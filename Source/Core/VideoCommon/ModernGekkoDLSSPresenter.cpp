// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#include "VideoCommon/ModernGekkoDLSSPresenter.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>
#include "Common/Logging/Log.h"
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/AbstractShader.h"
#include "VideoCommon/AbstractStagingTexture.h"
#include "VideoCommon/AbstractTexture.h"
#include "VideoCommon/ModernGekkoOpticalFlow.h"
#include "VideoCommon/ModernGekkoTemporalState.h"
#include "VideoCommon/RenderState.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VideoConfig.h"
#include "VideoBackends/Vulkan/CommandBufferManager.h"
#include "VideoBackends/Vulkan/ModernGekkoNGX.h"
#include "VideoBackends/Vulkan/StateTracker.h"
#include "VideoBackends/Vulkan/VKTexture.h"

namespace VideoCommon
{
namespace
{
namespace NGX = Vulkan::ModernGekkoNGX;
constexpr const char* RESAMPLE = R"(
layout(local_size_x=8, local_size_y=8, local_size_z=1) in;
UBO_BINDING(std140, 1) uniform Params { vec4 sizes; };
SAMPLER_BINDING(0) uniform sampler2DArray color_source;
SAMPLER_BINDING(1) uniform sampler2DArray depth_source;
IMAGE_BINDING(rgba16f, 0) uniform writeonly image2DArray color_target;
IMAGE_BINDING(r32f, 1) uniform writeonly image2DArray depth_target;
void main() {
  ivec2 p=ivec2(gl_GlobalInvocationID.xy);
  if(any(greaterThanEqual(p,ivec2(sizes.xy)))) return;
  vec2 uv=(vec2(p)+0.5)/sizes.xy;
  imageStore(color_target,ivec3(p,0),textureLod(color_source,vec3(uv,0),0));
  ivec2 q=clamp(ivec2(uv*sizes.zw),ivec2(0),ivec2(sizes.zw)-1);
  imageStore(depth_target,ivec3(p,0),vec4(texelFetch(depth_source,ivec3(q,0),0).r));
}
)";
constexpr const char* CUT_SAMPLE = R"(
layout(local_size_x=8, local_size_y=8, local_size_z=1) in;
SAMPLER_BINDING(0) uniform sampler2DArray color_source;
IMAGE_BINDING(rgba8, 0) uniform writeonly image2DArray cut_sample;
void main() {
  ivec2 p=ivec2(gl_GlobalInvocationID.xy);
  if(any(greaterThanEqual(p,ivec2(16)))) return;
  vec3 c=textureLod(color_source,vec3((vec2(p)+0.5)/16.0,0),0).rgb;
  imageStore(cut_sample,ivec3(p,0),vec4(c,1));
}
)";
bool Requested()
{
  const char* value=std::getenv("MODERNGEKKO_DLSS_EVALUATE");
  return value && std::string_view(value)=="1" && NGX::IsRequested() && NGX::IsInitialized() &&
    g_backend_info.api_type==APIType::Vulkan && g_ActiveConfig.stereo_mode==StereoMode::Off;
}
void Barrier(const AbstractTexture* texture, VkImageLayout layout, VkAccessFlags access)
{
  const auto* vk_texture=static_cast<const Vulkan::VKTexture*>(texture);
  auto* tracker=Vulkan::StateTracker::GetInstance();
  tracker->EndRenderPass();
  const auto command=Vulkan::g_command_buffer_mgr->GetCurrentCommandBuffer();
  // SDK writes are external to VKTexture's tracked compute state. Make them available
  // before a tracked transition, whose source scope alone may omit those writes.
  VkImageMemoryBarrier barrier{};
  barrier.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask=vk_texture->GetLayout()==VK_IMAGE_LAYOUT_UNDEFINED ? 0 :
    VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.oldLayout=barrier.newLayout=vk_texture->GetLayout();
  barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
  barrier.image=vk_texture->GetImage();
  barrier.subresourceRange={Vulkan::VKTexture::GetImageAspectForFormat(texture->GetFormat()),
    0,texture->GetLevels(),0,texture->GetLayers()};
  vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&barrier);
  vk_texture->TransitionToLayout(command,layout);
  barrier.srcAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask=access;
  barrier.oldLayout=barrier.newLayout=layout;
  vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&barrier);
}
NGX::Image Image(const AbstractTexture* texture)
{
  const auto* vk_texture=static_cast<const Vulkan::VKTexture*>(texture);
  return {vk_texture->GetImage(),vk_texture->GetView(),
    {Vulkan::VKTexture::GetImageAspectForFormat(texture->GetFormat()),0,1,0,1},
    vk_texture->GetVkFormat(),texture->GetWidth(),texture->GetHeight()};
}
}
struct ModernGekkoDLSSPresenter::State
{
  std::unique_ptr<ModernGekkoOpticalFlow> flow;
  std::unique_ptr<AbstractShader> resample,cut_shader;
  std::array<std::unique_ptr<AbstractTexture>,2> color,depth;
  std::unique_ptr<AbstractTexture> output,cut_texture;
  std::unique_ptr<AbstractStagingTexture> cut_readback;
  std::array<u8,16*16*4> previous_cut{};
  std::array<float,2> previous_jitter{},input_per_native{};
  MathUtil::Rectangle<int> source_rect{};
  u32 input_width=0,input_height=0,output_width=0,output_height=0,current=0;
  u64 content_id=std::numeric_limits<u64>::max(),evaluations=0,resource_generation=0;
  bool history=false,cut_history=false,output_valid=false,owns_feature=false,attempted=false;
  bool unavailable=false;
  u64 fallback_presents=0,rejected_packets=0,last_rejected_content=std::numeric_limits<u64>::max();
  u64 last_rejected_entry=std::numeric_limits<u64>::max();
  u32 consecutive_rejections=0;
  std::chrono::steady_clock::time_point last_frame{};
  void ResetHistory() { history=false;cut_history=false;output_valid=false;attempted=false; }
  void Release()
  {
    if(g_gfx && (owns_feature || output)) g_gfx->WaitForGPUIdle();
    if(owns_feature) NGX::DestroyFeature();
    owns_feature=false;
    flow.reset();color={};depth={};output.reset();cut_texture.reset();cut_readback.reset();
    input_width=input_height=output_width=output_height=0;
    ResetHistory();
  }
  const AbstractTexture* Fallback(std::string_view reason,u64 entry_id,u64 packet_id)
  {
    ++fallback_presents;
    if(entry_id!=last_rejected_entry || packet_id!=last_rejected_content) {
      last_rejected_entry=entry_id;last_rejected_content=packet_id;
      ++rejected_packets;++consecutive_rejections;
    }
    if(fallback_presents<=8 || fallback_presents%120==0)
      INFO_LOG_FMT(VIDEO,"ModernGekko DLSS: native fallback reason={}, rejected packets={}, consecutive={}, fallback presents={}, evaluations={}",
        reason,rejected_packets,consecutive_rejections,fallback_presents,evaluations);
    const bool attempted_this_packet=attempted && content_id==packet_id;
    ResetHistory();attempted=attempted_this_packet;
    if(!unavailable && consecutive_rejections>=16) {
      unavailable=true;
      ModernGekkoTemporal::SetActive(false);
      Release();
      WARN_LOG_FMT(VIDEO,"ModernGekko DLSS: unavailable for this run after {} rejected packets; raster jitter disabled; reason={}",
        consecutive_rejections,reason);
    }
    return nullptr;
  }
  bool Configure(u32 width,u32 height)
  {
    if(output && width==output_width && height==output_height) return true;
    Release();
    u32 iw=0,ih=0;
    if(!NGX::GetQualityRenderSize(width,height,&iw,&ih) || iw<32 || ih<32) return false;
    if(!resample) resample=g_gfx->CreateShaderFromSource(ShaderStage::Compute,RESAMPLE,nullptr,"DLSS matched input resample");
    if(!cut_shader) cut_shader=g_gfx->CreateShaderFromSource(ShaderStage::Compute,CUT_SAMPLE,nullptr,"DLSS scene cut sample");
    if(!resample || !cut_shader) return false;
    auto allocate=[](u32 w,u32 h,AbstractTextureFormat format,const char* name) {
      return g_gfx->CreateTexture(TextureConfig(w,h,1,1,1,format,AbstractTextureFlag_ComputeImage,
        AbstractTextureType::Texture_2DArray),name);
    };
    for(u32 i=0;i<2;++i) {
      color[i]=allocate(iw,ih,AbstractTextureFormat::RGBA16F,"DLSS matched color");
      depth[i]=allocate(iw,ih,AbstractTextureFormat::R32F,"DLSS matched depth");
      if(!color[i] || !depth[i]) {Release();return false;}
    }
    output=allocate(width,height,AbstractTextureFormat::RGBA16F,"DLSS output");
    cut_texture=allocate(16,16,AbstractTextureFormat::RGBA8,"DLSS private cut sample");
    cut_readback=g_gfx->CreateStagingTexture(StagingTextureType::Readback,
      TextureConfig(16,16,1,1,1,AbstractTextureFormat::RGBA8,0,AbstractTextureType::Texture_2DArray));
    if(!output || !cut_texture || !cut_readback) {Release();return false;}
    flow=std::make_unique<ModernGekkoOpticalFlow>();
    g_gfx->WaitForGPUIdle();
    Vulkan::StateTracker::GetInstance()->EndRenderPass();
    if(!NGX::CreateFeature(Vulkan::g_command_buffer_mgr->GetCurrentCommandBuffer(),iw,ih,
      width,height,g_backend_info.bSupportsReversedDepthRange)) {Release();return false;}
    owns_feature=true;input_width=iw;input_height=ih;output_width=width;output_height=height;
    ++resource_generation;
    Vulkan::StateTracker::GetInstance()->InvalidateCachedState();
    INFO_LOG_FMT(VIDEO,"ModernGekko DLSS: Quality input {}x{}, output {}x{}; experimental estimated motion",iw,ih,width,height);
    const u64 owned_texel_bytes=u64(iw)*ih*24+u64(width)*height*8+16*16*4;
    INFO_LOG_FMT(VIDEO,"ModernGekko DLSS: resource generation={}, presenter images=6, texel bytes={} (excludes flow, SDK and allocation padding)",
      resource_generation,owned_texel_bytes);
    return true;
  }
  void Bind(u32 slot,const AbstractTexture* texture)
  {
    Barrier(texture,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_ACCESS_SHADER_READ_BIT);
    g_gfx->SetTexture(slot,texture);g_gfx->SetSamplerState(slot,RenderState::GetLinearSamplerState());
  }
  bool IsCut(const AbstractTexture* texture)
  {
    Bind(0,texture);g_gfx->SetComputeImageTexture(0,cut_texture.get(),false,true);
    g_gfx->DispatchComputeShader(cut_shader.get(),8,8,1,2,2,1);
    cut_readback->CopyFromTexture(cut_texture.get());
    const auto start=std::chrono::steady_clock::now();
    cut_readback->Flush();
    if(!cut_readback->Map()) {cut_history=false;return true;}
    std::array<u8,16*16*4> sample{};
    for(u32 y=0;y<16;++y)
      std::copy_n(reinterpret_cast<const u8*>(cut_readback->GetMappedPointer())+
        y*cut_readback->GetMappedStride(),16*4,sample.data()+y*16*4);
    cut_readback->Unmap();
    double difference=0;
    for(u32 p=0;p<16*16;++p) for(u32 c=0;c<3;++c)
      difference+=std::abs(int(sample[p*4+c])-int(previous_cut[p*4+c]));
    difference/=16*16*3*255.0;
    const bool cut=cut_history && difference>0.30;
    previous_cut=sample;cut_history=true;
    if(cut || evaluations%120==0) {
      const double wait=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      INFO_LOG_FMT(VIDEO,"ModernGekko DLSS: cut metric {:.4f}, reset {}, readback wait {:.3f} ms",difference,cut,wait);
    }
    return cut;
  }
};
ModernGekkoDLSSPresenter::ModernGekkoDLSSPresenter() : m_state(std::make_unique<State>()) {}
ModernGekkoDLSSPresenter::~ModernGekkoDLSSPresenter()
{
  ModernGekkoTemporal::SetActive(false);
  m_state->Release();
}
const AbstractTexture* ModernGekkoDLSSPresenter::Render(const RcTcacheEntry& entry,
  const MathUtil::Rectangle<int>& source_rect,u32 width,u32 height)
{
  auto& s=*m_state;
  if(!Requested() || !width || !height) {
    ModernGekkoTemporal::SetActive(false);s.Release();return nullptr;
  }
  if(s.unavailable) {
    ModernGekkoTemporal::SetActive(false);
    return s.Fallback("experiment unavailable for this run; raster jitter disabled",entry ? entry->id : 0,
      ModernGekkoTemporal::GetState().content_id);
  }
  ModernGekkoTemporal::SetActive(true);
  const auto packet=TextureCacheBase::GetTemporalPacket(entry,source_rect);
  const u64 entry_id=entry ? entry->id : 0;
  const u64 packet_id=packet ? packet->content_id : ModernGekkoTemporal::GetState().content_id;
  auto fallback=[&](std::string_view reason) {return s.Fallback(reason,entry_id,packet_id);};
  if(!packet || !packet->color || !packet->depth) {
    return fallback(packet ? std::string_view(packet->reason) : "matched temporal packet absent");
  }
  if(packet->color->GetWidth()!=packet->depth->GetWidth() ||
     packet->color->GetHeight()!=packet->depth->GetHeight() ||
     !packet->color->GetWidth() || !packet->color->GetHeight() ||
     packet->efb_rect.GetWidth()<=0 || packet->efb_rect.GetHeight()<=0 ||
     !std::isfinite(packet->jitter_x) || !std::isfinite(packet->jitter_y)) {
    return fallback("invalid matched snapshot dimensions or jitter");
  }
  if(!s.Configure(width,height)) return fallback("SDK Quality configuration or allocation failed");
  const std::array<float,2> input_per_native={
    float(s.input_width)/packet->efb_rect.GetWidth(),
    float(s.input_height)/packet->efb_rect.GetHeight()};
  ModernGekkoTemporal::SetInputPixelsPerNative(input_per_native[0],input_per_native[1]);
  if(s.input_per_native!=input_per_native) {
    s.input_per_native=input_per_native;s.ResetHistory();
    // The captured packet predates this raster jitter scale; await a newly rendered packet.
    return fallback("awaiting newly rendered packet after raster jitter calibration");
  }
  if(!packet->valid) return fallback(packet->reason);
  const bool same_crop=s.source_rect==source_rect;
  if(packet->content_id==s.content_id && same_crop && s.attempted)
    return s.output_valid ? s.output.get() : nullptr;
  if(!same_crop) s.ResetHistory();
  s.source_rect=source_rect;s.content_id=packet->content_id;s.attempted=true;
  auto* color=s.color[s.current].get();auto* depth=s.depth[s.current].get();
  const std::array<float,4> sizes={float(s.input_width),float(s.input_height),
    float(packet->color->GetWidth()),float(packet->color->GetHeight())};
  g_vertex_manager->UploadUtilityUniforms(sizes.data(),sizeof(sizes));
  s.Bind(0,packet->color.get());s.Bind(1,packet->depth.get());
  g_gfx->SetComputeImageTexture(0,color,false,true);g_gfx->SetComputeImageTexture(1,depth,false,true);
  g_gfx->DispatchComputeShader(s.resample.get(),8,8,1,(s.input_width+7)/8,(s.input_height+7)/8,1);
  const bool cut=s.IsCut(color);
  const std::array<float,2> jitter={packet->jitter_x*input_per_native[0],
    packet->jitter_y*input_per_native[1]};
  bool reset=!s.history || cut || packet->history_reset;
  ModernGekkoOpticalFlow::Inputs inputs;
  inputs.current_color=color;inputs.previous_color=s.color[1-s.current].get();
  inputs.current_depth=depth;inputs.previous_depth=s.depth[1-s.current].get();
  inputs.width=s.input_width;inputs.height=s.input_height;
  inputs.current_jitter=jitter;inputs.previous_jitter=s.previous_jitter;
  inputs.history_valid=!reset;inputs.uniform_jitter_coverage=packet->uniform_jitter_coverage;
  const auto motion=s.flow->Run(inputs);
  if(!motion.motion || !motion.bias) return fallback("estimated motion pipeline failed");
  for(const AbstractTexture* texture:std::array<const AbstractTexture*,4>{color,depth,motion.motion,motion.bias})
    Barrier(texture,VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_SHADER_READ_BIT);
  Barrier(s.output.get(),VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_SHADER_WRITE_BIT);
  NGX::Frame frame{};
  frame.color=Image(color);frame.depth=Image(depth);frame.motion_vectors=Image(motion.motion);
  frame.bias=Image(motion.bias);frame.output=Image(s.output.get());
  frame.jitter_x=jitter[0];frame.jitter_y=jitter[1];frame.reset=reset;
  const auto now=std::chrono::steady_clock::now();
  frame.frame_time_ms=s.history ? std::clamp(float(std::chrono::duration<double,std::milli>(now-s.last_frame).count()),1.0f,100.0f) : 1000.0f/60.0f;
  s.output_valid=NGX::Evaluate(Vulkan::g_command_buffer_mgr->GetCurrentCommandBuffer(),frame);
  Barrier(s.output.get(),VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_ACCESS_SHADER_READ_BIT);
  Vulkan::StateTracker::GetInstance()->InvalidateCachedState();
  s.content_id=packet->content_id;s.previous_jitter=jitter;s.last_frame=now;
  s.history=s.output_valid;s.current=1-s.current;++s.evaluations;
  if(!s.output_valid) return fallback("SDK evaluation failed");
  s.consecutive_rejections=0;
  return s.output.get();
}
}
