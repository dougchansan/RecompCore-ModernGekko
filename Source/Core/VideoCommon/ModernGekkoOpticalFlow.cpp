// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#include "VideoCommon/ModernGekkoOpticalFlow.h"

#include <algorithm>
#include <cmath>
#include <string>
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/AbstractShader.h"
#include "VideoCommon/AbstractTexture.h"
#include "VideoCommon/RenderState.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VideoConfig.h"
#include "VideoBackends/Vulkan/CommandBufferManager.h"
#include "VideoBackends/Vulkan/StateTracker.h"
#include "VideoBackends/Vulkan/VKTexture.h"

namespace VideoCommon
{
namespace
{
// Utility UBO binding follows the existing texture decoding compute shaders.
constexpr const char* COMMON = R"(
layout(local_size_x=8, local_size_y=8, local_size_z=1) in;
UBO_BINDING(std140, 1) uniform Params {
  vec4 sizes; // destination xy, parent/source zw
  vec4 control; // has_parent, history, paired_depth, paired_jitter_mask
  vec4 jitter; // current xy, previous zw
  vec4 limits; // depth threshold, proven uniform jitter coverage
};
SAMPLER_BINDING(0) uniform sampler2DArray src0;
SAMPLER_BINDING(1) uniform sampler2DArray src1;
SAMPLER_BINDING(2) uniform sampler2DArray src2;
SAMPLER_BINDING(3) uniform sampler2DArray src3;
SAMPLER_BINDING(4) uniform sampler2DArray src4;
SAMPLER_BINDING(5) uniform sampler2DArray src5;
#ifndef FLOW_OUTPUT_FORMAT
#define FLOW_OUTPUT_FORMAT rgba16f
#endif
IMAGE_BINDING(FLOW_OUTPUT_FORMAT, 0) uniform writeonly image2DArray dst0;
vec3 sampleColor(sampler2DArray s, vec2 p, vec2 extent) {
  return textureLod(s, vec3((p+0.5)/extent, 0.0), 0.0).rgb;
}
float patchError(vec2 p, vec2 flow) {
  float e=0.0;
  for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
    vec2 offset=vec2(x,y);
    vec3 d=sampleColor(src0,p+offset,sizes.xy)-sampleColor(src1,p+offset+flow,sizes.xy);
    e+=dot(d,d)/27.0;
  }
  return e;
}
bool inside(vec2 p) { return all(greaterThanEqual(p,vec2(1.0))) &&
  all(lessThan(p,sizes.xy-vec2(1.0))); }
)";
constexpr const char* DOWNSAMPLE = R"(
void main() {
  ivec2 ip=ivec2(gl_GlobalInvocationID.xy);
  if(any(greaterThanEqual(ip,ivec2(sizes.xy)))) return;
  vec2 p=(vec2(ip)+0.5)*sizes.zw/sizes.xy-0.5;
  vec3 c=sampleColor(src0,p,sizes.zw);
  if(any(notEqual(sizes.xy,sizes.zw))) {
    const float weights[5]=float[5](1.0,4.0,6.0,4.0,1.0);
    c=vec3(0.0);
    for(int y=-2;y<=2;++y) for(int x=-2;x<=2;++x)
      c+=sampleColor(src0,p+vec2(x,y),sizes.zw)*weights[x+2]*weights[y+2]/256.0;
  }
  imageStore(dst0,ivec3(ip,0),vec4(c,1.0));
}
)";
constexpr const char* ESTIMATE = R"(
void main() {
  ivec2 ip=ivec2(gl_GlobalInvocationID.xy);
  if(any(greaterThanEqual(ip,ivec2(sizes.xy)))) return;
  vec2 p=vec2(ip), base=vec2(0.0);
  if(control.x>0.5) {
    vec2 parent=(p+0.5)*sizes.zw/sizes.xy-0.5;
    vec4 prior=textureLod(src2,vec3((parent+0.5)/sizes.zw,0.0),0.0);
    if(prior.w>0.1) base=prior.xy*sizes.xy/sizes.zw;
  }
  float best=1e10;
  vec2 flow=base;
  // At full resolution also test a local zero-centred basin: aliased coarse correspondence
  // must not trap small actual motion in a distant local minimum.
  for(int seed=0;seed<2;++seed) {
    if(seed==1 && control.y<0.5) continue;
    vec2 origin=seed==0 ? base : vec2(0.0);
    int radius=seed==0 ? 2 : 4;
    for(int y=-4;y<=4;++y) for(int x=-4;x<=4;++x) {
      if(abs(x)>radius || abs(y)>radius) continue;
      vec2 candidate=origin+vec2(x,y);
      if(!inside(p+candidate)) continue;
      float e=patchError(p,candidate);
      if(e+0.000001*dot(candidate,candidate)<best+0.000001*dot(flow,flow))
        { best=e; flow=candidate; }
    }
  }
  // Quadratic minimum at each axis gives subpixel correspondence from bilinear samples.
  vec2 sub=vec2(0.0);
  for(int axis=0;axis<2;++axis) {
    vec2 step=axis==0 ? vec2(1.0,0.0) : vec2(0.0,1.0);
    if(!inside(p+flow-step) || !inside(p+flow+step)) continue;
    float a=patchError(p,flow-step), b=patchError(p,flow+step);
    float curvature=a+b-2.0*best;
    if(curvature>1e-7) sub[axis]=clamp(0.5*(a-b)/curvature,-0.5,0.5);
  }
  flow+=sub;
  // Joint least-squares refinement accounts for coupled x/y gradients; the axis parabola
  // alone is inaccurate on oblique features. Reject singular (aperture/flat) systems.
  for(int iteration=0;iteration<3;++iteration) {
    float hxx=0.0,hxy=0.0,hyy=0.0,bx=0.0,by=0.0;
    for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
      vec2 at=p+vec2(x,y)+flow;
      vec3 d=sampleColor(src1,at,sizes.xy)-sampleColor(src0,p+vec2(x,y),sizes.xy);
      vec3 gx=(sampleColor(src1,at+vec2(0.25,0.0),sizes.xy)-
               sampleColor(src1,at-vec2(0.25,0.0),sizes.xy))*2.0;
      vec3 gy=(sampleColor(src1,at+vec2(0.0,0.25),sizes.xy)-
               sampleColor(src1,at-vec2(0.0,0.25),sizes.xy))*2.0;
      hxx+=dot(gx,gx);hxy+=dot(gx,gy);hyy+=dot(gy,gy);
      bx+=dot(gx,d);by+=dot(gy,d);
    }
    float determinant=hxx*hyy-hxy*hxy;
    if(determinant<1e-8) break;
    vec2 delta=vec2(hyy*bx-hxy*by,hxx*by-hxy*bx)/determinant;
    vec2 candidate=flow-clamp(delta,vec2(-0.5),vec2(0.5));
    if(!inside(p+candidate) || patchError(p,candidate)>patchError(p,flow)) break;
    flow=candidate;
  }
  float error=patchError(p,flow), alternative=1e10;
  // Separated alternatives reject repeated texture; adjacent samples are the same minimum.
  for(int y=-2;y<=2;++y) for(int x=-2;x<=2;++x) {
    if(x*x+y*y<4 || !inside(p+flow+vec2(x,y))) continue;
    alternative=min(alternative,patchError(p,flow+vec2(x,y)));
  }
  float unique=clamp((alternative-error)/max(alternative,0.0001),0.0,1.0);
  float mean=0.0, square=0.0;
  for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
    float l=dot(sampleColor(src0,p+vec2(x,y),sizes.xy),vec3(0.2126,0.7152,0.0722));
    mean+=l/9.0; square+=l*l/9.0;
  }
  unique*=smoothstep(0.00005,0.0005,max(0.0,square-mean*mean));
  imageStore(dst0,ivec3(ip,0),vec4(flow,error,unique));
}
)";
constexpr const char* CONFIDENCE = R"(
IMAGE_BINDING(rgba8, 1) uniform writeonly image2DArray dst1;
SAMPLER_BINDING(6) uniform sampler2DArray mask0;
SAMPLER_BINDING(7) uniform sampler2DArray mask1;
void main() {
  ivec2 ip=ivec2(gl_GlobalInvocationID.xy);
  if(any(greaterThanEqual(ip,ivec2(sizes.xy)))) return;
  vec2 p=vec2(ip);
  vec4 forward=textureLod(src2,vec3((p+0.5)/sizes.xy,0.0),0.0);
  vec2 flow=forward.xy, q=p+flow;
  vec4 backward=textureLod(src3,vec3((q+0.5)/sizes.xy,0.0),0.0);
  float agreement=length(flow+backward.xy);
  float confidence=(1.0-smoothstep(0.5,1.5,agreement)) *
    (1.0-smoothstep(0.001,0.012,forward.z)) * smoothstep(0.05,0.3,forward.w);
  float mean=0.0, square=0.0;
  for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
    float l=dot(sampleColor(src0,p+vec2(x,y),sizes.xy),vec3(0.2126,0.7152,0.0722));
    mean+=l/9.0; square+=l*l/9.0;
  }
  confidence*=smoothstep(0.00005,0.0005,max(0.0,square-mean*mean));
  if(!inside(p) || !inside(q) || control.y<0.5) confidence=0.0;
  if(control.z>0.5) {
    float d=sampleColor(src4,p,sizes.xy).r, old=sampleColor(src5,q,sizes.xy).r;
    confidence*=1.0-smoothstep(limits.x,limits.x*2.0,abs(d-old));
  }
  if(control.w>0.5) {
    float mask=min(sampleColor(mask0,p,sizes.xy).r,sampleColor(mask1,q,sizes.xy).r);
    // Only classified world pixels lose raster jitter; guest UI keeps its measured displacement.
    flow+=(jitter.xy-jitter.zw)*step(0.9,mask);
  } else if(limits.y>0.5) {
    flow+=jitter.xy-jitter.zw;
  } else if(any(greaterThan(abs(jitter.xy-jitter.zw),vec2(0.00001)))) {
    // Jittered inputs without classification cannot supply unjittered UI/world vectors.
    confidence=0.0;
  }
  if(control.y<0.5) flow=vec2(0.0);
  imageStore(dst0,ivec3(ip,0),vec4(flow,confidence,0.0));
  imageStore(dst1,ivec3(ip,0),vec4(1.0-confidence,0.0,0.0,1.0));
}
)";
struct alignas(16) Uniforms
{
  std::array<float, 4> sizes{}, control{}, jitter{}, limits{};
};
bool Matches(const AbstractTexture* texture, u32 width, u32 height)
{
  return texture && texture->GetConfig().width==width && texture->GetConfig().height==height &&
         texture->GetConfig().layers==1 && texture->GetConfig().samples==1 &&
         texture->GetConfig().type==AbstractTextureType::Texture_2DArray;
}
}  // namespace

ModernGekkoOpticalFlow::ModernGekkoOpticalFlow() = default;
ModernGekkoOpticalFlow::~ModernGekkoOpticalFlow() = default;

bool ModernGekkoOpticalFlow::Initialize(u32 width, u32 height)
{
  if(!m_downsample)
    m_downsample=g_gfx->CreateShaderFromSource(ShaderStage::Compute,
      std::string(COMMON)+DOWNSAMPLE,nullptr,"ModernGekko optical flow downsample");
  if(!m_estimate)
    m_estimate=g_gfx->CreateShaderFromSource(ShaderStage::Compute,
      std::string(COMMON)+ESTIMATE,nullptr,"ModernGekko optical flow estimate");
  if(!m_confidence)
    m_confidence=g_gfx->CreateShaderFromSource(ShaderStage::Compute,
      std::string("#define FLOW_OUTPUT_FORMAT rg16f\n")+COMMON+CONFIDENCE,nullptr,"ModernGekko optical flow confidence");
  if(!m_downsample || !m_estimate || !m_confidence) return false;
  if(m_width==width && m_height==height && m_motion && m_bias) return true;
  auto allocate=[&](u32 w,u32 h,AbstractTextureFormat format,const char* name) {
    return g_gfx->CreateTexture(TextureConfig(w,h,1,1,1,format,
      AbstractTextureFlag_ComputeImage,AbstractTextureType::Texture_2DArray),name);
  };
  m_width=0; m_height=0;
  for(size_t direction=0;direction<2;++direction)
    for(size_t level=0;level<LEVELS;++level) {
      const u32 w=std::max(1u,(width+(1u<<level)-1)>>level);
      const u32 h=std::max(1u,(height+(1u<<level)-1)>>level);
      m_color[direction][level]=allocate(w,h,AbstractTextureFormat::RGBA16F,"Flow pyramid");
      m_flow[direction][level]=allocate(w,h,AbstractTextureFormat::RGBA16F,"Flow correspondence");
      if(!m_color[direction][level] || !m_flow[direction][level]) return false;
    }
  m_motion=allocate(width,height,AbstractTextureFormat::RG16F,"Flow motion and confidence");
  m_bias=allocate(width,height,AbstractTextureFormat::RGBA8,"Flow current color bias");
  if(!m_motion || !m_bias) return false;
  m_width=width; m_height=height;
  return true;
}

void ModernGekkoOpticalFlow::BindSample(u32 slot,const AbstractTexture* texture)
{
  g_gfx->SetTexture(slot,texture);
  g_gfx->SetSamplerState(slot,RenderState::GetLinearSamplerState());
  // Existing sampled-texture transitions target fragment reads only. These compute shaders
  // need visibility too. This Vulkan-only module deliberately owns that additional dependency.
  const auto* vk_texture=static_cast<const Vulkan::VKTexture*>(texture);
  VkImageMemoryBarrier barrier{};
  barrier.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
  barrier.oldLayout=barrier.newLayout=vk_texture->GetLayout();
  barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
  barrier.image=vk_texture->GetImage();
  barrier.subresourceRange={Vulkan::VKTexture::GetImageAspectForFormat(texture->GetFormat()),
                           0,texture->GetLevels(),0,texture->GetLayers()};
  Vulkan::StateTracker::GetInstance()->EndRenderPass();
  vkCmdPipelineBarrier(Vulkan::g_command_buffer_mgr->GetCurrentCommandBuffer(),
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,
    0,nullptr,0,nullptr,1,&barrier);
}

void ModernGekkoOpticalFlow::Dispatch(const AbstractShader* shader,u32 width,u32 height)
{
  g_gfx->DispatchComputeShader(shader,8,8,1,(width+7)/8,(height+7)/8,1);
  // Order sampled compute reads before resource reuse as writes in this or the next Run.
  // VKTexture's generic sampled-layout transition currently tracks fragment reads only.
  VkMemoryBarrier barrier{};
  barrier.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vkCmdPipelineBarrier(Vulkan::g_command_buffer_mgr->GetCurrentCommandBuffer(),
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,
    1,&barrier,0,nullptr,0,nullptr);
}

ModernGekkoOpticalFlow::Result ModernGekkoOpticalFlow::Run(const Inputs& in)
{
  if(g_backend_info.api_type!=APIType::Vulkan || !g_backend_info.bSupportsComputeShaders ||
     !g_gfx || !g_vertex_manager || in.width<32 || in.height<32 ||
     !Matches(in.current_color,in.width,in.height)) return {};
  const bool history=in.history_valid && Matches(in.previous_color,in.width,in.height);
  if(!std::isfinite(in.depth_rejection_threshold) || in.depth_rejection_threshold<=0.0f)
    return {};
  for(float value:in.current_jitter) if(!std::isfinite(value)) return {};
  for(float value:in.previous_jitter) if(!std::isfinite(value)) return {};
  if(!Initialize(in.width,in.height)) return {};
  g_vertex_manager->Flush();
  Uniforms u;
  for(size_t direction=0;direction<2;++direction)
    for(size_t level=0;level<LEVELS;++level) {
      const AbstractTexture* source=level ? m_color[direction][level-1].get() :
        (direction && history ? in.previous_color : in.current_color);
      const auto& config=m_color[direction][level]->GetConfig();
      u.sizes={float(config.width),float(config.height),float(source->GetConfig().width),
               float(source->GetConfig().height)};
      g_vertex_manager->UploadUtilityUniforms(&u,sizeof(u));
      BindSample(0,source);
      g_gfx->SetComputeImageTexture(0,m_color[direction][level].get(),false,true);
      Dispatch(m_downsample.get(),config.width,config.height);
    }
  for(size_t direction=0;direction<2;++direction)
    for(size_t reverse=LEVELS;reverse>0;--reverse) {
      const size_t level=reverse-1;
      const auto& config=m_flow[direction][level]->GetConfig();
      const AbstractTexture* parent=level+1<LEVELS ? m_flow[direction][level+1].get() :
        m_color[direction][level].get();
      u.sizes={float(config.width),float(config.height),float(parent->GetConfig().width),
               float(parent->GetConfig().height)};
      u.control[0]=level+1<LEVELS ? 1.0f : 0.0f;
      u.control[1]=level==0 ? 1.0f : 0.0f;
      g_vertex_manager->UploadUtilityUniforms(&u,sizeof(u));
      BindSample(0,m_color[direction][level].get());
      BindSample(1,m_color[1-direction][level].get());
      BindSample(2,parent);
      g_gfx->SetComputeImageTexture(0,m_flow[direction][level].get(),false,true);
      Dispatch(m_estimate.get(),config.width,config.height);
    }
  const bool depth=history && Matches(in.current_depth,in.width,in.height) &&
                   Matches(in.previous_depth,in.width,in.height);
  const bool masks=history && Matches(in.current_jitter_mask,in.width,in.height) &&
                   Matches(in.previous_jitter_mask,in.width,in.height);
  u.sizes={float(in.width),float(in.height),float(in.width),float(in.height)};
  u.control={0.0f,history ? 1.0f : 0.0f,depth ? 1.0f : 0.0f,masks ? 1.0f : 0.0f};
  u.jitter={in.current_jitter[0],in.current_jitter[1],in.previous_jitter[0],in.previous_jitter[1]};
  u.limits[0]=in.depth_rejection_threshold;
  u.limits[1]=in.uniform_jitter_coverage ? 1.0f : 0.0f;
  g_vertex_manager->UploadUtilityUniforms(&u,sizeof(u));
  BindSample(0,m_color[0][0].get()); BindSample(1,m_color[1][0].get());
  BindSample(2,m_flow[0][0].get()); BindSample(3,m_flow[1][0].get());
  BindSample(4,depth ? in.current_depth : m_color[0][0].get());
  BindSample(5,depth ? in.previous_depth : m_color[1][0].get());
  BindSample(6,masks ? in.current_jitter_mask : m_color[0][0].get());
  BindSample(7,masks ? in.previous_jitter_mask : m_color[1][0].get());
  g_gfx->SetComputeImageTexture(0,m_motion.get(),false,true);
  g_gfx->SetComputeImageTexture(1,m_bias.get(),false,true);
  Dispatch(m_confidence.get(),in.width,in.height);
  // Force tracked compute-write -> read transitions before reuse or NGX evaluation.
  BindSample(0,m_motion.get()); BindSample(1,m_bias.get());
  Vulkan::StateTracker::GetInstance()->InvalidateCachedState();
  return {m_motion.get(),m_bias.get(),history};
}
}  // namespace VideoCommon
