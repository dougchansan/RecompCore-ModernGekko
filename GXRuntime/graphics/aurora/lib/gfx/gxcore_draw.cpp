#include "gxcore_draw.hpp"

#include "../webgpu/gpu.hpp"
#include "../gx/gx.hpp" // UseReversedZ + set_logical_viewport (substrate glue)
#include "texture.hpp"
#include "tex_copy_conv.hpp" // EFB-copy format conversion (63/S16)

#include <gxruntime/gxcore/gxcore.hpp> // EfbCopyCommand
#include <gxruntime/gxcore/texture_decode.hpp>

#include <absl/container/flat_hash_map.h>

#include <absl/container/flat_hash_set.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <cstring>
#include <mutex>
#include <vector>

namespace aurora::gfx::gxcore {

namespace gxc = gxruntime::gxcore;

namespace {

// Frame interpolation (AURORA_FRAME_INTERP=<layers>, 2-4). While it is on, a
// game frame's draws and EFB copies are logged here instead of encoded; at
// present the log is replayed once per in-between frame with each draw's
// position matrices blended from the previous game frame's, then once as the
// real frame (interp_replay). Same rules as Dolphin's FrameInterp
// (VideoCommon/XFStructs.cpp): a draw pairs with the previous frame's draw of
// the same vertex data, the nearest matrix among several; camera cuts, 2D
// (orthographic) draws and unpaired draws are drawn where they really are.
struct LoggedDraw {
  gfx::ClipRect scissor{};
  bool has_viewport = false;
  gfx::Viewport viewport{};
  PipelineRef pipeline = 0;
  BindGroupRef textureBindGroup = 0;
  std::vector<float> vertices;
  std::vector<uint16_t> indices;
  gxc::VertexShaderConstants constants{};
  gxc::PixelShaderConstants pixel_constants{};
  bool tev = false;
  bool pos_mtx_idx = false;
  // A draw sampling an EFB copy of this frame keeps its plan and binds its
  // textures at replay: when it was logged the copy had not run yet (it is
  // logged too), so it bound guest memory - a black square behind screen-space
  // effects - or last frame's copy texture.
  std::unique_ptr<gxc::DrawPlan> rebind;
  uint64_t key = 0;   // vertex data + index count: names the object
  int32_t pair = -1;  // index of its previous-frame draw, -1 = none
};
struct LogEntry {
  bool copy = false;
  gxc::EfbCopyCommand copy_cmd{};
  uint32_t draw = 0;  // index into draws
};
struct InterpLog {
  std::vector<LogEntry> entries;
  std::vector<LoggedDraw> draws;
  bool paired = false;
  void clear();
};
// Logged draws of finished frames, kept for their buffers: a frame logs ~2000
// draws, and allocating each one's vertex and index storage anew cost more
// than copying into it.
std::vector<LoggedDraw> g_interpPool;
void InterpLog::clear() {
  entries.clear();
  for (LoggedDraw& d : draws) {
    d.rebind.reset();
    g_interpPool.push_back(std::move(d));
  }
  draws.clear();
  paired = false;
}
uint32_t g_interpLayers = [] {
  const char* v = std::getenv("AURORA_FRAME_INTERP");
  const long n = v != nullptr ? std::strtol(v, nullptr, 10) : 0;
  return static_cast<uint32_t>(n >= 2 && n <= 4 ? n : 0);
}();
InterpLog g_interpCur, g_interpPrev;
// Layers the presenter currently asks for (interp_set_active): 1 renders
// live, logging nothing.
uint32_t g_interpActive = g_interpLayers;
absl::flat_hash_set<uint32_t> g_interpCopyDests;  // EFB copy destinations logged this frame
bool g_interpReplaying = false;

struct InterpTiming { double record = 0, pair = 0, replay = 0; uint64_t frames = 0, draws = 0, copies = 0, rebinds = 0; };
InterpTiming g_interpTiming;
const bool g_interpLog = std::getenv("AURORA_INTERP_LOG") != nullptr;
double interp_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool interp_recording() { return g_interpActive > 1 && !g_interpReplaying; }
}  // namespace

using webgpu::g_device;
using webgpu::g_graphicsConfig;

static Module Log("aurora::gfx::gxcore");

namespace {

wgpu::CompareFunction to_compare(gxc::CompareMode func) {
  // GC compare flipped for the substrate's reversed-Z (gx/gx.cpp:526 shape).
  switch (func) {
  case gxc::CompareMode::Never:
    return wgpu::CompareFunction::Never;
  case gxc::CompareMode::Less:
    return gx::UseReversedZ ? wgpu::CompareFunction::Greater
                            : wgpu::CompareFunction::Less;
  case gxc::CompareMode::Equal:
    return wgpu::CompareFunction::Equal;
  case gxc::CompareMode::LEqual:
    return gx::UseReversedZ ? wgpu::CompareFunction::GreaterEqual
                            : wgpu::CompareFunction::LessEqual;
  case gxc::CompareMode::Greater:
    return gx::UseReversedZ ? wgpu::CompareFunction::Less
                            : wgpu::CompareFunction::Greater;
  case gxc::CompareMode::NEqual:
    return wgpu::CompareFunction::NotEqual;
  case gxc::CompareMode::GEqual:
    return gx::UseReversedZ ? wgpu::CompareFunction::LessEqual
                            : wgpu::CompareFunction::GreaterEqual;
  case gxc::CompareMode::Always:
  default:
    return wgpu::CompareFunction::Always;
  }
}

wgpu::BlendFactor to_blend_factor_src(gxc::SrcBlendFactor factor) {
  switch (factor) {
  case gxc::SrcBlendFactor::Zero:
    return wgpu::BlendFactor::Zero;
  case gxc::SrcBlendFactor::One:
    return wgpu::BlendFactor::One;
  case gxc::SrcBlendFactor::DstClr:
    return wgpu::BlendFactor::Dst;
  case gxc::SrcBlendFactor::InvDstClr:
    return wgpu::BlendFactor::OneMinusDst;
  case gxc::SrcBlendFactor::SrcAlpha:
    return wgpu::BlendFactor::SrcAlpha;
  case gxc::SrcBlendFactor::InvSrcAlpha:
    return wgpu::BlendFactor::OneMinusSrcAlpha;
  case gxc::SrcBlendFactor::DstAlpha:
    return wgpu::BlendFactor::DstAlpha;
  case gxc::SrcBlendFactor::InvDstAlpha:
  default:
    return wgpu::BlendFactor::OneMinusDstAlpha;
  }
}

wgpu::BlendFactor to_blend_factor_dst(gxc::DstBlendFactor factor) {
  switch (factor) {
  case gxc::DstBlendFactor::Zero:
    return wgpu::BlendFactor::Zero;
  case gxc::DstBlendFactor::One:
    return wgpu::BlendFactor::One;
  case gxc::DstBlendFactor::SrcClr:
    return wgpu::BlendFactor::Src;
  case gxc::DstBlendFactor::InvSrcClr:
    return wgpu::BlendFactor::OneMinusSrc;
  case gxc::DstBlendFactor::SrcAlpha:
    return wgpu::BlendFactor::SrcAlpha;
  case gxc::DstBlendFactor::InvSrcAlpha:
    return wgpu::BlendFactor::OneMinusSrcAlpha;
  case gxc::DstBlendFactor::DstAlpha:
    return wgpu::BlendFactor::DstAlpha;
  case gxc::DstBlendFactor::InvDstAlpha:
  default:
    return wgpu::BlendFactor::OneMinusDstAlpha;
  }
}

// Texture bind group layout for a used-texmap set (63/Mfin multi-texmap): texmap
// t occupies binding 2t (texture) + 2t+1 (sampler), matching the WGSL. Cached per
// mask; used_mask=1 (texmap 0 only) reproduces the pre-Mfin single-texmap layout.
// Called by draws on the recording thread and by create_pipeline on the pipeline
// thread: unlocked, an insert's rehash on one let the other miss, create its own
// layout and get back an uncached temporary that the bind group outlived.
wgpu::BindGroupLayout texture_bind_group_layout(uint32_t used_mask = 1u) {
  static std::mutex mutex;
  static absl::flat_hash_map<uint32_t, wgpu::BindGroupLayout> cache;
  std::lock_guard lock{mutex};
  auto it = cache.find(used_mask);
  if (it != cache.end())
    return it->second;
  std::vector<wgpu::BindGroupLayoutEntry> entries;
  for (uint32_t t = 0; t < 8u; ++t) {
    if ((used_mask & (1u << t)) == 0u)
      continue;
    entries.push_back(wgpu::BindGroupLayoutEntry{
        .binding = 2u * t,
        .visibility = wgpu::ShaderStage::Fragment,
        .texture =
            wgpu::TextureBindingLayout{
                .sampleType = wgpu::TextureSampleType::Float,
                .viewDimension = wgpu::TextureViewDimension::e2D,
            },
    });
    entries.push_back(wgpu::BindGroupLayoutEntry{
        .binding = 2u * t + 1u,
        .visibility = wgpu::ShaderStage::Fragment,
        .sampler =
            wgpu::SamplerBindingLayout{
                .type = wgpu::SamplerBindingType::Filtering,
            },
    });
  }
  const wgpu::BindGroupLayoutDescriptor descriptor{
      .label = "GXCore Texture Bind Group Layout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  auto layout = g_device.CreateBindGroupLayout(&descriptor);
  cache.emplace(used_mask, layout);
  return layout;
}

// Guest-identity texture cache (S13 A3). Keyed by the texture's guest identity
// AND the TLUT identity it indexes: the same CI image bytes re-palettized to a
// different TLUT decode to different pixels, so the palette address/format/
// entries are part of the key. 63/Mfin adds `content_hash` (xxh3 of the actual
// source texels + TLUT): the live game reuses one guest texture-buffer address
// for different images across screens (stadium select overwrites the captain
// art) and streams movie frames through a single buffer, so identity alone would
// bind the stale decode. Hashing the content makes a content change under an
// unchanged identity a distinct entry (Dolphin/Aurora texel-hash model). Static
// replay fixtures never rewrite an address, so the no-reconvert property holds.
struct TextureKey {
  uint32_t address;
  uint32_t size;
  uint32_t format;
  uint32_t width;
  uint32_t height;
  uint32_t tlut_address;
  uint32_t tlut_format;
  uint32_t tlut_entries;
  uint64_t content_hash;
  bool operator==(const TextureKey&) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const TextureKey& key) {
    return H::combine(std::move(h), key.address, key.size, key.format,
                      key.width, key.height, key.tlut_address, key.tlut_format,
                      key.tlut_entries, key.content_hash);
  }
};

absl::flat_hash_map<TextureKey, TextureHandle> g_textureCache;
// address -> the key currently cached at that guest address. When new content
// arrives at an address (movie streaming, screen transitions), the prior entry
// for that address is evicted so the content-hashed cache stays bounded to one
// live entry per buffer instead of accumulating every historical frame.
// The frame it was last bound in rides along: content replaced at an address
// whose texture no draw of the current frame has bound yet is written into
// that texture in place instead of creating a new one (games rewrite some
// buffers every frame; creating a GPU texture per rewrite dominated the feed).
struct AddrEntry {
  TextureKey key;
  uint32_t last_used_frame;
};
absl::flat_hash_map<uint32_t, AddrEntry> g_textureAddrKey;
// Content hashes taken this frame, keyed by identity (content_hash 0): one
// texture is typically bound by many draws per frame and hashing its full
// texel data for each of them cost more than the draws. Guest memory is read
// live while the frame streams in, so a texture rewritten between two of its
// own draws in one frame is not something this path could observe anyway.
absl::flat_hash_map<TextureKey, uint64_t> g_textureHashMemo;
uint32_t g_textureHashMemoFrame = UINT32_MAX;
TextureCacheStats g_textureCacheStats;
// GPU vertex fetch: arrays already in this frame's storage buffer, by host
// pointer -> byte offset and the span uploaded.
struct FetchUpload {
  uint32_t offset = 0;
  uint32_t size = 0;
};
absl::flat_hash_map<uintptr_t, FetchUpload> g_fetchArrays;
uint32_t g_fetchArraysFrame = UINT32_MAX;

// EFB copy-to-texture destinations (63/S16): dest guest address -> resolved EFB
// texture. A draw binding a texture at one of these addresses samples the copied
// EFB content instead of the (stale, never-written-by-us) guest memory there.
// Mirrors aurora lib/gx GXState::copyTextures. Copies key by dest address only
// (a re-copy to the same address updates in place, matching the live path).
absl::flat_hash_map<uint32_t, TextureHandle> g_efbCopyTextures;

} // namespace

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  const gxc::PipelineKey& key = config.key;
  const std::string wgsl = gxc::generate_wgsl(key.shader);
  wgpu::ShaderSourceWGSL sourceDescriptor{};
  sourceDescriptor.code = wgsl.c_str();
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &sourceDescriptor,
      .label = "GXCore Shader Module",
  };
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);

  // Group 0 keeps the pass preamble's static bind group compatible (the shader
  // never references it); 1 = shared dynamic VS uniform. On the TEV path (S14)
  // 2 = shared dynamic PS uniform and 3 = texture; else 2 = texture. Putting
  // the PS uniform before the texture keeps an untextured TEV draw gap-free.
  const bool tev = key.shader.tev_valid != 0;
  const bool textured = key.shader.textured != 0;
  // Multi-texmap: the texture group's layout matches the set of texmaps the WGSL
  // declares (derived identically from the shader key), so pipeline and bind
  // group agree. Untextured draws never use the texture group (layoutCount below).
  const uint32_t tex_mask = textured ? gxc::used_texmap_mask(key.shader) : 1u;
  std::array<wgpu::BindGroupLayout, 4> bindGroupLayouts{
      g_staticBindGroupLayout,
      g_uniformBindGroupLayout,
      tev ? g_uniformBindGroupLayout : texture_bind_group_layout(tex_mask),
      texture_bind_group_layout(tex_mask),
  };
  const size_t layoutCount = tev ? (textured ? 4 : 3) : (textured ? 3 : 2);
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .label = "GXCore Pipeline Layout",
      .bindGroupLayoutCount = layoutCount,
      .bindGroupLayouts = bindGroupLayouts.data(),
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);

  // Fixed decoded-vertex layout (gxruntime/gxcore/shader.hpp). The normal
  // (location 8) is only declared by the lit shader, so add it to the pipeline
  // only when the key is lit — WGSL requires the vertex layout to satisfy every
  // shader input.
  std::vector<wgpu::VertexAttribute> attributes{
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x3,
          .offset = gxc::kVertexPosOffset,
          .shaderLocation = 0,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Uint32,
          .offset = gxc::kVertexPosMtxOffset,
          .shaderLocation = 1,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x4,
          .offset = gxc::kVertexColor0Offset,
          .shaderLocation = 2,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x4,
          .offset = gxc::kVertexColor1Offset,
          .shaderLocation = 3,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset,
          .shaderLocation = 4,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset + 8,
          .shaderLocation = 5,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset + 16,
          .shaderLocation = 6,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset + 24,
          .shaderLocation = 7,
      },
  };
  // Item-5 texgen inputs must mirror the generator's VertexIn: emboss needs the
  // NBT normal/binormal/tangent + light dir; a Color1/emboss key that is unlit
  // still declares the normal (location 8). Detect emboss demand from the key.
  bool has_emboss = false;
  for (std::uint32_t i = 0; i < key.shader.num_tex_gens; ++i) {
    if (static_cast<gxc::TexGenType>(key.shader.tex_gens[i].texgentype) ==
        gxc::TexGenType::EmbossMap)
      has_emboss = true;
  }
  // Mirror the generator exactly: locations 8/10/11 are declared only when the
  // vertex FORMAT carries that attribute. A lit/emboss draw whose format omits
  // it reads the cached fallback from the uniform instead (I_CACHED_NORMAL), so
  // the shader does not declare the input and the layout must not provide it.
  if ((key.shader.lit_valid != 0 || has_emboss ||
       gxc::texgen_reads_row(key.shader, gxc::TexSourceRow::Normal)) &&
      key.shader.has_vertex_normal != 0) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x3,
        .offset = gxc::kVertexNormalOffset,
        .shaderLocation = 8,
    });
  }
  if (key.shader.has_tex_mtx_idx != 0) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Uint32,
        .offset = gxc::kVertexTexMtxIdxOffset,
        .shaderLocation = 9,
    });
  }
  if ((has_emboss || gxc::texgen_reads_row(key.shader, gxc::TexSourceRow::BinormalB)) &&
      key.shader.has_vertex_binormal != 0) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x3,
        .offset = gxc::kVertexBinormalOffset,
        .shaderLocation = 10,
    });
  }
  if ((has_emboss || gxc::texgen_reads_row(key.shader, gxc::TexSourceRow::BinormalT)) &&
      key.shader.has_vertex_tangent != 0) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x3,
        .offset = gxc::kVertexTangentOffset,
        .shaderLocation = 11,
    });
  }
  const wgpu::VertexBufferLayout vertexLayout{
      .stepMode = wgpu::VertexStepMode::Vertex,
      .arrayStride = gxc::kVertexStrideBytes,
      .attributeCount = attributes.size(),
      .attributes = attributes.data(),
  };

  const bool depthCompare = key.depth_test != 0;
  const wgpu::DepthStencilState depthStencil{
      .format = g_graphicsConfig.depthFormat,
      .depthWriteEnabled = depthCompare && key.depth_update != 0,
      .depthCompare =
          depthCompare ? to_compare(static_cast<gxc::CompareMode>(
                             key.depth_func))
                       : wgpu::CompareFunction::Always,
  };

  // GC subtract mode forces ONE/ONE with dst - src (Dolphin RenderState).
  wgpu::BlendState blendState{};
  if (key.blend_subtract != 0) {
    blendState.color = {
        .operation = wgpu::BlendOperation::ReverseSubtract,
        .srcFactor = wgpu::BlendFactor::One,
        .dstFactor = wgpu::BlendFactor::One,
    };
    blendState.alpha = blendState.color;
  } else {
    blendState.color = {
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = to_blend_factor_src(
            static_cast<gxc::SrcBlendFactor>(key.src_factor)),
        .dstFactor = to_blend_factor_dst(
            static_cast<gxc::DstBlendFactor>(key.dst_factor)),
    };
    blendState.alpha = blendState.color;
  }
  auto writeMask = wgpu::ColorWriteMask::None;
  if (key.color_update != 0) {
    writeMask |= wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green |
                 wgpu::ColorWriteMask::Blue;
  }
  if (key.alpha_update != 0)
    writeMask |= wgpu::ColorWriteMask::Alpha;
  const bool blending =
      key.blend_enable != 0 || key.blend_subtract != 0;
  const wgpu::ColorTargetState colorTarget{
      .format = g_graphicsConfig.surfaceConfiguration.format,
      .blend = blending ? &blendState : nullptr,
      .writeMask = writeMask,
  };
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = 1,
      .targets = &colorTarget,
  };

  auto cullMode = wgpu::CullMode::None;
  switch (static_cast<gxc::CullMode>(key.cull_mode)) {
  case gxc::CullMode::Back:
    cullMode = wgpu::CullMode::Back;
    break;
  case gxc::CullMode::Front:
    cullMode = wgpu::CullMode::Front;
    break;
  default:
    break; // None here; All was skipped at plan time
  }

  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "GXCore Pipeline",
      .layout = pipelineLayout,
      .vertex =
          wgpu::VertexState{
              .module = module,
              .entryPoint = "vs_main",
              // GPU vertex fetch reads its vertices from storage instead.
              .bufferCount = config.key.shader.fetch.enabled != 0 ? 0u : 1u,
              .buffers = config.key.shader.fetch.enabled != 0 ? nullptr : &vertexLayout,
          },
      .primitive =
          wgpu::PrimitiveState{
              .topology = wgpu::PrimitiveTopology::TriangleList,
              // Substrate winding convention (gx/gx.cpp to_primitive_state).
              .frontFace = wgpu::FrontFace::CW,
              .cullMode = cullMode,
          },
      .depthStencil = &depthStencil,
      .multisample =
          wgpu::MultisampleState{
              .count = config.msaaSamples,
          },
      .fragment = &fragmentState,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  if (!bind_pipeline(data.pipeline, pass)) {
    return;
  }
  const std::array vsOffsets{data.uniformRange.offset};
  pass.SetBindGroup(1, g_uniformBindGroup, vsOffsets.size(), vsOffsets.data());
  if (data.tev) {
    // group 2 = PS uniform (same dynamic-uniform bind group, its own offset);
    // group 3 = texture.
    const std::array psOffsets{data.pixelUniformRange.offset};
    pass.SetBindGroup(2, g_uniformBindGroup, psOffsets.size(),
                      psOffsets.data());
    if (data.textureBindGroup != 0) {
      pass.SetBindGroup(3, find_bind_group(data.textureBindGroup));
    }
  } else if (data.textureBindGroup != 0) {
    pass.SetBindGroup(2, find_bind_group(data.textureBindGroup));
  }
  if (!data.fetch)
    pass.SetVertexBuffer(0, g_vertexBuffer, data.vertRange.offset,
                         data.vertRange.size);
  pass.SetIndexBuffer(g_indexBuffer, wgpu::IndexFormat::Uint16,
                      data.idxRange.offset, data.idxRange.size);
  pass.DrawIndexed(data.indexCount);
}

void reset_texture_cache() {
  g_textureCache.clear();
  g_textureAddrKey.clear();
  g_textureHashMemo.clear();
  g_textureHashMemoFrame = UINT32_MAX;
  g_efbCopyTextures.clear();
  g_textureCacheStats = {};
}

const TextureCacheStats& texture_cache_stats() { return g_textureCacheStats; }

// Resolve the current EFB region into a texture the guest-identity path can bind
// later, keyed by the copy's destination address (63/S16). Mirrors aurora
// lib/dolphin/gx/GXFrameBuffer.cpp copy_tex on the same substrate: map the EFB
// source rect through the logical->render scaling, allocate (or reuse) a resolve
// target, then gfx::resolve_pass — which converts per format (tex_copy_conv,
// incl. depth targets) and optionally clears the EFB to the game's copy-clear
// color/Z afterwards. The bound draw's geometry is already in the pass (the
// sink flushed the pending draw before firing this).
void copy_efb_to_texture(const gxc::EfbCopyCommand& cmd) {
  if (interp_recording()) {
    g_interpCur.entries.push_back(LogEntry{.copy = true, .copy_cmd = cmd});
    g_interpCopyDests.insert(cmd.dest_address);
    return;
  }
  // Mirror the copy-clear color/Z (BP 0x4F-0x51 at this copy) into the gx
  // state GXSetCopyClear would have written: begin_frame's pass-0 EFB clear
  // reads g_gxState.clearColor/clearDepth, and in gxcore mode the live gx
  // layer that normally maintains them is bypassed. Without this every frame
  // cleared to the default alpha=1 and e.g. the Strikers shadow-grab alpha
  // background inverted (glxSwap display-copy clears to {0,0,0,0}).
  gx::g_gxState.clearColor = {
      static_cast<float>(cmd.clear_r) / 255.f,
      static_cast<float>(cmd.clear_g) / 255.f,
      static_cast<float>(cmd.clear_b) / 255.f,
      static_cast<float>(cmd.clear_a) / 255.f,
  };
  gx::g_gxState.clearDepth = cmd.clear_z;
  if (cmd.format == 0xFu) {
    // Display copy (GXCopyDisp): no texture destination; its requested clear
    // becomes the next frame's EFB clear via the state mirrored above.
    return;
  }
  const auto fmt = static_cast<GXTexFmt>(cmd.format);
  // Source rect in render pixels, each edge rounded inward. At a fractional
  // EFB scale a draw clipped to a viewport edge does not reach the partly
  // covered pixel past it (D3D12 truncates the implicit viewport clip), so
  // rounding outward, as the scissor mapping does, takes in a fringe row or
  // column the game never drew. A clamped sampler reading past the copy's edge
  // (projected shadow masks) then sees the clear colour instead of the copy.
  const gfx::Viewport mapped = gx::map_logical_viewport(gfx::Viewport{
      .left = static_cast<float>(cmd.src_x),
      .top = static_cast<float>(cmd.src_y),
      .width = static_cast<float>(cmd.width),
      .height = static_cast<float>(cmd.height),
  });
  const auto [targetWidth, targetHeight] = gfx::get_render_target_size();
  constexpr float kEdgeSlack = 1e-3f;  // float noise at integer scales
  const auto inward = [](float edge, bool low) {
    return static_cast<int32_t>(low ? std::ceil(edge - kEdgeSlack) : std::floor(edge + kEdgeSlack));
  };
  const int32_t copyLeft = std::clamp(inward(mapped.left, true), 0, static_cast<int32_t>(targetWidth));
  const int32_t copyTop = std::clamp(inward(mapped.top, true), 0, static_cast<int32_t>(targetHeight));
  const int32_t copyRight =
      std::clamp(inward(mapped.left + mapped.width, false), copyLeft, static_cast<int32_t>(targetWidth));
  const int32_t copyBottom =
      std::clamp(inward(mapped.top + mapped.height, false), copyTop, static_cast<int32_t>(targetHeight));
  const gfx::ClipRect srcRect{
      .x = copyLeft,
      .y = copyTop,
      .width = copyRight - copyLeft,
      .height = copyBottom - copyTop,
  };
  const uint32_t dstWidth = static_cast<uint32_t>(std::max(srcRect.width, 1));
  const uint32_t dstHeight = static_cast<uint32_t>(std::max(srcRect.height, 1));

  auto it = g_efbCopyTextures.find(cmd.dest_address);
  // A destination reused for a copy of another size or format (a 384x384
  // shadow mask where a full-frame copy lived) needs a new texture: resolving
  // into the old one leaves the copy at the wrong scale for its samplers.
  const bool conv = gfx::tex_copy_conv::needs_conversion(fmt);
  if (it != g_efbCopyTextures.end() && it->second &&
      (it->second->size.width != dstWidth || it->second->size.height != dstHeight ||
       (conv ? it->second->gxFormat != static_cast<u32>(fmt) : it->second->gxFormat != GX_TF_RGBA8))) {
    g_efbCopyTextures.erase(it);
    it = g_efbCopyTextures.end();
  }
  if (it == g_efbCopyTextures.end() || !it->second) {
    TextureHandle handle;
    if (gfx::tex_copy_conv::needs_conversion(fmt)) {
      handle = gfx::new_conv_texture(dstWidth, dstHeight, fmt, "GXCore Copy Conv");
    } else {
      handle = gfx::new_render_texture(dstWidth, dstHeight, GX_TF_RGBA8,
                                       "GXCore Copy");
    }
    it = g_efbCopyTextures.insert_or_assign(cmd.dest_address, handle).first;
  }
  if (!it->second) {
    return;
  }
  float clearDepthValue = static_cast<float>(cmd.clear_z) / 16777215.f;
  if (gx::UseReversedZ) {
    clearDepthValue = 1.f - clearDepthValue;
  }
  gfx::resolve_pass(it->second, srcRect, cmd.clear && cmd.color_update,
                    cmd.clear && cmd.alpha_update,
                    cmd.clear && cmd.depth_update,
                    Vec4<float>{static_cast<float>(cmd.clear_r) / 255.f,
                                static_cast<float>(cmd.clear_g) / 255.f,
                                static_cast<float>(cmd.clear_b) / 255.f,
                                static_cast<float>(cmd.clear_a) / 255.f},
                    clearDepthValue, fmt);
}

// The texture bind group for a plan's texmaps (0 when untextured); false when a
// referenced texmap does not resolve and the draw is not drawable.
bool bind_plan_textures(const gxc::DrawPlan& plan, BindGroupRef* out) {
  // Resolve one texmap's texture to a GPU handle: EFB-copy shadow first, else the
  // guest-identity + content-hash decode cache. Shared by the single-texmap fast
  // path and the multi-texmap path (63/Mfin, e.g. THP YUV Y/U/V on texmap 0/1/2).
  auto resolve_texture_handle =
      [&](uint32_t address, uint32_t tsize, uint32_t format, uint32_t width,
          uint32_t height, const void* data, uint32_t available, bool has_tlut,
          uint32_t tlut_address, uint32_t tlut_format, uint32_t tlut_entries,
          const void* tlut_data, uint32_t tlut_available) -> TextureHandle {
    auto efbIt = g_efbCopyTextures.find(address);
    if (efbIt != g_efbCopyTextures.end() && efbIt->second) {
      ++g_textureCacheStats.hits;
      return efbIt->second;
    }
    if (data == nullptr)
      return TextureHandle{};
    const auto* bytes = static_cast<const uint8_t*>(data);
    const uint32_t size = std::min(tsize, available);
    const uint32_t frame = current_frame();
    if (frame != g_textureHashMemoFrame) {
      g_textureHashMemo.clear();
      g_textureHashMemoFrame = frame;
    }
    TextureKey key{address,     tsize,        format,      width,
                   height,      tlut_address, tlut_format, tlut_entries,
                   0};
    // Hash the actual source texels (+ TLUT for CI) so a content change under an
    // unchanged guest identity is a distinct cache entry — see TextureKey.
    if (const auto memo = g_textureHashMemo.find(key); memo != g_textureHashMemo.end()) {
      key.content_hash = memo->second;
    } else {
      uint64_t content_hash = XXH3_64bits(bytes, size);
      // Only the palette itself: the resolver's available count runs to the
      // end of guest RAM, and hashing that made every unrelated store a new key
      // (each CI texture re-decoded and re-created every frame).
      if (gxc::is_ci_format(format) && has_tlut && tlut_data != nullptr)
        content_hash = XXH3_64bits_withSeed(tlut_data, std::min(tlut_available, tlut_entries * 2u), content_hash);
      g_textureHashMemo.emplace(key, content_hash);
      key.content_hash = content_hash;
    }
    auto it = g_textureCache.find(key);
    if (it != g_textureCache.end()) {
      ++g_textureCacheStats.hits;
      g_textureAddrKey[address].last_used_frame = frame;
      return it->second;
    }
    // New content for this identity: the prior entry cached at the same guest
    // address is stale (its buffer was overwritten). Evict it to keep the cache
    // bounded, keeping its texture for an in-place rewrite when no draw of
    // this frame has bound it yet.
    TextureHandle reusable;
    auto addrIt = g_textureAddrKey.find(address);
    if (addrIt != g_textureAddrKey.end()) {
      if (auto old = g_textureCache.find(addrIt->second.key); old != g_textureCache.end()) {
        if (addrIt->second.last_used_frame != frame)
          reusable = std::move(old->second);
        g_textureCache.erase(old);
      }
    }
    // gxcore owns the decode for every format: produce tightly-packed RGBA8 and
    // upload it as a pre-decoded PC texture (no substrate re-conversion).
    std::vector<uint8_t> decoded;
    if (gxc::is_ci_format(format)) {
      if (has_tlut && tlut_data != nullptr)
        decoded = gxc::decode_ci(format, width, height, bytes, size, tlut_format,
                                 tlut_entries,
                                 static_cast<const uint8_t*>(tlut_data),
                                 tlut_available);
    } else {
      decoded = gxc::decode_texture(format, width, height, bytes, size);
    }
    TextureHandle handle;
    if (!decoded.empty()) {
      if (reusable && reusable->gxFormat == GX_TF_RGBA8_PC && reusable->size.width == width &&
          reusable->size.height == height && reusable->mipCount == 1) {
        handle = std::move(reusable);
        write_texture(*handle, ArrayRef<uint8_t>{decoded.data(), decoded.size()});
      } else {
        handle = new_static_texture_2d(
            width, height, 1, GX_TF_RGBA8_PC,
            ArrayRef<uint8_t>{decoded.data(), decoded.size()}, false,
            "GXCore Texture");
      }
      ++g_textureCacheStats.uploads;
      if (gxc::is_ci_format(format))
        ++g_textureCacheStats.ci_uploads;
    } else {
      // CI without a resolved palette, or a format gxcore does not decode: upload
      // the raw GX bytes under the original format (old behavior).
      handle = new_static_texture_2d(width, height, 1, format,
                                     ArrayRef<uint8_t>{bytes, size}, false,
                                     "GXCore Texture");
      ++g_textureCacheStats.uploads;
      ++g_textureCacheStats.raw_fallback;
    }
    g_textureCache.emplace(key, handle);
    g_textureAddrKey[address] = AddrEntry{key, frame};
    return handle;
  };

  BindGroupRef textureBindGroup = 0;
  if (plan.pipeline.shader.textured != 0) {
    // The texmap's TX_SETMODE0: wrap S (bits 0-1) and T (2-3) clamp/repeat/
    // mirror, mag filter (bit 4) and min filter (5-7, linear for LINEAR,
    // LIN_MIP_NEAR, LIN_MIP_LIN). Projected shadows rely on clamp: the
    // projector's back side lands outside the texture and must read its edge.
    auto sampler_for = [](bool valid, uint32_t mode0) {
      auto wrap = [](uint32_t w) {
        return w == 0u ? wgpu::AddressMode::ClampToEdge
                       : w == 2u ? wgpu::AddressMode::MirrorRepeat : wgpu::AddressMode::Repeat;
      };
      const uint32_t minf = (mode0 >> 5) & 7u;
      const wgpu::SamplerDescriptor samplerDescriptor{
          .label = "GXCore Sampler",
          .addressModeU = valid ? wrap(mode0 & 3u) : wgpu::AddressMode::Repeat,
          .addressModeV = valid ? wrap((mode0 >> 2) & 3u) : wgpu::AddressMode::Repeat,
          .magFilter = !valid || ((mode0 >> 4) & 1u) ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest,
          .minFilter = !valid || minf == 1u || minf == 3u || minf == 5u ? wgpu::FilterMode::Linear
                                                                        : wgpu::FilterMode::Nearest,
          .mipmapFilter = wgpu::MipmapFilterMode::Nearest,
      };
      return sampler_ref(samplerDescriptor);
    };
    const auto sampler = sampler_for(plan.tex_mode0_valid, plan.tex_mode0);
    if (plan.texmap_mask == 0u) {
      // Single-texmap fast path: primary texture at binding 0/1 (unchanged).
      if (plan.has_texture) {
        TextureHandle bound = resolve_texture_handle(
            plan.tex_address, plan.tex_size, plan.tex_format, plan.tex_width,
            plan.tex_height, plan.tex_data, plan.tex_available, plan.has_tlut,
            plan.tlut_address, plan.tlut_format, plan.tlut_entries,
            plan.tlut_data, plan.tlut_available);
        if (bound) {
          const std::array entries{
              WGPUBindGroupEntry{.binding = 0,
                                 .textureView =
                                     bound->sampleTextureView.Get()},
              WGPUBindGroupEntry{.binding = 1, .sampler = sampler.Get()},
          };
          const WGPUBindGroupDescriptor descriptor{
              .label = {"GXCore Texture Bind Group", WGPU_STRLEN},
              .layout = texture_bind_group_layout(1u).Get(),
              .entryCount = entries.size(),
              .entries = entries.data(),
          };
          textureBindGroup = bind_group_ref(descriptor);
        }
      }
    } else {
      // Multi-texmap: bind each used texmap at 2t (texture) / 2t+1 (sampler),
      // matching the WGSL declarations. If any referenced texmap fails to
      // resolve the draw is not drawable (avoids a missing-bind-group error).
      std::vector<TextureHandle> held;
      std::vector<WGPUBindGroupEntry> entries;
      bool complete = true;
      for (uint32_t t = 0; t < 8u; ++t) {
        if ((plan.texmap_mask & (1u << t)) == 0u)
          continue;
        const gxc::PlanTexture& pt = plan.textures[t];
        TextureHandle bound;
        if (pt.valid)
          bound = resolve_texture_handle(
              pt.address, pt.size, pt.format, pt.width, pt.height, pt.data,
              pt.available, pt.has_tlut, pt.tlut_address, pt.tlut_format,
              pt.tlut_entries, pt.tlut_data, pt.tlut_available);
        if (!bound) {
          complete = false;
          break;
        }
        held.push_back(bound);
        entries.push_back(WGPUBindGroupEntry{
            .binding = 2u * t, .textureView = bound->sampleTextureView.Get()});
        const auto texmap_sampler = sampler_for(pt.mode0_valid, pt.mode0);
        entries.push_back(WGPUBindGroupEntry{.binding = 2u * t + 1u,
                                             .sampler = texmap_sampler.Get()});
      }
      if (!complete)
        return false;
      const WGPUBindGroupDescriptor descriptor{
          .label = {"GXCore Texture Bind Group", WGPU_STRLEN},
          .layout = texture_bind_group_layout(plan.texmap_mask).Get(),
          .entryCount = entries.size(),
          .entries = entries.data(),
      };
      textureBindGroup = bind_group_ref(descriptor);
    }
  }

  *out = textureBindGroup;
  return true;
}

bool submit_draw_plan(const gxc::DrawPlan& plan) {
  if (!plan.ok || plan.vertex_count == 0 || plan.indices.empty()) {
    return false;
  }

  // GX scissor, as Dolphin BPFunctions::ComputeScissorRects and
  // SetScissorAndViewport: closed [left, right] x [top, bottom] in GX
  // coordinates, made EFB-relative by the scissor offset (stored halved; both
  // carry GX's +342, which cancels). Coordinates wrap at 1024, so each axis
  // tries the offset in 1024 steps; of the resulting rectangles the one
  // covering most of the viewport wins (then the largest), and its offset
  // also places the viewport origin.
  gfx::ClipRect scissor{};
  bool has_viewport = false;
  gfx::Viewport viewport{};
  int32_t scissorXOff = 0;
  int32_t scissorYOff = 0;
  {
    const int32_t top = static_cast<int32_t>(plan.scissor_tl & 0x7FFu);
    const int32_t left = static_cast<int32_t>((plan.scissor_tl >> 12) & 0x7FFu);
    const int32_t bottom = static_cast<int32_t>(plan.scissor_br & 0x7FFu);
    const int32_t right = static_cast<int32_t>((plan.scissor_br >> 12) & 0x7FFu);
    if (left > right || top > bottom)
      return true;  // nothing inside the scissor: the draw renders no pixels
    struct Range {
      int32_t offset, start, end;
    };
    // At most one candidate per 1024 step in -4096..4096: a fixed buffer, as
    // this runs for every draw.
    struct Ranges {
      Range items[9];
      int count = 0;
      const Range* begin() const { return items; }
      const Range* end() const { return items + count; }
      bool empty() const { return count == 0; }
    };
    auto ranges = [](int32_t start, int32_t end, int32_t offset, int32_t dim) {
      Ranges out;
      for (int32_t extra = -4096; extra <= 4096; extra += 1024) {
        const int32_t off = offset + extra;
        const int32_t s = std::clamp(start - off, 0, dim);
        const int32_t e = std::clamp(end - off + 1, 0, dim);
        if (s < e)
          out.items[out.count++] = {off, s, e};
      }
      return out;
    };
    const auto xs = ranges(left, right, static_cast<int32_t>(plan.scissor_offset & 0x1FFu) << 1, 640);
    const auto ys = ranges(top, bottom, static_cast<int32_t>((plan.scissor_offset >> 10) & 0x1FFu) << 1, 528);
    if (xs.empty() || ys.empty())
      return true;
    // Viewport extent in raw GX coordinates (origin includes the +342).
    float vl = -1e9f, vr = 1e9f, vt = -1e9f, vb = 1e9f;
    if (plan.viewport_valid) {
      vl = std::min(plan.viewport[3] - plan.viewport[0], plan.viewport[3] + plan.viewport[0]);
      vr = std::max(plan.viewport[3] - plan.viewport[0], plan.viewport[3] + plan.viewport[0]);
      vt = std::min(plan.viewport[4] - plan.viewport[1], plan.viewport[4] + plan.viewport[1]);
      vb = std::max(plan.viewport[4] - plan.viewport[1], plan.viewport[4] + plan.viewport[1]);
    }
    auto viewport_area = [&](const Range& x, const Range& y) {
      const float x0 = std::clamp(static_cast<float>(x.start + x.offset), vl, vr);
      const float x1 = std::clamp(static_cast<float>(x.end + x.offset), vl, vr);
      const float y0 = std::clamp(static_cast<float>(y.start + y.offset), vt, vb);
      const float y1 = std::clamp(static_cast<float>(y.end + y.offset), vt, vb);
      return (x1 - x0) * (y1 - y0);
    };
    const Range* bx = nullptr;
    const Range* by = nullptr;
    float best_vp = -1.f;
    int64_t best_area = -1;
    for (const auto& x : xs) {
      for (const auto& y : ys) {
        const float vp = viewport_area(x, y);
        const int64_t area = int64_t(x.end - x.start) * (y.end - y.start);
        if (vp > best_vp || (vp == best_vp && area > best_area)) {
          best_vp = vp;
          best_area = area;
          bx = &x;
          by = &y;
        }
      }
    }
    scissorXOff = bx->offset;
    scissorYOff = by->offset;
    scissor = (
        gfx::ClipRect{.x = bx->start, .y = by->start, .width = bx->end - bx->start, .height = by->end - by->start});
  }

  if (plan.viewport_valid) {
    // Raw XF viewport -> logical viewport, the raw path's own formula
    // (gx/command_processor.cpp XF 0x1A case), with the origin taken
    // from the scissor offset like Dolphin SetScissorAndViewport.
    const float sx = plan.viewport[0];
    const float sy = plan.viewport[1];
    const float sz = plan.viewport[2];
    const float ox = plan.viewport[3];
    const float oy = plan.viewport[4];
    const float oz = plan.viewport[5];
    const float width = sx * 2.0f;
    const float height = -sy * 2.0f;
    has_viewport = true;
    viewport = gfx::Viewport({
        .left = ox - static_cast<float>(scissorXOff) - width / 2.0f,
        .top = oy - static_cast<float>(scissorYOff) - height / 2.0f,
        .width = width,
        .height = height,
        .znear = (oz - sz) / 1.6777215e7f,
        .zfar = oz / 1.6777215e7f,
    });
  }

  BindGroupRef textureBindGroup = 0;
  if (!bind_plan_textures(plan, &textureBindGroup))
    return false;

  const bool fetch = plan.pipeline.shader.fetch.enabled != 0;
  if (!fetch && interp_recording()) {
    const double rec_start = g_interpLog ? interp_now() : 0;
    if (g_interpPool.empty()) {
      g_interpCur.draws.emplace_back();
    } else {
      g_interpCur.draws.push_back(std::move(g_interpPool.back()));
      g_interpPool.pop_back();
    }
    LoggedDraw& d = g_interpCur.draws.back();
    d.scissor = scissor;
    d.has_viewport = has_viewport;
    d.viewport = viewport;
    d.pipeline = pipeline_ref(PipelineConfig{
        .version = GXCorePipelineConfigVersion,
        .key = plan.pipeline,
        .msaaSamples = get_sample_count(),
    });
    d.textureBindGroup = textureBindGroup;
    if (plan.pipeline.shader.textured != 0) {
      auto from_copy = [](uint32_t address) {
        return g_interpCopyDests.contains(address) || g_efbCopyTextures.contains(address);
      };
      bool copy_sourced = plan.texmap_mask == 0u && plan.has_texture && from_copy(plan.tex_address);
      for (uint32_t t = 0; t < 8u && !copy_sourced; ++t)
        copy_sourced = (plan.texmap_mask & (1u << t)) != 0u && plan.textures[t].valid &&
                       from_copy(plan.textures[t].address);
      if (copy_sourced)
        d.rebind = std::make_unique<gxc::DrawPlan>(plan);
    }
    d.vertices.assign(plan.vertices.begin(), plan.vertices.end());
    d.indices.assign(plan.indices.begin(), plan.indices.end());
    d.constants = plan.constants;
    d.pixel_constants = plan.pixel_constants;
    d.tev = plan.pipeline.shader.tev_valid != 0;
    d.pos_mtx_idx = plan.pipeline.shader.has_pos_mtx_idx != 0;
    d.key = XXH3_64bits_withSeed(plan.vertices.data(), plan.vertices.size() * sizeof(float),
                                 plan.indices.size() * 2u + d.pos_mtx_idx);
    g_interpCur.entries.push_back(LogEntry{.draw = static_cast<uint32_t>(g_interpCur.draws.size() - 1)});
    if (g_interpLog)
      g_interpTiming.record += interp_now() - rec_start;
    return true;
  }
  gx::set_logical_scissor(scissor);
  if (has_viewport)
    gx::set_logical_viewport(viewport);
  Range vertRange{};
  if (fetch) {
    // GPU vertex fetch: the raw GX vertex bytes into the vertex buffer and each
    // indexed array into the storage buffer, where the shader reads them by
    // byte offset (group 0, the static bind group). An array several draws of
    // a frame share goes up once. The offsets go into the plan's uniform; the
    // plan is the sink's own scratch, filled for this submission.
    auto& constants = const_cast<gxc::VertexShaderConstants&>(plan.constants);
    vertRange = push_verts(plan.fetch_payload, plan.fetch_payload_size, 4);
    constants.fetch_base[0] = vertRange.offset;
    const uint32_t frame = current_frame();
    if (frame != g_fetchArraysFrame) {
      g_fetchArrays.clear();
      g_fetchArraysFrame = frame;
    }
    for (uint32_t attr = 0; attr < 16u; ++attr) {
      const auto& array = plan.fetch_arrays[attr];
      if (array.data == nullptr)
        continue;
      // Keyed by the host pointer (one per guest array per RAM snapshot): each
      // draw asks for the span its own indices reach, so an upload is reused
      // while it covers the request and replaced by a longer one when not.
      auto& upload = g_fetchArrays[reinterpret_cast<uintptr_t>(array.data)];
      if (upload.size < array.size)
        upload = {push_storage(static_cast<const uint8_t*>(array.data), array.size).offset, array.size};
      constants.fetch_arrays[attr][0] = upload.offset;
    }
  } else {
    vertRange = push_verts(reinterpret_cast<const uint8_t*>(plan.vertices.data()),
                           plan.vertices.size() * sizeof(float), 4);
  }
  const auto idxRange = push_indices(
      reinterpret_cast<const uint8_t*>(plan.indices.data()),
      plan.indices.size() * sizeof(uint16_t), 4);
  const auto uniformRange = push_uniform(
      reinterpret_cast<const uint8_t*>(&plan.constants),
      sizeof(plan.constants));
  const bool tev = plan.pipeline.shader.tev_valid != 0;
  Range pixelUniformRange{};
  if (tev) {
    pixelUniformRange = push_uniform(
        reinterpret_cast<const uint8_t*>(&plan.pixel_constants),
        sizeof(plan.pixel_constants));
  }

  push_draw_command(DrawData{
      .pipeline = pipeline_ref(PipelineConfig{
          .version = GXCorePipelineConfigVersion,
          .key = plan.pipeline,
          .msaaSamples = get_sample_count(),
      }),
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = uniformRange,
      .pixelUniformRange = pixelUniformRange,
      .indexCount = static_cast<uint32_t>(plan.indices.size()),
      .textureBindGroup = textureBindGroup,
      .tev = tev,
      .fetch = fetch,
  });
  return true;
}

namespace {
float as_f(const float* m, int w) { return m[w]; }

// Dolphin's IsMatrixCut on two 3x4 matrices (rows of 4): the change D =
// now * prev^-1 is the camera's motion for a static object; a large turn or a
// jump large next to the object's distance is a new view, not motion.
bool matrix_cut(const float* now, const float* prev) {
  auto A = [](const float* m, int r, int k) { return double(m[r * 4 + k]); };
  const double det = A(prev, 0, 0) * (A(prev, 1, 1) * A(prev, 2, 2) - A(prev, 1, 2) * A(prev, 2, 1)) -
                     A(prev, 0, 1) * (A(prev, 1, 0) * A(prev, 2, 2) - A(prev, 1, 2) * A(prev, 2, 0)) +
                     A(prev, 0, 2) * (A(prev, 1, 0) * A(prev, 2, 1) - A(prev, 1, 1) * A(prev, 2, 0));
  if (!std::isfinite(det) || std::fabs(det) < 1e-9)
    return std::memcmp(now, prev, 12 * sizeof(float)) != 0;
  double inv[3][3];
  inv[0][0] = (A(prev, 1, 1) * A(prev, 2, 2) - A(prev, 1, 2) * A(prev, 2, 1)) / det;
  inv[0][1] = (A(prev, 0, 2) * A(prev, 2, 1) - A(prev, 0, 1) * A(prev, 2, 2)) / det;
  inv[0][2] = (A(prev, 0, 1) * A(prev, 1, 2) - A(prev, 0, 2) * A(prev, 1, 1)) / det;
  inv[1][0] = (A(prev, 1, 2) * A(prev, 2, 0) - A(prev, 1, 0) * A(prev, 2, 2)) / det;
  inv[1][1] = (A(prev, 0, 0) * A(prev, 2, 2) - A(prev, 0, 2) * A(prev, 2, 0)) / det;
  inv[1][2] = (A(prev, 0, 2) * A(prev, 1, 0) - A(prev, 0, 0) * A(prev, 1, 2)) / det;
  inv[2][0] = (A(prev, 1, 0) * A(prev, 2, 1) - A(prev, 1, 1) * A(prev, 2, 0)) / det;
  inv[2][1] = (A(prev, 0, 1) * A(prev, 2, 0) - A(prev, 0, 0) * A(prev, 2, 1)) / det;
  inv[2][2] = (A(prev, 0, 0) * A(prev, 1, 1) - A(prev, 0, 1) * A(prev, 1, 0)) / det;
  double d[3][3];
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 3; ++k)
      d[r][k] = A(now, r, 0) * inv[0][k] + A(now, r, 1) * inv[1][k] + A(now, r, 2) * inv[2][k];
  const double d_det = d[0][0] * (d[1][1] * d[2][2] - d[1][2] * d[2][1]) -
                       d[0][1] * (d[1][0] * d[2][2] - d[1][2] * d[2][0]) +
                       d[0][2] * (d[1][0] * d[2][1] - d[1][1] * d[2][0]);
  if (!std::isfinite(d_det) || d_det <= 0)
    return true;
  const double scale = std::cbrt(d_det);
  if (scale < 0.5 || scale > 2.0)
    return true;
  const double cos_angle = ((d[0][0] + d[1][1] + d[2][2]) / scale - 1.0) * 0.5;
  constexpr double kMaxTurn = 0.6;  // rad per game frame
  if (cos_angle < std::cos(kMaxTurn))
    return true;
  double jump = 0, distance = 0;
  for (int r = 0; r < 3; ++r) {
    const double moved = A(now, r, 3) - (d[r][0] * A(prev, 0, 3) + d[r][1] * A(prev, 1, 3) + d[r][2] * A(prev, 2, 3));
    jump += moved * moved;
    distance = std::max({distance, std::fabs(A(now, r, 3)), std::fabs(A(prev, r, 3))});
  }
  return std::sqrt(jump) > 0.3 * distance + 2.0;
}

// The matrix a draw positions with when it has no per-vertex index.
const float* draw_matrix(const LoggedDraw& d) { return &d.constants.posnormalmatrix[0][0]; }

double matrix_distance(const float* a, const float* b) {
  double d = 0;
  for (int w = 0; w < 12; ++w) {
    const double delta = double(a[w]) - b[w];
    d += delta * delta;
  }
  return d;
}

// Pairs every draw of the current frame with one of the previous frame's.
// Several draws can share vertex data (the battle HUD draws every sprite from
// one quad): take the unused one nearest in matrix, and only a close match.
void pair_log(InterpLog& cur, const InterpLog& prev) {
  absl::flat_hash_map<uint64_t, std::vector<uint32_t>> by_key;
  for (uint32_t i = 0; i < prev.draws.size(); ++i)
    by_key[prev.draws[i].key].push_back(i);
  std::vector<bool> taken(prev.draws.size(), false);
  for (LoggedDraw& d : cur.draws) {
    d.pair = -1;
    const auto it = by_key.find(d.key);
    if (it == by_key.end())
      continue;
    const float* now = draw_matrix(d);
    double norm = 0;
    for (int w = 0; w < 12; ++w)
      norm += double(now[w]) * now[w];
    double best = 0;
    int32_t best_index = -1;
    for (const uint32_t c : it->second) {
      if (taken[c])
        continue;
      const double dist = it->second.size() == 1 ? 0.0 : matrix_distance(now, draw_matrix(prev.draws[c]));
      if (best_index < 0 || dist < best) {
        best = dist;
        best_index = static_cast<int32_t>(c);
      }
    }
    if (best_index >= 0 && (it->second.size() == 1 || std::sqrt(best) <= 0.05 * std::sqrt(norm) + 0.5)) {
      taken[best_index] = true;
      d.pair = best_index;
    }
  }
  cur.paired = true;
}

void lerp_matrix(float* out, const float* prev, const float* now, float t) {
  if (std::memcmp(prev, now, 12 * sizeof(float)) == 0 || matrix_cut(now, prev))
    return;  // out already holds now
  for (int w = 0; w < 12; ++w)
    out[w] = prev[w] + (now[w] - prev[w]) * t;
}

void emit_draw(const LoggedDraw& d, const gxc::VertexShaderConstants& constants) {
  BindGroupRef textureBindGroup = d.textureBindGroup;
  if (d.rebind && !bind_plan_textures(*d.rebind, &textureBindGroup))
    return;
  gx::set_logical_scissor(d.scissor);
  if (d.has_viewport)
    gx::set_logical_viewport(d.viewport);
  const auto vertRange = push_verts(reinterpret_cast<const uint8_t*>(d.vertices.data()),
                                    d.vertices.size() * sizeof(float), 4);
  const auto idxRange = push_indices(reinterpret_cast<const uint8_t*>(d.indices.data()),
                                     d.indices.size() * sizeof(uint16_t), 4);
  const auto uniformRange = push_uniform(reinterpret_cast<const uint8_t*>(&constants), sizeof(constants));
  Range pixelUniformRange{};
  if (d.tev)
    pixelUniformRange =
        push_uniform(reinterpret_cast<const uint8_t*>(&d.pixel_constants), sizeof(d.pixel_constants));
  push_draw_command(DrawData{
      .pipeline = d.pipeline,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = uniformRange,
      .pixelUniformRange = pixelUniformRange,
      .indexCount = static_cast<uint32_t>(d.indices.size()),
      .textureBindGroup = textureBindGroup,
      .tev = d.tev,
      .fetch = false,
  });
}
}  // namespace

uint32_t interp_layers() { return g_interpLayers; }

void interp_set_active(uint32_t layers) {
  g_interpActive = std::min(layers, g_interpLayers);
  if (g_interpActive <= 1) {
    // Live from the next frame: nothing to pair the first logged one with.
    g_interpCur.clear();
    g_interpPrev.clear();
  }
}

bool interp_active() { return g_interpActive > 1; }

void interp_replay(float t) {
  InterpLog& cur = g_interpCur;
  const InterpLog& prev = g_interpPrev;
  const double t0 = g_interpLog ? interp_now() : 0;
  if (t < 1.f && !cur.paired)
    pair_log(cur, prev);
  const double t1 = g_interpLog ? interp_now() : 0;
  g_interpTiming.pair += t1 - t0;
  g_interpReplaying = true;
  gxc::VertexShaderConstants blended;
  for (const LogEntry& e : cur.entries) {
    if (e.copy) {
      copy_efb_to_texture(e.copy_cmd);
      continue;
    }
    const LoggedDraw& d = cur.draws[e.draw];
    // 2D draws (orthographic: projection row 3 is 0 0 0 1) gain nothing.
    const bool ortho = d.constants.projection[3][3] != 0.f;
    if (t >= 1.f || d.pair < 0 || ortho) {
      emit_draw(d, d.constants);
      continue;
    }
    const LoggedDraw& p = prev.draws[d.pair];
    blended = d.constants;
    if (d.pos_mtx_idx) {
      for (int m = 0; m < 10; ++m)
        lerp_matrix(&blended.transformmatrices[m * 3][0], &p.constants.transformmatrices[m * 3][0],
                    &d.constants.transformmatrices[m * 3][0], t);
    } else {
      lerp_matrix(&blended.posnormalmatrix[0][0], &p.constants.posnormalmatrix[0][0],
                  &d.constants.posnormalmatrix[0][0], t);
    }
    emit_draw(d, blended);
  }
  g_interpReplaying = false;
  if (g_interpLog)
    g_interpTiming.replay += interp_now() - t1;
}

void interp_finish_frame() {
  if (g_interpLog) {
    InterpTiming& m = g_interpTiming;
    ++m.frames;
    m.draws += g_interpCur.draws.size();
    for (const auto& e : g_interpCur.entries) m.copies += e.copy;
    for (const auto& d : g_interpCur.draws) m.rebinds += d.rebind != nullptr;
    if (m.frames == 120) {
      std::fprintf(stderr, "[interp] per frame: record %.2f ms, pair %.2f ms, replay %.2f ms (all layers); draws %.0f, copies %.1f, rebinds %.1f\n",
                   m.record * 1e3 / 120, m.pair * 1e3 / 120, m.replay * 1e3 / 120, m.draws / 120.0, m.copies / 120.0, m.rebinds / 120.0);
      m = {};
    }
  }
  std::swap(g_interpPrev, g_interpCur);
  g_interpCur.clear();
  g_interpCopyDests.clear();
}

} // namespace aurora::gfx::gxcore
