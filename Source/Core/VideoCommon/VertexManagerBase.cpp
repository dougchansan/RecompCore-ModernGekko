// Copyright 2010 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/VertexManagerBase.h"

#include <array>
#include <cmath>
#include <memory>
#include <limits>
#include <string>
#include <cstdlib>
#include <string_view>

#include "Common/ChunkFile.h"
#include "Common/CommonTypes.h"
#include "Common/Contains.h"
#include "Common/EnumMap.h"
#include "Common/Logging/Log.h"
#include "Common/MathUtil.h"
#include "Common/SmallVector.h"

#include "Core/DolphinAnalytics.h"
#include "Core/HW/SystemTimers.h"
#include "Core/System.h"

#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/BPMemory.h"
#include "VideoCommon/BPFunctions.h"
#include "VideoCommon/BoundingBox.h"
#include "VideoCommon/DataReader.h"
#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/GeometryShaderManager.h"
#include "VideoCommon/GraphicsModSystem/Runtime/CustomShaderCache.h"
#include "VideoCommon/GraphicsModSystem/Runtime/GraphicsModActionData.h"
#include "VideoCommon/GraphicsModSystem/Runtime/GraphicsModManager.h"
#include "VideoCommon/IndexGenerator.h"
#include "VideoCommon/NativeVertexFormat.h"
#include "VideoCommon/OpcodeDecoding.h"
#include "VideoCommon/PerfQueryBase.h"
#include "VideoCommon/PixelShaderGen.h"
#include "VideoCommon/PixelShaderManager.h"
#include "VideoCommon/Statistics.h"
#include "VideoCommon/TextureCacheBase.h"
#include "VideoCommon/VertexLoaderManager.h"
#include "VideoCommon/VertexShaderManager.h"
#include "VideoCommon/VideoBackendBase.h"
#include "VideoCommon/VideoCommon.h"
#include "VideoCommon/VideoConfig.h"
#include "VideoCommon/XFMemory.h"
#include "VideoCommon/XFStateManager.h"
#ifdef HAS_VULKAN
#include "VideoBackends/Vulkan/VKGfx.h"
#include "VideoBackends/Vulkan/StateTracker.h"
#include "VideoCommon/ModernGekkoTemporalState.h"
#ifdef Always
#undef Always
#endif
#endif

std::unique_ptr<VertexManagerBase> g_vertex_manager;

using OpcodeDecoder::Primitive;

// GX primitive -> RenderState primitive, no primitive restart
constexpr Common::EnumMap<PrimitiveType, Primitive::GX_DRAW_POINTS> primitive_from_gx{
    PrimitiveType::Triangles,  // GX_DRAW_QUADS
    PrimitiveType::Triangles,  // GX_DRAW_QUADS_2
    PrimitiveType::Triangles,  // GX_DRAW_TRIANGLES
    PrimitiveType::Triangles,  // GX_DRAW_TRIANGLE_STRIP
    PrimitiveType::Triangles,  // GX_DRAW_TRIANGLE_FAN
    PrimitiveType::Lines,      // GX_DRAW_LINES
    PrimitiveType::Lines,      // GX_DRAW_LINE_STRIP
    PrimitiveType::Points,     // GX_DRAW_POINTS
};

// GX primitive -> RenderState primitive, using primitive restart
constexpr Common::EnumMap<PrimitiveType, Primitive::GX_DRAW_POINTS> primitive_from_gx_pr{
    PrimitiveType::TriangleStrip,  // GX_DRAW_QUADS
    PrimitiveType::TriangleStrip,  // GX_DRAW_QUADS_2
    PrimitiveType::TriangleStrip,  // GX_DRAW_TRIANGLES
    PrimitiveType::TriangleStrip,  // GX_DRAW_TRIANGLE_STRIP
    PrimitiveType::TriangleStrip,  // GX_DRAW_TRIANGLE_FAN
    PrimitiveType::Lines,          // GX_DRAW_LINES
    PrimitiveType::Lines,          // GX_DRAW_LINE_STRIP
    PrimitiveType::Points,         // GX_DRAW_POINTS
};

// Due to the BT.601 standard which the GameCube is based on being a compromise
// between PAL and NTSC, neither standard gets square pixels. They are each off
// by ~9% in opposite directions.
// Just in case any game decides to take this into account, we do both these
// tests with a large amount of slop.

static float CalculateProjectionViewportRatio(const Projection::Raw& projection,
                                              const Viewport& viewport)
{
  const float projection_ar = projection[2] / projection[0];
  const float viewport_ar = viewport.wd / viewport.ht;

  return std::abs(projection_ar / viewport_ar);
}

static bool IsAnamorphicProjection(const Projection::Raw& projection, const Viewport& viewport,
                                   const VideoConfig& config)
{
  // If ratio between our projection and viewport aspect ratios is similar to 16:9 / 4:3
  // we have an anamorphic projection. This value can be overridden by a GameINI.
  // Game cheats that change the aspect ratio to natively unsupported ones
  // won't be automatically recognized here.

  return std::abs(CalculateProjectionViewportRatio(projection, viewport) -
                  config.widescreen_heuristic_widescreen_ratio) <
         config.widescreen_heuristic_aspect_ratio_slop;
}

static bool IsNormalProjection(const Projection::Raw& projection, const Viewport& viewport,
                               const VideoConfig& config)
{
  return std::abs(CalculateProjectionViewportRatio(projection, viewport) -
                  config.widescreen_heuristic_standard_ratio) <
         config.widescreen_heuristic_aspect_ratio_slop;
}

VertexManagerBase::VertexManagerBase()
    : m_cpu_vertex_buffer(MAXVBUFFERSIZE), m_cpu_index_buffer(MAXIBUFFERSIZE)
{
}

VertexManagerBase::~VertexManagerBase() = default;

bool VertexManagerBase::Initialize()
{
  auto& video_events = GetVideoEvents();

  m_frame_end_event =
      video_events.after_frame_event.Register([this](Core::System&) { OnEndFrame(); });
  m_after_present_event = video_events.after_present_event.Register(
      [this](const PresentInfo& pi) { m_ticks_elapsed = pi.emulated_timestamp; });
  m_index_generator.Init();
  m_custom_shader_cache = std::make_unique<CustomShaderCache>();
  m_cpu_cull.Init();
  return true;
}

u32 VertexManagerBase::GetRemainingSize() const
{
  return static_cast<u32>(m_end_buffer_pointer - m_cur_buffer_pointer);
}

void VertexManagerBase::AddIndices(OpcodeDecoder::Primitive primitive, u32 num_vertices)
{
  m_index_generator.AddIndices(primitive, num_vertices);
}

bool VertexManagerBase::AreAllVerticesCulled(VertexLoaderBase* loader,
                                             OpcodeDecoder::Primitive primitive, const u8* src,
                                             u32 count)
{
  return m_cpu_cull.AreAllVerticesCulled(loader, primitive, src, count);
}

DataReader VertexManagerBase::PrepareForAdditionalData(OpcodeDecoder::Primitive primitive,
                                                       u32 count, u32 stride, bool cullall)
{
  // Flush all EFB pokes. Since the buffer is shared, we can't draw pokes+primitives concurrently.
  g_framebuffer_manager->FlushEFBPokes();

  // The SSE vertex loader can write up to 4 bytes past the end
  u32 const needed_vertex_bytes = count * stride + 4;

  // We can't merge different kinds of primitives, so we have to flush here
  PrimitiveType new_primitive_type = g_backend_info.bSupportsPrimitiveRestart ?
                                         primitive_from_gx_pr[primitive] :
                                         primitive_from_gx[primitive];
  if (m_current_primitive_type != new_primitive_type) [[unlikely]]
  {
    Flush();

    // Have to update the rasterization state for point/line cull modes.
    m_current_primitive_type = new_primitive_type;
    SetRasterizationStateChanged();
  }

  u32 remaining_indices = GetRemainingIndices(primitive);
  u32 remaining_index_generator_indices = m_index_generator.GetRemainingIndices(primitive);

  // Check for size in buffer, if the buffer gets full, call Flush()
  if (!m_is_flushed && (count > remaining_index_generator_indices || count > remaining_indices ||
                        needed_vertex_bytes > GetRemainingSize())) [[unlikely]]
  {
    Flush();
  }

  m_cull_all = cullall;

  // need to alloc new buffer
  if (m_is_flushed) [[unlikely]]
  {
    if (cullall)
    {
      // This buffer isn't getting sent to the GPU. Just allocate it on the cpu.
      m_cur_buffer_pointer = m_base_buffer_pointer = m_cpu_vertex_buffer.data();
      m_end_buffer_pointer = m_base_buffer_pointer + m_cpu_vertex_buffer.size();
      m_index_generator.Start(m_cpu_index_buffer.data());
    }
    else
    {
      ResetBuffer(stride);
    }

    remaining_index_generator_indices = m_index_generator.GetRemainingIndices(primitive);
    remaining_indices = GetRemainingIndices(primitive);
    m_is_flushed = false;
  }

  // Now that we've reset the buffer, there should be enough space. It's possible that we still
  // won't have enough space in a few rare cases, such as vertex shader line/point expansion with a
  // ton of lines in one draw command, in which case we will either need to add support for
  // splitting a single draw command into multiple draws or using bigger indices.
  ASSERT_MSG(VIDEO, count <= remaining_index_generator_indices,
             "VertexManager: Too few remaining index values ({} > {}). "
             "32-bit indices or primitive breaking needed.",
             count, remaining_index_generator_indices);
  ASSERT_MSG(VIDEO, count <= remaining_indices,
             "VertexManager: Buffer not large enough for all indices! ({} > {}) "
             "Increase MAXIBUFFERSIZE or we need primitive breaking after all.",
             count, remaining_indices);
  ASSERT_MSG(VIDEO, needed_vertex_bytes <= GetRemainingSize(),
             "VertexManager: Buffer not large enough for all vertices! ({} > {}) "
             "Increase MAXVBUFFERSIZE or we need primitive breaking after all.",
             needed_vertex_bytes, GetRemainingSize());

  return DataReader(m_cur_buffer_pointer, m_end_buffer_pointer);
}

DataReader VertexManagerBase::DisableCullAll(u32 stride)
{
  if (m_cull_all)
  {
    m_cull_all = false;
    ResetBuffer(stride);
  }
  return DataReader(m_cur_buffer_pointer, m_end_buffer_pointer);
}

void VertexManagerBase::FlushData(u32 count, u32 stride)
{
  m_cur_buffer_pointer += count * stride;
}

u32 VertexManagerBase::GetRemainingIndices(OpcodeDecoder::Primitive primitive) const
{
  const u32 index_len = MAXIBUFFERSIZE - m_index_generator.GetIndexLen();

  if (primitive >= Primitive::GX_DRAW_LINES)
  {
    if (g_Config.UseVSForLinePointExpand())
    {
      if (g_backend_info.bSupportsPrimitiveRestart)
      {
        switch (primitive)
        {
        case Primitive::GX_DRAW_LINES:
          return index_len / 5 * 2;
        case Primitive::GX_DRAW_LINE_STRIP:
          return index_len / 5 + 1;
        case Primitive::GX_DRAW_POINTS:
          return index_len / 5;
        default:
          return 0;
        }
      }
      else
      {
        switch (primitive)
        {
        case Primitive::GX_DRAW_LINES:
          return index_len / 6 * 2;
        case Primitive::GX_DRAW_LINE_STRIP:
          return index_len / 6 + 1;
        case Primitive::GX_DRAW_POINTS:
          return index_len / 6;
        default:
          return 0;
        }
      }
    }
    else
    {
      switch (primitive)
      {
      case Primitive::GX_DRAW_LINES:
        return index_len;
      case Primitive::GX_DRAW_LINE_STRIP:
        return index_len / 2 + 1;
      case Primitive::GX_DRAW_POINTS:
        return index_len;
      default:
        return 0;
      }
    }
  }
  else if (g_backend_info.bSupportsPrimitiveRestart)
  {
    switch (primitive)
    {
    case Primitive::GX_DRAW_QUADS:
    case Primitive::GX_DRAW_QUADS_2:
      return index_len / 5 * 4;
    case Primitive::GX_DRAW_TRIANGLES:
      return index_len / 4 * 3;
    case Primitive::GX_DRAW_TRIANGLE_STRIP:
      return index_len / 1 - 1;
    case Primitive::GX_DRAW_TRIANGLE_FAN:
      return index_len / 6 * 4 + 1;
    default:
      return 0;
    }
  }
  else
  {
    switch (primitive)
    {
    case Primitive::GX_DRAW_QUADS:
    case Primitive::GX_DRAW_QUADS_2:
      return index_len / 6 * 4;
    case Primitive::GX_DRAW_TRIANGLES:
      return index_len;
    case Primitive::GX_DRAW_TRIANGLE_STRIP:
      return index_len / 3 + 2;
    case Primitive::GX_DRAW_TRIANGLE_FAN:
      return index_len / 3 + 2;
    default:
      return 0;
    }
  }
}

auto VertexManagerBase::ResetFlushAspectRatioCount() -> FlushStatistics
{
  const auto result = m_flush_statistics;
  m_flush_statistics = {};
  return result;
}

void VertexManagerBase::ResetBuffer(u32 vertex_stride)
{
  m_base_buffer_pointer = m_cpu_vertex_buffer.data();
  m_cur_buffer_pointer = m_cpu_vertex_buffer.data();
  m_end_buffer_pointer = m_base_buffer_pointer + m_cpu_vertex_buffer.size();
  m_index_generator.Start(m_cpu_index_buffer.data());
}

void VertexManagerBase::CommitBuffer(u32 num_vertices, u32 vertex_stride, u32 num_indices,
                                     u32* out_base_vertex, u32* out_base_index)
{
  *out_base_vertex = 0;
  *out_base_index = 0;
}

void VertexManagerBase::DrawCurrentBatch(u32 base_index, u32 num_indices, u32 base_vertex)
{
  // If bounding box is enabled, we need to flush any changes first, then invalidate what we have.
  if (g_bounding_box->IsEnabled() && g_ActiveConfig.bBBoxEnable && g_backend_info.bSupportsBBox)
  {
    g_bounding_box->Flush();
  }

  g_gfx->DrawIndexed(base_index, num_indices, base_vertex);
}

void VertexManagerBase::UploadUniforms()
{
}

void VertexManagerBase::InvalidateConstants()
{
  auto& system = Core::System::GetInstance();
  auto& vertex_shader_manager = system.GetVertexShaderManager();
  auto& geometry_shader_manager = system.GetGeometryShaderManager();
  auto& pixel_shader_manager = system.GetPixelShaderManager();
  vertex_shader_manager.dirty = true;
  geometry_shader_manager.dirty = true;
  pixel_shader_manager.dirty = true;
}

void VertexManagerBase::UploadUtilityUniforms(const void* uniforms, u32 uniforms_size)
{
}

void VertexManagerBase::UploadUtilityVertices(const void* vertices, u32 vertex_stride,
                                              u32 num_vertices, const u16* indices, u32 num_indices,
                                              u32* out_base_vertex, u32* out_base_index)
{
  // The GX vertex list should be flushed before any utility draws occur.
  ASSERT(m_is_flushed);

  // Copy into the buffers usually used for GX drawing.
  ResetBuffer(std::max(vertex_stride, 1u));
  if (vertices)
  {
    const u32 copy_size = vertex_stride * num_vertices;
    ASSERT((m_cur_buffer_pointer + copy_size) <= m_end_buffer_pointer);
    std::memcpy(m_cur_buffer_pointer, vertices, copy_size);
    m_cur_buffer_pointer += copy_size;
  }
  if (indices)
    m_index_generator.AddExternalIndices(indices, num_indices, num_vertices);

  CommitBuffer(num_vertices, vertex_stride, num_indices, out_base_vertex, out_base_index);
}

u32 VertexManagerBase::GetTexelBufferElementSize(TexelBufferFormat buffer_format)
{
  // R8 - 1, R16 - 2, RGBA8 - 4, R32G32 - 8
  return 1u << static_cast<u32>(buffer_format);
}

bool VertexManagerBase::UploadTexelBuffer(const void* data, u32 data_size, TexelBufferFormat format,
                                          u32* out_offset)
{
  return false;
}

bool VertexManagerBase::UploadTexelBuffer(const void* data, u32 data_size, TexelBufferFormat format,
                                          u32* out_offset, const void* palette_data,
                                          u32 palette_size, TexelBufferFormat palette_format,
                                          u32* palette_offset)
{
  return false;
}

BitSet32 VertexManagerBase::UsedTextures() const
{
  BitSet32 usedtextures;
  for (u32 i = 0; i < bpmem.genMode.numtevstages + 1u; ++i)
    if (bpmem.tevorders[i / 2].getEnable(i & 1))
      usedtextures[bpmem.tevorders[i / 2].getTexMap(i & 1)] = true;

  if (bpmem.genMode.numindstages > 0)
    for (unsigned int i = 0; i < bpmem.genMode.numtevstages + 1u; ++i)
      if (bpmem.tevind[i].IsActive() && bpmem.tevind[i].bt < bpmem.genMode.numindstages)
        usedtextures[bpmem.tevindref.getTexMap(bpmem.tevind[i].bt)] = true;

  return usedtextures;
}

void VertexManagerBase::Flush()
{
  if (m_is_flushed)
    return;

  m_is_flushed = true;

  if (m_draw_counter == 0)
  {
    // This is more or less the start of the Frame
    GetVideoEvents().before_frame_event.Trigger();
  }

  if (xfmem.numTexGen.numTexGens != bpmem.genMode.numtexgens ||
      xfmem.numChan.numColorChans != bpmem.genMode.numcolchans)
  {
    ERROR_LOG_FMT(
        VIDEO,
        "Mismatched configuration between XF and BP stages - {}/{} texgens, {}/{} colors. "
        "Skipping draw. Please report on the issue tracker.",
        xfmem.numTexGen.numTexGens, bpmem.genMode.numtexgens.Value(), xfmem.numChan.numColorChans,
        bpmem.genMode.numcolchans.Value());

    // Analytics reporting so we can discover which games have this problem, that way when we
    // eventually simulate the behavior we have test cases for it.
    if (xfmem.numTexGen.numTexGens != bpmem.genMode.numtexgens)
    {
      DolphinAnalytics::Instance().ReportGameQuirk(GameQuirk::MismatchedGPUTexGensBetweenXFAndBP);
    }
    if (xfmem.numChan.numColorChans != bpmem.genMode.numcolchans)
    {
      DolphinAnalytics::Instance().ReportGameQuirk(GameQuirk::MismatchedGPUColorsBetweenXFAndBP);
    }

    return;
  }

#if defined(_DEBUG) || defined(DEBUGFAST)
  PRIM_LOG("frame{}:\n texgen={}, numchan={}, dualtex={}, ztex={}, cole={}, alpe={}, ze={}",
           g_ActiveConfig.iSaveTargetId, xfmem.numTexGen.numTexGens, xfmem.numChan.numColorChans,
           xfmem.dualTexTrans.enabled, bpmem.ztex2.op.Value(), bpmem.blendmode.color_update.Value(),
           bpmem.blendmode.alpha_update.Value(), bpmem.zmode.update_enable.Value());

  for (u32 i = 0; i < xfmem.numChan.numColorChans; ++i)
  {
    LitChannel* ch = &xfmem.color[i];
    PRIM_LOG("colchan{}: matsrc={}, light={:#x}, ambsrc={}, diffunc={}, attfunc={}", i,
             ch->matsource.Value(), ch->GetFullLightMask(), ch->ambsource.Value(),
             ch->diffusefunc.Value(), ch->attnfunc.Value());
    ch = &xfmem.alpha[i];
    PRIM_LOG("alpchan{}: matsrc={}, light={:#x}, ambsrc={}, diffunc={}, attfunc={}", i,
             ch->matsource.Value(), ch->GetFullLightMask(), ch->ambsource.Value(),
             ch->diffusefunc.Value(), ch->attnfunc.Value());
  }

  for (u32 i = 0; i < xfmem.numTexGen.numTexGens; ++i)
  {
    TexMtxInfo tinfo = xfmem.texMtxInfo[i];
    if (tinfo.texgentype != TexGenType::EmbossMap)
      tinfo.hex &= 0x7ff;
    if (tinfo.texgentype != TexGenType::Regular)
      tinfo.projection = TexSize::ST;

    PRIM_LOG("txgen{}: proj={}, input={}, gentype={}, srcrow={}, embsrc={}, emblght={}, "
             "postmtx={}, postnorm={}",
             i, tinfo.projection.Value(), tinfo.inputform.Value(), tinfo.texgentype.Value(),
             tinfo.sourcerow.Value(), tinfo.embosssourceshift.Value(),
             tinfo.embosslightshift.Value(), xfmem.postMtxInfo[i].index.Value(),
             xfmem.postMtxInfo[i].normalize.Value());
  }

  PRIM_LOG("pixel: tev={}, ind={}, texgen={}, dstalpha={}, alphatest={:#x}",
           bpmem.genMode.numtevstages.Value() + 1, bpmem.genMode.numindstages.Value(),
           bpmem.genMode.numtexgens.Value(), bpmem.dstalpha.enable.Value(),
           (bpmem.alpha_test.hex >> 16) & 0xff);
#endif

  // Track some stats used elsewhere by the anamorphic widescreen heuristic.
  auto& system = Core::System::GetInstance();
  if (!system.IsWii())
  {
    const bool is_perspective = xfmem.projection.type == ProjectionType::Perspective;

    auto& counts =
        is_perspective ? m_flush_statistics.perspective : m_flush_statistics.orthographic;

    const auto& projection = xfmem.projection.rawProjection;
    // TODO: Potentially the viewport size could be used as weight for the flush count average.
    // This way a small minimap would have less effect than a fullscreen projection.
    const auto& viewport = xfmem.viewport;

    // FYI: This average is based on flushes.
    // It doesn't look at vertex counts like the heuristic does.
    counts.average_ratio.Push(CalculateProjectionViewportRatio(projection, viewport));

    if (IsAnamorphicProjection(projection, viewport, g_ActiveConfig))
    {
      ++counts.anamorphic_flush_count;
      counts.anamorphic_vertex_count += m_index_generator.GetIndexLen();
    }
    else if (IsNormalProjection(projection, viewport, g_ActiveConfig))
    {
      ++counts.normal_flush_count;
      counts.normal_vertex_count += m_index_generator.GetIndexLen();
    }
    else
    {
      ++counts.other_flush_count;
      counts.other_vertex_count += m_index_generator.GetIndexLen();
    }
  }

  auto& pixel_shader_manager = system.GetPixelShaderManager();
  auto& geometry_shader_manager = system.GetGeometryShaderManager();
  auto& vertex_shader_manager = system.GetVertexShaderManager();
  auto& xf_state_manager = system.GetXFStateManager();

  if (g_ActiveConfig.bGraphicMods)
  {
    const double seconds_elapsed =
        static_cast<double>(m_ticks_elapsed) / system.GetSystemTimers().GetTicksPerSecond();
    pixel_shader_manager.constants.time_ms = seconds_elapsed * 1000;
  }

  CalculateNormals(VertexLoaderManager::GetCurrentVertexFormat());
  // Calculate ZSlope for zfreeze
  const auto used_textures = UsedTextures();
  std::vector<std::string> texture_names;
  Common::SmallVector<u32, 8> texture_units;
  std::array<SamplerState, 8> samplers;
  if (!m_cull_all)
  {
    if (!g_ActiveConfig.bGraphicMods)
    {
      for (const u32 i : used_textures)
      {
        const auto cache_entry = g_texture_cache->Load(i);
        if (!cache_entry)
          continue;
        const float custom_tex_scale = cache_entry->GetWidth() / float(cache_entry->native_width);
        samplers[i] = TextureCacheBase::GetSamplerState(
            i, custom_tex_scale, cache_entry->is_custom_tex, cache_entry->has_arbitrary_mips);
      }
    }
    else
    {
      for (const u32 i : used_textures)
      {
        const auto cache_entry = g_texture_cache->Load(i);
        if (cache_entry)
        {
          if (!Common::Contains(texture_names, cache_entry->texture_info_name))
          {
            texture_names.push_back(cache_entry->texture_info_name);
            texture_units.push_back(i);
          }

          const float custom_tex_scale = cache_entry->GetWidth() / float(cache_entry->native_width);
          samplers[i] = TextureCacheBase::GetSamplerState(
              i, custom_tex_scale, cache_entry->is_custom_tex, cache_entry->has_arbitrary_mips);
        }
      }
    }
  }
  vertex_shader_manager.SetConstants(texture_names, xf_state_manager);
  if (!bpmem.genMode.zfreeze)
  {
    // Must be done after VertexShaderManager::SetConstants()
    CalculateZSlope(VertexLoaderManager::GetCurrentVertexFormat());
  }
  else if (m_zslope.dirty && !m_cull_all)  // or apply any dirty ZSlopes
  {
    pixel_shader_manager.SetZSlope(m_zslope.dfdx, m_zslope.dfdy, m_zslope.f0);
    m_zslope.dirty = false;
  }

  if (!m_cull_all)
  {
    CustomPixelShaderContents custom_pixel_shader_contents;
    std::optional<CustomPixelShader> custom_pixel_shader;
    std::vector<std::string> custom_pixel_texture_names;
    std::span<u8> custom_pixel_shader_uniforms;
    bool skip = false;
    for (size_t i = 0; i < texture_names.size(); i++)
    {
      GraphicsModActionData::DrawStarted draw_started{texture_units, &skip, &custom_pixel_shader,
                                                      &custom_pixel_shader_uniforms};
      for (const auto& action : g_graphics_mod_manager->GetDrawStartedActions(texture_names[i]))
      {
        action->OnDrawStarted(&draw_started);
        if (custom_pixel_shader)
        {
          custom_pixel_shader_contents.shaders.push_back(*custom_pixel_shader);
          custom_pixel_texture_names.push_back(texture_names[i]);
        }
        custom_pixel_shader = std::nullopt;
      }
    }

    // Now the vertices can be flushed to the GPU. Everything following the CommitBuffer() call
    // must be careful to not upload any utility vertices, as the binding will be lost otherwise.
    const u32 num_indices = m_index_generator.GetIndexLen();
    if (num_indices == 0)
      return;

    // Texture loading can cause palettes to be applied (-> uniforms -> draws).
    // Palette application does not use vertices, only a full-screen quad, so this is okay.
    // Same with GPU texture decoding, which uses compute shaders.
    g_texture_cache->BindTextures(used_textures, samplers);

    if (!skip)
    {
      UpdatePipelineConfig();
      UpdatePipelineObject();
      if (m_current_pipeline_object)
      {
        const AbstractPipeline* pipeline_object = m_current_pipeline_object;
        if (!custom_pixel_shader_contents.shaders.empty())
        {
          if (const auto custom_pipeline =
                  GetCustomPipeline(custom_pixel_shader_contents, m_current_pipeline_config,
                                    m_current_uber_pipeline_config, m_current_pipeline_object))
          {
            pipeline_object = custom_pipeline;
          }
        }
        RenderDrawCall(pixel_shader_manager, geometry_shader_manager, custom_pixel_shader_contents,
                       custom_pixel_shader_uniforms, m_current_primitive_type, pipeline_object);
      }
    }

    // Even if we skip the draw, emulated state should still be impacted
    OnDraw();

    // The EFB cache is now potentially stale.
    g_framebuffer_manager->FlagPeekCacheAsOutOfDate();
  }

  if (xfmem.numTexGen.numTexGens != bpmem.genMode.numtexgens)
  {
    ERROR_LOG_FMT(VIDEO,
                  "xf.numtexgens ({}) does not match bp.numtexgens ({}). Error in command stream.",
                  xfmem.numTexGen.numTexGens, bpmem.genMode.numtexgens.Value());
  }
}

void VertexManagerBase::DoState(PointerWrap& p)
{
  if (p.IsReadMode())
  {
    // Flush old vertex data before loading state.
    Flush();
  }

  p.Do(m_zslope);
  p.Do(VertexLoaderManager::normal_cache);
  p.Do(VertexLoaderManager::tangent_cache);
  p.Do(VertexLoaderManager::binormal_cache);
}

void VertexManagerBase::CalculateZSlope(NativeVertexFormat* format)
{
  float out[12];
  float viewOffset[2] = {xfmem.viewport.xOrig - bpmem.scissorOffset.x * 2,
                         xfmem.viewport.yOrig - bpmem.scissorOffset.y * 2};

  if (m_current_primitive_type != PrimitiveType::Triangles &&
      m_current_primitive_type != PrimitiveType::TriangleStrip)
  {
    return;
  }

  // Global matrix ID.
  u32 mtxIdx = g_main_cp_state.matrix_index_a.PosNormalMtxIdx;
  const PortableVertexDeclaration vert_decl = format->GetVertexDeclaration();

  // Make sure the buffer contains at least 3 vertices.
  if ((m_cur_buffer_pointer - m_base_buffer_pointer) < (vert_decl.stride * 3))
    return;

  // Lookup vertices of the last rendered triangle and software-transform them
  // This allows us to determine the depth slope, which will be used if z-freeze
  // is enabled in the following flush.
  auto& system = Core::System::GetInstance();
  auto& vertex_shader_manager = system.GetVertexShaderManager();
  for (unsigned int i = 0; i < 3; ++i)
  {
    // If this vertex format has per-vertex position matrix IDs, look it up.
    if (vert_decl.posmtx.enable)
      mtxIdx = VertexLoaderManager::position_matrix_index_cache[2 - i];

    if (vert_decl.position.components == 2)
      VertexLoaderManager::position_cache[2 - i][2] = 0;

    vertex_shader_manager.TransformToClipSpace(&VertexLoaderManager::position_cache[2 - i][0],
                                               &out[i * 4], mtxIdx);

    // Transform to Screenspace
    float inv_w = 1.0f / out[3 + i * 4];

    out[0 + i * 4] = out[0 + i * 4] * inv_w * xfmem.viewport.wd + viewOffset[0];
    out[1 + i * 4] = out[1 + i * 4] * inv_w * xfmem.viewport.ht + viewOffset[1];
    out[2 + i * 4] = out[2 + i * 4] * inv_w * xfmem.viewport.zRange + xfmem.viewport.farZ;
  }

  float dx31 = out[8] - out[0];
  float dx12 = out[0] - out[4];
  float dy12 = out[1] - out[5];
  float dy31 = out[9] - out[1];

  float DF31 = out[10] - out[2];
  float DF21 = out[6] - out[2];
  float a = DF31 * -dy12 - DF21 * dy31;
  float b = dx31 * DF21 + dx12 * DF31;
  float c = -dx12 * dy31 - dx31 * -dy12;

  // Sometimes we process de-generate triangles. Stop any divide by zeros
  if (c == 0)
    return;

  m_zslope.dfdx = -a / c;
  m_zslope.dfdy = -b / c;
  m_zslope.f0 = out[2] - (out[0] * m_zslope.dfdx + out[1] * m_zslope.dfdy);
  m_zslope.dirty = true;
}

void VertexManagerBase::CalculateNormals(NativeVertexFormat* format)
{
  const PortableVertexDeclaration vert_decl = format->GetVertexDeclaration();

  // Only update the binormal/tangent vertex shader constants if the vertex format lacks binormals
  // (VertexLoaderManager::binormal_cache gets updated by the vertex loader when binormals are
  // present, though)
  if (vert_decl.normals[1].enable)
    return;

  VertexLoaderManager::tangent_cache[3] = 0;
  VertexLoaderManager::binormal_cache[3] = 0;

  auto& system = Core::System::GetInstance();
  auto& vertex_shader_manager = system.GetVertexShaderManager();
  if (vertex_shader_manager.constants.cached_tangent != VertexLoaderManager::tangent_cache)
  {
    vertex_shader_manager.constants.cached_tangent = VertexLoaderManager::tangent_cache;
    vertex_shader_manager.dirty = true;
  }
  if (vertex_shader_manager.constants.cached_binormal != VertexLoaderManager::binormal_cache)
  {
    vertex_shader_manager.constants.cached_binormal = VertexLoaderManager::binormal_cache;
    vertex_shader_manager.dirty = true;
  }

  if (vert_decl.normals[0].enable)
    return;

  VertexLoaderManager::normal_cache[3] = 0;
  if (vertex_shader_manager.constants.cached_normal != VertexLoaderManager::normal_cache)
  {
    vertex_shader_manager.constants.cached_normal = VertexLoaderManager::normal_cache;
    vertex_shader_manager.dirty = true;
  }
}

void VertexManagerBase::UpdatePipelineConfig()
{
  NativeVertexFormat* vertex_format = VertexLoaderManager::GetCurrentVertexFormat();
  if (vertex_format != m_current_pipeline_config.vertex_format)
  {
    m_current_pipeline_config.vertex_format = vertex_format;
    m_current_uber_pipeline_config.vertex_format =
        VertexLoaderManager::GetUberVertexFormat(vertex_format->GetVertexDeclaration());
    m_pipeline_config_changed = true;
  }

  VertexShaderUid vs_uid = GetVertexShaderUid();
  if (vs_uid != m_current_pipeline_config.vs_uid)
  {
    m_current_pipeline_config.vs_uid = vs_uid;
    m_current_uber_pipeline_config.vs_uid = UberShader::GetVertexShaderUid();
    m_pipeline_config_changed = true;
  }

  PixelShaderUid ps_uid = GetPixelShaderUid();
  if (ps_uid != m_current_pipeline_config.ps_uid)
  {
    m_current_pipeline_config.ps_uid = ps_uid;
    m_current_uber_pipeline_config.ps_uid = UberShader::GetPixelShaderUid();
    m_pipeline_config_changed = true;
  }

  GeometryShaderUid gs_uid = GetGeometryShaderUid(GetCurrentPrimitiveType());
  if (gs_uid != m_current_pipeline_config.gs_uid)
  {
    m_current_pipeline_config.gs_uid = gs_uid;
    m_current_uber_pipeline_config.gs_uid = gs_uid;
    m_pipeline_config_changed = true;
  }

  if (m_rasterization_state_changed)
  {
    m_rasterization_state_changed = false;

    RasterizationState new_rs = {};
    new_rs.Generate(bpmem, m_current_primitive_type);
    if (new_rs != m_current_pipeline_config.rasterization_state)
    {
      m_current_pipeline_config.rasterization_state = new_rs;
      m_current_uber_pipeline_config.rasterization_state = new_rs;
      m_pipeline_config_changed = true;
    }
  }

  if (m_depth_state_changed)
  {
    m_depth_state_changed = false;

    DepthState new_ds = {};
    new_ds.Generate(bpmem);
    if (new_ds != m_current_pipeline_config.depth_state)
    {
      m_current_pipeline_config.depth_state = new_ds;
      m_current_uber_pipeline_config.depth_state = new_ds;
      m_pipeline_config_changed = true;
    }
  }

  if (m_blending_state_changed)
  {
    m_blending_state_changed = false;

    BlendingState new_bs = {};
    new_bs.Generate(bpmem);
    if (new_bs != m_current_pipeline_config.blending_state)
    {
      m_current_pipeline_config.blending_state = new_bs;
      m_current_uber_pipeline_config.blending_state = new_bs;
      m_pipeline_config_changed = true;
    }
  }
}

void VertexManagerBase::UpdatePipelineObject()
{
  if (!m_pipeline_config_changed)
    return;

  m_current_pipeline_object = nullptr;
  m_pipeline_config_changed = false;

  switch (g_ActiveConfig.iShaderCompilationMode)
  {
  case ShaderCompilationMode::Synchronous:
  {
    // Ubershaders disabled? Block and compile the specialized shader.
    m_current_pipeline_object = g_shader_cache->GetPipelineForUid(m_current_pipeline_config);
  }
  break;

  case ShaderCompilationMode::SynchronousUberShaders:
  {
    // Exclusive ubershader mode, always use ubershaders.
    m_current_pipeline_object =
        g_shader_cache->GetUberPipelineForUid(m_current_uber_pipeline_config);
  }
  break;

  case ShaderCompilationMode::AsynchronousUberShaders:
  case ShaderCompilationMode::AsynchronousSkipRendering:
  {
    // Can we background compile shaders? If so, get the pipeline asynchronously.
    auto res = g_shader_cache->GetPipelineForUidAsync(m_current_pipeline_config);
    if (res)
    {
      // Specialized shaders are ready, prefer these.
      m_current_pipeline_object = *res;
      return;
    }

    if (g_ActiveConfig.iShaderCompilationMode == ShaderCompilationMode::AsynchronousUberShaders)
    {
      // Specialized shaders not ready, use the ubershaders.
      m_current_pipeline_object =
          g_shader_cache->GetUberPipelineForUid(m_current_uber_pipeline_config);
    }
    else
    {
      // Ensure we try again next draw. Otherwise, if no registers change between frames, the
      // object will never be drawn, even when the shader is ready.
      m_pipeline_config_changed = true;
    }
  }
  break;
  }
}

void VertexManagerBase::OnConfigChange()
{
  // Reload index generator function tables in case VS expand config changed
  m_index_generator.Init();
}

void VertexManagerBase::OnDraw()
{
  m_draw_counter++;

  // If the last efb copy was too close to the one before it, don't forget about it until the next
  // efb copy happens (which might not be for a long time)
  u32 diff = m_draw_counter - m_last_efb_copy_draw_counter;
  if (m_unflushed_efb_copy && diff > MINIMUM_DRAW_CALLS_PER_COMMAND_BUFFER_FOR_READBACK)
  {
    g_gfx->Flush();
    m_unflushed_efb_copy = false;
    m_last_efb_copy_draw_counter = m_draw_counter;
  }

  // If we didn't have any CPU access last frame, do nothing.
  if (m_scheduled_command_buffer_kicks.empty() || !m_allow_background_execution)
    return;

  // Check if this draw is scheduled to kick a command buffer.
  // The draw counters will always be sorted so a binary search is possible here.
  if (std::ranges::binary_search(m_scheduled_command_buffer_kicks, m_draw_counter))
  {
    // Kick a command buffer on the background thread.
    g_gfx->Flush();
    m_unflushed_efb_copy = false;
    m_last_efb_copy_draw_counter = m_draw_counter;
  }
}

void VertexManagerBase::OnCPUEFBAccess()
{
  // Check this isn't another access without any draws in between.
  if (!m_cpu_accesses_this_frame.empty() && m_cpu_accesses_this_frame.back() == m_draw_counter)
    return;

  // Store the current draw counter for scheduling in OnEndFrame.
  m_cpu_accesses_this_frame.emplace_back(m_draw_counter);
}

void VertexManagerBase::OnEFBCopyToRAM()
{
  // If we're not deferring, try to preempt it next frame.
  if (!g_ActiveConfig.bDeferEFBCopies)
  {
    OnCPUEFBAccess();
    return;
  }

  // Otherwise, only execute if we have at least 10 objects between us and the last copy.
  const u32 diff = m_draw_counter - m_last_efb_copy_draw_counter;
  m_last_efb_copy_draw_counter = m_draw_counter;
  if (diff < MINIMUM_DRAW_CALLS_PER_COMMAND_BUFFER_FOR_READBACK)
  {
    m_unflushed_efb_copy = true;
    return;
  }

  m_unflushed_efb_copy = false;
  g_gfx->Flush();
}

void VertexManagerBase::OnEndFrame()
{
  m_draw_counter = 0;
  m_last_efb_copy_draw_counter = 0;
  m_scheduled_command_buffer_kicks.clear();

  // If we have no CPU access at all, leave everything in the one command buffer for maximum
  // parallelism between CPU/GPU, at the cost of slightly higher latency.
  if (m_cpu_accesses_this_frame.empty())
    return;

  // In order to reduce CPU readback latency, we want to kick a command buffer roughly halfway
  // between the draw counters that invoked the readback, or every 250 draws, whichever is
  // smaller.
  if (g_ActiveConfig.iCommandBufferExecuteInterval > 0)
  {
    u32 last_draw_counter = 0;
    u32 interval = static_cast<u32>(g_ActiveConfig.iCommandBufferExecuteInterval);
    for (u32 draw_counter : m_cpu_accesses_this_frame)
    {
      // We don't want to waste executing command buffers for only a few draws, so set a minimum.
      // Leave last_draw_counter as-is, so we get the correct number of draws between submissions.
      u32 draw_count = draw_counter - last_draw_counter;
      if (draw_count < MINIMUM_DRAW_CALLS_PER_COMMAND_BUFFER_FOR_READBACK)
        continue;

      if (draw_count <= interval)
      {
        u32 mid_point = draw_count / 2;
        m_scheduled_command_buffer_kicks.emplace_back(last_draw_counter + mid_point);
      }
      else
      {
        u32 counter = interval;
        while (counter < draw_count)
        {
          m_scheduled_command_buffer_kicks.emplace_back(last_draw_counter + counter);
          counter += interval;
        }
      }

      last_draw_counter = draw_counter;
    }
  }

  m_cpu_accesses_this_frame.clear();

  // We invalidate the pipeline object at the start of the frame.
  // This is for the rare case where only a single pipeline configuration is used,
  // and hybrid ubershaders have compiled the specialized shader, but without any
  // state changes the specialized shader will not take over.
  InvalidatePipelineObject();
}

void VertexManagerBase::NotifyCustomShaderCacheOfHostChange(const ShaderHostConfig& host_config)
{
  m_custom_shader_cache->SetHostConfig(host_config);
  m_custom_shader_cache->Reload();
}

void VertexManagerBase::RenderDrawCall(
    PixelShaderManager& pixel_shader_manager, GeometryShaderManager& geometry_shader_manager,
    const CustomPixelShaderContents& custom_pixel_shader_contents,
    std::span<u8> custom_pixel_shader_uniforms, PrimitiveType primitive_type,
    const AbstractPipeline* current_pipeline)
{
  // Now we can upload uniforms, as nothing else will override them.
  geometry_shader_manager.SetConstants(primitive_type);
  pixel_shader_manager.SetConstants();
  if (!custom_pixel_shader_uniforms.empty() &&
      pixel_shader_manager.custom_constants.data() != custom_pixel_shader_uniforms.data())
  {
    pixel_shader_manager.custom_constants_dirty = true;
  }
  pixel_shader_manager.custom_constants = custom_pixel_shader_uniforms;
#ifdef HAS_VULKAN
  const TemporalDrawPlan temporal_plan = PrepareTemporalDraw(
      current_pipeline, !custom_pixel_shader_contents.shaders.empty());
  // Native rendering and all guest-visible callbacks always use the unjittered lane.
  if (g_backend_info.api_type == APIType::Vulkan)
    Core::System::GetInstance().GetVertexShaderManager().SetTemporalRasterJitter(false);
#endif
  UploadUniforms();

  g_gfx->SetPipeline(current_pipeline);

  u32 base_vertex, base_index;
  CommitBuffer(m_index_generator.GetNumVerts(),
               VertexLoaderManager::GetCurrentVertexFormat()->GetVertexStride(),
               m_index_generator.GetIndexLen(), &base_vertex, &base_index);

  if (g_backend_info.api_type != APIType::D3D && g_ActiveConfig.UseVSForLinePointExpand() &&
      (primitive_type == PrimitiveType::Points || primitive_type == PrimitiveType::Lines))
  {
    // VS point/line expansion puts the vertex id at gl_VertexID << 2
    // That means the base vertex has to be adjusted to match
    // (The shader adds this after shifting right on D3D, so no need to do this)
    base_vertex <<= 2;
  }

  if (PerfQueryBase::ShouldEmulate())
    g_perf_query->EnableQuery(bpmem.zcontrol.early_ztest ? PQG_ZCOMP_ZCOMPLOC : PQG_ZCOMP);

  DrawCurrentBatch(base_index, m_index_generator.GetIndexLen(), base_vertex);

  // Track the total emulated state draws
  INCSTAT(g_stats.this_frame.num_draw_calls);

  if (PerfQueryBase::ShouldEmulate())
    g_perf_query->DisableQuery(bpmem.zcontrol.early_ztest ? PQG_ZCOMP_ZCOMPLOC : PQG_ZCOMP);
#ifdef HAS_VULKAN
  ReplayTemporalDraw(temporal_plan, base_index, m_index_generator.GetIndexLen(), base_vertex);
#endif
}

#ifdef HAS_VULKAN
VertexManagerBase::TemporalDrawPlan
VertexManagerBase::PrepareTemporalDraw(const AbstractPipeline* native_pipeline, bool custom_shader)
{
  TemporalDrawPlan plan;
  const char* requested = std::getenv("MODERNGEKKO_DLSS_EVALUATE");
  const char* ngx_requested = std::getenv("MODERNGEKKO_DLSS");
  if (!requested || std::string_view(requested) != "1" || !ngx_requested ||
      std::string_view(ngx_requested) != "1" || g_backend_info.api_type != APIType::Vulkan ||
      !ModernGekkoTemporal::GetState().active)
    return plan;

  auto* gfx = Vulkan::VKGfx::GetInstance();
  const auto native_state = gfx->SaveTemporalRasterState();
  bool allow_native_geometry_fields = false;
  bool no_indirect = false;
  auto trace = [&](bool known, bool overwrite, bool alpha_gate, std::string_view reason) {
    const char* enabled = std::getenv("MODERNGEKKO_TEMPORAL_TRACE_FRAME");
    const auto state = ModernGekkoTemporal::GetState();
    if (!enabled || std::string_view(enabled) != "1" || state.content_id < 90 || state.content_id > 91)
      return;
    static u64 traced_content = std::numeric_limits<u64>::max();
    static u32 order = 0;
    if (traced_content != state.content_id) { traced_content = state.content_id; order = 0; }
    if (++order > 512)
      return;
    const auto scissor = BPFunctions::ComputeScissorRects(
        bpmem.scissorTL, bpmem.scissorBR, bpmem.scissorOffset, xfmem.viewport);
    std::string rectangles;
    for (const auto& rc : scissor.rectangles)
      rectangles += fmt::format("({},{},{},{})@{},{};", rc.rect.left, rc.rect.top,
                                rc.rect.right, rc.rect.bottom, rc.x_off, rc.y_off);
    const auto blend = m_current_pipeline_config.blending_state;
    const auto depth = m_current_pipeline_config.depth_state;
    const auto* ps = m_current_pipeline_config.ps_uid.GetUidData();
    INFO_LOG_FMT(VIDEO,
      "Temporal draw trace: content={} order={} indices={} known={} overwrite={} alpha_gate={} "
      "RGB={} A={} ztest={} zwrite={} zfunc={} blend={} src={} dst={} projection={} "
      "viewport=({},{},{},{}) scissor_count={} scissors={} "
      "gpuVP=({},{},{},{},{},{}) gpuScissor=({},{},{},{}) "
      "uid_ztest={} uid_fragdepth={} uid_ztex={} uid_alpha_pretest={} uid_bbox={} "
      "allow_fields={} no_indirect={} configured_indirect={} sampled_indirect={} reason={}",
      state.content_id, order, m_index_generator.GetIndexLen(), known, overwrite, alpha_gate,
      bool(blend.color_update), bool(blend.alpha_update), bool(depth.test_enable),
      bool(depth.update_enable), static_cast<u32>(depth.func.Value()), bool(blend.blend_enable),
      static_cast<u32>(blend.src_factor.Value()), static_cast<u32>(blend.dst_factor.Value()),
      static_cast<u32>(xfmem.projection.type), xfmem.viewport.xOrig, xfmem.viewport.yOrig,
      2.0f*xfmem.viewport.wd, -2.0f*xfmem.viewport.ht, scissor.rectangles.size(), rectangles,
      native_state.viewport.x, native_state.viewport.y, native_state.viewport.width,
      native_state.viewport.height, native_state.viewport.minDepth, native_state.viewport.maxDepth,
      native_state.scissor.offset.x, native_state.scissor.offset.y, native_state.scissor.extent.width,
      native_state.scissor.extent.height, static_cast<u32>(ps->ztest), static_cast<u32>(ps->per_pixel_depth),
      static_cast<u32>(ps->ztex_op), static_cast<u32>(ps->Pretest), static_cast<u32>(ps->bounding_box), allow_native_geometry_fields,
      no_indirect, bpmem.genMode.numindstages.Value(), static_cast<u32>(ps->nIndirectStagesUsed), reason);
  };
  auto reject = [&](std::string_view reason) {
    g_texture_cache->RejectTemporalNativeAncestry();
    trace(false, false, false, reason);
    static u64 rejected_draws = 0;
    if (++rejected_draws <= 8 || rejected_draws % 120 == 0)
      INFO_LOG_FMT(VIDEO, "Temporal mirror: rejected draws={} reason={}", rejected_draws, reason);
    g_framebuffer_manager->InvalidateTemporalSplit();
    g_framebuffer_manager->InvalidateTemporalMirrorCoverage();
    gfx->RestoreTemporalRasterState(native_state);
    return TemporalDrawPlan{};
  };
  // Late depth, alpha rejection and fragment-depth metadata ordering have matched
  // generated-shader GPU raster proof. These remaining paths still need separate proof.
  const auto* ps = m_current_pipeline_config.ps_uid.GetUidData();
  if (custom_shader || g_ActiveConfig.bGraphicMods ||
      g_ActiveConfig.stereo_mode != StereoMode::Off ||
      g_framebuffer_manager->GetEFBSamples() != 1 || bpmem.genMode.zfreeze ||
      !std::isfinite(xfmem.viewport.wd) || !std::isfinite(xfmem.viewport.ht) ||
      !std::isfinite(xfmem.viewport.zRange) || !std::isfinite(xfmem.viewport.farZ) ||
      xfmem.viewport.wd == 0.0f || xfmem.viewport.ht == 0.0f || ps->uint_output ||
      m_current_pipeline_config.blending_state.logic_op_enable ||
      ps->blend_enable || ps->logic_op_enable || ps->ztest == EmulatedZ::EarlyWithFBFetch)
    return reject("unsupported shader, sample count or depth mode");
  if (!g_framebuffer_manager->EnsureTemporalMirror())
    return reject("mirror framebuffer unavailable");

  // Clip-space displacement uses clip.w for both projection types. Generated-shader
  // raster fixtures prove the same signed pixel offset for perspective and orthographic
  // geometry; texture provenance, rather than projection type, classifies the inputs.
  const auto native_blend = m_current_pipeline_config.blending_state;
  const auto native_depth = m_current_pipeline_config.depth_state;
  const bool depth_write = native_depth.test_enable && native_depth.update_enable;
  no_indirect = bpmem.genMode.numindstages == 0 && ps->nIndirectStagesUsed == 0;
  for (const auto& indirect : bpmem.tevind)
  {
    no_indirect = no_indirect && !indirect.IsActive() && !indirect.fb_addprev &&
                  indirect.sw == IndTexWrap::ITW_OFF && indirect.tw == IndTexWrap::ITW_OFF;
  }
  const auto temporal_state = ModernGekkoTemporal::GetState();
  allow_native_geometry_fields = xfmem.projection.type == ProjectionType::Perspective &&
      native_blend.color_update && depth_write && temporal_state.depth_convention_known &&
      !(temporal_state.reason_bits & static_cast<u32>(ModernGekkoTemporal::CoverageReason::MixedDepthConvention)) &&
      ps->ztex_op == ZTexOp::Disabled && !ps->per_pixel_depth && no_indirect;
  const bool known = !g_texture_cache->HasUnclassifiedTemporalInputs(
      UsedTextures(), allow_native_geometry_fields);
  const bool overwrite = known && native_blend.color_update && !native_blend.blend_enable &&
                         !native_blend.logic_op_enable;
  const bool opacity_blend = native_blend.color_update && native_blend.blend_enable &&
      !native_blend.subtract && native_blend.src_factor == SrcBlendFactor::SrcAlpha &&
      native_blend.dst_factor == DstBlendFactor::InvSrcAlpha;
  const bool alpha_overwrite = known && native_blend.alpha_update &&
      (!native_blend.blend_enable || (!native_blend.subtract_alpha &&
       native_blend.src_factor_alpha == SrcBlendFactor::One &&
       native_blend.dst_factor_alpha == DstBlendFactor::Zero));
  const bool alpha_dependency = native_blend.color_update && native_blend.blend_enable &&
      (native_blend.src_factor == SrcBlendFactor::DstAlpha ||
       native_blend.src_factor == SrcBlendFactor::InvDstAlpha ||
       native_blend.dst_factor == DstBlendFactor::DstAlpha ||
       native_blend.dst_factor == DstBlendFactor::InvDstAlpha);
  const auto known_mode = known ? TemporalPixelShaderMode::MetadataKnown :
                                 TemporalPixelShaderMode::MetadataUnknown;
  const auto color_mode = opacity_blend ?
      (known ? TemporalPixelShaderMode::MetadataKnownOpaqueAlpha :
               TemporalPixelShaderMode::MetadataUnknownOpaqueAlpha) : known_mode;
  const bool physical_early = ps->ztest == EmulatedZ::ForcedEarly;
  const auto depth_mode = physical_early ?
      (known ? TemporalPixelShaderMode::DepthKnownEarly : TemporalPixelShaderMode::DepthUnknownEarly) :
      known_mode;
  enum class Pass { Color, Alpha, AlphaGate, DepthMetadata, Mirror, BoundaryProof, OverlayColor };
  auto pipeline_for = [&](TemporalPixelShaderMode shader_mode, Pass pass) {
    auto specialized = m_current_pipeline_config;
    auto uber = m_current_uber_pipeline_config;
    SetTemporalPixelShaderMode(&specialized.ps_uid, shader_mode, native_blend.use_dual_src);
    UberShader::SetTemporalPixelShaderMode(&uber.ps_uid, shader_mode, native_blend.use_dual_src);
    if (pass != Pass::Mirror)
    {
      DepthState depth = specialized.depth_state;
      depth.update_enable = pass == Pass::BoundaryProof;
      if (pass == Pass::OverlayColor)
      {
        depth.test_enable = false;
        depth.update_enable = false;
      }
      BlendingState blend{};
      blend.blend_enable = true;
      blend.color_update = pass != Pass::Alpha;
      blend.alpha_update = pass == Pass::Alpha;
      blend.src_factor = SrcBlendFactor::Zero;
      blend.dst_factor = DstBlendFactor::SrcClr;
      blend.src_factor_alpha = alpha_overwrite ? SrcBlendFactor::One : SrcBlendFactor::Zero;
      blend.dst_factor_alpha = alpha_overwrite ? DstBlendFactor::Zero : DstBlendFactor::SrcAlpha;
      if (pass == Pass::AlphaGate)
        blend.dst_factor = DstBlendFactor::DstAlpha;
      else if (pass == Pass::DepthMetadata || (pass == Pass::Color && overwrite))
      {
        blend.src_factor = SrcBlendFactor::One;
        blend.dst_factor = DstBlendFactor::Zero;
      }
      else if (pass == Pass::Color && opacity_blend)
      {
        // Secondary alpha is an exact selector: known opaque repairs, fractional
        // known retains history, and any nonzero unknown contribution taints.
        blend.use_dual_src = true;
        blend.src_factor = SrcBlendFactor::SrcAlpha;
        blend.dst_factor = DstBlendFactor::InvSrcAlpha;
      }
      if (pass == Pass::BoundaryProof)
      {
        blend.blend_enable = false;
        blend.color_update = true;
        blend.alpha_update = false;
      }
      else if (pass == Pass::OverlayColor)
      {
        blend.use_dual_src = false;
        blend.color_update = true;
        blend.alpha_update = true;
        blend.src_factor = SrcBlendFactor::SrcAlpha;
        blend.dst_factor = DstBlendFactor::InvSrcAlpha;
        blend.src_factor_alpha = SrcBlendFactor::One;
        blend.dst_factor_alpha = DstBlendFactor::InvSrcAlpha;
      }
      specialized.depth_state = depth;
      uber.depth_state = depth;
      specialized.blending_state = blend;
      uber.blending_state = blend;
    }
    if (native_pipeline->m_config.usage == AbstractPipelineUsage::GXUber)
      return g_shader_cache->GetUberPipelineForUid(uber);
    return g_shader_cache->GetPipelineForUid(specialized);
  };
  plan.known = known;
  plan.overwrite = overwrite;
  if (g_framebuffer_manager->IsTemporalOverlayActive())
  {
    if (!known || xfmem.projection.type != ProjectionType::Orthographic ||
        native_depth.test_enable || native_depth.update_enable || !opacity_blend)
    {
      g_framebuffer_manager->RejectTemporalSplit();
      return reject("unsupported draw after verified world/overlay boundary");
    }
    plan.overlay_pipeline = pipeline_for(TemporalPixelShaderMode::Overlay, Pass::OverlayColor);
    plan.overlay_framebuffer = g_framebuffer_manager->GetTemporalOverlayFramebuffer();
    if (!plan.overlay_pipeline || !plan.overlay_framebuffer)
    {
      g_framebuffer_manager->RejectTemporalSplit();
      return reject("overlay pipeline or framebuffer unavailable");
    }
    trace(known, false, false, "unjittered overlay");
    gfx->RestoreTemporalRasterState(native_state);
    return plan;
  }
  const bool boundary_candidate = known && !native_blend.color_update && !native_blend.alpha_update &&
      depth_write && native_depth.func == CompareMode::Always &&
      xfmem.projection.type == ProjectionType::Orthographic &&
      (static_cast<u32>(ps->Pretest) & 3u) == static_cast<u32>(AlphaTestResult::Pass) &&
      !ps->per_pixel_depth && ps->ztex_op == ZTexOp::Disabled &&
      (m_current_primitive_type == PrimitiveType::Triangles ||
       m_current_primitive_type == PrimitiveType::TriangleStrip);
  if (boundary_candidate)
  {
    if (!g_framebuffer_manager->EnsureTemporalBoundary())
      return reject("boundary scratch framebuffer unavailable");
    plan.boundary_pipeline = pipeline_for(TemporalPixelShaderMode::Boundary, Pass::BoundaryProof);
    plan.boundary_framebuffer = g_framebuffer_manager->GetTemporalBoundaryFramebuffer();
    if (!plan.boundary_pipeline || !plan.boundary_framebuffer)
      return reject("boundary proof pipeline unavailable");
  }
  plan.color_pipeline = pipeline_for(TemporalPixelShaderMode::MirrorColor, Pass::Mirror);
  if (native_blend.color_update)
    plan.metadata_pipeline = pipeline_for(color_mode, Pass::Color);
  if (native_blend.alpha_update)
    plan.alpha_pipeline = pipeline_for(known_mode, Pass::Alpha);
  if (alpha_dependency)
    plan.alpha_gate_pipeline = pipeline_for(TemporalPixelShaderMode::MetadataKnown, Pass::AlphaGate);
  if (depth_write)
    plan.depth_pipeline = pipeline_for(depth_mode, Pass::DepthMetadata);
  plan.color_framebuffer = g_framebuffer_manager->GetTemporalMirrorFramebuffer();
  plan.metadata_framebuffer = g_framebuffer_manager->GetTemporalCoverageFramebuffer();
  plan.depth_framebuffer = g_framebuffer_manager->GetTemporalDepthCoverageFramebuffer();
  if (!plan.color_pipeline || (native_blend.color_update && !plan.metadata_pipeline) ||
      (native_blend.alpha_update && !plan.alpha_pipeline) ||
      (alpha_dependency && !plan.alpha_gate_pipeline) || (depth_write && !plan.depth_pipeline))
    return reject("paired shader pipeline unavailable");
  trace(known, overwrite, alpha_dependency, "paired");
  gfx->RestoreTemporalRasterState(native_state);
  return plan;
}

void VertexManagerBase::ReplayTemporalDraw(const TemporalDrawPlan& plan, u32 base_index,
                                         u32 num_indices, u32 base_vertex)
{
  if (!plan.color_pipeline && !plan.overlay_pipeline)
    return;
  auto* gfx = Vulkan::VKGfx::GetInstance();
  const auto native_state = gfx->SaveTemporalRasterState();
  auto& vertex_shader_manager = Core::System::GetInstance().GetVertexShaderManager();
  auto restore_native = [&]() {
    gfx->RestoreTemporalRasterState(native_state);
    InvalidateConstants();
    vertex_shader_manager.SetTemporalRasterJitter(false);
    UploadUniforms();
  };
  if (plan.overlay_pipeline)
  {
    vertex_shader_manager.SetTemporalRasterJitter(false);
    UploadUniforms();
    gfx->SetFramebuffer(plan.overlay_framebuffer);
    gfx->SetPipeline(plan.overlay_pipeline);
    gfx->DrawIndexed(base_index, num_indices, base_vertex);
    restore_native();
    return;
  }
  if (plan.boundary_pipeline)
  {
    // Native guest effects have already happened once. This separate scratch draw
    // proves actual raster coverage and stored depth without touching the mirror.
    vertex_shader_manager.SetTemporalRasterJitter(false);
    UploadUniforms();
    g_framebuffer_manager->ResetTemporalBoundary();
    gfx->SetFramebuffer(plan.boundary_framebuffer);
    gfx->SetPipeline(plan.boundary_pipeline);
    gfx->DrawIndexed(base_index, num_indices, base_vertex);
    u64 mask_invalid = ~u64{0}, nonfinite = ~u64{0};
    float depth_min = std::numeric_limits<float>::quiet_NaN();
    float depth_max = depth_min;
    const bool proven = g_framebuffer_manager->CertifyTemporalBoundary(
        g_framebuffer_manager->GetTemporalCaptureRect(), &mask_invalid, &depth_min, &depth_max,
        &nonfinite, false);
    bool frozen = false;
    if (proven)
    {
      frozen = g_framebuffer_manager->FreezeWorldBeforeOverlay() &&
               g_framebuffer_manager->ActivateTemporalOverlay();
      if (!frozen)
        g_framebuffer_manager->RejectTemporalSplit();
    }
    INFO_LOG_FMT(VIDEO, "Temporal boundary: content={} proven={} mask_invalid={} depth_min={} "
                       "depth_max={} nonfinite={} world_frozen={}",
                 ModernGekkoTemporal::GetState().content_id, proven, mask_invalid, depth_min,
                 depth_max, nonfinite, frozen);
    restore_native();
    if (proven)
      return;
    // Proof utilities may upload utility vertices and uniforms; restore committed
    // geometry bindings and upload fresh unchanged GX constants before replay.
  }
  vertex_shader_manager.SetTemporalRasterJitter(true);
  UploadUniforms();
  if (plan.alpha_gate_pipeline)
  {
    // Propagate OLD destination-alpha taint to RGB before the ordinary metadata
    // pass can repair alpha. This pass writes only RGB: R := R * prior A.
    gfx->SetFramebuffer(plan.metadata_framebuffer);
    gfx->SetPipeline(plan.alpha_gate_pipeline);
    gfx->DrawIndexed(base_index, num_indices, base_vertex);
  }
  if (plan.metadata_pipeline)
  {
    gfx->SetFramebuffer(plan.metadata_framebuffer);
    gfx->SetPipeline(plan.metadata_pipeline);
    gfx->DrawIndexed(base_index, num_indices, base_vertex);
  }
  if (plan.alpha_pipeline)
  {
    gfx->SetFramebuffer(plan.metadata_framebuffer);
    gfx->SetPipeline(plan.alpha_pipeline);
    gfx->DrawIndexed(base_index, num_indices, base_vertex);
  }
  if (plan.depth_pipeline)
  {
    gfx->SetFramebuffer(plan.depth_framebuffer);
    gfx->SetPipeline(plan.depth_pipeline);
    gfx->DrawIndexed(base_index, num_indices, base_vertex);
  }
  gfx->SetFramebuffer(plan.color_framebuffer);
  gfx->SetPipeline(plan.color_pipeline);
  gfx->DrawIndexed(base_index, num_indices, base_vertex);
  static u64 paired_draws = 0;
  if (++paired_draws <= 8 || paired_draws % 120 == 0)
    INFO_LOG_FMT(VIDEO, "Temporal mirror: paired draws={} metadata={} known={} overwrite={}",
                 paired_draws, plan.metadata_pipeline != nullptr, plan.known, plan.overwrite);
  // Replay bypasses DrawCurrentBatch/OnDraw, bounding-box flushes, statistics, and queries.
  restore_native();
}
#endif

const AbstractPipeline* VertexManagerBase::GetCustomPipeline(
    const CustomPixelShaderContents& custom_pixel_shader_contents,
    const VideoCommon::GXPipelineUid& current_pipeline_config,
    const VideoCommon::GXUberPipelineUid& current_uber_pipeline_config,
    const AbstractPipeline* current_pipeline) const
{
  if (current_pipeline)
  {
    if (!custom_pixel_shader_contents.shaders.empty())
    {
      CustomShaderInstance custom_shaders;
      custom_shaders.pixel_contents = custom_pixel_shader_contents;
      switch (g_ActiveConfig.iShaderCompilationMode)
      {
      case ShaderCompilationMode::Synchronous:
      case ShaderCompilationMode::AsynchronousSkipRendering:
      {
        if (auto pipeline = m_custom_shader_cache->GetPipelineAsync(
                current_pipeline_config, custom_shaders, current_pipeline->m_config))
        {
          return *pipeline;
        }
      }
      break;
      case ShaderCompilationMode::SynchronousUberShaders:
      {
        // D3D has issues compiling large custom ubershaders
        // use specialized shaders instead
        if (g_backend_info.api_type == APIType::D3D)
        {
          if (auto pipeline = m_custom_shader_cache->GetPipelineAsync(
                  current_pipeline_config, custom_shaders, current_pipeline->m_config))
          {
            return *pipeline;
          }
        }
        else
        {
          if (auto pipeline = m_custom_shader_cache->GetPipelineAsync(
                  current_uber_pipeline_config, custom_shaders, current_pipeline->m_config))
          {
            return *pipeline;
          }
        }
      }
      break;
      case ShaderCompilationMode::AsynchronousUberShaders:
      {
        if (auto pipeline = m_custom_shader_cache->GetPipelineAsync(
                current_pipeline_config, custom_shaders, current_pipeline->m_config))
        {
          return *pipeline;
        }
        else if (auto uber_pipeline = m_custom_shader_cache->GetPipelineAsync(
                     current_uber_pipeline_config, custom_shaders, current_pipeline->m_config))
        {
          return *uber_pipeline;
        }
      }
      break;
      };
    }
  }

  return nullptr;
}
