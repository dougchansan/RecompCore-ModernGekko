// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/VertexShaderManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iterator>

#include "Common/ChunkFile.h"
#include "Common/CommonTypes.h"
#include "Common/Logging/Log.h"
#include "Common/Matrix.h"
#include "VideoCommon/BPFunctions.h"
#include "VideoCommon/BPMemory.h"
#include "VideoCommon/ColosseumProjection.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <tuple>
#include "VideoCommon/CPMemory.h"
#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/FreeLookCamera.h"
#include "VideoCommon/GraphicsModSystem/Runtime/GraphicsModActionData.h"
#include "VideoCommon/GraphicsModSystem/Runtime/GraphicsModManager.h"
#include "VideoCommon/Statistics.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VideoCommon.h"
#include "VideoCommon/VideoConfig.h"
#include "VideoCommon/XFMemory.h"
#include "VideoCommon/XFStateManager.h"
#include "VideoCommon/XFStructs.h"

void VertexShaderManager::Init()
{
  // Initialize state tracking variables
  m_projection_graphics_mod_change = false;

  constants = {};

  m_projection_matrix = Common::Matrix44::Identity().data;

  dirty = true;
}

Common::Matrix44 VertexShaderManager::LoadProjectionMatrix()
{
  return LoadProjectionMatrix(xfmem.projection.rawProjection, &m_projection_matrix, true);
}

Common::Matrix44 VertexShaderManager::LoadProjectionMatrix(const std::array<float, 6>& rawProjection,
                                                          std::array<float, 16>* projection_matrix,
                                                          bool update_stats)
{
  auto& pm = *projection_matrix;

  switch (xfmem.projection.type)
  {
  case ProjectionType::Perspective:
  {
    const Common::Vec2 fov_multiplier = g_freelook_camera.IsActive() ?
                                            g_freelook_camera.GetFieldOfViewMultiplier() :
                                            Common::Vec2{1, 1};
    // Colosseum can retain a cached 16:9 projection after loading a state or
    // changing output modes even though its game-side aspect constant has
    // already been updated. Correct only projections that still describe a
    // roughly 16:9 frustum; projections rebuilt by the game for the native
    // target aspect pass through unchanged. This preserves the original
    // vertical framing and character size while revealing more world
    // horizontally (Hor+).
    const float raw_aspect =
        rawProjection[0] != 0.0f ? std::abs(rawProjection[2] / rawProjection[0]) : 0.0f;
    // Only centered frustums qualify. The stale field camera this targets is
    // symmetric, while battle arenas build an off-center (skewed) ~1.98-aspect
    // frustum every frame for a pass that has to line up with the main camera;
    // widening it left whole floor regions black at 32:9.
    const bool off_center_projection =
        std::abs(rawProjection[1]) > 0.01f || std::abs(rawProjection[3]) > 0.01f;
    const bool is_cached_widescreen_projection =
        !off_center_projection && raw_aspect >= 1.6f && raw_aspect <= 2.0f;
    const bool square_offscreen_viewport =
        g_ActiveConfig.bWidescreenHudSafeArea &&
        VideoCommon::IsColosseumSquareOffscreenViewport(xfmem.viewport.wd,
                                                        xfmem.viewport.ht);
    const float authored_projection_scale =
        std::clamp(g_ActiveConfig.fWidescreenHudSafeAreaScale, 0.25f, 1.0f);
    // Colosseum renders its character and world shadows through a square
    // offscreen viewport. That buffer is authored square and is sampled back
    // through a projector, so its frustum has nothing to do with the output
    // aspect: it must pass through untouched. Scaling it by the inverse safe
    // area (which is what this branch used to do) stretched every shadow
    // horizontally by 1/scale - 2x at 32:9 - and amplified the shadow camera's
    // own per-draw aspect drift by the same factor, which read on screen as
    // shadows swinging back and forth on a completely static scene.
    constexpr float offscreen_projection_scale = 1.0f;
    const float authored_menu_scale = std::max(
        VideoCommon::GetColosseumAuthoredMenuHorizontalScale(
            raw_aspect, g_ActiveConfig.bWidescreenAuthoredMenu),
        g_ActiveConfig.bWidescreenAuthoredMenu ?
            VideoCommon::COLOSSEUM_AUTHORED_MENU_MINIMUM_SCALE :
            1.0f);
    const float naming_scale = VideoCommon::GetColosseumNamingHorizontalScale(
        g_ActiveConfig.bColosseumNamingPresentation);
    const float native_hor_plus_scale =
        g_ActiveConfig.bColosseumNamingPresentation ? naming_scale :
        g_ActiveConfig.bWidescreenAuthoredMenu ? authored_menu_scale :
        square_offscreen_viewport ? offscreen_projection_scale :
        g_ActiveConfig.bWidescreenHudSafeArea && is_cached_widescreen_projection ?
            authored_projection_scale :
            1.0f;
    const float horizontal_scale =
        g_ActiveConfig.fAspectRatioHackW * fov_multiplier.x * native_hor_plus_scale;
    // MODERNGEKKO_PROJ_TRACE=1: log each distinct perspective pass (viewport,
    // raw aspect, applied horizontal scale) - temporary culling diagnostic.
    static const bool proj_trace = [] {
      const char* v = std::getenv("MODERNGEKKO_PROJ_TRACE");
      return v && v[0] == '1';
    }();
    if (proj_trace && update_stats)
    {
      static std::set<std::tuple<int, int, int, int, int, int>> seen;
      const auto key = std::make_tuple(
          static_cast<int>(xfmem.viewport.wd * 2), static_cast<int>(xfmem.viewport.ht * 2),
          static_cast<int>(xfmem.viewport.xOrig), static_cast<int>(xfmem.viewport.yOrig),
          static_cast<int>(raw_aspect * 1000), static_cast<int>(horizontal_scale * 1000));
      if (seen.insert(key).second && seen.size() < 200)
      {
        std::fprintf(stderr,
                     "[proj] vp=%.1fx%.1f orig=%.1f,%.1f raw_aspect=%.3f hscale=%.3f "
                     "p0=%.4f p1=%.4f p2=%.4f p3=%.4f square=%d cached16x9=%d\n",
                     xfmem.viewport.wd * 2, xfmem.viewport.ht * 2, xfmem.viewport.xOrig,
                     xfmem.viewport.yOrig, raw_aspect, horizontal_scale, rawProjection[0],
                     rawProjection[1], rawProjection[2], rawProjection[3],
                     square_offscreen_viewport ? 1 : 0, is_cached_widescreen_projection ? 1 : 0);
      }
    }
    pm[0] = rawProjection[0] * horizontal_scale;
    pm[1] = 0.0f;
    pm[2] = rawProjection[1] * horizontal_scale;
    pm[3] = 0.0f;

    pm[4] = 0.0f;
    pm[5] = rawProjection[2] * g_ActiveConfig.fAspectRatioHackH * fov_multiplier.y;
    pm[6] = rawProjection[3] * g_ActiveConfig.fAspectRatioHackH * fov_multiplier.y;
    pm[7] = 0.0f;

    pm[8] = 0.0f;
    pm[9] = 0.0f;
    pm[10] = rawProjection[4];
    pm[11] = rawProjection[5];

    pm[12] = 0.0f;
    pm[13] = 0.0f;

    pm[14] = -1.0f;
    pm[15] = 0.0f;

    if (update_stats)
      g_stats.gproj = pm;
  }
  break;

  case ProjectionType::Orthographic:
  {
    // Colosseum composites the perspective world and its 2D overlays through
    // the same orthographic full-screen quad. Scaling this projection shrinks
    // the whole camera and exposes stale EFB regions, so preserve it verbatim.
    // Complete menu canvases are handled by the presenter instead. The naming
    // screen is the one exception: its confirmation panels were authored out
    // to x=660 for a 602-pixel VI aperture, so center the complete 640-pixel
    // composition with the matching 602/640 horizontal scale.
    const float naming_scale = VideoCommon::GetColosseumNamingHorizontalScale(
        g_ActiveConfig.bColosseumNamingPresentation);
    pm[0] = rawProjection[0] * naming_scale;
    pm[1] = 0.0f;
    pm[2] = 0.0f;
    pm[3] = rawProjection[1] * naming_scale;

    pm[4] = 0.0f;
    pm[5] = rawProjection[2];
    pm[6] = 0.0f;
    pm[7] = rawProjection[3];

    pm[8] = 0.0f;
    pm[9] = 0.0f;
    pm[10] = rawProjection[4];
    pm[11] = rawProjection[5];

    pm[12] = 0.0f;
    pm[13] = 0.0f;

    pm[14] = 0.0f;
    pm[15] = 1.0f;

    if (update_stats)
    {
      g_stats.g2proj = pm;
      g_stats.proj = rawProjection;
    }
  }
  break;

  default:
    ERROR_LOG_FMT(VIDEO, "Unknown projection type: {}", xfmem.projection.type);
  }

  PRIM_LOG("Projection: {} {} {} {} {} {}", rawProjection[0], rawProjection[1], rawProjection[2],
           rawProjection[3], rawProjection[4], rawProjection[5]);

  auto corrected_matrix = Common::Matrix44::FromArray(pm);

  if (g_freelook_camera.IsActive() && xfmem.projection.type == ProjectionType::Perspective)
    corrected_matrix *= g_freelook_camera.GetView();

  g_freelook_camera.GetController()->SetClean();

  return corrected_matrix;
}

void VertexShaderManager::SetProjectionMatrix(XFStateManager& xf_state_manager)
{
  if (xf_state_manager.DidProjectionChange() || g_freelook_camera.GetController()->IsDirty())
  {
    xf_state_manager.ResetProjection();
    auto corrected_matrix = LoadProjectionMatrix();
    memcpy(constants.projection.data(), corrected_matrix.data.data(), 4 * sizeof(float4));
  }
}

bool VertexShaderManager::UseVertexDepthRange()
{
  // Backend has full support for unrestricted depth ranges including the ability to clamp the
  // final depth value to MAX_EFB_DEPTH.
  if (g_backend_info.bSupportsUnrestrictedDepthRange)
    return false;

  // We can't compute the depth range in the vertex shader if we don't support depth clamp.
  if (!g_backend_info.bSupportsDepthClamp)
    return false;

  // We need a full depth range if a ztexture is used.
  if (bpmem.ztex2.op != ZTexOp::Disabled && !bpmem.zcontrol.early_ztest)
    return true;

  // If an inverted depth range is unsupported, we also need to check if the range is inverted.
  if (!g_backend_info.bSupportsReversedDepthRange)
  {
    if (xfmem.viewport.zRange < 0.0f)
      return true;

    if (xfmem.viewport.zRange > xfmem.viewport.farZ)
      return true;
  }

  // If an oversized depth range or a ztexture is used, we need to calculate the depth range
  // in the vertex shader.
  return fabs(xfmem.viewport.zRange) > 16777215.0f || fabs(xfmem.viewport.farZ) > 16777215.0f;
}

// Syncs the shader constant buffers with xfmem
// TODO: A cleaner way to control the matrices without making a mess in the parameters field
void VertexShaderManager::SetConstants(std::span<const std::string> textures,
                                       XFStateManager& xf_state_manager)
{
  if (constants.missing_color_hex != g_ActiveConfig.iMissingColorValue)
  {
    const float a = (g_ActiveConfig.iMissingColorValue) & 0xFF;
    const float b = (g_ActiveConfig.iMissingColorValue >> 8) & 0xFF;
    const float g = (g_ActiveConfig.iMissingColorValue >> 16) & 0xFF;
    const float r = (g_ActiveConfig.iMissingColorValue >> 24) & 0xFF;
    constants.missing_color_hex = g_ActiveConfig.iMissingColorValue;
    constants.missing_color_value = {r / 255, g / 255, b / 255, a / 255};

    dirty = true;
  }

  const auto per_vertex_transform_matrix_changes =
      xf_state_manager.GetPerVertexTransformMatrixChanges();
  if (per_vertex_transform_matrix_changes[0] >= 0)
  {
    int startn = per_vertex_transform_matrix_changes[0] / 4;
    int endn = (per_vertex_transform_matrix_changes[1] + 3) / 4;
    memcpy(constants.transformmatrices[startn].data(), &xfmem.posMatrices[startn * 4],
           (endn - startn) * sizeof(float4));
    dirty = true;
    xf_state_manager.ResetPerVertexTransformMatrixChanges();
  }

  const auto per_vertex_normal_matrices_changed =
      xf_state_manager.GetPerVertexNormalMatrixChanges();
  if (per_vertex_normal_matrices_changed[0] >= 0)
  {
    int startn = per_vertex_normal_matrices_changed[0] / 3;
    int endn = (per_vertex_normal_matrices_changed[1] + 2) / 3;
    for (int i = startn; i < endn; i++)
    {
      memcpy(constants.normalmatrices[i].data(), &xfmem.normalMatrices[3 * i], 12);
    }
    dirty = true;
    xf_state_manager.ResetPerVertexNormalMatrixChanges();
  }

  const auto post_transform_matrices_changed = xf_state_manager.GetPostTransformMatrixChanges();
  if (post_transform_matrices_changed[0] >= 0)
  {
    int startn = post_transform_matrices_changed[0] / 4;
    int endn = (post_transform_matrices_changed[1] + 3) / 4;
    memcpy(constants.posttransformmatrices[startn].data(), &xfmem.postMatrices[startn * 4],
           (endn - startn) * sizeof(float4));
    dirty = true;
    xf_state_manager.ResetPostTransformMatrixChanges();
  }

  const auto light_changes = xf_state_manager.GetLightsChanged();
  if (light_changes[0] >= 0)
  {
    // TODO: Outdated comment
    // lights don't have a 1 to 1 mapping, the color component needs to be converted to 4 floats
    const int istart = light_changes[0] / 0x10;
    const int iend = (light_changes[1] + 15) / 0x10;

    for (int i = istart; i < iend; ++i)
    {
      const Light& light = xfmem.lights[i];
      VertexShaderConstants::Light& dstlight = constants.lights[i];

      // xfmem.light.color is packed as abgr in u8[4], so we have to swap the order
      dstlight.color[0] = light.color[3];
      dstlight.color[1] = light.color[2];
      dstlight.color[2] = light.color[1];
      dstlight.color[3] = light.color[0];

      dstlight.cosatt[0] = light.cosatt[0];
      dstlight.cosatt[1] = light.cosatt[1];
      dstlight.cosatt[2] = light.cosatt[2];

      if (fabs(light.distatt[0]) < 0.00001f && fabs(light.distatt[1]) < 0.00001f &&
          fabs(light.distatt[2]) < 0.00001f)
      {
        // dist attenuation, make sure not equal to 0!!!
        dstlight.distatt[0] = .00001f;
      }
      else
      {
        dstlight.distatt[0] = light.distatt[0];
      }
      dstlight.distatt[1] = light.distatt[1];
      dstlight.distatt[2] = light.distatt[2];

      dstlight.pos[0] = light.dpos[0];
      dstlight.pos[1] = light.dpos[1];
      dstlight.pos[2] = light.dpos[2];

      // TODO: Hardware testing is needed to confirm that this normalization is correct
      auto sanitize = [](float f) {
        if (std::isnan(f))
          return 0.0f;
        else if (std::isinf(f))
          return f > 0.0f ? 1.0f : -1.0f;
        else
          return f;
      };
      double norm = double(light.ddir[0]) * double(light.ddir[0]) +
                    double(light.ddir[1]) * double(light.ddir[1]) +
                    double(light.ddir[2]) * double(light.ddir[2]);
      norm = 1.0 / sqrt(norm);
      dstlight.dir[0] = sanitize(static_cast<float>(light.ddir[0] * norm));
      dstlight.dir[1] = sanitize(static_cast<float>(light.ddir[1] * norm));
      dstlight.dir[2] = sanitize(static_cast<float>(light.ddir[2] * norm));
    }
    dirty = true;

    xf_state_manager.ResetLightsChanged();
  }

  for (int i : xf_state_manager.GetMaterialChanges())
  {
    u32 data = i >= 2 ? xfmem.matColor[i - 2] : xfmem.ambColor[i];
    constants.materials[i][0] = (data >> 24) & 0xFF;
    constants.materials[i][1] = (data >> 16) & 0xFF;
    constants.materials[i][2] = (data >> 8) & 0xFF;
    constants.materials[i][3] = data & 0xFF;
    dirty = true;
  }
  xf_state_manager.ResetMaterialChanges();

  if (xf_state_manager.DidPosNormalChange())
  {
    xf_state_manager.ResetPosNormalChange();
    const float* pos = &xfmem.posMatrices[g_main_cp_state.matrix_index_a.PosNormalMtxIdx * 4];
    const float* norm =
        &xfmem.normalMatrices[3 * (g_main_cp_state.matrix_index_a.PosNormalMtxIdx & 31)];

    memcpy(constants.posnormalmatrix.data(), pos, 3 * sizeof(float4));
    memcpy(constants.posnormalmatrix[3].data(), norm, 3 * sizeof(float));
    memcpy(constants.posnormalmatrix[4].data(), norm + 3, 3 * sizeof(float));
    memcpy(constants.posnormalmatrix[5].data(), norm + 6, 3 * sizeof(float));
    dirty = true;
  }

  if (xf_state_manager.DidTexMatrixAChange())
  {
    xf_state_manager.ResetTexMatrixAChange();
    const std::array<const float*, 4> pos_matrix_ptrs{
        &xfmem.posMatrices[g_main_cp_state.matrix_index_a.Tex0MtxIdx * 4],
        &xfmem.posMatrices[g_main_cp_state.matrix_index_a.Tex1MtxIdx * 4],
        &xfmem.posMatrices[g_main_cp_state.matrix_index_a.Tex2MtxIdx * 4],
        &xfmem.posMatrices[g_main_cp_state.matrix_index_a.Tex3MtxIdx * 4],
    };

    for (size_t i = 0; i < pos_matrix_ptrs.size(); ++i)
    {
      memcpy(constants.texmatrices[3 * i].data(), pos_matrix_ptrs[i], 3 * sizeof(float4));
    }
    dirty = true;
  }

  if (xf_state_manager.DidTexMatrixBChange())
  {
    xf_state_manager.ResetTexMatrixBChange();
    const std::array<const float*, 4> pos_matrix_ptrs{
        &xfmem.posMatrices[g_main_cp_state.matrix_index_b.Tex4MtxIdx * 4],
        &xfmem.posMatrices[g_main_cp_state.matrix_index_b.Tex5MtxIdx * 4],
        &xfmem.posMatrices[g_main_cp_state.matrix_index_b.Tex6MtxIdx * 4],
        &xfmem.posMatrices[g_main_cp_state.matrix_index_b.Tex7MtxIdx * 4],
    };

    for (size_t i = 0; i < pos_matrix_ptrs.size(); ++i)
    {
      memcpy(constants.texmatrices[3 * i + 12].data(), pos_matrix_ptrs[i], 3 * sizeof(float4));
    }
    dirty = true;
  }

  if (xf_state_manager.DidViewportChange())
  {
    xf_state_manager.ResetViewportChange();

    // The console GPU places the pixel center at 7/12 unless antialiasing
    // is enabled, while D3D and OpenGL place it at 0.5. See the comment
    // in VertexShaderGen.cpp for details.
    // NOTE: If we ever emulate antialiasing, the sample locations set by
    // BP registers 0x01-0x04 need to be considered here.
    const float pixel_center_correction = 7.0f / 12.0f - 0.5f;
    const bool bUseVertexRounding = g_ActiveConfig.UseVertexRounding();
    const float viewport_width = bUseVertexRounding ?
                                     (2.f * xfmem.viewport.wd) :
                                     g_framebuffer_manager->EFBToScaledXf(2.f * xfmem.viewport.wd);
    const float viewport_height = bUseVertexRounding ?
                                      (2.f * xfmem.viewport.ht) :
                                      g_framebuffer_manager->EFBToScaledXf(2.f * xfmem.viewport.ht);
    const float pixel_size_x = 2.f / viewport_width;
    const float pixel_size_y = 2.f / viewport_height;
    constants.pixelcentercorrection[0] = pixel_center_correction * pixel_size_x;
    constants.pixelcentercorrection[1] = pixel_center_correction * pixel_size_y;

    // By default we don't change the depth value at all in the vertex shader.
    constants.pixelcentercorrection[2] = 1.0f;
    constants.pixelcentercorrection[3] = 0.0f;

    constants.viewport[0] = (2.f * xfmem.viewport.wd);
    constants.viewport[1] = (2.f * xfmem.viewport.ht);

    if (UseVertexDepthRange())
    {
      // Oversized depth ranges are handled in the vertex shader. We need to reverse
      // the far value to use the reversed-Z trick.
      if (g_backend_info.bSupportsReversedDepthRange)
      {
        // Sometimes the console also tries to use the reversed-Z trick. We can only do
        // that with the expected accuracy if the backend can reverse the depth range.
        constants.pixelcentercorrection[2] = fabs(xfmem.viewport.zRange) / 16777215.0f;
        if (xfmem.viewport.zRange < 0.0f)
          constants.pixelcentercorrection[3] = xfmem.viewport.farZ / 16777215.0f;
        else
          constants.pixelcentercorrection[3] = 1.0f - xfmem.viewport.farZ / 16777215.0f;
      }
      else
      {
        // For backends that don't support reversing the depth range we can still render
        // cases where the console uses the reversed-Z trick. But we simply can't provide
        // the expected accuracy, which might result in z-fighting.
        constants.pixelcentercorrection[2] = xfmem.viewport.zRange / 16777215.0f;
        constants.pixelcentercorrection[3] = 1.0f - xfmem.viewport.farZ / 16777215.0f;
      }
    }

    dirty = true;
    BPFunctions::SetScissorAndViewport(g_framebuffer_manager.get(), bpmem.scissorTL,
                                       bpmem.scissorBR, bpmem.scissorOffset, xfmem.viewport);
    g_stats.AddScissorRect();
  }

  std::vector<GraphicsModAction*> projection_actions;
  if (g_ActiveConfig.bGraphicMods)
  {
    for (const auto& action : g_graphics_mod_manager->GetProjectionActions(xfmem.projection.type))
    {
      projection_actions.push_back(action);
    }

    for (const auto& texture : textures)
    {
      for (const auto& action :
           g_graphics_mod_manager->GetProjectionTextureActions(xfmem.projection.type, texture))
      {
        projection_actions.push_back(action);
      }
    }
  }

  if (xf_state_manager.DidProjectionChange() || g_freelook_camera.GetController()->IsDirty() ||
      !projection_actions.empty() || m_projection_graphics_mod_change)
  {
    xf_state_manager.ResetProjection();
    m_projection_graphics_mod_change = !projection_actions.empty();

    auto corrected_matrix = LoadProjectionMatrix();

    GraphicsModActionData::Projection projection{&corrected_matrix};
    for (const auto& action : projection_actions)
    {
      action->OnProjection(&projection);
    }

    memcpy(constants.projection.data(), corrected_matrix.data.data(), 4 * sizeof(float4));
    dirty = true;
  }

  if (xf_state_manager.DidTexMatrixInfoChange())
  {
    xf_state_manager.ResetTexMatrixInfoChange();
    constants.xfmem_dualTexInfo = xfmem.dualTexTrans.enabled;
    for (size_t i = 0; i < std::size(xfmem.texMtxInfo); i++)
      constants.xfmem_pack1[i][0] = xfmem.texMtxInfo[i].hex;
    for (size_t i = 0; i < std::size(xfmem.postMtxInfo); i++)
      constants.xfmem_pack1[i][1] = xfmem.postMtxInfo[i].hex;

    dirty = true;
  }

  if (xf_state_manager.DidLightingConfigChange())
  {
    xf_state_manager.ResetLightingConfigChange();

    for (size_t i = 0; i < 2; i++)
    {
      constants.xfmem_pack1[i][2] = xfmem.color[i].hex;
      constants.xfmem_pack1[i][3] = xfmem.alpha[i].hex;
    }
    constants.xfmem_numColorChans = xfmem.numChan.numColorChans;
    dirty = true;
  }

  if (FrameInterp::Enabled())
    SetInterpConstants();
}

// Frame interpolation: the layer-1 copies of the position matrices and the
// projection, from FrameInterp's blended shadow of XF memory. Kept separate
// from the dirty tracking above because they change whenever a paired load
// blends, not only when the live XF state changes.
void VertexShaderManager::SetInterpConstants()
{
  if (FrameInterp::g_pos_dirty)
  {
    FrameInterp::g_pos_dirty = false;
    memcpy(constants.transformmatrices_b.data(), FrameInterp::g_pos, sizeof(FrameInterp::g_pos));
    dirty = true;
  }

  const u32 pn = g_main_cp_state.matrix_index_a.PosNormalMtxIdx;
  const float* pos = &FrameInterp::g_pos[(pn & 0x3f) * 4];
  if (memcmp(constants.posnormalmatrix_b.data(), pos, 3 * sizeof(float4)) != 0)
  {
    memcpy(constants.posnormalmatrix_b.data(), pos, 3 * sizeof(float4));
    dirty = true;
  }

  std::array<float, 6> raw;
  std::copy(std::begin(FrameInterp::g_proj), std::end(FrameInterp::g_proj), raw.begin());
  const Common::Matrix44 projection = LoadProjectionMatrix(raw, &m_projection_matrix_b, false);
  if (memcmp(constants.projection_b.data(), projection.data.data(), 4 * sizeof(float4)) != 0)
  {
    memcpy(constants.projection_b.data(), projection.data.data(), 4 * sizeof(float4));
    dirty = true;
  }
}

void VertexShaderManager::TransformToClipSpace(const float* data, float* out, u32 MtxIdx)
{
  const float* world_matrix = &xfmem.posMatrices[(MtxIdx & 0x3f) * 4];

  // We use the projection matrix calculated by VertexShaderManager, because it
  // includes any free look transformations.
  // Make sure VertexShaderManager::SetConstants() has been called first.
  const float* proj_matrix = &m_projection_matrix[0];

  const float t[3] = {data[0] * world_matrix[0] + data[1] * world_matrix[1] +
                          data[2] * world_matrix[2] + world_matrix[3],
                      data[0] * world_matrix[4] + data[1] * world_matrix[5] +
                          data[2] * world_matrix[6] + world_matrix[7],
                      data[0] * world_matrix[8] + data[1] * world_matrix[9] +
                          data[2] * world_matrix[10] + world_matrix[11]};

  out[0] = t[0] * proj_matrix[0] + t[1] * proj_matrix[1] + t[2] * proj_matrix[2] + proj_matrix[3];
  out[1] = t[0] * proj_matrix[4] + t[1] * proj_matrix[5] + t[2] * proj_matrix[6] + proj_matrix[7];
  out[2] = t[0] * proj_matrix[8] + t[1] * proj_matrix[9] + t[2] * proj_matrix[10] + proj_matrix[11];
  out[3] =
      t[0] * proj_matrix[12] + t[1] * proj_matrix[13] + t[2] * proj_matrix[14] + proj_matrix[15];
}

void VertexShaderManager::DoState(PointerWrap& p)
{
  p.DoArray(m_projection_matrix);
  g_freelook_camera.DoState(p);

  // Serialize only the original constants: the frame-interpolation fields
  // appended to VertexShaderConstants are derived state, and including them
  // would change the savestate layout and reject every existing state.
  p.DoArray(reinterpret_cast<u8*>(&constants),
            static_cast<u32>(offsetof(VertexShaderConstants, transformmatrices_b)));

  if (p.IsReadMode())
  {
    dirty = true;
    FrameInterp::g_pos_dirty = true;
  }
}
