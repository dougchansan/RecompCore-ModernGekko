// Copyright 2014 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/GeometryShaderManager.h"

#include <cmath>
#include <cstddef>

#include "Common/ChunkFile.h"
#include "Common/CommonTypes.h"
#include "Core/System.h"
#include "VideoCommon/BPMemory.h"
#include "VideoCommon/CameraWarp.h"
#include "VideoCommon/VertexShaderManager.h"
#include "VideoCommon/RenderState.h"
#include "VideoCommon/ShaderGenCommon.h"
#include "VideoCommon/VideoConfig.h"
#include "VideoCommon/XFMemory.h"

static constexpr int LINE_PT_TEX_OFFSETS[8] = {0, 16, 8, 4, 2, 1, 1, 1};

static void SetIdentityCamera(std::array<float4, 16>* m)
{
  for (u32 i = 0; i < 16; ++i)
    (*m)[i] = {i % 4 == 0 ? 1.0f : 0.0f, i % 4 == 1 ? 1.0f : 0.0f, i % 4 == 2 ? 1.0f : 0.0f,
               i % 4 == 3 ? 1.0f : 0.0f};
}

void GeometryShaderManager::Init()
{
  constants = {};
  SetIdentityCamera(&constants.interp_camera);
  m_camera_warp_active = false;

  // Init any initial constants which aren't zero when bpmem is zero.
  SetViewportChanged();
  SetProjectionChanged();

  dirty = true;
}

void GeometryShaderManager::Dirty()
{
  // This function is called after a savestate is loaded.
  // Any constants that can changed based on settings should be re-calculated
  m_projection_changed = true;

  // Uses EFB scale config
  SetLinePtWidthChanged();

  dirty = true;
}

void GeometryShaderManager::SetVSExpand(VSExpand expand)
{
  if (constants.vs_expand != expand)
  {
    constants.vs_expand = expand;
    dirty = true;
  }
}

void GeometryShaderManager::SetConstants(PrimitiveType prim)
{
  if (m_projection_changed && g_ActiveConfig.stereo_mode != StereoMode::Off)
  {
    m_projection_changed = false;

    if (xfmem.projection.type == ProjectionType::Perspective)
    {
      const float offset = g_ActiveConfig.stereo_depth;
      constants.stereoparams[0] = g_ActiveConfig.bStereoSwapEyes ? offset : -offset;
      constants.stereoparams[1] = g_ActiveConfig.bStereoSwapEyes ? -offset : offset;
    }
    else
    {
      constants.stereoparams[0] = constants.stereoparams[1] = 0;
    }

    constants.stereoparams[2] = g_ActiveConfig.stereo_convergence;

    dirty = true;
  }

  if (g_ActiveConfig.UseVSForLinePointExpand())
  {
    if (prim == PrimitiveType::Points)
      SetVSExpand(VSExpand::Point);
    else if (prim == PrimitiveType::Lines)
      SetVSExpand(VSExpand::Line);
    else
      SetVSExpand(VSExpand::None);
  }

  if (m_viewport_changed)
  {
    m_viewport_changed = false;

    constants.lineptparams[0] = 2.0f * xfmem.viewport.wd;
    constants.lineptparams[1] = -2.0f * xfmem.viewport.ht;

    dirty = true;
  }

  if (g_ActiveConfig.stereo_mode == StereoMode::FrameInterp)
    UpdateCameraWarp();
}

// 240 Hz camera (see CameraWarp.h): only the main perspective pass - the full
// viewport excludes the 384x384 shadow passes; HUD and composite are
// orthographic.
void GeometryShaderManager::UpdateCameraWarp()
{
  bool active = xfmem.projection.type == ProjectionType::Perspective &&
                std::abs(xfmem.viewport.wd) * 2.0f >= 560.0f;
  if (active)
  {
    const auto& vs = Core::System::GetInstance().GetVertexShaderManager().constants;
    const u32 layers = ShaderHostConfig::GetCurrent().frame_interp4 ? 4 : 2;
    std::array<float4, 16> matrices;
    active = CameraWarp::GetLayerMatrices(layers, vs.projection, vs.pixelcentercorrection,
                                          g_backend_info.bSupportsClipControl, &matrices);
    if (active && matrices != constants.interp_camera)
    {
      constants.interp_camera = matrices;
      dirty = true;
    }
  }
  if (!active && m_camera_warp_active)
  {
    SetIdentityCamera(&constants.interp_camera);
    dirty = true;
  }
  m_camera_warp_active = active;
}

void GeometryShaderManager::SetViewportChanged()
{
  m_viewport_changed = true;
}

void GeometryShaderManager::SetProjectionChanged()
{
  m_projection_changed = true;
}

void GeometryShaderManager::SetLinePtWidthChanged()
{
  constants.lineptparams[2] = bpmem.lineptwidth.linesize / 6.f;
  constants.lineptparams[3] = bpmem.lineptwidth.pointsize / 6.f;
  constants.texoffset[2] = LINE_PT_TEX_OFFSETS[bpmem.lineptwidth.lineoff];
  constants.texoffset[3] = LINE_PT_TEX_OFFSETS[bpmem.lineptwidth.pointoff];
  dirty = true;
}

void GeometryShaderManager::SetTexCoordChanged(u8 texmapid)
{
  TCoordInfo& tc = bpmem.texcoords[texmapid];
  int bitmask = 1 << texmapid;
  constants.texoffset[0] &= ~bitmask;
  constants.texoffset[0] |= tc.s.line_offset << texmapid;
  constants.texoffset[1] &= ~bitmask;
  constants.texoffset[1] |= tc.s.point_offset << texmapid;
  dirty = true;
}

void GeometryShaderManager::DoState(PointerWrap& p)
{
  p.Do(m_projection_changed);
  p.Do(m_viewport_changed);

  // Only the original constants: interp_camera is derived each frame, and
  // saving it would change the savestate layout.
  p.DoArray(reinterpret_cast<u8*>(&constants),
            static_cast<u32>(offsetof(GeometryShaderConstants, interp_camera)));

  if (p.IsReadMode())
  {
    // Fixup the current state from global GPU state
    // NOTE: This requires that all GPU memory has been loaded already.
    Dirty();
  }
}
