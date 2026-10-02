// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include "Common/CommonTypes.h"

class AbstractShader;
class AbstractTexture;

namespace VideoCommon
{
// Experimental image correspondence, not geometry-derived motion. Inputs must be matched XFB
// snapshots with identical crop/scale and one layer. History advances on new XFBs, not presents.
// MODERNGEKKO_DLSS_FLOW_HALF=1 opts into a ceil-half estimation grid; outputs stay full sized.
class ModernGekkoOpticalFlow
{
public:
  struct Inputs
  {
    const AbstractTexture* current_color = nullptr;
    const AbstractTexture* previous_color = nullptr;
    const AbstractTexture* current_depth = nullptr;
    const AbstractTexture* previous_depth = nullptr;
    // Optional paired masks: 1 only where world projection received jitter; UI must be 0.
    const AbstractTexture* current_jitter_mask = nullptr;
    const AbstractTexture* previous_jitter_mask = nullptr;
    u32 width = 0, height = 0;
    std::array<float, 2> current_jitter{}, previous_jitter{};
    bool history_valid = false;
    // Explicit caller guarantee: EVERY input pixel received the reported projection jitter.
    // Do not set for mixed world/UI images. Paired masks take precedence when supplied.
    bool uniform_jitter_coverage = false;
    // Raw normalized-depth disagreement is conservative rejection, not 3D reprojection.
    float depth_rejection_threshold = 0.015f;
  };
  struct Result
  {
    // RG16F: xy current->previous unjittered render pixels.
    const AbstractTexture* motion = nullptr;
    // RGBA8: R=1-confidence (current-color bias), alpha=1. SDK format must be validated.
    const AbstractTexture* bias = nullptr;
    bool history_used = false;
    // No CPU readback. Optional timestamp profiling is nonblocking; coverage remains caller-owned.
  };

  ModernGekkoOpticalFlow();
  ~ModernGekkoOpticalFlow();
  ModernGekkoOpticalFlow(const ModernGekkoOpticalFlow&) = delete;
  ModernGekkoOpticalFlow& operator=(const ModernGekkoOpticalFlow&) = delete;
  // Results remain owned here and are overwritten by the next Run; destroy before g_gfx teardown.
  // A null result indicates unsupported inputs/allocation/shader failure.
  Result Run(const Inputs& inputs);

private:
  static constexpr std::size_t LEVELS = 5;
  bool Initialize(u32 width, u32 height);
  void BindSample(u32 slot, const AbstractTexture* texture);
  void Dispatch(const AbstractShader* shader, u32 width, u32 height);
  std::unique_ptr<AbstractShader> m_downsample, m_estimate, m_confidence;
  std::array<std::array<std::unique_ptr<AbstractTexture>, LEVELS>, 2> m_color;
  std::array<std::array<std::unique_ptr<AbstractTexture>, LEVELS>, 2> m_flow;
  std::unique_ptr<AbstractTexture> m_motion, m_bias;
  u32 m_width = 0, m_height = 0;
  bool m_half_resolution = false;
};
}  // namespace VideoCommon
