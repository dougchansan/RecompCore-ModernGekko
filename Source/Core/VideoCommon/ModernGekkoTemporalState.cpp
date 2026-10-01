// SPDX-License-Identifier: GPL-2.0-or-later
#include "VideoCommon/ModernGekkoTemporalState.h"
#include <cmath>
#include <algorithm>

namespace ModernGekkoTemporal
{
namespace
{
State s_state;
bool s_has_frame = false;
bool s_coverage_invalid = false;
std::array<int, 4> s_current_scissor{0, 0, 640, 528};
}
void SetActive(bool active)
{
  if (s_state.active == active)
    return;
  s_state.active = active;
  s_state.coverage_valid = false;
  s_state.depth_convention_known = false;
  s_state.depth_inverted = false;
  s_state.reason_bits = static_cast<u32>(CoverageReason::NoRecognizedDraw);
  s_state.recognized_draws = s_state.viewport_changes = s_state.depth_projection_changes = s_state.clear_count = 0;
  s_state.jitter_x = s_state.jitter_y = 0.0f;
  s_has_frame = false;
  s_coverage_invalid = false;
  ++s_state.serial;
}
void SetInputPixelsPerNative(float x, float y)
{
  if (!std::isfinite(x) || !std::isfinite(y) || x <= 0.0f || y <= 0.0f)
  {
    MarkCoverageInvalid(CoverageReason::InputScaleChanged);
    return;
  }
  if (s_state.input_per_native_x == x && s_state.input_per_native_y == y)
    return;
  s_state.input_per_native_x = x;
  s_state.input_per_native_y = y;
  // The already-rendering frame used the previous amplitude; do not relabel it.
  MarkCoverageInvalid(CoverageReason::InputScaleChanged);
}
void SetFrameJitter(float x, float y, u64 content_id)
{
  if (!s_state.active)
    return;
  const bool finite = std::isfinite(x) && std::isfinite(y);
  if (!s_has_frame || s_state.content_id != content_id)
  {
    s_state.coverage_valid = false;  // Finite jitter alone does not establish coverage.
    s_state.depth_convention_known = false;
    s_state.depth_inverted = false;
    s_coverage_invalid = !finite;
    s_state.reason_bits = static_cast<u32>(CoverageReason::NoRecognizedDraw);
    s_state.recognized_draws = s_state.viewport_changes = s_state.depth_projection_changes = s_state.clear_count = 0;
    if (!finite)
      s_state.reason_bits |= static_cast<u32>(CoverageReason::InvalidJitter);
    s_state.scissor_left = s_current_scissor[0];
    s_state.scissor_top = s_current_scissor[1];
    s_state.scissor_right = s_current_scissor[2];
    s_state.scissor_bottom = s_current_scissor[3];
  }
  else if (s_state.jitter_x != x || s_state.jitter_y != y)
  {
    s_state.coverage_valid = false;
    s_state.reason_bits |= static_cast<u32>(CoverageReason::InvalidJitter);
    s_coverage_invalid = true;  // One content frame must have one raster offset.
  }
  s_has_frame = true;
  s_state.content_id = content_id;
  s_state.jitter_x = finite ? x : 0.0f;
  s_state.jitter_y = finite ? y : 0.0f;
  if (!finite)
  {
    s_state.coverage_valid = false;
    s_coverage_invalid = true;
    s_state.reason_bits |= static_cast<u32>(CoverageReason::InvalidJitter);
  }
  ++s_state.serial;
}
void MarkCoverageInvalid(CoverageReason reason)
{
  s_state.reason_bits |= static_cast<u32>(reason);
  s_coverage_invalid = true;
  s_state.coverage_valid = false;
  ++s_state.serial;
}
void MarkGuestDrawCoverage(bool recognized_mapping)
{
  if (!s_state.active || !s_has_frame)
    return;
  ++s_state.recognized_draws;
  s_state.reason_bits &= ~static_cast<u32>(CoverageReason::NoRecognizedDraw);
  if (!recognized_mapping)
    MarkCoverageInvalid(CoverageReason::MixedViewport);
  else if (!s_coverage_invalid)
    s_state.coverage_valid = true;
}
void RecordClearRegion(int left, int top, int right, int bottom)
{
  if (!s_state.active)
    return;
  if (s_state.clear_count == s_state.clear_regions.size())
  {
    MarkCoverageInvalid(CoverageReason::PartialClear);
    return;
  }
  s_state.clear_regions[s_state.clear_count++] = {left, top, right, bottom};
}
void RecordDepthConvention(bool inverted)
{
  if (!s_state.active || !s_has_frame)
    return;
  if (!s_state.depth_convention_known)
  {
    s_state.depth_convention_known = true;
    s_state.depth_inverted = inverted;
  }
  else if (s_state.depth_inverted != inverted)
  {
    MarkCoverageInvalid(CoverageReason::MixedDepthConvention);
  }
}
void RecordDepthProjectionChange()
{
  ++s_state.depth_projection_changes;
  // Diagnostic; mixed viewport remains a separate conservative eligibility rejection.
  s_state.reason_bits |= static_cast<u32>(CoverageReason::DepthProjectionChanged);
}
void RecordViewportChange()
{
  ++s_state.viewport_changes;
}
void MarkScissorCoverage(int left, int top, int right, int bottom)
{
  // Keep the current guest configuration even while temporal rendering is disabled.
  s_current_scissor = {left, top, right, bottom};
  if (!s_state.active)
    return;
  s_state.scissor_left = std::max(s_state.scissor_left, left);
  s_state.scissor_top = std::max(s_state.scissor_top, top);
  s_state.scissor_right = std::min(s_state.scissor_right, right);
  s_state.scissor_bottom = std::min(s_state.scissor_bottom, bottom);
  if (left >= right || top >= bottom)
    MarkCoverageInvalid(CoverageReason::Scissor);
}
bool HasClearCoverageForRect(const State& state, int left, int top, int right, int bottom)
{
  for (u32 i = 0; i < state.clear_count; ++i)
  {
    const auto& clear = state.clear_regions[i];
    const bool intersects = clear[0] < right && clear[2] > left &&
                            clear[1] < bottom && clear[3] > top;
    const bool encloses = clear[0] <= left && clear[1] <= top &&
                          clear[2] >= right && clear[3] >= bottom;
    if (intersects && !encloses)
      return false;
  }
  return true;
}
State GetState()
{
  return s_state;
}
std::array<float, 2> ComputeClipJitter(float x, float y, float width, float height)
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) ||
      !std::isfinite(height) || width == 0.0f || height == 0.0f)
    return {};
  return {2.0f * x / std::abs(width), -2.0f * y / std::abs(height)};
}
}
