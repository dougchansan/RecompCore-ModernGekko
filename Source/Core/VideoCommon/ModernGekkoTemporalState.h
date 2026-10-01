// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include "Common/CommonTypes.h"

// Graphics thread only. Native EFB pixels, +right/+down. Packet owner advances content IDs.
namespace ModernGekkoTemporal
{
enum class CoverageReason : u32
{
  Unclassified = 1u << 0,
  NoRecognizedDraw = 1u << 1,
  NonXFBCopy = 1u << 2,
  PartialClear = 1u << 3,
  Poke = 1u << 4,
  Reinterpret = 1u << 5,
  MixedViewport = 1u << 6,
  InvalidViewport = 1u << 7,
  Scissor = 1u << 8,
  InputScaleChanged = 1u << 9,
  InvalidJitter = 1u << 10,
  DepthProjectionChanged = 1u << 11,
  MixedDepthConvention = 1u << 12
};
struct State
{
  bool active = false;
  bool coverage_valid = false;
  bool depth_convention_known = false;
  bool depth_inverted = false;
  u32 reason_bits = static_cast<u32>(CoverageReason::NoRecognizedDraw);
  u32 recognized_draws = 0;
  u32 viewport_changes = 0;
  u32 depth_projection_changes = 0;
  u32 clear_count = 0;
  std::array<std::array<int, 4>, 16> clear_regions{};
  float jitter_x = 0.0f;
  float jitter_y = 0.0f;
  float input_per_native_x = 1.0f;
  float input_per_native_y = 1.0f;
  int scissor_left = 0;
  int scissor_top = 0;
  int scissor_right = 640;
  int scissor_bottom = 528;
  u64 serial = 0;
  u64 content_id = 0;
};
void SetActive(bool active);
void SetFrameJitter(float x, float y, u64 content_id);
// SDK optimal input pixels / native crop pixels. Halton amplitude is one input pixel.
void SetInputPixelsPerNative(float x, float y);
void MarkCoverageInvalid(CoverageReason reason = CoverageReason::Unclassified);
void RecordViewportChange();
void RecordDepthProjectionChange();
// Raw captured EFB near/far convention; reject mixed conventions in one content frame.
void RecordDepthConvention(bool inverted);
void RecordClearRegion(int left, int top, int right, int bottom);
void MarkGuestDrawCoverage(bool recognized_mapping);
void MarkScissorCoverage(int left, int top, int right, int bottom);
State GetState();
bool HasClearCoverageForRect(const State& state, int left, int top, int right, int bottom);
// Signed viewport dimensions; shader mirroring precedes jitter. Before backend Y flip.
std::array<float, 2> ComputeClipJitter(float x, float y, float width, float height);
}
