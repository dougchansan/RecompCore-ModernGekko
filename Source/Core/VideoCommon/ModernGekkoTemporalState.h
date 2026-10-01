// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include "Common/CommonTypes.h"

// Graphics thread only. Native EFB pixels, +right/+down. Packet owner advances content IDs.
namespace ModernGekkoTemporal
{
struct State
{
  bool active = false;
  bool coverage_valid = false;
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
void MarkCoverageInvalid();
void MarkGuestDrawCoverage(bool recognized_mapping);
void MarkScissorCoverage(int left, int top, int right, int bottom);
State GetState();
// Signed viewport dimensions; shader mirroring precedes jitter. Before backend Y flip.
std::array<float, 2> ComputeClipJitter(float x, float y, float width, float height);
}
