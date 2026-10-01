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
  u64 serial = 0;
  u64 content_id = 0;
};
void SetActive(bool active);
void SetFrameJitter(float x, float y, u64 content_id);
void MarkCoverageInvalid();
State GetState();
// Signed viewport dimensions; shader mirroring precedes jitter. Before backend Y flip.
std::array<float, 2> ComputeClipJitter(float x, float y, float width, float height);
}
