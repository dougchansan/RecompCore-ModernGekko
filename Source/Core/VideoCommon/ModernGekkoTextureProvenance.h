// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "Common/CommonTypes.h"
struct BPMemory;
namespace ModernGekkoTextureProvenance
{
struct DependencyReport
{
  u32 rgb_slots = 0;
  u32 alpha_slots = 0;
  u32 indirect_slots = 0;
  // Conservative existing policy: all requested slots when Z-texture depth is enabled.
  u32 z_slots = 0;
  bool invalid_slots = false;
};
DependencyReport AnalyzeDependencies(const BPMemory& bp, u32 auxiliary_slots = 0xffu);
// Proves only that auxiliary texture slots do not feed final TEV RGB, indirect
// coordinates or Z-texture depth. Caller still proves opacity use and blend/depth survival.
bool CanUseAsOpacityData(const BPMemory& bp, u32 auxiliary_slots);
}
