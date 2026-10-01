// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "Common/CommonTypes.h"
struct BPMemory;
namespace ModernGekkoTextureProvenance
{
// Proves only that auxiliary texture slots do not feed final TEV RGB, indirect
// coordinates or Z-texture depth. Caller still proves opacity use and blend/depth survival.
bool CanUseAsOpacityData(const BPMemory& bp, u32 auxiliary_slots);
}
