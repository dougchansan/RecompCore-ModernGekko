// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <utility>

#include "VideoCommon/XFMemory.h"

std::pair<std::string, std::string> GetXFRegInfo(u32 address, u32 value);
std::string GetXFMemName(u32 address);
std::string GetXFMemDescription(u32 address, u32 value);
std::pair<std::string, std::string> GetXFTransferInfo(u16 base_address, u8 transfer_size,
                                                      const u8* data);
std::pair<std::string, std::string> GetXFIndexedLoadInfo(CPArray array, u32 index, u16 address,
                                                         u8 size);

// Frame interpolation shadow state (see XFStructs.cpp). g_pos mirrors
// xfmem.posMatrices and g_proj the raw projection, with each load blended
// halfway toward the matching load of the previous game frame.
namespace FrameInterp
{
extern float g_pos[256];
extern float g_proj[6];
extern bool g_pos_dirty;
bool Enabled();
void Frame();  // call at each XFB copy
}  // namespace FrameInterp
