// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "Common/CommonTypes.h"
#include "InputCommon/ControllerInterface/CoreDevice.h"

class InputConfig;
enum class PadGroup;
struct GCPadStatus;

namespace ControllerEmu
{
class ControlGroup;
}

namespace Pad
{
void Shutdown();
void Initialize();
void LoadConfig();
void GenerateDynamicInputTextures();
bool IsInitialized();

InputConfig* GetConfig();

GCPadStatus GetStatus(int pad_num);

// Force buttons to be held on a port regardless of the attached device, for
// benchmark automation. A race savestate loads with the kart parked at the
// line: the AI field races and the scene renders, but the camera never moves,
// so no track geometry streams in and the measured load understates driving.
// Pass 0 to clear.
void SetHeldButtons(int pad_num, u16 buttons);
ControllerEmu::ControlGroup* GetGroup(int pad_num, PadGroup group);
void Rumble(int pad_num, ControlState strength);
void ResetRumble(int pad_num);

bool GetMicButton(int pad_num);
}  // namespace Pad
