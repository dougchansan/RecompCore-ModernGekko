// Copyright 2010 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/HW/GCPad.h"

#include "Common/Common.h"
#include "Core/HW/GCPadEmu.h"
#include "InputCommon/ControllerEmu/ControlGroup/ControlGroup.h"
#include "InputCommon/GCPadStatus.h"
#include "InputCommon/InputConfig.h"

#include <array>
#include <atomic>

namespace Pad
{
static InputConfig s_config("GCPadNew", _trans("Pad"), "GCPad", "Pad");
InputConfig* GetConfig()
{
  return &s_config;
}

void Shutdown()
{
  s_config.UnregisterHotplugCallback();

  s_config.ClearControllers();
}

void Initialize()
{
  if (s_config.ControllersNeedToBeCreated())
  {
    for (unsigned int i = 0; i < 4; ++i)
      s_config.CreateController<GCPad>(i);
  }

  s_config.RegisterHotplugCallback();

  // Load the saved controller config
  s_config.LoadConfig();
}

void LoadConfig()
{
  s_config.LoadConfig();
}

void GenerateDynamicInputTextures()
{
  s_config.GenerateControllerTextures();
}

bool IsInitialized()
{
  return !s_config.ControllersNeedToBeCreated();
}

static std::array<std::atomic<u16>, 4> s_held_buttons{};

void SetHeldButtons(int pad_num, u16 buttons)
{
  if (pad_num >= 0 && pad_num < static_cast<int>(s_held_buttons.size()))
    s_held_buttons[pad_num].store(buttons, std::memory_order_relaxed);
}

GCPadStatus GetStatus(int pad_num)
{
  GCPadStatus status = static_cast<GCPad*>(s_config.GetController(pad_num))->GetInput();
  if (pad_num >= 0 && pad_num < static_cast<int>(s_held_buttons.size()))
  {
    const u16 held = s_held_buttons[pad_num].load(std::memory_order_relaxed);
    if (held != 0)
    {
      status.button |= held;
      // The analog triggers are what MKDD actually reads for acceleration on
      // a real pad; setting only the digital bit leaves it stationary.
      if (held & PAD_TRIGGER_R)
        status.triggerRight = 255;
      if (held & PAD_TRIGGER_L)
        status.triggerLeft = 255;
    }
  }
  return status;
}

ControllerEmu::ControlGroup* GetGroup(int pad_num, PadGroup group)
{
  return static_cast<GCPad*>(s_config.GetController(pad_num))->GetGroup(group);
}

void Rumble(const int pad_num, const ControlState strength)
{
  static_cast<GCPad*>(s_config.GetController(pad_num))->SetOutput(strength);
}

void ResetRumble(const int pad_num)
{
  static_cast<GCPad*>(s_config.GetController(pad_num))->SetOutput(0.0);
}

bool GetMicButton(const int pad_num)
{
  return static_cast<GCPad*>(s_config.GetController(pad_num))->GetMicButton();
}
}  // namespace Pad
