// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <memory>
#include "Common/CommonTypes.h"
#include "Common/MathUtil.h"
#include "VideoCommon/TextureCacheBase.h"
class AbstractTexture;
namespace VideoCommon
{
// Vulkan-only experimental temporal presentation. Owns resources until GPU-complete teardown.
class ModernGekkoDLSSPresenter
{
public:
  ModernGekkoDLSSPresenter();
  ~ModernGekkoDLSSPresenter();
  const AbstractTexture* Render(const RcTcacheEntry& entry,
      const MathUtil::Rectangle<int>& source_rect, u32 output_width, u32 output_height);
private:
  struct State;
  std::unique_ptr<State> m_state;
};
}
