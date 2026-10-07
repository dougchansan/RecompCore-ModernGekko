// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cmath>

namespace VideoCommon
{
constexpr float COLOSSEUM_AUTHORED_ASPECT = 4.0f / 3.0f;
constexpr float COLOSSEUM_WIDE_ASPECT = 16.0f / 9.0f;
constexpr float COLOSSEUM_AUTHORED_MENU_MINIMUM_SCALE =
    COLOSSEUM_WIDE_ASPECT / COLOSSEUM_AUTHORED_ASPECT;
constexpr float COLOSSEUM_NAMING_HORIZONTAL_SCALE = 602.0f / 640.0f;

inline float GetColosseumNamingHorizontalScale(bool naming_screen_active)
{
  return naming_screen_active ? COLOSSEUM_NAMING_HORIZONTAL_SCALE : 1.0f;
}

inline float GetColosseumPresentationAspect(float source_aspect,
                                             bool authored_menu_active)
{
  return authored_menu_active ? COLOSSEUM_AUTHORED_ASPECT : source_aspect;
}

inline float GetColosseumAuthoredMenuHorizontalScale(float raw_aspect,
                                                      bool authored_menu_active)
{
  if (!authored_menu_active || !std::isfinite(raw_aspect) || raw_aspect <= 0.0f)
    return 1.0f;

  // Colosseum's embedded Pokemon cameras cache the active field aspect when a
  // PC, party, or summary screen opens. Those models are then drawn into a
  // 4:3 viewport, so normalize only that menu-owned perspective projection.
  return raw_aspect / COLOSSEUM_AUTHORED_ASPECT;
}

inline bool IsColosseumSquareOffscreenViewport(float viewport_half_width,
                                                float viewport_half_height)
{
  if (!std::isfinite(viewport_half_width) || !std::isfinite(viewport_half_height))
    return false;

  const float width = std::abs(viewport_half_width);
  const float height = std::abs(viewport_half_height);
  return width >= 32.0f && std::abs(width - height) <= 0.5f;
}
}  // namespace VideoCommon
