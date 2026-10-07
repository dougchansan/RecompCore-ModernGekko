// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace ColosseumTextUpscale
{
// Keep most of the source coverage at the center of each texel. The narrower reconstruction
// radius rounds native pixel shoulders without spreading enough coverage to close the one-pixel
// counters used by lowercase e and other small Colosseum glyphs.
constexpr float COVERAGE_LOW = 0.40f;
constexpr float COVERAGE_HIGH = 0.68f;
constexpr float COLOR_COVERAGE_LOW = 0.10f;
constexpr float COLOR_COVERAGE_HIGH = 0.90f;
constexpr float RECONSTRUCTION_RADIUS = 0.55f;
constexpr float CENTER_WEIGHT = 0.68f;
constexpr float AXIAL_WEIGHT = 0.065f;
constexpr float DIAGONAL_WEIGHT = 0.015f;

// Battle status labels (Lv/HP/EXP) and their neighboring HUD sprites share this RGB5A3 atlas.
// Keying the color-preserving path by both dimensions and hash keeps the text option from
// sharpening unrelated scene textures that happen to use the same format.
constexpr std::uint64_t BATTLE_STATUS_ATLAS_HASH = 0x592cd77b475e4a80ULL;

enum class Filter
{
  None,
  Coverage,
  ColorCoverage,
};

constexpr float ReconstructCoverage(float coverage)
{
  if (coverage <= COVERAGE_LOW)
    return 0.0f;
  if (coverage >= COVERAGE_HIGH)
    return 1.0f;

  const float t = (coverage - COVERAGE_LOW) / (COVERAGE_HIGH - COVERAGE_LOW);
  return t * t * (3.0f - 2.0f * t);
}

constexpr float CombineCoverage(float center, float axial_sum, float diagonal_sum)
{
  return center * CENTER_WEIGHT + axial_sum * AXIAL_WEIGHT +
         diagonal_sum * DIAGONAL_WEIGHT;
}

struct Plan
{
  bool enabled = false;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  Filter filter = Filter::None;
};

constexpr Plan GetPlan(bool enabled, bool is_i4, bool is_rgb5a3, std::uint32_t width,
                       std::uint32_t height, std::uint32_t level_count,
                       std::uint64_t texture_hash)
{
  constexpr std::uint32_t scale = 6;
  if (!enabled || level_count != 1)
    return {};

  if (is_i4 && width == 512 && height == 512)
  {
    return {.enabled = true,
            .width = width * scale,
            .height = height * scale,
            .filter = Filter::Coverage};
  }

  if (is_rgb5a3 && width == 256 && height == 239 &&
      texture_hash == BATTLE_STATUS_ATLAS_HASH)
  {
    return {.enabled = true,
            .width = width * scale,
            .height = height * scale,
            .filter = Filter::ColorCoverage};
  }

  return {};
}
}  // namespace ColosseumTextUpscale
