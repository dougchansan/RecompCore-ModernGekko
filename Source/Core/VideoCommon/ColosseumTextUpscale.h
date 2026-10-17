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

// Glyph surfaces are rebuilt from a Gaussian-weighted sum of the 4x4 native texels around each
// point, cut at EDGE_LEVEL. The game's font is already antialiased at native size, so this
// smooth field has straight diagonals and round curves where the centre-weighted pass above kept
// the texel steps; cutting a little above half keeps the small font's counters (e, g, %) open.
constexpr float SMOOTH_SIGMA = 0.6f;  // native texels
constexpr float EDGE_LEVEL = 0.6f;
// Half the antialiased ramp in field units; the field changes about 0.5 per native texel at an
// edge, so this spreads the edge over about one target pixel at the 6x surface scale.
constexpr float EDGE_HALF_WIDTH = 0.08f;
constexpr int SURFACE_SCALE = 6;

constexpr float SmoothEdge(float field, float half_width)
{
  const float t = (field - (EDGE_LEVEL - half_width)) / (2.0f * half_width);
  const float c = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
  return c * c * (3.0f - 2.0f * c);
}

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
  constexpr std::uint32_t scale = SURFACE_SCALE;
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
