// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
// Independent CPU correspondence oracle for GPU fixture expectations. These tests establish
// signed/fractional/ambiguous correspondence cases; they do not execute or validate GPU shaders.
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <gtest/gtest.h>

namespace
{
using Image = std::function<double(double, double)>;
struct Correspondence
{
  double x = 0, y = 0, error = 0, uniqueness = 0, variance = 0;
};
Image Texture()
{
  return [](double x, double y) {
    return 0.5 + 0.15 * std::sin(x * 0.73 + y * 0.11) +
           0.13 * std::cos(y * 0.61 - x * 0.07) + 0.12 * std::sin(x * 0.29 + y * 0.37);
  };
}
double Error(const Image& current, const Image& previous, double x, double y,
             double dx, double dy)
{
  double error = 0;
  for (int j = -2; j <= 2; ++j)
    for (int i = -2; i <= 2; ++i)
    {
      const double difference = current(x + i, y + j) - previous(x + i + dx, y + j + dy);
      error += difference * difference / 25;
    }
  return error;
}
Correspondence Reference(const Image& current, const Image& previous, double x, double y)
{
  Correspondence result;
  result.error = std::numeric_limits<double>::infinity();
  for (int j = -16; j <= 16; ++j)
    for (int i = -16; i <= 16; ++i)
    {
      const double dx = i * 0.25, dy = j * 0.25;
      const double error = Error(current, previous, x, y, dx, dy);
      if (error < result.error)
      {
        result.x = dx;
        result.y = dy;
        result.error = error;
      }
    }
  double alternative = std::numeric_limits<double>::infinity(), mean = 0, square = 0;
  for (int j = -2; j <= 2; ++j)
    for (int i = -2; i <= 2; ++i)
    {
      if (i * i + j * j >= 4)
        alternative = std::min(alternative, Error(current, previous, x, y,
                                                  result.x + i, result.y + j));
      const double value = current(x + i, y + j);
      mean += value / 25;
      square += value * value / 25;
    }
  result.uniqueness = std::clamp((alternative - result.error) / std::max(alternative, 0.0001),
                                0.0, 1.0);
  result.variance = std::max(0.0, square - mean * mean);
  return result;
}

TEST(ModernGekkoOpticalFlowReference, CurrentToPreviousSignedTranslation)
{
  const Image previous = Texture();
  for (const double dx : {-3.0, 2.0})
    for (const double dy : {-2.0, 3.0})
    {
      const Image current = [=](double x, double y) { return previous(x + dx, y + dy); };
      const auto result = Reference(current, previous, 27, 31);
      EXPECT_NEAR(result.x, dx, 0.01);
      EXPECT_NEAR(result.y, dy, 0.01);
      EXPECT_LT(result.error, 1e-10);
      EXPECT_GT(result.uniqueness, 0.9);
    }
}
TEST(ModernGekkoOpticalFlowReference, FractionalTranslation)
{
  const Image previous = Texture();
  const Image current = [=](double x, double y) { return previous(x + 1.25, y - 0.75); };
  const auto result = Reference(current, previous, 27, 31);
  EXPECT_NEAR(result.x, 1.25, 0.01);
  EXPECT_NEAR(result.y, -0.75, 0.01);
}
TEST(ModernGekkoOpticalFlowReference, FlatAndRepeatedPatternsAreAmbiguous)
{
  const Image flat = [](double, double) { return 0.5; };
  auto result = Reference(flat, flat, 27, 31);
  EXPECT_EQ(result.variance, 0.0);
  EXPECT_EQ(result.uniqueness, 0.0);
  const Image repeated = [](double x, double) { return 0.5 + 0.2 * std::cos(x * 3.141592653589793); };
  result = Reference(repeated, repeated, 27, 31);
  EXPECT_LT(result.uniqueness, 0.001);
}
TEST(ModernGekkoOpticalFlowReference, NewlyOccludedRegionRejectsPhotometricMatch)
{
  const Image previous = Texture();
  const Image current = [](double, double) { return 0.0; };
  const auto result = Reference(current, previous, 27, 31);
  EXPECT_GT(result.error, 0.012);
  EXPECT_EQ(result.variance, 0.0);
}
}  // namespace
