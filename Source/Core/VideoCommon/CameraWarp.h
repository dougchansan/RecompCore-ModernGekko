// Copyright 2026 ModernGekko contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// 240 Hz camera for frame interpolation (StereoMode::FrameInterp).
//
// In-between frames blend each object, camera included, between the last two
// game frames, so turning the view shows up to a game frame late. For the
// camera mod's first-person and free cameras the view only rotates about the
// eye, so each EFB layer can instead be rotated to where the latest look
// input puts the camera when that layer is shown: a per-layer clip-space
// matrix (post-projection * projection * rotation * their inverses) applied by
// the geometry shader to the main 3D pass.
//
// The mod stamps every game frame's camera into the FIFO as an XF load to
// kStampAddress (outside XF memory), so the GPU thread receives it in order
// with that frame's draws; the host publishes the look stick at ~1 kHz.

#pragma once

#include <array>

#include "Common/CommonTypes.h"

namespace CameraWarp
{
// XF address of the camera stamp: [version<<16 | flags<<8 | mode, yaw, pitch,
// speed (rad/s at full deflection), time ns high, time ns low] as BE words.
constexpr u16 kStampAddress = 0x7E00;
constexpr u32 kStampVersion = 0xCA01;
// Version 2 adds the eye position (words 6-8) and the orbit radius (word 9).
constexpr u32 kStampVersion2 = 0xCA02;
constexpr u32 kModeFirstPerson = 1;
constexpr u32 kModeFree = 2;
constexpr u32 kModeOrbit = 3;  // third person: the eye circles a target
constexpr u32 kFlagInvertY = 1;

// GPU thread, in FIFO order (from LoadXFReg).
void Stamp(u32 words, const u8* be_data);
// GPU thread, at each XFB copy (end of a game frame).
void OnFrame();

// Any thread: the look stick, -1..1 after the mod's dead zone (0 when the
// game is paused or the stick is idle).
void SetLookInput(float x, float y);

// Off with MODERNGEKKO_CAMERA_WARP=0, or from the menu.
void SetEnabled(bool enabled);
bool IsEnabled();

// GPU thread: per-layer clip-space matrices (row-major, 4 rows per layer) for
// the current draw, or false for identity (no camera stamp, another camera
// mode, or not the main perspective pass).
bool GetLayerMatrices(u32 layers, const std::array<std::array<float, 4>, 4>& projection,
                      const std::array<float, 4>& pixel_center, bool clip_control,
                      std::array<std::array<float, 4>, 16>* out);

// GPU thread: the camera rotation from the previous game frame to the current
// one in view space (3 rows), applied to each vertex's previous-frame view
// position before the layers are blended. Frame interpolation then blends only
// object motion; the camera turn is applied exactly by GetLayerMatrices
// instead of as a straight-line blend of screen positions, which bent the
// scene (most at the screen edges) during fast turns. False = no stamps.
bool GetViewDelta(std::array<std::array<float, 4>, 3>* out);

// Forget stamps (state loads, video restarts).
void Reset();

// The latest camera stamp, for the developer info overlay.
struct DebugInfo
{
  bool active = false;  // stamps arriving (the camera mod owns the camera)
  u32 mode = 0;         // kModeFirstPerson / kModeFree / kModeOrbit
  float yaw = 0, pitch = 0, speed = 0;
  std::array<double, 3> eye{};
  double correction_degrees = 0;  // last prediction correction being faded out
};
DebugInfo GetDebugInfo();
}  // namespace CameraWarp
