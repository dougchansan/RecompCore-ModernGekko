// Copyright 2026 ModernGekko contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/CameraWarp.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "Common/Swap.h"
#include "VideoCommon/Present.h"

namespace CameraWarp
{
namespace
{
using Mat4 = std::array<double, 16>;  // row-major

struct CameraStamp
{
  u32 mode = 0;
  u32 flags = 0;
  float yaw = 0;
  float pitch = 0;
  float speed = 0;
  TimePoint time{};
  std::array<double, 3> eye{};  // version 2; zero for version 1 (rotation only)
  float radius = 0;             // third-person orbit: distance to its target
};

CameraStamp s_prev, s_cur;
u32 s_stamps = 0;  // valid stamps in a row (2 = prev and cur usable)
bool s_stamped_this_frame = false;
u32 s_unstamped_frames = 0;

std::atomic<float> s_look_x{0}, s_look_y{0};
std::atomic<bool> s_enabled{[] {
  const char* env = std::getenv("MODERNGEKKO_CAMERA_WARP");
  return !(env && env[0] == '0');
}()};

// MODERNGEKKO_CAMERA_WARP_TRACE=1: about once a second; =2: every stamp.
const int s_trace = [] {
  const char* env = std::getenv("MODERNGEKKO_CAMERA_WARP_TRACE");
  return env ? std::atoi(env) : 0;
}();

// MODERNGEKKO_CAMERA_WARP_TEST_YAW=<radians>: verification aid, adds a fixed
// yaw to every corrected layer (compare against the game turning that far).
const double s_test_yaw = [] {
  const char* env = std::getenv("MODERNGEKKO_CAMERA_WARP_TEST_YAW");
  return env ? std::atof(env) : 0.0;
}();

constexpr double kPi = 3.14159265358979323846;
constexpr float kPitchLimit = 1.45f;  // the mod's first-person / free camera limit
// The mod's orbit pitch range (eye below the target is positive pitch).
constexpr float kOrbitPitchMin = -1.35f;
constexpr float kOrbitPitchMax = 0.25f;

Mat4 Multiply(const Mat4& a, const Mat4& b)
{
  Mat4 r{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
    {
      double sum = 0;
      for (int k = 0; k < 4; ++k)
        sum += a[i * 4 + k] * b[k * 4 + j];
      r[i * 4 + j] = sum;
    }
  return r;
}

bool Invert(const Mat4& m, Mat4* out)
{
  // Gauss-Jordan with partial pivoting.
  double a[4][8];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      a[i][j] = j < 4 ? m[i * 4 + j] : (j - 4 == i ? 1.0 : 0.0);
  for (int c = 0; c < 4; ++c)
  {
    int pivot = c;
    for (int r = c + 1; r < 4; ++r)
      if (std::abs(a[r][c]) > std::abs(a[pivot][c]))
        pivot = r;
    if (std::abs(a[pivot][c]) < 1e-12)
      return false;
    if (pivot != c)
      for (int j = 0; j < 8; ++j)
        std::swap(a[c][j], a[pivot][j]);
    const double inv = 1.0 / a[c][c];
    for (int j = 0; j < 8; ++j)
      a[c][j] *= inv;
    for (int r = 0; r < 4; ++r)
      if (r != c)
      {
        const double f = a[r][c];
        for (int j = 0; j < 8; ++j)
          a[r][j] -= f * a[c][j];
      }
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      (*out)[i * 4 + j] = a[i][j + 4];
  return true;
}

// World-to-view rotation of the mod's camera: C_MTXLookAt(eye, eye + forward,
// up = +Y) with forward (sin yaw cos pitch, sin pitch, cos yaw cos pitch).
// Rows: right, up, and look = eye - target (the view looks down -Z).
Mat4 ViewRotation(double yaw, double pitch)
{
  const double fx = std::sin(yaw) * std::cos(pitch), fy = std::sin(pitch),
               fz = std::cos(yaw) * std::cos(pitch);
  const double lx = -fx, ly = -fy, lz = -fz;
  // right = normalize(up x look), up = (0, 1, 0)
  double rx = lz, ry = 0, rz = -lx;
  const double rlen = std::sqrt(rx * rx + rz * rz);
  rx /= rlen;
  rz /= rlen;
  // up' = look x right
  const double ux = ly * rz - lz * ry, uy = lz * rx - lx * rz, uz = lx * ry - ly * rx;
  return {rx, ry, rz, 0, ux, uy, uz, 0, lx, ly, lz, 0, 0, 0, 0, 1};
}

// View matrix (rotation + translation) of a camera at eye with yaw / pitch.
Mat4 ViewMatrix(double yaw, double pitch, const std::array<double, 3>& eye)
{
  Mat4 m = ViewRotation(yaw, pitch);
  for (int r = 0; r < 3; ++r)
    m[r * 4 + 3] = -(m[r * 4] * eye[0] + m[r * 4 + 1] * eye[1] + m[r * 4 + 2] * eye[2]);
  return m;
}

// Inverse of a rigid view matrix.
Mat4 InvertRigid(const Mat4& m)
{
  Mat4 r = m;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r[i * 4 + j] = m[j * 4 + i];
  for (int i = 0; i < 3; ++i)
    r[i * 4 + 3] = -(r[i * 4] * m[3] + r[i * 4 + 1] * m[7] + r[i * 4 + 2] * m[11]);
  return r;
}

std::array<double, 3> Forward(double yaw, double pitch)
{
  return {std::sin(yaw) * std::cos(pitch), std::sin(pitch), std::cos(yaw) * std::cos(pitch)};
}

// Third-person orbit: the eye circles the point it looks at (the player), so
// the camera turns with translation. First-person and free cameras turn about
// the eye, and their own movement stays blended between game frames.
bool Orbiting(const CameraStamp& s)
{
  return s.mode == kModeOrbit && s.radius > 0.001f;
}

std::array<double, 3> OrbitCenter(const CameraStamp& s)
{
  const auto f = Forward(s.yaw, s.pitch);
  return {s.eye[0] + f[0] * s.radius, s.eye[1] + f[1] * s.radius, s.eye[2] + f[2] * s.radius};
}

Mat4 Transpose3(const Mat4& m)
{
  Mat4 r = m;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r[i * 4 + j] = m[j * 4 + i];
  return r;
}

double Seconds(DT d)
{
  return std::chrono::duration<double>(d).count();
}
}  // namespace

void Stamp(u32 words, const u8* be_data)
{
  if (s_trace >= 3)
  {
    static u32 raw = 0;
    if (raw++ < 400)
      std::fprintf(stderr, "[camwarp] raw words=%u w0=%08x stamped=%d\n", words,
                   words ? Common::swap32(be_data) : 0, s_stamped_this_frame ? 1 : 0);
  }
  if (words < 6)
    return;
  u32 w[10] = {};
  for (u32 i = 0; i < std::min<u32>(words, 10); ++i)
    w[i] = Common::swap32(be_data + i * 4);
  const u32 version = w[0] >> 16;
  if (version != kStampVersion && !(version == kStampVersion2 && words >= 10))
    return;
  CameraStamp stamp;
  stamp.mode = w[0] & 0xff;
  stamp.flags = (w[0] >> 8) & 0xff;
  std::memcpy(&stamp.yaw, &w[1], 4);
  std::memcpy(&stamp.pitch, &w[2], 4);
  std::memcpy(&stamp.speed, &w[3], 4);
  if (version == kStampVersion2)
  {
    for (int i = 0; i < 3; ++i)
    {
      float value;
      std::memcpy(&value, &w[6 + i], 4);
      stamp.eye[i] = value;
    }
    std::memcpy(&stamp.radius, &w[9], 4);
  }
  const u64 ns = (u64(w[4]) << 32) | w[5];
  stamp.time = TimePoint(std::chrono::duration_cast<DT>(std::chrono::nanoseconds(ns)));
  // The mod reads the same monotonic clock; if not, fall back to arrival time.
  const TimePoint now = Clock::now();
  if (stamp.time > now + std::chrono::milliseconds(5) ||
      stamp.time < now - std::chrono::milliseconds(250))
    stamp.time = now;
  if (!std::isfinite(stamp.yaw) || !std::isfinite(stamp.pitch) || !std::isfinite(stamp.speed))
    return;

  // Several stamps in one frame (render passes reload the camera) describe the
  // same frame: replace it rather than treating it as motion.
  if (s_stamped_this_frame)
  {
    if (stamp.mode != s_cur.mode)
      s_stamps = 0;
    s_cur = stamp;
    return;
  }
  s_stamped_this_frame = true;
  if (s_trace)
  {
    static u32 count = 0;
    if (count++ % 60 == 0 || s_trace >= 2)
    {
      std::fprintf(stderr, "[camwarp] stamp mode=%u yaw=%.3f pitch=%.3f speed=%.2f age=%.1fms look=%.2f,%.2f\n",
                   stamp.mode, stamp.yaw, stamp.pitch, stamp.speed,
                   Seconds(Clock::now() - stamp.time) * 1000.0,
                   s_look_x.load(std::memory_order_relaxed), s_look_y.load(std::memory_order_relaxed));
    }
  }
  const bool continues = s_stamps > 0 && s_cur.mode == stamp.mode && stamp.mode != 0;
  s_prev = s_cur;
  s_cur = stamp;
  s_stamps = continues ? 2 : (stamp.mode != 0 ? 1 : 0);
}

void OnFrame()
{
  if (s_trace >= 3)
  {
    static u32 frames = 0;
    if (frames++ < 400)
      std::fprintf(stderr, "[camwarp] frame stamped=%d stamps=%u\n", s_stamped_this_frame ? 1 : 0, s_stamps);
  }
  // A frame without a stamp kept its camera (no new controller sample): the
  // next frame interpolates from the same camera.
  if (!s_stamped_this_frame && s_stamps > 0)
  {
    s_prev = s_cur;
    // The mod stamps every frame it owns the camera: a few in a row missing
    // means something else took over (a cutscene, a menu).
    if (++s_unstamped_frames > 2)
      s_stamps = 0;
  }
  else if (s_stamped_this_frame)
  {
    s_unstamped_frames = 0;
  }
  s_stamped_this_frame = false;
}

void SetLookInput(float x, float y)
{
  s_look_x.store(x, std::memory_order_relaxed);
  s_look_y.store(y, std::memory_order_relaxed);
}

void SetEnabled(bool enabled)
{
  s_enabled.store(enabled, std::memory_order_relaxed);
}

bool IsEnabled()
{
  return s_enabled.load(std::memory_order_relaxed);
}

void Reset()
{
  s_stamps = 0;
  s_stamped_this_frame = false;
  s_unstamped_frames = 0;
  s_prev = s_cur = {};
}

namespace
{
bool Active(TimePoint now)
{
  // Stamps older than ~15 frames mean the camera mod stopped stamping (menu,
  // cutscene, another camera): leave the view alone.
  return IsEnabled() && s_stamps >= 2 && now - s_cur.time <= std::chrono::milliseconds(250);
}
}  // namespace

bool GetViewDelta(std::array<std::array<float, 4>, 3>* out)
{
  if (!Active(Clock::now()))
    return false;
  const Mat4 delta =
      Orbiting(s_cur) && Orbiting(s_prev) ?
          Multiply(ViewMatrix(s_cur.yaw, s_cur.pitch, s_cur.eye),
                   InvertRigid(ViewMatrix(s_prev.yaw, s_prev.pitch, s_prev.eye))) :
          Multiply(ViewRotation(s_cur.yaw, s_cur.pitch),
                   Transpose3(ViewRotation(s_prev.yaw, s_prev.pitch)));
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c)
      (*out)[r][c] = static_cast<float>(delta[r * 4 + c]);
  return true;
}

bool GetLayerMatrices(u32 layers, const std::array<std::array<float, 4>, 4>& projection,
                      const std::array<float, 4>& pixel_center, bool clip_control,
                      std::array<std::array<float, 4>, 16>* out)
{
  const TimePoint now = Clock::now();
  if (layers < 2 || layers > 4 || !Active(now))
    return false;

  Mat4 p, a;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      p[i * 4 + j] = projection[i][j];
  // The vertex shader's post-projection steps (depth range remap, viewport
  // sign, pixel-center offset) as a matrix A: clip = A * P * view.
  const double sx = pixel_center[0] < 0 ? -1.0 : 1.0;
  const double sy = -pixel_center[1] < 0 ? -1.0 : 1.0;
  const double pz = pixel_center[2], pw = pixel_center[3];
  a = {sx, 0, 0, -pixel_center[0], 0, sy, 0, -pixel_center[1], 0, 0, -pz, pw, 0, 0, 0, 1};
  if (!clip_control)
    a = {sx, 0, 0, -pixel_center[0], 0, sy, 0, -pixel_center[1], 0, 0, -2 * pz, 2 * pw - 1,
         0, 0, 0, 1};
  const Mat4 ap = Multiply(a, p);
  Mat4 ap_inv;
  if (!Invert(ap, &ap_inv))
    return false;

  // When each layer is shown: in-between layers 1..n-1 from the next frame's
  // first slot, one step apart, then the real frame (layer 0).
  const double step = 1.0 / 60.0 / layers;
  TimePoint first = g_presenter ? g_presenter->InterpNextFrameSlot() : TimePoint{};
  if (first == TimePoint{} || first < now)
    first = now;

  const double look_x = s_look_x.load(std::memory_order_relaxed);
  const double look_y = s_look_y.load(std::memory_order_relaxed);
  const double invert = (s_cur.flags & kFlagInvertY) ? -1.0 : 1.0;
  const bool orbit = Orbiting(s_cur) && Orbiting(s_prev);
  const double pitch_min = orbit ? std::min<double>(kOrbitPitchMin, s_cur.pitch) : -kPitchLimit;
  const double pitch_max = orbit ? std::max<double>(kOrbitPitchMax, s_cur.pitch) : kPitchLimit;
  // The orbit center (the player) keeps its last velocity while extrapolated.
  std::array<double, 3> center{}, velocity{};
  Mat4 content_inv{};
  if (orbit)
  {
    center = OrbitCenter(s_cur);
    const auto before = OrbitCenter(s_prev);
    const double dt = std::clamp(Seconds(s_cur.time - s_prev.time), 1.0 / 120.0, 0.1);
    double distance = 0;
    for (int i = 0; i < 3; ++i)
    {
      velocity[i] = (center[i] - before[i]) / dt;
      distance += (center[i] - before[i]) * (center[i] - before[i]);
    }
    if (distance > 100.0 * 100.0)  // a warp, not walking
      velocity = {};
    content_inv = InvertRigid(ViewMatrix(s_cur.yaw, s_cur.pitch, s_cur.eye));
  }

  for (u32 k = 0; k < layers; ++k)
  {
    const u32 order = k == 0 ? layers - 1 : k - 1;
    const double shown = Seconds(first - s_cur.time) + order * step;
    const double ahead = std::clamp(shown, 0.0, 0.1);
    // Every layer's content is in the current frame's camera: the vertex
    // shader rotated the previous-frame positions into it (GetViewDelta).
    const double content_yaw = s_cur.yaw;
    const double content_pitch = s_cur.pitch;
    // Target: the current frame's camera advanced by the stick as the mod would.
    const double target_yaw = s_cur.yaw - look_x * s_cur.speed * ahead + s_test_yaw;
    const double target_pitch =
        std::clamp(s_cur.pitch + look_y * s_cur.speed * ahead * invert, pitch_min, pitch_max);
    Mat4 delta;
    if (orbit)
    {
      const auto f = Forward(target_yaw, target_pitch);
      std::array<double, 3> eye;
      for (int i = 0; i < 3; ++i)
        eye[i] = center[i] + velocity[i] * ahead - f[i] * s_cur.radius;
      delta = Multiply(ViewMatrix(target_yaw, target_pitch, eye), content_inv);
    }
    else
    {
      delta = Multiply(ViewRotation(target_yaw, target_pitch),
                       Transpose3(ViewRotation(content_yaw, content_pitch)));
    }
    if (s_trace && k == 1)
    {
      static TimePoint last{};
      if (now - last > std::chrono::seconds(1))
      {
        last = now;
        std::fprintf(stderr, "[camwarp] layer1 shown in %.1fms: content yaw %.4f -> target %.4f (%.2f deg)\n",
                     shown * 1000.0, content_yaw, target_yaw,
                     std::remainder(target_yaw - content_yaw, 2 * kPi) * 180.0 / kPi);
      }
    }
    const Mat4 m = Multiply(Multiply(ap, delta), ap_inv);
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        (*out)[k * 4 + r][c] = static_cast<float>(m[r * 4 + c]);
  }
  return true;
}
}  // namespace CameraWarp
