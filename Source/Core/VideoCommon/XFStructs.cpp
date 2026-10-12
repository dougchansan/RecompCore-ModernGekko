// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/XFStructs.h"

#include <bitset>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <string>
#include <fmt/format.h>

#include "Common/CommonTypes.h"
#include "Common/Logging/Log.h"
#include "Common/Swap.h"

#include "Core/DolphinAnalytics.h"
#include "Core/HW/Memmap.h"
#include "Core/System.h"

#include "VideoCommon/CPMemory.h"
#include "VideoCommon/CameraWarp.h"
#include "VideoCommon/Fifo.h"
#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/GeometryShaderManager.h"
#include "VideoCommon/PixelShaderManager.h"
#include "VideoCommon/VertexLoaderManager.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/XFMemory.h"
#include "VideoCommon/XFStateManager.h"
#include "VideoCommon/VideoConfig.h"

static void XFMemWritten(XFStateManager& xf_state_manager, u32 transferSize, u32 baseAddress)
{
  g_vertex_manager->Flush();
  xf_state_manager.InvalidateXFRange(baseAddress, baseAddress + transferSize);
}

static void XFRegWritten(Core::System& system, XFStateManager& xf_state_manager, u32 address,
                         u32 value)
{
  if (address >= XFMEM_REGISTERS_START && address < XFMEM_REGISTERS_END)
  {
    switch (address)
    {
    case XFMEM_ERROR:
    case XFMEM_DIAG:
    case XFMEM_STATE0:  // internal state 0
    case XFMEM_STATE1:  // internal state 1
    case XFMEM_CLOCK:
    case XFMEM_SETGPMETRIC:
      // Not implemented
      break;

    case XFMEM_CLIPDISABLE:
    {
      ClipDisable setting{.hex = value};
      if (setting.disable_clipping_detection)
        DolphinAnalytics::Instance().ReportGameQuirk(GameQuirk::SetsXFClipDisableBit0);
      if (setting.disable_trivial_rejection)
        DolphinAnalytics::Instance().ReportGameQuirk(GameQuirk::SetsXFClipDisableBit1);
      if (setting.disable_cpoly_clipping_acceleration)
        DolphinAnalytics::Instance().ReportGameQuirk(GameQuirk::SetsXFClipDisableBit2);
      break;
    }

    case XFMEM_VTXSPECS:  //__GXXfVtxSpecs, wrote 0004
      VertexLoaderManager::g_needs_cp_xf_consistency_check = true;
      break;

    case XFMEM_SETNUMCHAN:
      if (xfmem.numChan.numColorChans != (value & 3))
        g_vertex_manager->Flush();
      xf_state_manager.SetLightingConfigChanged();
      break;

    case XFMEM_SETCHAN0_AMBCOLOR:  // Channel Ambient Color
    case XFMEM_SETCHAN1_AMBCOLOR:
    {
      u8 chan = address - XFMEM_SETCHAN0_AMBCOLOR;
      if (xfmem.ambColor[chan] != value)
      {
        g_vertex_manager->Flush();
        xf_state_manager.SetMaterialColorChanged(chan);
      }
      break;
    }

    case XFMEM_SETCHAN0_MATCOLOR:  // Channel Material Color
    case XFMEM_SETCHAN1_MATCOLOR:
    {
      u8 chan = address - XFMEM_SETCHAN0_MATCOLOR;
      if (xfmem.matColor[chan] != value)
      {
        g_vertex_manager->Flush();
        xf_state_manager.SetMaterialColorChanged(chan + 2);
      }
      break;
    }

    case XFMEM_SETCHAN0_COLOR:  // Channel Color
    case XFMEM_SETCHAN1_COLOR:
    case XFMEM_SETCHAN0_ALPHA:  // Channel Alpha
    case XFMEM_SETCHAN1_ALPHA:
      if (((u32*)&xfmem)[address] != (value & 0x7fff))
        g_vertex_manager->Flush();
      xf_state_manager.SetLightingConfigChanged();
      break;

    case XFMEM_DUALTEX:
      if (xfmem.dualTexTrans.enabled != bool(value & 1))
        g_vertex_manager->Flush();
      xf_state_manager.SetTexMatrixInfoChanged(-1);
      break;

    case XFMEM_SETMATRIXINDA:
      xf_state_manager.SetTexMatrixChangedA(value);
      VertexLoaderManager::g_needs_cp_xf_consistency_check = true;
      break;
    case XFMEM_SETMATRIXINDB:
      xf_state_manager.SetTexMatrixChangedB(value);
      VertexLoaderManager::g_needs_cp_xf_consistency_check = true;
      break;

    case XFMEM_SETVIEWPORT:
    case XFMEM_SETVIEWPORT + 1:
    case XFMEM_SETVIEWPORT + 2:
    case XFMEM_SETVIEWPORT + 3:
    case XFMEM_SETVIEWPORT + 4:
    case XFMEM_SETVIEWPORT + 5:
      g_vertex_manager->Flush();
      xf_state_manager.SetViewportChanged();
      system.GetPixelShaderManager().SetViewportChanged();
      system.GetGeometryShaderManager().SetViewportChanged();
      break;

    case XFMEM_SETPROJECTION:
    case XFMEM_SETPROJECTION + 1:
    case XFMEM_SETPROJECTION + 2:
    case XFMEM_SETPROJECTION + 3:
    case XFMEM_SETPROJECTION + 4:
    case XFMEM_SETPROJECTION + 5:
    case XFMEM_SETPROJECTION + 6:
      g_vertex_manager->Flush();
      // First-person depth of field: the world's depth is only intact until the
      // game switches from the main perspective camera to its orthographic
      // overlay/composite pass (which overwrites depth). Blur at that moment,
      // once per frame; the full-size viewport excludes the 384x384 shadow-map
      // passes. (Handlers run before xfmem is updated: type is still the old one.)
      if (address == XFMEM_SETPROJECTION + 6 &&
          xfmem.projection.type == ProjectionType::Perspective &&
          static_cast<ProjectionType>(value) == ProjectionType::Orthographic &&
          std::abs(xfmem.viewport.wd) * 2.0f >= 560.0f)
      {
        // Ambient occlusion first (it darkens creases in the sharp image), then
        // depth of field. The pause menu sets the level; MODERNGEKKO_SSAO=1..3
        // overrides it (tests).
        static const int ssao_override = [] {
          const char* v = std::getenv("MODERNGEKKO_SSAO");
          return v ? std::clamp(std::atoi(v), 0, 3) : -1;
        }();
        const int ssao = ssao_override >= 0 ? ssao_override :
                                              g_ambient_occlusion.load(std::memory_order_relaxed);
        if (ssao > 0)
          g_framebuffer_manager->ApplyAmbientOcclusionOnce(ssao);
        // Cinematic look after occlusion (test hook MODERNGEKKO_CINE=1 vivid, 2 cinematic).
        static const int cine = [] {
          const char* v = std::getenv("MODERNGEKKO_CINE");
          return v ? std::clamp(std::atoi(v), 0, 2) : 0;
        }();
        if (cine > 0)
          g_framebuffer_manager->ApplyCinematicOnce(cine);
        if (const int dof = g_depth_of_field_level.load(std::memory_order_relaxed); dof > 0)
          g_framebuffer_manager->ApplyDepthOfFieldOnce(dof);
      }
      xf_state_manager.SetProjectionChanged();
      system.GetGeometryShaderManager().SetProjectionChanged();
      break;

    case XFMEM_SETNUMTEXGENS:  // GXSetNumTexGens
      if (xfmem.numTexGen.numTexGens != (value & 15))
        g_vertex_manager->Flush();
      break;

    case XFMEM_SETTEXMTXINFO:
    case XFMEM_SETTEXMTXINFO + 1:
    case XFMEM_SETTEXMTXINFO + 2:
    case XFMEM_SETTEXMTXINFO + 3:
    case XFMEM_SETTEXMTXINFO + 4:
    case XFMEM_SETTEXMTXINFO + 5:
    case XFMEM_SETTEXMTXINFO + 6:
    case XFMEM_SETTEXMTXINFO + 7:
      g_vertex_manager->Flush();
      xf_state_manager.SetTexMatrixInfoChanged(address - XFMEM_SETTEXMTXINFO);
      break;

    case XFMEM_SETPOSTMTXINFO:
    case XFMEM_SETPOSTMTXINFO + 1:
    case XFMEM_SETPOSTMTXINFO + 2:
    case XFMEM_SETPOSTMTXINFO + 3:
    case XFMEM_SETPOSTMTXINFO + 4:
    case XFMEM_SETPOSTMTXINFO + 5:
    case XFMEM_SETPOSTMTXINFO + 6:
    case XFMEM_SETPOSTMTXINFO + 7:
      g_vertex_manager->Flush();
      xf_state_manager.SetTexMatrixInfoChanged(address - XFMEM_SETPOSTMTXINFO);
      break;

    // --------------
    // Unknown Regs
    // --------------

    // Maybe these are for Normals?
    case 0x1048:  // xfmem.texcoords[0].nrmmtxinfo.hex = data; break; ??
    case 0x1049:
    case 0x104a:
    case 0x104b:
    case 0x104c:
    case 0x104d:
    case 0x104e:
    case 0x104f:
      DolphinAnalytics::Instance().ReportGameQuirk(GameQuirk::UsesUnknownXFCommand);
      DEBUG_LOG_FMT(VIDEO, "Possible Normal Mtx XF reg?: {:x}={:x}", address, value);
      break;

    case 0x1013:
    case 0x1014:
    case 0x1015:
    case 0x1016:
    case 0x1017:

    default:
      DolphinAnalytics::Instance().ReportGameQuirk(GameQuirk::UsesUnknownXFCommand);
      WARN_LOG_FMT(VIDEO, "Unknown XF Reg: {:x}={:x}", address, value);
      break;
    }
  }
}

// Frame-interpolation research trace. MODERNGEKKO_XF_TRACE=<file> writes one
// line per XFB copy listing, in order, every load into XF matrix memory
// (< 0x0600) plus the viewport (0x101A) and projection (0x1020) registers as
// "kind addr:size:fnv" tokens, so consecutive frames can be compared for a
// stable draw/matrix order. Off (one getenv) unless the variable is set.
namespace XFTrace
{
static FILE* s_file;
static bool s_init;
static int s_frames;
static std::string s_line;

static bool On()
{
  if (!s_init)
  {
    s_init = true;
    if (const char* path = std::getenv("MODERNGEKKO_XF_TRACE"))
      s_file = std::fopen(path, "w");
  }
  return s_file && s_frames < 3000;
}

void Load(char kind, u32 address, u32 words, const u8* be_data)
{
  if (!On() || (address >= 0x0600 && address != 0x101A && address != 0x1020))
    return;
  u32 hash = 2166136261u;
  for (u32 i = 0; i < words * 4; ++i)
    hash = (hash ^ be_data[i]) * 16777619u;
  s_line += fmt::format(" {}{:x}:{}:{:08x}", kind, address, words, hash);
}

void Frame()
{
  if (!On())
    return;
  std::fprintf(s_file, "%d%s\n", s_frames++, s_line.c_str());
  std::fflush(s_file);
  s_line.clear();
}
}  // namespace XFTrace

// Frame interpolation (StereoMode::FrameInterp). Every XF load that touches
// the position matrices (0x000-0x0FF) or the projection (0x1020-0x1025) is
// recorded in order. A load is paired with the load at the same position in
// the previous frame's sequence when the address and size match (consecutive
// frames issue the same sequence ~99% of the time); the shadow copies below
// then hold its previous-frame words, giving the vertex shader's pos_b, which the
// geometry shader blends toward pos for the in-between layers. Once a frame
// diverges, the rest of that frame uses its real values (no motion).
// See LoadXFReg: the camera mod's world-drawing marker (outside XF memory).
constexpr u16 kWorldMarkerAddress = 0x7E10;

namespace FrameInterp
{
float g_pos[256];
float g_proj[6];
bool g_pos_dirty = true;

namespace
{
// Loads are paired with the previous frame's by key: XF address, size, and the
// position array of the draw that uses them (the mesh). Pairing by position in
// the frame's load order (with only address and size checked) went wrong as
// soon as one model was culled or appeared - every later matrix paired with a
// neighbour's (most loads are 12 words to slot 0), so in-between frames drew
// floor pieces from another object's place (black flicker crossing a
// boundary at 120 / 240 FPS).
struct LoadRec
{
  u16 address;
  u16 words;
  u32 data;  // index into the data pool
  u32 base;  // position array base of the first draw after the load
};
std::vector<LoadRec> s_cur, s_pending;
std::vector<u32> s_prev_data, s_cur_data;
std::unordered_map<u64, std::vector<u32>> s_prev_map;  // key -> data indices, in order
std::unordered_map<u64, u32> s_key_uses;               // occurrences used this frame
bool s_have_prev = false;
// Shadow words written by this frame's loads; the rest are refreshed from XF
// memory at the frame's end so an unchanged matrix blends to itself.
std::bitset<256> s_pos_loaded;
// Shadow words holding the previous frame's value (a paired load); the rest
// hold the current value, already in the current camera.
std::bitset<256> s_pos_previous;
bool s_proj_loaded = false;
u32 s_stat_frames, s_stat_loads, s_stat_moved, s_stat_static, s_stat_unpaired, s_stat_cut;

constexpr u32 kNoDraw = 0xFFFFFFFFu;

float AsFloat(u32 bits)
{
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// A whole 3x4 position matrix (view * model): the change from the previous
// frame's, D = M_now * M_prev^-1, is the camera's motion for every static
// object. A camera cut is a large turn or a jump that is large next to the
// object's distance; judging single words instead let a cut through for big
// meshes (the floor blended between two unrelated views and left the screen
// for an in-between frame - black floor when the field camera switched).
// Returns -1 if the matrix cannot be judged this way.
int IsMatrixCut(const u32* now_words, const u32* prev_words)
{
  double c[12], p[12];
  for (int i = 0; i < 12; ++i)
  {
    c[i] = AsFloat(now_words[i]);
    p[i] = AsFloat(prev_words[i]);
  }
  // Rows r: A[r][0..2] = m[r*4+0..2], t[r] = m[r*4+3].
  auto A = [](const double* m, int r, int k) { return m[r * 4 + k]; };
  const double det = A(p, 0, 0) * (A(p, 1, 1) * A(p, 2, 2) - A(p, 1, 2) * A(p, 2, 1)) -
                     A(p, 0, 1) * (A(p, 1, 0) * A(p, 2, 2) - A(p, 1, 2) * A(p, 2, 0)) +
                     A(p, 0, 2) * (A(p, 1, 0) * A(p, 2, 1) - A(p, 1, 1) * A(p, 2, 0));
  if (!std::isfinite(det) || std::fabs(det) < 1e-9)
    return -1;
  double inv[3][3];
  inv[0][0] = (A(p, 1, 1) * A(p, 2, 2) - A(p, 1, 2) * A(p, 2, 1)) / det;
  inv[0][1] = (A(p, 0, 2) * A(p, 2, 1) - A(p, 0, 1) * A(p, 2, 2)) / det;
  inv[0][2] = (A(p, 0, 1) * A(p, 1, 2) - A(p, 0, 2) * A(p, 1, 1)) / det;
  inv[1][0] = (A(p, 1, 2) * A(p, 2, 0) - A(p, 1, 0) * A(p, 2, 2)) / det;
  inv[1][1] = (A(p, 0, 0) * A(p, 2, 2) - A(p, 0, 2) * A(p, 2, 0)) / det;
  inv[1][2] = (A(p, 0, 2) * A(p, 1, 0) - A(p, 0, 0) * A(p, 1, 2)) / det;
  inv[2][0] = (A(p, 1, 0) * A(p, 2, 1) - A(p, 1, 1) * A(p, 2, 0)) / det;
  inv[2][1] = (A(p, 0, 1) * A(p, 2, 0) - A(p, 0, 0) * A(p, 2, 1)) / det;
  inv[2][2] = (A(p, 0, 0) * A(p, 1, 1) - A(p, 0, 1) * A(p, 1, 0)) / det;
  double d[3][3];
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 3; ++k)
      d[r][k] = A(c, r, 0) * inv[0][k] + A(c, r, 1) * inv[1][k] + A(c, r, 2) * inv[2][k];
  // Rotation angle of D with its uniform scale (animated scaling) removed.
  const double d_det = d[0][0] * (d[1][1] * d[2][2] - d[1][2] * d[2][1]) -
                       d[0][1] * (d[1][0] * d[2][2] - d[1][2] * d[2][0]) +
                       d[0][2] * (d[1][0] * d[2][1] - d[1][1] * d[2][0]);
  if (!std::isfinite(d_det) || d_det <= 0)
    return 1;  // mirrored or degenerate: not motion
  const double scale = std::cbrt(d_det);
  if (scale < 0.5 || scale > 2.0)
    return 1;
  const double cos_angle = ((d[0][0] + d[1][1] + d[2][2]) / scale - 1.0) * 0.5;
  constexpr double kMaxTurn = 0.6;  // rad per game frame (~34 deg; fast turns are ~0.1)
  if (cos_angle < std::cos(kMaxTurn))
    return 1;
  // Translation of D: t_now - D * t_prev, against the object's distance.
  double jump = 0, distance = 0;
  for (int r = 0; r < 3; ++r)
  {
    const double moved =
        A(c, r, 3) - (d[r][0] * A(p, 0, 3) + d[r][1] * A(p, 1, 3) + d[r][2] * A(p, 2, 3));
    jump += moved * moved;
    distance = std::max({distance, std::fabs(A(c, r, 3)), std::fabs(A(p, r, 3))});
  }
  return std::sqrt(jump) > 0.3 * distance + 2.0 ? 1 : 0;
}

u64 Key(u32 address, u32 words, u32 base)
{
  return (u64(base) << 32) | (u64(address & 0xFFFF) << 16) | u64(words & 0xFFFF);
}

// Pairs one recorded load and writes its previous-frame (or current) words to
// the shadow.
void ResolveLoad(LoadRec rec, bool* flushed)
{
  const u32 address = rec.address, words = rec.words;
  const bool touches_pos = address < 0x100;
  const u32* now_words = &s_cur_data[rec.data];

  const u32* prev = nullptr;
  // Projections are not blended: a projection load paired with another pass's
  // (a different kind of view) gave in-between frames an invalid projection -
  // the field floor vanished from them while walking (black flicker at 120 /
  // 240 FPS only). They change rarely; a zoom simply steps at the game rate.
  // MODERNGEKKO_INTERP_PROJ=1 blends them again (diagnostic).
  static const bool blend_projection = [] {
    const char* v = std::getenv("MODERNGEKKO_INTERP_PROJ");
    return v && v[0] == '1';
  }();
  if (s_have_prev && (touches_pos || blend_projection))
  {
    const u64 key = Key(address, words, rec.base);
    if (const auto it = s_prev_map.find(key); it != s_prev_map.end())
    {
      const u32 use = s_key_uses[key]++;
      if (use < it->second.size())
        prev = &s_prev_data[it->second[use]];
    }
  }

  // Camera cuts: a paired load whose values jump far between frames is a new
  // view, not motion - blending it would show a bogus halfway frame, so it is
  // taken as-is. Rotation/scale entries are judged against their own size;
  // position-matrix translations (every 4th word) against the object's
  // distance from the camera, since turning swings an object near the screen
  // centre from x ~ 0 to x ~ 50 in a frame, which is motion, not a cut.
  ++s_stat_loads;
  if (!prev)
    ++s_stat_unpaired;
  if (prev && touches_pos && words == 12 && (address & 3) == 0)
  {
    const int cut = IsMatrixCut(now_words, prev);
    if (cut == 1)
    {
      prev = nullptr;
      ++s_stat_cut;
    }
    else if (cut == 0)
    {
      ++(std::memcmp(now_words, prev, 12 * sizeof(u32)) != 0 ? s_stat_moved : s_stat_static);
    }
    if (cut >= 0)
      goto judged;
  }
  if (prev)
  {
    float distance = 0;
    if (touches_pos)
    {
      for (u32 i = 0; i < words; ++i)
      {
        if (((address + i) & 3) != 3)
          continue;
        distance = std::max({distance, std::fabs(AsFloat(now_words[i])), std::fabs(AsFloat(prev[i]))});
      }
    }
    bool moved = false;
    for (u32 i = 0; i < words; ++i)
    {
      const float now = AsFloat(now_words[i]);
      const float old_value = AsFloat(prev[i]);
      const bool translation = touches_pos && address + i < 0x100 && ((address + i) & 3) == 3;
      const float size = translation ? std::max({distance, std::fabs(now), std::fabs(old_value)}) :
                                       std::max(std::fabs(now), std::fabs(old_value));
      const float limit = std::max(1.0f, 0.5f * size);
      if (!(std::fabs(now - old_value) <= limit))
      {
        prev = nullptr;
        ++s_stat_cut;
        break;
      }
      moved |= now != old_value;
    }
    if (prev)
      ++(moved ? s_stat_moved : s_stat_static);
  }
judged:

  for (u32 i = 0; i < words; ++i)
  {
    const float value = AsFloat(prev ? prev[i] : now_words[i]);
    const u32 a = address + i;
    if (a < 0x100)
    {
      s_pos_loaded.set(a);
      if (s_pos_previous[a] != (prev != nullptr))
      {
        // The camera correction for the pending batch depends on it.
        if (!*flushed)
        {
          g_vertex_manager->Flush();
          *flushed = true;
        }
        s_pos_previous.set(a, prev != nullptr);
      }
    }
    else if (a >= 0x1020 && a <= 0x1025)
    {
      s_proj_loaded = true;
    }
    float* slot = a < 0x100 ? &g_pos[a] : (a >= 0x1020 && a <= 0x1025) ? &g_proj[a - 0x1020] : nullptr;
    if (!slot || std::memcmp(slot, &value, sizeof(value)) == 0)
      continue;
    // The batch the vertex manager is still accumulating was issued under
    // the old matrices; draw it before its layer-1 copies change.
    if (!*flushed)
    {
      g_vertex_manager->Flush();
      *flushed = true;
    }
    *slot = value;
    if (a < 0x100)
      g_pos_dirty = true;
  }
  s_cur.push_back(rec);
}

void ResolvePending(u32 base)
{
  if (s_pending.empty())
    return;
  bool flushed = false;
  for (LoadRec rec : s_pending)
  {
    rec.base = base;
    ResolveLoad(rec, &flushed);
  }
  s_pending.clear();
}
}  // namespace

void Reset();

bool Enabled()
{
  const bool enabled = g_ActiveConfig.stereo_mode == StereoMode::FrameInterp;
  // Switched on mid-session: the shadow is whatever it held before.
  static bool was_enabled = false;
  if (enabled && !was_enabled)
  {
    was_enabled = true;
    Reset();
  }
  was_enabled = enabled;
  return enabled;
}

void Load(u32 address, u32 words, const u8* be_data)
{
  if (!Enabled())
    return;
  const bool touches_pos = address < 0x100;
  const bool touches_proj = address <= 0x1025 && address + words > 0x1020;
  if (!touches_pos && !touches_proj)
    return;
  // Paired when the draw that uses it arrives (BeforeDraw), which names the mesh.
  const u32 data_index = static_cast<u32>(s_cur_data.size());
  for (u32 i = 0; i < words; ++i)
    s_cur_data.push_back(Common::swap32(be_data + 4 * i));
  s_pending.push_back({static_cast<u16>(address), static_cast<u16>(words), data_index, kNoDraw});
}

void BeforeDraw()
{
  if (s_pending.empty() || !Enabled())
    return;
  ResolvePending(g_main_cp_state.array_bases[CPArray::Position]);
}

bool HoldsPreviousFrame(u32 address)
{
  return address < 0x100 && s_pos_previous[address];
}

// After a state load (or a frame-interpolation restart) the shadow holds another
// scene: take the live XF state and stop pairing until a full frame is recorded.
void Reset()
{
  s_pos_previous.reset();
  std::copy(std::begin(xfmem.posMatrices), std::end(xfmem.posMatrices), g_pos);
  std::copy(xfmem.projection.rawProjection.begin(), xfmem.projection.rawProjection.end(), g_proj);
  g_pos_dirty = true;
  s_cur.clear();
  s_pending.clear();
  s_prev_data.clear();
  s_cur_data.clear();
  s_prev_map.clear();
  s_key_uses.clear();
  s_have_prev = false;
  s_pos_loaded.reset();
  s_proj_loaded = false;
}

void Frame()
{
  if (!Enabled())
    return;
  ResolvePending(kNoDraw);
  // MODERNGEKKO_INTERP_STATS=1: every 120 frames, how many matrix loads were
  // paired and moved (interpolated), paired but unchanged, unpaired, or cut.
  static const bool stats = [] {
    const char* v = std::getenv("MODERNGEKKO_INTERP_STATS");
    return v && v[0] == '1';
  }();
  if (stats && ++s_stat_frames >= 120)
  {
    std::fprintf(stderr, "[interp] loads=%u moved=%u static=%u unpaired=%u cut=%u\n",
                 s_stat_loads, s_stat_moved, s_stat_static, s_stat_unpaired, s_stat_cut);
    s_stat_frames = s_stat_loads = s_stat_moved = s_stat_static = s_stat_unpaired = s_stat_cut = 0;
  }
  CameraWarp::OnFrame();
  // Words not loaded this frame keep their live value: the next frame's
  // layer-1 copy of an unchanged matrix must equal the matrix, not the blended
  // value from frames ago (which also outlives the scene that loaded it).
  for (u32 a = 0; a < 0x100; ++a)
  {
    if (!s_pos_loaded[a])
      s_pos_previous.reset(a);
    if (!s_pos_loaded[a] && std::memcmp(&g_pos[a], &xfmem.posMatrices[a], sizeof(float)) != 0)
    {
      g_pos[a] = xfmem.posMatrices[a];
      g_pos_dirty = true;
    }
  }
  if (!s_proj_loaded)
    std::copy(xfmem.projection.rawProjection.begin(), xfmem.projection.rawProjection.end(), g_proj);
  s_pos_loaded.reset();
  s_proj_loaded = false;
  // This frame's loads become the next frame's pairing table.
  s_prev_map.clear();
  for (const LoadRec& rec : s_cur)
    s_prev_map[Key(rec.address, rec.words, rec.base)].push_back(rec.data);
  std::swap(s_prev_data, s_cur_data);
  s_cur.clear();
  s_cur_data.clear();
  s_key_uses.clear();
  s_have_prev = true;
}
}  // namespace FrameInterp

void LoadXFReg(u16 base_address, u8 transfer_size, const u8* data)
{
  // The camera mod's per-frame camera stamp (outside XF memory).
  if (base_address == CameraWarp::kStampAddress)
  {
    CameraWarp::Stamp(transfer_size, data);
    return;
  }
  // The camera mod's world marker: nonzero while the 3D world draws (the
  // time-of-day grade applies), zero around 2D sprites. In FIFO order with
  // the draws; the pending batch belongs to the previous state.
  if (base_address == kWorldMarkerAddress)
  {
    const bool active = transfer_size >= 1 && Common::swap32(data) != 0;
    auto& pixel_shader_manager = Core::System::GetInstance().GetPixelShaderManager();
    if (active != pixel_shader_manager.m_world_tint_active)
    {
      g_vertex_manager->Flush();
      pixel_shader_manager.SetWorldTintActive(active);
    }
    return;
  }
  XFTrace::Load('x', base_address, transfer_size, data);
  FrameInterp::Load(base_address, transfer_size, data);
  if (base_address > XFMEM_REGISTERS_END)
  {
    WARN_LOG_FMT(VIDEO, "XF load base address past end of address space: {:x} {} bytes",
                 base_address, transfer_size);
    return;
  }

  u32 end_address = base_address + transfer_size;  // exclusive

  // do not allow writes past registers
  if (end_address > XFMEM_REGISTERS_END)
  {
    WARN_LOG_FMT(VIDEO, "XF load ends past end of address space: {:x} {} bytes", base_address,
                 transfer_size);
    end_address = XFMEM_REGISTERS_END;
  }

  auto& system = Core::System::GetInstance();
  auto& xf_state_manager = system.GetXFStateManager();

  // write to XF mem
  if (base_address < XFMEM_REGISTERS_START)
  {
    const u32 xf_mem_base = base_address;
    u32 xf_mem_transfer_size = transfer_size;

    if (end_address > XFMEM_REGISTERS_START)
    {
      xf_mem_transfer_size = XFMEM_REGISTERS_START - base_address;
      base_address = XFMEM_REGISTERS_START;
    }

    XFMemWritten(xf_state_manager, xf_mem_transfer_size, xf_mem_base);
    for (u32 i = 0; i < xf_mem_transfer_size; i++)
    {
      ((u32*)&xfmem)[xf_mem_base + i] = Common::swap32(data);
      data += 4;
    }
  }

  // write to XF regs
  if (base_address >= XFMEM_REGISTERS_START)
  {
    for (u32 address = base_address; address < end_address; address++)
    {
      const u32 value = Common::swap32(data);

      XFRegWritten(system, xf_state_manager, address, value);
      ((u32*)&xfmem)[address] = value;

      data += 4;
    }
  }
}

// TODO - verify that it is correct. Seems to work, though.
void LoadIndexedXF(CPArray array, u32 index, u16 address, u8 size)
{
  // load stuff from array to address in xf mem

  const u32 buf_size = size * sizeof(u32);
  u32* currData = reinterpret_cast<u32*>(&xfmem) + address;
  u32* newData;
  auto& system = Core::System::GetInstance();
  auto& fifo = system.GetFifo();
  if (fifo.UseDeterministicGPUThread())
  {
    newData = static_cast<u32*>(fifo.PopFifoAuxBuffer(buf_size));
  }
  else
  {
    auto& memory = system.GetMemory();
    newData = reinterpret_cast<u32*>(memory.GetPointerForRange(
        g_main_cp_state.array_bases[array] + g_main_cp_state.array_strides[array] * index,
        buf_size));
  }

  XFTrace::Load('i', address, size, reinterpret_cast<const u8*>(newData));
  FrameInterp::Load(address, size, reinterpret_cast<const u8*>(newData));
  auto& xf_state_manager = system.GetXFStateManager();
  bool changed = false;
  for (u32 i = 0; i < size; ++i)
  {
    if (currData[i] != Common::swap32(newData[i]))
    {
      changed = true;
      XFMemWritten(xf_state_manager, size, address);
      break;
    }
  }
  if (changed)
  {
    for (u32 i = 0; i < size; ++i)
      currData[i] = Common::swap32(newData[i]);
  }
}

void PreprocessIndexedXF(CPArray array, u32 index, u16 address, u8 size)
{
  const size_t buf_size = size * sizeof(u32);

  auto& system = Core::System::GetInstance();
  auto& memory = system.GetMemory();
  const u8* new_data = memory.GetPointerForRange(
      g_preprocess_cp_state.array_bases[array] + g_preprocess_cp_state.array_strides[array] * index,
      buf_size);

  system.GetFifo().PushFifoAuxBuffer(new_data, buf_size);
}

std::pair<std::string, std::string> GetXFRegInfo(u32 address, u32 value)
{
// Macro to set the register name and make sure it was written correctly via compile time assertion
#define RegName(reg) ((void)(reg), #reg)
#define DescriptionlessReg(reg) std::make_pair(RegName(reg), "");

  switch (address)
  {
  case XFMEM_ERROR:
    return DescriptionlessReg(XFMEM_ERROR);
  case XFMEM_DIAG:
    return DescriptionlessReg(XFMEM_DIAG);
  case XFMEM_STATE0:  // internal state 0
    return std::make_pair(RegName(XFMEM_STATE0), "internal state 0");
  case XFMEM_STATE1:  // internal state 1
    return std::make_pair(RegName(XFMEM_STATE1), "internal state 1");
  case XFMEM_CLOCK:
    return DescriptionlessReg(XFMEM_CLOCK);
  case XFMEM_SETGPMETRIC:
    return DescriptionlessReg(XFMEM_SETGPMETRIC);

  case XFMEM_CLIPDISABLE:
    return std::make_pair(RegName(XFMEM_CLIPDISABLE), fmt::to_string(ClipDisable{.hex = value}));

  case XFMEM_VTXSPECS:
    return std::make_pair(RegName(XFMEM_VTXSPECS), fmt::to_string(INVTXSPEC{.hex = value}));

  case XFMEM_SETNUMCHAN:
    return std::make_pair(RegName(XFMEM_SETNUMCHAN),
                          fmt::format("Number of color channels: {}", value & 3));
    break;

  case XFMEM_SETCHAN0_AMBCOLOR:
    return std::make_pair(RegName(XFMEM_SETCHAN0_AMBCOLOR),
                          fmt::format("Channel 0 Ambient Color: {:08x}", value));
  case XFMEM_SETCHAN1_AMBCOLOR:
    return std::make_pair(RegName(XFMEM_SETCHAN1_AMBCOLOR),
                          fmt::format("Channel 1 Ambient Color: {:08x}", value));

  case XFMEM_SETCHAN0_MATCOLOR:
    return std::make_pair(RegName(XFMEM_SETCHAN0_MATCOLOR),
                          fmt::format("Channel 0 Material Color: {:08x}", value));
  case XFMEM_SETCHAN1_MATCOLOR:
    return std::make_pair(RegName(XFMEM_SETCHAN1_MATCOLOR),
                          fmt::format("Channel 1 Material Color: {:08x}", value));

  case XFMEM_SETCHAN0_COLOR:  // Channel Color
    return std::make_pair(RegName(XFMEM_SETCHAN0_COLOR),
                          fmt::format("Channel 0 Color config:\n{}", LitChannel{.hex = value}));
  case XFMEM_SETCHAN1_COLOR:
    return std::make_pair(RegName(XFMEM_SETCHAN1_COLOR),
                          fmt::format("Channel 1 Color config:\n{}", LitChannel{.hex = value}));
  case XFMEM_SETCHAN0_ALPHA:  // Channel Alpha
    return std::make_pair(RegName(XFMEM_SETCHAN0_ALPHA),
                          fmt::format("Channel 0 Alpha config:\n{}", LitChannel{.hex = value}));
  case XFMEM_SETCHAN1_ALPHA:
    return std::make_pair(RegName(XFMEM_SETCHAN1_ALPHA),
                          fmt::format("Channel 1 Alpha config:\n{}", LitChannel{.hex = value}));

  case XFMEM_DUALTEX:
    return std::make_pair(RegName(XFMEM_DUALTEX),
                          fmt::format("Dual Tex Trans {}", (value & 1) ? "enabled" : "disabled"));

  case XFMEM_SETMATRIXINDA:
    return std::make_pair(RegName(XFMEM_SETMATRIXINDA),
                          fmt::format("Matrix index A:\n{}", TMatrixIndexA{.Hex = value}));
  case XFMEM_SETMATRIXINDB:
    return std::make_pair(RegName(XFMEM_SETMATRIXINDB),
                          fmt::format("Matrix index B:\n{}", TMatrixIndexB{.Hex = value}));

  case XFMEM_SETVIEWPORT:
    return std::make_pair(RegName(XFMEM_SETVIEWPORT + 0),
                          fmt::format("Viewport width: {}", std::bit_cast<float>(value)));
  case XFMEM_SETVIEWPORT + 1:
    return std::make_pair(RegName(XFMEM_SETVIEWPORT + 1),
                          fmt::format("Viewport height: {}", std::bit_cast<float>(value)));
  case XFMEM_SETVIEWPORT + 2:
    return std::make_pair(RegName(XFMEM_SETVIEWPORT + 2),
                          fmt::format("Viewport z range: {}", std::bit_cast<float>(value)));
  case XFMEM_SETVIEWPORT + 3:
    return std::make_pair(RegName(XFMEM_SETVIEWPORT + 3),
                          fmt::format("Viewport x origin: {}", std::bit_cast<float>(value)));
  case XFMEM_SETVIEWPORT + 4:
    return std::make_pair(RegName(XFMEM_SETVIEWPORT + 4),
                          fmt::format("Viewport y origin: {}", std::bit_cast<float>(value)));
  case XFMEM_SETVIEWPORT + 5:
    return std::make_pair(RegName(XFMEM_SETVIEWPORT + 5),
                          fmt::format("Viewport far z: {}", std::bit_cast<float>(value)));
    break;

  case XFMEM_SETPROJECTION:
    return std::make_pair(RegName(XFMEM_SETPROJECTION + 0),
                          fmt::format("Projection[0]: {}", std::bit_cast<float>(value)));
  case XFMEM_SETPROJECTION + 1:
    return std::make_pair(RegName(XFMEM_SETPROJECTION + 1),
                          fmt::format("Projection[1]: {}", std::bit_cast<float>(value)));
  case XFMEM_SETPROJECTION + 2:
    return std::make_pair(RegName(XFMEM_SETPROJECTION + 2),
                          fmt::format("Projection[2]: {}", std::bit_cast<float>(value)));
  case XFMEM_SETPROJECTION + 3:
    return std::make_pair(RegName(XFMEM_SETPROJECTION + 3),
                          fmt::format("Projection[3]: {}", std::bit_cast<float>(value)));
  case XFMEM_SETPROJECTION + 4:
    return std::make_pair(RegName(XFMEM_SETPROJECTION + 4),
                          fmt::format("Projection[4]: {}", std::bit_cast<float>(value)));
  case XFMEM_SETPROJECTION + 5:
    return std::make_pair(RegName(XFMEM_SETPROJECTION + 5),
                          fmt::format("Projection[5]: {}", std::bit_cast<float>(value)));
  case XFMEM_SETPROJECTION + 6:
    return std::make_pair(RegName(XFMEM_SETPROJECTION + 6),
                          fmt::to_string(static_cast<ProjectionType>(value)));

  case XFMEM_SETNUMTEXGENS:
    return std::make_pair(RegName(XFMEM_SETNUMTEXGENS),
                          fmt::format("Number of tex gens: {}", value & 15));

  case XFMEM_SETTEXMTXINFO:
  case XFMEM_SETTEXMTXINFO + 1:
  case XFMEM_SETTEXMTXINFO + 2:
  case XFMEM_SETTEXMTXINFO + 3:
  case XFMEM_SETTEXMTXINFO + 4:
  case XFMEM_SETTEXMTXINFO + 5:
  case XFMEM_SETTEXMTXINFO + 6:
  case XFMEM_SETTEXMTXINFO + 7:
    return std::make_pair(
        fmt::format("XFMEM_SETTEXMTXINFO Matrix {}", address - XFMEM_SETTEXMTXINFO),
        fmt::to_string(TexMtxInfo{.hex = value}));

  case XFMEM_SETPOSTMTXINFO:
  case XFMEM_SETPOSTMTXINFO + 1:
  case XFMEM_SETPOSTMTXINFO + 2:
  case XFMEM_SETPOSTMTXINFO + 3:
  case XFMEM_SETPOSTMTXINFO + 4:
  case XFMEM_SETPOSTMTXINFO + 5:
  case XFMEM_SETPOSTMTXINFO + 6:
  case XFMEM_SETPOSTMTXINFO + 7:
    return std::make_pair(
        fmt::format("XFMEM_SETPOSTMTXINFO Matrix {}", address - XFMEM_SETPOSTMTXINFO),
        fmt::to_string(PostMtxInfo{.hex = value}));

  // --------------
  // Unknown Regs
  // --------------

  // Maybe these are for Normals?
  case 0x1048:  // xfmem.texcoords[0].nrmmtxinfo.hex = data; break; ??
  case 0x1049:
  case 0x104a:
  case 0x104b:
  case 0x104c:
  case 0x104d:
  case 0x104e:
  case 0x104f:
    return std::make_pair(
        fmt::format("Possible Normal Mtx XF reg?: {:x}={:x}", address, value),
        "Maybe these are for Normals? xfmem.texcoords[0].nrmmtxinfo.hex = data; break; ??");
    break;

  case 0x1013:
  case 0x1014:
  case 0x1015:
  case 0x1016:
  case 0x1017:

  default:
    return std::make_pair(fmt::format("Unknown XF Reg: {:x}={:x}", address, value), "");
  }
#undef RegName
#undef DescriptionlessReg
}

std::string GetXFMemName(u32 address)
{
  if (address >= XFMEM_POSMATRICES && address < XFMEM_POSMATRICES_END)
  {
    const u32 row = (address - XFMEM_POSMATRICES) / 4;
    const u32 col = (address - XFMEM_POSMATRICES) % 4;
    return fmt::format("Position matrix row {:2d} col {:2d}", row, col);
  }
  else if (address >= XFMEM_NORMALMATRICES && address < XFMEM_NORMALMATRICES_END)
  {
    const u32 row = (address - XFMEM_NORMALMATRICES) / 3;
    const u32 col = (address - XFMEM_NORMALMATRICES) % 3;
    return fmt::format("Normal matrix row {:2d} col {:2d}", row, col);
  }
  else if (address >= XFMEM_POSTMATRICES && address < XFMEM_POSTMATRICES_END)
  {
    const u32 row = (address - XFMEM_POSTMATRICES) / 4;
    const u32 col = (address - XFMEM_POSTMATRICES) % 4;
    return fmt::format("Post matrix row {:2d} col {:2d}", row, col);
  }
  else if (address >= XFMEM_LIGHTS && address < XFMEM_LIGHTS_END)
  {
    const u32 light = (address - XFMEM_LIGHTS) / 16;
    const u32 offset = (address - XFMEM_LIGHTS) % 16;
    switch (offset)
    {
    default:
      return fmt::format("Light {} unused param {}", light, offset);
    case 3:
      return fmt::format("Light {} color", light);
    case 4:
    case 5:
    case 6:
      return fmt::format("Light {} cosine attenuation {}", light, offset - 4);
    case 7:
    case 8:
    case 9:
      return fmt::format("Light {} distance attenuation {}", light, offset - 7);
    case 10:
    case 11:
    case 12:
      // Yagcd says light pos or "inf ldir", while dolphin has a union for dpos and sdir with only
      // dpos being used nowadays. As far as I can tell only the DX9 engine once at
      // Source/Plugins/Plugin_VideoDX9/Src/TransformEngine.cpp used sdir directly...
      return fmt::format("Light {0} {1} position or inf ldir {1}", light, "xyz"[offset - 10]);
    case 13:
    case 14:
    case 15:
      // Yagcd says light dir or "1/2 angle", dolphin has union for ddir or shalfangle.
      // It would make sense if d stood for direction and s for specular, but it's ddir and
      // shalfangle that have the comment "specular lights only", both at the same offset,
      // while dpos and sdir have none...
      return fmt::format("Light {0} {1} direction or half angle {1}", light, "xyz"[offset - 13]);
    }
  }
  else
  {
    return fmt::format("Unknown memory {:04x}", address);
  }
}

std::string GetXFMemDescription(u32 address, u32 value)
{
  if ((address >= XFMEM_POSMATRICES && address < XFMEM_POSMATRICES_END) ||
      (address >= XFMEM_NORMALMATRICES && address < XFMEM_NORMALMATRICES_END) ||
      (address >= XFMEM_POSTMATRICES && address < XFMEM_POSTMATRICES_END))
  {
    // The matrices all use floats
    return fmt::format("{} = {}", GetXFMemName(address), std::bit_cast<float>(value));
  }
  else if (address >= XFMEM_LIGHTS && address < XFMEM_LIGHTS_END)
  {
    // Each light is 16 words; for this function we don't care which light it is
    const u32 offset = (address - XFMEM_LIGHTS) % 16;
    if (offset <= 3)
    {
      // The unused parameters (0, 1, 2) and the color (3) should be hex-formatted
      return fmt::format("{} = {:08x}", GetXFMemName(address), value);
    }
    else
    {
      // Everything else is a float
      return fmt::format("{} = {}", GetXFMemName(address), std::bit_cast<float>(value));
    }
  }
  else
  {
    // Unknown address
    return fmt::format("{} = {:08x}", GetXFMemName(address), value);
  }
}

std::pair<std::string, std::string> GetXFTransferInfo(u16 base_address, u8 transfer_size,
                                                      const u8* data)
{
  if (base_address > XFMEM_REGISTERS_END)
  {
    return std::make_pair("Invalid XF Transfer", "Base address past end of address space");
  }
  else if (transfer_size == 1 && base_address >= XFMEM_REGISTERS_START)
  {
    // Write directly to a single register
    const u32 value = Common::swap32(data);
    return GetXFRegInfo(base_address, value);
  }

  // More complicated cases
  fmt::memory_buffer name, desc;
  u32 end_address = base_address + transfer_size;  // exclusive

  // do not allow writes past registers
  if (end_address > XFMEM_REGISTERS_END)
  {
    fmt::format_to(std::back_inserter(name), "Invalid XF Transfer ");
    fmt::format_to(std::back_inserter(desc), "Transfer ends past end of address space\n\n");
    end_address = XFMEM_REGISTERS_END;
  }

  // write to XF mem
  if (base_address < XFMEM_REGISTERS_START)
  {
    const u32 xf_mem_base = base_address;
    u32 xf_mem_transfer_size = transfer_size;

    if (end_address > XFMEM_REGISTERS_START)
    {
      xf_mem_transfer_size = XFMEM_REGISTERS_START - base_address;
      base_address = XFMEM_REGISTERS_START;
    }

    fmt::format_to(std::back_inserter(name), "Write {} XF mem words at {:04x}",
                   xf_mem_transfer_size, xf_mem_base);

    for (u32 i = 0; i < xf_mem_transfer_size; i++)
    {
      const auto mem_desc = GetXFMemDescription(xf_mem_base + i, Common::swap32(data));
      fmt::format_to(std::back_inserter(desc), "{}{}", i != 0 ? "\n" : "", mem_desc);
      data += 4;
    }

    if (end_address > XFMEM_REGISTERS_START)
      fmt::format_to(std::back_inserter(name), "; ");
  }

  // write to XF regs
  if (base_address >= XFMEM_REGISTERS_START)
  {
    fmt::format_to(std::back_inserter(name), "Write {} XF regs at {:04x}",
                   end_address - base_address, base_address);

    for (u32 address = base_address; address < end_address; address++)
    {
      const u32 value = Common::swap32(data);

      const auto [regname, regdesc] = GetXFRegInfo(address, value);
      fmt::format_to(std::back_inserter(desc), "{}\n{}\n", regname, regdesc);

      data += 4;
    }
  }

  return std::make_pair(fmt::to_string(name), fmt::to_string(desc));
}

std::pair<std::string, std::string> GetXFIndexedLoadInfo(CPArray array, u32 index, u16 address,
                                                         u8 size)
{
  const auto desc = fmt::format("Load {} words to XF address {:03x} from CP array {} row {}", size,
                                address, array, index);
  fmt::memory_buffer written;
  for (u32 i = 0; i < size; i++)
  {
    fmt::format_to(std::back_inserter(written), "{}\n", GetXFMemName(address + i));
  }

  return std::make_pair(desc, fmt::to_string(written));
}
