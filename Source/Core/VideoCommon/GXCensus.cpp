// Copyright 2026 ModernGekko
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/GXCensus.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "VideoCommon/BPMemory.h"
#include "VideoCommon/CPMemory.h"
#include "VideoCommon/TextureInfo.h"
#include "VideoCommon/XFMemory.h"

namespace GXCensus
{
bool g_enabled = false;

namespace
{
// BPMemory and XFMemory mirror the hardware register files word for word, so a register's
// address is its index into the struct viewed as u32s.
static_assert(sizeof(BPMemory) == 0x100 * sizeof(u32));

enum class Feature : u16
{
  BpWrite,          // value: address | in_display_list << 8
  CpWrite,          // value: command | in_display_list << 8
  XfWrite,          // value: first address | count << 16 | in_display_list << 24
  XfIndexed,        // value: array | size << 8 | address << 16
  Primitive,        // value: primitive | vat << 8 | in_display_list << 12
  VertexFormat,     // value: hash of VCD + VAT words (see vertex_formats for the words)
  DisplayListCall,  // value: size rounded up to a power of two
  UnknownOpcode,    // value: opcode
  DrawPrimitive,    // value: primitive type at draw time
  TextureFormat,    // value: format | tlut << 8 | from_tmem << 16 | levels << 20
  TextureSize,      // value: width << 16 | height
  TextureSampler,   // value: texmode0 (wrap, filter, lod bias, aniso)
  EfbCopy,          // value: to_xfb | format << 1 | depth << 6 | half << 7 | intensity << 8
                    //        | clear << 9 | gamma << 10
  EfbCopySize,      // value: width << 16 | height
  DrawDone,         // value: 0
  PeToken,          // value: interrupt
  EfbPeek,          // value: depth
  EfbPoke,          // value: depth
  Count
};

constexpr std::array<const char*, static_cast<size_t>(Feature::Count)> FEATURE_NAMES = {
    "bp_write",       "cp_write",        "xf_write",        "xf_indexed",
    "primitive",      "vertex_format",   "display_list",    "unknown_opcode",
    "draw_primitive", "texture_format",  "texture_size",    "texture_sampler",
    "efb_copy",       "efb_copy_size",   "draw_done",       "pe_token",
    "efb_peek",       "efb_poke",
};

struct Counter
{
  u64 count = 0;
  u32 first_frame = 0;
  u32 last_frame = 0;
};

struct Snapshot
{
  Counter counter;
  std::vector<u32> words;
};

// Registers that define how a draw is shaded and blended (BP) and transformed/lit (XF).
constexpr std::array<u8, 11> BP_SINGLE = {0x00, 0x0f, 0x27, 0x40, 0x41, 0x42,
                                          0x43, 0xee, 0xf3, 0xf4, 0xf5};
constexpr std::array<u16, 6> XF_SINGLE = {0x1009, 0x100e, 0x100f, 0x1010, 0x1011, 0x1018};

std::mutex s_mutex;
std::string s_path;
u32 s_frame = 0;
u64 s_draws = 0;
u8 s_last_vat = 0;
std::unordered_map<u64, Counter> s_features;
std::unordered_map<u64, Snapshot> s_draw_states;
std::unordered_map<u64, Snapshot> s_vertex_formats;

u64 Key(Feature feature, u64 value)
{
  return (static_cast<u64>(feature) << 48) | (value & 0xFFFF'FFFF'FFFFull);
}

void Touch(Counter& counter)
{
  if (counter.count == 0)
    counter.first_frame = s_frame;
  counter.last_frame = s_frame;
  ++counter.count;
}

void Record(Feature feature, u64 value)
{
  Touch(s_features[Key(feature, value)]);
}

u64 HashWords(const std::vector<u32>& words)
{
  u64 hash = 0xcbf29ce484222325ull;
  for (u32 word : words)
  {
    for (int i = 0; i < 4; ++i)
    {
      hash ^= (word >> (i * 8)) & 0xff;
      hash *= 0x100000001b3ull;
    }
  }
  return hash;
}

void RecordSnapshot(std::unordered_map<u64, Snapshot>& map, std::vector<u32>&& words)
{
  const u64 hash = HashWords(words);
  Snapshot& snapshot = map[hash];
  if (snapshot.counter.count == 0)
    snapshot.words = std::move(words);
  Touch(snapshot.counter);
}

std::vector<u32> VertexFormatWords(u8 vat)
{
  const CPState& cp = g_main_cp_state;
  const VAT& attr = cp.vtx_attr[vat & 7];
  return {cp.vtx_desc.low.Hex, cp.vtx_desc.high.Hex, attr.g0.Hex, attr.g1.Hex, attr.g2.Hex};
}

void WriteCounter(std::FILE* file, const Counter& counter)
{
  std::fprintf(file, "\"count\": %llu, \"first_frame\": %u, \"last_frame\": %u",
               static_cast<unsigned long long>(counter.count), counter.first_frame,
               counter.last_frame);
}

void WriteSnapshots(std::FILE* file, const char* name,
                    const std::unordered_map<u64, Snapshot>& map)
{
  std::fprintf(file, ",\n  \"%s\": [", name);
  bool first = true;
  for (const auto& [hash, snapshot] : map)
  {
    std::fprintf(file, "%s\n    {\"hash\": \"%016llx\", ", first ? "" : ",",
                 static_cast<unsigned long long>(hash));
    WriteCounter(file, snapshot.counter);
    std::fprintf(file, ", \"words\": [");
    for (size_t i = 0; i < snapshot.words.size(); ++i)
      std::fprintf(file, "%s%u", i ? ", " : "", snapshot.words[i]);
    std::fprintf(file, "]}");
    first = false;
  }
  std::fprintf(file, "\n  ]");
}

void WriteLocked()
{
  if (s_path.empty())
    return;

  const std::string tmp_path = s_path + ".tmp";
  std::FILE* file = std::fopen(tmp_path.c_str(), "w");
  if (!file)
    return;

  std::fprintf(file, "{\n  \"schema\": 1,\n  \"frames\": %u,\n  \"draws\": %llu", s_frame,
               static_cast<unsigned long long>(s_draws));

  // The draw-state word layout, so the decoder does not have to hard-code it.
  std::fprintf(file, ",\n  \"draw_state_layout\": {\"bp_single\": [");
  for (size_t i = 0; i < BP_SINGLE.size(); ++i)
    std::fprintf(file, "%s%u", i ? ", " : "", BP_SINGLE[i]);
  std::fprintf(file, "], \"bp_ranges\": [[16, 16], [40, 8], [192, 32], [246, 8]], \"xf_single\": [");
  for (size_t i = 0; i < XF_SINGLE.size(); ++i)
    std::fprintf(file, "%s%u", i ? ", " : "", XF_SINGLE[i]);
  std::fprintf(file, "], \"xf_ranges\": [[4159, 1], [4128, 7], [4160, 8], [4176, 8]], "
                     "\"tail\": [\"primitive_type\"]}");

  std::fprintf(file, ",\n  \"features\": [");
  bool first = true;
  for (const auto& [key, counter] : s_features)
  {
    const auto feature = static_cast<size_t>(key >> 48);
    const u64 value = key & 0xFFFF'FFFF'FFFFull;
    std::fprintf(file, "%s\n    {\"feature\": \"%s\", \"value\": %llu, ", first ? "" : ",",
                 feature < FEATURE_NAMES.size() ? FEATURE_NAMES[feature] : "unknown",
                 static_cast<unsigned long long>(value));
    WriteCounter(file, counter);
    std::fprintf(file, "}");
    first = false;
  }
  std::fprintf(file, "\n  ]");

  WriteSnapshots(file, "draw_states", s_draw_states);
  WriteSnapshots(file, "vertex_formats", s_vertex_formats);
  std::fprintf(file, "\n}\n");
  std::fclose(file);

  std::error_code ec;
  std::filesystem::rename(tmp_path, s_path, ec);
}

struct AutoInit
{
  AutoInit() { Init(); }
};
AutoInit s_auto_init;
}  // namespace

void Init()
{
  static bool s_done = false;
  if (s_done)
    return;
  s_done = true;

  const char* path = std::getenv("MODERNGEKKO_GX_CENSUS");
  if (!path || !*path)
    return;

  s_path = path;
  g_enabled = true;
  std::atexit([] { Flush(); });
}

void OnBP(u8 address, u32, bool in_display_list)
{
  std::lock_guard lock(s_mutex);
  Record(Feature::BpWrite, address | (u64(in_display_list) << 8));
}

void OnCP(u8 command, u32, bool in_display_list)
{
  std::lock_guard lock(s_mutex);
  Record(Feature::CpWrite, command | (u64(in_display_list) << 8));
}

void OnXF(u16 address, u8 count, bool in_display_list)
{
  std::lock_guard lock(s_mutex);
  Record(Feature::XfWrite,
         address | (u64(count) << 16) | (u64(in_display_list) << 24));
}

void OnIndexedLoad(u8 array, u16 address, u8 size, bool in_display_list)
{
  std::lock_guard lock(s_mutex);
  Record(Feature::XfIndexed,
         array | (u64(size) << 8) | (u64(address) << 16) | (u64(in_display_list) << 32));
}

void OnPrimitive(u8 primitive, u8 vat, u16, bool in_display_list)
{
  std::lock_guard lock(s_mutex);
  s_last_vat = vat;
  Record(Feature::Primitive, primitive | (u64(vat) << 8) | (u64(in_display_list) << 12));
  RecordSnapshot(s_vertex_formats, VertexFormatWords(vat));
}

void OnDisplayList(u32, u32 size)
{
  std::lock_guard lock(s_mutex);
  u64 bucket = 1;
  while (bucket < size)
    bucket <<= 1;
  Record(Feature::DisplayListCall, bucket);
}

void OnUnknownOpcode(u8 opcode)
{
  std::lock_guard lock(s_mutex);
  Record(Feature::UnknownOpcode, opcode);
}

void OnDraw(u8 primitive_type, u32)
{
  std::lock_guard lock(s_mutex);
  ++s_draws;
  Record(Feature::DrawPrimitive, primitive_type);

  const u32* bp = reinterpret_cast<const u32*>(&bpmem);
  const u32* xf = reinterpret_cast<const u32*>(&xfmem);

  std::vector<u32> words;
  words.reserve(128);
  for (u8 address : BP_SINGLE)
    words.push_back(bp[address]);
  for (u32 a = 0x10; a < 0x20; ++a)  // indirect TEV stages
    words.push_back(bp[a]);
  for (u32 a = 0x28; a < 0x30; ++a)  // TEV orders
    words.push_back(bp[a]);
  for (u32 a = 0xc0; a < 0xe0; ++a)  // TEV color/alpha combiners
    words.push_back(bp[a]);
  for (u32 a = 0xf6; a < 0xfe; ++a)  // TEV konstant selects and swap tables
    words.push_back(bp[a]);
  for (u16 address : XF_SINGLE)
    words.push_back(xf[address]);
  words.push_back(xf[0x103f]);           // numTexGen
  for (u32 a = 0x1020; a < 0x1027; ++a)  // projection (type at 0x1026)
    words.push_back(xf[a]);
  for (u32 a = 0x1040; a < 0x1048; ++a)  // texgen info
    words.push_back(xf[a]);
  for (u32 a = 0x1050; a < 0x1058; ++a)  // post-transform texgen info
    words.push_back(xf[a]);
  words.push_back(primitive_type);
  RecordSnapshot(s_draw_states, std::move(words));
}

void OnTextureLoad(const TextureInfo& info)
{
  const u32 stage = info.GetStage();
  const u32* bp = reinterpret_cast<const u32*>(&bpmem);
  const u32 mode0 = bp[(stage < 4 ? 0x80 : 0xa0) + (stage & 3)];

  std::lock_guard lock(s_mutex);
  Record(Feature::TextureFormat, static_cast<u64>(info.GetTextureFormat()) |
                                     (static_cast<u64>(info.GetTlutFormat()) << 8) |
                                     (u64(info.IsFromTmem()) << 16) |
                                     (u64(info.GetLevelCount()) << 20));
  Record(Feature::TextureSize, (u64(info.GetRawWidth()) << 16) | info.GetRawHeight());
  Record(Feature::TextureSampler, mode0 & 0xFFFFFF);
}

void OnEfbCopy(bool to_xfb, u32 format, bool depth, u32 width, u32 height, bool half_scale,
               bool intensity, bool clear, u32 gamma)
{
  std::lock_guard lock(s_mutex);
  Record(Feature::EfbCopy, u64(to_xfb) | (u64(format & 0x1f) << 1) | (u64(depth) << 6) |
                               (u64(half_scale) << 7) | (u64(intensity) << 8) |
                               (u64(clear) << 9) | (u64(gamma & 3) << 10));
  Record(Feature::EfbCopySize, (u64(width) << 16) | height);
}

void OnDrawDone()
{
  std::lock_guard lock(s_mutex);
  Record(Feature::DrawDone, 0);
}

void OnPeToken(bool interrupt)
{
  std::lock_guard lock(s_mutex);
  Record(Feature::PeToken, interrupt);
}

void OnEfbPeek(bool depth)
{
  std::lock_guard lock(s_mutex);
  Record(Feature::EfbPeek, depth);
}

void OnEfbPoke(bool depth)
{
  std::lock_guard lock(s_mutex);
  Record(Feature::EfbPoke, depth);
}

void OnFrame()
{
  std::lock_guard lock(s_mutex);
  ++s_frame;
  if (s_frame % 600 == 0)
    WriteLocked();
}

void Flush()
{
  if (!g_enabled)
    return;
  std::lock_guard lock(s_mutex);
  WriteLocked();
}
}  // namespace GXCensus
