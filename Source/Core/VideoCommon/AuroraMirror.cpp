// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/AuroraMirror.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Common/Logging/Log.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace AuroraMirror
{
bool g_on = [] {
  const char* exe = std::getenv("MODERNGEKKO_AURORA_MIRROR");
  return exe && *exe;
}();

namespace
{
// Shared with tools/aurora-mirror. Records are {u32 type, u32 size, payload},
// padded to 8 bytes; positions only grow and wrap modulo the capacity.
constexpr u32 kMagic = 0x4D525541;  // "AURM"
constexpr u32 kVersion = 1;
constexpr u64 kCapacity = 128ull << 20;
constexpr u64 kDataOffset = 4096;
enum : u32
{
  RecordFifo = 1,
  RecordFrameEnd = 2,
};

struct Header
{
  u32 magic;
  u32 version;
  u64 capacity;
  std::atomic<u64> write;
  std::atomic<u64> read;
  std::atomic<u64> frames;
  std::atomic<u64> dropped_frames;
  std::atomic<u32> desync;
  u32 producer_pid;
};

Header* s_header = nullptr;
u8* s_data = nullptr;
bool s_failed = false;
std::vector<u8> s_frame;  // the current frame's commands, GPU thread only

bool Init()
{
#ifdef _WIN32
  const std::string name = "moderngekko-aurora-mirror." + std::to_string(GetCurrentProcessId());
  const u64 size = kDataOffset + kCapacity;
  HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                      static_cast<DWORD>(size >> 32), static_cast<DWORD>(size),
                                      name.c_str());
  if (!mapping)
    return false;
  void* view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size);
  if (!view)
    return false;
  s_header = new (view) Header{};
  s_header->magic = kMagic;
  s_header->version = kVersion;
  s_header->capacity = kCapacity;
  s_header->producer_pid = GetCurrentProcessId();
  s_data = static_cast<u8*>(view) + kDataOffset;

  std::string cmd = "\"" + std::string(std::getenv("MODERNGEKKO_AURORA_MIRROR")) + "\" " +
                    std::to_string(GetCurrentProcessId());
  STARTUPINFOA si{sizeof(si)};
  PROCESS_INFORMATION pi{};
  if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi))
  {
    ERROR_LOG_FMT(VIDEO, "Aurora mirror: could not start {}", cmd);
    return false;
  }
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  NOTICE_LOG_FMT(VIDEO, "Aurora mirror: started {}", cmd);
  return true;
#else
  return false;
#endif
}

void Put(u64 pos, const void* src, u64 size)
{
  const u64 at = pos % kCapacity;
  const u64 first = std::min(size, kCapacity - at);
  std::memcpy(s_data + at, src, first);
  std::memcpy(s_data, static_cast<const u8*>(src) + first, size - first);
}

u64 Padded(u64 size)
{
  return (8 + size + 7) & ~u64(7);
}

void Write(u64* pos, u32 type, const u8* payload, u32 size)
{
  const u32 head[2] = {type, size};
  Put(*pos, head, sizeof(head));
  if (size)
    Put(*pos + 8, payload, size);
  *pos += Padded(size);
}
}  // namespace

void Feed(const u8* data, u32 size)
{
  s_frame.insert(s_frame.end(), data, data + size);
}

void FeedIndexedXF(u16 address, u8 size, const u32* big_endian_words)
{
  const u32 header = (static_cast<u32>(size - 1) << 16) | address;
  const u8 command[5] = {0x10, static_cast<u8>(header >> 24), static_cast<u8>(header >> 16),
                         static_cast<u8>(header >> 8), static_cast<u8>(header)};
  s_frame.insert(s_frame.end(), command, command + sizeof(command));
  const u8* words = reinterpret_cast<const u8*>(big_endian_words);
  s_frame.insert(s_frame.end(), words, words + size * 4u);
}

void FrameEnd()
{
  if (!s_header && !s_failed && !Init())
    s_failed = true;
  if (s_failed)
  {
    s_frame.clear();
    return;
  }
  const u64 need = Padded(s_frame.size()) + Padded(0);
  u64 pos = s_header->write.load(std::memory_order_relaxed);
  const u64 used = pos - s_header->read.load(std::memory_order_acquire);
  if (used + need > kCapacity)
  {
    s_header->dropped_frames.fetch_add(1, std::memory_order_relaxed);
    s_header->desync.store(1, std::memory_order_relaxed);
  }
  else
  {
    Write(&pos, RecordFifo, s_frame.data(), static_cast<u32>(s_frame.size()));
    Write(&pos, RecordFrameEnd, nullptr, 0);
    s_header->write.store(pos, std::memory_order_release);
    s_header->frames.fetch_add(1, std::memory_order_relaxed);
  }
  s_frame.clear();
}
}  // namespace AuroraMirror
