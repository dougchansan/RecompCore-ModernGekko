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
#include "Core/HW/Memmap.h"
#include "Core/System.h"
#include "VideoCommon/Present.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace AuroraMirror
{
bool g_on = [] {
  const char* exe = std::getenv("MODERNGEKKO_AURORA_MIRROR");
  const char* dll = std::getenv("MODERNGEKKO_AURORA_NATIVE");
  return (exe && *exe) || (dll && *dll);
}();
bool g_native = [] {
  const char* dll = std::getenv("MODERNGEKKO_AURORA_NATIVE");
  return dll && *dll;
}();

namespace
{
// MODERNGEKKO_AURORA_NATIVE=<aurora-native.dll>: the renderer runs in this
// process instead (tools/aurora-mirror/native.cpp), normally under the Null
// video backend. Commands go to it from the GPU thread in batches of about
// kNativeBatch bytes, so its texture and vertex reads stay close to the draws
// that make them, as Dolphin's own are; each XFB copy presents.
struct Native
{
  bool (*init)(const u8* ram, u32 ram_size);
  void (*feed)(const u8* data, u32 size);
  void (*frame_end)(float aspect);
  // Optional: GXDrawDone. A renderer with its own thread snapshots guest RAM
  // here and renders the batch while the game runs on.
  void (*sync)();
  // Optional: stop the renderer (its thread, its window) at backend shutdown.
  void (*shutdown)();
  // Optional: guest memory the GPU read at this point in the stream.
  void (*capture)(u32 address, u32 size);
};
constexpr size_t kNativeBatch = 8 * 1024;
const bool s_native_requested = [] {
  const char* dll = std::getenv("MODERNGEKKO_AURORA_NATIVE");
  return dll && *dll;
}();
Native s_native{};
bool s_native_failed = false;
bool s_draws_since_sync = false;  // GPU thread

bool NativeReady()
{
  if (s_native.feed || s_native_failed)
    return s_native.feed != nullptr;
  s_native_failed = true;  // until everything below succeeds
#ifdef _WIN32
  const char* path = std::getenv("MODERNGEKKO_AURORA_NATIVE");
  // Its own dependencies (Dawn, SDL, the shader compiler) sit beside it.
  HMODULE dll = LoadLibraryExA(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!dll)
  {
    ERROR_LOG_FMT(VIDEO, "Aurora native: could not load {} (error {})", path, GetLastError());
    return false;
  }
  Native native{
      reinterpret_cast<decltype(Native::init)>(GetProcAddress(dll, "aurora_native_init")),
      reinterpret_cast<decltype(Native::feed)>(GetProcAddress(dll, "aurora_native_feed")),
      reinterpret_cast<decltype(Native::frame_end)>(GetProcAddress(dll, "aurora_native_frame_end")),
      reinterpret_cast<decltype(Native::sync)>(GetProcAddress(dll, "aurora_native_sync")),
      reinterpret_cast<decltype(Native::shutdown)>(GetProcAddress(dll, "aurora_native_shutdown")),
      reinterpret_cast<decltype(Native::capture)>(GetProcAddress(dll, "aurora_native_capture")),
  };
  if (!native.init || !native.feed || !native.frame_end)
  {
    ERROR_LOG_FMT(VIDEO, "Aurora native: {} lacks the aurora_native_* entry points", path);
    return false;
  }
  auto& memory = Core::System::GetInstance().GetMemory();
  if (!native.init(memory.GetRAM(), memory.GetRamSizeReal()))
  {
    ERROR_LOG_FMT(VIDEO, "Aurora native: {} failed to start", path);
    return false;
  }
  s_native = native;
  s_native_failed = false;
  NOTICE_LOG_FMT(VIDEO, "Aurora native: rendering with {}", path);
  return true;
#else
  return false;
#endif
}

// Shared with tools/aurora-mirror. Records are {u32 type, u32 size, payload},
// padded to 8 bytes; positions only grow and wrap modulo the capacity.
constexpr u32 kMagic = 0x4D525541;  // "AURM"
constexpr u32 kVersion = 2;  // 2: frame-end records carry the draw aspect
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
#ifdef _WIN32
// <ring>.data: the producer wrote a frame. <ring>.done: the mirror consumed up
// to header->read.
HANDLE s_data_event = nullptr;
HANDLE s_done_event = nullptr;
#endif
// Synchronous (the default; MODERNGEKKO_AURORA_MIRROR_SYNC=0 turns it off):
// the GPU thread waits at each frame end until the mirror has drawn the frame.
// The mirror reads textures and vertex arrays from live guest RAM, which the
// game rewrites; waiting keeps the game no further ahead than a real GPU lets
// it be, so those reads see what Dolphin saw.
bool s_sync = [] {
  const char* v = std::getenv("MODERNGEKKO_AURORA_MIRROR_SYNC");
  return !(v && std::strcmp(v, "0") == 0);
}();

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
  s_data_event = CreateEventA(nullptr, FALSE, FALSE, (name + ".data").c_str());
  s_done_event = CreateEventA(nullptr, FALSE, FALSE, (name + ".done").c_str());

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

// Native mode: hand the commands gathered so far to the renderer.
void NativeFlush()
{
  if (!s_frame.empty() && NativeReady())
    s_native.feed(s_frame.data(), static_cast<u32>(s_frame.size()));
  s_frame.clear();
}
}  // namespace

void Feed(const u8* data, u32 size)
{
  // Primitive commands (0x80-0xBF) read guest memory; nothing else does
  // except TLUT loads, which CaptureMemory covers.
  if ((data[0] & 0xC0) == 0x80)
    s_draws_since_sync = true;
  s_frame.insert(s_frame.end(), data, data + size);
  if (s_native_requested && s_frame.size() >= kNativeBatch)
    NativeFlush();
}

void FeedIndexedXF(u16 address, u8 size, const u32* big_endian_words)
{
  const u32 header = (static_cast<u32>(size - 1) << 16) | address;
  const u8 command[5] = {0x10, static_cast<u8>(header >> 24), static_cast<u8>(header >> 16),
                         static_cast<u8>(header >> 8), static_cast<u8>(header)};
  s_frame.insert(s_frame.end(), command, command + sizeof(command));
  const u8* words = reinterpret_cast<const u8*>(big_endian_words);
  s_frame.insert(s_frame.end(), words, words + size * 4u);
  if (s_native_requested && s_frame.size() >= kNativeBatch)
    NativeFlush();
}

void DrawDone()
{
  if (!s_native_requested)
    return;
  NativeFlush();
  // With no draw since the last sync there is nothing a snapshot of guest RAM
  // would serve (Colosseum syncs twice a frame, once with no draws between);
  // the batch just carries on.
  if (!s_draws_since_sync)
    return;
  s_draws_since_sync = false;
  if (NativeReady() && s_native.sync)
    s_native.sync();
}

void CaptureMemory(u32 address, u32 size)
{
  if (!s_native_requested)
    return;
  // After the commands before it, so the renderer applies it in order.
  NativeFlush();
  if (NativeReady() && s_native.capture)
    s_native.capture(address, size);
}

void Shutdown()
{
  if (s_native.shutdown)
    s_native.shutdown();
  s_native = {};
  s_native_failed = true;  // stays stopped for the rest of the process
}

void FrameEnd()
{
  if (s_native_requested)
  {
    NativeFlush();
    if (NativeReady())
      s_native.frame_end(g_presenter ? g_presenter->CalculateDrawAspectRatio() : 0.0f);
    return;
  }
  if (!s_header && !s_failed && !Init())
    s_failed = true;
  if (s_failed)
  {
    s_frame.clear();
    return;
  }
  const u64 need = Padded(s_frame.size()) + Padded(sizeof(float));
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
    // The aspect Dolphin presents this frame at (menus pillarboxed 4:3,
    // widescreen gameplay wider), so the mirror can present the same way.
    const float aspect = g_presenter ? g_presenter->CalculateDrawAspectRatio() : 0.0f;
    Write(&pos, RecordFrameEnd, reinterpret_cast<const u8*>(&aspect), sizeof(aspect));
    s_header->write.store(pos, std::memory_order_release);
    s_header->frames.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    SetEvent(s_data_event);
    // Give up on a mirror that stops answering (closed, crashed, stuck)
    // rather than hang the game; the rest of the run is asynchronous. Its
    // first frame waits for the mirror to start up (Dawn, pipelines).
    const int limit = s_header->read.load(std::memory_order_relaxed) == 0 ? 30000 : 5000;
    for (int waited = 0; s_sync && s_header->read.load(std::memory_order_acquire) < pos;)
    {
      if (WaitForSingleObject(s_done_event, 100) == WAIT_TIMEOUT && (waited += 100) >= limit)
      {
        WARN_LOG_FMT(VIDEO, "Aurora mirror: no answer for 5 s, no longer waiting for it");
        s_sync = false;
      }
    }
#endif
  }
  s_frame.clear();
}
}  // namespace AuroraMirror
