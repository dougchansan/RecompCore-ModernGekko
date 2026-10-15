// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// MODERNGEKKO_WAIT_STATS=1: how long the emulation thread blocks on the video
// thread, by reason, reported to stderr every 2 s as milliseconds per second of
// wall time ("[waits] ..."). Off (one cached getenv) otherwise.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "Common/CommonTypes.h"

namespace WaitStats
{
enum class Kind
{
  BlockingEvent,     // AsyncRequests::PushBlockingEvent (EFB peek, bounding box, ...)
  EmptyQueue,        // AsyncRequests::WaitForEmptyQueue
  SyncGPU,           // Fifo::SyncGPU (register access, swap, ...)
  FlushGpu,          // Fifo::FlushGpu
  WaitForGpuThread,  // Fifo::WaitForGpuThread (sync-GPU timing)
  Throttle,          // CoreTiming::Throttle sleep (ahead of real time: not a stall)
  Count
};
inline constexpr std::array<const char*, static_cast<size_t>(Kind::Count)> kNames = {
    "blocking-event", "empty-queue", "sync-gpu", "flush-gpu", "wait-gpu-thread", "throttle"};

inline bool Enabled()
{
  static const bool enabled = std::getenv("MODERNGEKKO_WAIT_STATS") != nullptr;
  return enabled;
}

inline std::array<std::atomic<u64>, static_cast<size_t>(Kind::Count)> s_ns{};
inline std::array<std::atomic<u64>, static_cast<size_t>(Kind::Count)> s_count{};

class Scope
{
public:
  explicit Scope(Kind kind) : m_kind(kind), m_on(Enabled())
  {
    if (m_on)
      m_start = std::chrono::steady_clock::now();
  }
  ~Scope()
  {
    if (!m_on)
      return;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - m_start)
                        .count();
    s_ns[static_cast<size_t>(m_kind)].fetch_add(static_cast<u64>(ns), std::memory_order_relaxed);
    s_count[static_cast<size_t>(m_kind)].fetch_add(1, std::memory_order_relaxed);
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

private:
  Kind m_kind;
  bool m_on;
  std::chrono::steady_clock::time_point m_start{};
};

// Emulation thread, once per emulated field: prints and resets every 2 s.
inline void MaybeReport()
{
  if (!Enabled())
    return;
  static auto last = std::chrono::steady_clock::now();
  static u64 fields = 0;
  ++fields;
  const auto now = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(now - last).count();
  if (seconds < 2.0)
    return;
  char line[512];
  int n = std::snprintf(line, sizeof(line), "[waits] %.0f fields/s, ms per s:", fields / seconds);
  double total = 0;
  for (size_t i = 0; i < kNames.size(); ++i)
  {
    const double ms = s_ns[i].exchange(0, std::memory_order_relaxed) / 1e6 / seconds;
    const u64 count = s_count[i].exchange(0, std::memory_order_relaxed);
    total += ms;
    if (count && n > 0 && n < static_cast<int>(sizeof(line)))
    {
      n += std::snprintf(line + n, sizeof(line) - n, " %s %.1f (%.0f/s)", kNames[i], ms,
                         count / seconds);
    }
  }
  std::fprintf(stderr, "%s; total %.1f\n", line, total);
  last = now;
  fields = 0;
}
}  // namespace WaitStats
