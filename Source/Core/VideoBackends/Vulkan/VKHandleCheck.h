// Copyright 2026 ModernGekko
// SPDX-License-Identifier: GPL-2.0-or-later

// Diagnostic: tracks which image views and samplers are alive, so a descriptor
// write can be checked before it reaches the driver. A write that names a
// destroyed handle crashes inside the driver (amdvlk64.dll on Radeon) with no
// hint of which texture it was; this says which slot, which handle, and what
// that handle used to be. Enabled by MODERNGEKKO_VK_CHECK=1; otherwise every
// call is one predictable branch.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace Vulkan::HandleCheck
{
inline bool Enabled()
{
  static const bool enabled = [] {
    const char* value = std::getenv("MODERNGEKKO_VK_CHECK");
    return value && *value == '1';
  }();
  return enabled;
}

struct Registry
{
  std::mutex mutex;
  std::unordered_map<std::uint64_t, std::string> live;
  std::unordered_map<std::uint64_t, std::string> dead;
};

inline Registry& Views()
{
  static Registry registry;
  return registry;
}

inline Registry& Samplers()
{
  static Registry registry;
  return registry;
}

inline void Created(Registry& registry, std::uint64_t handle, std::string_view what)
{
  if (!Enabled() || !handle)
    return;
  std::lock_guard lock(registry.mutex);
  registry.dead.erase(handle);
  registry.live[handle] = std::string(what);
}

inline void Destroyed(Registry& registry, std::uint64_t handle)
{
  if (!Enabled() || !handle)
    return;
  std::lock_guard lock(registry.mutex);
  const auto it = registry.live.find(handle);
  registry.dead[handle] = it != registry.live.end() ? it->second : std::string("?");
  if (it != registry.live.end())
    registry.live.erase(it);
}

// True when the handle is alive (or checking is off); otherwise logs it.
inline bool Check(Registry& registry, std::uint64_t handle, const char* kind, unsigned slot)
{
  if (!Enabled())
    return true;
  if (!handle)
  {
    // A null handle in a combined image sampler is invalid without the
    // nullDescriptor feature; drivers may dereference it.
    std::fprintf(stderr, "[vk-check] null %s in sampler slot %u\n", kind, slot);
    std::fflush(stderr);
    return false;
  }
  std::lock_guard lock(registry.mutex);
  if (registry.live.count(handle))
    return true;
  const auto it = registry.dead.find(handle);
  std::fprintf(stderr, "[vk-check] stale %s in sampler slot %u: 0x%llx (%s%s)\n", kind, slot,
               static_cast<unsigned long long>(handle),
               it != registry.dead.end() ? "destroyed: " : "never registered",
               it != registry.dead.end() ? it->second.c_str() : "");
  std::fflush(stderr);
  return false;
}
}  // namespace Vulkan::HandleCheck
