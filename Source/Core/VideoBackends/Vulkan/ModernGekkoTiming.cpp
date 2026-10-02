// Copyright 2026 ModernGekko contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "VideoBackends/Vulkan/ModernGekkoTiming.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "Common/Logging/Log.h"
#include "VideoBackends/Vulkan/CommandBufferManager.h"
#include "VideoBackends/Vulkan/StateTracker.h"
#include "VideoBackends/Vulkan/VulkanContext.h"
#include "VideoBackends/Vulkan/VulkanLoader.h"

namespace ModernGekkoTiming
{
namespace
{
constexpr u32 PAIRS = 2048;
constexpr u32 WINDOW = 256;
enum class State { Free, Open, Pending };
struct Slot
{
  State state = State::Free;
  u32 token = 0;
  u32 generation = 0;
  u64 end_fence = 0;
  std::string stage;
};
struct Statistics
{
  u64 count = 0;
  double sum = 0;
  std::array<double, WINDOW> samples{};
  u32 used = 0;
  u32 next = 0;
  void Add(double milliseconds)
  {
    ++count;
    sum += milliseconds;
    samples[next] = milliseconds;
    next = (next + 1) % WINDOW;
    used = std::min(used + 1, WINDOW);
  }
};
VkQueryPool s_pool = VK_NULL_HANDLE;
VkDevice s_device = VK_NULL_HANDLE;
std::array<Slot, PAIRS> s_slots;
u32 s_next = 0;
u32 s_polls = 0;
u64 s_dropped = 0;
u64 s_dropped_in_pass = 0;
u64 s_invalid_results = 0;
u64 s_tick_mask = 0;
double s_tick_ms = 0;
bool s_unavailable = false;
std::mutex s_statistics_mutex;
std::map<std::string, Statistics> s_gpu, s_cpu;

void Report()
{
  std::lock_guard lock(s_statistics_mutex);
  const auto report = [](const char* kind, const auto& stages) {
    for (const auto& [name, statistics] : stages)
    {
      if (!statistics.count)
        continue;
      std::vector<double> sorted(statistics.samples.begin(),
                                 statistics.samples.begin() + statistics.used);
      std::sort(sorted.begin(), sorted.end());
      const auto percentile = [&](double fraction) {
        return sorted[static_cast<size_t>((sorted.size() - 1) * fraction)];
      };
      INFO_LOG_FMT(VIDEO, "ModernGekko profile: kind={} stage={} count={} mean_ms={} rolling_samples={} p50_ms={} p95_ms={}",
                   kind, name, statistics.count, statistics.sum / statistics.count,
                   statistics.used, percentile(0.5), percentile(0.95));
    }
  };
  report("gpu_queue", s_gpu);
  report("cpu", s_cpu);
  u32 pending = 0;
  for (const Slot& slot : s_slots)
    pending += slot.state != State::Free;
  INFO_LOG_FMT(VIDEO, "ModernGekko profile: pairs={} pending={} dropped={} dropped_in_pass={} invalid_results={}",
               PAIRS, pending, s_dropped, s_dropped_in_pass, s_invalid_results);
}

bool Initialize()
{
  if (s_pool != VK_NULL_HANDLE)
    return true;
  if (s_unavailable || !Vulkan::g_vulkan_context || !Vulkan::g_command_buffer_mgr)
    return false;
  const auto& context = *Vulkan::g_vulkan_context;
  const u32 valid_bits = context.GetGraphicsQueueProperties().timestampValidBits;
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.GetPhysicalDevice(), &properties);
  if (!valid_bits || !std::isfinite(properties.limits.timestampPeriod) ||
      properties.limits.timestampPeriod <= 0)
  {
    WARN_LOG_FMT(VIDEO, "ModernGekko profile: graphics queue timestamps unavailable");
    s_unavailable = true;
    return false;
  }
  s_tick_mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
  s_tick_ms = properties.limits.timestampPeriod / 1000000.0;
  VkQueryPoolCreateInfo create{};
  create.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  create.queryType = VK_QUERY_TYPE_TIMESTAMP;
  create.queryCount = PAIRS * 2;
  s_device = context.GetDevice();
  const VkResult result = vkCreateQueryPool(s_device, &create, nullptr, &s_pool);
  if (result != VK_SUCCESS)
  {
    WARN_LOG_FMT(VIDEO, "ModernGekko profile: query pool unavailable result={}", static_cast<int>(result));
    s_pool = VK_NULL_HANDLE;
    s_device = VK_NULL_HANDLE;
    s_unavailable = true;
    return false;
  }
  INFO_LOG_FMT(VIDEO, "ModernGekko profile: enabled pairs={} timestamp_valid_bits={} timestamp_period_ns={} no_wait=true",
               PAIRS, valid_bits, properties.limits.timestampPeriod);
  return true;
}

void Collect(bool gpu_idle)
{
  if (s_pool == VK_NULL_HANDLE || !Vulkan::g_command_buffer_mgr)
    return;
  const u64 completed = Vulkan::g_command_buffer_mgr->GetCompletedFenceCounter();
  for (Slot& slot : s_slots)
  {
    if (slot.state != State::Pending || (!gpu_idle && slot.end_fence > completed))
      continue;
    const u32 pair = static_cast<u32>(&slot - s_slots.data());
    // Availability can still describe the previous use until the reset executes.
    // The completed recording-fence check above prevents that stale result.
    std::array<u64, 4> values{};
    const VkResult result = vkGetQueryPoolResults(
        s_device, s_pool, pair * 2, 2, sizeof(values), values.data(), sizeof(u64) * 2,
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS && result != VK_NOT_READY)
    {
      ++s_invalid_results;
      slot.state = State::Free;
      continue;
    }
    if (result == VK_NOT_READY || !values[1] || !values[3])
      continue;
    const double milliseconds = static_cast<double>((values[2] - values[0]) & s_tick_mask) * s_tick_ms;
    if (std::isfinite(milliseconds))
    {
      std::lock_guard lock(s_statistics_mutex);
      s_gpu[slot.stage].Add(milliseconds);
    }
    else
      ++s_invalid_results;
    slot.state = State::Free;
  }
}
}  // namespace

bool Enabled()
{
  static const bool enabled = [] {
    const char* value = std::getenv("MODERNGEKKO_DLSS_PROFILE");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

u32 Begin(const char* stage)
{
  if (!Enabled() || !stage || !*stage || !Initialize())
    return 0;
  if (Vulkan::StateTracker::GetInstance() && Vulkan::StateTracker::GetInstance()->InRenderPass())
  {
    ++s_dropped_in_pass;
    return 0;
  }
  for (u32 attempt = 0; attempt < PAIRS; ++attempt)
  {
    const u32 pair = s_next;
    s_next = (s_next + 1) % PAIRS;
    Slot& slot = s_slots[pair];
    if (slot.state != State::Free)
      continue;
    slot.generation = (slot.generation + 1) & 0xfffff;
    slot.token = (slot.generation << 12) | (pair + 1);
    slot.stage = stage;
    slot.state = State::Open;
    const auto commands = Vulkan::g_command_buffer_mgr->GetCurrentCommandBuffer();
    vkCmdResetQueryPool(commands, s_pool, pair * 2, 2);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s_pool, pair * 2);
    return slot.token;
  }
  ++s_dropped;
  return 0;
}

void End(u32 token)
{
  if (!Enabled() || !token || !Vulkan::g_command_buffer_mgr)
    return;
  const u32 encoded_pair = token & 0xfff;
  if (!encoded_pair || encoded_pair > PAIRS)
    return;
  Slot& slot = s_slots[encoded_pair - 1];
  if (slot.token != token || slot.state != State::Open)
    return;
  vkCmdWriteTimestamp(Vulkan::g_command_buffer_mgr->GetCurrentCommandBuffer(),
                      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s_pool,
                      (encoded_pair - 1) * 2 + 1);
  slot.end_fence = Vulkan::g_command_buffer_mgr->GetCurrentFenceCounter();
  slot.state = State::Pending;
}

void Poll()
{
  if (!Enabled())
    return;
  Collect(false);
  if (++s_polls % 120 == 0)
    Report();
}

void CPU(const char* stage, double milliseconds)
{
  if (!Enabled() || !stage || !*stage || !std::isfinite(milliseconds) || milliseconds < 0)
    return;
  std::lock_guard lock(s_statistics_mutex);
  s_cpu[stage].Add(milliseconds);
}

void Shutdown()
{
  if (!Enabled())
    return;
  Collect(true);  // Caller guarantees completed GPU work.
  Report();
  if (s_pool != VK_NULL_HANDLE)
    vkDestroyQueryPool(s_device, s_pool, nullptr);
  s_pool = VK_NULL_HANDLE;
  s_device = VK_NULL_HANDLE;
  s_slots = {};
  {
    std::lock_guard lock(s_statistics_mutex);
    s_gpu.clear();
    s_cpu.clear();
  }
  s_next = s_polls = 0;
  s_dropped = s_dropped_in_pass = s_invalid_results = 0;
  s_unavailable = false;
}
}  // namespace ModernGekkoTiming
