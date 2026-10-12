// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoBackends/Vulkan/VKLowLatency.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <mutex>

#include "VideoCommon/VideoConfig.h"

namespace Vulkan::LowLatency
{
namespace
{
bool s_enabled = false;
VkDevice s_device = VK_NULL_HANDLE;
VkSwapchainKHR s_swap_chain = VK_NULL_HANDLE;
VkSemaphore s_semaphore = VK_NULL_HANDLE;  // timeline, signalled by vkLatencySleepNV
u64 s_semaphore_value = 0;
u64 s_frame = 0;          // present ID of the frame being built (video thread)
int s_applied_mode = -1;  // last vkSetLatencySleepModeNV
u32 s_frames_since_report = 0;
// Swapchain calls (present, sleep, markers) need external synchronization; the
// present can run on the submission thread.
std::mutex s_mutex;
VkSwapchainLatencyCreateInfoNV s_create_info = {VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV,
                                                nullptr, VK_TRUE};

void Marker(u64 present_id, VkLatencyMarkerNV marker)
{
  const VkSetLatencyMarkerInfoNV info = {VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV, nullptr,
                                         present_id, marker};
  vkSetLatencyMarkerNV(s_device, s_swap_chain, &info);
}

void ApplyMode(int mode)
{
  if (mode == s_applied_mode)
    return;
  const VkLatencySleepModeInfoNV info = {VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV, nullptr,
                                         mode > 0 ? VK_TRUE : VK_FALSE,
                                         mode > 1 ? VK_TRUE : VK_FALSE, 0};
  if (vkSetLatencySleepModeNV(s_device, s_swap_chain, &info) == VK_SUCCESS)
  {
    s_applied_mode = mode;
    std::fprintf(stderr, "[reflex] mode %s\n", mode == 0 ? "off" : mode == 1 ? "on" : "on + boost");
  }
}

// Mean simulation-start to present-end (and to GPU render end) over the
// driver's recent frame reports.
void Report(int mode)
{
  std::array<VkLatencyTimingsFrameReportNV, 64> reports{};
  for (auto& report : reports)
    report.sType = VK_STRUCTURE_TYPE_LATENCY_TIMINGS_FRAME_REPORT_NV;
  VkGetLatencyMarkerInfoNV info = {VK_STRUCTURE_TYPE_GET_LATENCY_MARKER_INFO_NV, nullptr,
                                   static_cast<u32>(reports.size()), reports.data()};
  vkGetLatencyTimingsNV(s_device, s_swap_chain, &info);
  double present_sum = 0, gpu_sum = 0;
  u32 count = 0;
  for (u32 i = 0; i < info.timingCount && i < reports.size(); ++i)
  {
    const auto& r = reports[i];
    if (r.simStartTimeUs == 0 || r.presentEndTimeUs < r.simStartTimeUs ||
        r.gpuRenderEndTimeUs < r.simStartTimeUs)
      continue;
    present_sum += static_cast<double>(r.presentEndTimeUs - r.simStartTimeUs);
    gpu_sum += static_cast<double>(r.gpuRenderEndTimeUs - r.simStartTimeUs);
    ++count;
  }
  if (count == 0)
    return;
  const double present_ms = present_sum / count / 1000.0, gpu_ms = gpu_sum / count / 1000.0;
  g_low_latency_ms.store(static_cast<float>(gpu_ms), std::memory_order_relaxed);
  std::fprintf(stderr,
               "[reflex] mode=%d frames=%u sim->gpu-end %.2f ms sim->present-end %.2f ms\n", mode,
               count, gpu_ms, present_ms);
}
}  // namespace

void Init(VkDevice device)
{
  VkSemaphoreTypeCreateInfo type_info = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, nullptr,
                                         VK_SEMAPHORE_TYPE_TIMELINE, 0};
  const VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type_info, 0};
  if (!vkSetLatencySleepModeNV || !vkLatencySleepNV || !vkSetLatencyMarkerNV ||
      !vkGetLatencyTimingsNV || !vkWaitSemaphores ||
      vkCreateSemaphore(device, &info, nullptr, &s_semaphore) != VK_SUCCESS)
  {
    std::fprintf(stderr, "[reflex] VK_NV_low_latency2 entry points missing: not used\n");
    return;
  }
  s_device = device;
  s_semaphore_value = 0;
  s_enabled = true;
  g_low_latency_supported.store(true, std::memory_order_relaxed);
  std::fprintf(stderr, "[reflex] VK_NV_low_latency2 available\n");
}

void Shutdown()
{
  if (s_semaphore != VK_NULL_HANDLE)
    vkDestroySemaphore(s_device, s_semaphore, nullptr);
  s_semaphore = VK_NULL_HANDLE;
  s_enabled = false;
  s_swap_chain = VK_NULL_HANDLE;
  g_low_latency_supported.store(false, std::memory_order_relaxed);
}

bool Enabled()
{
  return s_enabled;
}

void* SwapChainCreateInfo(const void* next)
{
  if (!s_enabled)
    return nullptr;
  s_create_info.pNext = next;
  return &s_create_info;
}

void OnSwapChainCreated(VkSwapchainKHR swap_chain)
{
  if (!s_enabled)
    return;
  std::lock_guard lock(s_mutex);
  s_swap_chain = swap_chain;
  s_applied_mode = -1;
}

void OnSwapChainDestroyed(VkSwapchainKHR swap_chain)
{
  std::lock_guard lock(s_mutex);
  if (s_swap_chain == swap_chain)
    s_swap_chain = VK_NULL_HANDLE;
}

void FrameStart()
{
  if (!s_enabled)
    return;
  const int mode = g_low_latency_mode.load(std::memory_order_relaxed);
  bool wait = false;
  {
    std::lock_guard lock(s_mutex);
    if (s_swap_chain == VK_NULL_HANDLE)
      return;
    ApplyMode(mode);
    if (++s_frames_since_report >= 240)
    {
      s_frames_since_report = 0;
      Report(mode);
    }
    ++s_frame;
    if (mode > 0)
    {
      const VkLatencySleepInfoNV info = {VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV, nullptr,
                                         s_semaphore, ++s_semaphore_value};
      wait = vkLatencySleepNV(s_device, s_swap_chain, &info) == VK_SUCCESS;
    }
  }
  if (wait)
  {
    const VkSemaphoreWaitInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, nullptr, 0, 1,
                                      &s_semaphore, &s_semaphore_value};
    vkWaitSemaphores(s_device, &info, 100'000'000);  // at most 100 ms
  }
  std::lock_guard lock(s_mutex);
  if (s_swap_chain != VK_NULL_HANDLE)
    Marker(s_frame, VK_LATENCY_MARKER_SIMULATION_START_NV);
}

u64 BeginPresent()
{
  if (!s_enabled)
    return 0;
  std::lock_guard lock(s_mutex);
  if (s_swap_chain == VK_NULL_HANDLE || s_frame == 0)
    return 0;
  Marker(s_frame, VK_LATENCY_MARKER_SIMULATION_END_NV);
  Marker(s_frame, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
  Marker(s_frame, VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
  return s_frame;
}

VkResult Present(VkQueue queue, VkPresentInfoKHR* info, u64 present_id)
{
  if (!s_enabled || present_id == 0)
    return vkQueuePresentKHR(queue, info);
  VkPresentIdKHR ids = {VK_STRUCTURE_TYPE_PRESENT_ID_KHR, info->pNext, 1, &present_id};
  info->pNext = &ids;
  std::lock_guard lock(s_mutex);
  const bool marked = s_swap_chain != VK_NULL_HANDLE && info->pSwapchains[0] == s_swap_chain;
  if (marked)
    Marker(present_id, VK_LATENCY_MARKER_PRESENT_START_NV);
  const VkResult result = vkQueuePresentKHR(queue, info);
  if (marked)
    Marker(present_id, VK_LATENCY_MARKER_PRESENT_END_NV);
  info->pNext = ids.pNext;
  return result;
}
}  // namespace Vulkan::LowLatency
