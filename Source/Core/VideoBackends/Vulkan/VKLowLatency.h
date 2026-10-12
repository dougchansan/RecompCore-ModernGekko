// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "Common/CommonTypes.h"
#include "VideoBackends/Vulkan/VulkanLoader.h"

// NVIDIA Reflex low-latency mode through VK_NV_low_latency2 (NVIDIA Vulkan
// drivers on Windows and Linux): the renderer's counterpart to AMD Anti-Lag.
// Each frame is tagged with a present ID and latency markers; with the mode on,
// the video thread sleeps at the start of a frame for as long as the driver
// says, so the CPU does not queue frames ahead of the GPU. The mode comes from
// g_low_latency_mode (0 off, 1 on, 2 on + boost); the measured simulation-start
// to present latency goes to g_low_latency_ms and the log ("[reflex]").
namespace Vulkan::LowLatency
{
// After device creation, when VulkanContext enabled the extension.
void Init(VkDevice device);
void Shutdown();
bool Enabled();

// Swapchain creation: chain this element into VkSwapchainCreateInfoKHR::pNext
// (nullptr when not enabled), then report the created / destroyed swapchain.
void* SwapChainCreateInfo(const void* next);
void OnSwapChainCreated(VkSwapchainKHR swap_chain);
void OnSwapChainDestroyed(VkSwapchainKHR swap_chain);

// Video thread, at the start of a frame: sleeps (mode on), then marks the
// simulation start of the next frame.
void FrameStart();
// Video thread, before submitting the frame that presents. Returns its present
// ID (0 when not enabled) for the submit and present.
u64 BeginPresent();
// Present on whichever thread submits: wraps vkQueuePresentKHR with the
// present markers and VkPresentIdKHR.
VkResult Present(VkQueue queue, VkPresentInfoKHR* info, u64 present_id);
}  // namespace Vulkan::LowLatency
