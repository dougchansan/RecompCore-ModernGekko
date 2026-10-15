// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "Common/CommonTypes.h"
#include "VideoBackends/Vulkan/VulkanLoader.h"

// NVIDIA Reflex low-latency mode through VK_NV_low_latency2 (NVIDIA Vulkan
// drivers on Windows and Linux): the renderer's counterpart to AMD Anti-Lag.
// The game simulation is Dolphin's emulation thread, not the video thread, so
// the frame boundary is the emulated field boundary (g_low_latency_field_hook):
// there the previous frame's simulation ends, the thread sleeps for as long as
// the driver says (mode on), and the next frame's simulation and input sample
// start. The video thread tags the present of each emulated frame with that
// frame's ID, found from the XFB's CoreTiming ticks. The mode comes from
// g_low_latency_mode (0 off, 1 on, 2 on + boost); the measured simulation-start
// to GPU-end latency goes to g_low_latency_ms and the log ("[reflex]").
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

// Video thread, before submitting a frame that presents the XFB queued with
// these ticks. Returns the emulated frame's present ID for the submit and
// present, or 0 (present untagged): not enabled, an unknown XFB, or a repeat
// present of a frame already tagged (frame interpolation, duplicate XFBs).
u64 BeginPresent(u64 xfb_ticks);
// Present on whichever thread submits: wraps vkQueuePresentKHR with the
// present markers and VkPresentIdKHR.
VkResult Present(VkQueue queue, VkPresentInfoKHR* info, u64 present_id);
}  // namespace Vulkan::LowLatency
