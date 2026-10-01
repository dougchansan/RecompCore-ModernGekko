// Copyright 2026 ModernGekko contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once
#include <string>
#include <vector>
#include "VideoBackends/Vulkan/VulkanLoader.h"

namespace Vulkan
{
class VulkanContext;

namespace ModernGekkoNGX
{
// Phase-1 bootstrap for NVIDIA NGX on the existing Vulkan device.
//
// The SDK is compile-time optional and runtime opt-in. Set
// MODERNGEKKO_DLSS=1 to request initialization in an NGX-enabled build.
bool IsRequested();
bool GetInstanceExtensions(std::vector<std::string>* names);
bool GetDeviceExtensions(VkInstance instance, VkPhysicalDevice device,
                         std::vector<std::string>* names);
bool Initialize(VulkanContext& context);
void Shutdown(VulkanContext& context);
bool IsInitialized();

struct Image
{
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkImageSubresourceRange range = {};
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t width = 0;
  uint32_t height = 0;
};

struct Frame
{
  Image color, depth, motion_vectors, output, bias;
  // Input pixel jitter; motion vectors are unjittered current-to-previous pixel offsets.
  float jitter_x = 0;
  float jitter_y = 0;
  bool reset = false;
  float frame_time_ms = 0;
};

bool GetQualityRenderSize(uint32_t output_width, uint32_t output_height,
                          uint32_t* render_width, uint32_t* render_height);
// Caller must WaitForGPUIdle before replacing or destroying a live feature.
// Caller owns command recording/submission, image barriers/layouts (GENERAL),
// and restoring Dolphin state invalidated by NGX. No synchronization is implicit.
bool CreateFeature(VkCommandBuffer commands, uint32_t render_width, uint32_t render_height,
                   uint32_t output_width, uint32_t output_height, bool reversed_depth);
bool Evaluate(VkCommandBuffer commands, const Frame& frame);
bool HasFeature();
void DestroyFeature();
}  // namespace ModernGekkoNGX
}  // namespace Vulkan
