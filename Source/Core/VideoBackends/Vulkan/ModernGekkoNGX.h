// Copyright 2026 ModernGekko contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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
bool Initialize(VulkanContext& context);
void Shutdown(VulkanContext& context);
bool IsInitialized();
}  // namespace ModernGekkoNGX
}  // namespace Vulkan
