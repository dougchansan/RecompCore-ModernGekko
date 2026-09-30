// Copyright 2026 ModernGekko contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoBackends/Vulkan/ModernGekkoNGX.h"

#include "Common/Logging/Log.h"
#include "VideoBackends/Vulkan/VulkanContext.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#ifdef MODERNGEKKO_ENABLE_NGX
#include <nvsdk_ngx_vk.h>
#endif

namespace Vulkan::ModernGekkoNGX
{
namespace
{
bool s_initialized = false;

bool ParseTruthy(const char* value)
{
  if (!value)
    return false;

  const std::string_view text{value};
  return text == "1" || text == "true" || text == "TRUE" || text == "on" ||
         text == "ON" || text == "yes" || text == "YES";
}

#ifdef MODERNGEKKO_ENABLE_NGX
std::filesystem::path GetApplicationDataPath()
{
  std::filesystem::path base;
  if (const char* xdg_cache = std::getenv("XDG_CACHE_HOME"); xdg_cache && *xdg_cache)
    base = xdg_cache;
  else if (const char* home = std::getenv("HOME"); home && *home)
    base = std::filesystem::path{home} / ".cache";
  else
  {
    std::error_code ec;
    base = std::filesystem::temp_directory_path(ec);
    if (ec)
      base = ".";
  }

  const std::filesystem::path path = base / "moderngekko" / "ngx";
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec)
    WARN_LOG_FMT(VIDEO, "ModernGekko NGX: could not create data directory {}: {}",
                 path.string(), ec.message());
  return path;
}
#endif
}  // namespace

bool IsRequested()
{
  return ParseTruthy(std::getenv("MODERNGEKKO_DLSS"));
}

bool Initialize(VulkanContext& context)
{
  if (!IsRequested())
    return false;

#ifndef MODERNGEKKO_ENABLE_NGX
  WARN_LOG_FMT(VIDEO,
               "ModernGekko NGX: MODERNGEKKO_DLSS is set, but this build was "
               "compiled without MODERNGEKKO_ENABLE_NGX");
  return false;
#else
  if (s_initialized)
    return true;

  const std::wstring data_path = GetApplicationDataPath().wstring();
  const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
      MODERNGEKKO_NGX_PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM,
      "ModernGekko-NGX-PoC-1", data_path.c_str(), context.GetVulkanInstance(),
      context.GetPhysicalDevice(), context.GetDevice());

  if (NVSDK_NGX_FAILED(result))
  {
    ERROR_LOG_FMT(VIDEO,
                  "ModernGekko NGX: Vulkan initialization failed (result {:#x}). "
                  "Check the NGX log under {} and verify the NVIDIA driver/SDK.",
                  static_cast<unsigned>(result), data_path.string());
    return false;
  }

  s_initialized = true;
  INFO_LOG_FMT(VIDEO,
               "ModernGekko NGX: Vulkan SDK initialized on {}. "
               "Phase 1 only validates the SDK/driver handshake; DLSS "
               "Super Resolution evaluation is not enabled yet.",
               context.GetDeviceInfo().deviceName);
  return true;
#endif
}

void Shutdown(VulkanContext& context)
{
#ifdef MODERNGEKKO_ENABLE_NGX
  if (!s_initialized)
    return;

  const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Shutdown1(context.GetDevice());
  if (NVSDK_NGX_FAILED(result))
    WARN_LOG_FMT(VIDEO, "ModernGekko NGX: shutdown returned {:#x}",
                 static_cast<unsigned>(result));
  s_initialized = false;
#else
  (void)context;
#endif
}

bool IsInitialized()
{
  return s_initialized;
}
}  // namespace Vulkan::ModernGekkoNGX
