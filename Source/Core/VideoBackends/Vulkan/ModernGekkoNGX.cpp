// Copyright 2026 ModernGekko contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoBackends/Vulkan/ModernGekkoNGX.h"

#include "Common/Logging/Log.h"
#include "VideoBackends/Vulkan/VulkanContext.h"

#include <algorithm>
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
void NVSDK_CONV LogSDKMessage(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature)
{
  if (message)
    INFO_LOG_FMT(VIDEO, "ModernGekko NGX: SDK {}", message);
}

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

namespace
{
#ifdef MODERNGEKKO_ENABLE_NGX
// All pointers passed to discovery refer to storage owned for the entire query.
struct Discovery
{
  std::wstring data_path = GetApplicationDataPath().wstring();
  std::wstring feature_path;
  const wchar_t* path_pointer = nullptr;
  NVSDK_NGX_FeatureCommonInfo common = {};
  NVSDK_NGX_FeatureDiscoveryInfo info = {};
  Discovery()
  {
    common.LoggingInfo.LoggingCallback = LogSDKMessage;
    common.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
    if (const char* path = std::getenv("MODERNGEKKO_NGX_FEATURE_PATH"); path && *path)
    {
      feature_path = std::filesystem::path{path}.wstring();
      path_pointer = feature_path.c_str();
      common.PathListInfo.Path = &path_pointer;
      common.PathListInfo.Length = 1;
    }
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    info.Identifier.v.ProjectDesc = {MODERNGEKKO_NGX_PROJECT_ID,
                                     NVSDK_NGX_ENGINE_TYPE_CUSTOM, "ModernGekko-NGX-PoC-1"};
    info.ApplicationDataPath = data_path.c_str();
    info.FeatureInfo = &common;
  }
};

bool CopyRequirements(NVSDK_NGX_Result result, uint32_t count,
                      const VkExtensionProperties* properties, std::vector<std::string>* names)
{
  names->clear();
  if (NVSDK_NGX_FAILED(result) || (count && !properties))
  {
    WARN_LOG_FMT(VIDEO, "ModernGekko NGX: extension query failed ({:#x}); using ordinary Vulkan",
                 static_cast<unsigned>(result));
    return false;
  }
  for (uint32_t i = 0; i < count; ++i)
  {
    const auto& name = properties[i].extensionName;
    const auto end = std::find(std::begin(name), std::end(name), '\0');
    if (end == std::begin(name) || end == std::end(name))
    {
      names->clear();
      WARN_LOG_FMT(VIDEO, "ModernGekko NGX: invalid extension name; using ordinary Vulkan");
      return false;
    }
    std::string owned(std::begin(name), end);
    if (std::find(names->begin(), names->end(), owned) == names->end())
      names->push_back(std::move(owned));
  }
  return true;
}
#endif
}  // namespace

bool GetInstanceExtensions(std::vector<std::string>* names)
{
  names->clear();
#ifdef MODERNGEKKO_ENABLE_NGX
  if (!IsRequested())
    return false;
  Discovery discovery;
  uint32_t count = 0;
  VkExtensionProperties* properties = nullptr;
  const auto result = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
      &discovery.info, &count, &properties);
  return CopyRequirements(result, count, properties, names);
#else
  return false;
#endif
}

bool GetDeviceExtensions(VkInstance instance, VkPhysicalDevice device,
                         std::vector<std::string>* names)
{
  names->clear();
#ifdef MODERNGEKKO_ENABLE_NGX
  if (!IsRequested())
    return false;
  Discovery discovery;
  uint32_t count = 0;
  VkExtensionProperties* properties = nullptr;
  const auto result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
      instance, device, &discovery.info, &count, &properties);
  return CopyRequirements(result, count, properties, names);
#else
  (void)instance;
  (void)device;
  return false;
#endif
}

bool Initialize(VulkanContext& context)
{
  INFO_LOG_FMT(VIDEO, "ModernGekko runtime: selected Vulkan GPU {} (vendor {:#x}, device {:#x})",
               context.GetDeviceInfo().deviceName, context.GetDeviceInfo().vendorID,
               context.GetDeviceInfo().deviceID);
  if (!IsRequested())
    return false;

#ifndef MODERNGEKKO_ENABLE_NGX
  WARN_LOG_FMT(VIDEO,
               "ModernGekko NGX: MODERNGEKKO_DLSS is set, but this build was "
               "compiled without MODERNGEKKO_ENABLE_NGX");
  return false;
#else
  if (!context.HasNGXExtensions())
  {
    WARN_LOG_FMT(VIDEO, "ModernGekko NGX: required extensions unavailable; using ordinary Vulkan");
    return false;
  }
  if (s_initialized)
    return true;

  const std::filesystem::path data_path = GetApplicationDataPath();
  const std::wstring data_path_wide = data_path.wstring();
  // Keep SDK-referenced search-path storage alive through Shutdown.
  static NVSDK_NGX_FeatureCommonInfo feature_info;
  feature_info = {};
  feature_info.LoggingInfo.LoggingCallback = LogSDKMessage;
  feature_info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
  static std::wstring feature_path_wide;
  feature_path_wide.clear();
  static const wchar_t* feature_path;
  feature_path = nullptr;
  if (const char* path = std::getenv("MODERNGEKKO_NGX_FEATURE_PATH"); path && *path)
  {
    feature_path_wide = std::filesystem::path{path}.wstring();
    feature_path = feature_path_wide.c_str();
    feature_info.PathListInfo.Path = &feature_path;
    feature_info.PathListInfo.Length = 1;
    INFO_LOG_FMT(VIDEO, "ModernGekko NGX: feature search path {}", path);
  }
  const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
      MODERNGEKKO_NGX_PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM,
      "ModernGekko-NGX-PoC-1", data_path_wide.c_str(), context.GetVulkanInstance(),
      context.GetPhysicalDevice(), context.GetDevice(), nullptr, nullptr, &feature_info);

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
               "ModernGekko NGX: Vulkan SDK initialized on {} (result {:#x}). "
               "Phase 1 only validates the SDK/driver handshake; DLSS "
               "Super Resolution evaluation is not enabled yet.",
               context.GetDeviceInfo().deviceName, static_cast<unsigned>(result));

  NVSDK_NGX_Parameter* parameters = nullptr;
  const NVSDK_NGX_Result capability_result =
      NVSDK_NGX_VULKAN_GetCapabilityParameters(&parameters);
  INFO_LOG_FMT(VIDEO, "ModernGekko NGX: GetCapabilityParameters result {:#x}",
               static_cast<unsigned>(capability_result));
  if (!NVSDK_NGX_FAILED(capability_result) && parameters)
  {
    const auto log_capability = [parameters](const char* name, const char* key, auto value) {
      const NVSDK_NGX_Result get_result = parameters->Get(key, &value);
      if (NVSDK_NGX_FAILED(get_result))
        WARN_LOG_FMT(VIDEO, "ModernGekko NGX: {}=unknown (Get result {:#x})", name,
                     static_cast<unsigned>(get_result));
      else
        INFO_LOG_FMT(VIDEO, "ModernGekko NGX: {}={} (Get result {:#x})", name, value,
                     static_cast<unsigned>(get_result));
    };
    log_capability("SuperSampling_Available", NVSDK_NGX_Parameter_SuperSampling_Available, 0);
    log_capability("NeedsUpdatedDriver", NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, 0);
    log_capability("MinDriverVersionMajor", NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor,
                   0u);
    log_capability("MinDriverVersionMinor", NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor,
                   0u);
  }
  else
  {
    WARN_LOG_FMT(VIDEO, "ModernGekko NGX: capability values are unknown; no usable parameter map");
  }
  if (parameters)
  {
    const NVSDK_NGX_Result destroy_result = NVSDK_NGX_VULKAN_DestroyParameters(parameters);
    INFO_LOG_FMT(VIDEO, "ModernGekko NGX: DestroyParameters result {:#x}",
                 static_cast<unsigned>(destroy_result));
  }
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
