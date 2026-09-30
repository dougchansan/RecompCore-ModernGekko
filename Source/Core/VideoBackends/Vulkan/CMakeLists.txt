add_library(videovulkan
  CommandBufferManager.cpp
  CommandBufferManager.h
  Constants.h
  ObjectCache.cpp
  ObjectCache.h
  ShaderCompiler.cpp
  ShaderCompiler.h
  StagingBuffer.cpp
  StagingBuffer.h
  StateTracker.cpp
  StateTracker.h
  VKBoundingBox.cpp
  VKBoundingBox.h
  VKMain.cpp
  VKPerfQuery.cpp
  VKPerfQuery.h
  VKPipeline.cpp
  VKPipeline.h
  VKGfx.cpp
  VKGfx.h
  VKShader.cpp
  VKShader.h
  VKStreamBuffer.cpp
  VKStreamBuffer.h
  VKSwapChain.cpp
  VKSwapChain.h
  VKTexture.cpp
  VKTexture.h
  VKVertexFormat.cpp
  VKVertexFormat.h
  VKVertexManager.cpp
  VKVertexManager.h
  VulkanContext.cpp
  VulkanContext.h
  VulkanLoader.cpp
  VulkanLoader.h
  ModernGekkoNGX.cpp
  ModernGekkoNGX.h
)

target_link_libraries(videovulkan
PUBLIC
  common
  videocommon

PRIVATE
  xxhash::xxhash
)

if (ANDROID AND _M_ARM_64)
  target_link_libraries(videovulkan
  PRIVATE
      adrenotools
  )
endif()

# Only include the Vulkan headers when building the Vulkan backend
target_include_directories(videovulkan
PRIVATE
  ${PROJECT_SOURCE_DIR}/Externals/Vulkan-Headers/include
  ${PROJECT_SOURCE_DIR}/Externals/VulkanMemoryAllocator/include
  ${PROJECT_SOURCE_DIR}/Externals/libadrenotools/include
)

# ModernGekko's first DLSS integration checkpoint uses NVIDIA NGX directly on
# Linux/Vulkan. It is deliberately opt-in so normal Dolphin/RecompCore builds
# keep their existing dependency surface and behavior.
if(MODERNGEKKO_ENABLE_NGX)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    message(FATAL_ERROR
      "MODERNGEKKO_ENABLE_NGX currently supports Linux x86-64 only")
  endif()

  if(NOT MODERNGEKKO_NGX_SDK_ROOT)
    message(FATAL_ERROR
      "MODERNGEKKO_ENABLE_NGX requires MODERNGEKKO_NGX_SDK_ROOT")
  endif()

  set(_moderngekko_ngx_include "${MODERNGEKKO_NGX_SDK_ROOT}/include")
  set(_moderngekko_ngx_library
      "${MODERNGEKKO_NGX_SDK_ROOT}/lib/Linux_x86_64/libnvsdk_ngx.a")

  if(NOT EXISTS "${_moderngekko_ngx_include}/nvsdk_ngx_vk.h")
    message(FATAL_ERROR
      "NVIDIA NGX Vulkan headers not found under MODERNGEKKO_NGX_SDK_ROOT")
  endif()
  if(NOT EXISTS "${_moderngekko_ngx_library}")
    message(FATAL_ERROR
      "NVIDIA NGX Linux library not found: ${_moderngekko_ngx_library}")
  endif()

  if(NOT MODERNGEKKO_NGX_PROJECT_ID)
    message(FATAL_ERROR
      "MODERNGEKKO_NGX_PROJECT_ID must be a GUID-like project identifier")
  endif()

  target_include_directories(videovulkan PRIVATE "${_moderngekko_ngx_include}")
  target_link_libraries(videovulkan PRIVATE "${_moderngekko_ngx_library}" ${CMAKE_DL_LIBS})
  target_compile_definitions(videovulkan PRIVATE
    MODERNGEKKO_ENABLE_NGX=1
    MODERNGEKKO_NGX_PROJECT_ID="\"${MODERNGEKKO_NGX_PROJECT_ID}\""
  )
endif()

if(MSVC)
  # Add precompiled header
  target_link_libraries(videovulkan PRIVATE use_pch)
endif()
