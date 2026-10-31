#pragma once

#include "texture.hpp"
#include <optional>

namespace aurora::gfx::texture_replacement {
void initialize() noexcept;
void shutdown() noexcept;
void register_tlut(const GXTlutObj* obj, const void* data, GXTlutFmt format, uint16_t entries) noexcept;
void load_tlut(const GXTlutObj* obj, uint32_t idx) noexcept;
std::optional<TextureHandle> find_replacement(const GXTexObj_& obj) noexcept;
std::optional<TextureHandle> find_replacement(const GXTexObj_& obj, const GXTlutObj_& tlut) noexcept;
// The replacement for a guest texture given as raw GX data (the gxcore path,
// which has no GXTexObj): Dolphin's tex1_ name from its base level and, for a
// color-indexed format, the palette entries it uses. Render thread.
std::optional<TextureHandle> find_replacement(uint32_t format, uint32_t width, uint32_t height,
                                              const uint8_t* data, uint32_t available, const uint8_t* tlut,
                                              uint32_t tlutAvailable) noexcept;
std::string build_texture_replacement_name(const GXTexObj_& obj) noexcept;
} // namespace aurora::gfx::texture_replacement
