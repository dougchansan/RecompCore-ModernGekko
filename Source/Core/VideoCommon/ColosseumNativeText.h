// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <vector>

#include "Common/CommonTypes.h"

struct AuroraTextGlyph;

// Colosseum text drawn with a real font at output resolution.
//
// The game copies each glyph from its bitmap font into a 512x512 I4 scratch
// texture and draws a quad from that slot. The recomp module reports every copy
// as writes to BP registers 0x6A-0x6E (dolrecomp_native_msg_glyph_blit in
// GXRuntime native_sdk.c), in FIFO order with the draws. The texture cache keeps
// an output-resolution mirror of the scratch texture, writes each reported glyph
// into it from the host font, and hands the mirror out whenever the game samples
// the scratch texture. The game's layout (pen advance, line breaks, slot rect)
// is unchanged; only the pixels inside each slot come from the host font.
namespace ColosseumNativeText
{
struct Glyph
{
  u16 x = 0, y = 0;  // slot in the 512x512 scratch texture, game pixels
  u8 w = 0, h = 0;   // slot size
  s8 y_offset = 0;   // slot top relative to the line top
  u8 font = 0;       // game font id (0: 22px, 1: 18px, 2: 13px, 3/4: condensed)
  u16 code = 0;      // UTF-16 code unit
  u32 bitmap = 0;    // physical address of the game's own 4bpp glyph bitmap
};

// BP writes 0x6A-0x6E; returns false for any other register.
bool OnBPWrite(u8 reg, u32 value);

// Whether text should come from the host font: the HD text option is on and
// the font loaded. Pending glyphs are dropped while it is off.
bool Active();

// Physical address of the scratch texture (0 until the first glyph).
u32 AtlasAddress();

// Glyphs reported since the last TakePending.
bool HasPending();
std::vector<Glyph> TakePending();

// The slot rect a glyph's copy clears, in game pixels: two columns left and
// right of the bitmap and one row above and below.
struct Rect
{
  int x0, y0, x1, y1;
};
Rect ClearedRect(const Glyph& glyph);

// RGBA8 (white, alpha = coverage, as the game's I4 decodes) for the cleared
// rect of `glyph` at `scale` output pixels per game pixel.
void Rasterize(const Glyph& glyph, u32 scale, std::vector<u8>* rgba, u32* width, u32* height);

// A key that identifies the rasterized image of a glyph.
u64 GlyphKey(const Glyph& glyph);

// For the in-process Aurora renderer (AuroraMirror, gxruntime/aurora_backend.h),
// which sees the BP writes in its own command stream: one glyph's five words to
// its pixels, clipped to the scratch texture. False while native text is off.
// Called on Aurora's render thread only.
bool RasterizeForAurora(const u32 words[5], u32 scale, AuroraTextGlyph* out);
}  // namespace ColosseumNativeText
