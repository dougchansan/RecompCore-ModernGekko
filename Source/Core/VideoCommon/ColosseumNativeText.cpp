// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/ColosseumNativeText.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Common/FileUtil.h"
#include "Common/Logging/Log.h"
#include "Core/HW/Memmap.h"
#include "Core/System.h"
#include "VideoCommon/AuroraMirror.h"
#include "VideoCommon/ColosseumTextUpscale.h"
#include "VideoCommon/TextureCacheBase.h"

// ImGui's copy of stb_truetype; imgui_draw.cpp compiles its own static one.
#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include <imstb_truetype.h>

#include "gxruntime/aurora_backend.h"

namespace ColosseumNativeText
{
namespace
{
// Per game font: the em the host font is drawn at (the game's cell height) and
// the baseline below the line top (bottom of the 'H' ink, from
// tools/colosseum_font).
constexpr float FONT_EM[5] = {22, 18, 13, 18, 13};
constexpr float FONT_BASELINE[5] = {22, 18, 14, 19, 13};

constexpr size_t MAX_PENDING = 8192;

u32 s_words[5];
u32 s_atlas;
std::vector<Glyph> s_pending;

struct Font
{
  std::once_flag tried;  // the GPU thread and Aurora's render thread both ask
  bool loaded = false;
  std::vector<unsigned char> data;
  stbtt_fontinfo info{};
  // MODERNGEKKO_NATIVE_TEXT_SIZE / 22. MS PGothic is narrower than the game
  // font, so 26px fills the game's advance boxes where 22px looks letter-spaced.
  float size_scale = 26.0f / 22.0f;
};
Font s_font;

std::string DefaultFontPath()
{
#ifdef _WIN32
  const char* windir = std::getenv("WINDIR");
  return std::string(windir ? windir : "C:\\Windows") + "\\Fonts\\msgothic.ttc";
#else
  return {};
#endif
}

void LoadFont()
{
  const char* off = std::getenv("MODERNGEKKO_NATIVE_TEXT");
  if (off && std::strcmp(off, "0") == 0)
    return;
  const char* path_env = std::getenv("MODERNGEKKO_NATIVE_TEXT_FONT");
  const std::string path = path_env && *path_env ? path_env : DefaultFontPath();
  std::string bytes;
  if (path.empty() || !File::ReadFileToString(path, bytes))
  {
    WARN_LOG_FMT(VIDEO, "Colosseum native text: no font at '{}'", path);
    return;
  }
  s_font.data.assign(bytes.begin(), bytes.end());
  const char* face_env = std::getenv("MODERNGEKKO_NATIVE_TEXT_FACE");
  const char* face = face_env && *face_env ? face_env : "MS PGothic";
  int offset = stbtt_FindMatchingFont(s_font.data.data(), face, STBTT_MACSTYLE_DONTCARE);
  if (offset < 0)
  {
    // msgothic.ttc is MS Gothic, MS UI Gothic, MS PGothic.
    const int count = stbtt_GetNumberOfFonts(s_font.data.data());
    offset = stbtt_GetFontOffsetForIndex(s_font.data.data(), count > 2 ? 2 : 0);
  }
  if (offset < 0 || !stbtt_InitFont(&s_font.info, s_font.data.data(), offset))
  {
    WARN_LOG_FMT(VIDEO, "Colosseum native text: '{}' has no usable face '{}'", path, face);
    s_font.data.clear();
    return;
  }
  if (const char* size = std::getenv("MODERNGEKKO_NATIVE_TEXT_SIZE"))
  {
    const float px = static_cast<float>(std::atof(size));
    if (px > 4 && px < 128)
      s_font.size_scale = px / FONT_EM[0];
  }
  s_font.loaded = true;
  INFO_LOG_FMT(VIDEO, "Colosseum native text: '{}' from '{}'", face, path);
}

// Codes the game reuses for its own symbols, which the host font would draw as
// the Unicode character: the Pokédollar, font 0's yen-for-backslash, the "e+"/"e"
// stand-ins for alpha and beta, the ㍼㍽㍾ "e" glyphs and the speech bubble.
bool IsGameSymbol(const Glyph& glyph)
{
  switch (glyph.code)
  {
  case 0x24:
  case 0x2031:
  case 0x337C:
  case 0x337D:
  case 0x337E:
    return true;
  case 0x5C:
    return glyph.font == 0;
  case 0x3B1:
  case 0x3B2:
    return glyph.font <= 2;
  default:
    return false;
  }
}

void Put(std::vector<u8>* rgba, u32 width, int x, int y, float coverage)
{
  const u8 a = static_cast<u8>(std::clamp(coverage, 0.0f, 1.0f) * 255.0f + 0.5f);
  if (a == 0)
    return;
  u8* p = rgba->data() + (static_cast<size_t>(y) * width + x) * 4;
  p[0] = p[1] = p[2] = p[3] = std::max(p[3], a);
}

// The host font's glyph, centered in the slot's advance (squeezed to it when
// wider) and sat on the game font's baseline.
bool DrawFontGlyph(const Glyph& glyph, u32 scale, std::vector<u8>* rgba, u32 width, int bx0,
                   int by0, int bw, int bh)
{
  const int index = stbtt_FindGlyphIndex(&s_font.info, glyph.code);
  if (index == 0)
    return false;
  const int font = std::min<int>(glyph.font, 4);
  const float em = FONT_EM[font] * s_font.size_scale * scale;
  const float sy = stbtt_ScaleForMappingEmToPixels(&s_font.info, em);
  int advance = 0, lsb = 0;
  stbtt_GetGlyphHMetrics(&s_font.info, index, &advance, &lsb);
  const float advance_px = advance * sy;
  const float squeeze = advance_px > bw ? bw / advance_px : 1.0f;
  const float sx = sy * squeeze;
  const float pen_x = bx0 + (bw - advance_px * squeeze) * 0.5f;
  const float base_y = by0 + (FONT_BASELINE[font] - glyph.y_offset) * scale;
  const int ipx = static_cast<int>(std::floor(pen_x));
  const int ipy = static_cast<int>(std::floor(base_y));
  const float fx = pen_x - ipx, fy = base_y - ipy;
  int ix0, iy0, ix1, iy1;
  stbtt_GetGlyphBitmapBoxSubpixel(&s_font.info, index, sx, sy, fx, fy, &ix0, &iy0, &ix1, &iy1);
  const int gw = ix1 - ix0, gh = iy1 - iy0;
  if (gw <= 0 || gh <= 0)
    return true;  // blank (space and the like)
  std::vector<unsigned char> coverage(static_cast<size_t>(gw) * gh);
  stbtt_MakeGlyphBitmapSubpixel(&s_font.info, coverage.data(), gw, gh, gw, sx, sy, fx, fy, index);
  for (int y = 0; y < gh; ++y)
  {
    const int ty = ipy + iy0 + y;
    if (ty < by0 || ty >= by0 + bh)
      continue;
    for (int x = 0; x < gw; ++x)
    {
      const int tx = ipx + ix0 + x;
      if (tx < bx0 || tx >= bx0 + bw)
        continue;
      Put(rgba, width, tx, ty, coverage[static_cast<size_t>(y) * gw + x] / 255.0f);
    }
  }
  return true;
}

// The game's own bitmap, rebuilt from the same smooth field as the HD text pass
// (ColosseumTextUpscale.h).
void DrawGameGlyph(const Glyph& glyph, u32 scale, std::vector<u8>* rgba, u32 width, int bx0,
                   int by0, int bw, int bh)
{
  const int stride = (glyph.w + 1) / 2;
  auto& memory = Core::System::GetInstance().GetMemory();
  const u8* bits = memory.GetPointerForRange(glyph.bitmap, static_cast<size_t>(stride) * glyph.h);
  if (!bits)
    return;
  auto texel = [&](int x, int y) -> float {
    if (x < 0 || y < 0 || x >= glyph.w || y >= glyph.h)
      return 0.0f;
    const u8 b = bits[y * stride + x / 2];
    return ((x & 1) ? (b & 0xF) : (b >> 4)) / 15.0f;
  };
  constexpr float k =
      1.0f / (2.0f * ColosseumTextUpscale::SMOOTH_SIGMA * ColosseumTextUpscale::SMOOTH_SIGMA);
  const float half_width = 0.48f / scale;
  for (int py = 0; py < bh; ++py)
  {
    const float v = (py + 0.5f) / scale;
    for (int px = 0; px < bw; ++px)
    {
      const float u = (px + 0.5f) / scale;
      const int cx = static_cast<int>(std::floor(u - 0.5f));
      const int cy = static_cast<int>(std::floor(v - 0.5f));
      float field = 0, weights = 0;
      for (int ty = cy - 1; ty <= cy + 2; ++ty)
        for (int tx = cx - 1; tx <= cx + 2; ++tx)
        {
          const float dx = tx + 0.5f - u, dy = ty + 0.5f - v;
          const float w = std::exp(-(dx * dx + dy * dy) * k);
          field += w * texel(tx, ty);
          weights += w;
        }
      Put(rgba, width, bx0 + px, by0 + py,
          ColosseumTextUpscale::SmoothEdge(field / weights, half_width));
    }
  }
}

Glyph Decode(const u32 words[5])
{
  Glyph glyph;
  glyph.x = words[0] & 0x1FF;
  glyph.y = (words[0] >> 9) & 0x1FF;
  glyph.font = (words[0] >> 18) & 7;
  glyph.w = words[1] & 0x3F;
  glyph.h = (words[1] >> 6) & 0x3F;
  glyph.y_offset = static_cast<s8>((words[1] >> 12) & 0xFF);
  glyph.code = words[3] & 0xFFFF;
  glyph.bitmap = ((words[3] >> 16) & 0xFF) | ((words[4] & 0xFFFFFF) << 8);
  return glyph;
}

u32 DecodeAtlas(const u32 words[5])
{
  return (words[2] & 0xFFFFFF) << 5;
}
}  // namespace

bool OnBPWrite(u8 reg, u32 value)
{
  if (reg < 0x6A || reg > 0x6E)
    return false;
  s_words[reg - 0x6A] = value & 0xFFFFFF;
  // An in-process Aurora renderer draws these from its own command stream.
  if (reg != 0x6E || AuroraMirror::NativeRendering() || !Active())
    return true;

  const Glyph glyph = Decode(s_words);
  s_atlas = DecodeAtlas(s_words);
  if (s_pending.size() >= MAX_PENDING)
    s_pending.erase(s_pending.begin(), s_pending.begin() + MAX_PENDING / 2);
  s_pending.push_back(glyph);
  return true;
}

bool Active()
{
  if (!IsColosseumTextUpscaleEnabled())
  {
    s_pending.clear();
    return false;
  }
  std::call_once(s_font.tried, LoadFont);
  return s_font.loaded;
}

u32 AtlasAddress()
{
  return s_atlas;
}

bool HasPending()
{
  return !s_pending.empty();
}

std::vector<Glyph> TakePending()
{
  std::vector<Glyph> out;
  out.swap(s_pending);
  return out;
}

Rect ClearedRect(const Glyph& glyph)
{
  const int x0 = glyph.x - 2;
  const int y0 = glyph.y - 1;
  return {x0, y0, x0 + 2 * ((glyph.w + 5) >> 1), glyph.y + glyph.h + 1};
}

u64 GlyphKey(const Glyph& glyph)
{
  // The bitmap address is one entry of one font, which fixes the size; the
  // code and font are there for the symbols the host font does not draw.
  return (static_cast<u64>(glyph.bitmap) << 32) | (static_cast<u64>(glyph.code) << 16) |
         (static_cast<u64>(glyph.font) << 8) | static_cast<u8>(glyph.y_offset);
}

void Rasterize(const Glyph& glyph, u32 scale, std::vector<u8>* rgba, u32* width, u32* height)
{
  const Rect r = ClearedRect(glyph);
  *width = static_cast<u32>(r.x1 - r.x0) * scale;
  *height = static_cast<u32>(r.y1 - r.y0) * scale;
  rgba->assign(static_cast<size_t>(*width) * *height * 4, 0);
  const int bx0 = (glyph.x - r.x0) * scale;
  const int by0 = (glyph.y - r.y0) * scale;
  const int bw = glyph.w * scale;
  const int bh = glyph.h * scale;
  if (IsGameSymbol(glyph) || !DrawFontGlyph(glyph, scale, rgba, *width, bx0, by0, bw, bh))
    DrawGameGlyph(glyph, scale, rgba, *width, bx0, by0, bw, bh);
}

int PlaceSlot(u32 code, u32 font, s32 height, s32* y_offset)
{
  if (!IsColosseumTextUpscaleEnabled())
    return 0;
  std::call_once(s_font.tried, LoadFont);
  if (!s_font.loaded)
    return 0;
  Glyph glyph;
  glyph.code = static_cast<u16>(code);
  glyph.font = static_cast<u8>(font);
  if (IsGameSymbol(glyph))
    return 0;
  const int index = stbtt_FindGlyphIndex(&s_font.info, glyph.code);
  int ix0, iy0, ix1, iy1;
  if (index == 0 || !stbtt_GetGlyphBox(&s_font.info, index, &ix0, &iy0, &ix1, &iy1))
    return 0;
  const int f = std::min<int>(font, 4);
  const float sy = stbtt_ScaleForMappingEmToPixels(&s_font.info, FONT_EM[f] * s_font.size_scale);
  // The host ink's rows relative to the line top, as DrawFontGlyph places it.
  const float top = FONT_BASELINE[f] - iy1 * sy;
  const float bottom = FONT_BASELINE[f] - iy0 * sy;
  int offset = *y_offset;
  if (bottom - top > height)
    offset = static_cast<int>(std::floor((top + bottom - height) * 0.5f));
  else if (top < offset)
    offset = static_cast<int>(std::floor(top));
  else if (bottom > offset + height)
    offset = static_cast<int>(std::ceil(bottom)) - height;
  if (offset == *y_offset || offset < -128 || offset > 127)
    return 0;
  *y_offset = offset;
  return 1;
}

bool RasterizeForAurora(const u32 words[5], u32 scale, AuroraTextGlyph* out)
{
  // Not Active(): that drops the GPU thread's pending glyphs.
  if (!IsColosseumTextUpscaleEnabled())
    return false;
  std::call_once(s_font.tried, LoadFont);
  if (!s_font.loaded)
    return false;
  const Glyph glyph = Decode(words);
  const Rect r = ClearedRect(glyph);
  const int x0 = std::max(r.x0, 0), y0 = std::max(r.y0, 0);
  const int x1 = std::min(r.x1, 512), y1 = std::min(r.y1, 512);
  if (x0 >= x1 || y0 >= y1)
    return false;

  // Clipped images by glyph and scale; a few thousand cover any screen.
  struct Image
  {
    std::vector<u8> rgba;
    u32 x = 0, y = 0, width = 0, height = 0;
  };
  static std::unordered_map<u64, Image> cache;
  static u32 cache_scale = 0;
  if (cache_scale != scale || cache.size() > 4096)
  {
    cache.clear();
    cache_scale = scale;
  }
  // The clip depends on the slot, not just the image.
  const u64 key = GlyphKey(glyph) ^ (static_cast<u64>(x0 - r.x0) << 6) ^
                  (static_cast<u64>(y0 - r.y0) << 4) ^ (static_cast<u64>(r.x1 - x1) << 2) ^
                  static_cast<u64>(r.y1 - y1);
  Image& image = cache[key];
  if (image.rgba.empty())
  {
    std::vector<u8> full;
    u32 width = 0, height = 0;
    Rasterize(glyph, scale, &full, &width, &height);
    image.width = static_cast<u32>(x1 - x0) * scale;
    image.height = static_cast<u32>(y1 - y0) * scale;
    image.rgba.resize(static_cast<size_t>(image.width) * image.height * 4);
    const size_t row = static_cast<size_t>(image.width) * 4;
    for (u32 y = 0; y < image.height; ++y)
    {
      const size_t sx = static_cast<size_t>(x0 - r.x0) * scale;
      const size_t sy = static_cast<size_t>(y0 - r.y0) * scale + y;
      std::memcpy(image.rgba.data() + y * row, full.data() + (sy * width + sx) * 4, row);
    }
  }
  out->atlas = DecodeAtlas(words);
  out->x = static_cast<u32>(x0) * scale;
  out->y = static_cast<u32>(y0) * scale;
  out->width = image.width;
  out->height = image.height;
  out->rgba = image.rgba.data();
  return true;
}
}  // namespace ColosseumNativeText
