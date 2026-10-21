// Copyright 2026 ModernGekko
// SPDX-License-Identifier: GPL-2.0-or-later

// GX feature census: records which GPU features a game actually uses, so a native renderer can be
// scoped from evidence instead of guesses. Enabled by setting MODERNGEKKO_GX_CENSUS to an output
// path; when unset every hook is a single predictable branch.
//
// The census stores raw register words (BP/XF/CP) for each distinct draw configuration and leaves
// decoding to the offline tool, so this file stays small and does not depend on shader generation.

#pragma once

#include "Common/CommonTypes.h"

class TextureInfo;

namespace GXCensus
{
extern bool g_enabled;

// Reads MODERNGEKKO_GX_CENSUS once. Safe to call repeatedly.
void Init();

inline bool On()
{
  return g_enabled;
}

// Command stream (GPU thread, non-preprocess decode only).
void OnBP(u8 address, u32 value, bool in_display_list);
void OnCP(u8 command, u32 value, bool in_display_list);
void OnXF(u16 address, u8 count, bool in_display_list);
void OnIndexedLoad(u8 array, u16 address, u8 size, bool in_display_list);
void OnPrimitive(u8 primitive, u8 vat, u16 num_vertices, bool in_display_list);
void OnDisplayList(u32 address, u32 size);
void OnUnknownOpcode(u8 opcode);

// A batch is about to be drawn with the current BP/XF/CP state.
void OnDraw(u8 primitive_type, u32 index_count);

// A texture stage is being (re)bound from guest memory or TMEM.
void OnTextureLoad(const TextureInfo& info);

// EFB copy to a texture or to the XFB.
void OnEfbCopy(bool to_xfb, u32 format, bool depth, u32 width, u32 height, bool half_scale,
               bool intensity, bool clear, u32 gamma);

// Pixel engine synchronisation.
void OnDrawDone();
void OnPeToken(bool interrupt);

// CPU-side EFB access (CPU thread).
void OnEfbPeek(bool depth);
void OnEfbPoke(bool depth);

// End of an emulated frame (XFB copy). Periodically rewrites the output file.
void OnFrame();

// Writes the output file now.
void Flush();
}  // namespace GXCensus
