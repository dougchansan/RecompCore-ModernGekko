// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GXRUNTIME_AURORA_BACKEND_H
#define GXRUNTIME_AURORA_BACKEND_H

#include "gxruntime/platform.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AuroraBackendConfig {
    const char* app_name;
    unsigned window_width;
    unsigned window_height;
    bool vsync;
    bool allow_texture_dumps;
    bool info_logging;
    bool graphics_logging;
    bool force_untextured;
} AuroraBackendConfig;

bool dol_aurora_initialize(int argc, char** argv,
                           const AuroraBackendConfig* config);
void dol_aurora_shutdown(void);

/* Colosseum text from a host font. The recomp reports each glyph the game
 * copies into its glyph texture as BP writes 0x6A-0x6E (native_sdk.c); the
 * rasterizer turns those five words into the glyph's pixels at `scale` output
 * pixels per game pixel. It returns false while native text is off. */
typedef struct AuroraTextGlyph {
    u32 atlas;           /* physical address of the game's glyph texture */
    u32 x, y;            /* top left of the pixels, in output pixels */
    u32 width, height;   /* of the pixels */
    const u8* rgba;      /* RGBA8, alpha = coverage; valid until the next call */
} AuroraTextGlyph;
typedef bool (*AuroraTextRasterizer)(const u32 words[5], u32 scale, AuroraTextGlyph* out);
void aurora_backend_set_text_rasterizer(AuroraTextRasterizer rasterizer);

#ifdef __cplusplus
}
#endif

#endif
