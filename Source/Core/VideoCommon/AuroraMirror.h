// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "Common/CommonTypes.h"

// MODERNGEKKO_AURORA_MIRROR=<aurora-mirror.exe>: replay the GX command stream
// in a second renderer, side by side with Dolphin's. Every command the GPU
// thread executes (display lists flattened in place, as the FIFO recorder
// does) goes into a shared-memory ring, with a marker at each XFB copy; the
// mirror process maps the ring and guest RAM and draws with Aurora. The game
// never waits on it: when the ring is full the frame is dropped and the
// mirror is told its GX state is no longer exact.
//
// MODERNGEKKO_AURORA_NATIVE=<aurora-native.dll> takes the same stream to
// GXRuntime/Aurora loaded into this process instead, which then renders the
// game (normally under the Null video backend).
namespace AuroraMirror
{
extern bool g_on;
inline bool On()
{
  return g_on;
}
// The game is drawn by an in-process renderer (MODERNGEKKO_AURORA_NATIVE),
// not by Dolphin's video backend.
extern bool g_native;
inline bool NativeRendering()
{
  return g_native;
}

// One whole GX command.
void Feed(const u8* data, u32 size);
// An indexed XF load (LOAD_INDX_A-D), as an immediate XF load of the words it
// read from guest memory when it ran: the game reuses those buffers.
void FeedIndexedXF(u16 address, u8 size, const u32* big_endian_words);
inline bool IsIndexedLoad(u8 opcode)
{
  return opcode == 0x20 || opcode == 0x28 || opcode == 0x30 || opcode == 0x38;
}
// The frame's XFB copy has executed.
void FrameEnd();
// GXDrawDone, before its finish interrupt is raised: the game will wait for
// the GPU, so guest RAM now holds what the draws so far read.
void DrawDone();
// The video backend is shutting down: stop an in-process renderer.
void Shutdown();
// A savestate restored the GPU registers: replay BP, CP and XF state into the
// stream (as Dolphin's FIFO player loads a recording's registers) before the
// next command, so a renderer fed from the stream starts from the same state.
void ResyncState();
// The GPU read guest memory at this point in the stream (a TLUT load): a
// renderer working from a later copy of guest RAM needs these bytes as they
// are now.
void CaptureMemory(u32 address, u32 size);
}  // namespace AuroraMirror
