// Copyright 2026 ModernGekko contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "Common/CommonTypes.h"

namespace ModernGekkoTiming
{
bool Enabled();
// GPU scopes are renderer-thread only. Begin must be outside a render pass;
// it resets only its own query pair and never ends a pass or submits work.
// Both timestamps use BOTTOM_OF_PIPE: elapsed queue time may include submission gaps.
u32 Begin(const char* stage);
void End(u32 token);
// Host-only, nonblocking query collection. No query resets, GPU waits or flushes.
void Poll();
void CPU(const char* stage, double milliseconds);
// Caller must have completed GPU work; call before Vulkan device destruction.
void Shutdown();
}
