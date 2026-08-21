// Live inspection overlay for static-recomp builds.
//
// Phase 1: read-only. Shows the guest program counter and link register with
// their function names resolved from the decomp symbol table, the general
// purpose register file, and the condition/count registers.
//
// VideoCommon deliberately does NOT include the CPU runtime's headers. Drawing
// a window is no reason to take a dependency on the guest CPU, and the layout
// of CPUState is the runtime's business. Instead the recomp core registers a
// reader callback that copies the handful of fields this window shows; the
// overlay invokes it once per frame from the video thread.
//
// The copy is not synchronised against the CPU thread. That is deliberate: the
// alternative is stalling the guest once per frame to populate a debug window.
// A register that is a few instructions stale is fine here, and nothing in this
// overlay feeds a decision.
//
// Off unless KART_DEBUG_OVERLAY is set, so stock builds and qualification runs
// are byte-for-byte unaffected.
#pragma once

#include "Common/CommonTypes.h"

namespace KartDebug
{
struct GuestRegs
{
  u32 gpr[32];
  u32 pc;
  u32 lr;
  u32 ctr;
  u32 cr;
  u32 xer;
  u32 msr;
  u64 timebase;
};

// Fills `out` from the live guest state. Registered by the recomp core, which
// owns that state and knows its layout.
using RegReader = void (*)(GuestRegs* out);

// Pass nullptr to retract, which the overlay renders as "no guest state".
void SetRegReader(RegReader reader);

// True when KART_DEBUG_OVERLAY is set. Cached after the first call.
bool OverlayEnabled();

// Called by the frontend once it has routed mouse events into ImGui. Until
// then the window is pinned and non-interactive, because a window that looks
// draggable and ignores the mouse is worse than one that is honestly static.
// It also turns on ImGui's software cursor: the system cursor is hidden over
// the render surface, so without it there is nothing to aim with.
void SetInputRouted(bool routed);

// Bumped by the frontend for every mouse event it forwards. Shown in the
// overlay so a dead input path is self-diagnosing: "routed but zero events"
// and "not routed" and "events arriving but no cursor" are three different
// faults and there is no way to tell them apart from the outside.
void NoteMouseEvent();

// Draws the overlay. Call with the ImGui lock held, from the same place as the
// other on-screen windows.
void DrawOverlay(float backbuffer_scale);
}  // namespace KartDebug
