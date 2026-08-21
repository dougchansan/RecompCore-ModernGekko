// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/KartDebugOverlay.h"

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <imgui.h>

#include "VideoCommon/KartDebugSymbols.h"

namespace KartDebug
{
namespace
{
// Written once by the CPU thread at core start, read every frame by the video
// thread. Relaxed ordering suffices: the function it points at is static and
// outlives both threads.
std::atomic<RegReader> s_reader{nullptr};

bool s_enabled_cached = false;
bool s_enabled_valid = false;

// Nothing feeds ImGui mouse input in this frontend -- only DolphinQt's
// RenderWidget ever called Presenter::SetMousePos, and the SDL frontend polls
// its events in the aurora window layer, which knows nothing about VideoCommon.
// So the window cannot be dragged, and pinning it by environment variable is
// the honest workaround until input is routed properly.
//
//   KART_DEBUG_OVERLAY=1     top-left (default)
//   KART_DEBUG_OVERLAY=tr    top-right
//   KART_DEBUG_OVERLAY=bl    bottom-left
//   KART_DEBUG_OVERLAY=br    bottom-right
ImVec2 CornerPivot(const char* mode, ImVec2* pivot)
{
  const bool right = mode != nullptr && (mode[0] == 't' || mode[0] == 'b') &&
                     mode[1] == 'r';
  const bool bottom = mode != nullptr && mode[0] == 'b';
  const ImGuiViewport* vp = ImGui::GetMainViewport();
  const float pad = 20.0f;
  *pivot = ImVec2(right ? 1.0f : 0.0f, bottom ? 1.0f : 0.0f);
  return ImVec2(right ? vp->WorkPos.x + vp->WorkSize.x - pad : vp->WorkPos.x + pad,
                bottom ? vp->WorkPos.y + vp->WorkSize.y - pad : vp->WorkPos.y + pad);
}

const char* s_mode = nullptr;
std::atomic<bool> s_input_routed{false};
std::atomic<unsigned> s_mouse_events{0};

// "name +0xoff", or the bare address when the PC lies outside every known
// function -- itself informative, since it means the guest is somewhere the
// decomp does not describe.
void TextAddress(const char* label, u32 address)
{
  u32 offset = 0;
  const char* name = ResolveSymbol(address, &offset);
  if (name != nullptr)
    ImGui::Text("%-3s %08X  %s +0x%X", label, address, name, offset);
  else
    ImGui::Text("%-3s %08X  (no symbol)", label, address);
}
}  // namespace

void SetInputRouted(bool routed)
{
  s_input_routed.store(routed, std::memory_order_relaxed);
}

void NoteMouseEvent()
{
  s_mouse_events.fetch_add(1, std::memory_order_relaxed);
}

void SetRegReader(RegReader reader)
{
  s_reader.store(reader, std::memory_order_relaxed);
}

bool OverlayEnabled()
{
  if (!s_enabled_valid)
  {
    const char* v = std::getenv("KART_DEBUG_OVERLAY");
    s_enabled_cached = v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    s_mode = v;
    s_enabled_valid = true;
  }
  return s_enabled_cached;
}

void DrawOverlay(const float backbuffer_scale)
{
  (void)backbuffer_scale;
  if (!OverlayEnabled())
    return;

  const RegReader reader = s_reader.load(std::memory_order_relaxed);

  // Interactive only once mouse events have actually arrived, not merely
  // because the frontend claimed to route them. On the SDL frontend they never
  // do: the watch registers (even with SDL's event subsystem brought up first)
  // and fires zero times, so the window would otherwise advertise dragging it
  // cannot deliver. Deriving this from observed events rather than from a
  // promise makes the failure self-correcting.
  const bool interactive = s_input_routed.load(std::memory_order_relaxed) &&
                           s_mouse_events.load(std::memory_order_relaxed) > 0;

  // Without input, pin every frame: a window that drifts off screen could
  // never be recovered. With input, place it once and let it be dragged.
  ImVec2 pivot;
  const ImVec2 pos = CornerPivot(s_mode, &pivot);
  ImGui::SetNextWindowPos(pos, interactive ? ImGuiCond_FirstUseEver : ImGuiCond_Always,
                          pivot);
  ImGui::SetNextWindowBgAlpha(0.85f);

  // The system cursor is hidden over the render surface, so ImGui has to draw
  // its own or there is nothing to aim with.
  if (interactive)
    ImGui::GetIO().MouseDrawCursor = true;

  // NoMove/NoNav because none of them can be honoured without input, and an
  // interactive-looking window that ignores the mouse is worse than a static one.
  ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
  if (!interactive)
    flags |= ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoNav |
             ImGuiWindowFlags_NoFocusOnAppearing;

  if (ImGui::Begin("Kart Debug", nullptr, flags))
  {
    if (reader == nullptr)
    {
      // Not an error: this is what a non-recomp core, or a run before the core
      // has started, correctly looks like.
      ImGui::TextUnformatted("no guest state published");
    }
    else
    {
      GuestRegs regs{};
      reader(&regs);

      TextAddress("PC", regs.pc);
      TextAddress("LR", regs.lr);
      ImGui::Separator();

      ImGui::Text("CTR %08X   CR %08X   XER %08X", regs.ctr, regs.cr, regs.xer);
      ImGui::Text("MSR %08X   TB %llu", regs.msr,
                  static_cast<unsigned long long>(regs.timebase));
      ImGui::Separator();

      // Four columns of eight keeps all thirty-two visible without scrolling,
      // which is the point of an overlay rather than a docked panel.
      if (ImGui::BeginTable("gpr", 4, ImGuiTableFlags_SizingFixedFit))
      {
        for (int row = 0; row < 8; ++row)
        {
          ImGui::TableNextRow();
          for (int col = 0; col < 4; ++col)
          {
            const int index = col * 8 + row;
            ImGui::TableSetColumnIndex(col);
            ImGui::Text("r%-2d %08X", index, regs.gpr[index]);
          }
        }
        ImGui::EndTable();
      }

      ImGui::Separator();
      const ImGuiIO& io = ImGui::GetIO();
      ImGui::Text("input %s | events %u | pos %.0f,%.0f | cursor %d",
                  interactive ? "routed" : "NOT routed",
                  s_mouse_events.load(std::memory_order_relaxed), io.MousePos.x,
                  io.MousePos.y, io.MouseDrawCursor ? 1 : 0);
      ImGui::Text("%zu symbols", SymbolCount());

      // Echo the same line to stderr about once a second. The overlay can only
      // be read by someone sitting at the window; this makes the input path
      // diagnosable from a log, which is the only channel the automation
      // harness has.
      // Behind its own variable: at one line a second this drowns the probe
      // output that the defect hunts actually depend on.
      static const bool log_to_stderr = std::getenv("KART_DEBUG_OVERLAY_LOG") != nullptr;
      static unsigned frames;
      if (log_to_stderr && (frames++ % 60u) == 0u)
      {
        std::fprintf(stderr,
                     "[overlay] input=%s events=%u pos=%.0f,%.0f cursor=%d "
                     "pc=%08X\n",
                     interactive ? "routed" : "NOT-routed",
                     s_mouse_events.load(std::memory_order_relaxed), io.MousePos.x,
                     io.MousePos.y, io.MouseDrawCursor ? 1 : 0, regs.pc);
        std::fflush(stderr);
      }
    }
  }
  ImGui::End();
}
}  // namespace KartDebug
