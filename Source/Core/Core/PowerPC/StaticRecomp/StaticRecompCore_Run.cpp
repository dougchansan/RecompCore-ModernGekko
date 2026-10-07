// RecompCore: StaticRecomp CPU core - Main execution loop.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/PowerPC/StaticRecomp/StaticRecompCore.h"
#include "Core/System.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/PowerPC/Interpreter/Interpreter.h"
#include "Core/PowerPC/StaticRecomp/StaticRecompLockstep.h"
#include "Common/Swap.h"
#include "Core/CoreTiming.h"
#include "Core/HW/CPU.h"
#include "Core/Config/MainSettings.h"
#include "Core/Config/ConfigManager.h"
#include "Core/HW/SystemTimers.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace
{
// Guest task node layout, read from the decomp (gappBackgroundCallback,
// src/game/gs_gapp.c) and confirmed against the generated chunk: stride 0x18,
// prev at 0x0, next at 0x4, state at 0x8, priority at 0xC, blocked at 0xD.
constexpr u32 GAPP_TASK_NEXT = 0x4;
constexpr u32 GAPP_TASK_STATE = 0x8;
constexpr u32 GAPP_TASK_BLOCKED = 0xD;
constexpr u32 GAPP_TASK_RUNNABLE_STATE = 2;

// A walk of a corrupt or mid-update list must terminate. The guest list is
// bounded by the task array, so anything longer than this is not a list.
constexpr u32 GAPP_TASK_WALK_LIMIT = 256;

bool GuestRead32(const u8* ram, u32 ram_size, u32 address, u32* out)
{
  if (address < 0x80000000u)
    return false;
  const u32 offset = address - 0x80000000u;
  if (offset + sizeof(u32) > ram_size)
    return false;
  *out = Common::swap32(&ram[offset]);
  return true;
}

bool GuestRead8(const u8* ram, u32 ram_size, u32 address, u8* out)
{
  if (address < 0x80000000u)
    return false;
  const u32 offset = address - 0x80000000u;
  if (offset >= ram_size)
    return false;
  *out = ram[offset];
  return true;
}

constexpr u32 SYNC_EXCEPTION_MASK = ~static_cast<u32>(
    EXCEPTION_EXTERNAL_INT | EXCEPTION_DECREMENTER | EXCEPTION_PERFORMANCE_MONITOR);
constexpr u32 ASYNC_EXCEPTION_MASK =
    EXCEPTION_EXTERNAL_INT | EXCEPTION_DECREMENTER | EXCEPTION_PERFORMANCE_MONITOR;
constexpr u32 MSR_EE = 0x00008000u;

struct FileCloser
{
  void operator()(std::FILE* file) const
  {
    if (file)
      std::fclose(file);
  }
};

using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

FilePtr OpenDispatchTrace()
{
  const char* path = std::getenv("STATICRECOMP_TRACE_FILE");
  if (!path || !*path)
    return {};

  FilePtr file(std::fopen(path, "w"));
  if (file)
  {
    std::fprintf(file.get(),
                 "event,dispatch,pc,lr,r0,r3,r13,r29,r30,timebase,ppc_downcount\n");
    std::fflush(file.get());
  }
  return file;
}
}

// True when the loop iteration about to start would do nothing at all.
//
// gappBackgroundCallback is the guest's OSSetIdleFunction body: a bare for(;;)
// that walks the active task list, runs any runnable callback, then splices in
// whatever the pending list accumulated. An iteration is a no-op exactly when
// no active node is runnable AND the pending list is empty -- and since we are
// called at the instruction after the loop's final bl, with the walk finished
// and nothing else able to run in between, that is also the state the next
// iteration starts from. Anything unreadable answers "not empty", so a bad read
// costs performance rather than correctness.
bool StaticRecompCore::TaskIdleIterationIsEmpty() const
{
  const u8* ram = m_guest.ram;
  const u32 ram_size = m_guest.ram_size;
  if (ram == nullptr || m_task_list_head == 0 || m_task_pending_head == 0)
    return false;

  u32 pending = 0;
  if (!GuestRead32(ram, ram_size, m_task_pending_head, &pending) || pending != 0)
    return false;

  u32 node = 0;
  if (!GuestRead32(ram, ram_size, m_task_list_head, &node))
    return false;

  for (u32 steps = 0; node != 0; ++steps)
  {
    if (steps >= GAPP_TASK_WALK_LIMIT)
      return false;

    u32 state = 0;
    u8 blocked = 0;
    if (!GuestRead32(ram, ram_size, node + GAPP_TASK_STATE, &state) ||
        !GuestRead8(ram, ram_size, node + GAPP_TASK_BLOCKED, &blocked))
      return false;
    if (state == GAPP_TASK_RUNNABLE_STATE && blocked == 0)
      return false;

    if (!GuestRead32(ram, ram_size, node + GAPP_TASK_NEXT, &node))
      return false;
  }
  return true;
}

void StaticRecompCore::Run()
{
  auto& core_timing = m_system.GetCoreTiming();
  auto& power_pc = m_system.GetPowerPC();
  auto& ppc = power_pc.GetPPCState();
  auto& interpreter = m_system.GetInterpreter();
  auto& memory = m_system.GetMemory();
  const CPU::State* state_ptr = m_system.GetCPU().GetStatePtr();
  FilePtr dispatch_trace = OpenDispatchTrace();

  m_guest.ram = memory.GetRAM();
  m_guest.ram_size = memory.GetRamSizeReal();
  m_guest.exram = memory.GetEXRAM();
  m_guest.exram_size = memory.GetExRamSizeReal();
  InitLookupTable(m_guest.ram_size, m_guest.exram_size);
  const bool lockstep_enabled = m_lockstep_verifier->IsEnabled();
  const auto fast_dispatchable_at = [this](u32 address) {
    if (m_has_rel_modules || !m_forced_fallback_ranges.empty() ||
        (m_module_source.host_call_contains && !m_hook_aware_module))
      return FastDispatchableAt(address);
    if (!m_module_active || m_chunk_lookup_table.empty())
      return false;

    int lookup_index = -1;
    if (address >= 0x80000000u && address < 0x80000000u + m_lookup_ram_size)
    {
      lookup_index = static_cast<int>((address - 0x80000000u) >> 2);
    }
    else if (address >= 0x90000000u && address < 0x90000000u + m_lookup_exram_size)
    {
      lookup_index = static_cast<int>((m_lookup_ram_size >> 2) + ((address - 0x90000000u) >> 2));
    }
    if (lookup_index < 0 || lookup_index >= static_cast<int>(m_chunk_lookup_table.size()))
      return false;
    const int chunk = m_chunk_lookup_table[lookup_index];
    return chunk >= 0 && m_chunk_state[chunk] == CHUNK_VERIFIED;
  };

  const std::string initial_game_id = SConfig::GetInstance().GetGameID();
  m_module_active = m_module && (initial_game_id.empty() || initial_game_id == m_module->game_id);

  if (!m_module_active && m_fallback_jit && !m_guest.host_call)
  {
    m_fallback_jit->Run();
    return;
  }

  while (*state_ptr == CPU::State::Running)
  {
    core_timing.Advance();
    const std::string current_game_id = SConfig::GetInstance().GetGameID();
    m_module_active = m_module && (current_game_id.empty() || current_game_id == m_module->game_id);

    do
    {
      // MSR.FP needs no gate here: generated FPU instructions raise the
      // FP-unavailable exception themselves (ppc_fp_available).
      // A hook-aware module's chassis_dispatch (dolrecomp_call) runs the mod
      // host call itself on every dispatch, and its generated code returns to
      // this loop at every hooked address and hooked return, so hooked
      // addresses need no detour through the JIT.
      if (m_module_active && DispatchableAt(ppc.pc) &&
          (m_hook_aware_module || !(m_guest.host_call && IsHostCallAddress(ppc.pc))))
      {
        SyncIn();
        ++m_bursts;
        do
        {
          const bool trace_sample = (m_native_dispatches & 0xFFFFFu) == 0;
          const bool trace_wait_entry = m_guest.pc == 0x800d3190u;
          if (dispatch_trace && (trace_sample || trace_wait_entry))
          {
            std::fprintf(dispatch_trace.get(),
                         "%s,%llu,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%llu,%d\n",
                         trace_wait_entry ? "wait-entry" : "sample",
                         static_cast<unsigned long long>(m_native_dispatches), m_guest.pc,
                         m_guest.lr, m_guest.gpr[0], m_guest.gpr[3], m_guest.gpr[13],
                         m_guest.gpr[29], m_guest.gpr[30],
                         static_cast<unsigned long long>(m_guest.timebase), ppc.downcount);
            std::fflush(dispatch_trace.get());
          }
          const bool do_ls = lockstep_enabled && m_lockstep_verifier->ShouldCheck(m_guest.pc);
          if (do_ls)
          {
            m_lockstep_verifier->Prepare(m_guest);
          }

          if (m_collect_dispatch_samples && (m_native_dispatches & 4095u) == 0)
            ++m_dispatch_samples[m_guest.pc];
          const u32 runtime_dispatch_address = m_guest.pc;
          u32 linked_dispatch_address = runtime_dispatch_address;
          if (m_has_rel_modules)
            ResolveNativeAddress(runtime_dispatch_address, &linked_dispatch_address, nullptr);
          m_guest.pc = linked_dispatch_address;
          m_module->dispatch(&m_guest, linked_dispatch_address);
          if (m_has_rel_modules)
            m_guest.pc = TranslateRelAddress(m_guest.pc);
          ++m_native_dispatches;

          if (do_ls)
          {
            m_lockstep_verifier->Verify(m_guest);
          }

          // Flush the module's per-block cycle charges into Dolphin's
          // downcount. A dispatch that charged nothing (PC-switch default,
          // pure embedded data) still costs 1 so the burst always makes
          // downcount progress; this per-dispatch flush is also the
          // dispatcher back-edge timing check — CoreTiming regains control
          // with at least CachedInterpreter's per-block frequency, so
          // external-interrupt latency matches stock.
          const s64 charge = -m_guest.downcount;
          m_guest.downcount = 0;
          const u64 effective_charge = static_cast<u64>(charge > 0 ? charge : 1);
          ppc.downcount -= static_cast<int>(effective_charge);
          m_charged_cycles += effective_charge;
          AdvanceGuestTimebase(effective_charge);

          // Yield only after execution has resumed inside the configured loop
          // body and branched back to its start. Initial arrival at idle_pc
          // still runs every callback pump once, while resume points after any
          // nested callback can skip the completed wait iteration.
          const bool completed_idle_loop =
              m_idle_pc != 0 && m_idle_loop_end_pc > m_idle_pc &&
              runtime_dispatch_address > m_idle_pc &&
              runtime_dispatch_address < m_idle_loop_end_pc && m_guest.pc == m_idle_pc;
          if (completed_idle_loop)
          {
            ++m_idle_hits;
            if (dispatch_trace && (m_idle_hits & (m_idle_hits - 1)) == 0)
            {
              std::fprintf(dispatch_trace.get(),
                           "idle,%llu,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%llu,%d\n",
                           static_cast<unsigned long long>(m_native_dispatches), m_guest.pc,
                           m_guest.lr, m_guest.gpr[0], m_guest.gpr[3], m_guest.gpr[13],
                           m_guest.gpr[29], m_guest.gpr[30],
                           static_cast<unsigned long long>(m_guest.timebase), ppc.downcount);
              std::fflush(dispatch_trace.get());
            }
            m_system.GetCoreTiming().Idle();
          }

          // Second idle trigger, for a loop whose back edge never reaches the
          // dispatcher. m_idle_pc cannot fire on gappBackgroundCallback: its
          // back edge is an intra-chunk branch, so the dispatcher never regains
          // control with pc == the loop top. What it does see every iteration
          // is the return from the loop's final bl, which is what this matches.
          if (m_task_idle_pc != 0 && runtime_dispatch_address == m_task_idle_pc &&
              TaskIdleIterationIsEmpty())
          {
            ++m_task_idle_hits;
            if (dispatch_trace && (m_task_idle_hits & (m_task_idle_hits - 1)) == 0)
            {
              std::fprintf(dispatch_trace.get(), "task-idle,%llu,%08x,%llu,%d\n",
                           static_cast<unsigned long long>(m_native_dispatches), m_guest.pc,
                           static_cast<unsigned long long>(m_task_idle_hits), ppc.downcount);
              std::fflush(dispatch_trace.get());
            }
            m_system.GetCoreTiming().Idle();
          }

          // ctx->timebase is refreshed at burst start (SyncIn), and here we
          // incrementally advance it by the exact block cycle charges to
          // prevent guest busy-wait loops from spinning on a stale timebase.
          if (m_guest.exception)
          {
            // DolRecomp's runtime already redirected pc/msr/srr to the guest
            // exception vector; the flag only signals that it happened.
            m_guest.exception = 0;
            m_guest.program_exception = 0;
            ++m_native_exceptions;
          }
          if ((ppc.Exceptions & SYNC_EXCEPTION_MASK) != 0)
            break;  // Hook-raised synchronous exception: deliver via Dolphin below.
          if ((ppc.Exceptions & ASYNC_EXCEPTION_MASK) != 0 && (m_guest.msr & MSR_EE) != 0)
            break;  // rfi/mtmsr re-enabled interrupts while one was pending.
        } while (m_module_active && fast_dispatchable_at(m_guest.pc) &&
                 (m_hook_aware_module || !(m_guest.host_call && IsHostCallAddress(m_guest.pc))) &&
                 ppc.downcount > 0 &&
                 *state_ptr == CPU::State::Running);
        SyncOut();
        if ((ppc.Exceptions & SYNC_EXCEPTION_MASK) != 0)
          power_pc.CheckExceptions();
        else if ((ppc.Exceptions & ASYNC_EXCEPTION_MASK) != 0)
          power_pc.CheckExternalExceptions();
      }
      else
      {
        if (m_guest.host_call && IsHostCallAddress(ppc.pc))
        {
          ++m_host_call_sites[ppc.pc];
          SyncIn();
          bool handled = m_guest.host_call(&m_guest, m_guest.pc);
          if (!handled && m_guest.pc < m_guest.ram_size)
            handled = m_guest.host_call(&m_guest, m_guest.pc | 0x80000000u);
          if (m_fallback_jit && IsHostCallAddress(m_guest.lr))
            m_fallback_jit->GetBlockCache()->InvalidateICache(m_guest.lr, 4, true);
          if (handled)
          {
            const s64 charge = -m_guest.downcount;
            m_guest.downcount = 0;
            const u64 effective_charge = static_cast<u64>(charge > 0 ? charge : 1);
            ppc.downcount -= static_cast<int>(effective_charge);
            AdvanceGuestTimebase(effective_charge);
            SyncOut();
            continue;
          }
          SyncOut();
          if (m_fallback_jit)
          {
            m_host_call_passthrough_pc = ppc.pc;
            m_host_call_passthrough = true;
          }
        }
        // SingleStepInner delivers synchronous exceptions itself; external
        // interrupts are delivered at slice start, as in Interpreter::Run.
        if (m_module_active && IsForcedFallbackAddress(ppc.pc))
        {
          ppc.downcount -= interpreter.SingleStepInner();
          ++m_fallback_steps;
        }
        else if (m_fallback_jit)
        {
          m_fallback_jit->Run();
        }
        else
        {
          do
          {
            ppc.downcount -= interpreter.SingleStepInner();
            ++m_fallback_steps;
          } while (!(m_module_active && DispatchableAt(ppc.pc)) &&
                   !IsHostCallAddress(ppc.pc) && ppc.downcount > 0 &&
                   *state_ptr == CPU::State::Running);
        }
      }
    } while (ppc.downcount > 0 && *state_ptr == CPU::State::Running);
  }
}

void StaticRecompCore::SingleStep()
{
  // Debugger stepping runs through the interpreter; state outside Run() lives
  // in PowerPCState, so no sync is needed.
  auto& system = m_system;
  system.GetCoreTiming().Advance();
  system.GetPPCState().downcount -= system.GetInterpreter().SingleStepInner();
}
