// RecompCore: StaticRecomp CPU core - Memory and instruction fallback HLE hooks.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/PowerPC/StaticRecomp/StaticRecompCore.h"
#include "Core/System.h"
#include "Core/PowerPC/MMU.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/PowerPC/JitInterface.h"
#include "Core/PowerPC/Interpreter/Interpreter.h"
#include "Core/PowerPC/StaticRecomp/StaticRecompLockstep.h"
#include "Core/HW/GPFifo.h"
#include "Core/HW/SystemTimers.h"
#include "Common/Logging/Log.h"

#include <cstdio>
#include <cstdlib>

namespace
{
constexpr u32 LOCKED_CACHE_BASE = 0xE0000000u;
}

bool StaticRecompCore::HookHostCall(CPUState* cpu, u32 address)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  // The LLVM backend's generated regions run without per-instruction host-call
  // dispatch, so each one asks at entry whether the host has to intercept
  // anything inside it. Acknowledging the query is what distinguishes "nothing
  // to intercept here" from "this host does not implement the query": leaving
  // external_rid alone makes the module take the safe path instead.
  // A whole-range cache operation from a native substitution. Doing this line
  // by line is what made DCFlushRange the single largest consumer of guest CPU
  // time in a battle scene: JitInterface::InvalidateICacheLines folds a line
  // count into one InvalidateICache, and its own comment describes exactly the
  // dcbf loop this replaces.
  if (address == PPC_HOST_CALL_CACHE_RANGE)
  {
    core->HandleCacheRange(cpu, cpu->external_rid, cpu->external_addr, cpu->external_value);
    return false;
  }

  if (address == PPC_HOST_CALL_NATIVE_REGION_QUERY)
  {
    // STATICRECOMP_NATIVE_REGION=0 declines the query instead of answering it,
    // which is exactly what a host without native-region support does: the
    // module then takes the safe path at every region entry. Keeping the two
    // behaviours on one binary is what makes "is the fast path at fault?" a
    // one-variable question.
    static const bool answer_query = [] {
      const char* setting = std::getenv("STATICRECOMP_NATIVE_REGION");
      return !setting || std::atoi(setting) != 0;
    }();
    if (!answer_query)
      return true;
    const bool blocked = core->RegionNeedsInterception(cpu->external_addr, cpu->external_value);
    cpu->external_rid = PPC_NATIVE_REGION_QUERY_HANDLED;
    return blocked;
  }
  return core->m_module_source.host_call &&
         core->m_module_source.host_call(cpu, address, core->m_module_source.host_call_user);
}

bool StaticRecompCore::CallGuest(CPUState* cpu, u32 address)
{
  auto* core = cpu ? static_cast<StaticRecompCore*>(cpu->external_user_data) : nullptr;
  if (!core || !core->m_module || !core->m_module->dispatch)
    return false;
  // Returning to this address ends the call: low memory globals, never code.
  constexpr u32 kReturn = 0x80000010u;
  CPUState state = *cpu;
  state.lr = kReturn;
  state.pc = address;
  for (u32 dispatches = 0; dispatches < 10000000u; ++dispatches)
  {
    if (state.pc == kReturn)
    {
      cpu->gpr[3] = state.gpr[3];
      cpu->gpr[4] = state.gpr[4];
      return true;
    }
    // `sc` (the OS uses it to sync caches around thread switches): the module
    // has redirected to the system call vector with srr0/srr1 saved. The GC
    // OS handler only toggles HID0 and returns (rfi), so return directly.
    if (state.pc == 0x00000C00u || state.pc == 0x80000C00u)
    {
      state.pc = state.srr0;
      state.msr = state.srr1;
      state.exception = 0;
      continue;
    }
    // FP unavailable: the OS enables the FPU lazily per thread. The call runs
    // on the hooked thread's own state, so enable it and retry the instruction.
    if (state.pc == 0x00000800u || state.pc == 0x80000800u)
    {
      state.pc = state.srr0;
      state.msr = state.srr1 | 0x2000u;  // MSR[FP]
      state.exception = 0;
      continue;
    }
    state.exception = 0;
    if (!core->DispatchableAt(state.pc))
    {
      std::fprintf(stderr, "[mod] guest call %08x left module code at %08x\n", address,
                   state.pc);
      return false;
    }
    u32 linked = state.pc;
    if (core->m_has_rel_modules)
      core->ResolveNativeAddress(state.pc, &linked, nullptr);
    state.pc = linked;
    core->m_module->dispatch(&state, linked);
    if (core->m_has_rel_modules)
      state.pc = core->TranslateRelAddress(state.pc);
    state.downcount = 0;
  }
  std::fprintf(stderr, "[mod] guest call %08x did not return\n", address);
  return false;
}

// Profiling: external (non-RAM) accesses bucketed by the top address nibble.
u64 g_static_recomp_ext_counts[2][16];

u64 StaticRecompCore::HookExternalRead(CPUState* cpu, u32 ea, u8 size)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ++g_static_recomp_ext_counts[0][ea >> 28];
  ea = core->TranslateRelAddress(ea);
  if (ea == 0)
    std::fprintf(stderr, "[zero-access] read size=%u guest_pc=%08x ppc_pc=%08x lr=%08x\n", size,
                 cpu->pc, core->m_system.GetPPCState().pc, cpu->lr);
  core->PropagateGuestMSR();
  auto& mmu = core->m_system.GetMMU();
  u64 value;
  switch (size)
  {
  case 1:
    value = mmu.Read<u8>(ea);
    break;
  case 2:
    value = mmu.Read<u16>(ea);
    break;
  case 4:
    value = mmu.Read<u32>(ea);
    break;
  case 8:
    value = mmu.Read<u64>(ea);
    break;
  default:
    ERROR_LOG_FMT(POWERPC, "StaticRecomp: external read of bad size {} at 0x{:08X}", size, ea);
    return 0;
  }
  if (core->m_lockstep_verifier->m_ls_journaling &&
      StaticRecompLockstep::LsHwAccessInScope(mmu, ea))
  {
    core->m_lockstep_verifier->m_journal.native_reads.push_back({ea, static_cast<u32>(value), size});
  }
  return value;
}

void StaticRecompCore::HookExternalWrite(CPUState* cpu, u32 ea, u64 value, u8 size)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ++g_static_recomp_ext_counts[1][ea >> 28];
  ea = core->TranslateRelAddress(ea);
  if (ea == 0)
    std::fprintf(stderr, "[zero-access] write size=%u guest_pc=%08x ppc_pc=%08x lr=%08x\n", size,
                 cpu->pc, core->m_system.GetPPCState().pc, cpu->lr);

  // Gather-pipe fast path: stores to the write-gather pipe page at effective
  // 0xCC008000 go straight to GPFifo, mirroring the MMU's masked-write
  // special case without an MMU round trip. Keying on the effective page is
  // the same shortcut Dolphin's JITs take (optimizeGatherPipe). GPFifo
  // maintains ppc_state.gather_pipe_ptr internally.
  //
  // FastWrite + FastCheckGatherPipe rather than Write*: Write* also calls
  // JitInterface::CompileExceptionCheck, which teaches the JIT to recompile
  // the storing block with an inline FIFO check. These stores come from the
  // static module, which that never applies to - and ppc_state.pc does not
  // name the store here, so every burst paid for an instruction fetch and
  // decode, and could invalidate an unrelated fallback block. The FIFO work
  // is unchanged; an 8-byte store flushes once instead of after each byte,
  // which leaves the same pipe and FIFO state.
  if ((ea & 0xFFFFF000) == 0xCC008000u)
  {
    if (core->m_lockstep_verifier->m_ls_journaling)
      core->m_lockstep_verifier->m_journal.native_mmio.push_back({ea, static_cast<u32>(value), size});
    auto& gpfifo = core->m_system.GetGPFifo();
    switch (size)
    {
    case 1:
      gpfifo.FastWrite8(static_cast<u8>(value));
      break;
    case 2:
      gpfifo.FastWrite16(static_cast<u16>(value));
      break;
    case 4:
      gpfifo.FastWrite32(static_cast<u32>(value));
      break;
    default:
      gpfifo.FastWrite64(value);
      break;
    }
    gpfifo.FastCheckGatherPipe();
    return;
  }

  core->PropagateGuestMSR();
  auto& mmu = core->m_system.GetMMU();
  if (core->m_lockstep_verifier->m_ls_journaling &&
      StaticRecompLockstep::LsHwAccessInScope(mmu, ea))
  {
    core->m_lockstep_verifier->m_journal.native_mmio.push_back({ea, static_cast<u32>(value), size});
  }
  switch (size)
  {
  case 1:
    mmu.Write<u8>(static_cast<u8>(value), ea);
    break;
  case 2:
    mmu.Write<u16>(static_cast<u16>(value), ea);
    break;
  case 4:
    mmu.Write<u32>(static_cast<u32>(value), ea);
    break;
  case 8:
    mmu.Write<u64>(value, ea);
    break;
  default:
    ERROR_LOG_FMT(POWERPC, "StaticRecomp: external write of bad size {} at 0x{:08X}", size, ea);
    break;
  }
}

u32 StaticRecompCore::HookExternalRead32(CPUState* cpu, u32 ea, u8 rid)
{
  // eciwx external-control read. EAR-enable and alignment were checked by the
  // generated helper; Dolphin's interpreter services the access as a plain
  // MMU read (the rid is carried in EAR only).
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ea = core->TranslateRelAddress(ea);
  core->PropagateGuestMSR();
  auto& mmu = core->m_system.GetMMU();
  const u32 value = mmu.Read<u32>(ea);
  if (core->m_lockstep_verifier->m_ls_journaling &&
      StaticRecompLockstep::LsHwAccessInScope(mmu, ea))
  {
    core->m_lockstep_verifier->m_journal.native_reads.push_back({ea, value, 4});
  }
  return value;
}

void StaticRecompCore::HookExternalWrite32(CPUState* cpu, u32 ea, u32 value, u8 rid)
{
  // ecowx external-control write; see HookExternalRead32.
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ea = core->TranslateRelAddress(ea);
  core->PropagateGuestMSR();
  auto& mmu = core->m_system.GetMMU();
  if (core->m_lockstep_verifier->m_ls_journaling &&
      StaticRecompLockstep::LsHwAccessInScope(mmu, ea))
  {
    core->m_lockstep_verifier->m_journal.native_mmio.push_back({ea, value, 4});
  }
  mmu.Write<u32>(value, ea);
}

void* StaticRecompCore::HookExternalPointer(CPUState* cpu, u32 ea, u32 size)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  // Gather-pipe query (size 0 on the WGPIPE page): hand the module
  // &ppc_state.gather_pipe_ptr, which is immediately followed by
  // gather_pipe_base_ptr (the JITs rely on the same layout), so generated
  // stores can do GPFifo::FastWrite themselves and only call out to flush.
  if (ea == 0xCC008000u && size == 0)
  {
    auto& ppc = core->m_system.GetPPCState();
    static_assert(offsetof(PowerPC::PowerPCState, gather_pipe_base_ptr) ==
                  offsetof(PowerPC::PowerPCState, gather_pipe_ptr) + sizeof(u8*));
    return &ppc.gather_pipe_ptr;
  }
  auto& memory = core->m_system.GetMemory();
  // Logical-view query (size 0 at 0xFFFFF000): the base of Dolphin's 4 GiB
  // fastmem view of effective addresses, built on first request. The MMU
  // keeps it in step with the DBATs from then on, the same way it does for
  // the JIT; pages no BAT maps stay reserved (an access there faults).
  // Not under lockstep: direct stores would bypass its write journal.
  if (ea == 0xFFFFF000u && size == 0)
  {
    if (core->m_lockstep_verifier->IsEnabled() || std::getenv("DOLRECOMP_NO_FASTMEM"))
      return nullptr;
    if (!memory.GetLogicalBase())
    {
      if (!memory.InitFastmemArena())
        return nullptr;
      core->m_system.GetMMU().DBATUpdated();
    }
    return memory.GetLogicalBase();
  }
  if (ea >= LOCKED_CACHE_BASE && size != 0 &&
      (ea - LOCKED_CACHE_BASE) + size <= memory.GetL1CacheSize())
  {
    return memory.GetL1Cache() + (ea - LOCKED_CACHE_BASE);
  }
  // Everything else stays on the per-access MMU hooks: this hook receives
  // *effective* addresses, and whether one maps to RAM depends on live
  // MSR/BAT state that only the MMU can answer. Handing out a raw pointer
  // here would bypass MMIO and translation. (Memory::GetPointerForRange was
  // considered and rejected for exactly that reason.)
  return nullptr;
}

u32 StaticRecompCore::HookSPRRead(CPUState* cpu, u16 spr, u32 cia)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  auto& ppc = core->m_system.GetPPCState();
  if (spr >= 1024)
  {
    ppc_program_exception(cpu, PPC_PROGRAM_ILLEGAL, cia);
    return 0;
  }

  switch (spr)
  {
  case SPR_DEC:
    if ((ppc.spr[SPR_DEC] & 0x80000000u) == 0)
      ppc.spr[SPR_DEC] = core->m_system.GetSystemTimers().GetFakeDecrementer();
    break;
  case SPR_WPAR:
    if (core->m_system.GetGPFifo().IsBNE())
      ppc.spr[SPR_WPAR] |= 1;
    else
      ppc.spr[SPR_WPAR] &= ~1u;
    break;
  case SPR_UPMC1:
    return ppc.spr[SPR_PMC1];
  case SPR_UPMC2:
    return ppc.spr[SPR_PMC2];
  case SPR_UPMC3:
    return ppc.spr[SPR_PMC3];
  case SPR_UPMC4:
    return ppc.spr[SPR_PMC4];
  case SPR_IABR:
    return ppc.spr[SPR_IABR] & ~1u;
  default:
    break;
  }
  return ppc.spr[spr];
}

void StaticRecompCore::HookSPRWrite(CPUState* cpu, u16 spr, u32 value, u32 cia)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  auto& system = core->m_system;
  auto& ppc = system.GetPPCState();
  if (spr >= 1024)
  {
    ppc_program_exception(cpu, PPC_PROGRAM_ILLEGAL, cia);
    return;
  }

  const u32 old_value = ppc.spr[spr];
  ppc.spr[spr] = value;

  switch (spr)
  {
  case SPR_TL_W:
    TL(ppc) = value;
    system.GetSystemTimers().TimeBaseSet();
    return;
  case SPR_TU_W:
    TU(ppc) = value;
    system.GetSystemTimers().TimeBaseSet();
    return;
  case SPR_PVR:
    ppc.spr[SPR_PVR] = old_value;
    return;
  case SPR_HID0:
  {
    if (HID0(ppc).ICFI)
    {
      HID0(ppc).ICFI = 0;
      ppc.iCache.Reset(system.GetJitInterface());
    }
    return;
  }
  case SPR_HID1:
    ppc.spr[SPR_HID1] &= 0xF8000000u;
    return;
  case SPR_HID4:
    if (old_value != value)
    {
      system.GetMMU().IBATUpdated();
      system.GetMMU().DBATUpdated();
    }
    return;
  case SPR_WPAR:
    system.GetGPFifo().ResetGatherPipe();
    return;
  case SPR_DMAL:
    if (DMAL(ppc).DMA_T)
    {
      const u32 mem_address = DMAU(ppc).MEM_ADDR << 5;
      const u32 cache_address = DMAL(ppc).LC_ADDR << 5;
      u32 length = (DMAU(ppc).DMA_LEN_U << 2) | DMAL(ppc).DMA_LEN_L;
      if (length == 0)
        length = 128;
      if (DMAL(ppc).DMA_LD)
        system.GetMMU().DMA_MemoryToLC(cache_address, mem_address, length);
      else
        system.GetMMU().DMA_LCToMemory(mem_address, cache_address, length);
    }
    DMAL(ppc).DMA_T = 0;
    return;
  case SPR_DEC:
    if ((old_value >> 31) == 0 && (value >> 31) != 0)
      ppc.Exceptions |= EXCEPTION_DECREMENTER;
    system.GetSystemTimers().DecrementerSet();
    return;
  case SPR_SDR:
    system.GetMMU().SDRUpdated();
    return;
  case SPR_MMCR0:
  case SPR_MMCR1:
    PowerPC::MMCRUpdated(ppc);
    return;
  case SPR_THRM1:
  case SPR_THRM2:
  case SPR_THRM3:
  {
    const auto update = [&ppc](UReg_THRM12* reg) {
      if (!THRM3(ppc).E || !reg->V)
      {
        reg->TIV = 0;
      }
      else
      {
        reg->TIV = 1;
        reg->TIN = reg->TID ? 42 < reg->THRESHOLD : 42 > reg->THRESHOLD;
      }
    };
    update(&THRM1(ppc));
    update(&THRM2(ppc));
    return;
  }
  default:
    break;
  }

  const bool ibat = (spr >= SPR_IBAT0U && spr <= SPR_IBAT3L) ||
                    (spr >= SPR_IBAT4U && spr <= SPR_IBAT7L);
  const bool dbat = (spr >= SPR_DBAT0U && spr <= SPR_DBAT3L) ||
                    (spr >= SPR_DBAT4U && spr <= SPR_DBAT7L);
  if (old_value != value && ibat)
    system.GetMMU().IBATUpdated();
  else if (old_value != value && dbat)
    system.GetMMU().DBATUpdated();
}


void StaticRecompCore::HandleCacheRange(CPUState* cpu, u8 operation, u32 start, u32 bytes)
{
  if (!bytes)
    return;
  start = TranslateRelAddress(start);
  PropagateGuestMSR();
  auto& ppc = m_system.GetPPCState();
  const u32 count = (bytes + 31u) / 32u;

  if (operation == PPC_CACHE_ICBI)
  {
    auto& memory = m_system.GetMemory();
    auto& jit = m_system.GetJitInterface();
    for (u32 i = 0; i < count; ++i)
      ppc.iCache.Invalidate(memory, jit, start + i * 32u);
    return;
  }

  if (!ppc.m_enable_dcache)
  {
    // The batched form. This is the whole point of the range hook.
    m_system.GetJitInterface().InvalidateICacheLines(start, count);
    return;
  }

  // With dcache emulation on there is no range form, so fall back to per line,
  // which is still no worse than the guest loop would have been.
  auto& mmu = m_system.GetMMU();
  for (u32 i = 0; i < count; ++i)
  {
    const u32 ea = start + i * 32u;
    switch (operation)
    {
    case PPC_CACHE_DCBST:
      mmu.StoreDCacheLine(ea);
      break;
    case PPC_CACHE_DCBF:
      mmu.FlushDCacheLine(ea);
      break;
    case PPC_CACHE_DCBI:
      mmu.InvalidateDCacheLine(ea);
      break;
    default:
      break;
    }
  }
}

void StaticRecompCore::HookCacheControl(CPUState* cpu, u8 operation, u32 ea, u32 cia)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ea = core->TranslateRelAddress(ea);
  core->PropagateGuestMSR();
  auto& ppc = core->m_system.GetPPCState();
  auto& mmu = core->m_system.GetMMU();

  if (operation == PPC_CACHE_ICBI)
  {
    ppc.iCache.Invalidate(core->m_system.GetMemory(), core->m_system.GetJitInterface(), ea);
    return;
  }

  if (!ppc.m_enable_dcache)
  {
    core->m_system.GetJitInterface().InvalidateICacheLine(ea);
    return;
  }

  switch (operation)
  {
  case PPC_CACHE_DCBST:
    mmu.StoreDCacheLine(ea);
    break;
  case PPC_CACHE_DCBF:
    mmu.FlushDCacheLine(ea);
    break;
  case PPC_CACHE_DCBI:
    mmu.InvalidateDCacheLine(ea);
    break;
  default:
    ppc_program_exception(cpu, PPC_PROGRAM_ILLEGAL, cia);
    break;
  }
}

void StaticRecompCore::HookInstructionFallback(CPUState* cpu, u32 raw, u32 cia)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  cia = core->TranslateRelAddress(cia);
  ++core->m_hook_fallback_instructions;

  // Lockstep: a block that fell back to the interpreter for an unmodeled
  // instruction (DMA mtspr, cache op, ...) performed side effects not captured
  // by the RAM journal / MMIO hooks, so re-running it on the shadow would
  // double-issue them. Mark it unsafe to differentially check.
  if (core->m_lockstep_verifier->m_ls_journaling)
    core->m_lockstep_verifier->m_ls_fallback_seen = true;

  auto& system = core->m_system;
  auto& ppc = system.GetPPCState();

  // Fast path for dcbf/dcbst/dcbi/icbi: streaming code flushes caches in
  // 32-byte loops (thousands per frame), and these ops read two GPRs and
  // change no CPU state, so they run straight off ctx without the full
  // SyncOut/interpreter/SyncIn round trip. This mirrors Dolphin's
  // interpreter with dcache emulation off: every one funnels into
  // InvalidateICacheLine (keeping the SMC guard exact). dcbi's PR!=0
  // privilege trap and dcache-on configs take the slow path.
  if ((raw >> 26) == 31u && !ppc.m_enable_dcache)
  {
    const u32 xo = (raw >> 1) & 0x3FFu;
    if (xo == 86u || xo == 54u || xo == 982u || (xo == 470u && (cpu->msr & 0x4000u) == 0))
    {
      const u32 ra = (raw >> 16) & 31u;
      const u32 rb = (raw >> 11) & 31u;
      const u32 ea = (ra ? cpu->gpr[ra] : 0u) + cpu->gpr[rb];
      if (xo == 982u)
        ppc.iCache.Invalidate(system.GetMemory(), system.GetJitInterface(), ea);
      else
        system.GetJitInterface().InvalidateICacheLine(ea);
      // These bypass SingleStepInner, so charge Dolphin's PPCTables cost
      // here (icbi 4, dcbf/dcbst/dcbi 5); their emitted block cost is zero.
      ppc.downcount -= (xo == 982u) ? 4 : 5;
      cpu->pc = cia + 4u;
      return;
    }
  }

  // The recompiled segment resumes via the dispatcher at the PC this leaves
  // behind, so this must execute exactly the instruction at cia via
  // Dolphin's interpreter and hand the register state back.
  core->SyncOut();
  ppc.pc = cia;
  ppc.npc = cia + 4;
  ppc.downcount -= system.GetInterpreter().SingleStepInner();
  core->SyncIn();
}
