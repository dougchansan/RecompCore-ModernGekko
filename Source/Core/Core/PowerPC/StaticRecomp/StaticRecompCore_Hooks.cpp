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
#include "Common/ChunkFile.h"
#include "Common/Logging/Log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace
{
constexpr u32 LOCKED_CACHE_BASE = 0xE0000000u;

// --- MKDD vehicle-scaling extension window -------------------------------
//
// Per-kart arrays are sized 8 in the retail image and cannot be grown in
// place, because the recompiled code reaches every field at a baked-in
// offset. The generated module is rewritten so karts >= 8 address a separate
// window instead (see kart_ext.h). Nothing maps that window: MEM2 is only
// allocated for Wii titles, so these accesses fall through to the external
// hooks below.
//
// Backing it here keeps the change off Dolphin's memory model entirely. Only
// karts >= 8 ever reach this path, so the indirect call costs nothing for an
// unmodified 8-kart race. Storage is big-endian to match guest semantics, so
// a 32-bit write read back as two 16-bit halves behaves as on hardware.
constexpr u32 KART_EXT_BASE = 0x90000000u;
// Stable chassis capacity. Modules negotiate a smaller live layout, so counts
// 9 through 16 use the same ModernGekko executable.
constexpr u32 KART_EXT_SIZE = 0x100000u;

// KART_EXT_SIZE is the chassis capacity. The live range is negotiated from
// the generated module's kart_ext_abi descriptor below, so a module built for
// 9 through 16 karts never depend on this translation unit being rebuilt with a
// matching KART_EXTENDED_COUNT.
u32 g_kart_ext_base = KART_EXT_BASE;
u32 g_kart_ext_live_size = KART_EXT_SIZE;
u32 g_kart_ext_count = 9u;

constexpr u32 KART_EXT_ABI_MAGIC = 0x4B455854u;
constexpr u32 KART_EXT_ABI_VERSION = 2u;
constexpr u32 KART_EXT_ORIGINAL_COUNT = 8u;
constexpr u32 KART_EXT_MAX_COUNT = 16u;
struct KartExtAbiHost
{
  u32 magic;
  u32 abi_version;
  u32 struct_size;
  u32 guest_base;
  u32 host_size;
  u32 layout_size;
  u32 original_count;
  u32 extended_count;
};

void KartExtConfigure(const Common::DynamicLibrary& library)
{
  // Defaults preserve modules produced before the descriptor was introduced;
  // a descriptor, when present, is authoritative and is validated before any
  // host hook accepts an extension address.
  g_kart_ext_base = KART_EXT_BASE;
  g_kart_ext_live_size = KART_EXT_SIZE;
  g_kart_ext_count = 9u;
  if (!library.IsOpen())
    return;

  const auto* abi = reinterpret_cast<const KartExtAbiHost*>(
      library.GetSymbolAddress("kart_ext_abi"));
  if (!abi)
    return;
  const bool valid = abi->magic == KART_EXT_ABI_MAGIC &&
                     abi->abi_version == KART_EXT_ABI_VERSION &&
                     abi->struct_size >= sizeof(KartExtAbiHost) &&
                     abi->guest_base != 0 &&
                     abi->host_size <= KART_EXT_SIZE &&
                     abi->layout_size != 0 &&
                     abi->layout_size <= abi->host_size &&
                     abi->original_count == KART_EXT_ORIGINAL_COUNT &&
                     abi->extended_count >= abi->original_count &&
                     abi->extended_count <= KART_EXT_MAX_COUNT &&
                     abi->guest_base <= 0xffffffffu - abi->host_size;
  if (!valid)
  {
    // Fail closed for a module that advertises an incompatible contract. Do
    // not reinterpret its addresses using the legacy default layout.
    g_kart_ext_live_size = 0;
    return;
  }
  g_kart_ext_base = abi->guest_base;
  g_kart_ext_live_size = abi->layout_size;
  g_kart_ext_count = abi->extended_count;
}

u8 g_kart_ext[KART_EXT_SIZE]{};           // zero-initialised == NULL pointers

// Diagnostics. Zero traffic is ambiguous on its own -- it means either "karts
// 8+ were never created" or "the rewrite never reached the window" -- so
// record the high-water offset too, which bounds how far into the extension
// the run actually got. Reported once at exit.
u64 g_kart_ext_reads = 0;
u64 g_kart_ext_writes = 0;
// A NULL store counts the same as a real one, so split the tally:
// writes prove the constructor ran, not that it built anything.
u64 g_kart_ext_writes_null = 0;
// Which window offsets receive a NULL pointer. A NULL stored into the
// extension means that member's construction produced nothing for karts 8+,
// and the offset names the member.
constexpr int KART_NULL_MAX = 64;
u32 g_kart_null_off[KART_NULL_MAX]{};
u32 g_kart_null_pc[KART_NULL_MAX]{};
int g_kart_null_n = 0;
u32 g_kart_ext_high = 0;

// Which guest PCs touch the window, and at what offset. Knowing that a read
// returned zero is not actionable; knowing *which site* read it identifies the
// member whose construction store is missing. Bounded so a hot loop cannot
// flood the log.
constexpr int KART_EXT_TRACE_MAX = 96;
u32 g_kart_ext_pc[KART_EXT_TRACE_MAX]{};
u32 g_kart_ext_pc_off[KART_EXT_TRACE_MAX]{};
int g_kart_ext_pc_n = 0;

void KartExtNote(u32 pc, u32 off)
{
  for (int i = 0; i < g_kart_ext_pc_n; ++i)
    if (g_kart_ext_pc[i] == pc)
      return;
  if (g_kart_ext_pc_n < KART_EXT_TRACE_MAX)
  {
    g_kart_ext_pc[g_kart_ext_pc_n] = pc;
    g_kart_ext_pc_off[g_kart_ext_pc_n] = off;
    ++g_kart_ext_pc_n;
  }
}

struct KartExtReporter
{
  ~KartExtReporter()
  {
    std::fprintf(stderr,
                 "[kart_ext] reads=%llu writes=%llu (of which NULL: %llu) high_offset=0x%x\n",
                 static_cast<unsigned long long>(g_kart_ext_reads),
                 static_cast<unsigned long long>(g_kart_ext_writes),
                 static_cast<unsigned long long>(g_kart_ext_writes_null), g_kart_ext_high);
    for (int i = 0; i < g_kart_ext_pc_n; ++i)
      std::fprintf(stderr, "[kart_ext] site pc=%08x off=0x%x\n", g_kart_ext_pc[i],
                   g_kart_ext_pc_off[i]);
    for (int i = 0; i < g_kart_null_n; ++i)
      std::fprintf(stderr, "[kart_ext] NULL stored at off=0x%x by pc=%08x\n",
                   g_kart_null_off[i], g_kart_null_pc[i]);
    // Site records dedupe by PC, so a region written by code that already
    // appeared under a different offset leaves no trace in the list above.
    // The window's own contents are unambiguous: dump them. Zero rows are
    // reported as a range so a wholly-unpopulated member is still visible.
    std::fprintf(stderr, "[kart_ext] window dump 0x000..0x540\n");
    u32 zero_from = 0xffffffffu;
    for (u32 off = 0; off < 0x540u; off += 16)
    {
      bool row_zero = true;
      for (u32 i = 0; i < 16; ++i)
        if (g_kart_ext[off + i] != 0)
          row_zero = false;
      if (row_zero)
      {
        if (zero_from == 0xffffffffu)
          zero_from = off;
        continue;
      }
      if (zero_from != 0xffffffffu)
      {
        std::fprintf(stderr, "[kart_ext]   0x%03x..0x%03x  zero\n", zero_from, off - 1);
        zero_from = 0xffffffffu;
      }
      std::fprintf(stderr, "[kart_ext]   0x%03x  ", off);
      for (u32 i = 0; i < 16; i += 4)
        std::fprintf(stderr, "%02x%02x%02x%02x ", g_kart_ext[off + i], g_kart_ext[off + i + 1],
                     g_kart_ext[off + i + 2], g_kart_ext[off + i + 3]);
      std::fprintf(stderr, "\n");
    }
    if (zero_from != 0xffffffffu)
      std::fprintf(stderr, "[kart_ext]   0x%03x..0x53f  zero\n", zero_from);
  }
};
KartExtReporter g_kart_ext_reporter;

bool KartExtContains(u32 ea, u8 size)
{
  if (g_kart_ext_live_size == 0 || ea < g_kart_ext_base)
    return false;
  const u64 offset = static_cast<u64>(ea) - g_kart_ext_base;
  return offset <= g_kart_ext_live_size &&
         static_cast<u64>(size) <= g_kart_ext_live_size - offset;
}

u64 KartExtRead(u32 ea, u8 size, u32 pc)
{
  const u32 off = ea - g_kart_ext_base;
  u64 value = 0;
  for (u8 i = 0; i < size; ++i)
    value = (value << 8) | g_kart_ext[off + i];
  ++g_kart_ext_reads;
  if (off + size > g_kart_ext_high)
    g_kart_ext_high = off + size;
  KartExtNote(pc, off);
  return value;
}

void KartExtWrite(u32 ea, u64 value, u8 size, u32 pc)
{
  const u32 off = ea - g_kart_ext_base;
  for (u8 i = 0; i < size; ++i)
    g_kart_ext[off + i] = static_cast<u8>(value >> ((size - 1 - i) * 8));
  ++g_kart_ext_writes;
  if (size == 4 && value == 0)
  {
    ++g_kart_ext_writes_null;
    bool seen = false;
    for (int i = 0; i < g_kart_null_n; ++i)
      if (g_kart_null_off[i] == off)
        seen = true;
    if (!seen && g_kart_null_n < KART_NULL_MAX)
    {
      g_kart_null_off[g_kart_null_n] = off;
      g_kart_null_pc[g_kart_null_n] = pc;
      ++g_kart_null_n;
    }
  }
  if (off + size > g_kart_ext_high)
    g_kart_ext_high = off + size;
  KartExtNote(pc, off);
}

// --- on-demand mKartInfo seed for karts 8+ ---------------------------------
//
// These mirror the module's kart_ext.h. The count is read from kart_ext_abi at
// runtime; the host binary therefore works with all supported module layouts.
constexpr u32 KART_ORIGINAL_KARTS = KART_EXT_ORIGINAL_COUNT;
constexpr u32 KART_EXT_MAX_EXTRA_KARTS = KART_EXT_MAX_COUNT - KART_ORIGINAL_KARTS;
constexpr u32 KartExtRegion(u32 stride, u32 extra)
{
  return (stride * extra + 0x1fu) & ~0x1fu;
}
constexpr u32 KARTINFO_STRIDE = 0x18u; // sizeof(KartInfo)
constexpr u32 KARTINFO_PREFIX_STRIDES[] = {
  4u, 8u, 4u, 4u, 4u, 16u, 8u, 4u, 4u, 4u, 4u, 4u, 4u, 4u,
};

u32 KartInfoOffset(u32 count)
{
  const u32 extra = count - KART_ORIGINAL_KARTS;
  u32 offset = 0;
  for (const u32 stride : KARTINFO_PREFIX_STRIDES)
    offset += KartExtRegion(stride, extra);
  return offset;
}
// Seed from the LAST original kart, not kart 0. Kart 0 is the player, and a
// KartInfo copied from it carries real gamepads -- so the rival-pool formula
// (mKartNum - humans - consoles, at 0x80247138) counted kart 8 as a human and
// handed out one fewer entry, while the rival-creation path still built a
// RivalSpeedCtrl for it. Measured at 9 karts: pool=5, six rivals, the sixth
// found an empty JSUPtrList and dereferenced NULL. Kart 7 is CPU-controlled in
// a 1P race, so copying it keeps both sides of that arithmetic agreeing.
constexpr u32 KARTINFO_SEED_SRC_INDEX = 7;
constexpr u32 RACEINFO_KARTINFO0 =
    0x803B145Cu + 0x30u + KARTINFO_SEED_SRC_INDEX * 0x18u;

bool g_kart_seed_on = std::getenv("KART_SEED") != nullptr;
u64 g_kart_seed_fills = 0;
bool g_kart_seeded[KART_EXT_MAX_EXTRA_KARTS]{};

// Enabled with KART_SEED=1, so the black screen can be compared with and
// without it in otherwise identical unattended runs.
#define KART_EXT_SEED_ENABLE (g_kart_seed_on)

void KartExtSeedKartInfoOnDemand(CPUState* cpu, u32 ea, u8 size)
{
  if (g_kart_ext_count < KART_ORIGINAL_KARTS ||
      g_kart_ext_count > KART_EXT_MAX_COUNT)
    return;
  const u32 extra = g_kart_ext_count - KART_ORIGINAL_KARTS;
  const u32 kartinfo_off = KartInfoOffset(g_kart_ext_count);
  const u32 off = ea - g_kart_ext_base;
  if (off + size <= kartinfo_off ||
      off >= kartinfo_off + KARTINFO_STRIDE * extra)
    return;

  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  auto& mmu = core->m_system.GetMMU();

  for (u32 k = 0; k < extra; ++k)
  {
    const u32 dst = kartinfo_off + k * KARTINFO_STRIDE;
    // An extra slot can receive a partial placeholder through
    // RaceInfo::setKart before its first read.  "Any nonzero byte" is not
    // proof that all character, kart, and pad fields are initialized.  When
    // KART_SEED is explicitly enabled, seed exactly once from a complete CPU
    // kart regardless of partial destination contents.
    if (g_kart_seeded[k])
      continue;

    // Only copy a source that actually holds something. Before character
    // select mKartInfo[0] is still zeroed, and latching that in would recreate
    // the very bug this works around.
    u32 words[KARTINFO_STRIDE / 4];
    bool src_live = false;
    for (u32 i = 0; i < KARTINFO_STRIDE / 4; ++i)
    {
      words[i] = mmu.Read<u32>(RACEINFO_KARTINFO0 + i * 4);
      if (words[i] != 0)
        src_live = true;
    }
    if (!src_live)
      continue;

    for (u32 i = 0; i < KARTINFO_STRIDE / 4; ++i)
      for (u32 b = 0; b < 4; ++b)
        g_kart_ext[dst + i * 4 + b] = static_cast<u8>(words[i] >> ((3 - b) * 8));
    g_kart_seeded[k] = true;
    ++g_kart_seed_fills;
  }
}
}

// --- MKDD kart construction write-watch -----------------------------------
//
// Static analysis cannot find what populates KartCtrl::mKartBodies[8]: the
// scanner is rooted at the singleton, and 1076 loads against 5 stores says the
// writer never reaches the array through it. So watch the address range at
// runtime instead and report the PC that writes it.
//
// Module writes to ordinary RAM do NOT pass through the external hooks --
// mem_write32() is ALWAYS_INLINE in the module and stores straight into the
// RAM buffer. The one seam is g_mem_write_journal, which the module already
// exports a setter for (ppc_set_mem_write_journal). That callback gets a RAM
// offset and a size but no PC, hence `user` is the core: CPUState m_guest
// carries the pc.
//
// Enable with KART_WATCH=1. Off by default: this fires on EVERY guest write.
// It also takes the journal slot that lockstep uses, so it refuses to install
// when STATICRECOMP_LOCKSTEP is on rather than silently fighting over it.
namespace
{
// Watch target is configurable so the same instrument can follow the chain
// upstream: KartCtrl's constructor copies from RaceMgr's arrays, so when a
// KartCtrl slot lands NULL the question moves to who fills RaceMgr.
//   KART_WATCH_PTR    guest address of the singleton POINTER (default KartCtrl)
//   KART_WATCH_BASE   direct guest object address (for static objects such as gRaceInfo)
//   KART_WATCH_OFF    member offset within that object
//   KART_WATCH_LEN    bytes to watch
u32 g_watch_ptr_addr = 0x803CC588u;              // KartCtrl
u32 KART_WATCH_MEMBER = 0xA0u;                   // mKartBodies[8]
u32 KART_WATCH_LEN = 0x20u;                      // 8 * 4 bytes
constexpr u32 RAM_MASK = 0x01FFFFFFu;            // MEM1 guest -> RAM offset

bool g_kart_watch_on = false;
u32 g_kart_watch_ctrl = 0;      // cached KartCtrl pointer
// Highest KartCtrl::mKartCount ever observed. If the override works this
// reaches KART_EXTENDED_COUNT; if it stays 8, nothing downstream can ever
// iterate far enough to touch karts 8+, whatever the addressing does.
u32 g_kart_count_max = 0;
bool g_kart_dumped = false;   // one-shot per-kart array dump (KART_DUMP=1)
// Cached: this is tested from the per-write journal callback, so a getenv()
// here costs an environment scan on EVERY guest write. Leaving it inline cost
// 25x (0.04x speed) once the dump was made to fire late instead of instantly.
const bool g_kart_dump_on = std::getenv("KART_DUMP") != nullptr;
// Writes to wait for before dumping. KART_DUMP_AFTER overrides.
u64 g_kart_dump_after_writes = std::getenv("KART_DUMP_AFTER")
                                   ? std::strtoull(std::getenv("KART_DUMP_AFTER"), nullptr, 0)
                                   : 3000000000ull;
bool g_threads_dumped = false; // one-shot OS thread dump (KART_THREADS=1)
struct KartBody36Event { u32 kind, pc, lr, ea, value, r3, r4, r5, r29, r30, r31; };
KartBody36Event g_body36_events[32]{};
u32 g_body36_count = 0;

// First invalid-looking effective addresses reached through the generated
// module's external-memory seam. Keep this silent in the hot path: printing
// every bad access changes timing drastically and can produce multi-gigabyte
// logs before shutdown. Enable with KART_INVALID_TRACE=1.
struct KartInvalidEvent
{
  u32 kind, ea, size, pc, lr, ctr, cr;
  u32 r1, r3, r4, r5, r6, r7, r8, r28, r29, r30, r31;
};
constexpr u32 KART_INVALID_EVENT_COUNT = 64;
KartInvalidEvent g_invalid_events[KART_INVALID_EVENT_COUNT]{};
u64 g_invalid_count = 0;

void KartRecordInvalid(CPUState* cpu, u32 ea, u8 size, u32 kind)
{
  static const bool on = std::getenv("KART_INVALID_TRACE") != nullptr;
  if (!on || !((ea < 0x01000000u) || (ea >= 0x3F000000u && ea < 0x40000000u)))
    return;
  const u64 ordinal = g_invalid_count++;
  if (ordinal >= KART_INVALID_EVENT_COUNT)
    return;
  g_invalid_events[ordinal] = {kind, ea, size, cpu->pc, cpu->lr, cpu->ctr, cpu->cr,
                               cpu->gpr[1], cpu->gpr[3], cpu->gpr[4], cpu->gpr[5],
                               cpu->gpr[6], cpu->gpr[7], cpu->gpr[8], cpu->gpr[28],
                               cpu->gpr[29], cpu->gpr[30], cpu->gpr[31]};
  std::fprintf(stderr,
               "[invalid-now] #%llu %c%u ea=%08x pc=%08x lr=%08x ctr=%08x "
               "r1=%08x r3=%08x r4=%08x r5=%08x r6=%08x r7=%08x r8=%08x "
               "r28=%08x r29=%08x r30=%08x r31=%08x\n",
               ordinal, kind ? 'W' : 'R', size, ea, cpu->pc, cpu->lr, cpu->ctr,
               cpu->gpr[1], cpu->gpr[3], cpu->gpr[4], cpu->gpr[5], cpu->gpr[6],
               cpu->gpr[7], cpu->gpr[8], cpu->gpr[28], cpu->gpr[29], cpu->gpr[30],
               cpu->gpr[31]);
}

// GameCube OS thread structures. SelectThread showing up hot means threads are
// parked, and a profile cannot say what they are parked ON. These can.
//   OSThread: context 0x000 (srr0 at +0x198), state 0x2C8, suspend 0x2CC,
//             priority 0x2D0, queue 0x2DC, link.next 0x2E0, mutex 0x2F0
//   OSThreadQueue: { head, tail }
constexpr u32 OS_CURR_THREAD    = 0x800000E4u;
constexpr u32 DVD_THREAD_QUEUE  = 0x803CAFE8u;   // __DVDThreadQueue
constexpr u32 RUN_QUEUE         = 0x80375678u;   // RunQueue[32]
constexpr u32 OST_SRR0 = 0x198u, OST_STATE = 0x2C8u, OST_SUSPEND = 0x2CCu;
constexpr u32 OST_PRIO = 0x2D0u, OST_QUEUE = 0x2DCu, OST_NEXT = 0x2E0u;
constexpr u32 OST_MUTEX = 0x2F0u;

const char* OSThreadState(u16 st)
{
  switch (st)
  {
  case 1: return "READY";
  case 2: return "RUNNING";
  case 4: return "WAITING";
  case 8: return "MORIBUND";
  default: return "?";
  }
}

// Coarse PC profile. Three hypotheses about WHY the game hangs have each been
// rejected by measurement; this asks the blunter question of WHERE, by
// sampling the guest pc as writes go by. A hang concentrates the samples.
constexpr int KART_PROF_N = 64;
u32 g_prof_pc[KART_PROF_N]{};
u32 g_prof_lr[KART_PROF_N]{};
u32 g_prof_hits[KART_PROF_N]{};
int g_prof_used = 0;

// Sample the link register with the pc. Knowing the code spins in J3DStruct
// says nothing about who is driving it; the caller does. Bucketed on the pair
// so one row means "this loop, entered from here".
void KartProfSample(u32 pc, u32 lr)
{
  const u32 bucket = pc & ~0x3Fu;
  for (int i = 0; i < g_prof_used; ++i)
    if (g_prof_pc[i] == bucket && g_prof_lr[i] == lr) { ++g_prof_hits[i]; return; }
  if (g_prof_used < KART_PROF_N)
  {
    g_prof_pc[g_prof_used] = bucket;
    g_prof_lr[g_prof_used] = lr;
    g_prof_hits[g_prof_used] = 1;
    ++g_prof_used;
  }
}
// Module-side total of kart_ea/kart_ea_x calls, including the karts 0-7 case
// that never reaches this file. Resolved out of the module at init.
const volatile unsigned long long* g_kart_ea_calls = nullptr;
// Sampled copy. The reporter is a static destructor, which runs AFTER the
// module is unloaded, so reading through the pointer there dereferences a
// dead mapping. Snapshot it while the module is still live instead.
unsigned long long g_kart_ea_calls_last = 0;
}  // namespace

// The extension window is process memory, not emulated RAM, so no other
// DoState in the chain carries it. Without this a savestate restores MEM1
// faithfully and leaves every extended kart's sidecar at whatever the loading
// process happens to hold -- all zeroes in a fresh one -- so karts 8..15 come
// back with NULL KartCheckers, NULL panes and zeroed item tables while the
// guest heap still believes they are racing. Savestate-based investigation of
// any 9-through-16 kart defect is worthless until this exists.
//
// The full chassis is written rather than the negotiated live size, so the
// stream layout does not depend on which module produced the state. It is
// almost all zeroes and compresses accordingly.
void KartExtDoState(PointerWrap& p)
{
  // The window's meaning depends on the layout that produced it. A state
  // saved under a different kart count describes a different sidecar, and its
  // bytes cannot be reinterpreted here.
  u32 base = g_kart_ext_base;
  u32 live = g_kart_ext_live_size;
  u32 count = g_kart_ext_count;
  p.Do(base);
  p.Do(live);
  p.Do(count);

  const bool matches = base == g_kart_ext_base && live == g_kart_ext_live_size &&
                       count == g_kart_ext_count;

  // Consume the payload either way: the stream must stay aligned for whatever
  // follows, whether or not the bytes are usable.
  p.DoArray(g_kart_ext);

  // Self-evidence. A restored window cannot be demonstrated by dumping bytes
  // afterwards, because a live race repopulates it within a frame of loading;
  // the count has to be taken here, at the moment of transfer.
  if (p.IsReadMode() || p.IsWriteMode())
  {
    u32 nonzero = 0;
    for (u32 i = 0; i < g_kart_ext_live_size && i < sizeof(g_kart_ext); ++i)
      if (g_kart_ext[i] != 0)
        ++nonzero;
    std::fprintf(stderr, "[kart_ext] savestate %s: %u non-zero of %u live\n",
                 p.IsReadMode() ? "restored" : "saved", nonzero,
                 g_kart_ext_live_size);
  }

  if (!matches && p.IsReadMode())
  {
    // Loud and defined beats silently wrong. A zeroed window is exactly the
    // state a fresh process starts in, so the failure mode is the familiar
    // one rather than a plausible-looking mixture of two layouts.
    std::memset(g_kart_ext, 0, sizeof(g_kart_ext));
    std::fprintf(stderr,
                 "[kart_ext] savestate layout mismatch: state base=%08x "
                 "live=%u count=%u, running base=%08x live=%u count=%u; "
                 "extension window cleared\n",
                 base, live, count, g_kart_ext_base, g_kart_ext_live_size,
                 g_kart_ext_count);
  }
}

// Sampled from the core's shutdown path, where the module is still mapped. A
// static destructor is too late: the dylib is unloaded by then and the read
// segfaults, truncating the report just before the count is printed.
void KartMaybeDumpThreads(StaticRecompCore* core);

void KartSampleEaCalls()
{
  if (g_kart_ea_calls)
    g_kart_ea_calls_last = *g_kart_ea_calls;
}

namespace
{
u64 g_kart_watch_calls = 0;
int g_kart_watch_hits = 0;

struct KartWatchHit
{
  u32 pc, offset, size;
};
constexpr int KART_WATCH_MAX = 32;
KartWatchHit g_kart_watch_hit[KART_WATCH_MAX]{};

struct KartItemStateHit
{
  u32 pc, lr, ea, size, value;
  u32 r0, r3, r4, r5, r6, r7, r8, r28, r29, r30, r31;
};
constexpr u32 KART_ITEM_STATE_MAX = 512;
KartItemStateHit g_item_state_hit[KART_ITEM_STATE_MAX]{};
u64 g_item_state_count = 0;

struct KartWatchReporter
{
  ~KartWatchReporter()
  {
    if (!g_kart_watch_on)
      return;
    if (g_prof_used)
    {
      // Simple selection of the hottest buckets; the list is tiny.
      for (int r = 0; r < 8 && r < g_prof_used; ++r)
      {
        int best = r;
        for (int i = r + 1; i < g_prof_used; ++i)
          if (g_prof_hits[i] > g_prof_hits[best]) best = i;
        std::swap(g_prof_pc[r], g_prof_pc[best]);
        std::swap(g_prof_hits[r], g_prof_hits[best]);
        std::swap(g_prof_lr[r], g_prof_lr[best]);
        std::fprintf(stderr, "[kart_prof] pc~%08x lr=%08x  samples=%u\n",
                     g_prof_pc[r], g_prof_lr[r], g_prof_hits[r]);
      }
    }
    std::fprintf(stderr, "[kart_watch] max mKartCount seen=%u\n", g_kart_count_max);
    // NOTE: do NOT dereference g_kart_ea_calls here. This runs from a static
    // destructor, by which point the module dylib is unloaded, and the read
    // segfaults -- truncating the report immediately before this line.
    // KartSampleEaCalls() is called from the core's shutdown path instead,
    // while the module is still mapped.
    std::fprintf(stderr, "[kart_watch] kart_ea calls (all karts)=%llu%s\n",
                 g_kart_ea_calls_last, g_kart_ea_calls ? "" : " (symbol not found)");
    std::fprintf(stderr, "[kart_watch] writes seen=%llu, KartCtrl=%08x, hits=%d\n",
                 static_cast<unsigned long long>(g_kart_watch_calls), g_kart_watch_ctrl,
                 g_kart_watch_hits);
    for (int i = 0; i < g_kart_watch_hits && i < KART_WATCH_MAX; ++i)
    {
      const u32 slot = (g_kart_watch_hit[i].offset - KART_WATCH_MEMBER) / 4u;
      std::fprintf(stderr, "[kart_watch] pc=%08x wrote slot[%u] (off=0x%x size=%u)\n",
                   g_kart_watch_hit[i].pc, slot, g_kart_watch_hit[i].offset,
                   g_kart_watch_hit[i].size);
    }
    for (u32 i = 0; i < g_item_state_count && i < KART_ITEM_STATE_MAX; ++i)
    {
      const auto& e = g_item_state_hit[i];
      std::fprintf(stderr,
                   "[item-state] n=%u pc=%08x lr=%08x ea=%08x size=%u value=%08x "
                   "r0=%08x r3=%08x r4=%08x r5=%08x r6=%08x r7=%08x r8=%08x "
                   "r28=%08x r29=%08x r30=%08x r31=%08x\n",
                   i, e.pc, e.lr, e.ea, e.size, e.value, e.r0, e.r3, e.r4,
                   e.r5, e.r6, e.r7, e.r8, e.r28, e.r29, e.r30, e.r31);
    }
    if (g_item_state_count)
      std::fprintf(stderr, "[item-state-summary] writes=%llu stored=%u\n",
                   static_cast<unsigned long long>(g_item_state_count),
                   static_cast<u32>(std::min<u64>(g_item_state_count, KART_ITEM_STATE_MAX)));
  }
};
KartWatchReporter g_kart_watch_reporter;

}  // namespace

void StaticRecompCore::KartWatchJournal(u32 vmem_offset, u32 size, void* user)
{
  ++g_kart_watch_calls;
  auto* core = static_cast<StaticRecompCore*>(user);

  // Sample before any early return: the hung state has KartCtrl == NULL, and
  // the return below would skip exactly the case worth profiling.
  if ((g_kart_watch_calls & 0x3FFu) == 0)
    KartProfSample(core->m_guest.pc, core->m_guest.lr);

  KartMaybeDumpThreads(core);
  KartWatchJournal2(vmem_offset, size, user);
}

// Thread dump, callable from anywhere with a live core.
//
// It used to live inline in the journal above, which made it useless twice
// over: KART_WATCH_LIGHT never installs the journal, so it never ran at all;
// and it was one-shot on the FIRST guest write, so even in full mode it
// photographed the boot sequence rather than the hang. It is now called from
// the core's shutdown path, when the game is actually wedged.
void KartMaybeDumpThreads(StaticRecompCore* core)
{
  static const bool on = std::getenv("KART_THREADS") != nullptr;
  if (!on || g_threads_dumped)
    return;
  {
    g_threads_dumped = true;
    auto& mmu = core->m_system.GetMMU();
    auto ok = [](u32 p) { return p >= 0x80000000u && p < 0x81800000u; };
    auto thread_line = [&](const char* tag, u32 t) {
      if (!ok(t)) return;
      const u32 saved_r1 = mmu.Read<u32>(t + 0x04u);
      const u32 saved_caller_lr = ok(saved_r1) ? mmu.Read<u32>(saved_r1 + 36u) : 0;
      std::fprintf(stderr,
                   "[threads] %-10s %08x state=%-8s suspend=%d prio=%d "
                   "base=%d queue=%08x mutex=%08x srr0=%08x lr=%08x "
                   "r1=%08x r3=%08x r4=%08x r5=%08x r6=%08x "
                   "r26=%08x r27=%08x r28=%08x r29=%08x r30=%08x r31=%08x "
                   "caller_lr=%08x stack=%08x..%08x\n",
                   tag, t, OSThreadState(mmu.Read<u16>(t + OST_STATE)),
                   mmu.Read<u32>(t + OST_SUSPEND), mmu.Read<u32>(t + OST_PRIO),
                   mmu.Read<u32>(t + 0x2D4u), mmu.Read<u32>(t + OST_QUEUE),
                   mmu.Read<u32>(t + OST_MUTEX), mmu.Read<u32>(t + OST_SRR0),
                   mmu.Read<u32>(t + 0x84u), mmu.Read<u32>(t + 0x04u),
                   mmu.Read<u32>(t + 0x0Cu), mmu.Read<u32>(t + 0x10u),
                   mmu.Read<u32>(t + 0x14u), mmu.Read<u32>(t + 0x18u),
                   mmu.Read<u32>(t + 0x68u), mmu.Read<u32>(t + 0x6Cu),
                   mmu.Read<u32>(t + 0x70u), mmu.Read<u32>(t + 0x74u),
                   mmu.Read<u32>(t + 0x78u), mmu.Read<u32>(t + 0x7Cu),
                   saved_caller_lr,
                   mmu.Read<u32>(t + 0x304u), mmu.Read<u32>(t + 0x308u));
    };

    // Sanity-check the reads themselves before believing any of them. The
    // disc id sits at 0x80000000 and must read 'GM4E'; if it does not, every
    // conclusion drawn from mmu.Read here -- including "KartCtrl is NULL" --
    // is measuring the instrument rather than the game.
    {
      const u32 magic = mmu.Read<u32>(0x80000000u);
      std::fprintf(stderr,
                   "[threads] SANITY disc_id=%08x (%c%c%c%c) expect 474d3445  "
                   "osthreadq=%08x/%08x default=%08x curr=%08x\n",
                   magic, (magic >> 24) & 0xFF, (magic >> 16) & 0xFF,
                   (magic >> 8) & 0xFF, magic & 0xFF,
                   mmu.Read<u32>(0x800000DCu), mmu.Read<u32>(0x800000E0u),
                   mmu.Read<u32>(0x800000D8u), mmu.Read<u32>(0x800000E4u));
    }

    // Print raw first: a silent range-check drop looks identical to "no
    // threads", which is how the first run read as an empty system while the
    // game was busy at 99% CPU.
    for (u32 probe = 0x800000D0u; probe <= 0x800000FCu; probe += 4)
    {
      const u32 v = mmu.Read<u32>(probe);
      if (ok(v))
        std::fprintf(stderr, "[threads] lowmem %08x -> %08x %s\n", probe, v,
                     OSThreadState(mmu.Read<u16>(v + OST_STATE)));
    }
    thread_line("current", mmu.Read<u32>(OS_CURR_THREAD));

    // Anything parked here is blocked on a disc read that has not completed --
    // which is what a black screen during course load would look like.
    const u32 dvd_head = mmu.Read<u32>(DVD_THREAD_QUEUE);
    std::fprintf(stderr, "[threads] __DVDThreadQueue head=%08x tail=%08x\n",
                 dvd_head, mmu.Read<u32>(DVD_THREAD_QUEUE + 4));
    u32 t = dvd_head;
    for (int i = 0; i < 8 && ok(t); ++i)
    {
      thread_line("dvd-wait", t);
      t = mmu.Read<u32>(t + OST_NEXT);
    }

    int runnable = 0;
    for (u32 pri = 0; pri < 32; ++pri)
    {
      u32 h = mmu.Read<u32>(RUN_QUEUE + pri * 8);
      for (int i = 0; i < 8 && ok(h); ++i)
      {
        thread_line("runnable", h);
        ++runnable;
        h = mmu.Read<u32>(h + OST_NEXT);
      }
    }
    std::fprintf(stderr, "[threads] %d runnable\n", runnable);
  }
}

void KartReportInvalid()
{
  if (g_invalid_count == 0)
    return;
  std::fprintf(stderr, "[invalid-trace] total=%llu stored=%u\n",
               static_cast<unsigned long long>(g_invalid_count),
               static_cast<unsigned>(std::min<u64>(g_invalid_count, KART_INVALID_EVENT_COUNT)));
  const u32 stored = static_cast<u32>(std::min<u64>(g_invalid_count, KART_INVALID_EVENT_COUNT));
  for (u32 i = 0; i < stored; ++i)
  {
    const auto& e = g_invalid_events[i];
    std::fprintf(stderr,
                 "[invalid-trace] #%u %c%u ea=%08x pc=%08x lr=%08x ctr=%08x cr=%08x "
                 "r1=%08x r3=%08x r4=%08x r5=%08x r6=%08x r7=%08x r8=%08x "
                 "r28=%08x r29=%08x r30=%08x r31=%08x\n",
                 i, e.kind ? 'W' : 'R', e.size, e.ea, e.pc, e.lr, e.ctr, e.cr,
                 e.r1, e.r3, e.r4, e.r5, e.r6, e.r7, e.r8,
                 e.r28, e.r29, e.r30, e.r31);
  }
}

void StaticRecompCore::KartWatchJournal2(u32 vmem_offset, u32 size, void* user)
{
  auto* core = static_cast<StaticRecompCore*>(user);
  auto& mmu = core->m_system.GetMMU();


  // Refresh the cached singleton periodically: it is NULL until KartCtrl is
  // allocated, and an MMU read on every guest write would be ruinous.
  if (g_kart_watch_ctrl == 0 && (g_kart_watch_calls & 0xFFFFu) == 0)
  {
    const u32 p = core->m_system.GetMMU().Read<u32>(g_watch_ptr_addr);
    if (p >= 0x80000000u && p < 0x81800000u)
      g_kart_watch_ctrl = p;
  }
  if (g_kart_watch_ctrl == 0)
    return;

  // Sample the live count cheaply: once every 64k writes is plenty to catch
  // whether it ever rises above 8.
  if ((g_kart_watch_calls & 0xFFFFu) == 0)
  {
    if (g_kart_ea_calls)
      g_kart_ea_calls_last = *g_kart_ea_calls;
    const u32 n = core->m_system.GetMMU().Read<u32>(g_kart_watch_ctrl + 0x22Cu);
    if (n <= 64u && n > g_kart_count_max)
      g_kart_count_max = n;
  }

  // One-shot dump of every per-kart pointer array, entries 0..8. In a hung
  // state a savestate cannot replay the fault, but it still holds the wreckage:
  // whichever member reads 00000000 for kart 8 while 0..7 are populated is the
  // one whose creation never covered the extra kart.
  // Fires KART_DUMP_AFTER seconds' worth of writes in, not the instant
  // KartCtrl appears. The old trigger ran mid-construction, so karts 4-7 read
  // as uninitialised garbage and the dump described nothing. Default is late
  // enough to be past race creation.
  if (!g_kart_dumped && g_kart_watch_ctrl && g_kart_dump_on &&
      g_kart_watch_calls > g_kart_dump_after_writes)
  {
    g_kart_dumped = true;
    auto& mmu = core->m_system.GetMMU();
    // ext is the slot's offset in the host window. Kart 8 does NOT live in
    // guest RAM: reading index 8 from the object walks off the end of an
    // 8-entry array and lands on the next member, which is why every array's
    // "entry 8" used to equal the next array's entry 0.
    struct { const char* name; u32 off, stride, ext; } kc[] = {
        {"KartCtrl.mKartLoaders", 0x000, 4, 0x000}, {"KartCtrl.mGamePads", 0x020, 8, 0x020},
        {"KartCtrl.mKartPads", 0x060, 4, 0x040},    {"KartCtrl.mKartAnimes", 0x080, 4, 0x060},
        {"KartCtrl.mKartBodies", 0x0A0, 4, 0x080},  {"KartCtrl.mKartSus", 0x0C0, 16, 0x0a0},
        {"KartCtrl.mKartAppendix", 0x140, 8, 0x0c0},{"KartCtrl.mRivalKarts", 0x180, 4, 0x0e0},
        {"KartCtrl.mKartSounds", 0x1A0, 4, 0x100},  {"KartCtrl.mKartTargets", 0x1C0, 4, 0x120},
        {"KartCtrl.mKartDisps", 0x1E0, 4, 0x140},   {"KartCtrl.mKartCams", 0x200, 4, 0x160},
    };
    auto ext_word = [](u32 off) {
      return (u32)((g_kart_ext[off] << 24) | (g_kart_ext[off + 1] << 16) |
                   (g_kart_ext[off + 2] << 8) | g_kart_ext[off + 3]);
    };
    std::fprintf(stderr, "[kart_dump] KartCtrl=%08x mKartCount=%u\n",
                 g_kart_watch_ctrl, mmu.Read<u32>(g_kart_watch_ctrl + 0x22Cu));
    for (auto& m : kc)
    {
      std::fprintf(stderr, "[kart_dump] %-24s", m.name);
      for (u32 i = 0; i < 8; ++i)
        std::fprintf(stderr, " %08x",
                     mmu.Read<u32>(g_kart_watch_ctrl + m.off + i * m.stride));
      std::fprintf(stderr, " | k8=%08x\n", ext_word(m.ext));
    }
    // Kart 8's KartLoader owns its ExModels inline (decomp: mBodyModel @0x14,
    // mWheelModels[6] @0xA4 stride 0x8A). KartSus::InitSettingParam copies
    // those into KartSus::mWheel @0. The draw path NULLs on mWheel, so the
    // question is whether the loader ever produced models at all.
    {
      const u32 loader8 = (u32)((g_kart_ext[0x1a0] << 24) | (g_kart_ext[0x1a1] << 16) |
                                (g_kart_ext[0x1a2] << 8) | g_kart_ext[0x1a3]);
      const u32 sus8 = (u32)((g_kart_ext[0x0a0] << 24) | (g_kart_ext[0x0a1] << 16) |
                             (g_kart_ext[0x0a2] << 8) | g_kart_ext[0x0a3]);
      std::fprintf(stderr, "[kart8] loader=%08x sus=%08x\n", loader8, sus8);
      if (loader8 >= 0x80000000u && loader8 < 0x81800000u)
      {
        // The decomp gives mWheelModels[6] @0xA4 stride 0x8A, but 0x8A is not
        // 4-aligned so strided reads land mid-word. Dump the region raw and let
        // the pointers show their own spacing.
        std::fprintf(stderr, "[kart8] loader.mBodyModel=%08x\n",
                     mmu.Read<u32>(loader8 + 0x14u));
        for (u32 off = 0xA0; off <= 0x120; off += 16)
        {
          std::fprintf(stderr, "[kart8] loader+%03x:", off);
          for (u32 i = 0; i < 16; i += 4)
            std::fprintf(stderr, " %08x", mmu.Read<u32>(loader8 + off + i));
          std::fprintf(stderr, "\n");
        }
      }
      if (sus8 >= 0x80000000u && sus8 < 0x81800000u)
        std::fprintf(stderr, "[kart8] sus.mWheel=%08x mArm=%08x mShock=%08x\n",
                     mmu.Read<u32>(sus8 + 0x0u), mmu.Read<u32>(sus8 + 0x4u),
                     mmu.Read<u32>(sus8 + 0x8u));
      // kart 7 for comparison: same fields, known good
      const u32 l7 = mmu.Read<u32>(mmu.Read<u32>(0x803CB7E8u) + 0x68u + 7 * 4);
      if (l7 >= 0x80000000u && l7 < 0x81800000u)
        std::fprintf(stderr, "[kart7] loader=%08x mBodyModel=%08x\n", l7,
                     mmu.Read<u32>(l7 + 0x14u));
        for (u32 off = 0xA0; off <= 0x120; off += 16)
        {
          std::fprintf(stderr, "[kart7] loader+%03x:", off);
          for (u32 i = 0; i < 16; i += 4)
            std::fprintf(stderr, " %08x", mmu.Read<u32>(l7 + off + i));
          std::fprintf(stderr, "\n");
        }
    }
    const u32 rm = mmu.Read<u32>(0x803CB7E8u);
    if (rm >= 0x80000000u && rm < 0x81800000u)
    {
      for (auto& m : {std::pair<const char*, u32>{"RaceMgr.mKartChecker", 0x48},
                      std::pair<const char*, u32>{"RaceMgr.mKartLoader", 0x68}})
      {
        std::fprintf(stderr, "[kart_dump] %-24s", m.first);
        for (u32 i = 0; i < 8; ++i)
          std::fprintf(stderr, " %08x", mmu.Read<u32>(rm + m.second + i * 4));
        std::fprintf(stderr, " | k8=%08x\n",
                     ext_word(m.second == 0x48 ? 0x180 : 0x1a0));
      }
    }
  }

  const u32 lo = (g_kart_watch_ctrl & RAM_MASK) + KART_WATCH_MEMBER;
  if (vmem_offset < lo || vmem_offset >= lo + KART_WATCH_LEN)
    return;

  if (g_kart_watch_hits < KART_WATCH_MAX)
  {
    g_kart_watch_hit[g_kart_watch_hits] = {core->m_guest.pc,
                                           vmem_offset - (g_kart_watch_ctrl & RAM_MASK), size};
  }
  ++g_kart_watch_hits;
}

void StaticRecompCore::KartBody36Journal(u32 vmem_offset, u32 size, void* user)
{
  auto* core = static_cast<StaticRecompCore*>(user);
  auto& mmu = core->m_system.GetMMU();
  const u32 ctrl = mmu.Read<u32>(0x803CC588u);
  if (ctrl < 0x80000000u || ctrl >= 0x81800000u)
    return;
  const u32 body = mmu.Read<u32>(ctrl + 0xA0u);
  if (body < 0x80000000u || body >= 0x81800000u)
    return;
  const u32 body_target = (body + 0x24u) & RAM_MASK;
  const u32 mgr = mmu.Read<u32>(0x803CB7E8u);
  const u32 loader = mgr >= 0x80000000u && mgr < 0x81800000u ?
                         mmu.Read<u32>(mgr + 0x68u) : 0u;
  const u32 loader_target = loader >= 0x80000000u && loader < 0x81800000u ?
                               ((loader + 0x10u) & RAM_MASK) : 0xffffffffu;
  const bool hit_body = vmem_offset <= body_target && vmem_offset + size > body_target;
  const bool hit_loader = vmem_offset <= loader_target && vmem_offset + size > loader_target;
  if (!hit_body && !hit_loader)
    return;
  const u32 n = g_body36_count++;
  auto& e = g_body36_events[n & 31u];
  e.kind = hit_body ? 1u : 2u;
  e.pc = core->m_guest.pc; e.lr = core->m_guest.lr;
  e.ea = hit_body ? body + 0x24u : loader + 0x10u;
  e.value = mmu.Read<u32>(e.ea);
  e.r3 = core->m_guest.gpr[3]; e.r4 = core->m_guest.gpr[4];
  e.r5 = core->m_guest.gpr[5]; e.r29 = core->m_guest.gpr[29];
  e.r30 = core->m_guest.gpr[30]; e.r31 = core->m_guest.gpr[31];
}

void StaticRecompCore::KartItemStateJournal(u32 vmem_offset, u32 size, void* user)
{
  auto* core = static_cast<StaticRecompCore*>(user);
  auto& mmu = core->m_system.GetMMU();
  const u32 mgr = mmu.Read<u32>(0x803CBF40u);
  if (mgr < 0x80000000u || mgr >= 0x81800000u)
    return;
  const u32 off = vmem_offset - (mgr & RAM_MASK);
  // mStockItem, mEquipItem, hit/equip/use flags, heart and shuffle pointers.
  if (off < 0x24cu || off >= 0x39cu)
    return;
  const u64 ordinal = g_item_state_count++;
  if (ordinal >= KART_ITEM_STATE_MAX)
    return;
  u32 value = 0;
  if (size == 1) value = mmu.Read<u8>(mgr + off);
  else if (size == 2) value = mmu.Read<u16>(mgr + off);
  else value = mmu.Read<u32>(mgr + off);
  const auto& c = core->m_guest;
  g_item_state_hit[ordinal] = {c.pc, c.lr, mgr + off, size, value,
                               c.gpr[0], c.gpr[3], c.gpr[4], c.gpr[5], c.gpr[6],
                               c.gpr[7], c.gpr[8], c.gpr[28], c.gpr[29],
                               c.gpr[30], c.gpr[31]};
}

void KartReportBody36()
{
  const u32 used = std::min<u32>(g_body36_count, 32u);
  const u32 first = g_body36_count - used;
  std::fprintf(stderr, "[body36-summary] writes=%u stored=%u\n", g_body36_count, used);
  for (u32 i = 0; i < used; ++i)
  {
    const u32 n = first + i;
    const auto& e = g_body36_events[n & 31u];
    std::fprintf(stderr,
                 "[body36] n=%u kind=%u pc=%08x lr=%08x ea=%08x value=%08x "
                 "r3=%08x r4=%08x r5=%08x r29=%08x r30=%08x r31=%08x\n",
                 n, e.kind, e.pc, e.lr, e.ea, e.value, e.r3, e.r4, e.r5,
                 e.r29, e.r30, e.r31);
  }
}

void StaticRecompCore::KartWatchInit()
{
  // Negotiate the sidecar contract independently of the optional write-watch
  // diagnostics. This runs for every loaded module, including normal release
  // runs where KART_WATCH is unset.
  KartExtConfigure(m_library);
  if (const char* v = std::getenv("KART_WATCH_BASE"))
    g_kart_watch_ctrl = static_cast<u32>(std::strtoul(v, nullptr, 0));
  if (const char* v = std::getenv("KART_WATCH_PTR"))
    g_watch_ptr_addr = static_cast<u32>(std::strtoul(v, nullptr, 0));
  if (const char* v = std::getenv("KART_WATCH_OFF"))
    KART_WATCH_MEMBER = static_cast<u32>(std::strtoul(v, nullptr, 0));
  if (const char* v = std::getenv("KART_WATCH_LEN"))
    KART_WATCH_LEN = static_cast<u32>(std::strtoul(v, nullptr, 0));
  const char* on = std::getenv("KART_WATCH");
  if (!on || on[0] == '0' || on[0] == '\0')
    return;
  const char* ls = std::getenv("STATICRECOMP_LOCKSTEP");
  if (ls && ls[0] != '0' && ls[0] != '\0')
  {
    std::fprintf(stderr, "[kart_watch] refusing to install: lockstep owns the "
                         "write journal. Unset STATICRECOMP_LOCKSTEP.\n");
    return;
  }
  if (!m_module)
  {
    std::fprintf(stderr, "[kart_watch] no module loaded\n");
    return;
  }
  using SetFn = void (*)(void (*)(u32, u32, void*), void*);
  auto set = reinterpret_cast<SetFn>(m_library.GetSymbolAddress("ppc_set_mem_write_journal"));
  if (!set)
  {
    std::fprintf(stderr, "[kart_watch] module lacks ppc_set_mem_write_journal\n");
    return;
  }
  g_kart_ea_calls = reinterpret_cast<const volatile unsigned long long*>(
      m_library.GetSymbolAddress("kart_ea_calls"));
  if (std::getenv("KART_BODY36_TRACE"))
  {
    set(&StaticRecompCore::KartBody36Journal, this);
    g_kart_watch_on = true;
    std::fprintf(stderr, "[kart_watch] lightweight KartBody[0]+0x24 writer trace\n");
    return;
  }
  if (std::getenv("KART_ITEM_STATE_TRACE"))
  {
    set(&StaticRecompCore::KartItemStateJournal, this);
    g_kart_watch_on = true;
    std::fprintf(stderr, "[kart_watch] ItemObjMgr stock/equip register trace\n");
    return;
  }
  // The journal fires on EVERY guest write -- billions per run -- and costs
  // roughly 3x wall clock on a cold boot. KART_WATCH_LIGHT keeps the kart_ea
  // total (a plain module counter, free) and drops the window watch, the
  // KartCtrl watch, the dump and the profiler with it.
  if (std::getenv("KART_WATCH_LIGHT"))
  {
    g_kart_watch_on = true;
    std::fprintf(stderr, "[kart_watch] light mode: kart_ea only, no write journal\n");
    return;
  }
  set(&StaticRecompCore::KartWatchJournal, this);
  g_kart_watch_on = true;
  std::fprintf(stderr, "[kart_watch] watching *(0x%08x)+0x%02x..0x%02x\n",
               g_watch_ptr_addr, KART_WATCH_MEMBER, KART_WATCH_MEMBER + KART_WATCH_LEN);
}

bool StaticRecompCore::HookHostCall(CPUState* cpu, u32 address)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  return core->m_module_source.host_call &&
         core->m_module_source.host_call(cpu, address, core->m_module_source.host_call_user);
}

u64 StaticRecompCore::HookExternalRead(CPUState* cpu, u32 ea, u8 size)
{
  if (KartExtContains(ea, size))
  {
    // Seeding RaceInfo::mKartInfo for karts 8+ at RaceInfo::setRace copies
    // nothing: setRace opens with reset(), and character select populates
    // mKartInfo only afterwards, so kart 0's entry is still all zeroes there
    // (measured -- the seed writes six NULLs). Rather than guess which later
    // guest function is both after character select and before the first
    // consumer, fill the slot the first time anyone reads it, from whatever
    // kart 0 holds right then. A source that is still blank is left alone, so
    // an early read cannot latch zeroes in permanently.
    if (KART_EXT_SEED_ENABLE)
      KartExtSeedKartInfoOnDemand(cpu, ea, size);
    return KartExtRead(ea, size, cpu->pc);
  }

  KartRecordInvalid(cpu, ea, size, 0u);

  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ea = core->TranslateRelAddress(ea);
  if (ea == 0)
  {
    // Dump the pointer-ish registers too. A failed translation says an object
    // was NULL; knowing WHICH object is what identifies the structure whose
    // per-kart array still ends at 8. Guest addresses live in 0x80xxxxxx.
    std::fprintf(stderr,
                 "[zero-access] read size=%u guest_pc=%08x ppc_pc=%08x lr=%08x"
                 " r3=%08x r4=%08x r5=%08x r6=%08x r31=%08x\n",
                 size, cpu->pc, core->m_system.GetPPCState().pc, cpu->lr,
                 cpu->gpr[3], cpu->gpr[4], cpu->gpr[5], cpu->gpr[6], cpu->gpr[31]);
  }
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
  if (KartExtContains(ea, size))
  {
    KartExtWrite(ea, value, size, cpu->pc);
    return;
  }


  KartRecordInvalid(cpu, ea, size, 1u);

  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ea = core->TranslateRelAddress(ea);
  if (ea == 0)
    std::fprintf(stderr, "[zero-access] write size=%u guest_pc=%08x ppc_pc=%08x lr=%08x\n", size,
                 cpu->pc, core->m_system.GetPPCState().pc, cpu->lr);

  // Gather-pipe fast path: stores to the write-gather pipe page at effective
  // 0xCC008000 go straight to GPFifo, mirroring the MMU's masked-write
  // special case without an MMU round trip. Keying on the effective page is
  // the same shortcut Dolphin's JITs take (optimizeGatherPipe). GPFifo
  // maintains ppc_state.gather_pipe_ptr internally.
  if ((ea & 0xFFFFF000) == 0xCC008000u)
  {
    if (core->m_lockstep_verifier->m_ls_journaling)
      core->m_lockstep_verifier->m_journal.native_mmio.push_back({ea, static_cast<u32>(value), size});
    auto& gpfifo = core->m_system.GetGPFifo();
    switch (size)
    {
    case 1:
      gpfifo.Write8(static_cast<u8>(value));
      return;
    case 2:
      gpfifo.Write16(static_cast<u16>(value));
      return;
    case 4:
      gpfifo.Write32(static_cast<u32>(value));
      return;
    default:
      for (u32 i = size * 8u; i > 0;)
      {
        i -= 8;
        gpfifo.Write8(static_cast<u8>(value >> i));
      }
      return;
    }
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
  auto& memory = core->m_system.GetMemory();
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
