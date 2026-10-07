// Native replacements for recompiled guest library functions.
//
// Selected by DOLRECOMP_NATIVE_SUBST at codegen time: the emitter guards the
// matched function's entry label with a call to one of these, so the
// substitution costs nothing beyond the dispatch the guest already pays for the
// call. Doing the same thing through a mod patch was measured 34% *slower*,
// because a mod patch round-trips the dispatcher and copies the whole CPU state
// in and out on every call.
//
// CONTRACT. Each routine returns 1 if it fully performed the guest function,
// having set the return value where the ABI puts it and set pc to lr; or 0 to
// decline, having touched nothing, in which case the emitter falls through to
// the translated guest code.
//
// Declining is the important half. The first version of memcpy/memset here
// bounds-checked the guest pointers and, when the range was not wholly inside
// main RAM, silently did nothing and returned as though it had copied - so any
// buffer staged through the L1 locked cache at 0xE0000000 (which this game uses
// for its graphics state) came out as garbage, and field dialogue boxes
// rendered as white slivers. A substitution that cannot handle its inputs must
// hand them back, never swallow them.
//
// Guest floats are big-endian, so every load and store swaps. The arithmetic
// mirrors the guest's operation order and rounding exactly: these routines are
// hand-written paired-single assembly whose ps_madds steps are *fused*
// multiply-adds, and the module compiles with -ffp-contract=off so the fusing
// must be explicit. Bit-exactness is not cosmetic - the lockstep verifier
// compares recompiled execution against the interpreter, and netplay assumes
// both sides agree.

#include "gxruntime/gx_recomp.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#define DOLRECOMP_BSWAP32(x) _byteswap_ulong(x)
#else
#define DOLRECOMP_BSWAP32(x) __builtin_bswap32(x)
#endif

// Debug switches, read once:
//   DOLRECOMP_NATIVE_SUBST_OFF    every substitution declines, so the module
//                                 runs entirely on translated guest code. Same
//                                 binary, so a rendering or timing difference
//                                 can be bisected without a rebuild.
//   DOLRECOMP_NATIVE_SUBST_TRACE  report each routine's handled/declined counts
//                                 at exit. A routine that declines constantly
//                                 is not earning its place; one that declines
//                                 occasionally is exactly the case that used to
//                                 corrupt memory silently.
enum {
  DOLRECOMP_SUBST_MEMCPY,
  DOLRECOMP_SUBST_MEMSET,
  DOLRECOMP_SUBST_MTX_SCALED_ADD,
  DOLRECOMP_SUBST_MTX_CONCAT,
  DOLRECOMP_SUBST_MTX_COPY,
  DOLRECOMP_SUBST_MTX_IDENTITY,
  DOLRECOMP_SUBST_MTX_MULT_VEC,
  DOLRECOMP_SUBST_DCFLUSHRANGE,
  DOLRECOMP_SUBST_COUNT
};

static const char *const dolrecomp_subst_names[DOLRECOMP_SUBST_COUNT] = {
    "memcpy", "memset", "HSD_MtxScaledAdd", "PSMTXConcat",
    "PSMTXCopy", "PSMTXIdentity", "PSMTXMultVec"};

static unsigned long long dolrecomp_subst_handled[DOLRECOMP_SUBST_COUNT];
static unsigned long long dolrecomp_subst_declined[DOLRECOMP_SUBST_COUNT];
static int dolrecomp_subst_state = -1;  // -1 unread, 0 off, 1 on, 2 on+trace

// Declines are bucketed by the top byte of the offending guest address, so a
// routine that declines a lot says *where* its pointers live rather than just
// how often it gave up. 0x80/0xC0 mean main RAM (a size overrun, since the
// window itself is accepted), 0xE0 the L1 locked cache, 0x00 a null.
static unsigned long long dolrecomp_subst_decline_dst[DOLRECOMP_SUBST_COUNT][256];
static unsigned long long dolrecomp_subst_decline_src[DOLRECOMP_SUBST_COUNT][256];

static void dolrecomp_subst_report_buckets(const char *label, int which,
                                           unsigned long long (*table)[256]) {
  for (int b = 0; b < 256; b++) {
    if (table[which][b])
      fprintf(stderr, "[subst]     %s %02x......  %llu\n", label, b, table[which][b]);
  }
}

static void dolrecomp_subst_report(void) {
  fprintf(stderr, "[subst] native substitution counts (handled / declined)\n");
  for (int i = 0; i < DOLRECOMP_SUBST_COUNT; i++) {
    if (dolrecomp_subst_handled[i] || dolrecomp_subst_declined[i]) {
      fprintf(stderr, "[subst]   %-18s %12llu / %llu\n", dolrecomp_subst_names[i],
              dolrecomp_subst_handled[i], dolrecomp_subst_declined[i]);
      dolrecomp_subst_report_buckets("dst", i, dolrecomp_subst_decline_dst);
      dolrecomp_subst_report_buckets("src", i, dolrecomp_subst_decline_src);
    }
  }
  fflush(stderr);
}

static int dolrecomp_subst_init(void) {
  if (getenv("DOLRECOMP_NATIVE_SUBST_OFF"))
    return 0;
  if (getenv("DOLRECOMP_NATIVE_SUBST_TRACE")) {
    atexit(dolrecomp_subst_report);
    return 2;
  }
  return 1;
}

static inline int dolrecomp_subst_enabled(void) {
  if (dolrecomp_subst_state < 0)
    dolrecomp_subst_state = dolrecomp_subst_init();
  return dolrecomp_subst_state != 0;
}

static inline int dolrecomp_handled(int which) {
  if (dolrecomp_subst_state == 2)
    dolrecomp_subst_handled[which]++;
  return 1;
}

static inline int dolrecomp_declined(int which) {
  if (dolrecomp_subst_state == 2)
    dolrecomp_subst_declined[which]++;
  return 0;
}

// As above, but also records which of the two pointers was unusable and where
// it pointed. dst_bad/src_bad are the results of the caller's own checks, so a
// call declined for both reasons is counted in both buckets.
static inline int dolrecomp_declined_at(int which, uint32_t dst_addr, int dst_bad,
                                        uint32_t src_addr, int src_bad) {
  if (dolrecomp_subst_state == 2) {
    dolrecomp_subst_declined[which]++;
    if (dst_bad)
      dolrecomp_subst_decline_dst[which][(dst_addr >> 24) & 0xFFu]++;
    if (src_bad)
      dolrecomp_subst_decline_src[which][(src_addr >> 24) & 0xFFu]++;
  }
  return 0;
}

// Main RAM is reachable through two windows: cached at 0x80000000 and uncached
// at 0xC0000000, both mapping to the same bytes. Games use the uncached mirror
// heavily for DMA and GX data, so accepting only the cached one is not a
// conservative choice - it declined 73% of memcpy calls, which is most of the
// win on the hottest non-matrix function. This mirrors resolve_addr in the CPU
// runtime. Everything else - the L1 locked cache at 0xE0000000, ARAM, MMIO -
// still declines and is left to the guest code.
static uint8_t *dolrecomp_guest_bytes(CPUState *ctx, uint32_t address,
                                      uint32_t bytes) {
  if (!ctx->ram || bytes > ctx->ram_size)
    return 0;
  uint32_t offset;
  if (address >= GC_RAM_BASE && address - GC_RAM_BASE < ctx->ram_size)
    offset = address - GC_RAM_BASE;
  else if (address >= GC_RAM_UNCACHED && address - GC_RAM_UNCACHED < ctx->ram_size)
    offset = address - GC_RAM_UNCACHED;
  else
    return 0;
  if (offset > ctx->ram_size - bytes)
    return 0;
  return ctx->ram + offset;
}

// Anything outside the two RAM windows is asked of the host rather than guessed
// at. Instrumenting the declines showed every single one was the L1 locked
// cache at 0xE0000000 - 2.0M of 2.8M memcpy calls, because the graphics state
// this game stages there is copied in and out constantly. The host's
// external_pointer hook already hands out a real pointer for exactly that range
// and returns null for everything else (MMIO, unmapped, anything needing live
// MMU state to resolve), so deferring to it widens the substitution to the case
// that matters without inventing a second opinion about which addresses are
// safe. Null still means decline, and decline still means the guest code runs.
static uint8_t *dolrecomp_guest_bytes_ext(CPUState *ctx, uint32_t address,
                                          uint32_t bytes) {
  uint8_t *direct = dolrecomp_guest_bytes(ctx, address, bytes);
  if (direct || bytes == 0 || !ctx->external_pointer)
    return direct;
  return (uint8_t *)ctx->external_pointer(ctx, address, bytes);
}

static inline float *dolrecomp_guest_floats(CPUState *ctx, uint32_t address,
                                            uint32_t count) {
  return (float *)(void *)dolrecomp_guest_bytes(ctx, address,
                                                count * (uint32_t)sizeof(float));
}

static inline float dolrecomp_load_be(const float *p) {
  uint32_t bits;
  memcpy(&bits, p, sizeof(bits));
  bits = DOLRECOMP_BSWAP32(bits);
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline void dolrecomp_store_be(float *p, float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  bits = DOLRECOMP_BSWAP32(bits);
  memcpy(p, &bits, sizeof(bits));
}

// void* memcpy(void* dst, const void* src, u32 count)
//
// The guest's memcpy copies forward when src >= dst and backward otherwise,
// which is memmove's contract, not memcpy's; substituting host memcpy would be
// undefined on the overlapping ranges the game relies on. Returns dst in r3.
int dolrecomp_native_memcpy(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  const uint32_t dst_addr = (uint32_t)ctx->gpr[3];
  const uint32_t src_addr = (uint32_t)ctx->gpr[4];
  const uint32_t count = (uint32_t)ctx->gpr[5];
  uint8_t *dst = dolrecomp_guest_bytes_ext(ctx, dst_addr, count);
  const uint8_t *src = dolrecomp_guest_bytes_ext(ctx, src_addr, count);
  if (count && (!dst || !src))
    return dolrecomp_declined_at(DOLRECOMP_SUBST_MEMCPY, dst_addr, !dst, src_addr,
                                 !src);
  if (count)
    memmove(dst, src, count);
  ctx->gpr[3] = dst_addr;
  ctx->pc = ctx->lr;
  return dolrecomp_handled(DOLRECOMP_SUBST_MEMCPY);
}

// void* memset(void* dest, int val, u32 count)
int dolrecomp_native_memset(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  const uint32_t dest_addr = (uint32_t)ctx->gpr[3];
  const uint32_t count = (uint32_t)ctx->gpr[5];
  uint8_t *dest = dolrecomp_guest_bytes_ext(ctx, dest_addr, count);
  if (count && !dest)
    return dolrecomp_declined(DOLRECOMP_SUBST_MEMSET);
  if (count)
    memset(dest, (int)(uint8_t)ctx->gpr[4], count);
  ctx->gpr[3] = dest_addr;
  ctx->pc = ctx->lr;
  return dolrecomp_handled(DOLRECOMP_SUBST_MEMSET);
}


// void DCFlushRange(void* addr, u32 nBytes)
//
// The guest routine is a dcbf loop over 32-byte lines followed by an `sc`. Each
// dcbf reaches the host, and the host does per-line work there (a JIT icache
// invalidation plus a binary search over chunk ranges), so a range flush over a
// large buffer costs thousands of host round trips. PC sampling put this routine
// at roughly half of all guest CPU time in a battle scene.
//
// This flushes every line BUT THE LAST in one range call, then rewrites r3/r4 so
// the guest runs its own loop for exactly that final line. It deliberately
// returns 0, which resumes the translated guest code from the top of the
// routine: the trailing `sc` and the return then execute exactly as they would
// have. That matters because `sc` raises a system call the OS handler sees, and
// setting pc = lr the way a fully-handled substitution does would skip it.
// Re-flushing the final line is idempotent, so the visible result is unchanged.
int dolrecomp_native_dcflushrange(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  const uint32_t addr = (uint32_t)ctx->gpr[3];
  const uint32_t bytes = (uint32_t)ctx->gpr[4];
  // The guest checks nBytes <= 0 first and returns without the `sc`; leave that
  // path alone rather than reproducing it.
  if ((int32_t)bytes <= 0)
    return dolrecomp_declined(DOLRECOMP_SUBST_DCFLUSHRANGE);

  const uint32_t start = addr & ~31u;
  const uint32_t lines = (bytes + (addr & 31u) + 31u) >> 5;

  // Below a threshold this substitution is a PESSIMISATION, which cold-boot
  // timing caught and the steady-state scenes hid: the fixed cost here is the
  // entry guard plus one host round trip for the range, and the guest still
  // flushes the final line, so for a two or three line flush that is more work
  // than simply letting the guest loop do it. Boot is dominated by small,
  // frequent flushes and measured 9.6% SLOWER before this check; the battle and
  // field scenes issue larger ones and gained 7-19%.
  // DOLRECOMP_DCF_MIN_LINES tunes it. 8 was chosen by sweeping the value on one
  // build (it is read at runtime, so no rebuild is needed to change it):
  //
  //                  none      4        8       16
  //   postbattle    +19.1%    --     +16.0%   +15.3%
  //   battle-mid     +8.2%    --     +11.4%    +8.0%
  //   field-phenac   +7.2%   +4.2%    +4.3%    +0.9%
  //   cold boot      -9.6%    --      -1.5%    -2.8%   (negative = slower)
  //
  // 16 throws away field-phenac, whose flushes are mostly under that size.
  static int min_lines = -1;
  if (min_lines < 0) {
    const char *setting = getenv("DOLRECOMP_DCF_MIN_LINES");
    min_lines = (setting && *setting) ? atoi(setting) : 8;
    if (min_lines < 2)
      min_lines = 2;
  }
  if (lines < (uint32_t)min_lines)
    return dolrecomp_declined(DOLRECOMP_SUBST_DCFLUSHRANGE);

  ppc_cache_range(ctx, PPC_CACHE_DCBF, start, (lines - 1u) * 32u);
  ctx->gpr[3] = start + (lines - 1u) * 32u;
  ctx->gpr[4] = 32u;
  dolrecomp_handled(DOLRECOMP_SUBST_DCFLUSHRANGE);
  return 0;
}

// void HSD_MtxScaledAdd(f32* src, f32 scale, f32* add, f32* dst)
//
// dst[i] = scale * src[i] + add[i] over 12 floats. The decomp compiles this
// under "#pragma fp_contract on", so each term is one fused multiply-add.
// scale arrives in f1 per the EABI, not in a GPR.
int dolrecomp_native_hsd_mtx_scaled_add(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  const float *src = dolrecomp_guest_floats(ctx, (uint32_t)ctx->gpr[3], 12u);
  const float *add = dolrecomp_guest_floats(ctx, (uint32_t)ctx->gpr[4], 12u);
  float *dst = dolrecomp_guest_floats(ctx, (uint32_t)ctx->gpr[5], 12u);
  if (!src || !add || !dst)
    return dolrecomp_declined(DOLRECOMP_SUBST_MTX_SCALED_ADD);

  const float scale = (float)ctx->fpr[1];
  float out[12];
  for (int i = 0; i < 12; i++)
    out[i] = fmaf(scale, dolrecomp_load_be(src + i), dolrecomp_load_be(add + i));
  for (int i = 0; i < 12; i++)
    dolrecomp_store_be(dst + i, out[i]);

  ctx->pc = ctx->lr;
  return dolrecomp_handled(DOLRECOMP_SUBST_MTX_SCALED_ADD);
}

// void PSMTXConcat(const Mtx a, const Mtx b, Mtx ab)
// ab = a * b, 3x4 row-major. ab may alias a or b, so nothing is stored until
// the whole result is computed.
int dolrecomp_native_psmtx_concat(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  const float *a = dolrecomp_guest_floats(ctx, (uint32_t)ctx->gpr[3], 12u);
  const float *b = dolrecomp_guest_floats(ctx, (uint32_t)ctx->gpr[4], 12u);
  float *ab = dolrecomp_guest_floats(ctx, (uint32_t)ctx->gpr[5], 12u);
  if (!a || !b || !ab)
    return dolrecomp_declined(DOLRECOMP_SUBST_MTX_CONCAT);

  float av[12], bv[12], out[12];
  for (int i = 0; i < 12; i++) {
    av[i] = dolrecomp_load_be(a + i);
    bv[i] = dolrecomp_load_be(b + i);
  }
  for (int row = 0; row < 3; row++) {
    const float r0 = av[row * 4 + 0];
    const float r1 = av[row * 4 + 1];
    const float r2 = av[row * 4 + 2];
    const float r3 = av[row * 4 + 3];
    out[row * 4 + 0] = fmaf(r2, bv[8], fmaf(r1, bv[4], r0 * bv[0]));
    out[row * 4 + 1] = fmaf(r2, bv[9], fmaf(r1, bv[5], r0 * bv[1]));
    out[row * 4 + 2] = fmaf(r2, bv[10], fmaf(r1, bv[6], r0 * bv[2]));
    // The last step is ps_madds1 against the constant pair (0.0, 1.0) at
    // lbl_804789B0: a fused multiply-add by exactly 1.0, i.e. an exact add.
    out[row * 4 + 3] = fmaf(r2, bv[11], fmaf(r1, bv[7], r0 * bv[3])) + r3;
  }
  for (int i = 0; i < 12; i++)
    dolrecomp_store_be(ab + i, out[i]);

  ctx->pc = ctx->lr;
  return dolrecomp_handled(DOLRECOMP_SUBST_MTX_CONCAT);
}

// void PSMTXCopy(const Mtx src, Mtx dst)
int dolrecomp_native_psmtx_copy(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  const uint8_t *src = dolrecomp_guest_bytes(ctx, (uint32_t)ctx->gpr[3], 48u);
  uint8_t *dst = dolrecomp_guest_bytes(ctx, (uint32_t)ctx->gpr[4], 48u);
  if (!src || !dst)
    return dolrecomp_declined(DOLRECOMP_SUBST_MTX_COPY);
  if (src != dst)
    memcpy(dst, src, 48u);  // raw bytes: endianness is irrelevant to a copy
  ctx->pc = ctx->lr;
  return dolrecomp_handled(DOLRECOMP_SUBST_MTX_COPY);
}

// void PSMTXIdentity(Mtx m)
int dolrecomp_native_psmtx_identity(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  float *m = dolrecomp_guest_floats(ctx, (uint32_t)ctx->gpr[3], 12u);
  if (!m)
    return dolrecomp_declined(DOLRECOMP_SUBST_MTX_IDENTITY);
  for (int i = 0; i < 12; i++)
    dolrecomp_store_be(m + i, 0.0f);
  dolrecomp_store_be(m + 0, 1.0f);
  dolrecomp_store_be(m + 5, 1.0f);
  dolrecomp_store_be(m + 10, 1.0f);
  ctx->pc = ctx->lr;
  return dolrecomp_handled(DOLRECOMP_SUBST_MTX_IDENTITY);
}

// void PSMTXMultVec(const Mtx m, const Vec* src, Vec* dst)
//
// Follows the SDK's paired-single sequence, which is not a left-to-right dot
// product: each row is two lanes (ps_mul then a fused ps_madd) joined by
// ps_sum0, giving (m0*x + m2*z) + (m1*y + m3). Float addition is not
// associative, so the grouping is part of the contract.
int dolrecomp_native_psmtx_mult_vec(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  const float *m = dolrecomp_guest_floats(ctx, (uint32_t)ctx->gpr[3], 12u);
  const float *src = dolrecomp_guest_floats(ctx, (uint32_t)ctx->gpr[4], 3u);
  float *dst = dolrecomp_guest_floats(ctx, (uint32_t)ctx->gpr[5], 3u);
  if (!m || !src || !dst)
    return dolrecomp_declined(DOLRECOMP_SUBST_MTX_MULT_VEC);

  const float x = dolrecomp_load_be(src + 0);
  const float y = dolrecomp_load_be(src + 1);
  const float z = dolrecomp_load_be(src + 2);
  float mv[12];
  for (int i = 0; i < 12; i++)
    mv[i] = dolrecomp_load_be(m + i);

  const float ox = fmaf(mv[2], z, mv[0] * x) + (mv[1] * y + mv[3]);
  const float oy = fmaf(mv[6], z, mv[4] * x) + (mv[5] * y + mv[7]);
  const float oz = fmaf(mv[10], z, mv[8] * x) + (mv[9] * y + mv[11]);

  dolrecomp_store_be(dst + 0, ox);
  dolrecomp_store_be(dst + 1, oy);
  dolrecomp_store_be(dst + 2, oz);

  ctx->pc = ctx->lr;
  return dolrecomp_handled(DOLRECOMP_SUBST_MTX_MULT_VEC);
}
