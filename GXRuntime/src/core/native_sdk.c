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
  DOLRECOMP_SUBST_GX_POS_MTX,
  DOLRECOMP_SUBST_GX_NRM_MTX,
  DOLRECOMP_SUBST_GX_TEX_MTX,
  DOLRECOMP_SUBST_GS_QUEUE_COMMAND,
  DOLRECOMP_SUBST_COUNT
};

static const char *const dolrecomp_subst_names[DOLRECOMP_SUBST_COUNT] = {
    "memcpy", "memset", "HSD_MtxScaledAdd", "PSMTXConcat",
    "PSMTXCopy", "PSMTXIdentity", "PSMTXMultVec", "DCFlushRange",
    "GXLoadPosMtxImm", "GXLoadNrmMtxImm", "GXLoadTexMtxImm",
    "GSQueueCommand"};

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


// ---------------------------------------------------------------------------
// Native GX command writers.
//
// GXLoadPosMtxImm / GXLoadNrmMtxImm / GXLoadTexMtxImm only stream a command
// byte, a register word and a matrix into the write-gather pipe. The native
// versions build the identical byte stream and append it straight to the
// gather-pipe buffer (see dolrecomp_gp_write in cpu.h), handing only the last
// word to external_write so GPFifo flushes every completed 32-byte burst
// exactly as the per-store guest code would have.
//
// Exactness: each matrix word goes through the same conversion the generated
// code applies - psq_l/psq_st (GQR type 0) is convert_to_double followed by
// convert_to_single_ftz, which flushes single denormals; lfs/stfs round-trips
// bit for bit. Anything the guest code would have handled differently
// (lazy-FP unavailable, a quantising GQR0, LSQE clear, an unreachable matrix
// pointer, no gather-pipe pointer from the runtime) declines before any side
// effect. Only volatile registers (r0, r4, r5, f0-f5) differ afterwards, which
// the ABI leaves undefined.

static inline int dolrecomp_gx_ps_ok(CPUState *ctx) {
  if (g_ppc_lazy_fp_enabled && !(ctx->msr & PPC_MSR_FP))
    return 0;
  if ((ctx->hid2 & PPC_HID2_LSQE) == 0u)
    return 0;
  const uint32_t gqr0 = ctx->gqr[0];
  return ((gqr0 >> 16) & 7u) == 0u && (gqr0 & 7u) == 0u;
}

static inline uint32_t dolrecomp_gx_ps_word(const uint8_t *p) {
  return convert_to_single_ftz(convert_to_double(read_be32(p)));
}

// Append `n` (>= 5) bytes of an already-big-endian command stream. Returns 0,
// touching nothing, if the runtime gave no gather-pipe pointer.
static inline int dolrecomp_gx_emit(CPUState *ctx, const uint8_t *bytes, uint32_t n) {
  uint8_t **gp = g_dolrecomp_gather_pipe ? g_dolrecomp_gather_pipe
                                         : dolrecomp_gather_pipe_slow(ctx);
  if (gp == (uint8_t **)1 || !ctx->external_write)
    return 0;
  uint8_t *p = gp[0];
  const size_t queued = (size_t)(p - gp[1]);
  // GPFifo keeps 16 bursts of slack beyond the 32-byte window; stay well
  // inside it (callers emit at most 53 bytes onto fewer than 32 queued).
  if (queued + n > 32u * 8u)
    return 0;
  memcpy(p, bytes, n - 4u);
  p += n - 4u;
  const uint32_t last = read_be32(bytes + n - 4u);
  if (queued + n < 32u) {
    write_be32(p, last);
    gp[0] = p + 4;
  } else {
    gp[0] = p;
    ctx->external_write(ctx, DOLRECOMP_GATHER_PIPE_PAGE, last, 4);
  }
  return 1;
}

// void GXLoadPosMtxImm(Mtx mtx, u32 id)   0x800BD4B4
int dolrecomp_native_gx_load_pos_mtx_imm(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  if (!dolrecomp_gx_ps_ok(ctx))
    return dolrecomp_declined(DOLRECOMP_SUBST_GX_POS_MTX);
  const uint8_t *m = dolrecomp_guest_bytes_ext(ctx, (uint32_t)ctx->gpr[3], 48u);
  if (!m)
    return dolrecomp_declined(DOLRECOMP_SUBST_GX_POS_MTX);
  uint8_t s[53];
  s[0] = 0x10;
  write_be32(s + 1, (((uint32_t)ctx->gpr[4]) << 2) | 0x000B0000u);
  for (int i = 0; i < 12; i++)
    write_be32(s + 5 + 4 * i, dolrecomp_gx_ps_word(m + 4 * i));
  if (!dolrecomp_gx_emit(ctx, s, sizeof(s)))
    return dolrecomp_declined(DOLRECOMP_SUBST_GX_POS_MTX);
  ctx->pc = ctx->lr;
  return dolrecomp_handled(DOLRECOMP_SUBST_GX_POS_MTX);
}

// void GXLoadNrmMtxImm(Mtx mtx, u32 id)   0x800BD504
// Columns 0-1 of each row go through psq, column 2 through lfs/stfs.
int dolrecomp_native_gx_load_nrm_mtx_imm(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  if (!dolrecomp_gx_ps_ok(ctx))
    return dolrecomp_declined(DOLRECOMP_SUBST_GX_NRM_MTX);
  const uint8_t *m = dolrecomp_guest_bytes_ext(ctx, (uint32_t)ctx->gpr[3], 44u);
  if (!m)
    return dolrecomp_declined(DOLRECOMP_SUBST_GX_NRM_MTX);
  uint8_t s[41];
  s[0] = 0x10;
  write_be32(s + 1, ((uint32_t)ctx->gpr[4] * 3u + 0x400u) | 0x00080000u);
  for (int row = 0; row < 3; row++) {
    const uint8_t *r = m + 16 * row;
    write_be32(s + 5 + 12 * row + 0, dolrecomp_gx_ps_word(r + 0));
    write_be32(s + 5 + 12 * row + 4, dolrecomp_gx_ps_word(r + 4));
    memcpy(s + 5 + 12 * row + 8, r + 8, 4);
  }
  if (!dolrecomp_gx_emit(ctx, s, sizeof(s)))
    return dolrecomp_declined(DOLRECOMP_SUBST_GX_NRM_MTX);
  ctx->pc = ctx->lr;
  return dolrecomp_handled(DOLRECOMP_SUBST_GX_NRM_MTX);
}

// void GXLoadTexMtxImm(Mtx mtx, u32 id, GXTexMtxType type)   0x800BD58C
// Register count is 8 for type 1 else 12; the copy is 4x3 only for type 0.
int dolrecomp_native_gx_load_tex_mtx_imm(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  if (!dolrecomp_gx_ps_ok(ctx))
    return dolrecomp_declined(DOLRECOMP_SUBST_GX_TEX_MTX);
  const uint32_t id = (uint32_t)ctx->gpr[4];
  const uint32_t type = (uint32_t)ctx->gpr[5];
  const uint32_t words = type == 0u ? 12u : 8u;
  const uint8_t *m = dolrecomp_guest_bytes_ext(ctx, (uint32_t)ctx->gpr[3], words * 4u);
  if (!m)
    return dolrecomp_declined(DOLRECOMP_SUBST_GX_TEX_MTX);
  const uint32_t addr = id >= 0x40u ? (id - 0x40u) * 4u + 0x500u : id * 4u;
  const uint32_t count = type == 1u ? 8u : 12u;
  uint8_t s[53];
  s[0] = 0x10;
  write_be32(s + 1, addr | ((count - 1u) << 16));
  for (uint32_t i = 0; i < words; i++)
    write_be32(s + 5 + 4 * i, dolrecomp_gx_ps_word(m + 4 * i));
  if (!dolrecomp_gx_emit(ctx, s, 5u + words * 4u))
    return dolrecomp_declined(DOLRECOMP_SUBST_GX_TEX_MTX);
  ctx->pc = ctx->lr;
  return dolrecomp_handled(DOLRECOMP_SUBST_GX_TEX_MTX);
}


// ---------------------------------------------------------------------------
// GS command recorder.
//
// void fn_800D4F98(u32 opcode, u32 paramCount, ...)   0x800D4F98
//
// Colosseum's GS layer records most rendering calls into a command buffer
// (replayed later by fn_800D461C): the opcode, then paramCount's arguments
// fetched through __va_arg. Each call pays a 144-byte frame, the register
// save area and one __va_arg call per argument.
//
// The varargs walk is the SDK's (gs_range_800C45A0.c): GPR class from
// r5..r10 then the caller's overflow area (r1 + 8), FPR class from f1..f8
// then the overflow area aligned to 8. Floats are recorded as the guest does
// it - stfd of the register, lfd, frsp, stfs - so ppc_frsp runs on f0 in the
// same order and leaves the same f0 and FPSCR. Everything is read and
// resolved first; an unreachable pointer, a float taken from f1..f8 without
// CR6 set (the guest would read a stale stack slot), or a command buffer that
// overlaps anything read declines before any side effect. Only the callee's
// dead stack frame and volatile registers other than f0 differ.

enum { GSQ_WORD, GSQ_FLOAT, GSQ_DEREF, GSQ_COPY };
typedef struct {
  uint8_t op;
  uint8_t bytes;    // GSQ_COPY length
  uint8_t advance;  // GSQ_COPY write-pointer step
} DolrecompGsqStep;

// paramCount 1..10 record that many words, 11..14 that many minus 10 floats;
// 0 and anything above 20 record only the opcode.
static const DolrecompGsqStep dolrecomp_gsq_special[6][4] = {
    /* 15 */ {{GSQ_WORD, 0, 0}, {GSQ_FLOAT, 0, 0}, {GSQ_FLOAT, 0, 0}},
    /* 16 */ {{GSQ_COPY, 0x30, 0x30}},
    /* 17 */ {{GSQ_WORD, 0, 0}, {GSQ_COPY, 0x30, 0x30}},
    /* 18 */ {{GSQ_WORD, 0, 0}, {GSQ_COPY, 0x18, 0x60}, {GSQ_WORD, 0, 0}},
    /* 19 */ {{GSQ_WORD, 0, 0}, {GSQ_WORD, 0, 0}, {GSQ_WORD, 0, 0}, {GSQ_COPY, 0x30, 0x30}},
    /* 20 */ {{GSQ_WORD, 0, 0}, {GSQ_DEREF, 0, 0}},
};
static const uint8_t dolrecomp_gsq_special_len[6] = {3, 1, 2, 3, 4, 2};

typedef struct {
  uint8_t op;
  uint32_t word;       // GSQ_WORD / GSQ_DEREF value
  uint64_t dbits;      // GSQ_FLOAT: the stfd image
  const uint8_t *src;  // GSQ_COPY source
  uint32_t bytes, advance;
} DolrecompGsqItem;

static inline int dolrecomp_ranges_overlap(const uint8_t *a, uint32_t an,
                                           const uint8_t *b, uint32_t bn) {
  return a < b + bn && b < a + an;
}

int dolrecomp_native_gs_queue_command(CPUState *ctx) {
  if (!dolrecomp_subst_enabled())
    return 0;
  const int which = DOLRECOMP_SUBST_GS_QUEUE_COMMAND;
  const uint32_t opcode = (uint32_t)ctx->gpr[3];
  const uint32_t count = (uint32_t)ctx->gpr[4];
  const int cr6 = (ctx->cr & 0x02000000u) != 0;
  const int fp_ok = !g_ppc_lazy_fp_enabled || (ctx->msr & PPC_MSR_FP);
  if (cr6 && !fp_ok)  // the prologue's stfd f1..f8 would trap
    return dolrecomp_declined(which);

  DolrecompGsqStep script[10];
  uint32_t steps = 0;
  if (count >= 1 && count <= 10) {
    for (; steps < count; steps++)
      script[steps] = (DolrecompGsqStep){GSQ_WORD, 0, 0};
  } else if (count >= 11 && count <= 14) {
    for (; steps < count - 10; steps++)
      script[steps] = (DolrecompGsqStep){GSQ_FLOAT, 0, 0};
  } else if (count >= 15 && count <= 20) {
    for (; steps < dolrecomp_gsq_special_len[count - 15]; steps++)
      script[steps] = dolrecomp_gsq_special[count - 15][steps];
  }

  // Phase 1: fetch every argument the way __va_arg would.
  const uint8_t *reads[12];
  uint32_t read_bytes[12];
  uint32_t nreads = 0;
  DolrecompGsqItem items[10];
  uint32_t gpr = 2, fpr = 0;
  uint32_t overflow = (uint32_t)ctx->gpr[1] + 8u;
  uint32_t span = 4;  // the opcode
  for (uint32_t i = 0; i < steps; i++) {
    DolrecompGsqItem *it = &items[i];
    it->op = script[i].op;
    if (it->op == GSQ_FLOAT) {
      if (!fp_ok)
        return dolrecomp_declined(which);
      if (fpr < 8) {
        if (!cr6)
          return dolrecomp_declined(which);
        it->dbits = f64_bits(ctx->fpr[1 + fpr]);
        fpr++;
      } else {
        overflow = (overflow + 7u) & ~7u;
        const uint8_t *p = dolrecomp_guest_bytes_ext(ctx, overflow, 8u);
        if (!p)
          return dolrecomp_declined(which);
        it->dbits = read_be64(p);
        reads[nreads] = p;
        read_bytes[nreads++] = 8;
        overflow += 8u;
      }
      span += 4;
      continue;
    }
    uint32_t value;
    if (gpr < 8) {
      value = (uint32_t)ctx->gpr[3 + gpr];
      gpr++;
    } else {
      const uint8_t *p = dolrecomp_guest_bytes_ext(ctx, overflow, 4u);
      if (!p)
        return dolrecomp_declined(which);
      value = read_be32(p);
      reads[nreads] = p;
      read_bytes[nreads++] = 4;
      overflow += 4u;
    }
    if (it->op == GSQ_WORD) {
      it->word = value;
      span += 4;
    } else if (it->op == GSQ_DEREF) {
      const uint8_t *p = dolrecomp_guest_bytes_ext(ctx, value, 4u);
      if (!p)
        return dolrecomp_declined(which);
      it->word = read_be32(p);
      reads[nreads] = p;
      read_bytes[nreads++] = 4;
      span += 4;
    } else {
      it->bytes = script[i].bytes;
      it->advance = script[i].advance;
      it->src = dolrecomp_guest_bytes_ext(ctx, value, it->bytes);
      if (!it->src)
        return dolrecomp_declined(which);
      reads[nreads] = it->src;
      read_bytes[nreads++] = it->bytes;
      span += it->advance;
    }
  }

  // The GS state block (r13 - 23968) and its command write pointer (+0x494).
  const uint8_t *sp = dolrecomp_guest_bytes_ext(ctx, (uint32_t)ctx->gpr[13] - 23968u, 4u);
  if (!sp)
    return dolrecomp_declined(which);
  uint8_t *wp = dolrecomp_guest_bytes_ext(ctx, read_be32(sp) + 0x494u, 4u);
  if (!wp)
    return dolrecomp_declined(which);
  const uint32_t cmd_addr = read_be32(wp);
  uint8_t *cmd = dolrecomp_guest_bytes_ext(ctx, cmd_addr, span);
  if (!cmd || dolrecomp_ranges_overlap(cmd, span, wp, 4u) ||
      dolrecomp_ranges_overlap(cmd, span, sp, 4u))
    return dolrecomp_declined(which);
  for (uint32_t i = 0; i < nreads; i++)
    if (dolrecomp_ranges_overlap(cmd, span, reads[i], read_bytes[i]))
      return dolrecomp_declined(which);

  // Phase 2: record.
  uint8_t *out = cmd;
  write_be32(out, opcode);
  out += 4;
  for (uint32_t i = 0; i < steps; i++) {
    const DolrecompGsqItem *it = &items[i];
    if (it->op == GSQ_FLOAT) {
      ctx->fpr[0] = f64_value(it->dbits);
      ppc_frsp(ctx, 0, 0);
      write_be32(out, convert_to_single(f64_bits(ctx->fpr[0])));
      out += 4;
    } else if (it->op == GSQ_COPY) {
      memmove(out, it->src, it->bytes);
      out += it->advance;
    } else {
      write_be32(out, it->word);
      out += 4;
    }
  }
  write_be32(wp, cmd_addr + span);
  ctx->pc = ctx->lr;
  return dolrecomp_handled(which);
}
