// Native substitutions for profile-identified hot guest functions.
//
// DolRecomp emits a stub for each guest address named in
// DOLRECOMP_SUBSTITUTIONS (addr=symbol,...) at module generation time; the
// stub calls the symbol with the CPUState and then returns to the guest LR.
// The implementations below mutate guest-visible state exactly as the
// original guest function would: argument registers are read from ctx->gpr /
// ctx->fpr per the PowerPC EABI, memory goes through the runtime's mem_*
// helpers so RAM, EXRAM, MMIO and the locked cache all keep their existing
// semantics (including the write journal), and the return value lands in r3.
//
// Callers can only rely on what the guest ABI guarantees, so scratch
// registers are deliberately left untouched.
//
// Each substitute charges an approximate Gekko cycle count into
// ctx->downcount so guest time keeps advancing at a plausible rate; the
// figures are rough (loads/stores plus loop overhead), not calibrated.

#include "core/cpu.h"

static inline f32 sub_read_f32(CPUState* cpu, u32 addr) {
    union { u32 u; f32 f; } v;
    v.u = mem_read32(cpu, addr);
    return v.f;
}

static inline void sub_write_f32(CPUState* cpu, u32 addr, f32 value) {
    union { u32 u; f32 f; } v;
    v.f = value;
    mem_write32(cpu, addr, v.u);
}

static inline void sub_charge(CPUState* cpu, s64 cycles) {
    cpu->downcount -= cycles;
}

// PSMTXIdentity(Mtx m /*r3*/)
void dolrecomp_native_psmtx_identity(CPUState* cpu) {
    u32 m = cpu->gpr[3];
    for (u32 row = 0; row < 3; row++)
        for (u32 col = 0; col < 4; col++)
            sub_write_f32(cpu, m + (row * 4u + col) * 4u, row == col ? 1.0f : 0.0f);
    sub_charge(cpu, 30);
}

// PSMTXCopy(const Mtx src /*r3*/, Mtx dst /*r4*/)
void dolrecomp_native_psmtx_copy(CPUState* cpu) {
    u32 src = cpu->gpr[3];
    u32 dst = cpu->gpr[4];
    for (u32 i = 0; i < 12u; i++)
        mem_write32(cpu, dst + i * 4u, mem_read32(cpu, src + i * 4u));
    sub_charge(cpu, 30);
}

// PSMTXConcat(const Mtx a /*r3*/, const Mtx b /*r4*/, Mtx ab /*r5*/)
// ab may alias a or b, so compute into a temporary first, like the SDK does.
void dolrecomp_native_psmtx_concat(CPUState* cpu) {
    u32 a = cpu->gpr[3];
    u32 b = cpu->gpr[4];
    u32 ab = cpu->gpr[5];
    f32 ma[12], mb[12], mo[12];
    for (u32 i = 0; i < 12u; i++) {
        ma[i] = sub_read_f32(cpu, a + i * 4u);
        mb[i] = sub_read_f32(cpu, b + i * 4u);
    }
    for (u32 row = 0; row < 3u; row++) {
        for (u32 col = 0; col < 4u; col++) {
            f32 sum = ma[row * 4u + 0u] * mb[0u * 4u + col] +
                      ma[row * 4u + 1u] * mb[1u * 4u + col] +
                      ma[row * 4u + 2u] * mb[2u * 4u + col];
            if (col == 3u)
                sum += ma[row * 4u + 3u];
            mo[row * 4u + col] = sum;
        }
    }
    for (u32 i = 0; i < 12u; i++)
        sub_write_f32(cpu, ab + i * 4u, mo[i]);
    sub_charge(cpu, 120);
}

// PSMTXMultVec(const Mtx m /*r3*/, const Vec* src /*r4*/, Vec* dst /*r5*/)
void dolrecomp_native_psmtx_mult_vec(CPUState* cpu) {
    u32 m = cpu->gpr[3];
    u32 src = cpu->gpr[4];
    u32 dst = cpu->gpr[5];
    f32 v[3], o[3];
    for (u32 i = 0; i < 3u; i++)
        v[i] = sub_read_f32(cpu, src + i * 4u);
    for (u32 row = 0; row < 3u; row++) {
        o[row] = sub_read_f32(cpu, m + (row * 4u + 0u) * 4u) * v[0] +
                 sub_read_f32(cpu, m + (row * 4u + 1u) * 4u) * v[1] +
                 sub_read_f32(cpu, m + (row * 4u + 2u) * 4u) * v[2] +
                 sub_read_f32(cpu, m + (row * 4u + 3u) * 4u);
    }
    for (u32 i = 0; i < 3u; i++)
        sub_write_f32(cpu, dst + i * 4u, o[i]);
    sub_charge(cpu, 40);
}

// HSD_MtxScaledAdd(f32* src /*r3*/, f32 scale /*f1*/, f32* add /*r4*/, f32* dst /*r5*/)
// dst[i] = scale * src[i] + add[i] for the 12 matrix elements.
void dolrecomp_native_hsd_mtx_scaled_add(CPUState* cpu) {
    u32 src = cpu->gpr[3];
    u32 add = cpu->gpr[4];
    u32 dst = cpu->gpr[5];
    f32 scale = (f32)cpu->fpr[1];
    for (u32 i = 0; i < 12u; i++)
        sub_write_f32(cpu, dst + i * 4u,
                      scale * sub_read_f32(cpu, src + i * 4u) +
                          sub_read_f32(cpu, add + i * 4u));
    sub_charge(cpu, 70);
}

// memcpy(void* dst /*r3*/, const void* src /*r4*/, size_t n /*r5*/) -> r3
void dolrecomp_native_memcpy(CPUState* cpu) {
    u32 dst = cpu->gpr[3];
    u32 src = cpu->gpr[4];
    u32 n = cpu->gpr[5];
    // Fast path: both ranges resolve to host memory with no wraparound.
    u32 dst_offset = (u32)-1;
    u8* dp = n ? get_ram_ptr(cpu, dst, n, &dst_offset) : NULL;
    u8* sp = n ? get_ram_ptr(cpu, src, n, NULL) : NULL;
    if (dp && sp) {
        clear_matching_reservation(cpu, dst);
        if (g_mem_write_journal && dst_offset != (u32)-1)
            g_mem_write_journal(dst_offset, n, g_mem_write_journal_user);
        memmove(dp, sp, n);
    } else {
        for (u32 i = 0; i < n; i++)
            mem_write8(cpu, dst + i, mem_read8(cpu, src + i));
    }
    cpu->gpr[3] = dst;
    sub_charge(cpu, 8 + (s64)(n / 4u));
}

// memset(void* dst /*r3*/, int value /*r4*/, size_t n /*r5*/) -> r3
void dolrecomp_native_memset(CPUState* cpu) {
    u32 dst = cpu->gpr[3];
    u8 value = (u8)cpu->gpr[4];
    u32 n = cpu->gpr[5];
    u32 dst_offset = (u32)-1;
    u8* dp = n ? get_ram_ptr(cpu, dst, n, &dst_offset) : NULL;
    if (dp) {
        clear_matching_reservation(cpu, dst);
        if (g_mem_write_journal && dst_offset != (u32)-1)
            g_mem_write_journal(dst_offset, n, g_mem_write_journal_user);
        memset(dp, value, n);
    } else {
        for (u32 i = 0; i < n; i++)
            mem_write8(cpu, dst + i, value);
    }
    cpu->gpr[3] = dst;
    sub_charge(cpu, 8 + (s64)(n / 4u));
}
