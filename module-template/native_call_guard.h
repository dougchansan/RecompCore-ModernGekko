// SPDX-License-Identifier: GPL-2.0-or-later
#ifndef STATICRECOMP_NATIVE_CALL_GUARD_H
#define STATICRECOMP_NATIVE_CALL_GUARD_H
#include "cpu/cpu.h"
typedef int (*StaticRecompCallGuard)(CPUState*, u32, u32, u32, void*);
#if defined(_WIN32)
#define STATICRECOMP_CALL_EXPORT __declspec(dllexport)
#else
#define STATICRECOMP_CALL_EXPORT __attribute__((visibility("default")))
#endif
STATICRECOMP_CALL_EXPORT void staticrecomp_set_call_guard_v2(StaticRecompCallGuard guard, void* user);
void staticrecomp_native_begin_dispatch(void);
int staticrecomp_native_call_enter(CPUState* cpu, u32 target, u32 continuation);
int staticrecomp_native_call_finish(CPUState* cpu, u32 target, u32 continuation);
#endif
