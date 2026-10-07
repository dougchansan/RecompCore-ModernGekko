// SPDX-License-Identifier: GPL-2.0-or-later
#ifndef STATICRECOMP_SDK_LEAF_GUARD_H
#define STATICRECOMP_SDK_LEAF_GUARD_H
#include "cpu/cpu.h"
#ifndef DOLRECOMP_C_LOOP_CYCLE_BUDGET
#define DOLRECOMP_C_LOOP_CYCLE_BUDGET 256
#endif
typedef int (*StaticRecompSdkGuard)(CPUState*, u32, u32, void*);
#if defined(_WIN32)
#define STATICRECOMP_SDK_EXPORT __declspec(dllexport)
#else
#define STATICRECOMP_SDK_EXPORT __attribute__((visibility("default")))
#endif
STATICRECOMP_SDK_EXPORT void staticrecomp_set_sdk_guard_v1(StaticRecompSdkGuard guard, void* user);
// 0: not handled, PC stays at callee; 1: handled, safe to resume;
// 2: handled, return to chassis at resulting PC before executing more code.
int staticrecomp_try_sdk_leaf(CPUState* cpu, u32 target, u32 continuation);
#endif
