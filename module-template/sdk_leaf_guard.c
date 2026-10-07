// SPDX-License-Identifier: GPL-2.0-or-later
#include "sdk_leaf_guard.h"
extern int dolrecomp_native_psmtx_identity(CPUState*);
extern int dolrecomp_native_psmtx_copy(CPUState*);
extern int dolrecomp_native_psmtx_concat(CPUState*);
extern int dolrecomp_native_psmtx_mult_vec(CPUState*);
extern int dolrecomp_native_hsd_mtx_scaled_add(CPUState*);
static StaticRecompSdkGuard s_guard;
static void* s_user;
void staticrecomp_set_sdk_guard_v1(StaticRecompSdkGuard guard, void* user)
{
  s_guard = guard;
  s_user = user;
}
int staticrecomp_try_sdk_leaf(CPUState* cpu, u32 target, u32 continuation)
{
  int (*leaf)(CPUState*) = 0;
  switch (target)
  {
  case 0x800A2D38u: leaf = dolrecomp_native_psmtx_identity; break;
  case 0x800A2D64u: leaf = dolrecomp_native_psmtx_copy; break;
  case 0x800A2D98u: leaf = dolrecomp_native_psmtx_concat; break;
  case 0x800A37CCu: leaf = dolrecomp_native_psmtx_mult_vec; break;
  case 0x801A85F0u: leaf = dolrecomp_native_hsd_mtx_scaled_add; break;
  default: return 0;
  }
  if (!cpu || !s_guard || cpu->exception || cpu->pc != target ||
      cpu->lr != continuation ||
      cpu->downcount <= -(s64)DOLRECOMP_C_LOOP_CYCLE_BUDGET ||
      !s_guard(cpu, target, continuation, s_user))
    return 0;
  // Substitutions decline without architectural changes. A successful native
  // dispatch costs one cycle in the chassis; retain that charge when inlining.
  if (!leaf(cpu))
    return 0;
  cpu->downcount -= 1;
  if (cpu->pc != continuation || cpu->exception ||
      cpu->downcount <= -(s64)DOLRECOMP_C_LOOP_CYCLE_BUDGET ||
      !s_guard(cpu, target, continuation, s_user))
    return 2;
  return 1;
}
