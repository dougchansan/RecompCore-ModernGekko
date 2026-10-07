// SPDX-License-Identifier: GPL-2.0-or-later
#include "native_call_guard.h"
static StaticRecompCallGuard s_guard;
static void* s_user;
static unsigned s_depth;
static int s_unwind;
void staticrecomp_set_call_guard_v2(StaticRecompCallGuard guard, void* user)
{
  s_guard = guard;
  s_user = user;
  s_depth = 0;
  s_unwind = 0;
}
void staticrecomp_native_begin_dispatch(void)
{
  s_depth = 0;
  s_unwind = 0;
}
int staticrecomp_native_call_enter(CPUState* cpu, u32 target, u32 continuation)
{
  if (!cpu || !s_guard || s_unwind || s_depth >= 8 ||
      cpu->pc != target || cpu->lr != continuation ||
      !s_guard(cpu, target, continuation, 0, s_user))
  {
    s_unwind = 1;
    return 0;
  }
  ++s_depth;
  return 1;
}
int staticrecomp_native_call_finish(CPUState* cpu, u32 target, u32 continuation)
{
  if (!s_depth || !s_guard || !cpu)
  {
    s_unwind = 1;
    return 0;
  }
  --s_depth;
  const int allowed = s_guard(cpu, target, continuation, 1, s_user);
  if (!allowed || cpu->pc != continuation)
    s_unwind = 1;
  return allowed && !s_unwind;
}
