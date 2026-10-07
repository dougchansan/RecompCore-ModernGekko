// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "Common/Config/Config.h"

namespace Config
{
extern const Info<bool> MAIN_STATICRECOMP_MODULE;
extern const Info<u32> MAIN_STATICRECOMP_IDLE_PC;
extern const Info<u32> MAIN_STATICRECOMP_IDLE_LOOP_END_PC;
extern const Info<std::string> MAIN_STATICRECOMP_FALLBACK_RANGES;

// Second, task-list-aware idle trigger. MAIN_STATICRECOMP_IDLE_PC only fires
// when the dispatcher regains control with pc == idle_pc, which a loop whose
// back edge is an intra-chunk branch never does. These name a PC the dispatcher
// *does* see -- the instruction after the loop's last bl -- plus the two guest
// list heads whose contents decide whether the coming iteration has any work.
extern const Info<u32> MAIN_STATICRECOMP_TASK_IDLE_PC;
extern const Info<u32> MAIN_STATICRECOMP_TASK_LIST_HEAD;
extern const Info<u32> MAIN_STATICRECOMP_TASK_PENDING_HEAD;
}  // namespace Config
