// SPDX-License-Identifier: GPL-2.0-or-later
#include "Core/Config/StaticRecompSettings.h"
#include "Core/System.h"

namespace Config
{
const Info<bool> MAIN_STATICRECOMP_MODULE{{System::Main, "Core", "StaticRecompModule"}, true};
const Info<u32> MAIN_STATICRECOMP_IDLE_PC{{System::Main, "Core", "StaticRecompIdlePC"}, 0};
const Info<u32> MAIN_STATICRECOMP_IDLE_LOOP_END_PC{
    {System::Main, "Core", "StaticRecompIdleLoopEndPC"}, 0};
const Info<std::string> MAIN_STATICRECOMP_FALLBACK_RANGES{
    {System::Main, "Core", "StaticRecompFallbackRanges"}, ""};
const Info<u32> MAIN_STATICRECOMP_TASK_IDLE_PC{
    {System::Main, "Core", "StaticRecompTaskIdlePC"}, 0};
const Info<u32> MAIN_STATICRECOMP_TASK_LIST_HEAD{
    {System::Main, "Core", "StaticRecompTaskListHead"}, 0};
const Info<u32> MAIN_STATICRECOMP_TASK_PENDING_HEAD{
    {System::Main, "Core", "StaticRecompTaskPendingHead"}, 0};
}  // namespace Config
