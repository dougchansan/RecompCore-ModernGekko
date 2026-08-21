// Generated-table lookup for the kart debug overlay.
#pragma once

#include <cstddef>
#include <iterator>

#include "Common/CommonTypes.h"

namespace KartDebug
{
// Returns the function containing `address`, or nullptr when the address falls
// outside every known function. A containment test, not nearest-below: a PC in
// a gap reads as unknown rather than being attributed to the function before it.
const char* ResolveSymbol(u32 address, u32* offset);
std::size_t SymbolCount();
}  // namespace KartDebug
