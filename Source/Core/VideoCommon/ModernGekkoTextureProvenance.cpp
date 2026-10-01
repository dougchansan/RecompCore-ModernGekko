// SPDX-License-Identifier: GPL-2.0-or-later
#include "VideoCommon/ModernGekkoTextureProvenance.h"
#include <array>
#include "VideoCommon/BPMemory.h"

namespace ModernGekkoTextureProvenance
{
namespace
{
struct Register
{
  bool rgb = false;
  bool alpha = false;
};
bool ColorInput(TevColorArg arg, const std::array<Register, 4>& regs, bool texture)
{
  const u32 value = static_cast<u32>(arg);
  if (value < 8)
    return value % 2 == 0 ? regs[value / 2].rgb : regs[value / 2].alpha;
  return arg == TevColorArg::TexColor || arg == TevColorArg::TexAlpha ? texture : false;
}
bool AlphaInput(TevAlphaArg arg, const std::array<Register, 4>& regs, bool texture)
{
  const u32 value = static_cast<u32>(arg);
  if (value < 4)
    return regs[value].alpha;
  return arg == TevAlphaArg::TexAlpha ? texture : false;
}
template <typename Arg>
bool LerpDependency(Arg a, Arg b, Arg c, Arg zero, bool da, bool db, bool dc)
{
  // TEV masks a/b/c to 8 bits and interpolates a + (b-a)*(c+(c>>7))/256.
  // Equality of operand selectors establishes equality before simultaneous writes.
  if (a == b)
    return da;
  if (c == zero)
    return da;
  if (a == zero && b == zero)
    return false;
  return da || db || dc;
}
bool CompareDependency(bool da, bool db, bool dc, bool equal_operands,
                       bool zero_c, TevComparison comparison)
{
  if (zero_c)
    return false;
  if (equal_operands)
    return comparison == TevComparison::EQ ? dc : false;
  return da || db || dc;
}
}
bool CanUseAsOpacityData(const BPMemory& bp, u32 auxiliary_slots)
{
  if ((auxiliary_slots & ~0xffu) != 0)
    return false;
  if (auxiliary_slots == 0)
    return true;
  // last_texture can also replace/add depth outside the TEV combiner.
  if (bp.ztex2.op != ZTexOp::Disabled)
    return false;
  std::array<Register, 4> registers{};
  const u32 stages = bp.genMode.numtevstages + 1u;
  for (u32 stage = 0; stage < stages; ++stage)
  {
    const auto& indirect = bp.tevind[stage];
    if (indirect.IsActive() && indirect.bt < bp.genMode.numindstages &&
        (auxiliary_slots & (1u << bp.tevindref.getTexMap(indirect.bt))))
      return false;  // Includes indirect bump alpha; its raster channel needs separate proof.
    const auto& order = bp.tevorders[stage / 2];
    const bool texture = bp.genMode.numtexgens > 0 && order.getEnable(stage & 1) &&
                         (auxiliary_slots & (1u << order.getTexMap(stage & 1)));
    // Whole-texture dependency is conservative under every texture swap permutation.
    const auto& cc = bp.combiners[stage].colorC;
    const auto& ac = bp.combiners[stage].alphaC;
    const bool ca = ColorInput(cc.a, registers, texture);
    const bool cb = ColorInput(cc.b, registers, texture);
    const bool cc_dep = ColorInput(cc.c, registers, texture);
    const bool cd = ColorInput(cc.d, registers, texture);
    const bool aa = AlphaInput(ac.a, registers, texture);
    const bool ab = AlphaInput(ac.b, registers, texture);
    const bool av = AlphaInput(ac.c, registers, texture);
    const bool ad = AlphaInput(ac.d, registers, texture);
    bool rgb = cd;
    bool alpha = ad;
    if (cc.bias == TevBias::Compare)
      rgb |= CompareDependency(ca, cb, cc_dep, cc.a == cc.b, cc.c == TevColorArg::Zero,
                               cc.comparison);
    else if (cc.c == TevColorArg::One)
      rgb |= cb;
    else
      rgb |= LerpDependency(cc.a.Value(), cc.b.Value(), cc.c.Value(), TevColorArg::Zero,
                            ca, cb, cc_dep);
    if (ac.bias == TevBias::Compare)
    {
      // R8/GR16/BGR24 alpha comparisons test COLOR a/b, not alpha a/b.
      const bool alpha_compare = ac.compare_mode == TevCompareMode::A8;
      alpha |= CompareDependency(alpha_compare ? aa : ca, alpha_compare ? ab : cb, av,
                                 alpha_compare ? ac.a == ac.b : cc.a == cc.b,
                                 ac.c == TevAlphaArg::Zero, ac.comparison);
    }
    else
      alpha |= LerpDependency(ac.a.Value(), ac.b.Value(), ac.c.Value(), TevAlphaArg::Zero,
                              aa, ab, av);
    // Both combinations read the old registers before either destination changes.
    registers[static_cast<u32>(cc.dest.Value())].rgb = rgb;
    registers[static_cast<u32>(ac.dest.Value())].alpha = alpha;
  }
  // The final stage's destination is displayed even if it is not PREV.
  const auto last = bp.combiners[stages - 1].colorC.dest.Value();
  return !registers[static_cast<u32>(last)].rgb;
}
}
