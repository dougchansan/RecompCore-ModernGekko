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
  u32 rgb = 0;
  u32 alpha = 0;
};
u32 ColorInput(TevColorArg arg, const std::array<Register, 4>& regs, u32 texture)
{
  const u32 value = static_cast<u32>(arg);
  if (value < 8)
    return value % 2 == 0 ? regs[value / 2].rgb : regs[value / 2].alpha;
  return arg == TevColorArg::TexColor || arg == TevColorArg::TexAlpha ? texture : 0u;
}
u32 AlphaInput(TevAlphaArg arg, const std::array<Register, 4>& regs, u32 texture)
{
  const u32 value = static_cast<u32>(arg);
  if (value < 4)
    return regs[value].alpha;
  return arg == TevAlphaArg::TexAlpha ? texture : 0u;
}
template <typename Arg>
u32 LerpDependency(Arg a, Arg b, Arg c, Arg zero, u32 da, u32 db, u32 dc)
{
  // TEV masks a/b/c to 8 bits and interpolates a + (b-a)*(c+(c>>7))/256.
  // Equality of operand selectors establishes equality before simultaneous writes.
  if (a == b)
    return da;
  if (c == zero)
    return da;
  if (a == zero && b == zero)
    return 0;
  return da | db | dc;
}
u32 CompareDependency(u32 da, u32 db, u32 dc, bool equal_operands,
                       bool zero_c, TevComparison comparison)
{
  if (zero_c)
    return 0;
  if (equal_operands)
    return comparison == TevComparison::EQ ? dc : 0u;
  return da | db | dc;
}
}
DependencyReport AnalyzeDependencies(const BPMemory& bp, u32 auxiliary_slots)
{
  DependencyReport report;
  report.invalid_slots = (auxiliary_slots & ~0xffu) != 0;
  auxiliary_slots &= 0xffu;
  // Preserve the existing conservative policy for the outside-combiner depth path.
  if (bp.ztex2.op != ZTexOp::Disabled)
    report.z_slots = auxiliary_slots;
  std::array<Register, 4> registers{};
  const u32 stages = bp.genMode.numtevstages + 1u;
  for (u32 stage = 0; stage < stages; ++stage)
  {
    const auto& indirect = bp.tevind[stage];
    if (indirect.IsActive() && indirect.bt < bp.genMode.numindstages)
      report.indirect_slots |= auxiliary_slots & (1u << bp.tevindref.getTexMap(indirect.bt));
    // Includes indirect bump alpha; no eligibility is inferred from its consumer role.
    const auto& order = bp.tevorders[stage / 2];
    const u32 texture = bp.genMode.numtexgens > 0 && order.getEnable(stage & 1) ?
                            auxiliary_slots & (1u << order.getTexMap(stage & 1)) : 0u;
    // Whole-texture dependency is conservative under every texture swap permutation.
    const auto& cc = bp.combiners[stage].colorC;
    const auto& ac = bp.combiners[stage].alphaC;
    const u32 ca = ColorInput(cc.a, registers, texture);
    const u32 cb = ColorInput(cc.b, registers, texture);
    const u32 cc_dep = ColorInput(cc.c, registers, texture);
    const u32 cd = ColorInput(cc.d, registers, texture);
    const u32 aa = AlphaInput(ac.a, registers, texture);
    const u32 ab = AlphaInput(ac.b, registers, texture);
    const u32 av = AlphaInput(ac.c, registers, texture);
    const u32 ad = AlphaInput(ac.d, registers, texture);
    u32 rgb = cd;
    u32 alpha = ad;
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
  const auto& last = bp.combiners[stages - 1];
  report.rgb_slots = registers[static_cast<u32>(last.colorC.dest.Value())].rgb;
  report.alpha_slots = registers[static_cast<u32>(last.alphaC.dest.Value())].alpha;
  return report;
}
bool CanUseAsOpacityData(const BPMemory& bp, u32 auxiliary_slots)
{
  const auto report = AnalyzeDependencies(bp, auxiliary_slots);
  return !report.invalid_slots && report.rgb_slots == 0 && report.indirect_slots == 0 &&
         report.z_slots == 0;
}
}
