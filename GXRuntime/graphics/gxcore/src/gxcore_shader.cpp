// SPDX-License-Identifier: GPL-3.0-or-later
#include "gxruntime/gxcore/shader.hpp"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

namespace gxruntime::gxcore {

namespace {

std::uint32_t bits(std::uint32_t value, std::uint32_t count,

                   std::uint32_t shift) {
  return (value >> shift) & ((1u << count) - 1u);
}

// FogParam0/FogParam3 FloatValue (Dolphin BPMemory.cpp): mant(11)/exp(8)/sign(1)
// re-expanded to an IEEE-754 f32 (mantissa scaled from 11 to 23 bits).
float fog_param_float(std::uint32_t raw) {
  const std::uint32_t integral =
      (bits(raw, 1, 19) << 31) | (bits(raw, 8, 11) << 23) | (bits(raw, 11, 0) << 12);
  float f;
  std::memcpy(&f, &integral, sizeof f);
  return f;
}

// --- WGSL emission -----------------------------------------------------------

void emit(std::string& out, const char* text) { out += text; }

void emitf(std::string& out, const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof buffer, fmt, args);
  va_end(args);
  out += buffer;
}

} // namespace

namespace {

// --- Integer TEV fragment (Dolphin PixelShaderGen port) ----------------------
//
// Tables mirror PixelShaderGen.cpp; GLSL int3/int4 become WGSL vec3i/vec4i.
// WGSL shifts require a matching-width unsigned shift operand (no scalar-on-
// vector broadcast), so rgb shifts use vec3u(Nu) while alpha shifts use Nu.

const char* const kTevCInput[16] = {
    "prev.rgb",           "prev.aaa",           "c0.rgb",   "c0.aaa",
    "c1.rgb",             "c1.aaa",             "c2.rgb",   "c2.aaa",
    "textemp.rgb",        "textemp.aaa",        "rastemp.rgb", "rastemp.aaa",
    "vec3i(255,255,255)", "vec3i(128,128,128)", "konsttemp.rgb", "vec3i(0,0,0)"};
const char* const kTevAInput[8] = {"prev.a",    "c0.a",       "c1.a",
                                   "c2.a",      "textemp.a",  "rastemp.a",
                                   "konsttemp.a", "0"};
// Destination register NAMES (not swizzles): WGSL forbids assigning to a
// multi-component swizzle, so combines write the whole vec4i, reconstructing
// the untouched channel (rgb keeps .a, alpha keeps .rgb).
const char* const kTevOutput[4] = {"prev", "c0", "c1", "c2"};
const char kRgbaSwizzle[4] = {'r', 'g', 'b', 'a'};

const char* const kKselC[32] = {
    "vec3i(255,255,255)", "vec3i(223,223,223)", "vec3i(191,191,191)",
    "vec3i(159,159,159)", "vec3i(128,128,128)", "vec3i(96,96,96)",
    "vec3i(64,64,64)",    "vec3i(32,32,32)",    "vec3i(0,0,0)",
    "vec3i(0,0,0)",       "vec3i(0,0,0)",       "vec3i(0,0,0)",
    "psc.kcolors[0].rgb", "psc.kcolors[1].rgb", "psc.kcolors[2].rgb",
    "psc.kcolors[3].rgb", "psc.kcolors[0].rrr", "psc.kcolors[1].rrr",
    "psc.kcolors[2].rrr", "psc.kcolors[3].rrr", "psc.kcolors[0].ggg",
    "psc.kcolors[1].ggg", "psc.kcolors[2].ggg", "psc.kcolors[3].ggg",
    "psc.kcolors[0].bbb", "psc.kcolors[1].bbb", "psc.kcolors[2].bbb",
    "psc.kcolors[3].bbb", "psc.kcolors[0].aaa", "psc.kcolors[1].aaa",
    "psc.kcolors[2].aaa", "psc.kcolors[3].aaa"};
const char* const kKselA[32] = {
    "255",  "223",  "191",  "159",  "128",  "96",   "64",   "32",
    "0",    "0",    "0",    "0",    "0",    "0",    "0",    "0",
    "psc.kcolors[0].r", "psc.kcolors[1].r", "psc.kcolors[2].r", "psc.kcolors[3].r",
    "psc.kcolors[0].g", "psc.kcolors[1].g", "psc.kcolors[2].g", "psc.kcolors[3].g",
    "psc.kcolors[0].b", "psc.kcolors[1].b", "psc.kcolors[2].b", "psc.kcolors[3].b",
    "psc.kcolors[0].a", "psc.kcolors[1].a", "psc.kcolors[2].a", "psc.kcolors[3].a"};

const char* const kAlphaLogic[4] = {" && ", " || ", " != ", " == "};

// Dolphin tev_alpha_funcs_table: prev.a <cmp> ref (Never/Always take no ref).
std::string alpha_cond(std::uint8_t comp, const char* ref) {
  switch (comp) {
  case 0: return "(false)";
  case 1: return std::string("(prev.a < ") + ref + ")";
  case 2: return std::string("(prev.a == ") + ref + ")";
  case 3: return std::string("(prev.a <= ") + ref + ")";
  case 4: return std::string("(prev.a > ") + ref + ")";
  case 5: return std::string("(prev.a != ") + ref + ")";
  case 6: return std::string("(prev.a >= ") + ref + ")";
  default: return "(true)";
  }
}

bool cc_uses(const TevStageKey& s, std::uint8_t arg) {
  return s.cc_a == arg || s.cc_b == arg || s.cc_c == arg || s.cc_d == arg;
}
bool ac_uses(const TevStageKey& s, std::uint8_t arg) {
  return s.ac_a == arg || s.ac_b == arg || s.ac_c == arg || s.ac_d == arg;
}

// Shift token that WGSL accepts for the given width. vec: vec3u operand.
std::string shl(bool vec, unsigned n) {
  char b[32];
  std::snprintf(b, sizeof b, vec ? " << vec3u(%uu)" : " << %uu", n);
  return b;
}
std::string shr(bool vec, unsigned n) {
  char b[32];
  std::snprintf(b, sizeof b, vec ? " >> vec3u(%uu)" : " >> %uu", n);
  return b;
}

// One "regular" combine (Dolphin WriteTevRegular): (d+bias)*scale +/- lerp,
// with the GC scale-fold-into-lerp and rounding bias. `comp` selects rgb/a.
std::string tev_regular(bool vec, const char* comp, std::uint8_t bias,
                        std::uint8_t op, std::uint8_t scale) {
  const std::string a = std::string("tevin_a.") + comp;
  const std::string b = std::string("tevin_b.") + comp;
  const std::string c = std::string("tevin_c.") + comp;
  const std::string d = std::string("tevin_d.") + comp;
  const char* biasstr = bias == 1 ? " + 128" : bias == 2 ? " - 128" : "";
  const char* opstr = op == 1 ? "-" : "+";
  const bool div2 = scale == 3;
  const unsigned sh = scale == 1 ? 1u : scale == 2 ? 2u : 0u; // Scale2/4 shift
  const char* lerpbias = div2 ? "" : (op == 1 ? " + 127" : " + 128");

  // lerp = (a<<8) + (b-a)*(c + (c>>7))   [c: 0..255 -> 0..256]
  std::string lerp = "((" + a + shl(vec, 8) + ") + (" + b + " - " + a +
                     ") * (" + c + " + (" + c + shr(vec, 7) + ")))";
  // fold scale into the lerp and add rounding bias (skipped for Divide2)
  if (!div2 && sh != 0)
    lerp = "((" + lerp + shl(vec, sh) + ")" + lerpbias + ")";
  else if (!div2)
    lerp = "(" + lerp + lerpbias + ")";
  lerp = "(" + lerp + shr(vec, 8) + ")";

  std::string dterm = "(" + d + biasstr + ")";
  if (!div2 && sh != 0)
    dterm = "(" + dterm + shl(vec, sh) + ")";
  std::string res = "(" + dterm + " " + opstr + " " + lerp + ")";
  if (div2)
    res = "(" + res + shr(vec, 1) + ")";
  return res;
}

// Compare-mode combine (bias==Compare): tevin_d + compare(a,b)?c:0.
std::string tev_compare(bool vec, std::uint8_t comparison,
                        std::uint8_t compare_mode) {
  const char* cmp = comparison == 1 ? "==" : ">"; // TevComparison EQ==1
  const char* comp = vec ? "rgb" : "a";
  const std::string d = std::string("tevin_d.") + comp;
  std::string zero = vec ? "vec3i(0,0,0)" : "0";
  std::string cval = vec ? "tevin_c.rgb" : "tevin_c.a";
  std::string cond;
  switch (compare_mode) {
  case 0: // R8
    cond = std::string("(tevin_a.r ") + cmp + " tevin_b.r)";
    break;
  case 1: // GR16
    cond = std::string("((tevin_a.r + tevin_a.g*256) ") + cmp +
           " (tevin_b.r + tevin_b.g*256))";
    break;
  case 2: // BGR24
    cond = std::string("((tevin_a.r + tevin_a.g*256 + tevin_a.b*65536) ") +
           cmp + " (tevin_b.r + tevin_b.g*256 + tevin_b.b*65536))";
    break;
  default: // RGB8/A8 componentwise
    if (comparison == 1)
      return "(" + d + " + (vec3i(1,1,1) - sign(abs(tevin_a.rgb - "
                       "tevin_b.rgb))) * tevin_c.rgb)";
    return "(" + d + " + max(sign(tevin_a.rgb - tevin_b.rgb), vec3i(0,0,0)) "
                     "* tevin_c.rgb)";
  }
  if (compare_mode == 3 && !vec) { // A8 uses .a comparison
    cond = std::string("(tevin_a.a ") + cmp + " tevin_b.a)";
  }
  return "(" + d + " + select(" + zero + ", " + cval + ", " + cond + "))";
}

// Fog (Dolphin PixelShaderGen WriteFog + PixelShaderManager fog constants).
// Applied to prev.rgb after the alpha test, exactly Dolphin's order. GC zCoord
// is the 24-bit screen depth (0=near, 0xFFFFFF=far); our substrate runs
// reversed-Z so in.pos.z is near=1/far=0 → zCoord = (1-in.pos.z)*2^24 (same as
// graphics/aurora lib/gx/shader.cpp:1501). Only called when fog_fsel != Off.
void emit_fog(std::string& out, const ShaderKey& key) {
  const auto fsel = static_cast<FogType>(key.fog_fsel);
  if (fsel == FogType::Off)
    return;
  emit(out, "    // Fog (Dolphin WriteFog)\n"
            "    var zCoord = i32((1.0 - in.pos.z) * 16777216.0);\n"
            "    zCoord = clamp(zCoord, 0, 16777215);\n");
  if (static_cast<FogProjection>(key.fog_proj) == FogProjection::Perspective) {
    // ze = A / (B - (Zs >> B_SHF))
    emit(out, "    var ze = (psc.fogf.x * 16777216.0) / "
              "f32(psc.fogi.y - (zCoord >> u32(psc.fogi.w)));\n");
  } else {
    // ze = A * Zs (orthographic; no B_SHF)
    emit(out, "    var ze = psc.fogf.x * f32(zCoord) / 16777216.0;\n");
  }
  // x_adjust = sqrt((x-center)^2 + k^2)/k, ze *= x_adjust.
  if (key.fog_range != 0) {
    emit(out,
         "    let offset = (2.0 * (in.pos.x / psc.fogf.w)) - 1.0 - psc.fogf.z;\n"
         "    let floatindex = clamp(9.0 - abs(offset) * 9.0, 0.0, 9.0);\n"
         "    let indexlower = u32(floatindex);\n"
         "    let indexupper = indexlower + 1u;\n"
         "    let klower = psc.fogrange[indexlower >> 2u][indexlower & 3u];\n"
         "    let kupper = psc.fogrange[indexupper >> 2u][indexupper & 3u];\n"
         "    let k = mix(klower, kupper, fract(floatindex));\n"
         "    let x_adjust = sqrt(offset * offset + k * k) / k;\n"
         "    ze = ze * x_adjust;\n");
  }
  emit(out, "    var fogv = clamp(ze - psc.fogf.y, 0.0, 1.0);\n");
  switch (fsel) {
  case FogType::Exp:
    emit(out, "    fogv = 1.0 - exp2(-8.0 * fogv);\n");
    break;
  case FogType::ExpSq:
    emit(out, "    fogv = 1.0 - exp2(-8.0 * fogv * fogv);\n");
    break;
  case FogType::BackwardsExp:
    emit(out, "    fogv = exp2(-8.0 * (1.0 - fogv));\n");
    break;
  case FogType::BackwardsExpSq:
    emit(out, "    fogv = 1.0 - fogv;\n"
              "    fogv = exp2(-8.0 * fogv * fogv);\n");
    break;
  case FogType::Linear:
  default:
    break; // linear: fogv unchanged
  }
  // Integer blend: prev.rgb = (prev.rgb*(256-ifog) + fogcolor*ifog) >> 8.
  emit(out, "    let ifog = i32(round(fogv * 256.0));\n"
            "    prev = vec4i((prev.rgb * (256 - ifog) + psc.fogcolor.rgb * "
            "ifog) >> vec3u(8u), prev.a);\n");
}

void emit_tev_fragment(std::string& out, const ShaderKey& key) {
  // Single-texmap (<=1 distinct texmap) samples tex0 exactly as before; a
  // multi-texmap TEV (e.g. THP YUV Y/U/V on texmap 0/1/2) samples tex{texmap}.
  const bool multi_texmap = texmap_popcount(used_texmap_mask(key)) > 1u;
  emit(out, "@fragment\nfn fs_main(in: VertexOut) -> @location(0) vec4f {\n");
  emit(out, "    var prev = psc.colors[0];\n"
            "    var c0 = psc.colors[1];\n"
            "    var c1 = psc.colors[2];\n"
            "    var c2 = psc.colors[3];\n"
            "    var rastemp = vec4i(0,0,0,0);\n"
            "    var rawtextemp = vec4i(0,0,0,0);\n"
            "    var textemp = vec4i(0,0,0,0);\n"
            "    var konsttemp = vec4i(0,0,0,0);\n"
            "    var tevin_a = vec4i(0,0,0,0);\n"
            "    var tevin_b = vec4i(0,0,0,0);\n"
            "    var tevin_c = vec4i(0,0,0,0);\n"
            "    var tevin_d = vec4i(0,0,0,0);\n"
            "    let col0i = vec4i(round(in.color0 * 255.0));\n"
            "    let col1i = vec4i(round(in.color1 * 255.0));\n");

  // Indirect texturing (Dolphin PixelShaderGen): on this path every stage's
  // texcoord is the s17.7 fixed-point tevcoord its indirect op produces, and
  // samples divide it back by the texmap's guest size (sampleTexture,
  // non-manual sampling; LOD bias is not applied, as on the plain path).
  const bool ind = tev_indirect_path(key);
  const bool can_sample = key.num_tex_gens > 0 && key.textured != 0;
  if (ind) {
    emit(out, "    var alphabump = 0;\n"
              "    var tevcoord = vec2i(0, 0);\n"
              "    var wrappedcoord = vec2i(0, 0);\n");
    // fixpoint_uv{i} = (uv / q) * I_TEXDIMS[i].zw * 128 (q == 0 keeps uv).
    if (key.num_tex_gens == 0)
      emit(out, "    let fixpoint_uv0 = vec2i(0, 0);\n");
    for (std::uint32_t i = 0; i < key.num_tex_gens; ++i)
      emitf(out,
            "    let fixpoint_uv%u = vec2i(select(in.uv%u.xy / in.uv%u.z, "
            "in.uv%u.xy, in.uv%u.z == 0.0) * vec2f(psc.texdims[%u].zw * 128));\n",
            i, i, i, i, i, i);
    // Indirect lookups: iindtex{i} = sample(texmap, fixpoint_uv >> texscale).abg.
    for (std::uint32_t i = 0; i < key.num_ind_stages && i < kMaxIndStages; ++i) {
      if ((key.ind_stage_used_mask & (1u << i)) == 0u)
        continue;
      std::uint32_t tc = key.ind_stages[i].texcoord;
      // Quirk (Dolphin): a texcoord past numtexgens reads texcoord 0.
      if (tc >= key.num_tex_gens)
        tc = 0;
      const std::uint32_t tmap = key.ind_stages[i].texmap & 7u;
      const std::uint32_t tunit = multi_texmap ? tmap : 0u;
      if (can_sample)
        emitf(out,
              "    let iindtex%u = vec4i(round(textureSample(tex%u, samp%u, "
              "vec2f(fixpoint_uv%u >> vec2u(psc.indtexscale[%u].%s)) / "
              "vec2f(psc.texdims[%u].xy * 128)) * 255.0)).abg;\n",
              i, tunit, tunit, tc, i / 2u, (i & 1u) ? "zw" : "xy", tmap);
      else
        emitf(out, "    let iindtex%u = vec3i(0, 0, 0);\n", i);
    }
  }

  for (std::uint32_t n = 0; n < key.num_tev_stages; ++n) {
    const TevStageKey& s = key.tev_stages[n];
    emitf(out, "    // TEV stage %u\n", n);

    std::uint32_t texcoord = s.tevorders_texcoord;
    if (texcoord >= key.num_tex_gens)
      texcoord = 0;

    // Indirect op (Dolphin WriteStage). A stage reading an indirect stage past
    // numindstages skips the offset; wrap and add-prev still apply.
    if (ind) {
      emit(out, "    // indirect op\n");
      const bool has_ind_stage = s.ind_bt < key.num_ind_stages;
      if (has_ind_stage && s.ind_bs != 0u) {
        // Bump alpha: the low bits of the selected component, 5 significant.
        static const char kAlphaSel[4] = {'x', 'x', 'y', 'z'};
        static const unsigned kAlphaShift[4] = {0u, 5u, 4u, 3u};
        emitf(out, "    alphabump = (iindtex%u.%c << %uu) & 248;\n", s.ind_bt,
              kAlphaSel[s.ind_bs & 3u], kAlphaShift[s.ind_fmt & 3u]);
      }
      if (has_ind_stage && s.ind_mtx_index != 0u) {
        // Format shift, then bias (-128 for ITF_8, else +1) on the selected
        // components.
        static const unsigned kFmtShift[4] = {0u, 3u, 4u, 5u};
        const int add = s.ind_fmt == 0u ? -128 : 1;
        emitf(out,
              "    let iindtevcrd%u = (iindtex%u >> vec3u(%uu)) + vec3i(%d, %d, %d);\n",
              n, s.ind_bt, kFmtShift[s.ind_fmt & 3u], (s.ind_bias & 1u) ? add : 0,
              (s.ind_bias & 2u) ? add : 0, (s.ind_bias & 4u) ? add : 0);
        // Matrix (each has two rows; .w is 17 - scale): static, or the
        // dynamic S/T forms that scale the regular coord by one offset.
        const std::uint32_t mtx = 2u * (s.ind_mtx_index - 1u);
        bool shift = true;
        if (s.ind_mtx_id == 0u)
          emitf(out,
                "    var indtevtrans%u = vec2i(dot(psc.indtexmtx[%u].xyz, iindtevcrd%u), "
                "dot(psc.indtexmtx[%u].xyz, iindtevcrd%u)) >> vec2u(3u);\n",
                n, mtx, n, mtx + 1u, n);
        else if (s.ind_mtx_id == 1u)
          emitf(out,
                "    var indtevtrans%u = (fixpoint_uv%u * iindtevcrd%u.xx) >> vec2u(8u);\n",
                n, texcoord, n);
        else if (s.ind_mtx_id == 2u)
          emitf(out,
                "    var indtevtrans%u = (fixpoint_uv%u * iindtevcrd%u.yy) >> vec2u(8u);\n",
                n, texcoord, n);
        else {
          emitf(out, "    var indtevtrans%u = vec2i(0, 0);\n", n);
          shift = false;
        }
        if (shift)
          emitf(out,
                "    { let w = psc.indtexmtx[%u].w; indtevtrans%u = select("
                "indtevtrans%u << vec2u(u32(-w)), indtevtrans%u >> vec2u(u32(w)), "
                "w >= 0); }\n",
                mtx, n, n, n);
      } else {
        emitf(out, "    let indtevtrans%u = vec2i(0, 0);\n", n);
      }
      // Wrap the regular coord: off, 256..16 texels (s17.7), or 0.
      auto wrapped = [&](std::uint8_t wrap, char comp) -> std::string {
        char b[96];
        if (wrap == 0u)
          std::snprintf(b, sizeof b, "fixpoint_uv%u.%c", texcoord, comp);
        else if (wrap >= 6u) // 7 (invalid) behaves as ITW_0
          std::snprintf(b, sizeof b, "0");
        else
          std::snprintf(b, sizeof b, "(fixpoint_uv%u.%c & ((%u << 7) - 1))", texcoord,
                        comp, 256u >> (wrap - 1u));
        return b;
      };
      emitf(out, "    wrappedcoord = vec2i(%s, %s);\n", wrapped(s.ind_sw, 'x').c_str(),
            wrapped(s.ind_tw, 'y').c_str());
      if (s.ind_fb_addprev != 0u)
        emitf(out, "    tevcoord = tevcoord + wrappedcoord + indtevtrans%u;\n", n);
      else
        emitf(out, "    tevcoord = wrappedcoord + indtevtrans%u;\n", n);
      // Emulate s24 overflows.
      emit(out, "    tevcoord = (tevcoord << vec2u(8u)) >> vec2u(8u);\n");
    }

    // Rasterized color, if referenced (RASC=10/RASA=11 color; RASA=5 alpha).
    // Channels 5/6 are the indirect bump alpha (Dolphin tev_ras_table), which
    // is zero off the indirect path.
    const bool uses_ras =
        cc_uses(s, 10) || cc_uses(s, 11) || ac_uses(s, 5);
    if (uses_ras) {
      const char* src = s.tevorders_colorchan == 0
                            ? "col0i"
                            : s.tevorders_colorchan == 1 ? "col1i"
                                                         : "vec4i(0,0,0,0)";
      if (ind && s.tevorders_colorchan == 5)
        src = "vec4i(alphabump)";
      else if (ind && s.tevorders_colorchan == 6)
        src = "vec4i(alphabump | (alphabump >> 5u))";
      emitf(out, "    rastemp = %s.%c%c%c%c;\n", src,
            kRgbaSwizzle[s.ras_swap[0]], kRgbaSwizzle[s.ras_swap[1]],
            kRgbaSwizzle[s.ras_swap[2]], kRgbaSwizzle[s.ras_swap[3]]);
    }

    // Texture sample for this stage. texunit is 0 on the single-texmap fast path
    // (byte-identical output) or the stage's texmap when >1 texmap is combined.
    const std::uint32_t texunit = multi_texmap ? (s.tevorders_texmap & 7u) : 0u;
    if (ind && s.tevorders_enable != 0 && can_sample) {
      emitf(out,
            "    rawtextemp = vec4i(round(textureSample(tex%u, samp%u, vec2f(tevcoord) / "
            "vec2f(psc.texdims[%u].xy * 128)) * 255.0));\n",
            texunit, texunit, s.tevorders_texmap & 7u);
      emitf(out, "    textemp = rawtextemp.%c%c%c%c;\n",
            kRgbaSwizzle[s.tex_swap[0]], kRgbaSwizzle[s.tex_swap[1]],
            kRgbaSwizzle[s.tex_swap[2]], kRgbaSwizzle[s.tex_swap[3]]);
    } else if (s.tevorders_enable != 0 && key.num_tex_gens > 0 && key.textured != 0) {
      // Dolphin PixelShaderGen divides every texcoord by its q (unless 0),
      // not only STQ texgens: a dual-transform post matrix can leave an ST
      // texgen's q != 1, and the hardware divides it all the same.
      const bool proj = texcoord < kMaxTexGens;
      if (proj)
        emitf(out,
              // Dolphin PixelShaderGen: q == 0 keeps st, otherwise st / q,
              // negative q included (a projector behind the surface).
              "    { let q = in.uv%u.z; let uv = select(in.uv%u.xy / q, "
              "in.uv%u.xy, q == 0.0); "
              "rawtextemp = vec4i(round(textureSample(tex%u, samp%u, uv) * "
              "255.0)); }\n",
              texcoord, texcoord, texcoord, texunit, texunit);
      else
        emitf(out,
              "    { let uv = in.uv%u.xy; rawtextemp = "
              "vec4i(round(textureSample(tex%u, samp%u, uv) * 255.0)); }\n",
              texcoord, texunit, texunit);
      emitf(out, "    textemp = rawtextemp.%c%c%c%c;\n",
            kRgbaSwizzle[s.tex_swap[0]], kRgbaSwizzle[s.tex_swap[1]],
            kRgbaSwizzle[s.tex_swap[2]], kRgbaSwizzle[s.tex_swap[3]]);
    } else if (key.num_tex_gens == 0) {
      emit(out, "    textemp = vec4i(0,0,0,0);\n");
    } else {
      emit(out, "    textemp = vec4i(255,255,255,255);\n");
    }

    // Konst, if referenced (KONST=14 color / 6 alpha).
    if (cc_uses(s, 14) || ac_uses(s, 6))
      emitf(out, "    konsttemp = vec4i(%s, %s);\n", kKselC[s.ksel_kc],
            kKselA[s.ksel_ka]);

    // tevin_{a,b,c} masked to 0..255; tevin_d keeps the 10-bit range.
    emitf(out, "    tevin_a = vec4i(%s, %s) & vec4i(255,255,255,255);\n",
          kTevCInput[s.cc_a], kTevAInput[s.ac_a]);
    emitf(out, "    tevin_b = vec4i(%s, %s) & vec4i(255,255,255,255);\n",
          kTevCInput[s.cc_b], kTevAInput[s.ac_b]);
    emitf(out, "    tevin_c = vec4i(%s, %s) & vec4i(255,255,255,255);\n",
          kTevCInput[s.cc_c], kTevAInput[s.ac_c]);
    emitf(out, "    tevin_d = vec4i(%s, %s);\n", kTevCInput[s.cc_d],
          kTevAInput[s.ac_d]);

    // Color combine.
    std::string crgb = s.cc_bias == 3
                           ? tev_compare(true, s.cc_op, s.cc_scale)
                           : tev_regular(true, "rgb", s.cc_bias, s.cc_op,
                                         s.cc_scale);
    const char* clo = s.cc_clamp ? "vec3i(0,0,0)" : "vec3i(-1024,-1024,-1024)";
    const char* chi =
        s.cc_clamp ? "vec3i(255,255,255)" : "vec3i(1023,1023,1023)";
    const char* cdest = kTevOutput[s.cc_dest];
    emitf(out, "    %s = vec4i(clamp(%s, %s, %s), %s.a);\n", cdest,
          crgb.c_str(), clo, chi, cdest);

    // Alpha combine.
    std::string aexp = s.ac_bias == 3
                           ? tev_compare(false, s.ac_op, s.ac_scale)
                           : tev_regular(false, "a", s.ac_bias, s.ac_op,
                                         s.ac_scale);
    const char* alo = s.ac_clamp ? "0" : "-1024";
    const char* ahi = s.ac_clamp ? "255" : "1023";
    const char* adest = kTevOutput[s.ac_dest];
    emitf(out, "    %s = vec4i(%s.rgb, clamp(%s, %s, %s));\n", adest, adest,
          aexp.c_str(), alo, ahi);
  }

  // The last stage's destination register is what reaches the framebuffer.
  const TevStageKey& last = key.tev_stages[key.num_tev_stages - 1];
  if (last.cc_dest != 0)
    emitf(out, "    prev = vec4i(%s.rgb, prev.a);\n", kTevOutput[last.cc_dest]);
  if (last.ac_dest != 0)
    emitf(out, "    prev = vec4i(prev.rgb, %s.a);\n", kTevOutput[last.ac_dest]);

  // Alpha test (Dolphin WriteAlphaTest): discard on failure.
  const std::string c0 = alpha_cond(key.alpha_comp0, "psc.alpha_ref.x");
  const std::string c1 = alpha_cond(key.alpha_comp1, "psc.alpha_ref.y");
  emitf(out, "    if (!( %s%s%s )) { discard; }\n", c0.c_str(),
        kAlphaLogic[key.alpha_logic], c1.c_str());

  // Fog runs after the alpha test (Dolphin order), modifying prev.rgb only.
  emit_fog(out, key);

  emit(out, "    return vec4f(prev) / 255.0;\n}\n");
}

// --- Integer lighting (Dolphin LightingShaderGen port) -----------------------
//
// GenerateLightShader for one light: emits the attenuation setup then the
// diffuse accumulation into `lacc` (int4). GLSL's `int3/float3` become WGSL
// `vec3i/vec3f`; GLSL `a ? b : c` becomes `select(c, b, a)`. WGSL forbids
// assigning to a multi-component swizzle (`lacc.rgb += ...`), so the rgb pass
// rebuilds the whole vector; the alpha pass writes the single `.a` component.
void emit_light(std::string& out, const LightChanKey& ch, int i, bool alpha) {
  const auto attn = static_cast<AttenuationFunc>(ch.attnfunc);
  const auto diff = static_cast<DiffuseFunc>(ch.diffusefunc);
  emitf(out, "        { // light %d\n", i);
  switch (attn) {
  case AttenuationFunc::None:
  case AttenuationFunc::Dir:
    emitf(out, "            ldir = normalize(vsc.lights[%d].pos.xyz - pos);\n", i);
    emit(out, "            attn = 1.0;\n"
              "            if (length(ldir) == 0.0) { ldir = _normal; }\n");
    break;
  case AttenuationFunc::Spec:
    emitf(out, "            ldir = normalize(vsc.lights[%d].pos.xyz - pos);\n", i);
    emitf(out,
          "            attn = select(0.0, max(0.0, dot(_normal, "
          "vsc.lights[%d].dir.xyz)), dot(_normal, ldir) >= 0.0);\n",
          i);
    emitf(out, "            cosAttn = vsc.lights[%d].cosatt.xyz;\n", i);
    if (diff == DiffuseFunc::None)
      emitf(out, "            distAttn = vsc.lights[%d].distatt.xyz;\n", i);
    else
      emitf(out, "            distAttn = normalize(vsc.lights[%d].distatt.xyz);\n", i);
    emit(out, "            attn = max(0.0, dot(cosAttn, vec3f(1.0, attn, "
              "attn*attn))) / dot(distAttn, vec3f(1.0, attn, attn*attn));\n");
    break;
  case AttenuationFunc::Spot:
    emitf(out, "            ldir = vsc.lights[%d].pos.xyz - pos;\n", i);
    emit(out, "            dist2 = dot(ldir, ldir);\n"
              "            dist = sqrt(dist2);\n"
              "            ldir = ldir / dist;\n");
    emitf(out, "            attn = max(0.0, dot(ldir, vsc.lights[%d].dir.xyz));\n", i);
    emitf(out,
          "            attn = max(0.0, vsc.lights[%d].cosatt.x + "
          "vsc.lights[%d].cosatt.y*attn + vsc.lights[%d].cosatt.z*attn*attn) / "
          "dot(vsc.lights[%d].distatt.xyz, vec3f(1.0, dist, dist2));\n",
          i, i, i, i);
    break;
  }
  // Diffuse term: attn * <diffuse> * color. Sign uses raw dot, Clamp max(0,dot),
  // None omits the dot (LightingShaderGen.cpp GenerateLightShader).
  const char* dterm = diff == DiffuseFunc::Sign  ? "(dot(ldir, _normal)) * "
                      : diff == DiffuseFunc::Clamp ? "max(0.0, dot(ldir, _normal)) * "
                                                   : "";
  if (!alpha) {
    emitf(out,
          "            lacc = lacc + vec4i(vec3i(round(attn * %svec3f("
          "vsc.lights[%d].color.rgb))), 0);\n",
          dterm, i);
  } else {
    emitf(out,
          "            lacc.a = lacc.a + i32(round(attn * %sf32("
          "vsc.lights[%d].color.a)));\n",
          dterm, i);
  }
  emit(out, "        }\n");
}

// GenerateLightingShaderHeader for one color channel j (litchan j = rgb,
// j+2 = alpha). Returns a WGSL function calc_lighting_chn{j}.
void emit_lighting_chan(std::string& out, const ShaderKey& key,
                        std::uint32_t j) {
  const LightChanKey& col = key.litchan[j];
  const LightChanKey& alp = key.litchan[j + 2];
  emitf(out,
        "fn calc_lighting_chn%u(base_color: vec4f, pos: vec3f, _normal: vec3f)"
        " -> vec4f {\n",
        j);
  emit(out, "    var lacc: vec4i;\n"
            "    var mat: vec4i;\n"
            "    var ldir: vec3f;\n    var cosAttn: vec3f;\n"
            "    var distAttn: vec3f;\n"
            "    var dist: f32;\n    var dist2: f32;\n    var attn: f32;\n");
  // mat rgb.
  if (static_cast<MatSource>(col.matsource) == MatSource::Vertex)
    emit(out, "    mat = vec4i(round(base_color * 255.0));\n");
  else
    emitf(out, "    mat = vsc.materials[%u];\n", j + 2);
  // lacc rgb (ambient / disabled-lighting seed).
  if (col.enablelighting != 0) {
    if (static_cast<MatSource>(col.ambsource) == MatSource::Vertex)
      emit(out, "    lacc = vec4i(round(base_color * 255.0));\n");
    else
      emitf(out, "    lacc = vsc.materials[%u];\n", j);
  } else {
    emit(out, "    lacc = vec4i(255, 255, 255, 255);\n");
  }
  // mat.w when the alpha material source differs from the color source.
  if (alp.matsource != col.matsource) {
    if (static_cast<MatSource>(alp.matsource) == MatSource::Vertex)
      emit(out, "    mat.w = i32(round(base_color.w * 255.0));\n");
    else
      emitf(out, "    mat.w = vsc.materials[%u].w;\n", j + 2);
  }
  // lacc.w (alpha ambient / disabled seed).
  if (alp.enablelighting != 0) {
    if (static_cast<MatSource>(alp.ambsource) == MatSource::Vertex)
      emit(out, "    lacc.w = i32(round(base_color.w * 255.0));\n");
    else
      emitf(out, "    lacc.w = vsc.materials[%u].w;\n", j);
  } else {
    emit(out, "    lacc.w = 255;\n");
  }
  // Color (rgb) lights.
  if (col.enablelighting != 0) {
    for (int i = 0; i < 8; ++i)
      if ((col.light_mask & (1u << i)) != 0u)
        emit_light(out, col, i, false);
  }
  // Alpha lights.
  if (alp.enablelighting != 0) {
    for (int i = 0; i < 8; ++i)
      if ((alp.light_mask & (1u << i)) != 0u)
        emit_light(out, alp, i, true);
  }
  emit(out, "    lacc = clamp(lacc, vec4i(0), vec4i(255));\n"
            "    return vec4f((mat * (lacc + (lacc >> vec4u(7)))) >> vec4u(8)) "
            "/ 255.0;\n}\n\n");
}

// True when color channel j (0/1) runs the full integer-lighting path rather
// than raw vertex-color passthrough. Dolphin runs dolphin_calculate_lighting_chn
// for every configured channel; the passthrough shortcut is only valid when the
// result would equal the vertex color, i.e. the channel is unlit AND both its
// color and alpha material sources are the vertex (mat=vertex, lacc=white =>
// output=vertex). A captured channel that enables lighting, or that selects the
// material register for either color or alpha, takes the full path. An
// unconfigured channel (bit clear in chan_captured_mask) stays passthrough even
// though MatSource::Register decodes to 0 — there is no material to apply.
} // namespace

bool channel_lit_path(const ShaderKey& k, unsigned j) {
  if ((k.chan_captured_mask & (1u << j)) == 0u)
    return false;
  const LightChanKey& col = k.litchan[j];
  const LightChanKey& alp = k.litchan[j + 2];
  return col.enablelighting != 0 || alp.enablelighting != 0 ||
         static_cast<MatSource>(col.matsource) == MatSource::Register ||
         static_cast<MatSource>(alp.matsource) == MatSource::Register;
}

// --- GPU vertex fetch ---------------------------------------------------------
// The vertex shader decodes the raw GX vertex stream (big-endian bytes in the
// shared vertex buffer) and the indexed arrays (in the shared storage buffer)
// as gxcore.cpp's CPU decoder does, from the walk in FetchKey.

void emit_fetch_buffers(std::string& out) {
  emit(out,
       "@group(0) @binding(0) var<storage, read> fetch_verts: array<u32>;\n"
       "@group(0) @binding(1) var<storage, read> fetch_store: array<u32>;\n"
       "fn vb8(b: u32) -> u32 { return (fetch_verts[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }\n"
       "fn vb16(b: u32) -> u32 { return (vb8(b) << 8u) | vb8(b + 1u); }\n"
       "fn vb24(b: u32) -> u32 { return (vb16(b) << 8u) | vb8(b + 2u); }\n"
       "fn vb32(b: u32) -> u32 { return (vb16(b) << 16u) | vb16(b + 2u); }\n"
       "fn sb8(b: u32) -> u32 { return (fetch_store[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }\n"
       "fn sb16(b: u32) -> u32 { return (sb8(b) << 8u) | sb8(b + 1u); }\n"
       "fn sb24(b: u32) -> u32 { return (sb16(b) << 8u) | sb8(b + 2u); }\n"
       "fn sb32(b: u32) -> u32 { return (sb16(b) << 16u) | sb16(b + 2u); }\n"
       "fn fetch_e4(x: u32) -> u32 { return x * 17u; }\n"
       "fn fetch_e5(x: u32) -> u32 { return (x << 3u) | (x >> 2u); }\n"
       "fn fetch_e6(x: u32) -> u32 { return (x << 2u) | (x >> 4u); }\n");
}

// One fixed-point or float component at byte address `addr` (a WGSL u32
// expression), read with the vb* (vertex) or sb* (array) helpers.
std::string fetch_component(const char* reader, const std::string& addr, std::uint32_t format,
                            std::uint32_t frac) {
  const std::string r = reader;
  std::string value;
  switch (format) {
  case 0u:
    value = "f32(" + r + "8(" + addr + "))";
    break;
  case 1u:
    value = "f32((i32(" + r + "8(" + addr + ")) << 24u) >> 24u)";
    break;
  case 2u:
    value = "f32(" + r + "16(" + addr + "))";
    break;
  case 3u:
    value = "f32((i32(" + r + "16(" + addr + ")) << 16u) >> 16u)";
    break;
  case 4u:
    return "bitcast<f32>(" + r + "32(" + addr + "))";
  default:
    return "0.0";
  }
  if (frac == 0u)
    return value;
  // 2^-frac: a power of two prints and parses back exactly.
  char scale[64];
  std::snprintf(scale, sizeof scale, "%.9e", std::ldexp(1.0, -static_cast<int>(frac)));
  return value + " * " + scale;
}

std::uint32_t fetch_scalar_size(std::uint32_t format) {
  return format <= 1u ? 1u : format <= 3u ? 2u : 4u;
}

// GX color at `a` as a vec4f in 0..1 (gxcore.cpp decode_color).
std::string fetch_color(const char* reader, const std::string& a, std::uint32_t format) {
  const std::string r = reader;
  auto byte = [&](int n) { return r + "8(" + a + " + " + std::to_string(n) + "u)"; };
  std::string rgba;
  switch (format) {
  case 0u: { // RGB565
    const std::string c = r + "16(" + a + ")";
    rgba = "fetch_e5((" + c + " >> 11u) & 31u), fetch_e6((" + c + " >> 5u) & 63u), fetch_e5(" + c +
           " & 31u), 255u";
    break;
  }
  case 1u: // RGB888
  case 2u: // RGB888x
    rgba = byte(0) + ", " + byte(1) + ", " + byte(2) + ", 255u";
    break;
  case 3u: { // RGBA4444
    const std::string c = r + "16(" + a + ")";
    rgba = "fetch_e4((" + c + " >> 12u) & 15u), fetch_e4((" + c + " >> 8u) & 15u), fetch_e4((" + c +
           " >> 4u) & 15u), fetch_e4(" + c + " & 15u)";
    break;
  }
  case 4u: { // RGBA6666
    const std::string c = r + "24(" + a + ")";
    rgba = "fetch_e6((" + c + " >> 18u) & 63u), fetch_e6((" + c + " >> 12u) & 63u), fetch_e6((" + c +
           " >> 6u) & 63u), fetch_e6(" + c + " & 63u)";
    break;
  }
  case 5u: // RGBA8888
    rgba = byte(0) + ", " + byte(1) + ", " + byte(2) + ", " + byte(3);
    break;
  default:
    return "vec4f(1.0)";
  }
  return "vec4f(vec4u(" + rgba + ")) / 255.0";
}

void emit_fetch_vertex(std::string& out, const FetchKey& fetch, bool in_normal, bool in_texmtxidx,
                       bool in_binormal, bool in_tangent) {
  emit(out, "fn fetch_vertex(vid: u32) -> VertexIn {\n"
            "    var v: VertexIn;\n"
            "    v.rawcolor0 = vec4f(1.0);\n"
            "    v.rawcolor1 = vec4f(1.0);\n"
            "    v.posmtx = vsc.fetch_base.y;\n");
  emitf(out, "    let base = vsc.fetch_base.x + vid * %uu;\n", fetch.vertex_size);
  for (std::uint32_t e = 0; e < fetch.entry_count && e < kMaxFetchEntries; ++e) {
    const FetchEntryKey& f = fetch.entries[e];
    const std::string a = "a" + std::to_string(e);
    const char* reader = "vb";
    // Kinds 0/1 are the matrix-index bytes, always in the vertex.
    if (f.vcd_type >= 2u && f.kind != 0u && f.kind != 1u) {
      emitf(out, "    let i%u = vb%u(base + %uu);\n", e, f.vcd_type == 2u ? 8u : 16u, f.offset);
      emitf(out, "    let %s = vsc.fetch_arrays[%u].x + i%u * vsc.fetch_arrays[%u].y;\n", a.c_str(), f.attr,
            e, f.attr);
      reader = "sb";
    } else {
      emitf(out, "    let %s = base + %uu;\n", a.c_str(), f.offset);
    }
    const std::uint32_t scalar = fetch_scalar_size(f.format);
    auto comp = [&](std::uint32_t n) {
      return fetch_component(reader, a + " + " + std::to_string(n * scalar) + "u", f.format, f.frac);
    };
    auto vec3 = [&](std::uint32_t first) {
      return "vec3f(" + comp(first) + ", " + comp(first + 1) + ", " + comp(first + 2) + ")";
    };
    switch (f.kind) {
    case 0u: // position-matrix index: the matrix-memory row
      emitf(out, "    v.posmtx = vb8(%s);\n", a.c_str());
      break;
    case 1u: // tex-matrix index, packed one byte per texgen 0-3
      if (in_texmtxidx && f.out_slot < 4u)
        emitf(out, "    v.texmtxidx = v.texmtxidx | (vb8(%s) << %uu);\n", a.c_str(), 8u * f.out_slot);
      break;
    case 2u: // position
      out += "    v.rawpos = vec3f(" + comp(0) + ", " + comp(1) + ", " +
             (f.count >= 3u ? comp(2) : std::string("0.0")) + ");\n";
      break;
    case 3u: // normal, or normal/binormal/tangent
      if (f.count >= 9u) {
        if (in_normal)
          out += "    v.rawnormal = " + vec3(0) + ";\n";
        if (in_binormal)
          out += "    v.rawbinormal = " + vec3(3) + ";\n";
        if (in_tangent)
          out += "    v.rawtangent = " + vec3(6) + ";\n";
      } else if (f.nbt_part == 0u && in_normal) {
        out += "    v.rawnormal = " + vec3(0) + ";\n";
      } else if (f.nbt_part == 1u && in_binormal) {
        out += "    v.rawbinormal = " + vec3(0) + ";\n";
      } else if (f.nbt_part == 2u && in_tangent) {
        out += "    v.rawtangent = " + vec3(0) + ";\n";
      }
      break;
    case 4u: // color 0/1
      out += "    v.rawcolor" + std::to_string(f.out_slot & 1u) + " = " + fetch_color(reader, a, f.format) + ";\n";
      break;
    case 5u: // texcoord 0-3 (the vertex layout carries four)
      if (f.out_slot < kMaxVertexTexCoords)
        out += "    v.rawtex" + std::to_string(f.out_slot) + " = vec2f(" + comp(0) + ", " +
               (f.count >= 2u ? comp(1) : std::string("0.0")) + ");\n";
      break;
    default:
      break;
    }
  }
  emit(out, "    return v;\n}\n");
}

std::string generate_wgsl(const ShaderKey& key) {

  std::string out;
  out.reserve(4096);
  const bool tev = key.tev_valid != 0;
  const bool lit = key.lit_valid != 0;
  // Color texgens (Dolphin VertexShaderGen texgentype switch, TexGenType::Color0/
  // Color1) drive the texcoord from a lit color channel. A Color1 texgen consumes
  // o.color1 even on a non-TEV draw, and an emboss texgen needs the light dir in
  // tangent space, so detect demand up front to widen the varyings/uniform/inputs.
  bool needs_color1 = false;
  bool has_emboss = false;
  for (std::uint32_t i = 0; i < key.num_tex_gens; ++i) {
    const auto t = static_cast<TexGenType>(key.tex_gens[i].texgentype);
    if (t == TexGenType::Color1)
      needs_color1 = true;
    else if (t == TexGenType::EmbossMap)
      has_emboss = true;
  }
  // Emboss transforms the light dir into tangent space, so the lights uniform +
  // the NBT binormal/tangent vertex attrs must be present even on an unlit draw.
  const bool emit_lights = lit || has_emboss;
  const bool emit_nbt = has_emboss; // binormal/tangent vertex inputs required
  // color1 occupies @location(1) whenever it is emitted; texcoords shift up one.
  const bool emit_color1 = tev || needs_color1;
  const std::uint32_t uv_loc_base = emit_color1 ? 2u : 1u;
  // Per-channel: run the full integer-lighting path (vs vertex-color
  // passthrough). Channel 1 only matters when its varying is emitted.
  const bool chan0_lit = channel_lit_path(key, 0u);
  const bool chan1_lit = emit_color1 && channel_lit_path(key, 1u);
  // The lit path reads vsc.materials (register material/ambient) and, when a
  // light is enabled, vsc.lights. Whenever either the light path or a
  // register-material channel is active we must expose the full uniform view
  // (both fields, in the fixed C struct order) so offsets line up.
  const bool needs_uniform_full = emit_lights || chan0_lit || chan1_lit;
  // Vertex-format N/B/T presence. A lit/emboss draw whose format omits an
  // attribute substitutes the cross-draw cached value from the uniform
  // (Dolphin I_CACHED_NORMAL fallback, VertexShaderGen.cpp:607-632) instead of
  // reading a per-vertex input that the decoder left zeroed (normalize(0)=NaN).
  const bool has_normal = key.has_vertex_normal != 0;
  const bool has_binormal = key.has_vertex_binormal != 0;
  const bool has_tangent = key.has_vertex_tangent != 0;
  const char* normal_in = has_normal ? "in.rawnormal" : "vsc.cached_normal.xyz";
  const char* tangent_in =
      has_tangent ? "in.rawtangent" : "vsc.cached_tangent.xyz";
  const char* binormal_in =
      has_binormal ? "in.rawbinormal" : "vsc.cached_binormal.xyz";
  const bool uses_cached_normal = (chan0_lit || chan1_lit) && lit && !has_normal;
  const bool uses_cached_tangent = has_emboss && !has_tangent;
  const bool uses_cached_binormal = has_emboss && !has_binormal;
  // Declare all three cached fields together when any is referenced so the WGSL
  // struct offsets stay aligned with the C VertexShaderConstants tail (they are
  // laid out normal, tangent, binormal — skipping one mid-run would misplace a
  // later one). Only emitted when actually used, so formats that carry the
  // attribute keep byte-identical goldens.
  const bool uses_cached =
      uses_cached_normal || uses_cached_tangent || uses_cached_binormal;
  // GPU vertex fetch: the shader reads the raw vertex bytes itself, and its
  // per-draw offsets sit at the end of the uniform, so the whole uniform view
  // is declared ahead of them.
  const bool fetch = key.fetch.enabled != 0;

  emit(out, "// gxcore generated shader (Dolphin VertexShaderGen shape)\n");
  if (needs_uniform_full || fetch)
    emit(out, "struct Light {\n"
              "    color: vec4i,\n"
              "    cosatt: vec4f,\n"
              "    distatt: vec4f,\n"
              "    pos: vec4f,\n"
              "    dir: vec4f,\n"
              "};\n");
  emit(out, "struct VertexShaderConstants {\n"
            "    posnormalmatrix: array<vec4f, 6>,\n"
            "    projection: array<vec4f, 4>,\n"
            "    texmatrices: array<vec4f, 24>,\n"
            "    transformmatrices: array<vec4f, 64>,\n"
            "    posttexmatrices: array<vec4f, 24>,\n");
  // The lit/emboss/register-material path binds the full uniform buffer (lights
  // + materials); the plain path declares only the leading fields (a smaller
  // WGSL struct over the same buffer is valid), keeping every existing golden
  // byte-identical.
  if (needs_uniform_full || uses_cached || fetch)
    emit(out, "    normalmatrices: array<vec4f, 32>,\n");
  if (needs_uniform_full || fetch)
    emit(out, "    lights: array<Light, 8>,\n"
              "    materials: array<vec4i, 4>,\n");
  if (uses_cached || fetch)
    emit(out, "    cached_normal: vec4f,\n"
              "    cached_tangent: vec4f,\n"
              "    cached_binormal: vec4f,\n");
  if (fetch)
    emit(out, "    fetch_base: vec4u,\n"
              "    fetch_arrays: array<vec4u, 16>,\n");
  emit(out, "};\n"
            "@group(1) @binding(0) var<uniform> vsc: VertexShaderConstants;\n");
  if (fetch)
    emit_fetch_buffers(out);
  if (tev) {
    // PixelShaderConstants (Dolphin I_COLORS/I_KCOLORS/I_ALPHA). Integer TEV.
    // group(2) here (texture shifts to group(3)) so an untextured TEV draw
    // leaves no gap in the pipeline's bind-group indices.
    emit(out, "struct PixelShaderConstants {\n"
              "    colors: array<vec4i, 4>,\n"
              "    kcolors: array<vec4i, 4>,\n"
              "    alpha_ref: vec4i,\n"
              "    fogcolor: vec4i,\n"
              "    fogi: vec4i,\n"
              "    fogf: vec4f,\n"
              "    fogrange: array<vec4f, 3>,\n");
    // Indirect constants (I_TEXDIMS/I_INDTEXSCALE/I_INDTEXMTX) sit at the end
    // of the C struct; declared only on the indirect path so every other TEV
    // shader stays byte-identical.
    if (tev_indirect_path(key))
      emit(out, "    texdims: array<vec4i, 8>,\n"
                "    indtexscale: array<vec4i, 2>,\n"
                "    indtexmtx: array<vec4i, 6>,\n");
    emit(out, "};\n"
              "@group(2) @binding(0) var<uniform> psc: PixelShaderConstants;\n");
  }
  if (key.textured != 0) {
    const std::uint32_t tex_group = tev ? 3u : 2u;
    const std::uint32_t used = used_texmap_mask(key);
    if (texmap_popcount(used) <= 1u) {
      emitf(out,
            "@group(%u) @binding(0) var tex0: texture_2d<f32>;\n"
            "@group(%u) @binding(1) var samp0: sampler;\n",
            tex_group, tex_group);
    } else {
      // Multi-texmap: texmap t -> binding 2t (texture) / 2t+1 (sampler), matching
      // the substrate bind group built from the same used-texmap set (63/Mfin).
      for (std::uint32_t t = 0; t < 8u; ++t) {
        if ((used & (1u << t)) == 0u)
          continue;
        emitf(out,
              "@group(%u) @binding(%u) var tex%u: texture_2d<f32>;\n"
              "@group(%u) @binding(%u) var samp%u: sampler;\n",
              tex_group, 2u * t, t, tex_group, 2u * t + 1u, t);
      }
    }
  }

  // Fixed vertex input layout (kVertexStrideBytes); unused inputs are ignored.
  // Under GPU vertex fetch the same struct is filled by fetch_vertex instead
  // (no vertex buffer, so no locations).
  const bool in_normal = (lit || emit_nbt) && has_normal;
  const bool in_texmtxidx = key.has_tex_mtx_idx != 0;
  const bool in_binormal = emit_nbt && has_binormal;
  const bool in_tangent = emit_nbt && has_tangent;
  auto loc = [&](unsigned n) -> std::string {
    return fetch ? std::string() : "@location(" + std::to_string(n) + ") ";
  };
  out += "struct VertexIn {\n";
  out += "    " + loc(0) + "rawpos: vec3f,\n";
  out += "    " + loc(1) + "posmtx: u32,\n";
  out += "    " + loc(2) + "rawcolor0: vec4f,\n";
  out += "    " + loc(3) + "rawcolor1: vec4f,\n";
  for (unsigned t = 0; t < 4; ++t)
    out += "    " + loc(4 + t) + "rawtex" + std::to_string(t) + ": vec2f,\n";
  if (in_normal)
    out += "    " + loc(8) + "rawnormal: vec3f,\n";
  if (in_texmtxidx)
    out += "    " + loc(9) + "texmtxidx: u32,\n";
  if (in_binormal)
    out += "    " + loc(10) + "rawbinormal: vec3f,\n";
  if (in_tangent)
    out += "    " + loc(11) + "rawtangent: vec3f,\n";
  emit(out, "};\n");
  if (fetch)
    emit_fetch_vertex(out, key.fetch, in_normal, in_texmtxidx, in_binormal, in_tangent);

  // @invariant: GX multi-pass draws (base, light, shadow passes over the same
  // geometry) depth-test against each other with EQUAL/LEQUAL, so every
  // pipeline must compute bit-identical positions or the passes z-fight.
  emit(out, "struct VertexOut {\n"
            "    @invariant @builtin(position) pos: vec4f,\n"
            "    @location(0) color0: vec4f,\n");
  if (emit_color1)
    emit(out, "    @location(1) color1: vec4f,\n");
  for (std::uint32_t i = 0; i < key.num_tex_gens; ++i)
    emitf(out, "    @location(%u) uv%u: vec3f,\n", uv_loc_base + i, i);
  emit(out, "};\n\n");

  // Integer lighting header (Dolphin GenerateLightingShaderHeader): one function
  // per color channel that takes the full path (lit or register-material).
  if (chan0_lit)
    emit_lighting_chan(out, key, 0u);
  if (chan1_lit)
    emit_lighting_chan(out, key, 1u);

  if (fetch)
    emit(out, "@vertex\nfn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {\n"
              "    let in = fetch_vertex(vid);\n"
              "    var o: VertexOut;\n");
  else
    emit(out, "@vertex\nfn vs_main(in: VertexIn) -> VertexOut {\n"
              "    var o: VertexOut;\n");
  if (key.has_pos_mtx_idx != 0) {
    emit(out, "    let posidx = i32(in.posmtx);\n"
              "    let p0 = vsc.transformmatrices[posidx];\n"
              "    let p1 = vsc.transformmatrices[posidx + 1];\n"
              "    let p2 = vsc.transformmatrices[posidx + 2];\n");
  } else {
    emit(out, "    let p0 = vsc.posnormalmatrix[0];\n"
              "    let p1 = vsc.posnormalmatrix[1];\n"
              "    let p2 = vsc.posnormalmatrix[2];\n");
  }
  // Normal-matrix rows for normals, tangents and binormals (Dolphin
  // VertexShaderGen dolphin_normal_matrix): per vertex from (posidx & 31) when
  // the format carries a matrix index, else the current matrix's.
  if (needs_uniform_full || uses_cached) {
    if (key.has_pos_mtx_idx != 0)
      emit(out, "    let nidx = posidx & 31;\n"
                "    let n0 = vsc.normalmatrices[nidx].xyz;\n"
                "    let n1 = vsc.normalmatrices[(nidx + 1) & 31].xyz;\n"
                "    let n2 = vsc.normalmatrices[(nidx + 2) & 31].xyz;\n");
    else
      emit(out, "    let n0 = vsc.posnormalmatrix[3].xyz;\n"
                "    let n1 = vsc.posnormalmatrix[4].xyz;\n"
                "    let n2 = vsc.posnormalmatrix[5].xyz;\n");
  }
  emit(out, "    let pos4 = vec4f(in.rawpos, 1.0);\n"
            "    let viewpos = vec4f(dot(p0, pos4), dot(p1, pos4), "
            "dot(p2, pos4), 1.0);\n"
            "    var clip = vec4f(dot(vsc.projection[0], viewpos), "
            "dot(vsc.projection[1], viewpos), dot(vsc.projection[2], "
            "viewpos), dot(vsc.projection[3], viewpos));\n"
            // GC clip z lands in [-w, 0]; the substrate runs reversed-Z
            // (graphics/aurora lib/gx/gx.hpp UseReversedZ), so flip.
            "    clip.z = -clip.z;\n"
            "    o.pos = clip;\n");
  // Vertex colors (Dolphin VertexShaderGen WriteVertexBody). First select each
  // channel's vertex color with the GX presence rules: channel 0 prefers
  // rawcolor0, falls back to rawcolor1 when only color1 is present; channel 1
  // uses rawcolor1 only when BOTH colors are present; the missing value is white
  // (Dolphin's default missing_color_value). Then every channel runs
  // dolphin_calculate_lighting_chn when it takes the full path (which internally
  // applies the register material/ambient and any enabled lights), else the
  // vertex color passes straight through. numColorChans zeroes unused channels.
  const bool use_color1 = key.has_color0 != 0 && key.has_color1 != 0;
  const char* vc0 = key.has_color0 != 0   ? "in.rawcolor0"
                    : key.has_color1 != 0 ? "in.rawcolor1"
                                          : "vec4f(1.0)";
  const char* vc1 = use_color1 ? "in.rawcolor1" : "vec4f(1.0)";
  if (chan0_lit || chan1_lit) {
    // View-space normal (posnormalmatrix rows 3-5). When no channel enables a
    // light the input carries no normal (register-material-only path), so seed a
    // constant — dolphin_calculate_lighting_chn only reads it inside a light and
    // there are none, leaving the material term intact.
    if (lit)
      emitf(out, "    let _normal = normalize(vec3f("
                 "dot(n0, %s), "
                 "dot(n1, %s), "
                 "dot(n2, %s)));\n",
            normal_in, normal_in, normal_in);
    else
      emit(out, "    let _normal = vec3f(0.0, 0.0, 1.0);\n");
  }
  if (chan0_lit)
    emitf(out, "    o.color0 = calc_lighting_chn0(%s, viewpos.xyz, _normal);\n",
          vc0);
  else
    emitf(out, "    o.color0 = %s;\n", vc0);
  if (emit_color1) {
    if (chan1_lit)
      emitf(out,
            "    o.color1 = calc_lighting_chn1(%s, viewpos.xyz, _normal);\n",
            vc1);
    else
      emitf(out, "    o.color1 = %s;\n", vc1);
  }
  // numColorChans gates channel output (Dolphin WriteVertexBody), regardless of
  // whether the channel was lit or passthrough.
  if (key.num_color_chans == 0u)
    emit(out, "    o.color0 = vec4f(0.0);\n");
  if (emit_color1 && key.num_color_chans <= 1u)
    emit(out, "    o.color1 = vec4f(0.0);\n");

  for (std::uint32_t i = 0; i < key.num_tex_gens; ++i) {
    const TexGenKey& tg = key.tex_gens[i];
    emit(out, "    {\n");
    emit(out, "        var coord = vec4f(0.0, 0.0, 1.0, 1.0);\n");
    const auto row = static_cast<TexSourceRow>(tg.sourcerow);
    if (row == TexSourceRow::Geom) {
      emit(out, "        coord = vec4f(in.rawpos, 1.0);\n");
    } else if (tg.sourcerow >= static_cast<std::uint8_t>(TexSourceRow::Tex0) &&
               tg.sourcerow <
                   static_cast<std::uint8_t>(TexSourceRow::Tex0) + 4u) {
      const std::uint32_t texnum =
          tg.sourcerow - static_cast<std::uint8_t>(TexSourceRow::Tex0);
      if ((key.uv_mask & (1u << texnum)) != 0u) {
        emitf(out,
              "        coord = vec4f(in.rawtex%u.x, in.rawtex%u.y, 1.0, "
              "1.0);\n",
              texnum, texnum);
      }
    }
    // Other source rows (normal/colors/binormals, tex4-7) are outside the
    // slice; the plan builder counted them and coord stays the default.
    if (tg.inputform == 0u) // AB11
      emit(out, "        coord.z = 1.0;\n");
    const auto tgtype = static_cast<TexGenType>(tg.texgentype);
    if (tgtype == TexGenType::Regular) {
      // Matrix rows: per-vertex TEXMTXIDX selects a matrix-memory row base from
      // the vertex attribute (like PNMTXIDX for position, the byte is the row
      // index into transformmatrices); otherwise use the static per-texgen slot
      // resolved from XF MatrixIndexA/B into texmatrices.
      const bool per_vertex =
          key.has_tex_mtx_idx != 0 && (key.tex_mtx_idx_mask & (1u << i)) != 0u;
      if (per_vertex) {
        emitf(out, "        let ti%u = (in.texmtxidx >> (8u * %uu)) & 0xFFu;\n",
              i, i);
        emitf(out, "        let m%u0 = vsc.transformmatrices[ti%u];\n", i, i);
        emitf(out, "        let m%u1 = vsc.transformmatrices[ti%u + 1u];\n", i, i);
        emitf(out, "        let m%u2 = vsc.transformmatrices[ti%u + 2u];\n", i, i);
        if (tg.projection != 0u)
          emitf(out, "        var uv = vec3f(dot(coord, m%u0), dot(coord, m%u1), "
                     "dot(coord, m%u2));\n",
                i, i, i);
        else
          emitf(out, "        var uv = vec3f(dot(coord, m%u0), dot(coord, m%u1), "
                     "1.0);\n",
                i, i);
      } else if (tg.projection != 0u) { // STQ, 3 rows
        emitf(out,
              "        var uv = vec3f(dot(coord, vsc.texmatrices[%u]), "
              "dot(coord, vsc.texmatrices[%u]), dot(coord, "
              "vsc.texmatrices[%u]));\n",
              3u * i, 3u * i + 1u, 3u * i + 2u);
      } else { // ST, 2 rows
        emitf(out,
              "        var uv = vec3f(dot(coord, vsc.texmatrices[%u]), "
              "dot(coord, vsc.texmatrices[%u]), 1.0);\n",
              3u * i, 3u * i + 1u);
      }
      // Dual texture transform (Dolphin VertexShaderGen): optional normalize,
      // then the post matrix over (s, t, q, 1). HSD sets every texture's UV
      // transform this way.
      if ((tg.posttransform & 1u) != 0u) {
        if ((tg.posttransform & 2u) != 0u)
          emit(out, "        uv = normalize(uv);\n");
        emitf(out,
              "        uv = vec3f(dot(vec4f(uv, 1.0), vsc.posttexmatrices[%u]), "
              "dot(vec4f(uv, 1.0), vsc.posttexmatrices[%u]), "
              "dot(vec4f(uv, 1.0), vsc.posttexmatrices[%u]));\n",
              3u * i, 3u * i + 1u, 3u * i + 2u);
      }
      // GC q==0 special case (Dolphin VertexShaderGen.cpp WriteTexCoordTransforms).
      emit(out, "        if (uv.z == 0.0) { uv = vec3f(clamp(uv.xy * 0.5, "
                "vec2f(-1.0), vec2f(1.0)), uv.z); }\n");
      emitf(out, "        o.uv%u = uv;\n", i);
    } else if (tgtype == TexGenType::Color0) {
      // Dolphin VertexShaderGen: texgen from a lit color channel (toon/ramp),
      // no matrix. rg of the channel become s,t; q = 1.
      emitf(out, "        o.uv%u = vec3f(o.color0.x, o.color0.y, 1.0);\n", i);
    } else if (tgtype == TexGenType::Color1) {
      emitf(out, "        o.uv%u = vec3f(o.color1.x, o.color1.y, 1.0);\n", i);
    } else { // EmbossMap
      // Dolphin VertexShaderGen: add the light dir (view space) projected onto
      // the view-space tangent/binormal to the emboss-source texgen's coords.
      // Tangent/binormal go to view space via the normal matrix (rows 3-5).
      emitf(out, "        let tn%u = vec3f("
                 "dot(n0, %s), "
                 "dot(n1, %s), "
                 "dot(n2, %s));\n",
            i, tangent_in, tangent_in, tangent_in);
      emitf(out, "        let bn%u = vec3f("
                 "dot(n0, %s), "
                 "dot(n1, %s), "
                 "dot(n2, %s));\n",
            i, binormal_in, binormal_in, binormal_in);
      emitf(out, "        let ld%u = normalize(vsc.lights[%uu].pos.xyz - "
                 "viewpos.xyz);\n",
            i, static_cast<unsigned>(tg.embosslightshift));
      emitf(out, "        o.uv%u = o.uv%u + vec3f(dot(ld%u, tn%u), "
                 "dot(ld%u, bn%u), 0.0);\n",
            i, static_cast<unsigned>(tg.embosssourceshift), i, i, i, i);
    }
    emit(out, "    }\n");
  }
  emit(out, "    return o;\n}\n\n");

  // Fragment. TEV path = ported integer combiner; else the S12/S13 passthrough
  // (kept byte-identical for keys whose combiner regs were never captured).
  if (tev) {
    emit_tev_fragment(out, key);
    return out;
  }
  emit(out, "@fragment\nfn fs_main(in: VertexOut) -> @location(0) vec4f {\n");
  emit(out, "    var prev = in.color0;\n");
  if (key.textured != 0) {
    if (key.num_tex_gens > 0) {  // every texcoord divides by q (see above)
      emit(out, "    let uv = select(in.uv0.xy / in.uv0.z, in.uv0.xy, in.uv0.z == 0.0);\n");
    } else {
      emit(out, "    let uv = in.uv0.xy;\n");
    }
    emit(out, "    prev = textureSample(tex0, samp0, uv);\n");
  }
  emit(out, "    return prev;\n}\n");
  return out;
}

} // namespace gxruntime::gxcore
