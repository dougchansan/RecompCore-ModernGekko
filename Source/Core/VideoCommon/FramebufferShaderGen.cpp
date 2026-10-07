// Copyright 2019 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/FramebufferShaderGen.h"

#include <array>
#include <cstdlib>
#include <string_view>

#include "Common/Logging/Log.h"

#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/ColosseumTextUpscale.h"
#include "VideoCommon/ShaderGenCommon.h"
#include "VideoCommon/TextureDecoder.h"
#include "VideoCommon/VertexShaderGen.h"
#include "VideoCommon/VideoCommon.h"
#include "VideoCommon/VideoConfig.h"

namespace FramebufferShaderGen
{
namespace
{
APIType GetAPIType()
{
  return g_backend_info.api_type;
}

void EmitUniformBufferDeclaration(ShaderCode& code)
{
  code.Write("UBO_BINDING(std140, 1) uniform PSBlock\n");
}

void EmitSamplerDeclarations(ShaderCode& code, u32 start = 0, u32 end = 1,
                             bool multisampled = false)
{
  switch (GetAPIType())
  {
  case APIType::D3D:
  case APIType::Metal:
  case APIType::OpenGL:
  case APIType::Vulkan:
  {
    const char* array_type = multisampled ? "sampler2DMSArray" : "sampler2DArray";

    for (u32 i = start; i < end; i++)
    {
      code.Write("SAMPLER_BINDING({}) uniform {} samp{};\n", i, array_type, i);
    }
  }
  break;
  default:
    break;
  }
}

void EmitSampleTexture(ShaderCode& code, u32 n, std::string_view coords)
{
  switch (GetAPIType())
  {
  case APIType::D3D:
  case APIType::Metal:
  case APIType::OpenGL:
  case APIType::Vulkan:
    code.Write("texture(samp{}, {})", n, coords);
    break;

  default:
    break;
  }
}

// Emits a texel fetch/load instruction. Assumes that "coords" is a 4-element vector, with z
// containing the layer, and w containing the mipmap level.
void EmitTextureLoad(ShaderCode& code, u32 n, std::string_view coords)
{
  switch (GetAPIType())
  {
  case APIType::D3D:
  case APIType::Metal:
  case APIType::OpenGL:
  case APIType::Vulkan:
    code.Write("texelFetch(samp{}, ({}).xyz, ({}).w)", n, coords, coords);
    break;

  default:
    break;
  }
}

void EmitVertexMainDeclaration(ShaderCode& code, u32 num_tex_inputs, u32 num_color_inputs,
                               bool position_input, u32 num_tex_outputs, u32 num_color_outputs,
                               std::string_view extra_inputs = {})
{
  switch (GetAPIType())
  {
  case APIType::D3D:
  case APIType::Metal:
  case APIType::OpenGL:
  case APIType::Vulkan:
  {
    for (u32 i = 0; i < num_tex_inputs; i++)
    {
      const auto attribute = ShaderAttrib::TexCoord0 + i;
      code.Write("ATTRIBUTE_LOCATION({:s}) in float3 rawtex{};\n", attribute, i);
    }
    for (u32 i = 0; i < num_color_inputs; i++)
    {
      const auto attribute = ShaderAttrib::Color0 + i;
      code.Write("ATTRIBUTE_LOCATION({:s}) in float4 rawcolor{};\n", attribute, i);
    }
    if (position_input)
      code.Write("ATTRIBUTE_LOCATION({:s}) in float4 rawpos;\n", ShaderAttrib::Position);

    if (g_backend_info.bSupportsGeometryShaders)
    {
      code.Write("VARYING_LOCATION(0) out VertexData {{\n");
      for (u32 i = 0; i < num_tex_outputs; i++)
        code.Write("  float3 v_tex{};\n", i);
      for (u32 i = 0; i < num_color_outputs; i++)
        code.Write("  float4 v_col{};\n", i);
      code.Write("}};\n");
    }
    else
    {
      for (u32 i = 0; i < num_tex_outputs; i++)
        code.Write("VARYING_LOCATION({}) out float3 v_tex{};\n", i, i);
      for (u32 i = 0; i < num_color_outputs; i++)
        code.Write("VARYING_LOCATION({}) out float4 v_col{};\n", num_tex_inputs + i, i);
    }
    code.Write("#define opos gl_Position\n");
    code.Write("{}\n", extra_inputs);
    code.Write("void main()\n");
  }
  break;
  default:
    break;
  }
}

void EmitPixelMainDeclaration(ShaderCode& code, u32 num_tex_inputs, u32 num_color_inputs,
                              std::string_view output_type = "float4",
                              std::string_view extra_vars = {}, bool emit_frag_coord = false)
{
  switch (GetAPIType())
  {
  case APIType::D3D:
  case APIType::Metal:
  case APIType::OpenGL:
  case APIType::Vulkan:
  {
    if (g_backend_info.bSupportsGeometryShaders)
    {
      code.Write("VARYING_LOCATION(0) in VertexData {{\n");
      for (u32 i = 0; i < num_tex_inputs; i++)
        code.Write("  float3 v_tex{};\n", i);
      for (u32 i = 0; i < num_color_inputs; i++)
        code.Write("  float4 v_col{};\n", i);
      code.Write("}};\n");
    }
    else
    {
      for (u32 i = 0; i < num_tex_inputs; i++)
        code.Write("VARYING_LOCATION({}) in float3 v_tex{};\n", i, i);
      for (u32 i = 0; i < num_color_inputs; i++)
        code.Write("VARYING_LOCATION({}) in float4 v_col{};\n", num_tex_inputs + i, i);
    }

    code.Write("FRAGMENT_OUTPUT_LOCATION(0) out {} ocol0;\n", output_type);
    code.Write("{}\n", extra_vars);
    if (emit_frag_coord)
      code.Write("#define frag_coord gl_FragCoord\n");
    code.Write("void main()\n");
  }
  break;

  default:
    break;
  }
}
}  // Anonymous namespace

std::string GenerateScreenQuadVertexShader()
{
  ShaderCode code;
  EmitVertexMainDeclaration(code, 0, 0, false, 1, 0,

                            "#define id gl_VertexID\n");
  code.Write(
      "{{\n"
      "  v_tex0 = float3(float((id << 1) & 2), float(id & 2), 0.0f);\n"
      "  opos = float4(v_tex0.xy * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);\n");

  // NDC space is flipped in Vulkan. We also flip in GL so that (0,0) is in the lower-left.
  if (GetAPIType() == APIType::Vulkan || GetAPIType() == APIType::OpenGL)
    code.Write("  opos.y = -opos.y;\n");

  code.Write("}}\n");

  return code.GetBuffer();
}

std::string GeneratePassthroughGeometryShader(u32 num_tex, u32 num_colors)
{
  // Layered EFB copies/clears: 2 layers for stereoscopy, up to 4 for frame
  // interpolation.
  const u32 layers = g_ActiveConfig.stereo_mode == StereoMode::FrameInterp ?
                         static_cast<u32>(g_ActiveConfig.iFrameInterpLayers) :
                         2;
  ShaderCode code;
  if (GetAPIType() == APIType::D3D)
  {
    code.Write("struct VS_OUTPUT\n"
               "{{\n");
    for (u32 i = 0; i < num_tex; i++)
      code.Write("  float3 tex{} : TEXCOORD{};\n", i, i);
    for (u32 i = 0; i < num_colors; i++)
      code.Write("  float4 color{} : TEXCOORD{};\n", i, i + num_tex);
    code.Write("  float4 position : SV_Position;\n"
               "}};\n");

    code.Write("struct GS_OUTPUT\n"
               "{{");
    for (u32 i = 0; i < num_tex; i++)
      code.Write("  float3 tex{} : TEXCOORD{};\n", i, i);
    for (u32 i = 0; i < num_colors; i++)
      code.Write("  float4 color{} : TEXCOORD{};\n", i, i + num_tex);
    code.Write("  float4 position : SV_Position;\n"
               "  uint slice : SV_RenderTargetArrayIndex;\n"
               "}};\n\n");

    code.Write("[maxvertexcount({})]\n"
               "void main(triangle VS_OUTPUT vso[3], inout TriangleStream<GS_OUTPUT> output)\n"
               "{{\n"
               "  for (uint slice = 0; slice < {}u; slice++)\n",
               3 * layers, layers);
    code.Write("  {{\n"
               "    for (int i = 0; i < 3; i++)\n"
               "    {{\n"
               "      GS_OUTPUT gso;\n"
               "      gso.position = vso[i].position;\n");
    for (u32 i = 0; i < num_tex; i++)
      code.Write("      gso.tex{} = float3(vso[i].tex{}.xy, float(slice));\n", i, i);
    for (u32 i = 0; i < num_colors; i++)
      code.Write("      gso.color{} = vso[i].color{};\n", i, i);
    code.Write("      gso.slice = slice;\n"
               "      output.Append(gso);\n"
               "    }}\n"
               "    output.RestartStrip();\n"
               "  }}\n"
               "}}\n");
  }
  else if (GetAPIType() == APIType::OpenGL || GetAPIType() == APIType::Vulkan)
  {
    code.Write("layout(triangles) in;\n"
               "layout(triangle_strip, max_vertices = {}) out;\n",
               3 * layers);

    if (num_tex > 0 || num_colors > 0)
    {
      code.Write("VARYING_LOCATION(0) in VertexData {{\n");
      for (u32 i = 0; i < num_tex; i++)
        code.Write("  float3 v_tex{};\n", i);
      for (u32 i = 0; i < num_colors; i++)
        code.Write("  float4 v_col{};\n", i);
      code.Write("}} v_in[];\n");

      code.Write("VARYING_LOCATION(0) out VertexData {{\n");
      for (u32 i = 0; i < num_tex; i++)
        code.Write("  float3 v_tex{};\n", i);
      for (u32 i = 0; i < num_colors; i++)
        code.Write("  float4 v_col{};\n", i);
      code.Write("}} v_out;\n");
    }
    code.Write("\n"
               "void main()\n"
               "{{\n"
               "  for (int j = 0; j < {}; j++)\n"
               "  {{\n"
               "    gl_Layer = j;\n",
               layers);

    // We have to explicitly unroll this loop otherwise the GL compiler gets cranky.
    for (u32 v = 0; v < 3; v++)
    {
      code.Write("    gl_Position = gl_in[{}].gl_Position;\n", v);
      for (u32 i = 0; i < num_tex; i++)
      {
        code.Write("    v_out.v_tex{} = float3(v_in[{}].v_tex{}.xy, float(j));\n", i, v, i);
      }
      for (u32 i = 0; i < num_colors; i++)
        code.Write("    v_out.v_col{} = v_in[{}].v_col{};\n", i, v, i);
      code.Write("    EmitVertex();\n\n");
    }
    code.Write("    EndPrimitive();\n"
               "  }}\n"
               "}}\n");
  }

  return code.GetBuffer();
}

std::string GenerateTextureCopyVertexShader()
{
  ShaderCode code;
  EmitUniformBufferDeclaration(code);
  code.Write("{{"
             "  float2 src_offset;\n"
             "  float2 src_size;\n"
             "}};\n\n");

  EmitVertexMainDeclaration(code, 0, 0, false, 1, 0,

                            "#define id gl_VertexID");
  code.Write("{{\n"
             "  v_tex0 = float3(float((id << 1) & 2), float(id & 2), 0.0f);\n"
             "  opos = float4(v_tex0.xy * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);\n"
             "  v_tex0 = float3(src_offset + (src_size * v_tex0.xy), 0.0f);\n");

  // NDC space is flipped in Vulkan. We also flip in GL so that (0,0) is in the lower-left.
  if (GetAPIType() == APIType::Vulkan || GetAPIType() == APIType::OpenGL)
    code.Write("  opos.y = -opos.y;\n");

  code.Write("}}\n");

  return code.GetBuffer();
}

std::string GenerateTextureCopyPixelShader()
{
  ShaderCode code;
  EmitSamplerDeclarations(code, 0, 1, false);
  EmitPixelMainDeclaration(code, 1, 0);
  code.Write("{{\n"
             "  ocol0 = ");
  EmitSampleTexture(code, 0, "v_tex0");
  code.Write(";\n"
             "}}\n");
  return code.GetBuffer();
}

std::string GenerateTextSharpenPixelShader()
{
  ShaderCode code;
  EmitSamplerDeclarations(code, 0, 1, false);
  EmitPixelMainDeclaration(code, 1, 0);
  code.Write("{{\n"
             // Reconstruct a rounded silhouette from a compact, center-weighted footprint in
             // native texel space. The small radius smooths curved shoulders while preserving
             // the one-pixel counters in lowercase e and other narrow Colosseum glyphs.
             "  float2 texel = float2(1.0f / 512.0f, 1.0f / 512.0f);\n"
             "  float2 radius = texel * {:.3f}f;\n"
             "  float center = ",
             ColosseumTextUpscale::RECONSTRUCTION_RADIUS);
  EmitSampleTexture(code, 0, "v_tex0");
  code.Write(".a;\n"
             "  float axial = ");
  EmitSampleTexture(code, 0, "float3(v_tex0.xy + float2(radius.x, 0.0f), v_tex0.z)");
  code.Write(".a + ");
  EmitSampleTexture(code, 0, "float3(v_tex0.xy - float2(radius.x, 0.0f), v_tex0.z)");
  code.Write(".a + ");
  EmitSampleTexture(code, 0, "float3(v_tex0.xy + float2(0.0f, radius.y), v_tex0.z)");
  code.Write(".a + ");
  EmitSampleTexture(code, 0, "float3(v_tex0.xy - float2(0.0f, radius.y), v_tex0.z)");
  code.Write(".a;\n"
             "  float diagonal = ");
  EmitSampleTexture(code, 0, "float3(v_tex0.xy + radius, v_tex0.z)");
  code.Write(".a + ");
  EmitSampleTexture(code, 0, "float3(v_tex0.xy - radius, v_tex0.z)");
  code.Write(".a + ");
  EmitSampleTexture(code, 0, "float3(v_tex0.xy + float2(radius.x, -radius.y), v_tex0.z)");
  code.Write(".a + ");
  EmitSampleTexture(code, 0, "float3(v_tex0.xy + float2(-radius.x, radius.y), v_tex0.z)");
  code.Write(".a;\n"
             "  float reconstructed = center * {:.3f}f + axial * {:.3f}f + "
             "diagonal * {:.3f}f;\n"
             "  float coverage = smoothstep({:.2f}f, {:.2f}f, reconstructed);\n"
             "  ocol0 = float4(coverage, coverage, coverage, coverage);\n"
             "}}\n",
             ColosseumTextUpscale::CENTER_WEIGHT, ColosseumTextUpscale::AXIAL_WEIGHT,
             ColosseumTextUpscale::DIAGONAL_WEIGHT, ColosseumTextUpscale::COVERAGE_LOW,
             ColosseumTextUpscale::COVERAGE_HIGH);
  return code.GetBuffer();
}

std::string GenerateColorTextSharpenPixelShader()
{
  ShaderCode code;
  EmitSamplerDeclarations(code, 0, 1, false);
  EmitPixelMainDeclaration(code, 1, 0);
  code.Write("{{\n"
             "  float4 source = ");
  EmitSampleTexture(code, 0, "v_tex0");
  code.Write(";\n"
             // The battle-status atlas stores colored labels and icons on transparent black.
             // Linear enlargement blends their RGB toward black, so recover straight color
             // before applying the broad antialiased coverage transition.
             "  float3 straight_color = source.a > 0.001f ? "
             "clamp(source.rgb / source.a, 0.0f, 1.0f) : float3(0.0f, 0.0f, 0.0f);\n"
             "  float coverage = smoothstep({:.2f}f, {:.2f}f, source.a);\n"
             "  ocol0 = float4(straight_color, coverage);\n"
             "}}\n",
             ColosseumTextUpscale::COLOR_COVERAGE_LOW,
             ColosseumTextUpscale::COLOR_COVERAGE_HIGH);
  return code.GetBuffer();
}

std::string GenerateColorPixelShader()
{
  ShaderCode code;
  EmitPixelMainDeclaration(code, 0, 1);
  code.Write("{{\n"
             "  ocol0 = v_col0;\n"
             "}}\n");
  return code.GetBuffer();
}

std::string GenerateResolveColorPixelShader(u32 samples)
{
  ShaderCode code;
  EmitSamplerDeclarations(code, 0, 1, true);
  EmitPixelMainDeclaration(code, 1, 0);
  code.Write("{{\n"
             "  int layer = int(v_tex0.z);\n"
             "  int3 coords = int3(int2(gl_FragCoord.xy), layer);\n"
             "  ocol0 = float4(0.0f);\n");
  code.Write("  for (int i = 0; i < {}; i++)\n", samples);
  code.Write("    ocol0 += texelFetch(samp0, coords, i);\n");
  code.Write("  ocol0 /= {}.0f;\n", samples);
  code.Write("}}\n");
  return code.GetBuffer();
}

std::string GenerateResolveDepthPixelShader(u32 samples)
{
  ShaderCode code;
  EmitSamplerDeclarations(code, 0, 1, true);
  EmitPixelMainDeclaration(code, 1, 0, "float", "");
  code.Write("{{\n"
             "  int layer = int(v_tex0.z);\n");
  code.Write("  int3 coords = int3(int2(gl_FragCoord.xy), layer);\n");

  // Take the minimum of all depth samples.
  code.Write("  ocol0 = texelFetch(samp0, coords, 0).r;\n");
  code.Write("  for (int i = 1; i < {}; i++)\n", samples);
  code.Write("    ocol0 = min(ocol0, texelFetch(samp0, coords, i).r);\n");

  code.Write("}}\n");
  return code.GetBuffer();
}

std::string GenerateClearVertexShader()
{
  ShaderCode code;
  EmitUniformBufferDeclaration(code);
  code.Write("{{\n"
             "  float4 clear_color;\n"
             "  float clear_depth;\n"
             "}};\n");

  EmitVertexMainDeclaration(code, 0, 0, false, 0, 1,

                            "#define id gl_VertexID\n");
  code.Write(
      "{{\n"
      "  float2 coord = float2(float((id << 1) & 2), float(id & 2));\n"
      "  opos = float4(coord * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), clear_depth, 1.0f);\n"
      "  v_col0 = clear_color;\n");

  // NDC space is flipped in Vulkan
  if (GetAPIType() == APIType::Vulkan)
    code.Write("  opos.y = -opos.y;\n");

  code.Write("}}\n");

  return code.GetBuffer();
}

std::string GenerateEFBPokeVertexShader()
{
  ShaderCode code;
  EmitVertexMainDeclaration(code, 0, 1, true, 0, 1);
  code.Write("{{\n"
             "  v_col0 = rawcolor0;\n"
             "  opos = float4(rawpos.xyz, 1.0f);\n");
  if (g_backend_info.bSupportsLargePoints)
    code.Write("  gl_PointSize = rawpos.w;\n");

  // NDC space is flipped in Vulkan.
  if (GetAPIType() == APIType::Vulkan)
    code.Write("  opos.y = -opos.y;\n");

  code.Write("}}\n");
  return code.GetBuffer();
}

std::string GenerateFormatConversionShader(EFBReinterpretType convtype, u32 samples)
{
  ShaderCode code;
  EmitSamplerDeclarations(code, 0, 1, samples > 1);
  EmitPixelMainDeclaration(code, 1, 0, "float4",

                           "");
  code.Write("{{\n"
             "  int layer = int(v_tex0.z);\n");
  code.Write("  int3 coords = int3(int2(gl_FragCoord.xy), layer);\n");

  if (samples == 1)
  {
    // No MSAA at all.
    code.Write("  float4 val = texelFetch(samp0, coords, 0);\n");
  }
  else if (g_ActiveConfig.bSSAA)
  {
    // Sample shading, shader runs once per sample
    code.Write("  float4 val = texelFetch(samp0, coords, gl_SampleID);");
  }
  else
  {
    // MSAA without sample shading, average out all samples.
    code.Write("  float4 val = float4(0.0f, 0.0f, 0.0f, 0.0f);\n");
    code.Write("  for (int i = 0; i < {}; i++)\n", samples);
    code.Write("    val += texelFetch(samp0, coords, i);\n");
    code.Write("  val /= float({});\n", samples);
  }

  switch (convtype)
  {
  case EFBReinterpretType::RGB8ToRGBA6:
    code.Write("  int4 src8 = int4(round(val * 255.f));\n"
               "  int4 dst6;\n"
               "  dst6.r = src8.r >> 2;\n"
               "  dst6.g = ((src8.r & 0x3) << 4) | (src8.g >> 4);\n"
               "  dst6.b = ((src8.g & 0xF) << 2) | (src8.b >> 6);\n"
               "  dst6.a = src8.b & 0x3F;\n"
               "  ocol0 = float4(dst6) / 63.f;\n");
    break;

  case EFBReinterpretType::RGB8ToRGB565:
    code.Write("  ocol0 = val;\n");
    break;

  case EFBReinterpretType::RGBA6ToRGB8:
    code.Write("  int4 src6 = int4(round(val * 63.f));\n"
               "  int4 dst8;\n"
               "  dst8.r = (src6.r << 2) | (src6.g >> 4);\n"
               "  dst8.g = ((src6.g & 0xF) << 4) | (src6.b >> 2);\n"
               "  dst8.b = ((src6.b & 0x3) << 6) | src6.a;\n"
               "  dst8.a = 255;\n"
               "  ocol0 = float4(dst8) / 255.f;\n");
    break;

  case EFBReinterpretType::RGBA6ToRGB565:
    code.Write("  ocol0 = val;\n");
    break;

  case EFBReinterpretType::RGB565ToRGB8:
    code.Write("  ocol0 = val;\n");
    break;

  case EFBReinterpretType::RGB565ToRGBA6:
    //
    code.Write("  ocol0 = val;\n");
    break;
  }

  code.Write("}}\n");
  return code.GetBuffer();
}

std::string GenerateTextureReinterpretShader(TextureFormat from_format, TextureFormat to_format)
{
  ShaderCode code;
  EmitSamplerDeclarations(code, 0, 1, false);
  EmitPixelMainDeclaration(code, 1, 0, "float4", "", true);
  code.Write("{{\n"
             "  int layer = int(v_tex0.z);\n"
             "  int4 coords = int4(int2(frag_coord.xy), layer, 0);\n");

  // Convert to a 32-bit value encompassing all channels, filling the most significant bits with
  // zeroes.
  code.Write("  uint raw_value;\n");
  switch (from_format)
  {
  case TextureFormat::I8:
  case TextureFormat::C8:
  {
    code.Write("  float4 temp_value = ");
    EmitTextureLoad(code, 0, "coords");
    code.Write(";\n"
               "  raw_value = uint(temp_value.r * 255.0);\n");
  }
  break;

  case TextureFormat::IA8:
  {
    code.Write("  float4 temp_value = ");
    EmitTextureLoad(code, 0, "coords");
    code.Write(";\n"
               "  raw_value = uint(temp_value.r * 255.0) | (uint(temp_value.a * 255.0) << 8);\n");
  }
  break;

  case TextureFormat::I4:
  {
    code.Write("  float4 temp_value = ");
    EmitTextureLoad(code, 0, "coords");
    code.Write(";\n"
               "  raw_value = uint(temp_value.r * 15.0);\n");
  }
  break;

  case TextureFormat::IA4:
  {
    code.Write("  float4 temp_value = ");
    EmitTextureLoad(code, 0, "coords");
    code.Write(";\n"
               "  raw_value = uint(temp_value.r * 15.0) | (uint(temp_value.a * 15.0) << 4);\n");
  }
  break;

  case TextureFormat::RGB565:
  {
    code.Write("  float4 temp_value = ");
    EmitTextureLoad(code, 0, "coords");
    code.Write(";\n"
               "  raw_value = uint(temp_value.b * 31.0) | (uint(temp_value.g * 63.0) << 5) |\n"
               "              (uint(temp_value.r * 31.0) << 11);\n");
  }
  break;

  case TextureFormat::RGB5A3:
  {
    code.Write("  float4 temp_value = ");
    EmitTextureLoad(code, 0, "coords");
    code.Write(";\n");

    // 0.8784 = 224 / 255 which is the maximum alpha value that can be represented in 3 bits
    code.Write(
        "  if (temp_value.a > 0.878f) {{\n"
        "    raw_value = (uint(temp_value.b * 31.0)) | (uint(temp_value.g * 31.0) << 5) |\n"
        "                (uint(temp_value.r * 31.0) << 10) | 0x8000u;\n"
        "  }} else {{\n"
        "     raw_value = (uint(temp_value.b * 15.0)) | (uint(temp_value.g * 15.0) << 4) |\n"
        "                 (uint(temp_value.r * 15.0) << 8) | (uint(temp_value.a * 7.0) << 12);\n"
        "  }}\n");
  }
  break;

  default:
    WARN_LOG_FMT(VIDEO, "From format {} is not supported", from_format);
    return "{}\n";
  }

  // Now convert it to its new representation.
  switch (to_format)
  {
  case TextureFormat::I8:
  case TextureFormat::C8:
  {
    code.Write("  float orgba = float(raw_value & 0xFFu) / 255.0;\n"
               "  ocol0 = float4(orgba, orgba, orgba, orgba);\n");
  }
  break;

  case TextureFormat::IA8:
  {
    code.Write("  float orgb = float(raw_value & 0xFFu) / 255.0;\n"
               "  ocol0 = float4(orgb, orgb, orgb, float((raw_value >> 8) & 0xFFu) / 255.0);\n");
  }
  break;

  case TextureFormat::IA4:
  {
    code.Write("  float orgb = float(raw_value & 0xFu) / 15.0;\n"
               "  ocol0 = float4(orgb, orgb, orgb, float((raw_value >> 4) & 0xFu) / 15.0);\n");
  }
  break;

  case TextureFormat::RGB565:
  {
    code.Write("  ocol0 = float4(float((raw_value >> 10) & 0x1Fu) / 31.0,\n"
               "                 float((raw_value >> 5) & 0x1Fu) / 31.0,\n"
               "                 float(raw_value & 0x1Fu) / 31.0, 1.0);\n");
  }
  break;

  case TextureFormat::RGB5A3:
  {
    code.Write("  if ((raw_value & 0x8000u) != 0u) {{\n"
               "    ocol0 = float4(float((raw_value >> 10) & 0x1Fu) / 31.0,\n"
               "                   float((raw_value >> 5) & 0x1Fu) / 31.0,\n"
               "                   float(raw_value & 0x1Fu) / 31.0, 1.0);\n"
               "  }} else {{\n"
               "    ocol0 = float4(float((raw_value >> 8) & 0x0Fu) / 15.0,\n"
               "                   float((raw_value >> 4) & 0x0Fu) / 15.0,\n"
               "                   float(raw_value & 0x0Fu) / 15.0,\n"
               "                   float((raw_value >> 12) & 0x07u) / 7.0);\n"
               "  }}\n");
  }
  break;
  default:
    WARN_LOG_FMT(VIDEO, "To format {} is not supported", to_format);
    return "{}\n";
  }

  code.Write("}}\n");
  return code.GetBuffer();
}

// Depth of field over the EFB (first-person view). samp0 = EFB color,
// samp1 = EFB depth. Raw EFB depth is affine in 1/distance, and so is a thin
// lens's circle of confusion, so the blur radius is simply proportional to the
// raw depth difference from the focus (the depth at the screen centre).
// params: xy = texel size, z = circle-of-confusion scale, w = max radius (texels).
std::string GenerateDepthOfFieldPixelShader()
{
  // 16-point Vogel disc, radius normalized to 1.
  static constexpr std::array<std::array<float, 2>, 16> kTaps{{
      {0.177f, 0.000f},  {-0.226f, 0.207f}, {0.033f, -0.395f}, {0.287f, 0.359f},
      {-0.513f, -0.108f}, {0.500f, -0.299f}, {-0.191f, 0.622f}, {-0.330f, -0.592f},
      {0.701f, 0.207f},  {-0.701f, 0.330f}, {0.264f, -0.758f}, {0.398f, 0.746f},
      {-0.865f, -0.243f}, {0.876f, -0.395f}, {-0.330f, 0.907f}, {-0.360f, -0.933f},
  }};
  ShaderCode code;
  EmitUniformBufferDeclaration(code);
  code.Write("{{\n"
             "  float4 params;\n"
             "  float4 params2;\n"
             "}};\n\n");
  EmitSamplerDeclarations(code, 0, 3, false);
  EmitPixelMainDeclaration(code, 1, 0);
  // q = inverse distance (up to a constant): the raw depth measured from the
  // far end (params2.x = 1 when far is stored as 1). Blur grows with the
  // relative difference |q - q_focus| / q_focus, independent of the projection,
  // minus an in-focus band (params2.y) so small depth changes stay sharp. The
  // focus comes from the smoothed 1x1 focus texture (samp2).
  code.Write("{{\n"
             "#define INVDEPTH(uv) abs(params2.x - texture(samp1, uv).r)\n"
             "  float focus = texture(samp2, float3(0.5, 0.5, 0.0)).r;\n"
             "  float rcp_focus = 1.0 / max(focus, 1e-6);\n"
             "#define COC(q) clamp((abs((q) - focus) * rcp_focus - params2.y) * params.z, 0.0, 1.0)\n"
             "  float coc = COC(INVDEPTH(v_tex0));\n"
             "  float4 sum = texture(samp0, v_tex0);\n"
             "  float weight = 1.0;\n"
             "  if (coc > 0.02)\n"
             "  {{\n"
             "    float2 radius = params.xy * (params.w * coc);\n");
  for (const auto& tap : kTaps)
  {
    code.Write("    {{\n"
               "      float3 uv = float3(v_tex0.xy + float2({:.3f}, {:.3f}) * radius, v_tex0.z);\n"
               "      float w = 0.25 + COC(INVDEPTH(uv));\n"
               "      sum += texture(samp0, uv) * w;\n"
               "      weight += w;\n"
               "    }}\n",
               tap[0], tap[1]);
  }
  code.Write("  }}\n"
             "  ocol0 = sum / weight;\n");
  // MODERNGEKKO_DOF_DEBUG=1: show the circle of confusion (white = most blur).
  if (const char* debug = std::getenv("MODERNGEKKO_DOF_DEBUG"); debug && debug[0] == '1')
    code.Write("  ocol0 = float4(coc, texture(samp1, v_tex0).r, clamp(params.z * 0.25, 0.0, 1.0), 1.0);\n");
  code.Write("}}\n");
  return code.GetBuffer();
}

// Depth-of-field focus, rendered into a 1x1 texture: the mean inverse depth of a
// 5x5 grid over a window around the screen centre (the first-person dot),
// eased toward from the previous frame's focus (samp0) so characters passing
// the centre don't snap the focus. samp1 = EFB depth.
// params: x = blend toward the new focus (1 = no smoothing), y = 1 when far is
// stored as 1, zw = window half-size (UV).
std::string GenerateDepthOfFieldFocusPixelShader()
{
  ShaderCode code;
  EmitUniformBufferDeclaration(code);
  code.Write("{{\n"
             "  float4 params;\n"
             "}};\n\n");
  EmitSamplerDeclarations(code, 0, 2, false);
  EmitPixelMainDeclaration(code, 1, 0);
  code.Write("{{\n"
             "  float sum = 0.0;\n");
  for (int y = -2; y <= 2; y++)
  {
    for (int x = -2; x <= 2; x++)
    {
      code.Write("  sum += abs(params.y - texture(samp1, float3(0.5 + {:.1f} * params.z, "
                 "0.5 + {:.1f} * params.w, 0.0)).r);\n",
                 x * 0.5f, y * 0.5f);
    }
  }
  code.Write("  float current = sum / 25.0;\n"
             "  float previous = texture(samp0, float3(0.5, 0.5, 0.0)).r;\n"
             "  ocol0 = float4(previous > 0.0 ? lerp(previous, current, params.x) : current, 0.0, 0.0, 1.0);\n"
             "}}\n");
  return code.GetBuffer();
}

std::string GenerateEFBRestorePixelShader()
{
  ShaderCode code;
  EmitSamplerDeclarations(code, 0, 2, false);
  EmitPixelMainDeclaration(code, 1, 0, "float4", "");
  code.Write("{{\n"
             "  ocol0 = ");
  EmitSampleTexture(code, 0, "v_tex0");
  code.Write(";\n");
  code.Write("  gl_FragDepth = ");
  EmitSampleTexture(code, 1, "v_tex0");
  code.Write(".r;\n"
             "}}\n");
  return code.GetBuffer();
}

std::string GenerateImGuiVertexShader()
{
  ShaderCode code;

  // Uniform buffer contains the viewport size, and we transform in the vertex shader.
  EmitUniformBufferDeclaration(code);
  code.Write("{{\n"
             "float2 u_rcp_viewport_size_mul2;\n"
             "}};\n\n");

  EmitVertexMainDeclaration(code, 1, 1, true, 1, 1);
  code.Write("{{\n"
             "  v_tex0 = float3(rawtex0.xy, 0.0);\n"
             "  v_col0 = rawcolor0;\n"
             "  opos = float4(rawpos.x * u_rcp_viewport_size_mul2.x - 1.0,"
             "                1.0 - rawpos.y * u_rcp_viewport_size_mul2.y, 0.0, 1.0);\n");

  // NDC space is flipped in Vulkan.
  if (GetAPIType() == APIType::Vulkan)
    code.Write("  opos.y = -opos.y;\n");

  code.Write("}}\n");
  return code.GetBuffer();
}

std::string GenerateImGuiPixelShader(bool linear_space_output)
{
  ShaderCode code;
  EmitSamplerDeclarations(code, 0, 1, false);
  EmitPixelMainDeclaration(code, 1, 1);
  code.Write("{{\n"
             "  ocol0 = ");
  EmitSampleTexture(code, 0, "float3(v_tex0.xy, 0.0)");
  // We approximate to gamma 2.2 instead of sRGB as it barely matters for this case.
  // Note that if HDR is enabled, ideally we should multiply by
  // the paper white brightness for readability.
  if (linear_space_output)
    code.Write(" * pow(v_col0, float4(2.2f, 2.2f, 2.2f, 1.0f));\n}}\n");
  else
    code.Write(" * v_col0;\n}}\n");

  return code.GetBuffer();
}

}  // namespace FramebufferShaderGen
