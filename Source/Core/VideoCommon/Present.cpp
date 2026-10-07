// Copyright 2023 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/Present.h"

#include <algorithm>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "Common/ChunkFile.h"
#include "Common/Timer.h"
#include "Core/Config/GraphicsSettings.h"
#include "Core/Config/MainSettings.h"
#include "Core/CoreTiming.h"
#include "Core/HW/SystemTimers.h"
#include "Core/HW/VideoInterface.h"
#include "Core/Host.h"
#include "Core/System.h"

#include "InputCommon/ControllerInterface/ControllerInterface.h"

#include "Present.h"
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/ColosseumProjection.h"
#include "VideoCommon/FrameDumper.h"
#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/OnScreenUI.h"
#include "VideoCommon/PostProcessing.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VideoConfig.h"
#include "VideoCommon/VideoEvents.h"
#include "VideoCommon/Widescreen.h"

std::unique_ptr<VideoCommon::Presenter> g_presenter;

namespace VideoCommon
{
// Stretches the native/internal analog resolution aspect ratio from ~4:3 to ~16:9
static float SourceAspectRatioToWidescreen(float source_aspect)
{
  return source_aspect * ((16.0f / 9.0f) / (4.0f / 3.0f));
}

static std::tuple<int, int> FindClosestIntegerResolution(float width, float height,
                                                         float aspect_ratio)
{
  // We can't round both the x and y resolution as that might generate an aspect ratio
  // further away from the target one, we also can't either ceil or floor both sides,
  // so we find the combination or flooring and ceiling that is closest to the target ar.
  const int ceiled_width = static_cast<int>(std::ceil(width));
  const int ceiled_height = static_cast<int>(std::ceil(height));
  const int floored_width = static_cast<int>(std::floor(width));
  const int floored_height = static_cast<int>(std::floor(height));

  int int_width = floored_width;
  int int_height = floored_height;

  float min_aspect_ratio_distance = std::numeric_limits<float>::max();
  for (const int new_width : std::array<int, 2>{ceiled_width, floored_width})
  {
    for (const int new_height : std::array<int, 2>{ceiled_height, floored_height})
    {
      const float new_aspect_ratio = static_cast<float>(new_width) / new_height;
      const float aspect_ratio_distance = std::abs((new_aspect_ratio / aspect_ratio) - 1.f);
      if (aspect_ratio_distance < min_aspect_ratio_distance)
      {
        min_aspect_ratio_distance = aspect_ratio_distance;
        int_width = new_width;
        int_height = new_height;
      }
    }
  }

  return std::make_tuple(int_width, int_height);
}

static void TryToSnapToXFBSize(int& width, int& height, int xfb_width, int xfb_height)
{
  // Screen is blanking (e.g. game booting up), nothing to do here
  if (xfb_width == 0 || xfb_height == 0)
    return;

  // If there's only 1 pixel of either horizontal or vertical resolution difference,
  // make the output size match a multiple of the XFB native resolution,
  // to achieve the highest quality (least scaling).
  // The reason why the threshold is 1 pixel (per internal resolution multiplier) is because of
  // minor inaccuracies of the VI aspect ratio (and because some resolutions are rounded
  // while other are floored).
  const unsigned int efb_scale = g_framebuffer_manager->GetEFBScale();
  const unsigned int pixel_difference_width = std::abs(width - xfb_width);
  const unsigned int pixel_difference_height = std::abs(height - xfb_height);
  // We ignore this if there's an offset on both hor and ver size,
  // as then we'd be changing the aspect ratio too much and would need to
  // re-calculate a lot of stuff (like black bars).
  if ((pixel_difference_width <= efb_scale && pixel_difference_height == 0) ||
      (pixel_difference_height <= efb_scale && pixel_difference_width == 0))
  {
    width = xfb_width;
    height = xfb_height;
  }
}

Presenter::Presenter()
{
  auto& video_events = GetVideoEvents();

  m_config_changed =
      video_events.config_changed_event.Register([this](u32 bits) { ConfigChanged(bits); });

  m_end_field_hook = video_events.vi_end_field_event.Register(
      [this] { m_immediate_swap_happened_this_field.store(false, std::memory_order_relaxed); });
}

Presenter::~Presenter()
{
  // Disable ControllerInterface's aspect ratio adjustments so mapping dialog behaves normally.
  g_controller_interface.SetAspectRatioAdjustment(1);
}

bool Presenter::Initialize()
{
  UpdateDrawRectangle();

  m_immediate_swap_happened_this_field.store(false, std::memory_order_relaxed);

  if (!g_gfx->IsHeadless())
  {
    SetBackbuffer(g_gfx->GetSurfaceInfo());

    m_post_processor = std::make_unique<VideoCommon::PostProcessing>();
    if (!m_post_processor->Initialize(m_backbuffer_format))
      return false;

    m_onscreen_ui = std::make_unique<OnScreenUI>();
    if (!m_onscreen_ui->Initialize(m_backbuffer_width, m_backbuffer_height, m_backbuffer_scale))
      return false;

    // Draw a blank frame (and complete OnScreenUI initialization)
    g_gfx->BindBackbuffer({{0.0f, 0.0f, 0.0f, 1.0f}});
    g_gfx->PresentBackbuffer();
  }

  return true;
}

bool Presenter::FetchXFB(u32 xfb_addr, u32 fb_width, u32 fb_stride, u32 fb_height, u64 ticks)
{
  ReleaseXFBContentLock();
  u64 old_xfb_id = m_last_xfb_id;

  if (fb_width == 0 || fb_height == 0)
  {
    // Game is blanking the screen
    m_xfb_entry.reset();
    m_xfb_rect = MathUtil::Rectangle<int>();
    m_last_xfb_id = std::numeric_limits<u64>::max();
  }
  else
  {
    m_xfb_entry =
        g_texture_cache->GetXFBTexture(xfb_addr, fb_width, fb_height, fb_stride, &m_xfb_rect);
    m_last_xfb_id = m_xfb_entry->id;

    m_xfb_entry->AcquireContentLock();
  }
  m_last_xfb_addr = xfb_addr;
  m_last_xfb_ticks = ticks;
  m_last_xfb_width = fb_width;
  m_last_xfb_stride = fb_stride;
  m_last_xfb_height = fb_height;

  return old_xfb_id == m_last_xfb_id;
}

void Presenter::ViSwap(u32 xfb_addr, u32 fb_width, u32 fb_stride, u32 fb_height, u64 ticks,
                       TimePoint presentation_time)
{
  bool is_duplicate = FetchXFB(xfb_addr, fb_width, fb_stride, fb_height, ticks);

  PresentInfo present_info{
      .present_count = m_present_count++,
      .emulated_timestamp = ticks,
      .intended_present_time = presentation_time,
  };

  if (is_duplicate)
  {
    present_info.frame_count = m_frame_count - 1;  // Previous frame
    present_info.reason = PresentInfo::PresentReason::VideoInterfaceDuplicate;
  }
  else
  {
    present_info.frame_count = m_frame_count++;
    present_info.reason = PresentInfo::PresentReason::VideoInterface;
  }

  if (m_xfb_entry)
  {
    // With no references, this XFB copy wasn't stitched together
    // so just use its name directly
    if (m_xfb_entry->references.empty())
    {
      if (!m_xfb_entry->texture_info_name.empty())
        present_info.xfb_copy_hashes.push_back(m_xfb_entry->texture_info_name);
    }
    else
    {
      for (const auto& reference : m_xfb_entry->references)
      {
        if (!reference->texture_info_name.empty())
          present_info.xfb_copy_hashes.push_back(reference->texture_info_name);
      }
    }
  }

  auto& video_events = GetVideoEvents();

  video_events.before_present_event.Trigger(present_info);

  if (!is_duplicate || !g_ActiveConfig.bSkipPresentingDuplicateXFBs)
  {
    if (g_ActiveConfig.stereo_mode == StereoMode::FrameInterp && m_xfb_entry && !is_duplicate)
    {
      // MODERNGEKKO_INTERP_PRESENT: "ba" (default) presents the in-between
      // layer then the real one; "a"/"b" present only one layer (diagnostic).
      static const std::string order = [] {
        const char* v = std::getenv("MODERNGEKKO_INTERP_PRESENT");
        return std::string(v && *v ? v : "ba");
      }();
      m_interp_layers = m_xfb_entry->texture ?
                            std::clamp<u32>(m_xfb_entry->texture->GetLayers(), 2, 4) :
                            2;
      // Behind schedule (the game frame arrived more than a step after its
      // slot): in-between presents cost CPU time on this thread, so show only
      // the real frame until emulation catches up - heavy scenes such as
      // battles lose smoothness instead of game speed.
      const bool behind = m_interp_grid != TimePoint{} &&
                          Clock::now() > m_interp_frame_end + std::chrono::milliseconds(4);
      if (order == "ba" && (behind || !InterpLayerAllowed()))
      {
        // Display or emulation cannot keep up: show only the real frame, and
        // expect the next one a field from now (if it is late again, keep
        // shedding; once frames arrive on time, in-between frames resume).
        m_interp_pending = false;
        const TimePoint shown = Clock::now();
        m_interp_grid = shown;
        m_interp_frame_end = shown + std::chrono::microseconds(16683);
        m_interp_present_layer = 0;
        Present(&present_info);
      }
      else if (order == "ba")
      {
        // In-between frames now and at 1/n-field steps, then the real frame,
        // held to a wall-clock grid while emulation keeps running in between.
        m_interp_present_layer = 1;
        WaitForInterpSlot();
        Present(&present_info);
        m_interp_present_layer = 0;
        ScheduleInterpRealPresent(present_info);
      }
      else
      {
        for (const char layer : order)
        {
          m_interp_present_layer = layer == 'b' ? 1 : 0;
          Present(&present_info);
        }
        m_interp_present_layer = 0;
      }
    }
    else
    {
      m_interp_present_layer = 0;
      Present(&present_info);
    }
    ProcessFrameDumping(ticks);

    video_events.after_present_event.Trigger(present_info);
  }
}

void Presenter::ScheduleInterpRealPresent(const PresentInfo& in_between)
{
  auto& system = Core::System::GetInstance();

  // n layers per game frame: layer 1 was just presented; layers 2..n-1 and
  // then the real frame (layer 0) follow at 1/n-field steps.
  m_interp_layers = m_xfb_entry && m_xfb_entry->texture ?
                        std::clamp<u32>(m_xfb_entry->texture->GetLayers(), 2, 4) :
                        2;
  m_interp_step_ticks = system.GetVideoInterface().GetTicksPerField() / m_interp_layers;
  m_interp_next_layer = m_interp_layers > 2 ? 2 : 0;

  m_interp_pending = true;
  m_interp_pending_xfb_id = m_last_xfb_id;
  m_interp_pending_info = in_between;
  // Wake well before each slot: GPU work for the next frame runs inline on this
  // thread and delays CoreTiming events by milliseconds, while waking early is
  // free (WaitForInterpSlot holds the present to its wall-clock slot). The
  // first step, right after the swap, is hit hardest.
  ScheduleInterpStep(m_interp_step_ticks / 4);
}

void Presenter::ScheduleInterpStep(u64 delay_ticks)
{
  auto& system = Core::System::GetInstance();
  auto& core_timing = system.GetCoreTiming();
  static CoreTiming::EventType* event = nullptr;
  if (!event)
  {
    event = core_timing.RegisterEvent(
        "FrameInterpRealPresent", [](Core::System&, u64, s64) {
          if (g_presenter)
            g_presenter->PresentInterpReal();
        });
  }
  const double step_seconds = static_cast<double>(m_interp_step_ticks) /
                              system.GetSystemTimers().GetTicksPerSecond();
  m_interp_pending_info.intended_present_time +=
      std::chrono::duration_cast<DT>(std::chrono::duration<double>(step_seconds));
  core_timing.RemoveEvent(event);
  core_timing.ScheduleEvent(static_cast<s64>(delay_ticks), event, 0,
                            CoreTiming::FromThread::ANY);
}

namespace
{
// MODERNGEKKO_INTERP_STATS: how presents met their wall-clock slots.
struct
{
  u32 late = 0;
  u32 late_layer[4] = {};
  u32 resyncs = 0;
  double late_max_ms = 0;
} g_interp_pace_stats;
}  // namespace

bool Presenter::InterpLayerAllowed()
{
  const double step_ms = 1000.0 / 60.0 / std::max<u32>(m_interp_layers, 2);
  const TimePoint now = Clock::now();
  if (m_interp_dropping)
  {
    if (now < m_interp_retry)
      return false;
    m_interp_dropping = false;  // try again
    m_interp_block_ms = 0;
    g_frame_interp_dropping.store(false, std::memory_order_relaxed);
  }
  else if (m_interp_block_ms > 0.5 * step_ms)
  {
    m_interp_dropping = true;
    m_interp_retry = now + std::chrono::seconds(2);
    g_frame_interp_dropping.store(true, std::memory_order_relaxed);
    return false;
  }
  return true;
}

void Presenter::WaitForInterpSlot()
{
  // Emulated time runs ahead in bursts and the frame limiter catches up later,
  // so CoreTiming-scheduled presents bunch together (0.5 ms then 8 ms apart),
  // which looks no smoother than 60 FPS. Hold each present to a wall-clock
  // grid of field/n steps instead. Sleeping here only happens while emulation
  // is ahead of that grid, i.e. time the limiter would otherwise sleep anyway.
  auto& system = Core::System::GetInstance();
  const double field_seconds =
      static_cast<double>(system.GetVideoInterface().GetTicksPerField()) /
      system.GetSystemTimers().GetTicksPerSecond();
  const auto step = std::chrono::duration_cast<DT>(
      std::chrono::duration<double>(field_seconds / std::max<u32>(m_interp_layers, 2)));
  static Common::PrecisionTimer timer;

  // Presents per game frame run 1, 2, ..., n-1, 0; this many steps remain from
  // this present to the next frame's first one.
  const u32 layers = std::max<u32>(m_interp_layers, 2);
  const u32 layer = static_cast<u32>(std::clamp(m_interp_present_layer, 0, 3));
  const u32 remaining = layer == 0 ? 1 : layers - layer + 1;

  const TimePoint now = Clock::now();
  if (m_interp_grid == TimePoint{} || now > m_interp_grid + step || m_interp_grid > now + 2 * step)
  {
    // Late by more than a step, or far off: restart the grid here.
    m_interp_grid = now;
    m_interp_frame_end = now + remaining * step;
    ++g_interp_pace_stats.resyncs;
  }
  else if (m_interp_grid > now)
  {
    timer.SleepUntil(m_interp_grid);
    g_interp_pace_stats.late_max_ms = std::max(
        g_interp_pace_stats.late_max_ms, DT_ms(Clock::now() - m_interp_grid).count());
  }
  else
  {
    ++g_interp_pace_stats.late;
    ++g_interp_pace_stats.late_layer[layer];
    g_interp_pace_stats.late_max_ms =
        std::max(g_interp_pace_stats.late_max_ms, DT_ms(now - m_interp_grid).count());
  }
  // A frame's first present fixes where the next frame starts (its slot plus n
  // steps), keeping the 60 Hz cadence. When a present runs late (usually that
  // first one, whose timing the game sets), the rest of the frame's presents
  // share the delay evenly instead of the next one bunching up behind it.
  if (layer == 1)
    m_interp_frame_end = m_interp_grid + remaining * step;
  const TimePoint shown = std::max(now, m_interp_grid);
  m_interp_grid = shown + std::max<DT>((m_interp_frame_end - shown) / remaining, step / 2);
}

void Presenter::PresentInterpReal()
{
  // Skip if a newer frame arrived (or the screen blanked) in the meantime.
  if (!m_interp_pending || !m_xfb_entry || m_last_xfb_id != m_interp_pending_xfb_id)
  {
    m_interp_pending = false;
    return;
  }
  const u32 layer = m_interp_next_layer;
  m_interp_present_layer = static_cast<int>(layer);
  WaitForInterpSlot();
  Present(&m_interp_pending_info);
  m_interp_present_layer = 0;
  if (layer == 0)
  {
    m_interp_pending = false;
    return;
  }
  m_interp_next_layer = layer + 1 < m_interp_layers ? layer + 1 : 0;
  ScheduleInterpStep(m_interp_step_ticks / 2);
}

void Presenter::ImmediateSwap(u32 xfb_addr, u32 fb_width, u32 fb_stride, u32 fb_height)
{
  if (m_immediate_swap_happened_this_field.exchange(true, std::memory_order_relaxed) &&
      Config::Get(Config::GFX_HACK_CAP_IMMEDIATE_XFB))
  {
    return;
  }

  const u64 ticks = m_next_swap_estimated_ticks;

  FetchXFB(xfb_addr, fb_width, fb_stride, fb_height, ticks);

  PresentInfo present_info{
      .frame_count = m_frame_count++,
      .present_count = m_present_count++,
      .reason = PresentInfo::PresentReason::Immediate,
      .emulated_timestamp = ticks,
      .intended_present_time = m_next_swap_estimated_time,
  };

  auto& video_events = GetVideoEvents();

  video_events.before_present_event.Trigger(present_info);

  Present(&present_info);
  ProcessFrameDumping(ticks);

  video_events.after_present_event.Trigger(present_info);
}

void Presenter::SetNextSwapEstimatedTime(u64 ticks, TimePoint host_time)
{
  m_next_swap_estimated_ticks = ticks;
  m_next_swap_estimated_time = host_time;
}

void Presenter::ProcessFrameDumping(u64 ticks) const
{
  if (g_frame_dumper->IsFrameDumping() && m_xfb_entry)
  {
    MathUtil::Rectangle<int> target_rect;
    switch (Config::Get(Config::GFX_FRAME_DUMPS_RESOLUTION_TYPE))
    {
    default:
    case FrameDumpResolutionType::WindowResolution:
    {
      if (!g_gfx->IsHeadless())
      {
        target_rect = GetTargetRectangle();
        break;
      }
      [[fallthrough]];
    }
    case FrameDumpResolutionType::XFBAspectRatioCorrectedResolution:
    {
      target_rect = m_xfb_rect;
      const bool allow_stretch = false;
      auto [float_width, float_height] =
          ScaleToDisplayAspectRatio(m_xfb_rect.GetWidth(), m_xfb_rect.GetHeight(), allow_stretch);
      const float draw_aspect_ratio = CalculateDrawAspectRatio(allow_stretch);
      auto [int_width, int_height] =
          FindClosestIntegerResolution(float_width, float_height, draw_aspect_ratio);
      target_rect = MathUtil::Rectangle<int>(0, 0, int_width, int_height);
      break;
    }
    case FrameDumpResolutionType::XFBRawResolution:
    {
      target_rect = m_xfb_rect;
      break;
    }
    }

    int width = target_rect.GetWidth();
    int height = target_rect.GetHeight();

    const int resolution_lcm = g_frame_dumper->GetRequiredResolutionLeastCommonMultiple();

    // Ensure divisibility by the dumper LCM and a min of 1 to make it compatible with all the
    // video encoders. Note that this is theoretically only necessary when recording videos and not
    // screenshots.
    // We always scale positively to make sure the least amount of information is lost.
    //
    // TODO: this should be added as black padding on the edges by the frame dumper.
    if ((width % resolution_lcm) != 0 || width == 0)
      width += resolution_lcm - (width % resolution_lcm);
    if ((height % resolution_lcm) != 0 || height == 0)
      height += resolution_lcm - (height % resolution_lcm);

    // Remove any black borders, there would be no point in including them in the recording
    target_rect.left = 0;
    target_rect.top = 0;
    target_rect.right = width;
    target_rect.bottom = height;

    // TODO: any scaling done by this won't be gamma corrected,
    // we should either apply post processing as well, or port its gamma correction code
    g_frame_dumper->DumpCurrentFrame(m_xfb_entry->texture.get(), m_xfb_rect, target_rect, ticks,
                                     m_frame_count);
  }
}

void Presenter::SetBackbuffer(int backbuffer_width, int backbuffer_height)
{
  const bool is_first = m_backbuffer_width == 0 && m_backbuffer_height == 0;
  const bool size_changed =
      (m_backbuffer_width != backbuffer_width || m_backbuffer_height != backbuffer_height);
  m_backbuffer_width = backbuffer_width;
  m_backbuffer_height = backbuffer_height;
  UpdateDrawRectangle();

  OnBackbufferSet(size_changed, is_first);
}

void Presenter::SetBackbuffer(SurfaceInfo info)
{
  const bool is_first = m_backbuffer_width == 0 && m_backbuffer_height == 0;
  const bool size_changed =
      (m_backbuffer_width != (int)info.width || m_backbuffer_height != (int)info.height);
  m_backbuffer_width = info.width;
  m_backbuffer_height = info.height;
  m_backbuffer_scale = info.scale;
  m_backbuffer_format = info.format;
  if (m_onscreen_ui)
    m_onscreen_ui->SetScale(info.scale);

  OnBackbufferSet(size_changed, is_first);
}

void Presenter::OnBackbufferSet(bool size_changed, bool is_first_set)
{
  UpdateDrawRectangle();

  // Automatically update the resolution scale if the window size changed,
  // or if the game XFB resolution changed.
  if (size_changed && !is_first_set && g_ActiveConfig.iEFBScale == EFB_SCALE_AUTO_INTEGRAL &&
      m_auto_resolution_scale != AutoIntegralScale())
  {
    g_framebuffer_manager->RecreateEFBFramebuffer(g_ActiveConfig.iEFBScale);
  }
  if (size_changed || is_first_set)
  {
    m_auto_resolution_scale = AutoIntegralScale();
  }
}

void Presenter::ConfigChanged(u32 changed_bits)
{
  // Check for post-processing shader changes. Done up here as it doesn't affect anything outside
  // the post-processor. Note that options are applied every frame, so no need to check those.
  if (changed_bits & ConfigChangeBits::CONFIG_CHANGE_BIT_POST_PROCESSING_SHADER && m_post_processor)
  {
    // The existing shader must not be in use when it's destroyed
    g_gfx->WaitForGPUIdle();

    m_post_processor->RecompileShader();
  }

  // Stereo mode change requires recompiling our post processing pipeline and imgui pipelines for
  // rendering the UI.
  if (changed_bits & ConfigChangeBits::CONFIG_CHANGE_BIT_STEREO_MODE)
  {
    if (m_onscreen_ui)
      m_onscreen_ui->RecompileImGuiPipeline();
    if (m_post_processor)
      m_post_processor->RecompilePipeline();
  }
}

std::tuple<MathUtil::Rectangle<int>, MathUtil::Rectangle<int>>
Presenter::ConvertStereoRectangle(const MathUtil::Rectangle<int>& rc) const
{
  // Resize target to half its original size
  auto draw_rc = rc;
  if (g_ActiveConfig.stereo_mode == StereoMode::TopAndBottom)
  {
    // The height may be negative due to flipped rectangles
    int height = rc.bottom - rc.top;
    draw_rc.top += height / 4;
    draw_rc.bottom -= height / 4;
  }
  else
  {
    int width = rc.right - rc.left;
    draw_rc.left += width / 4;
    draw_rc.right -= width / 4;
  }

  // Create two target rectangle offset to the sides of the backbuffer
  auto left_rc = draw_rc;
  auto right_rc = draw_rc;
  if (g_ActiveConfig.stereo_mode == StereoMode::TopAndBottom)
  {
    left_rc.top -= m_backbuffer_height / 4;
    left_rc.bottom -= m_backbuffer_height / 4;
    right_rc.top += m_backbuffer_height / 4;
    right_rc.bottom += m_backbuffer_height / 4;
  }
  else
  {
    left_rc.left -= m_backbuffer_width / 4;
    left_rc.right -= m_backbuffer_width / 4;
    right_rc.left += m_backbuffer_width / 4;
    right_rc.right += m_backbuffer_width / 4;
  }

  return std::make_tuple(left_rc, right_rc);
}

MathUtil::Rectangle<int> Presenter::GetCustomCrop(const MathUtil::Rectangle<int>& rect) const
{
  if (!g_ActiveConfig.bCropCustom)
    return MathUtil::Rectangle<int>(0, 0, 0, 0);

  const int uncropped_source_width = rect.GetWidth();
  const int uncropped_source_height = rect.GetHeight();
  const int efb_scale = g_framebuffer_manager->GetEFBScale();

  // Determine amount of pixels to crop from the source rect.
  const int source_crop_left =
      std::min(g_ActiveConfig.iCropCustomLeft * efb_scale, uncropped_source_width);
  const int source_crop_right = std::min(g_ActiveConfig.iCropCustomRight * efb_scale,
                                         uncropped_source_width - source_crop_left);
  const int source_crop_top =
      std::min(g_ActiveConfig.iCropCustomTop * efb_scale, uncropped_source_height);
  const int source_crop_bottom = std::min(g_ActiveConfig.iCropCustomBottom * efb_scale,
                                          uncropped_source_height - source_crop_top);

  return MathUtil::Rectangle<int>(source_crop_left, source_crop_top, source_crop_right,
                                  source_crop_bottom);
}

MathUtil::Rectangle<int> Presenter::AdjustForCustomCrop(const MathUtil::Rectangle<int>& rect) const
{
  const MathUtil::Rectangle<int> crop = GetCustomCrop(rect);
  const MathUtil::Rectangle<int> cropped(rect.left + crop.left, rect.top + crop.top,
                                         rect.right - crop.right, rect.bottom - crop.bottom);
  return cropped;
}

float Presenter::AdjustAspectRatioForCustomCrop(float input_aspect_ratio) const
{
  if (!g_ActiveConfig.bCropCustom)
    return input_aspect_ratio;

  const MathUtil::Rectangle<int> rect = m_xfb_rect;
  if (rect.GetWidth() <= 0 || rect.GetHeight() <= 0)
    return input_aspect_ratio;

  const MathUtil::Rectangle<int> cropped = AdjustForCustomCrop(rect);
  const float relative_width_difference =
      static_cast<float>(cropped.GetWidth()) / static_cast<float>(rect.GetWidth());
  const float relative_height_difference =
      static_cast<float>(cropped.GetHeight()) / static_cast<float>(rect.GetHeight());
  return input_aspect_ratio * (relative_width_difference / relative_height_difference);
}

float Presenter::CalculateDrawAspectRatio(bool allow_stretch) const
{
  auto aspect_mode = g_ActiveConfig.aspect_mode;
  float resulting_aspect_ratio;

  if (!allow_stretch && aspect_mode == AspectMode::Stretch)
    aspect_mode = AspectMode::Auto;

  // If stretch is enabled, we prefer the aspect ratio of the window.
  if (aspect_mode == AspectMode::Stretch)
  {
    resulting_aspect_ratio =
        (static_cast<float>(m_backbuffer_width) / static_cast<float>(m_backbuffer_height));
  }
  else
  {
    // The actual aspect ratio of the XFB texture is irrelevant, the VI one is the one that matters
    const auto& vi = Core::System::GetInstance().GetVideoInterface();
    const float vi_aspect_ratio = vi.GetAspectRatio();
    const float source_aspect_ratio = GetColosseumPresentationAspect(
        AdjustAspectRatioForCustomCrop(vi_aspect_ratio),
        g_ActiveConfig.bWidescreenAuthoredMenu);

    // This will scale up the source ~4:3 resolution to its equivalent ~16:9 resolution
    if (aspect_mode == AspectMode::ForceWide ||
        (aspect_mode == AspectMode::Auto && g_widescreen->IsGameWidescreen()))
    {
      resulting_aspect_ratio = SourceAspectRatioToWidescreen(source_aspect_ratio);
    }
    else if (aspect_mode == AspectMode::Custom)
    {
      resulting_aspect_ratio =
          source_aspect_ratio * (g_ActiveConfig.GetCustomAspectRatio() / (4.0f / 3.0f));
    }
    // For the "custom stretch" mode, we force the exact target aspect ratio, without
    // acknowledging the difference between the source aspect ratio and 4:3.
    else if (aspect_mode == AspectMode::CustomStretch)
    {
      resulting_aspect_ratio = g_ActiveConfig.GetCustomAspectRatio();
    }
    else if (aspect_mode == AspectMode::Raw)
    {
      resulting_aspect_ratio =
          m_xfb_entry ? (static_cast<float>(m_last_xfb_width) / m_last_xfb_height) : 1.f;
    }
    else
    {
      resulting_aspect_ratio = source_aspect_ratio;
    }
  }

  if (g_ActiveConfig.stereo_per_eye_resolution_full)
  {
    if (g_ActiveConfig.stereo_mode == StereoMode::SideBySide)
    {
      // Render twice as wide if using side-by-side 3D, since the 3D will halve the horizontal
      // resolution
      resulting_aspect_ratio *= 2.0;
    }
    else if (g_ActiveConfig.stereo_mode == StereoMode::TopAndBottom)
    {
      // Render twice as tall if using top-and-bottom 3D, since the 3D will halve the vertical
      // resolution
      resulting_aspect_ratio /= 2.0;
    }
  }

  return resulting_aspect_ratio;
}

void Presenter::AdjustRectanglesToFitBounds(MathUtil::Rectangle<int>* target_rect,
                                            MathUtil::Rectangle<int>* source_rect, int fb_width,
                                            int fb_height)
{
  const int orig_target_width = target_rect->GetWidth();
  const int orig_target_height = target_rect->GetHeight();
  const int orig_source_width = source_rect->GetWidth();
  const int orig_source_height = source_rect->GetHeight();
  if (target_rect->left < 0)
  {
    const int offset = -target_rect->left;
    target_rect->left = 0;
    source_rect->left += offset * orig_source_width / orig_target_width;
  }
  if (target_rect->right > fb_width)
  {
    const int offset = target_rect->right - fb_width;
    target_rect->right -= offset;
    source_rect->right -= offset * orig_source_width / orig_target_width;
  }
  if (target_rect->top < 0)
  {
    const int offset = -target_rect->top;
    target_rect->top = 0;
    source_rect->top += offset * orig_source_height / orig_target_height;
  }
  if (target_rect->bottom > fb_height)
  {
    const int offset = target_rect->bottom - fb_height;
    target_rect->bottom -= offset;
    source_rect->bottom -= offset * orig_source_height / orig_target_height;
  }
}

void Presenter::ReleaseXFBContentLock()
{
  if (m_xfb_entry)
    m_xfb_entry->ReleaseContentLock();
}

void Presenter::ChangeSurface(void* new_surface_handle)
{
  std::lock_guard<std::mutex> lock(m_swap_mutex);
  m_new_surface_handle = new_surface_handle;
  m_surface_changed.Set();
}

void Presenter::ResizeSurface()
{
  std::lock_guard<std::mutex> lock(m_swap_mutex);
  m_surface_resized.Set();
}

void* Presenter::GetNewSurfaceHandle()
{
  void* handle = m_new_surface_handle;
  m_new_surface_handle = nullptr;
  return handle;
}

u32 Presenter::AutoIntegralScale() const
{
  // Take the source/native resolution (XFB) and stretch it on the target (window) aspect ratio.
  // If the target resolution is larger (on either x or y), we scale the source
  // by a integer multiplier until it won't have to be scaled up anymore.
  // NOTE: this might conflict with "Config::MAIN_RENDER_WINDOW_AUTOSIZE",
  // as they mutually influence each other.
  u32 source_width = m_last_xfb_width;
  u32 source_height = m_last_xfb_height;
  const u32 target_width = m_target_rectangle.GetWidth();
  const u32 target_height = m_target_rectangle.GetHeight();
  const float source_aspect_ratio = (float)source_width / source_height;
  const float target_aspect_ratio = (float)target_width / target_height;
  if (source_aspect_ratio >= target_aspect_ratio)
    source_width = std::round(source_height * target_aspect_ratio);
  else
    source_height = std::round(source_width / target_aspect_ratio);
  const u32 width_scale =
      source_width > 0 ? ((target_width + (source_width - 1)) / source_width) : 1;
  const u32 height_scale =
      source_height > 0 ? ((target_height + (source_height - 1)) / source_height) : 1;
  // Limit to the max to avoid creating textures larger than their max supported resolution.
  return std::min(std::max(width_scale, height_scale),
                  static_cast<u32>(Config::Get(Config::GFX_MAX_EFB_SCALE)));
}

void Presenter::SetSuggestedWindowSize(int width, int height)
{
  // While trying to guess the best window resolution, we can't allow it to use the
  // "AspectMode::Stretch" setting because that would self influence the output result,
  // given it would be based on the previous frame resolution
  const bool allow_stretch = false;
  const auto [out_width, out_height] = CalculateOutputDimensions(width, height, allow_stretch);

  // Track the last values of width/height to avoid sending a window resize event every frame.
  if (out_width == m_last_window_request_width && out_height == m_last_window_request_height)
    return;

  m_last_window_request_width = out_width;
  m_last_window_request_height = out_height;
  // Pass in the suggested window size. This might not always be acknowledged.
  Host_RequestRenderWindowSize(out_width, out_height);
}

// Crop to exact forced aspect ratios if enabled and not AspectMode::Stretch.
std::tuple<float, float> Presenter::ApplyStandardAspectCrop(float width, float height,
                                                            bool allow_stretch) const
{
  auto aspect_mode = g_ActiveConfig.aspect_mode;

  if (!allow_stretch && aspect_mode == AspectMode::Stretch)
    aspect_mode = AspectMode::Auto;

  if (!g_ActiveConfig.bCropToAspectRatio || aspect_mode == AspectMode::Stretch ||
      aspect_mode == AspectMode::Raw)
  {
    return {width, height};
  }

  // Force aspect ratios by cropping the image.
  const float current_aspect = width / height;
  float expected_aspect;
  switch (aspect_mode)
  {
  default:
  case AspectMode::Auto:
    expected_aspect = g_widescreen->IsGameWidescreen() ? (16.0f / 9.0f) : (4.0f / 3.0f);
    break;
  case AspectMode::ForceWide:
    expected_aspect = 16.0f / 9.0f;
    break;
  case AspectMode::ForceStandard:
    expected_aspect = 4.0f / 3.0f;
    break;
  // For the custom (relative) case, we want to crop from the native aspect ratio
  // to the specific target one, as they likely have a small difference
  case AspectMode::Custom:
  // There should be no cropping needed in the custom stretch case,
  // as output should always exactly match the target aspect ratio
  case AspectMode::CustomStretch:
    expected_aspect = g_ActiveConfig.GetCustomAspectRatio();
    break;
  }

  if (current_aspect > expected_aspect)
  {
    // keep height, crop width
    width = height * expected_aspect;
  }
  else
  {
    // keep width, crop height
    height = width / expected_aspect;
  }

  return {width, height};
}

void Presenter::UpdateDrawRectangle()
{
  const float draw_aspect_ratio = CalculateDrawAspectRatio();

  // Update aspect ratio hack values
  // Won't take effect until next frame
  // Don't know if there is a better place for this code so there isn't a 1 frame delay
  if (g_ActiveConfig.bWidescreenHack)
  {
    const auto& vi = Core::System::GetInstance().GetVideoInterface();
    float source_aspect_ratio = vi.GetAspectRatio();
    // If the game is meant to be in widescreen (or forced to),
    // scale the source aspect ratio to it.
    if (g_widescreen->IsGameWidescreen())
      source_aspect_ratio = SourceAspectRatioToWidescreen(source_aspect_ratio);

    const float adjust = source_aspect_ratio / draw_aspect_ratio;
    if (adjust > 1)
    {
      // Vert+
      g_Config.fAspectRatioHackW = 1;
      g_Config.fAspectRatioHackH = 1 / adjust;
    }
    else
    {
      // Hor+
      g_Config.fAspectRatioHackW = adjust;
      g_Config.fAspectRatioHackH = 1;
    }
  }
  else
  {
    // Hack is disabled.
    g_Config.fAspectRatioHackW = 1;
    g_Config.fAspectRatioHackH = 1;
  }

  // The rendering window size
  const float win_width = static_cast<float>(m_backbuffer_width);
  const float win_height = static_cast<float>(m_backbuffer_height);
  const float win_aspect_ratio = win_width / win_height;

  // FIXME: this breaks at very low widget sizes
  // Make ControllerInterface aware of the render window region actually being used
  // to adjust mouse cursor inputs.
  // This also doesn't handle the image cropping settings.
  g_controller_interface.SetAspectRatioAdjustment(draw_aspect_ratio / win_aspect_ratio);

  float draw_width = draw_aspect_ratio;
  float draw_height = 1;

  // Crop the picture to a standard aspect ratio. (if enabled)
  auto [crop_width, crop_height] = ApplyStandardAspectCrop(draw_width, draw_height);
  const float crop_aspect_ratio = crop_width / crop_height;

  // scale the picture to fit the rendering window
  if (win_aspect_ratio >= crop_aspect_ratio)
  {
    // the window is flatter than the picture
    draw_width *= win_height / crop_height;
    crop_width *= win_height / crop_height;
    draw_height *= win_height / crop_height;
    crop_height = win_height;
  }
  else
  {
    // the window is skinnier than the picture
    draw_width *= win_width / crop_width;
    draw_height *= win_width / crop_width;
    crop_height *= win_width / crop_width;
    crop_width = win_width;
  }

  int int_draw_width;
  int int_draw_height;

  if (g_ActiveConfig.aspect_mode != AspectMode::Raw || !m_xfb_entry)
  {
    // Find the best integer resolution: the closest aspect ratio with the least black bars.
    // This should have no influence if "AspectMode::Stretch" is active.
    const float updated_draw_aspect_ratio = draw_width / draw_height;
    const auto int_draw_res =
        FindClosestIntegerResolution(draw_width, draw_height, updated_draw_aspect_ratio);
    int_draw_width = std::get<0>(int_draw_res);
    int_draw_height = std::get<1>(int_draw_res);
    if (!g_ActiveConfig.bCropToAspectRatio)
    {
      if (g_ActiveConfig.aspect_mode != AspectMode::Stretch)
      {
        const MathUtil::Rectangle<int> rect = AdjustForCustomCrop(m_xfb_rect);
        TryToSnapToXFBSize(int_draw_width, int_draw_height, rect.GetWidth(), rect.GetHeight());
      }
      // We can't draw something bigger than the window, it will crop
      int_draw_width = std::min(int_draw_width, static_cast<int>(win_width));
      int_draw_height = std::min(int_draw_height, static_cast<int>(win_height));
    }
  }
  else
  {
    const MathUtil::Rectangle<int> rect = AdjustForCustomCrop(m_xfb_rect);
    int_draw_width = rect.GetWidth();
    int_draw_height = rect.GetHeight();
  }

  m_target_rectangle.left = static_cast<int>(std::round(win_width / 2.0 - int_draw_width / 2.0));
  m_target_rectangle.top = static_cast<int>(std::round(win_height / 2.0 - int_draw_height / 2.0));
  m_target_rectangle.right = m_target_rectangle.left + int_draw_width;
  m_target_rectangle.bottom = m_target_rectangle.top + int_draw_height;
}

std::tuple<float, float> Presenter::ScaleToDisplayAspectRatio(const int width, const int height,
                                                              bool allow_stretch) const
{
  // Scale either the width or height depending the content aspect ratio.
  // This way we preserve as much resolution as possible when scaling.
  float scaled_width = static_cast<float>(width);
  float scaled_height = static_cast<float>(height);
  const float draw_aspect = CalculateDrawAspectRatio(allow_stretch);
  if (scaled_width / scaled_height >= draw_aspect)
    scaled_height = scaled_width / draw_aspect;
  else
    scaled_width = scaled_height * draw_aspect;
  return std::make_tuple(scaled_width, scaled_height);
}

std::tuple<int, int> Presenter::CalculateOutputDimensions(int width, int height,
                                                          bool allow_stretch) const
{
  // Protect against zero width and height, a minimum of 1 will do
  width = std::max(width, 1);
  height = std::max(height, 1);

  auto [scaled_width, scaled_height] = ScaleToDisplayAspectRatio(width, height, allow_stretch);

  // Apply crop if enabled.
  std::tie(scaled_width, scaled_height) =
      ApplyStandardAspectCrop(scaled_width, scaled_height, allow_stretch);

  auto aspect_mode = g_ActiveConfig.aspect_mode;

  if (!allow_stretch && aspect_mode == AspectMode::Stretch)
    aspect_mode = AspectMode::Auto;

  if (!g_ActiveConfig.bCropToAspectRatio && aspect_mode != AspectMode::Stretch)
  {
    // Find the closest integer resolution for the aspect ratio,
    // this avoids a small black line from being drawn on one of the four edges
    const float draw_aspect_ratio = CalculateDrawAspectRatio(allow_stretch);
    auto [int_width, int_height] =
        FindClosestIntegerResolution(scaled_width, scaled_height, draw_aspect_ratio);
    if (aspect_mode != AspectMode::Raw)
    {
      TryToSnapToXFBSize(int_width, int_height, m_xfb_rect.GetWidth(), m_xfb_rect.GetHeight());
    }
    width = int_width;
    height = int_height;
  }
  else
  {
    width = static_cast<int>(std::ceil(scaled_width));
    height = static_cast<int>(std::ceil(scaled_height));
  }

  return std::make_tuple(width, height);
}

void Presenter::RenderXFBToScreen(const MathUtil::Rectangle<int>& target_rc,
                                  const AbstractTexture* source_texture,
                                  const MathUtil::Rectangle<int>& source_rc)
{
  if (g_ActiveConfig.stereo_mode == StereoMode::QuadBuffer &&
      g_backend_info.bUsesExplictQuadBuffering)
  {
    // Quad-buffered stereo is annoying on GL.
    g_gfx->SelectLeftBuffer();
    m_post_processor->BlitFromTexture(target_rc, source_rc, source_texture, 0);

    g_gfx->SelectRightBuffer();
    m_post_processor->BlitFromTexture(target_rc, source_rc, source_texture, 1);

    g_gfx->SelectMainBuffer();
  }
  else if (g_ActiveConfig.stereo_mode == StereoMode::FrameInterp)
  {
    // Layer 0 = real frame, layers 1..n-1 = in-between frames (clamped: an XFB
    // copied before a 120 <-> 240 switch has fewer layers).
    const int last_layer = static_cast<int>(source_texture->GetLayers()) - 1;
    m_post_processor->BlitFromTexture(target_rc, source_rc, source_texture,
                                      std::min(m_interp_present_layer, last_layer));
  }
  else if (g_ActiveConfig.stereo_mode == StereoMode::SideBySide ||
           g_ActiveConfig.stereo_mode == StereoMode::TopAndBottom)
  {
    const auto [left_rc, right_rc] = ConvertStereoRectangle(target_rc);

    m_post_processor->BlitFromTexture(left_rc, source_rc, source_texture, 0);
    m_post_processor->BlitFromTexture(right_rc, source_rc, source_texture, 1);
  }
  // Every other case will be treated the same (stereo or not).
  // If there's multiple source layers, they should all be copied.
  else
  {
    m_post_processor->BlitFromTexture(target_rc, source_rc, source_texture);
  }
}

void Presenter::Present(PresentInfo* present_info)
{
  m_present_count++;

  if (g_gfx->IsHeadless() || (!m_onscreen_ui && !m_xfb_entry))
    return;

  // Exclusive fullscreen follows the platform's request (display mode, focus).
  // The swap chain picks it up when the backbuffer is next bound.
  if (const bool exclusive = g_exclusive_fullscreen_wanted.load(std::memory_order_relaxed);
      exclusive != m_exclusive_fullscreen_applied)
  {
    m_exclusive_fullscreen_applied = exclusive;
    g_gfx->SetFullscreen(exclusive);
    std::fprintf(stderr, "[display] exclusive fullscreen %s\n", exclusive ? "on" : "off");
  }

  if (!g_gfx->SupportsUtilityDrawing())
  {
    // Video Software doesn't support drawing a UI or doing post-processing
    // So just show the XFB
    if (m_xfb_entry)
    {
      const MathUtil::Rectangle<int> rect = AdjustForCustomCrop(m_xfb_rect);
      g_gfx->ShowImage(m_xfb_entry->texture.get(), rect);

      // Update the window size based on the frame that was just rendered.
      // Due to depending on guest state, we need to call this every frame.
      SetSuggestedWindowSize(rect.GetWidth(), rect.GetHeight());
    }
    return;
  }

  // Since we use the common pipelines here and draw vertices if a batch is currently being
  // built by the vertex loader, we end up trampling over its pointer, as we share the buffer
  // with the loader, and it has not been unmapped yet. Force a pipeline flush to avoid this.
  g_vertex_manager->Flush();

  UpdateDrawRectangle();

  g_gfx->BeginUtilityDrawing();
  const bool backbuffer_bound = g_gfx->BindBackbuffer({{0.0f, 0.0f, 0.0f, 1.0f}});

  // Render the XFB to the screen.
  if (backbuffer_bound && m_xfb_entry)
  {
    // Adjust the source rectangle instead of using an oversized viewport to render the XFB.
    MathUtil::Rectangle<int> render_target_rc = GetTargetRectangle();
    MathUtil::Rectangle<int> render_source_rc = AdjustForCustomCrop(m_xfb_rect);
    AdjustRectanglesToFitBounds(&render_target_rc, &render_source_rc, m_backbuffer_width,
                                m_backbuffer_height);
    RenderXFBToScreen(render_target_rc, m_xfb_entry->texture.get(), render_source_rc);
  }

  // Frame interpolation presents one game frame several times. Building the
  // on-screen UI (statistics, graphs, OSD) is CPU work on the emulation thread,
  // so build it once per game frame and redraw the same draw data for the
  // in-between presents; the next UI frame starts after the real (layer 0)
  // present. Doing it per present cost battles their full speed at 240 FPS.
  const bool more_presents_this_frame =
      g_ActiveConfig.stereo_mode == StereoMode::FrameInterp && m_interp_present_layer != 0;
  if (m_onscreen_ui)
  {
    if (!m_ui_rendered)
    {
      m_onscreen_ui->Finalize();
      m_ui_rendered = true;
    }
    if (backbuffer_bound)
      m_onscreen_ui->DrawImGui();
  }

  // Present to the window system.
  {
    std::lock_guard<std::mutex> guard(m_swap_mutex);

    if (present_info != nullptr)
    {
      const auto present_time = GetUpdatedPresentationTime(present_info->intended_present_time);

      Core::System::GetInstance().GetCoreTiming().SleepUntil(present_time);

      // Perhaps in the future a more accurate time can be acquired from the various backends.
      present_info->actual_present_time = Clock::now();
      present_info->present_time_accuracy = PresentInfo::PresentTimeAccuracy::PresentInProgress;
    }

    const auto before_present = Clock::now();
    g_gfx->PresentBackbuffer();
    if (g_ActiveConfig.stereo_mode == StereoMode::FrameInterp)
    {
      m_interp_block_ms +=
          (DT_ms(Clock::now() - before_present).count() - m_interp_block_ms) * 0.1;
    }
    // MODERNGEKKO_INTERP_STATS=1: present pacing - count, interval spread and
    // time spent blocked inside the swap chain's Present (VSync waits).
    static const bool stats = [] {
      const char* v = std::getenv("MODERNGEKKO_INTERP_STATS");
      return v && v[0] == '1';
    }();
    if (stats)
    {
      static TimePoint last{}, window_start = Clock::now();
      static u32 count = 0, layer_counts[4] = {};
      static double min_ms = 1e9, max_ms = 0, blocked_ms = 0;
      const auto now = Clock::now();
      if (last != TimePoint{})
      {
        const double ms = std::chrono::duration<double, std::milli>(now - last).count();
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
      }
      blocked_ms += std::chrono::duration<double, std::milli>(now - before_present).count();
      last = now;
      ++count;
      ++layer_counts[std::clamp(m_interp_present_layer, 0, 3)];
      const double window = std::chrono::duration<double>(now - window_start).count();
      if (window >= 2.0)
      {
        std::fprintf(stderr,
                     "[present] %.1f/s interval min=%.2f max=%.2f ms blocked=%.2f ms/present "
                     "layers=%u/%u/%u/%u late=%u (%u/%u/%u/%u) resync=%u late_max=%.2f ms\n",
                     count / window, min_ms, max_ms, blocked_ms / count, layer_counts[0],
                     layer_counts[1], layer_counts[2], layer_counts[3], g_interp_pace_stats.late, g_interp_pace_stats.late_layer[0],
                     g_interp_pace_stats.late_layer[1], g_interp_pace_stats.late_layer[2],
                     g_interp_pace_stats.late_layer[3],
                     g_interp_pace_stats.resyncs, g_interp_pace_stats.late_max_ms);
        g_interp_pace_stats = {};
        window_start = now;
        count = 0;
        min_ms = 1e9;
        max_ms = blocked_ms = 0;
        std::fill(std::begin(layer_counts), std::end(layer_counts), 0u);
      }
    }
  }

  if (m_xfb_entry)
  {
    // Update the window size based on the frame that was just rendered.
    // Due to depending on guest state, we need to call this every frame.
    const MathUtil::Rectangle<int> rect = AdjustForCustomCrop(m_xfb_rect);
    SetSuggestedWindowSize(rect.GetWidth(), rect.GetHeight());
  }

  if (m_onscreen_ui && !more_presents_this_frame)
  {
    m_onscreen_ui->BeginImGuiFrame(m_backbuffer_width, m_backbuffer_height);
    m_ui_rendered = false;
  }

  g_gfx->EndUtilityDrawing();
}

TimePoint Presenter::GetUpdatedPresentationTime(TimePoint intended_presentation_time)
{
  const auto now = Clock::now();
  const auto arrival_offset = std::min(now - intended_presentation_time, DT{});

  if (!Config::Get(Config::MAIN_SMOOTH_EARLY_PRESENTATION))
  {
    m_presentation_time_offset = arrival_offset;

    // When SmoothEarlyPresentation is off and ImmediateXFB or RushFramePresentation are on,
    //  present as soon as possible as the goal is to achieve low input latency.
    if (g_ActiveConfig.bImmediateXFB || Config::Get(Config::MAIN_RUSH_FRAME_PRESENTATION))
      return now;

    return intended_presentation_time;
  }

  // Adjust slowly backward in time but quickly forward in time.
  // This keeps the pacing moderately smooth even if games produce regular sporadic bumps.
  // This was tuned to handle the terrible pacing in Brawl with "Immediate XFB".
  // Super Mario Galaxy 1 + 2 still perform poorly here in SingleCore mode.
  const auto adjustment_divisor = (arrival_offset < m_presentation_time_offset) ? 100 : 2;

  m_presentation_time_offset += (arrival_offset - m_presentation_time_offset) / adjustment_divisor;

  return intended_presentation_time + m_presentation_time_offset;
}

void Presenter::SetKeyMap(const DolphinKeyMap& key_map)
{
  if (m_onscreen_ui)
    m_onscreen_ui->SetKeyMap(key_map);
}

void Presenter::SetKey(u32 key, bool is_down, const char* chars)
{
  if (m_onscreen_ui)
    m_onscreen_ui->SetKey(key, is_down, chars);
}

void Presenter::SetMousePos(float x, float y)
{
  if (m_onscreen_ui)
    m_onscreen_ui->SetMousePos(x, y);
}

void Presenter::SetMousePress(u32 button_mask)
{
  if (m_onscreen_ui)
    m_onscreen_ui->SetMousePress(button_mask);
}

void Presenter::DoState(PointerWrap& p)
{
  p.Do(m_frame_count);
  p.Do(m_last_xfb_ticks);
  p.Do(m_last_xfb_addr);
  p.Do(m_last_xfb_width);
  p.Do(m_last_xfb_stride);
  p.Do(m_last_xfb_height);

  // If we're loading and there is a last XFB, re-display it.
  if (p.IsReadMode() && m_last_xfb_stride != 0)
  {
    // This technically counts as the end of the frame
    GetVideoEvents().after_frame_event.Trigger(Core::System::GetInstance());

    m_next_swap_estimated_ticks = m_last_xfb_ticks;
    m_next_swap_estimated_time = Clock::now();

    m_immediate_swap_happened_this_field.store(false, std::memory_order_relaxed);

    ImmediateSwap(m_last_xfb_addr, m_last_xfb_width, m_last_xfb_stride, m_last_xfb_height);
  }
}

}  // namespace VideoCommon
