// Copyright 2019 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "DolphinNoGUI/Platform.h"
#include "pause_menu_host.hpp"

#include "Core/Config/MainSettings.h"
#include "Core/Config/ConfigManager.h"
#include "Core/Core.h"
#include "Core/SavestateLayout.h"
#include "Core/State.h"
#include "Core/System.h"

#include "Common/CommonPaths.h"
#include "Common/FileUtil.h"
#include "Common/StringUtil.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>
#include <windows.h>
#include <windowsx.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <dwmapi.h>
#include <thread>

#include "VideoCommon/Present.h"
#include "VideoCommon/VideoConfig.h"
#include "resource.h"

void Host_RequestGameReset();
bool Host_IsNavigationWidescreenEnabled();
void Host_ToggleNavigationWidescreen();
bool Host_IsNavigationOverlayVisible();
void Host_ToggleNavigationOverlay();
int Host_GetUltrawideEfbScale();
void Host_SetUltrawideEfbScale(int scale);
void Host_RequestNavigationMapFit();
bool Host_IsTextUpscaleEnabled();
void Host_ToggleTextUpscale();
bool Host_IsHighResolutionTexturesEnabled();
void Host_ToggleHighResolutionTextures();
bool Host_IsCommunityHdTexturePackEnabled();
void Host_ToggleCommunityHdTexturePack();
bool Host_IsTextureDumpingEnabled();
void Host_ToggleTextureDumping();
bool Host_IsInputOverlayEnabled();
void Host_ToggleInputOverlay();
bool Host_IsMinimapHighContrastEnabled();
void Host_ToggleMinimapHighContrast();
bool Host_IsFastForwardEnabled();
void Host_ToggleFastForwardEnabled();
void Host_SetFastForwardActive(bool active);
bool Host_IsSixtyFpsEnabled();
void Host_ToggleSixtyFps();
bool Host_IsAutosaveEnabled();
void Host_ToggleAutosave();

namespace
{
// Menu command ids. Load State entries are allocated a contiguous range,
// since the list is rebuilt from disk each time the menu opens.
constexpr UINT ID_SAVE_STATE = 41001;
constexpr UINT ID_PAUSE = 41002;
constexpr UINT ID_MUTE = 41003;
constexpr UINT ID_FULLSCREEN = 41004;
constexpr UINT ID_NAV_TOGGLE_MAP = 41005;
constexpr UINT ID_NAV_FIT_MAP = 41006;
constexpr UINT ID_NAV_WIDESCREEN = 41007;
constexpr UINT ID_NAV_RESTART = 41009;
constexpr UINT ID_NAV_ULTRAWIDE_SCALE_3X = 41013;
constexpr UINT ID_NAV_ULTRAWIDE_SCALE_4X = 41014;
constexpr UINT ID_NAV_ULTRAWIDE_SCALE_5X = 41015;
constexpr UINT ID_NAV_ULTRAWIDE_SCALE_6X = 41016;
constexpr UINT ID_NAV_TEXT_UPSCALE = 41017;
constexpr UINT ID_NAV_INPUT_OVERLAY = 41020;
constexpr UINT ID_NAV_MINIMAP_HIGH_CONTRAST = 41021;
constexpr UINT ID_NAV_FAST_FORWARD = 41022;
constexpr UINT ID_NAV_AUTOSAVE = 41023;
constexpr UINT ID_NAV_SIXTY_FPS = 41024;
constexpr UINT ID_NAV_COMMUNITY_HD_TEXTURE_PACK = 41025;
constexpr UINT ID_LOAD_STATE_FIRST = 41100;
constexpr UINT ID_LOAD_STATE_LAST = 41199;


class PlatformWin32 final : public Platform
{
public:
  ~PlatformWin32() override;

  bool Init() override;
  void SetTitle(const std::string& string) override;
  void MainLoop() override;
  void ToggleFullscreenFromMenu() override { ToggleFullscreen(); }
  void SetExclusiveFullscreen(bool exclusive) override
  {
    m_exclusive_fullscreen = exclusive;
    UpdateExclusiveFullscreen();
  }
  void RequestGraphicsSwitchReveal() override
  {
    if (m_hwnd && m_switch_pending)
      PostMessage(m_hwnd, WM_GRAPHICS_SWITCH_REVEAL, 0, 0);
  }

  WindowSystemInfo GetWindowSystemInfo() const override;

private:
  static constexpr TCHAR WINDOW_CLASS_NAME[] = _T("DolphinNoGUI");
  static constexpr UINT WM_GRAPHICS_SWITCH_REVEAL = WM_APP + 0x47;

  // Graphics API switch (MODERNGEKKO_SWITCH_EVENT): the window is created off
  // screen and moved into the previous session's place once rendering.
  void RevealAfterGraphicsSwitch();
  bool m_switch_pending = false;
  bool m_switch_fullscreen = false;
  bool m_app_active = true;
  int m_switch_x = 0, m_switch_y = 0, m_switch_width = 0, m_switch_height = 0;

  static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

  static bool RegisterRenderWindowClass();
  bool CreateRenderWindow();
  bool CreateMenus();
  void RefreshMenu(HMENU menu);
  void SaveStateToStatesDirectory();
  void ToggleFullscreen();
  void SetFullscreen(bool fullscreen);
  void UpdateExclusiveFullscreen();
  void UpdateWindowPosition();
  void ProcessEvents();

  HWND m_hwnd{};
  HMENU m_menu{};
  HMENU m_file_menu{};
  HMENU m_load_menu{};
  HMENU m_view_menu{};
  HMENU m_ultrawide_quality_menu{};
  // Parallel to the Load State menu entries, rebuilt whenever it opens.
  std::vector<std::string> m_load_state_paths;
  std::time_t m_last_save_time{};
  std::size_t m_save_sequence{};
  WINDOWPLACEMENT m_windowed_placement{sizeof(WINDOWPLACEMENT)};
  LONG_PTR m_windowed_style = WS_OVERLAPPEDWINDOW;
  LONG_PTR m_windowed_ex_style = WS_EX_CLIENTEDGE;
  u32 m_mouse_buttons = 0;

  int m_window_x = Config::Get(Config::MAIN_RENDER_WINDOW_XPOS);
  int m_window_y = Config::Get(Config::MAIN_RENDER_WINDOW_YPOS);
  int m_window_width = Config::Get(Config::MAIN_RENDER_WINDOW_WIDTH);
  int m_window_height = Config::Get(Config::MAIN_RENDER_WINDOW_HEIGHT);
};

PlatformWin32::~PlatformWin32()
{
  if (m_hwnd)
    DestroyWindow(m_hwnd);
}

bool PlatformWin32::RegisterRenderWindowClass()
{
  WNDCLASSEX wc = {};
  wc.cbSize = sizeof(WNDCLASSEX);
  wc.style = 0;
  wc.lpfnWndProc = WndProc;
  wc.cbClsExtra = 0;
  wc.cbWndExtra = 0;
  wc.hInstance = GetModuleHandle(nullptr);
  wc.hIcon = LoadIcon(nullptr, IDI_ICON1);
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
  wc.lpszMenuName = nullptr;
  wc.lpszClassName = WINDOW_CLASS_NAME;
  wc.hIconSm = LoadIcon(nullptr, IDI_ICON1);

  if (!RegisterClassEx(&wc))
  {
    MessageBox(nullptr, _T("Window registration failed."), _T("Error"), MB_ICONERROR | MB_OK);
    return false;
  }

  return true;
}

bool PlatformWin32::CreateRenderWindow()
{
  // MODERNGEKKO_START_RECT=x,y,w,h: set by the pause menu's Graphics API
  // switch so the relaunched session opens exactly over the previous window
  // (and, when fullscreen, on the same monitor).
  if (const char* rect = std::getenv("MODERNGEKKO_START_RECT"); rect && *rect)
  {
    int x, y, w, h;
    if (std::sscanf(rect, "%d,%d,%d,%d", &x, &y, &w, &h) == 4 && w > 0 && h > 0)
    {
      m_window_x = x;
      m_window_y = y;
      m_window_width = w;
      m_window_height = h;
    }
  }
  if (const char* event = std::getenv("MODERNGEKKO_SWITCH_EVENT"); event && *event)
  {
    // Keep the window off every monitor (but visible, so its swap chain
    // presents normally) until the new session is drawing.
    m_switch_pending = true;
    m_switch_x = m_window_x;
    m_switch_y = m_window_y;
    m_switch_width = m_window_width;
    m_switch_height = m_window_height;
    // Right of every monitor: CreateRenderWindow treats a negative x as
    // "default position", which would put the window on screen.
    m_window_x = GetSystemMetrics(SM_XVIRTUALSCREEN) + GetSystemMetrics(SM_CXVIRTUALSCREEN) + 64;
    m_window_y = (std::max)(GetSystemMetrics(SM_YVIRTUALSCREEN), 0);
  }
  m_hwnd = CreateWindowEx(WS_EX_CLIENTEDGE, WINDOW_CLASS_NAME, _T("Dolphin"), WS_OVERLAPPEDWINDOW,
                          m_window_x < 0 ? CW_USEDEFAULT : m_window_x,
                          m_window_y < 0 ? CW_USEDEFAULT : m_window_y, m_window_width,
                          m_window_height, nullptr, nullptr, GetModuleHandle(nullptr), this);
  if (!m_hwnd)
  {
    MessageBox(nullptr, _T("CreateWindowEx failed."), _T("Error"), MB_ICONERROR | MB_OK);
    return false;
  }

  ShowWindow(m_hwnd, SW_SHOW);
  UpdateWindow(m_hwnd);
  return true;
}

bool PlatformWin32::CreateMenus()
{
  m_menu = CreateMenu();
  m_file_menu = CreatePopupMenu();
  m_load_menu = CreatePopupMenu();
  m_view_menu = CreatePopupMenu();
  m_ultrawide_quality_menu = CreatePopupMenu();
  if (!m_menu || !m_file_menu || !m_load_menu || !m_view_menu ||
      !m_ultrawide_quality_menu)
    return false;

  AppendMenuW(m_file_menu, MF_STRING, ID_SAVE_STATE, L"&Save State\tF1");
  AppendMenuW(m_file_menu, MF_POPUP, reinterpret_cast<UINT_PTR>(m_load_menu), L"&Load State");
  AppendMenuW(m_file_menu, MF_SEPARATOR, 0, nullptr);
  AppendMenuW(m_file_menu, MF_STRING, ID_NAV_RESTART, L"&Restart Game");
  AppendMenuW(m_file_menu, MF_SEPARATOR, 0, nullptr);
  // Checked state is refreshed from the core when the menu opens, so it cannot
  // drift out of step with an emulation that was paused some other way.
  AppendMenuW(m_file_menu, MF_STRING, ID_PAUSE, L"&Pause");
  AppendMenuW(m_file_menu, MF_STRING, ID_NAV_AUTOSAVE, L"Rotating &Autosaves");

  AppendMenuW(m_view_menu, MF_STRING, ID_FULLSCREEN, L"&Fullscreen\tAlt+Enter");
  AppendMenuW(m_view_menu, MF_STRING, ID_MUTE, L"&Mute Audio");
  AppendMenuW(m_view_menu, MF_SEPARATOR, 0, nullptr);
  AppendMenuW(m_view_menu, MF_STRING, ID_NAV_TOGGLE_MAP, L"&Show Navigation Map");
  AppendMenuW(m_view_menu, MF_STRING, ID_NAV_FIT_MAP, L"&Fit Map Bounds");
  AppendMenuW(m_view_menu, MF_SEPARATOR, 0, nullptr);
  AppendMenuW(m_view_menu, MF_STRING, ID_NAV_WIDESCREEN, L"Native &Ultrawide");
  AppendMenuW(m_ultrawide_quality_menu, MF_STRING,
              ID_NAV_ULTRAWIDE_SCALE_3X, L"3x Performance");
  AppendMenuW(m_ultrawide_quality_menu, MF_STRING,
              ID_NAV_ULTRAWIDE_SCALE_4X, L"4x Balanced");
  AppendMenuW(m_ultrawide_quality_menu, MF_STRING,
              ID_NAV_ULTRAWIDE_SCALE_5X, L"5x Quality");
  AppendMenuW(m_ultrawide_quality_menu, MF_STRING,
              ID_NAV_ULTRAWIDE_SCALE_6X, L"6x Ultra");
  AppendMenuW(m_view_menu, MF_POPUP,
              reinterpret_cast<UINT_PTR>(m_ultrawide_quality_menu),
              L"Ultrawide Render &Quality");
  AppendMenuW(m_view_menu, MF_SEPARATOR, 0, nullptr);
  AppendMenuW(m_view_menu, MF_STRING, ID_NAV_TEXT_UPSCALE,
              L"High-Resolution &Text");
  AppendMenuW(m_view_menu, MF_STRING, ID_NAV_COMMUNITY_HD_TEXTURE_PACK,
              L"Community HD Texture &Pack");
  AppendMenuW(m_view_menu, MF_SEPARATOR, 0, nullptr);
  AppendMenuW(m_view_menu, MF_STRING, ID_NAV_INPUT_OVERLAY,
              L"Controller &Input Overlay");
  AppendMenuW(m_view_menu, MF_STRING, ID_NAV_MINIMAP_HIGH_CONTRAST,
              L"High-Contrast &Minimap");
  AppendMenuW(m_view_menu, MF_STRING, ID_NAV_SIXTY_FPS,
              L"60 &FPS Gameplay Patch (Relaunch Required)");
  AppendMenuW(m_view_menu, MF_STRING, ID_NAV_FAST_FORWARD,
              L"Hold Space to &Fast-Forward");
  AppendMenuW(m_menu, MF_POPUP, reinterpret_cast<UINT_PTR>(m_file_menu), L"&File");
  AppendMenuW(m_menu, MF_POPUP, reinterpret_cast<UINT_PTR>(m_view_menu), L"&View");
  return SetMenu(m_hwnd, m_menu) != FALSE;
}

// Rebuilt on open rather than cached: states are written by this process while
// the menu is closed, and by the launcher between sessions.
void PlatformWin32::RefreshMenu(const HMENU menu)
{
  if (menu == m_file_menu)
  {
    auto& system = Core::System::GetInstance();
    const bool paused = Core::GetState(system) == Core::State::Paused;
    CheckMenuItem(m_file_menu, ID_PAUSE, MF_BYCOMMAND | (paused ? MF_CHECKED : MF_UNCHECKED));
    CheckMenuItem(m_file_menu, ID_NAV_AUTOSAVE,
                  MF_BYCOMMAND |
                      (Host_IsAutosaveEnabled() ? MF_CHECKED : MF_UNCHECKED));
    return;
  }

  if (menu == m_view_menu)
  {
    CheckMenuItem(m_view_menu, ID_MUTE,
                  MF_BYCOMMAND |
                      (Config::Get(Config::MAIN_AUDIO_MUTED) ? MF_CHECKED : MF_UNCHECKED));
    CheckMenuItem(m_view_menu, ID_NAV_TOGGLE_MAP,
                  MF_BYCOMMAND |
                      (Host_IsNavigationOverlayVisible() ? MF_CHECKED :
                                                           MF_UNCHECKED));
    CheckMenuItem(m_view_menu, ID_NAV_WIDESCREEN,
                  MF_BYCOMMAND |
                      (Host_IsNavigationWidescreenEnabled() ? MF_CHECKED : MF_UNCHECKED));
    CheckMenuItem(m_view_menu, ID_FULLSCREEN,
                  MF_BYCOMMAND | (m_window_fullscreen ? MF_CHECKED : MF_UNCHECKED));
    CheckMenuItem(m_view_menu, ID_NAV_TEXT_UPSCALE,
                  MF_BYCOMMAND |
                      (Host_IsTextUpscaleEnabled() ? MF_CHECKED : MF_UNCHECKED));
    CheckMenuItem(m_view_menu, ID_NAV_COMMUNITY_HD_TEXTURE_PACK,
                  MF_BYCOMMAND |
                      (Host_IsCommunityHdTexturePackEnabled() ? MF_CHECKED : MF_UNCHECKED));
    CheckMenuItem(m_view_menu, ID_NAV_INPUT_OVERLAY,
                  MF_BYCOMMAND |
                      (Host_IsInputOverlayEnabled() ? MF_CHECKED : MF_UNCHECKED));
    CheckMenuItem(m_view_menu, ID_NAV_MINIMAP_HIGH_CONTRAST,
                  MF_BYCOMMAND |
                      (Host_IsMinimapHighContrastEnabled() ? MF_CHECKED : MF_UNCHECKED));
    CheckMenuItem(m_view_menu, ID_NAV_FAST_FORWARD,
                  MF_BYCOMMAND |
                      (Host_IsFastForwardEnabled() ? MF_CHECKED : MF_UNCHECKED));
    CheckMenuItem(m_view_menu, ID_NAV_SIXTY_FPS,
                  MF_BYCOMMAND |
                      (Host_IsSixtyFpsEnabled() ? MF_CHECKED : MF_UNCHECKED));
    return;
  }
  if (menu == m_ultrawide_quality_menu)
  {
    UINT selected = ID_NAV_ULTRAWIDE_SCALE_5X;
    switch (Host_GetUltrawideEfbScale())
    {
    case 3:
      selected = ID_NAV_ULTRAWIDE_SCALE_3X;
      break;
    case 4:
      selected = ID_NAV_ULTRAWIDE_SCALE_4X;
      break;
    case 5:
      selected = ID_NAV_ULTRAWIDE_SCALE_5X;
      break;
    case 6:
      selected = ID_NAV_ULTRAWIDE_SCALE_6X;
      break;
    default:
      break;
    }
    CheckMenuRadioItem(m_ultrawide_quality_menu,
                       ID_NAV_ULTRAWIDE_SCALE_3X,
                       ID_NAV_ULTRAWIDE_SCALE_6X, selected, MF_BYCOMMAND);
    return;
  }

  if (menu != m_load_menu)
    return;

  while (DeleteMenu(m_load_menu, 0, MF_BYPOSITION))
  {
  }
  m_load_state_paths.clear();

  // Location, extension and order all come from State::Layout, so this menu and
  // any frontend listing the same directory cannot disagree.
  const std::vector<std::filesystem::path> states =
      State::Layout::List(StringToPath(File::GetUserPath(D_STATESAVES_IDX)));

  if (states.empty())
  {
    AppendMenuW(m_load_menu, MF_STRING | MF_GRAYED, 0, L"(no savestates)");
    return;
  }

  const std::size_t limit = std::min<std::size_t>(
      states.size(), ID_LOAD_STATE_LAST - ID_LOAD_STATE_FIRST + 1);
  for (std::size_t i = 0; i < limit; ++i)
  {
    AppendMenuW(m_load_menu, MF_STRING, ID_LOAD_STATE_FIRST + i,
                states[i].filename().wstring().c_str());
    m_load_state_paths.push_back(PathToString(states[i]));
  }
}

void PlatformWin32::SaveStateToStatesDirectory()
{
  const std::string directory = File::GetUserPath(D_STATESAVES_IDX);
  File::CreateFullPath(directory);
  const std::time_t now = std::time(nullptr);
  if (now == m_last_save_time)
    ++m_save_sequence;
  else
  {
    m_last_save_time = now;
    m_save_sequence = 0;
  }

  const std::filesystem::path path =
      StringToPath(directory) /
      State::Layout::TimestampedName(now, m_save_sequence, State::Layout::MANUAL_PREFIX);
  State::SaveAs(Core::System::GetInstance(), PathToString(path));
}

void PlatformWin32::ToggleFullscreen()
{
  SetFullscreen(!m_window_fullscreen);
}

void PlatformWin32::SetFullscreen(bool fullscreen)
{
  if (!m_hwnd || fullscreen == m_window_fullscreen)
    return;

  if (fullscreen)
  {
    m_windowed_style = GetWindowLongPtr(m_hwnd, GWL_STYLE);
    m_windowed_ex_style = GetWindowLongPtr(m_hwnd, GWL_EXSTYLE);
    m_windowed_placement.length = sizeof(WINDOWPLACEMENT);
    GetWindowPlacement(m_hwnd, &m_windowed_placement);

    const HMONITOR monitor = MonitorFromWindow(m_hwnd, MONITOR_DEFAULTTONEAREST);
    MONITORINFO monitor_info{sizeof(MONITORINFO)};
    if (!GetMonitorInfo(monitor, &monitor_info))
      return;

    SetMenu(m_hwnd, nullptr);
    SetWindowLongPtr(m_hwnd, GWL_STYLE,
                     m_windowed_style & ~static_cast<LONG_PTR>(WS_OVERLAPPEDWINDOW));
    SetWindowLongPtr(m_hwnd, GWL_EXSTYLE,
                     m_windowed_ex_style & ~static_cast<LONG_PTR>(WS_EX_CLIENTEDGE));
    const RECT& bounds = monitor_info.rcMonitor;
    SetWindowPos(m_hwnd, HWND_TOP, bounds.left, bounds.top,
                 bounds.right - bounds.left, bounds.bottom - bounds.top,
                 SWP_FRAMECHANGED | SWP_NOOWNERZORDER);
    m_window_fullscreen = true;
  }
  else
  {
    SetWindowLongPtr(m_hwnd, GWL_STYLE, m_windowed_style);
    SetWindowLongPtr(m_hwnd, GWL_EXSTYLE, m_windowed_ex_style);
    SetMenu(m_hwnd, m_menu);
    SetWindowPlacement(m_hwnd, &m_windowed_placement);
    SetWindowPos(m_hwnd, nullptr, 0, 0, 0, 0,
                 SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE |
                     SWP_NOZORDER | SWP_NOOWNERZORDER);
    m_window_fullscreen = false;
  }

  Config::SetCurrent(Config::MAIN_FULLSCREEN, fullscreen);
  DrawMenuBar(m_hwnd);
  if (g_presenter)
    g_presenter->ResizeSurface();
  UpdateExclusiveFullscreen();
}

// Exclusive mode only while fullscreen, focused and not mid graphics switch, so
// Alt-Tab hands the display back and refocusing takes it again.
void PlatformWin32::UpdateExclusiveFullscreen()
{
  const bool wanted = m_exclusive_fullscreen && m_window_fullscreen && !m_switch_pending &&
                      m_hwnd && m_app_active;
  g_exclusive_fullscreen_wanted.store(wanted, std::memory_order_relaxed);
}

bool PlatformWin32::Init()
{
  // Keep the HWND client area, Vulkan swapchain, ImGui overlay, and mouse
  // coordinates on the same physical-pixel grid on scaled ultrawide displays.
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
  SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

  if (!RegisterRenderWindowClass() || !CreateRenderWindow() || !CreateMenus())
    return false;

  if (Config::Get(Config::MAIN_FULLSCREEN))
  {
    if (m_switch_pending)
      m_switch_fullscreen = true;  // applied when revealed
    else
      SetFullscreen(true);
  }

  if (Config::Get(Config::MAIN_DISABLE_SCREENSAVER))
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED);

  UpdateWindowPosition();
  return true;
}

void PlatformWin32::SetTitle(const std::string& string)
{
  SetWindowTextW(m_hwnd, UTF8ToWString(string).c_str());
}

void PlatformWin32::MainLoop()
{
  while (IsRunning())
  {
    UpdateRunningFlag();
    Core::HostDispatchJobs(Core::System::GetInstance());
    Host_PauseMenuTick();
    ProcessEvents();
    UpdateWindowPosition();

    // TODO: Is this sleep appropriate?
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

WindowSystemInfo PlatformWin32::GetWindowSystemInfo() const
{
  WindowSystemInfo wsi;
  wsi.type = WindowSystemType::Windows;
  wsi.render_window = reinterpret_cast<void*>(m_hwnd);
  wsi.render_surface = reinterpret_cast<void*>(m_hwnd);
  return wsi;
}

void PlatformWin32::RevealAfterGraphicsSwitch()
{
  if (!m_switch_pending)
    return;
  m_switch_pending = false;
  UpdateExclusiveFullscreen();
  SetWindowPos(m_hwnd, HWND_TOP, m_switch_x, m_switch_y, m_switch_width, m_switch_height,
               SWP_SHOWWINDOW);
  if (m_switch_fullscreen)
    SetFullscreen(true);
  SetForegroundWindow(m_hwnd);
  UpdateWindowPosition();
  // Tell the previous session it can hide its window and exit.
  if (const char* event = std::getenv("MODERNGEKKO_SWITCH_EVENT"); event && *event)
  {
    if (HANDLE ready = OpenEventA(EVENT_MODIFY_STATE, FALSE, event))
    {
      SetEvent(ready);
      CloseHandle(ready);
    }
  }
}

void PlatformWin32::UpdateWindowPosition()
{
  if (m_window_fullscreen)
    return;

  RECT rc = {};
  if (!GetWindowRect(m_hwnd, &rc))
    return;

  m_window_x = rc.left;
  m_window_y = rc.top;
  m_window_width = rc.right - rc.left;
  m_window_height = rc.bottom - rc.top;
}

void PlatformWin32::ProcessEvents()
{
  MSG msg;
  while (PeekMessage(&msg, m_hwnd, 0, 0, PM_REMOVE))
  {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}

LRESULT PlatformWin32::WndProc(const HWND hwnd, const UINT msg, const WPARAM wParam,
                               const LPARAM lParam)
{
  PlatformWin32* platform = reinterpret_cast<PlatformWin32*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
  switch (msg)
  {
  case WM_GRAPHICS_SWITCH_REVEAL:
    if (platform)
      platform->RevealAfterGraphicsSwitch();
    return 0;

  case WM_NCCREATE:
  {
    platform = static_cast<PlatformWin32*>(reinterpret_cast<CREATESTRUCT*>(lParam)->lpCreateParams);
    SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(platform));
    return DefWindowProc(hwnd, msg, wParam, lParam);
  }

  case WM_CREATE:
  {
    if (hwnd)
    {
      // Remove rounded corners from the render window on Windows 11
      constexpr DWM_WINDOW_CORNER_PREFERENCE corner_preference = DWMWCP_DONOTROUND;
      DwmSetWindowAttribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, &corner_preference,
                            sizeof(corner_preference));
    }
  }
  break;

  case WM_SIZE:
  {
    if (g_presenter)
      g_presenter->ResizeSurface();
  }
  break;

  case WM_KEYDOWN:
    Host_PauseMenuKey(static_cast<int>(wParam), true);
    if (wParam == VK_ESCAPE || Host_IsPauseMenuOpen())
      return 0;
    // Bit 30 is the previous key state. Ignore auto-repeat so holding F1 cannot
    // queue a new savestate every few milliseconds.
    if (wParam == VK_F1 && (static_cast<ULONG_PTR>(lParam) & (1u << 30)) == 0)
    {
      platform->SaveStateToStatesDirectory();
      return 0;
    }
    else if (wParam == VK_SPACE && Host_IsFastForwardEnabled())
    {
      Host_SetFastForwardActive(true);
      return 0;
    }
    else if (wParam == VK_F11)
    {
      platform->ToggleFullscreen();
      return 0;
    }
    break;

  case WM_KEYUP:
    Host_PauseMenuKey(static_cast<int>(wParam), false);
    if (wParam == VK_ESCAPE || Host_IsPauseMenuOpen())
      return 0;
    if (wParam == VK_SPACE)
    {
      Host_SetFastForwardActive(false);
      return 0;
    }
    break;

  case WM_ACTIVATEAPP:
    if (platform)
    {
      platform->m_app_active = wParam != FALSE;
      platform->UpdateExclusiveFullscreen();
    }
    break;

  case WM_KILLFOCUS:
    Host_PauseMenuFocusLost();
    platform->m_mouse_buttons = 0;
    if (g_presenter)
      g_presenter->SetMousePress(0);
    // Never leave emulation sped up if Space is released while another window has focus.
    Host_SetFastForwardActive(false);
    break;

  case WM_MOUSEMOVE:
    if (Host_IsPauseMenuOpen() && g_presenter)
      g_presenter->SetMousePos(static_cast<float>(GET_X_LPARAM(lParam)),
                               static_cast<float>(GET_Y_LPARAM(lParam)));
    break;
  case WM_LBUTTONDOWN:
  case WM_LBUTTONUP:
  case WM_RBUTTONDOWN:
  case WM_RBUTTONUP:
    if (platform && g_presenter &&
        (Host_IsPauseMenuOpen() || msg == WM_LBUTTONUP || msg == WM_RBUTTONUP))
    {
      const u32 bit = (msg == WM_LBUTTONDOWN || msg == WM_LBUTTONUP) ? 1u : 2u;
      if (msg == WM_LBUTTONDOWN || msg == WM_RBUTTONDOWN)
        platform->m_mouse_buttons |= bit;
      else
        platform->m_mouse_buttons &= ~bit;
      g_presenter->SetMousePress(platform->m_mouse_buttons);
      return 0;
    }
    break;
  case WM_SYSKEYDOWN:
    if (wParam == VK_RETURN && (GetKeyState(VK_MENU) & 0x8000) != 0)
    {
      platform->ToggleFullscreen();
      return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);

  case WM_INITMENUPOPUP:
    if (platform)
      platform->RefreshMenu(reinterpret_cast<HMENU>(wParam));
    break;

  case WM_COMMAND:
  {
    if (!platform)
      break;
    const UINT command = LOWORD(wParam);
    auto& system = Core::System::GetInstance();
    if (command == ID_SAVE_STATE)
    {
      platform->SaveStateToStatesDirectory();
    }
    else if (command == ID_PAUSE)
    {
      const bool paused = Core::GetState(system) == Core::State::Paused;
      Core::SetState(system, paused ? Core::State::Running : Core::State::Paused);
    }
    else if (command == ID_MUTE)
    {
      Config::SetCurrent(Config::MAIN_AUDIO_MUTED, !Config::Get(Config::MAIN_AUDIO_MUTED));
    }
    else if (command == ID_FULLSCREEN)
    {
      platform->ToggleFullscreen();
    }
    else if (command == ID_NAV_RESTART)
    {
      Host_RequestGameReset();
    }
    else if (command == ID_NAV_AUTOSAVE)
    {
      Host_ToggleAutosave();
    }
    else if (command == ID_NAV_TOGGLE_MAP)
    {
      Host_ToggleNavigationOverlay();
    }
    else if (command == ID_NAV_WIDESCREEN)
    {
      Host_ToggleNavigationWidescreen();
    }
    else if (command == ID_NAV_ULTRAWIDE_SCALE_3X)
      Host_SetUltrawideEfbScale(3);
    else if (command == ID_NAV_ULTRAWIDE_SCALE_4X)
      Host_SetUltrawideEfbScale(4);
    else if (command == ID_NAV_ULTRAWIDE_SCALE_5X)
      Host_SetUltrawideEfbScale(5);
    else if (command == ID_NAV_ULTRAWIDE_SCALE_6X)
      Host_SetUltrawideEfbScale(6);
    else if (command == ID_NAV_TEXT_UPSCALE)
      Host_ToggleTextUpscale();
    else if (command == ID_NAV_COMMUNITY_HD_TEXTURE_PACK)
      Host_ToggleCommunityHdTexturePack();
    else if (command == ID_NAV_INPUT_OVERLAY)
      Host_ToggleInputOverlay();
    else if (command == ID_NAV_MINIMAP_HIGH_CONTRAST)
      Host_ToggleMinimapHighContrast();
    else if (command == ID_NAV_FAST_FORWARD)
      Host_ToggleFastForwardEnabled();
    else if (command == ID_NAV_SIXTY_FPS)
      Host_ToggleSixtyFps();
    else if (command == ID_NAV_FIT_MAP)
    {
      Host_RequestNavigationMapFit();
    }
    else if (command >= ID_LOAD_STATE_FIRST && command <= ID_LOAD_STATE_LAST)
    {
      const std::size_t index = command - ID_LOAD_STATE_FIRST;
      if (index < platform->m_load_state_paths.size())
        State::LoadAs(system, platform->m_load_state_paths[index]);
    }
    break;
  }

  case WM_CLOSE:
    platform->RequestShutdown();
    break;

  default:
    return DefWindowProc(hwnd, msg, wParam, lParam);
  }

  return 0;
}
}  // namespace

std::unique_ptr<Platform> Platform::CreateWin32Platform()
{
  return std::make_unique<PlatformWin32>();
}
