// Copyright 2023 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "DolphinNoGUI/Platform.h"
#include "pause_menu_host.hpp"

#include "Common/CommonPaths.h"
#include "Common/FileUtil.h"
#include "Common/MsgHandler.h"
#include "Common/StringUtil.h"
#include "Core/Config/MainSettings.h"
#include "Core/Core.h"
#include "Core/SavestateLayout.h"
#include "Core/State.h"
#include "Core/System.h"
#include "VideoCommon/EFBInterface.h"
#include "VideoCommon/Present.h"

#include <AppKit/AppKit.h>
#include <Carbon/Carbon.h>
#include <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

// The runtime's side of the window menus (the same set the Windows window uses).
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
bool Host_IsCommunityHdTexturePackEnabled();
void Host_ToggleCommunityHdTexturePack();
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
void Host_SaveCameraForState(const std::string& state_path);
void Host_RestoreCameraForState(const std::string& state_path);
void Host_OpenPlayersPanel();
void Host_OpenAudioMixer();
bool Host_IsGbaPlayerNumbersShown();
void Host_ToggleGbaPlayerNumbers();
bool Host_WantsMouse();

namespace
{
// Menu commands, as item tags.
enum MenuCommand : NSInteger
{
  kSaveState = 1,
  kRestart,
  kPause,
  kAutosave,
  kFullscreen,
  kNavigationMap,
  kFitMap,
  kUltrawide,
  kUltrawide3x,
  kUltrawide4x,
  kUltrawide5x,
  kUltrawide6x,
  kTextUpscale,
  kHdTexturePack,
  kInputOverlay,
  kMinimapContrast,
  kSixtyFps,
  kFastForward,
  kPlayersPanel,
  kGbaNumbers,
  kAudioMixer,
  kMute,
  kLoadStateFirst = 1000,
};

// The pause menu and its friends read Windows virtual-key codes.
int VirtualKey(unsigned short key_code)
{
  switch (key_code)
  {
  case kVK_Escape:
    return 0x1B;
  case kVK_Return:
  case kVK_ANSI_KeypadEnter:
    return 0x0D;
  case kVK_Delete:
    return 0x08;
  case kVK_Tab:
    return 0x09;
  case kVK_Space:
    return 0x20;
  case kVK_LeftArrow:
    return 0x25;
  case kVK_UpArrow:
    return 0x26;
  case kVK_RightArrow:
    return 0x27;
  case kVK_DownArrow:
    return 0x28;
  case kVK_Shift:
  case kVK_RightShift:
    return 0x10;
  case kVK_F1:
    return 0x70;
  case kVK_F11:
    return 0x7A;
  default:
    return -1;
  }
}

std::vector<std::string> s_load_state_paths;
std::time_t s_last_save_time{};
std::size_t s_save_sequence{};

void SaveStateToStatesDirectory()
{
  const std::string directory = File::GetUserPath(D_STATESAVES_IDX);
  File::CreateFullPath(directory);
  const std::time_t now = std::time(nullptr);
  if (now == s_last_save_time)
    ++s_save_sequence;
  else
  {
    s_last_save_time = now;
    s_save_sequence = 0;
  }
  const std::filesystem::path path =
      StringToPath(directory) /
      State::Layout::TimestampedName(now, s_save_sequence, State::Layout::MANUAL_PREFIX);
  State::SaveAs(Core::System::GetInstance(), PathToString(path));
  Host_SaveCameraForState(PathToString(path));
}
}  // namespace

static Platform* s_application_platform = nullptr;

@interface NSApplication (DolphinNoGUI)
@property Platform* platform;
- (void)shutdown;
@end

@implementation NSApplication (DolphinNoGUI)
- (Platform*)platform
{
  return s_application_platform;
}

- (void)setPlatform:(Platform*)platform
{
  s_application_platform = platform;
}

- (void)shutdown;
{
  [self platform]->Stop();
  [self stop:nil];
}
@end

// Carries out the window menu's commands and keeps its checkmarks and the Load
// State list current each time a menu opens.
@interface MenuController : NSObject <NSMenuDelegate>
@property(assign) NSWindow* window;
@property(strong) NSMenu* loadMenu;
- (void)command:(id)sender;
- (void)loadState:(id)sender;
@end

@implementation MenuController
- (void)command:(id)sender
{
  auto& system = Core::System::GetInstance();
  switch ([sender tag])
  {
  case kSaveState:
    SaveStateToStatesDirectory();
    break;
  case kRestart:
    Host_RequestGameReset();
    break;
  case kPause:
    Core::SetState(system, Core::GetState(system) == Core::State::Paused ? Core::State::Running :
                                                                            Core::State::Paused);
    break;
  case kAutosave:
    Host_ToggleAutosave();
    break;
  case kFullscreen:
    [self.window toggleFullScreen:nil];
    break;
  case kNavigationMap:
    Host_ToggleNavigationOverlay();
    break;
  case kFitMap:
    Host_RequestNavigationMapFit();
    break;
  case kUltrawide:
    Host_ToggleNavigationWidescreen();
    break;
  case kUltrawide3x:
  case kUltrawide4x:
  case kUltrawide5x:
  case kUltrawide6x:
    Host_SetUltrawideEfbScale(static_cast<int>([sender tag] - kUltrawide3x) + 3);
    break;
  case kTextUpscale:
    Host_ToggleTextUpscale();
    break;
  case kHdTexturePack:
    Host_ToggleCommunityHdTexturePack();
    break;
  case kInputOverlay:
    Host_ToggleInputOverlay();
    break;
  case kMinimapContrast:
    Host_ToggleMinimapHighContrast();
    break;
  case kSixtyFps:
    Host_ToggleSixtyFps();
    break;
  case kFastForward:
    Host_ToggleFastForwardEnabled();
    break;
  case kPlayersPanel:
    Host_OpenPlayersPanel();
    break;
  case kGbaNumbers:
    Host_ToggleGbaPlayerNumbers();
    break;
  case kAudioMixer:
    Host_OpenAudioMixer();
    break;
  case kMute:
    Config::SetCurrent(Config::MAIN_AUDIO_MUTED, !Config::Get(Config::MAIN_AUDIO_MUTED));
    break;
  default:
    break;
  }
}

- (void)loadState:(id)sender
{
  const NSInteger index = [sender tag] - kLoadStateFirst;
  if (index < 0 || static_cast<std::size_t>(index) >= s_load_state_paths.size())
    return;
  State::LoadAs(Core::System::GetInstance(), s_load_state_paths[index]);
  Host_RestoreCameraForState(s_load_state_paths[index]);
}

- (void)menuNeedsUpdate:(NSMenu*)menu
{
  if (menu == self.loadMenu)
  {
    // Rebuilt on open: states are written by this process and by the launcher.
    [menu removeAllItems];
    s_load_state_paths.clear();
    const std::vector<std::filesystem::path> states =
        State::Layout::List(StringToPath(File::GetUserPath(D_STATESAVES_IDX)));
    if (states.empty())
    {
      NSMenuItem* none = [[NSMenuItem alloc] initWithTitle:@"(no savestates)"
                                                    action:nil
                                             keyEquivalent:@""];
      [none setEnabled:NO];
      [menu addItem:none];
      return;
    }
    for (std::size_t i = 0; i < std::min<std::size_t>(states.size(), 100); ++i)
    {
      NSString* title = [NSString stringWithUTF8String:states[i].filename().string().c_str()];
      NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title
                                                    action:@selector(loadState:)
                                             keyEquivalent:@""];
      [item setTarget:self];
      [item setTag:kLoadStateFirst + static_cast<NSInteger>(i)];
      [menu addItem:item];
      s_load_state_paths.push_back(PathToString(states[i]));
    }
    return;
  }

  auto& system = Core::System::GetInstance();
  const int scale = Host_GetUltrawideEfbScale();
  for (NSMenuItem* item in [menu itemArray])
  {
    bool on = false;
    switch ([item tag])
    {
    case kPause:
      on = Core::GetState(system) == Core::State::Paused;
      break;
    case kAutosave:
      on = Host_IsAutosaveEnabled();
      break;
    case kFullscreen:
      on = ([self.window styleMask] & NSWindowStyleMaskFullScreen) != 0;
      break;
    case kNavigationMap:
      on = Host_IsNavigationOverlayVisible();
      break;
    case kUltrawide:
      on = Host_IsNavigationWidescreenEnabled();
      break;
    case kUltrawide3x:
    case kUltrawide4x:
    case kUltrawide5x:
    case kUltrawide6x:
      on = scale == static_cast<int>([item tag] - kUltrawide3x) + 3;
      break;
    case kTextUpscale:
      on = Host_IsTextUpscaleEnabled();
      break;
    case kHdTexturePack:
      on = Host_IsCommunityHdTexturePackEnabled();
      break;
    case kInputOverlay:
      on = Host_IsInputOverlayEnabled();
      break;
    case kMinimapContrast:
      on = Host_IsMinimapHighContrastEnabled();
      break;
    case kSixtyFps:
      on = Host_IsSixtyFpsEnabled();
      break;
    case kFastForward:
      on = Host_IsFastForwardEnabled();
      break;
    case kGbaNumbers:
      on = Host_IsGbaPlayerNumbersShown();
      break;
    case kMute:
      on = Config::Get(Config::MAIN_AUDIO_MUTED);
      break;
    default:
      continue;
    }
    [item setState:on ? NSControlStateValueOn : NSControlStateValueOff];
  }
}
@end

@interface AppDelegate : NSObject <NSApplicationDelegate>

@property(readonly) Platform* platform;

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender;
- (id)initWithPlatform:(Platform*)platform;
@end

@implementation AppDelegate
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender
{
  return YES;
}

- (id)initWithPlatform:(Platform*)platform
{
  self = [super init];
  if (self)
  {
    _platform = platform;
  }
  return self;
}
@end

@interface WindowDelegate : NSObject <NSWindowDelegate>
- (void)windowDidResize:(NSNotification*)notification;
- (void)windowDidResignKey:(NSNotification*)notification;
@end

@implementation WindowDelegate
- (void)windowDidResize:(NSNotification*)notification
{
  if (g_presenter)
    g_presenter->ResizeSurface();
}

- (void)windowDidResignKey:(NSNotification*)notification
{
  Host_PauseMenuFocusLost();
  if (g_presenter)
    g_presenter->SetMousePress(0);
  // Never leave emulation sped up if Space is let go in another window.
  Host_SetFastForwardActive(false);
}
@end

namespace
{
class PlatformMacOS : public Platform
{
public:
  ~PlatformMacOS() override;

  bool Init() override;
  void SetTitle(const std::string& title) override;
  void MainLoop() override;
  void SaveWindowGeometry() override;
  void ToggleFullscreenFromMenu() override { [m_window toggleFullScreen:nil]; }

  WindowSystemInfo GetWindowSystemInfo() const override;

private:
  void ProcessEvents();
  // True when the event was used here and must not reach AppKit.
  bool HandleKey(NSEvent* event);
  void HandleMouse(NSEvent* event);
  void UpdateWindowPosition();
  void SetupMenu();

  NSRect m_window_rect;
  NSWindow* m_window;
  NSMenu* menuBar;
  AppDelegate* m_app_delegate;
  WindowDelegate* m_window_delegate;
  MenuController* m_menu_controller;
  unsigned int m_mouse_buttons = 0;

  int m_window_x = Config::Get(Config::MAIN_RENDER_WINDOW_XPOS);
  int m_window_y = Config::Get(Config::MAIN_RENDER_WINDOW_YPOS);
  unsigned int m_window_width = Config::Get(Config::MAIN_RENDER_WINDOW_WIDTH);
  unsigned int m_window_height = Config::Get(Config::MAIN_RENDER_WINDOW_HEIGHT);
  bool m_window_fullscreen = Config::Get(Config::MAIN_FULLSCREEN);
};

PlatformMacOS::~PlatformMacOS()
{
  [m_window close];
}

bool PlatformMacOS::Init()
{
  [NSApplication sharedApplication];

  m_app_delegate = [[AppDelegate alloc] initWithPlatform:this];
  [NSApp setDelegate:m_app_delegate];

  [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
  [NSApp setPlatform:this];
  [NSApp finishLaunching];

  unsigned long styleMask = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                            NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable;

  m_window_rect = CGRectMake(m_window_x, m_window_y, m_window_width, m_window_height);
  m_window = [NSWindow alloc];
  m_window = [m_window initWithContentRect:m_window_rect
                                 styleMask:styleMask
                                   backing:NSBackingStoreBuffered
                                     defer:NO];
  if (m_window == nil)
  {
    NSLog(@"Window is %@\n", m_window);
    return false;
  }
  if (m_window_x == -1 || m_window_y == -1)
  {
    [m_window center];
  }
  m_window_delegate = [[WindowDelegate alloc] init];
  [m_window setDelegate:m_window_delegate];
  // The panels and GBA screens follow the mouse even with no button held.
  [m_window setAcceptsMouseMovedEvents:YES];

  NSNotificationCenter* c = [NSNotificationCenter defaultCenter];
  [c addObserver:NSApp
        selector:@selector(shutdown)
            name:NSWindowWillCloseNotification
          object:m_window];

  if (Config::Get(Config::MAIN_SHOW_CURSOR) == Config::ShowCursor::Never)
    [NSCursor hide];

  if (Config::Get(Config::MAIN_FULLSCREEN))
  {
    m_window_fullscreen = true;
    [m_window toggleFullScreen:m_window];
  }

  [m_window makeKeyAndOrderFront:NSApp];
  [m_window makeMainWindow];
  [NSApp activateIgnoringOtherApps:YES];
  [m_window setTitle:@"Dolphin-emu-nogui"];

  SetupMenu();

  return true;
}

void PlatformMacOS::SetTitle(const std::string& title)
{
  @autoreleasepool
  {
    NSWindow* window = m_window;
    NSString* str = [NSString stringWithUTF8String:title.c_str()];
    dispatch_async(dispatch_get_main_queue(), ^{
      [window setTitle:str];
    });
  }
}

void PlatformMacOS::MainLoop()
{
  while (IsRunning())
  {
    UpdateRunningFlag();
    Core::HostDispatchJobs(Core::System::GetInstance());
    Host_PauseMenuTick();
    ProcessEvents();
    UpdateWindowPosition();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

WindowSystemInfo PlatformMacOS::GetWindowSystemInfo() const
{
  @autoreleasepool
  {
    WindowSystemInfo wsi;
    wsi.type = WindowSystemType::MacOS;
    wsi.render_window = (void*)CFBridgingRetain([m_window contentView]);
    wsi.render_surface = wsi.render_window;
    return wsi;
  }
}

bool PlatformMacOS::HandleKey(NSEvent* event)
{
  if ([event window] != m_window)
    return false;
  const bool down = [event type] == NSEventTypeKeyDown;
  const bool command = ([event modifierFlags] & NSEventModifierFlagCommand) != 0;
  if (command)
    return false;  // menu shortcuts
  const int key = VirtualKey([event keyCode]);
  if (key < 0)
    return Host_IsPauseMenuOpen();
  Host_PauseMenuKey(key, down);
  if (key == 0x1B || Host_IsPauseMenuOpen())
    return true;
  if (down && ![event isARepeat] && key == 0x70)
  {
    SaveStateToStatesDirectory();
    return true;
  }
  if (key == 0x20 && Host_IsFastForwardEnabled())
  {
    Host_SetFastForwardActive(down);
    return true;
  }
  if (down && key == 0x7A)
  {
    [m_window toggleFullScreen:nil];
    return true;
  }
  return false;
}

void PlatformMacOS::HandleMouse(NSEvent* event)
{
  if ([event window] != m_window || !g_presenter)
    return;
  NSView* view = [m_window contentView];
  const NSPoint at = [view convertPoint:[event locationInWindow] fromView:nil];
  // Backing pixels, top-down: what the renderer's UI uses.
  const CGFloat scale = [m_window backingScaleFactor];
  g_presenter->SetMousePos(static_cast<float>(at.x * scale),
                           static_cast<float>(([view bounds].size.height - at.y) * scale));

  unsigned int bit = 0;
  bool down = false;
  switch ([event type])
  {
  case NSEventTypeLeftMouseDown:
    bit = 1, down = true;
    break;
  case NSEventTypeLeftMouseUp:
    bit = 1;
    break;
  case NSEventTypeRightMouseDown:
    bit = 2, down = true;
    break;
  case NSEventTypeRightMouseUp:
    bit = 2;
    break;
  default:
    return;
  }
  // A press goes to the menus and panels only when they want it; a release
  // always does, so no button is left held.
  if (down && !Host_IsPauseMenuOpen() && !Host_WantsMouse())
    return;
  if (down)
    m_mouse_buttons |= bit;
  else
    m_mouse_buttons &= ~bit;
  g_presenter->SetMousePress(m_mouse_buttons);
}

void PlatformMacOS::ProcessEvents()
{
  @autoreleasepool
  {
    // Drain what has arrived without waiting: the loop also runs the menus,
    // host jobs and the GBA screens, which must not stall behind an idle mouse.
    while (NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                               untilDate:[NSDate distantPast]
                                                  inMode:NSDefaultRunLoopMode
                                                 dequeue:YES])
    {
      switch ([event type])
      {
      case NSEventTypeKeyDown:
      case NSEventTypeKeyUp:
        if (HandleKey(event))
          continue;
        break;
      case NSEventTypeMouseMoved:
      case NSEventTypeLeftMouseDragged:
      case NSEventTypeRightMouseDragged:
      case NSEventTypeLeftMouseDown:
      case NSEventTypeLeftMouseUp:
      case NSEventTypeRightMouseDown:
      case NSEventTypeRightMouseUp:
        HandleMouse(event);
        break;
      default:
        break;
      }
      [NSApp sendEvent:event];
    }

    // Need to update if m_window becomes fullscreen
    m_window_fullscreen = [m_window styleMask] & NSWindowStyleMaskFullScreen;
    m_window_focus = [m_window isMainWindow];
  }
}

void PlatformMacOS::UpdateWindowPosition()
{
  if (m_window_fullscreen)
    return;

  NSRect win = [m_window contentRectForFrameRect:[m_window frame]];
  m_window_x = win.origin.x;
  m_window_y = win.origin.y;
  m_window_width = win.size.width;
  m_window_height = win.size.height;
}

void PlatformMacOS::SetupMenu()
{
  @autoreleasepool
  {
    m_menu_controller = [[MenuController alloc] init];
    m_menu_controller.window = m_window;
    MenuController* target = m_menu_controller;

    menuBar = [NSMenu new];
    auto submenu = [&](NSString* title) {
      NSMenu* menu = [[NSMenu alloc] initWithTitle:title];
      [menu setDelegate:target];
      [menu setAutoenablesItems:NO];
      NSMenuItem* holder = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
      [holder setSubmenu:menu];
      [menuBar addItem:holder];
      return menu;
    };
    auto add = [&](NSMenu* menu, NSString* title, MenuCommand tag, NSString* key = @"",
                   NSEventModifierFlags mask = NSEventModifierFlagCommand) {
      NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title
                                                    action:@selector(command:)
                                             keyEquivalent:key];
      [item setTarget:target];
      [item setTag:tag];
      if ([key length])
        [item setKeyEquivalentModifierMask:mask];
      [menu addItem:item];
      return item;
    };

    // Application menu (shown under the app's name).
    NSMenu* app = submenu(@"App");
    NSMenuItem* quit = [[NSMenuItem alloc] initWithTitle:@"Quit"
                                                  action:@selector(shutdown)
                                           keyEquivalent:@"q"];
    [app addItem:quit];

    NSMenu* file = submenu(@"File");
    add(file, @"Save State", kSaveState, @"s");
    NSMenuItem* load = [[NSMenuItem alloc] initWithTitle:@"Load State" action:nil keyEquivalent:@""];
    NSMenu* load_menu = [[NSMenu alloc] initWithTitle:@"Load State"];
    [load_menu setDelegate:target];
    [load setSubmenu:load_menu];
    [file addItem:load];
    m_menu_controller.loadMenu = load_menu;
    [file addItem:[NSMenuItem separatorItem]];
    add(file, @"Restart Game", kRestart);
    [file addItem:[NSMenuItem separatorItem]];
    add(file, @"Pause", kPause, @"p");
    add(file, @"Rotating Autosaves", kAutosave);

    NSMenu* view = submenu(@"View");
    add(view, @"Fullscreen", kFullscreen, @"f",
        NSEventModifierFlagCommand | NSEventModifierFlagControl);
    [view addItem:[NSMenuItem separatorItem]];
    add(view, @"Show Navigation Map", kNavigationMap);
    add(view, @"Fit Map Bounds", kFitMap);
    [view addItem:[NSMenuItem separatorItem]];
    add(view, @"Native Ultrawide", kUltrawide);
    NSMenuItem* quality = [[NSMenuItem alloc] initWithTitle:@"Ultrawide Render Quality"
                                                     action:nil
                                              keyEquivalent:@""];
    NSMenu* quality_menu = [[NSMenu alloc] initWithTitle:@"Ultrawide Render Quality"];
    [quality_menu setDelegate:target];
    add(quality_menu, @"3x Performance", kUltrawide3x);
    add(quality_menu, @"4x Balanced", kUltrawide4x);
    add(quality_menu, @"5x Quality", kUltrawide5x);
    add(quality_menu, @"6x Ultra", kUltrawide6x);
    [quality setSubmenu:quality_menu];
    [view addItem:quality];
    [view addItem:[NSMenuItem separatorItem]];
    add(view, @"High-Resolution Text", kTextUpscale);
    add(view, @"Community HD Texture Pack", kHdTexturePack);
    [view addItem:[NSMenuItem separatorItem]];
    add(view, @"Controller Input Overlay", kInputOverlay);
    add(view, @"High-Contrast Minimap", kMinimapContrast);
    add(view, @"60 FPS Gameplay Patch (Relaunch Required)", kSixtyFps);
    add(view, @"Hold Space to Fast-Forward", kFastForward);

    NSMenu* players = submenu(@"Players");
    add(players, @"Controllers and GBAs…", kPlayersPanel, @"k");
    [players addItem:[NSMenuItem separatorItem]];
    add(players, @"Show GBA Player Numbers", kGbaNumbers);

    NSMenu* audio = submenu(@"Audio");
    add(audio, @"Volume Mixer…", kAudioMixer, @"m",
        NSEventModifierFlagCommand | NSEventModifierFlagShift);
    [audio addItem:[NSMenuItem separatorItem]];
    add(audio, @"Mute Audio", kMute);

    [NSApp setMainMenu:menuBar];
  }
}

void PlatformMacOS::SaveWindowGeometry()
{
  if (Config::Get(Config::MAIN_RENDER_WINDOW_SAVE_ON_EXIT))
  {
    Config::SetBaseOrCurrent(Config::MAIN_RENDER_WINDOW_WIDTH, static_cast<int>(m_window_width));
    Config::SetBaseOrCurrent(Config::MAIN_RENDER_WINDOW_HEIGHT, static_cast<int>(m_window_height));
    Config::SetBaseOrCurrent(Config::MAIN_RENDER_WINDOW_XPOS, static_cast<int>(m_window_x));
    Config::SetBaseOrCurrent(Config::MAIN_RENDER_WINDOW_YPOS, static_cast<int>(m_window_y));
    Config::Save();
  }
}

}  // namespace

std::unique_ptr<Platform> Platform::CreateMacOSPlatform()
{
  return std::make_unique<PlatformMacOS>();
}
