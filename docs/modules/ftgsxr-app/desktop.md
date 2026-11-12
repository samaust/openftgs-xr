# The desktop player: `ftgsxr play`, `LinuxPlatform`, the log and the keyboard

How `ftgsxr-desktop`'s `play` command and `LinuxPlatform` (§1.7 Command line, §2.1 `ftgsxr_desktop`, §2.5 Desktop, D-36), the app crate's desktop log (§1.13 Logs, §1.11) and `ftgsxr_app::desktop_input` (§1.7, Appendix D.1) make the Linux player. The spec items are those, with §1.15 and §2.1's band table; this doc explains them and never overrides them. Built in M-4 (#277, #284, #289). The headless commands are `headless.md`'s.

## `play`

`ftgsxr play [MODEL|URL|SCENE]... [--settings FILE] [--tier TIER] [--log-level LEVEL] [--xr]`. The desktop crate depends on bevy for the window and the `App` (§2.1).

1. **Arguments.** An argument that parses as an `http` or `https` URL is a URL, a path ending in `.ftgsxr-scene.json` the scene (one at most, else a usage error), any other path a file (Q-157).
2. **Settings and log.** The settings file is read once, before the log; the level and the tier come from it unless `--log-level` or `--tier` override them for the session, unsaved. The log is installed, the file's warnings logged, and `SettingsPath` and `SettingsRes` are inserted by the binary, so `AppPlugin` keeps them (Q-170, Q-163). A log that cannot be installed is a warning, and the player runs on.
3. **The `App`.** `DefaultPlugins` with the window, `AppPlugin` with the tier, the initial models and the scene, and with `--xr` the XR session through `add_xr`, its eyes at the running tier's sort mode, so `--tier` holds for them (Q-120, Q-121, Q-399). With `--xr` and no OpenXR runtime, `play` prints `ftgsxr: E-XR-RUNTIME: <user text>` and exits 2 before running, which `ftgsxr_xr::runtime_unavailable` tells it (Q-162, Q-220); a runtime whose session fails later leaves the desktop running with a toast.

**The window's first frames.** While Bevy's pipelines compile, it presents the window's surface texture with nothing drawn to it. The vendored wgpu-core clears and submits such a texture before presenting it, so those frames leave the validation layer nothing to report (§1.13, R-16; D-52, `ftgsxr-xr/session.md`), which `present_undrawn`'s `gpu_` test pins without Bevy. When the window is resized or changes present mode, Bevy drops a surface texture it holds unpresented (wgpu discards it) and reconfigures the surface. That sequence leaves no message, so the vendored wgpu-core leaves the discard unchanged; `discard_undrawn`'s `gpu_` test guards it (D-52, #408).

**Resizing the window.** When the window system resizes the window, its surface goes out of date before Bevy's render world sees the new size. The vendored bevy\_render reconfigures the surface at its current extent, both when it finds it out of date and when the window's size or present mode changes, so the window swapchain is created at the window's size (§1.13, R-16; D-53, `ftgsxr-xr/session.md`). For a frame or two afterwards the camera and the passes still size themselves from the old size. No pass of `play` outruns the window swapchain meanwhile: egui and the splats render into the camera's main texture, and the upscaling pass that writes the window swapchain sets a scissor only for an explicit camera viewport, which `play` doesn't set. A camera viewport added later would need checking against that lag. A resize that lands while wgpu configures the surface may still give one swapchain extent message, the race D-53 accepts. `window_resize`'s two `gpu_` tests pin both behaviours, resizing a Bevy window through winit and reading the layer's own log through a child run (Q-113).

## `LinuxPlatform`

The platform port's Linux adapter: `LinuxPlatform::new` takes the XDG locations of §1.11 (`AppPaths::linux`) and is E-IO without a home directory (Q-169). Pickers are rfd's default dialogs on a helper thread, polled by `poll_pick`; a dialog that closes or cannot open answers an empty pick, and the Library tab's path field is the fallback (Q-160). Every permission is granted; there are no documents and no camera.

## The log

`init_desktop_log` installs a subscriber writing to standard error and to the player's own size-rotated `logs/ftgsxr.log` (Q-151). An event with a `per_frame` field is limited to one per second per call site; every other event passes (Q-167). Bevy's `LogPlugin` is left for the caller to disable.

## `desktop_input`

`DesktopInputPlugin`, added by `AppPlugin`, turns every Appendix D.1 row into its `Intent` in `PreUpdate`, and the mouse ray from the active camera through the cursor into `Intent::Pointer { hand: Mouse }` each frame. `KeyMap` holds D.1's bindings by snake_case action name (Q-152), overridden by `Settings.desktop.keys` and rebuilt when the settings change, with its chord spelling and warnings (Q-197). Double clicks, wheel steps, nudges and turns follow Q-198, in the units `rig`'s camera controller takes (Q-196). Keys and buttons egui wants stay with the UI. Delete asks for confirmation through `UnloadRequest` (Q-199). Quit is the window's close, which Bevy already turns into the app's exit. The Help tab's shortcut list is this module's dock section.

## Q-n rows relied on

`play` and `LinuxPlatform`: Q-157, Q-160, Q-162, Q-163, Q-168 (the re-exported `ModelRef`, `Url` and `DocumentProvider`), Q-169, Q-170, Q-220, D-52, D-53; with Q-120, Q-121 and Q-14. The log: Q-151, Q-167. `desktop_input`: Q-152, Q-180, Q-196, Q-197, Q-198, Q-199; with Q-50.
