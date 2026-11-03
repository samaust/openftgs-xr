# The interface in `ftgsxr-app`

How `ftgsxr_app::ui` meets the panels (§1.7 Panels, D-21, §1.8 User interface), the toasts (§1.13 Logs, §1.15), the HUD's data (§1.13 HUD) and the ray-to-pointer mapping (§1.8, I-11, U-32), with `ftgsxr_app::capture`'s screenshots (§1.7 Screenshots). The spec items are those and §2.1's `ftgsxr_app::ui` and `ftgsxr_app::capture`; this doc explains them and never overrides them. Built in M-4 (#286, #287 and the #286 follow-up).

## Dependency direction

`ui` comes before `audio`, `persist`, `governor` and `desktop_input` in §2.1's order, so it names none of their types (PRD #275's "Dependency direction"). Instead the later modules use ui's crate-private seams:

- **`ToastRequest`**: every module asks for a toast through it; ui's limiter turns it into the `Toast` message (Q-208). Library, playback, audio and persist failures arrive this way (Q-173, Q-217, Q-189, Q-201); the XR crate, outside the app crate, triggers a `Toast` as an observer event, which ui turns into a `ToastRequest` (Q-221).
- **`UiCue`**: what ui wants heard; `audio` maps it to its UI sounds.
- **`SceneLoadConfirmed`**: the confirmed scene load, which `persist` turns into `PersistCommand::LoadScene` (Q-200).
- **`DockHooks`**: later modules add their own sections to the dock's tabs (Q-218): persist's display, environment and settings sections, audio's volumes, governor's tier and switch, desktop_input's Help shortcuts.

## Structure

What runs without egui (the toasts and their limit, the confirmations, the UI commands, `HudData`) is added in every app; the drawing and the panel pointer only when `EguiPlugin` is, which `AppPlugin` adds only when the render app exists (PRD #275's spike check 1 and its fallback).

- **The screen.** `ui/dock.rs` draws the bar above the viewport, the dock on the left with its seven tabs, the toast line and the confirmation dialogs in one exclusive system, which collects what the user did and sends it after the frame. egui draws no text without a font: the desktop loads the first readable system font of a fixed list at startup (Q-211).
- **Toasts.** The rate limit keys on a toast's text: one toast of the same text a second, a later one within the second dropped; a toast stays on the line for a few seconds, the newest few shown (Q-208). The capacity overflow has its own warning text, naming the capacity only, so it is the same each frame and a per-frame overflow shows once a second (Q-209). An unavailable XR feature gives a toast only when it was explicitly asked for, at most once per session per extension (Q-161).
- **Confirmations.** Delete's `UnloadRequest` (Q-199) and `Intent::Ui(LoadScene)` (Q-200) open a dialog; only a confirmation sends the unload or the load.
- **HUD.** `HudData` holds Q-159's fields, filled each frame from `SplatStats`, the clock, the instances and the tier state.
- **The world panel.** `panel_hit` maps a ray to a pixel of a `WorldPanel` quad, front side only, the top-left corner pixel (0, 0) (Q-210); the headset's drawing of it is later work.
- **Training cameras and requests.** `ui/cameras.rs` offers in the Scene tab the cameras of a manifest beside a model, or of the one an HTTP index entry names, which `library` downloads with the model (#374): the list by split, "jump to camera", the frustums in the instance's frame. It also replays a `ftgs-render-request` file found in a directory source from the Library tab. A jump moves the fly camera through `rig`'s `CameraJump` and seeks the clock to the image's timestamp (E9, Q-219). Cameras follow openftgs's conventions (E6, §1.14 Frames).

## Screenshots (`capture`)

F12's `Intent::Screenshot` becomes a `ScreenshotRequest` for the desktop camera, else the lowest-order active camera, captured from that camera's own render target (Q-212, spike check 4's fallback). The UI is hidden for the captured frame (`UiState::hide_for_one_frame`). One screenshot is in flight at a time, and one Bevy never answers is given up with a warning toast (Q-213). The PNG is encoded with `image` as 8-bit RGB at the target's full size (Q-214), an iTXt chunk with the version, the models' identities and poses, the clock and the camera is inserted after IHDR (Q-150), and the file is written atomically under the screenshots directory, named by the UTC date and time (Q-215, §1.11, Q-14).

## Q-n rows relied on

`ui`: Q-159, Q-161, Q-166, Q-173, Q-193, Q-199, Q-200, Q-208, Q-209, Q-210, Q-211, Q-217, Q-218, Q-219, Q-221; with Q-64 and Q-117 (the error texts). `capture`: Q-150, Q-166, Q-180, Q-212, Q-213, Q-214, Q-215; with Q-14.
