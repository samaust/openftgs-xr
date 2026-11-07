# Settings, scenes and poses in `ftgsxr-app`

How `ftgsxr_app::persist` meets §1.11's settings and scene files and §1.5's poses: `SettingsPath`, `SettingsRes`, `PersistCommand`, `scene_from_world` and `apply_scene`. The spec items are §1.5 Poses and Scene files, §1.11, D-11, D-23 and §2.1 `ftgsxr_app::persist`; this doc explains them and never overrides them. Built in M-4 (#277's minimal load, #285); the environment mode's gate, the Environment tab's XR controls and the scene anchor's handshake in M-6 (#567). The M-6 spec items are §1.9, §1.4 Backgrounds, §1.8 Locomotion, §1.11 `[environment]` and Scene files' `anchor`, and §1.15 E-XR-PERMISSION.

## Settings

`SettingsRes` is the source of truth (PRD #275). When it is absent at `AppPlugin`'s build, the plugin reads it from `SettingsPath`, else the platform's config directory; a caller that inserts it first (the `play` binary, the Quest's harness) keeps its own (Q-163, Q-170). A private `PersistPlugin`, after `AudioPlugin`, maps its tables into the resources the features read whenever it changes (the audio gains, locomotion, the clock's speed and loop, the display settings of each new instance), and back when the user changes one; a change marks it dirty, and a dirty file is saved at most once a second, and always on `SaveSettings` and at exit (Q-207). The load's warnings and errors are E-SETTINGS failures; a file that cannot be read is set aside.

`[xr] keep_upright` and `ManipulateSettings.keep_upright` are kept equal by a small system of their own beside the two-way mapping, with the same rule: the settings' value goes into the world at the first frame and when it changes, and a value changed in the world goes back into the settings (Q-330).

`persist/panels.rs` draws the Display, Environment and Settings tabs' sections through `ui`'s `DockHooks`, each editing `SettingsRes` (Q-218; the Environment tab's XR controls below); `[audio]` is `audio`'s section and the governor `governor`'s. The tier pickers (the Display tab's showing the running tier, the Settings tab's the saved one) write `[general] tier` and a crate-private `TierPick` that `governor` switches to (Q-240).

The dock's layout is saved too (§1.7 Panels, spec revision 24, #428): at startup, before the dock's first draw, `[desktop] dock_width` and `dock_open` become `ui`'s dock state, and Help's first run opens the dock after them (Q-226); while the UI is on the window, a drag of the dock's handle (in whole points, a change under half a point being none) or a collapse goes back into `[desktop]`. A window too narrow for the saved width, or a larger UI scale, clamps only the drawing, and the saved width is drawn again once the window is wide enough (Q-252, owner, #441); the Settings tab does not draw the two keys (Q-253).

## The environment mode and its controls

`[environment] mode` is the wanted environment mode and the one value `ftgsxr-app` reads (Q-335). `persist` counts it as passthrough only under the `Headset` marker, so the desktop and `play --xr` stay in VR mode whatever the setting says (§1.4 Backgrounds: "the desktop is always in VR mode"). From that gated value it maps `Locomotion.enabled`, off in passthrough mode (D-17, Q-195), and `DesktopView`'s passthrough flag, which hides `rig`'s floor grid there (Q-228): the grid is spawned with or without a window and shown only in VR mode with `[display] grid` on. A marker inserted after the first frame maps the mode again. The effective mode, which also needs the runtime's passthrough, is the XR crate's `Environment.mode` (`ftgsxr-xr/passthrough.md`); `persist` never reads it.

`Toggle(Feature::Occlusion)` flips `[environment] occlusion`, as the other toggles flip theirs; `Toggle(Feature::Passthrough)` is the XR crate's (Q-335, Q-336).

The Environment tab draws the VR-mode controls, background and grid, everywhere, and the XR controls only for the features `XrFeatures` reports (Q-336, §1.9 "On the desktop the Environment tab shows the VR-mode controls only"): with passthrough, the mode switch, which writes `Toggle(Feature::Passthrough)`, the three style sliders and the contact-shadows switch; with occlusion, its switch, soft or hard, δ's slider over §1.9's range and hand removal, with the note "off in the <tier> tier" while the running tier's `occlusion` flag turns it off (Q-338). Under a `Denied` feature the line says the permission was denied and "Ask again" writes its toggle again (§1.15 E-XR-PERMISSION); for occlusion, whose denial turns the setting off, that toggle turns it on and asks once more (Q-355). The other controls edit the settings directly.

## Poses

`Poses` is read from `<config>/poses.json` at `AppPlugin`'s build when absent. `manipulate` changes it in memory; `persist` forwards `SavePose` and `ForgetPose` to it as intents and writes the file atomically whenever the store's content differs from what was last written (Q-205).

## Scenes

A relative scene path names a file under `AppPaths.scenes`; `Intent::Ui(SaveScene)` saves under a UTC date-and-time name, and `Intent::Ui(LoadScene)` is `ui`'s to confirm first (Q-200). Model locations are relative to `AppPaths.models` by Q-3's rule.

- **`scene_from_world`** writes the instances, their poses, visibility (`ShowOnUpload` before the upload, Q-202), modes and time mappings, the clock with its range (Q-204), the rig's pose, the desktop camera and its bookmarks; each instance's `lighting` is its Scene-tab manual values, null at the identity, and a loaded scene's `lighting` is applied as its instance appears (Q-206, Q-250); `anchor` is `SceneAnchor::Ready`'s UUID, null otherwise (Q-343).
- **`LoadScene`** resolves the locations, unloads every instance and calls `apply_scene`. A local model whose file is missing becomes a library entry in its missing state with an E-SCENE-MISSING toast (Q-203). A model whose source now holds a file of another identity is refused once read and becomes a missing entry too, with E-SCENE-MISSING, and the clock does not wait for it (Q-245). A missing model's attributes are kept for a later load of its identity (Q-225), or for the file a `ReplaceEntry` puts in its place, whatever its identity: that instance carries the replaced identity in a crate-private `Replaces`, and the next save writes its own source and identity (§1.5 Library, Q-255, Q-258). A hidden instance stays hidden through its upload (Q-202). The clock's T and range are applied one frame after every instance appeared (Q-204).

## The scene anchor

`SceneAnchor` is the public resource at which `persist` and the XR crate's `anchors` meet (§1.9 Floor, shadows and anchors, Q-343); the XR side is `ftgsxr-xr/anchors.md`'s. Its states are `None`, `Creating`, `Ready(uuid)`, `Locating(uuid)` and `Lost(uuid)`.

- **The save.** A `SaveScene` under the `Headset` marker with `[environment] mode` passthrough sets `Creating` and defers the write, keeping §1.9's order, anchor first, file second, without blocking the frame loop; elsewhere the file is written at once. The waiting save is written with its anchor once the XR crate answers `Ready(uuid)`. Any other answer, `None` when `NoAnchors`' `create` errs say, or Q-343's wait in `Time<Real>` running out, the state then set to `None`, ends the wait as a failure: the file is written with a null anchor and a warning toast says the anchor was not created (Q-350). A second save while one waits replaces the waiting path. A headset without the extensions therefore never waits.
- **The load.** On the headset, loading a file with an anchor sets `Locating(uuid)`, and one without sets `None`; elsewhere the anchor is ignored and the file's rig-relative pose applies. `Lost(uuid)` raises §1.9's warning toast once per change to `Lost`, the group staying as the file left it (Q-343, Q-350).

## Failures

A crate-private `PersistFailure { code, text }` carries the settings' and the pose file's failures (raised in the first frame) and the scene's (as they occur); `persist` turns them into `ui`'s `ToastRequest` (Q-201). Its systems run before the library's commands (`LibraryCommands`).

## Q-n rows relied on

Q-163, Q-166, Q-170, Q-173, Q-185, Q-195, Q-228, Q-335, Q-336, Q-343, Q-350, Q-355, Q-188, Q-200, Q-201, Q-202, Q-203, Q-204, Q-205, Q-206, Q-207, Q-218, Q-225, Q-245, Q-252, Q-253, Q-255, Q-258, Q-330; with Q-3, Q-49, Q-50, Q-51 and Q-54 from M-1.
