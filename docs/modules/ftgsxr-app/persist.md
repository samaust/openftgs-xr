# Settings, scenes and poses in `ftgsxr-app`

How `ftgsxr_app::persist` meets §1.11's settings and scene files and §1.5's poses: `SettingsPath`, `SettingsRes`, `PersistCommand`, `scene_from_world` and `apply_scene`. The spec items are §1.5 Poses and Scene files, §1.11, D-11, D-23 and §2.1 `ftgsxr_app::persist`; this doc explains them and never overrides them. Built in M-4 (#277's minimal load, #285).

## Settings

`SettingsRes` is the source of truth (PRD #275). When it is absent at `AppPlugin`'s build, the plugin reads it from `SettingsPath`, else the platform's config directory; a caller that inserts it first (the `play` binary, the Quest's harness) keeps its own (Q-163, Q-170). A private `PersistPlugin`, after `AudioPlugin`, maps its tables into the resources the features read whenever it changes (the audio gains, locomotion, the clock's speed and loop, the display settings of each new instance), and back when the user changes one; a change marks it dirty, and a dirty file is saved at most once a second, and always on `SaveSettings` and at exit (Q-207). The load's warnings and errors are E-SETTINGS failures; a file that cannot be read is set aside.

`persist/panels.rs` draws the Display, Environment and Settings tabs' sections through `ui`'s `DockHooks`, each editing `SettingsRes` (Q-218); `[audio]` is `audio`'s section and the governor `governor`'s. The tier pickers (the Display tab's showing the running tier, the Settings tab's the saved one) write `[general] tier` and a crate-private `TierPick` that `governor` switches to (Q-240).

## Poses

`Poses` is read from `<config>/poses.json` at `AppPlugin`'s build when absent. `manipulate` changes it in memory; `persist` forwards `SavePose` and `ForgetPose` to it as intents and writes the file atomically whenever the store's content differs from what was last written (Q-205).

## Scenes

A relative scene path names a file under `AppPaths.scenes`; `Intent::Ui(SaveScene)` saves under a UTC date-and-time name, and `Intent::Ui(LoadScene)` is `ui`'s to confirm first (Q-200). Model locations are relative to `AppPaths.models` by Q-3's rule.

- **`scene_from_world`** writes the instances, their poses, visibility (`ShowOnUpload` before the upload, Q-202), modes and time mappings, the clock with its range (Q-204), the rig's pose, the desktop camera and its bookmarks; lighting and the anchor are written null until M-5 and M-9 hold them (Q-206).
- **`LoadScene`** resolves the locations, unloads every instance and calls `apply_scene`. A local model whose file is missing becomes a library entry in its missing state with an E-SCENE-MISSING toast (Q-203). A model whose source now holds a file of another identity is refused once read and becomes a missing entry too, with E-SCENE-MISSING, and the clock does not wait for it (Q-245). A hidden instance stays hidden through its upload (Q-202). The clock's T and range are applied one frame after every instance appeared (Q-204).

## Failures

A crate-private `PersistFailure { code, text }` carries the settings' and the pose file's failures (raised in the first frame) and the scene's (as they occur); `persist` turns them into `ui`'s `ToastRequest` (Q-201). Its systems run before the library's commands (`LibraryCommands`).

## Q-n rows relied on

Q-163, Q-166, Q-170, Q-173, Q-185, Q-188, Q-200, Q-201, Q-202, Q-203, Q-204, Q-205, Q-206, Q-207, Q-218, Q-245; with Q-3, Q-49, Q-50, Q-51 and Q-54 from M-1.
