// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr application: everything the desktop and Quest builds share (§1.5–§1.7,
//! §1.10–§1.13, §2.1). M-2 adds the headless commands (#117, #118); M-4 the intents, the
//! platform port and `AppPlugin` (#277).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod audio;
pub mod capture;
pub mod desktop_input;
mod desktop_log;
mod exit_wait;
pub mod governor;
pub mod headless;
pub mod intent;
pub mod library;
pub mod manipulate;
pub mod persist;
pub mod platform;
pub mod playback;
pub mod plugin;
pub mod rig;
pub mod ui;

// Q-67: `BenchReport` and `cmd_bench`.
pub use headless::{
    BenchReport, HeadlessError, cmd_bench, cmd_export, cmd_import, cmd_info, cmd_render,
    cmd_validate, exit_code,
};
// §2.1 `ftgsxr_app::desktop_input`; Q-197: `KeyMap`'s methods.
pub use desktop_input::{DesktopInputPlugin, KeyMap};
// §2.1 `ftgsxr_app::governor`; Q-153, Q-178: the knobs, the ladders and the pure step.
pub use governor::{DESKTOP_LADDER, Governor, GovernorCounters, HEADSET_LADDER, Knob, govern};
// §2.1 `ftgsxr_app::intent`.
pub use intent::{
    CameraCommand, Feature, GizmoMode, Hand, InstanceAction, Intent, OverlayToggle,
    PlaybackCommand, UiCommand, VolumeChannel,
};
// §2.1 `ftgsxr_app::library`; Q-173: `LibraryFailure`; Q-255: `ReplaceEntry`.
pub use library::{
    EntryState, Library, LibraryCommand, LibraryEntry, LibraryFailure, LoadedModel, ReplaceEntry,
};
// §2.1 `ftgsxr_app::playback`.
pub use playback::Playback;
// §2.1 `ftgsxr_app::audio`; Q-149, Q-187: the output port, its sinks and the stub.
pub use audio::{
    AudioOutput, AudioOutputRes, AudioState, PlayUiSound, SinkId, Soundtrack, StubAudioOutput,
    StubSink, UiSound,
};
// §2.1 `ftgsxr_app::manipulate`.
pub use manipulate::{Grabbed, Hovered, ManipulateSettings, Poses, Selected, pick_instance};
// §2.1 `ftgsxr_app::rig`.
pub use rig::{
    ContentGroup, Locomotion, PlayerRig, SnapTurnState, apply_locomotion, stick_response,
};
// §2.1 `ftgsxr_app::ui`; Q-159: `HudInstance`; Q-263: `PanelPointer`.
pub use ui::{
    HudData, HudInstance, PanelPointer, Toast, ToastLevel, UiSurface, WorldPanel, WristMenu,
    panel_hit,
};
// §2.1 `ftgsxr_app::capture`.
pub use capture::ScreenshotRequest;
// §2.1 `ftgsxr_app::persist` (M-3: `SettingsRes`, Q-127; #277: `SettingsPath`, Q-163).
// Q-224: `SettingsWarnings`.
pub use persist::{
    PersistCommand, SettingsPath, SettingsRes, SettingsWarnings, apply_scene, scene_from_world,
};
// §2.1 `ftgsxr_app::platform`; Q-266: `Headset`.
pub use platform::{
    CameraFrames, CameraThumb, Headset, Permission, PermissionHandle, PickHandle, PickKind,
    Platform, PlatformRes,
};
// §2.1 `ftgsxr_app::plugin`; Q-151: `init_desktop_log` and the `IoError` it returns.
pub use ftgsxr_io::IoError;
pub use plugin::{AppPlugin, init_desktop_log};
// Q-168: what `ftgsxr-desktop` names for `AppPlugin.initial` and `LinuxPlatform`, for
// callers that depend on the app alone (§2.1's `ftgsxr-desktop`), as Q-129.
pub use ftgsxr_io::{DocumentProvider, ModelRef, Url};
// Q-T458.2: what `ftgsxr_quest::QuestPlatform`'s `DocumentProvider` returns and classifies
// with (§2.1's `ftgsxr-quest` depends on the app alone), as Q-168.
pub use ftgsxr_io::{Entry, EntryKind, classify_name};
// Q-89: the tier `cmd_render` (and M-4's `AppPlugin`) takes, for callers that depend on
// the app alone (§2.1's `ftgsxr-desktop`).
pub use ftgsxr_render::QualityTier;
// Q-129: what the Quest's provisional harness (`ftgsxr_quest::main` before M-4's
// `AppPlugin`) takes, for callers that depend on the app alone (§2.1's `ftgsxr-quest`).
pub use ftgsxr_io::{AppPaths, ModelBytes, Settings, load_settings};
pub use ftgsxr_render::{SortMode, SplatLoad, SplatPlugin, SplatTime};
// Q-137: what the Quest's device test `dv2` reads of the adapter (§3.4, Dv-2).
pub use ftgsxr_render::{SplatCamera, SplatEvent, SplatStats};
// Q-146 (superseding Q-135's re-exports): R-4's checks on the device, for the
// Quest's Dv-3 runner (§2.1's `ftgsxr-quest` depends on the app alone).
pub use headless::sort_check;
