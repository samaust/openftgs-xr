// SPDX-License-Identifier: Apache-2.0

//! The intents (§2.1 `ftgsxr_app::intent`, §1.7, §1.8): what the input adapters make of
//! the keyboard, the mouse, the controllers and the hands, and the one message every
//! feature reads.

use bevy::ecs::entity::Entity;
use bevy::math::{Vec2, Vec3};
use bevy::prelude::{Event, Message};
use ftgsxr_core::{EndRule, LoopMode, Pose};
use ftgsxr_render::RenderMode;

/// Which hand, or the desktop's mouse, an intent comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hand {
    /// The left controller or hand.
    Left,
    /// The right controller or hand.
    Right,
    /// The desktop's mouse.
    Mouse,
}

/// A feature a toggle switches on and off (§1.7, §1.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Feature {
    /// Passthrough (§1.9).
    Passthrough,
    /// The sound (§1.10).
    Sound,
    /// The HUD (§1.13).
    Hud,
    /// Occlusion by the environment's depth (§1.9).
    Occlusion,
    /// The main panel.
    Panel,
    /// The lighting adaptation (§1.9).
    Lighting,
}

/// A playback control (§1.6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlaybackCommand {
    /// Play.
    Play,
    /// Pause.
    Pause,
    /// Play when paused, pause when playing.
    Toggle,
    /// Seek to the clock time in seconds.
    Seek(f64),
    /// Step by whole frames, forward when positive.
    StepFrames(i32),
    /// Set the speed factor.
    Speed(f64),
    /// Set the loop mode.
    Loop(LoopMode),
    /// Set or clear the playback range in seconds.
    Range(Option<(f64, f64)>),
    /// Set an instance's time offset in seconds.
    Offset {
        /// The instance.
        instance: Entity,
        /// The offset in seconds.
        offset_s: f64,
    },
    /// Set an instance's end rule.
    EndRule {
        /// The instance.
        instance: Entity,
        /// The rule.
        rule: EndRule,
    },
}

/// An action on one instance (§1.5, §1.7, §1.8).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InstanceAction {
    /// Save its pose in the pose store.
    SavePose,
    /// Reset its pose to the stored or fitted one.
    ResetPose,
    /// Remove its pose from the pose store.
    ForgetPose,
    /// Fit it (§1.5).
    Fit,
    /// Snap it to the floor.
    SnapToFloor,
    /// Set its render mode.
    SetMode(RenderMode),
    /// Show or hide it.
    SetVisible(bool),
    /// Unload it.
    Unload,
    /// Select it.
    Select,
    /// Deselect it.
    Deselect,
}

/// The one message both input adapters produce and every feature consumes (§1.7, §1.8).
///
/// It derives both Bevy's `Message`, so the adapters write it with a `MessageWriter`
/// and the features read it with a `MessageReader` in the frame's order, and `Event`, as
/// §2.1 writes it, so it can also be triggered for observers.
// Q-166
#[derive(Message, Event, Clone, Copy, Debug, PartialEq)]
pub enum Intent {
    /// A pointing ray in stage space.
    Pointer {
        /// The hand.
        hand: Hand,
        /// The ray's origin.
        origin: Vec3,
        /// The ray's unit direction.
        dir: Vec3,
    },
    /// The select button went down or up.
    Select {
        /// The hand.
        hand: Hand,
        /// Down when true.
        pressed: bool,
    },
    /// A grab began at the hand's pose.
    GrabBegin {
        /// The hand.
        hand: Hand,
        /// The hand's pose.
        pose: Pose,
    },
    /// A held grab moved to the hand's pose.
    GrabMove {
        /// The hand.
        hand: Hand,
        /// The hand's pose.
        pose: Pose,
    },
    /// A grab ended.
    GrabEnd {
        /// The hand.
        hand: Hand,
    },
    /// Push or pull the grabbed instance along the ray.
    PushPull {
        /// The hand.
        hand: Hand,
        /// The stick's axis in \[-1, 1\].
        axis: f32,
    },
    /// Smooth locomotion: E12's stick values after `stick_response`.
    Locomote {
        /// The horizontal move.
        move_xz: Vec2,
        /// The turn.
        turn: f32,
    },
    /// Snap turns, positive to the right.
    SnapTurn {
        /// The steps.
        steps: i32,
    },
    /// Teleport to a floor point.
    Teleport {
        /// The target in stage space.
        target: Vec3,
    },
    /// Recenter the view.
    Recenter,
    /// §1.7's camera controls.
    Camera(CameraCommand),
    /// Choose the gizmo's handle set.
    Gizmo(GizmoMode),
    /// Nudge an instance.
    Nudge {
        /// The instance.
        instance: Entity,
        /// The move in metres.
        delta: Vec3,
    },
    /// Turn an instance about y.
    Turn {
        /// The instance.
        instance: Entity,
        /// The turn in degrees.
        yaw_deg: f32,
    },
    /// A playback control.
    Playback(PlaybackCommand),
    /// An action on one instance.
    Instance {
        /// The instance.
        instance: Entity,
        /// The action.
        action: InstanceAction,
    },
    /// Toggle a feature.
    Toggle(Feature),
    /// Toggle an overlay.
    Overlay(OverlayToggle),
    /// A UI command.
    Ui(UiCommand),
    /// Write a screenshot (§1.7).
    Screenshot,
    /// Set a volume.
    Volume {
        /// The channel.
        channel: VolumeChannel,
        /// The gain in \[0, 1\].
        value: f32,
    },
}

/// §1.7's camera controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraCommand {
    /// Move in the camera's horizontal frame (fly mode), per second.
    Move(Vec3),
    /// Look by the mouse's motion.
    Look(Vec2),
    /// Adjust the base speed by wheel steps.
    Speed(f32),
    /// Switch between fly and orbit.
    ToggleOrbit,
    /// Drag the orbit.
    Orbit(Vec2),
    /// Pan.
    Pan(Vec2),
    /// Zoom by wheel steps.
    Zoom(f32),
    /// Frame the selected instance.
    Frame,
    /// Reset the view.
    ResetView,
    /// Save the camera as bookmark 1 to 9.
    SaveBookmark(u8),
    /// Recall bookmark 1 to 9.
    RecallBookmark(u8),
}

/// The gizmo's handle set (§1.7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GizmoMode {
    /// Axis arrows.
    Translate,
    /// Rotation rings.
    Rotate,
    /// The uniform-scale cube.
    Scale,
}

/// An overlay (§1.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OverlayToggle {
    /// The instances' boxes.
    Boxes,
    /// The velocities.
    Velocities,
    /// The time tint.
    TimeTint,
}

/// A UI command (§1.7, Appendix D).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiCommand {
    /// Open Help.
    Help,
    /// Hide or show the whole UI.
    HideUi,
    /// Show or hide the HUD.
    Hud,
    /// Toggle fullscreen.
    Fullscreen,
    /// Open the file dialog.
    OpenFiles,
    /// Save the scene.
    SaveScene,
    /// Load a scene.
    LoadScene,
    /// Cycle the loop mode.
    CycleLoop,
}

/// A volume channel (§1.10).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VolumeChannel {
    /// The master gain.
    Master,
    /// The UI sounds.
    Ui,
    /// The soundtracks.
    Soundtrack,
}

/// Appendix D.1's Delete: a request to unload `.0` that `ui`'s confirmation dialog
/// turns into `Intent::Instance { action: Unload }` once confirmed; `desktop_input`
/// writes it, since D.1 unloads only after confirmation and the intent itself asks
/// nothing (PRD #275).
// Q-199
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UnloadRequest(
    #[allow(dead_code, reason = "read by ui's confirmation dialog (#286)")] pub(crate) Entity,
);
