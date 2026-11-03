// SPDX-License-Identifier: Apache-2.0

//! The clock resource and per-instance time (§2.1 `ftgsxr_app::playback`, §1.6, E9,
//! D-32): `Playback` holds the one clock T; each frame the playback intents are applied,
//! T_len follows the instances, the clock advances by the frame's Δt and every instance's
//! `SplatTime` is written from its `ModelTime`. Nothing here calls `audio` (PRD #275's
//! "Dependency direction"): the soundtrack follows `Playback` and `SplatTime`.

use bevy::app::{App, Plugin, Update};
use bevy::ecs::prelude::*;
use bevy::log::warn;
use bevy::time::Time;
use ftgsxr_core::{Clock, LoopMode, Model, PlayState};
use ftgsxr_render::SplatTime;

use crate::intent::{Intent, PlaybackCommand, UiCommand};
use crate::library::LoadedModel;

/// The playback clock T (§1.6, E9), one for every instance.
#[derive(Resource, Clone, Debug)]
pub struct Playback {
    /// The clock.
    pub clock: Clock,
}

impl Default for Playback {
    fn default() -> Self {
        Playback {
            clock: Clock::new(0.0),
        }
    }
}

/// The instance whose frame step `StepFrames` uses: set by the selection once it exists
/// (`manipulate`, after this module in §2.1's order); `None` falls back (Q-175).
#[derive(Resource, Default, Clone, Copy, Debug)]
pub(crate) struct StepTarget(pub(crate) Option<Entity>);

/// The loop mode last set on the clock, which `Clock` does not report: `CycleLoop` steps
/// from it, so the loop mode is set through `PlaybackCommand::Loop` (Q-176).
// Q-176
#[derive(Resource, Clone, Copy, Debug)]
pub(crate) struct LoopSetting(pub(crate) LoopMode);

impl Default for LoopSetting {
    fn default() -> Self {
        // Q-19: `Clock::new` starts in `Loop`.
        LoopSetting(LoopMode::Loop)
    }
}

/// §1.6's speed presets, in both directions, ascending.
#[allow(
    dead_code,
    reason = "read by desktop_input's speed keys (PRD #275 step 7)"
)]
const PRESETS: [f64; 12] = [
    -4.0, -2.0, -1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0,
];

/// The next of §1.6's speed presets above (`up`) or below `speed`, through zero from
/// ±0.1 to ∓0.1; `speed` itself at either end (Q-176).
// Q-176
#[allow(
    dead_code,
    reason = "read by desktop_input's speed keys (PRD #275 step 7)"
)]
pub(crate) fn speed_preset(speed: f64, up: bool) -> f64 {
    let next = if up {
        PRESETS.iter().copied().find(|&p| p > speed + 1e-9)
    } else {
        PRESETS.iter().rev().copied().find(|&p| p < speed - 1e-9)
    };
    next.unwrap_or(speed)
}

/// The loop mode after `m` in Appendix D.1's cycle: loop, ping-pong, once.
pub(crate) fn next_loop(m: LoopMode) -> LoopMode {
    match m {
        LoopMode::Loop => LoopMode::PingPong,
        LoopMode::PingPong => LoopMode::Once,
        LoopMode::Once => LoopMode::Loop,
    }
}

/// The playback systems, added by `AppPlugin`.
pub(crate) struct PlaybackPlugin;

impl Plugin for PlaybackPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Intent>()
            .init_resource::<Playback>()
            .init_resource::<StepTarget>()
            .init_resource::<LoopSetting>()
            .add_systems(Update, (run_intents, follow_length, advance).chain());
    }
}

/// Δ_frame of an instance (D-32). Q-174: the time centers are not kept after a load,
/// so the median branch is not reached: a checkpoint's Δt_f, else 1/30 s.
// Q-174
fn frame_step(m: &LoadedModel) -> f64 {
    let model = Model {
        position: Vec::new(),
        time_center: Vec::new(),
        duration: Vec::new(),
        velocity: Vec::new(),
        scale: Vec::new(),
        rotation: Vec::new(),
        opacity: Vec::new(),
        sh: Vec::new(),
        sh_degree: m.info.sh_degree,
        time: m.info.time.clone(),
        conventions: m.info.conventions.clone(),
        producer: m.info.producer.clone(),
        identity: [0; 32],
        kind: m.info.kind,
    };
    m.time.frame_step_s(&model)
}

/// The frame step `StepFrames` uses: the target's when it is a timed instance, else the
/// smallest over the timed instances, else D-32's 1/30 s (Q-175).
// Q-175
fn step_of(target: Option<Entity>, models: &Query<(Entity, &LoadedModel)>) -> f64 {
    let timed = |m: &LoadedModel| !m.time.span.is_single();
    if let Some((_, m)) = target.and_then(|e| models.get(e).ok())
        && timed(m)
    {
        return frame_step(m);
    }
    models
        .iter()
        .filter(|(_, m)| timed(m))
        .map(|(_, m)| frame_step(m))
        .reduce(f64::min)
        .unwrap_or(1.0 / 30.0)
}

/// Applies `Intent::Playback` and `UiCommand::CycleLoop` (§1.6, Appendix D.1).
fn run_intents(
    mut reader: MessageReader<Intent>,
    mut playback: ResMut<Playback>,
    target: Res<StepTarget>,
    mut looping: ResMut<LoopSetting>,
    mut models: Query<(Entity, &mut LoadedModel)>,
) {
    for intent in reader.read() {
        let clock = &mut playback.clock;
        match *intent {
            Intent::Ui(UiCommand::CycleLoop) => {
                looping.0 = next_loop(looping.0);
                clock.set_loop(looping.0);
            }
            Intent::Playback(c) => match c {
                PlaybackCommand::Play => clock.set_state(PlayState::Playing),
                PlaybackCommand::Pause => clock.set_state(PlayState::Paused),
                PlaybackCommand::Toggle => clock.set_state(match clock.state() {
                    PlayState::Playing => PlayState::Paused,
                    PlayState::Paused => PlayState::Playing,
                }),
                PlaybackCommand::Seek(t) => clock.seek(t),
                // §1.6: a frame step only while paused.
                PlaybackCommand::StepFrames(n) => {
                    if clock.state() == PlayState::Paused {
                        let step = step_of(target.0, &models.as_readonly());
                        let t = clock.time() + f64::from(n) * step;
                        clock.seek(t);
                    }
                }
                PlaybackCommand::Speed(s) => {
                    if let Err(e) = clock.set_speed(s) {
                        warn!("speed {s} refused: {e}");
                    }
                }
                PlaybackCommand::Loop(m) => {
                    looping.0 = m;
                    clock.set_loop(m);
                }
                PlaybackCommand::Range(r) => clock.set_range(r),
                PlaybackCommand::Offset { instance, offset_s } => {
                    // Q-177: a non-finite offset is refused.
                    if !offset_s.is_finite() {
                        warn!("offset {offset_s} refused");
                    } else if let Ok((_, mut m)) = models.get_mut(instance) {
                        m.time.offset_s = offset_s;
                    }
                }
                PlaybackCommand::EndRule { instance, rule } => {
                    if let Ok((_, mut m)) = models.get_mut(instance) {
                        m.time.end_rule = rule;
                    }
                }
            },
            _ => {}
        }
    }
}

/// T_len (E9) on every load, unload and offset change: the largest `length_s` over the
/// instances that are not single-timestamp, 0 when there is none.
fn follow_length(
    changed: Query<(), Changed<LoadedModel>>,
    mut removed: RemovedComponents<LoadedModel>,
    models: Query<&LoadedModel>,
    mut playback: ResMut<Playback>,
) {
    let removed = removed.read().count() > 0;
    if changed.is_empty() && !removed {
        return;
    }
    let length = models
        .iter()
        .filter(|m| !m.time.span.is_single())
        .map(|m| m.time.length_s())
        .fold(0.0, f64::max);
    if length != playback.clock.length() {
        playback.clock.set_length(length);
    }
}

/// Advances the clock by the frame's Δt and writes every instance's `SplatTime` (E9).
fn advance(
    time: Res<Time>,
    mut playback: ResMut<Playback>,
    mut models: Query<(Entity, &LoadedModel, Option<&mut SplatTime>)>,
    mut commands: Commands,
) {
    playback.clock.advance(time.delta_secs_f64());
    let t = playback.clock.time();
    for (e, m, current) in &mut models {
        let (tau, extrapolating) = m.time.tau_at(t);
        let new = SplatTime { tau, extrapolating };
        match current {
            Some(mut s) => {
                s.set_if_neq(new);
            }
            None => {
                // `try_insert`: an instance the library despawns in the same frame needs no time.
                commands.entity(e).try_insert(new);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_preset_steps_through_both_directions() {
        assert_eq!(speed_preset(1.0, true), 2.0);
        assert_eq!(speed_preset(1.0, false), 0.5);
        assert_eq!(speed_preset(0.1, false), -0.1);
        assert_eq!(speed_preset(-0.1, true), 0.1);
        assert_eq!(speed_preset(4.0, true), 4.0);
        assert_eq!(speed_preset(-4.0, false), -4.0);
        assert_eq!(speed_preset(1.5, true), 2.0);
        assert_eq!(speed_preset(1.5, false), 1.0);
    }

    #[test]
    fn next_loop_cycles() {
        assert_eq!(next_loop(LoopMode::Loop), LoopMode::PingPong);
        assert_eq!(next_loop(LoopMode::PingPong), LoopMode::Once);
        assert_eq!(next_loop(LoopMode::Once), LoopMode::Loop);
    }
}
