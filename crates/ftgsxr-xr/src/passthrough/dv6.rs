// SPDX-License-Identifier: Apache-2.0

//! Dv-6 on the device (§3.4, §1.9, D-14), as Q-346 reads it without a person. It lives
//! here, not in `ftgsxr-quest`, because it reads `layer`'s crate-private record and the
//! render world's layer list; the quest runner starts it by spawning an entity named
//! [`TRIGGER`] (Q-308), and it prints the verdict line `ftgsxr-test: dv6 PASS|FAIL
//! <detail>` itself (§3.4, Q-122).
//!
//! Once the session runs and the layer exists, the test flips `[environment] mode` to
//! passthrough and back, twice (the second time with a brightness of 0.2, so a style
//! change is sent). It passes when, in passthrough mode, the render world's layer list
//! holds `layer`'s provider first of two and the layer is resumed, in VR mode the
//! projection layer alone and the layer paused, each held for [`HOLD`] with no frame
//! breaking it, the session's blend mode OPAQUE in every frame, and every pause, resume
//! and style call accepted (PRD #565's spike check 3 is the host driver's scan of the
//! log for compositor errors over the same switches).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::app::{App, PostUpdate};
use bevy::log::info;
use bevy::prelude::{
    Commands, IntoScheduleConfigs, Name, Query, Res, ResMut, Resource, not, resource_exists,
};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy_mod_openxr::environment_blend_mode::OxrEnvironmentBlendModes;
use bevy_mod_openxr::resources::{OxrPassthroughLayerFB, OxrRenderLayers};
use bevy_mod_xr::session::XrState;
use ftgsxr_app::SettingsRes;

use super::layer::Record;
use super::{Environment, EnvironmentMode, LayerRequest, PassthroughStyle, style_map};

/// The name of the entity the quest runner spawns to start the test (Q-308).
// Q-308
pub(crate) const TRIGGER: &str = "ftgsxr-test: dv6";
/// The test's name in the verdict line.
const NAME: &str = "dv6";
/// How long each mode is held once reached.
const HOLD: Duration = Duration::from_secs(2);
/// How long a mode may take to be reached in the render world.
const REACH_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the session and the layer may take to appear.
const START_TIMEOUT: Duration = Duration::from_secs(60);
/// The brightness of the second passthrough switch, a style change.
const BRIGHTNESS: f32 = 0.2;

/// One render frame as seen after `layer` applied the request.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Frame {
    /// The mode the render world was asked for.
    mode: EnvironmentMode,
    /// The layer exists.
    created: bool,
    /// `layer`'s provider is first in the list.
    first: bool,
    /// `layer`'s provider is anywhere in the list.
    in_list: bool,
    /// The list's length.
    len: usize,
    /// The layer is resumed.
    resumed: bool,
    /// The style last sent and whether it was accepted.
    style: Option<(PassthroughStyle, bool)>,
    /// The pause, resume and style calls that failed in the session.
    failures: u32,
}

impl Frame {
    /// Q-346's state for `mode`: passthrough's provider first of two and resumed, or the
    /// projection layer alone and the layer paused.
    fn holds(&self, mode: EnvironmentMode) -> bool {
        self.mode == mode
            && match mode {
                EnvironmentMode::Passthrough => self.first && self.len == 2 && self.resumed,
                EnvironmentMode::Vr => !self.in_list && self.len == 1 && !self.resumed,
            }
    }
}

/// What the render world saw: the last frame, the frames seen and those whose blend mode
/// was not OPAQUE.
#[derive(Clone, Debug, Default)]
pub(crate) struct Seen {
    last: Frame,
    frames: u64,
    not_opaque: u64,
}

/// `Seen`, shared by the two worlds.
#[derive(Resource, Clone, Default)]
pub(crate) struct SeenShared(Arc<Mutex<Seen>>);

impl SeenShared {
    fn get(&self) -> Seen {
        self.0.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

/// The test's phase.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    /// Until the session runs and the layer exists.
    Start,
    /// Switching to `mode` in cycle `cycle` (0 or 1); `reached` once the render world
    /// shows it.
    Switch {
        cycle: u32,
        mode: EnvironmentMode,
        reached: Option<Instant>,
    },
    Done,
}

/// The test's state, inserted when the quest runner's entity appears.
#[derive(Resource, Debug)]
pub(crate) struct Dv6 {
    phase: Phase,
    since: Instant,
    /// Frames that broke a held mode, with the first such frame.
    broken: u32,
    first_broken: Option<Frame>,
    parts: Vec<String>,
    pass: bool,
}

/// Adds the test's systems: inert until an entity named [`TRIGGER`] exists.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn add(app: &mut App) {
    let shared = SeenShared::default();
    app.insert_resource(shared.clone()).add_systems(
        PostUpdate,
        (
            detect.run_if(not(resource_exists::<Dv6>)),
            run.run_if(resource_exists::<Dv6>),
        ),
    );
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.insert_resource(shared).add_systems(
            Render,
            observe
                .in_set(RenderSystems::Prepare)
                .after(super::layer::apply),
        );
    }
}

/// Starts the test when the quest runner's entity appears (Q-308).
#[cfg_attr(test, allow(dead_code))]
fn detect(names: Query<&Name>, mut commands: Commands) {
    if names.iter().any(|n| n.as_str() == TRIGGER) {
        info!("dv6: started");
        commands.insert_resource(Dv6 {
            phase: Phase::Start,
            since: Instant::now(),
            broken: 0,
            first_broken: None,
            parts: Vec::new(),
            pass: true,
        });
    }
}

/// One render frame: what `layer` did, after it applied the request.
#[cfg_attr(test, allow(dead_code))]
fn observe(
    request: Option<Res<LayerRequest>>,
    record: Option<Res<Record>>,
    layers: Option<Res<OxrRenderLayers>>,
    layer: Option<Res<OxrPassthroughLayerFB>>,
    blend: Option<Res<OxrEnvironmentBlendModes>>,
    shared: Res<SeenShared>,
) {
    let (Some(request), Some(record), Some(layers)) = (request, record, layers) else {
        return;
    };
    let frame = Frame {
        mode: request.mode,
        created: layer.is_some(),
        first: record.first_of(&layers),
        in_list: record.in_list(&layers),
        len: layers.len(),
        resumed: record.resumed,
        style: record.style,
        failures: record.failures,
    };
    let opaque = blend.is_some_and(|b| b.blend_mode() == openxr::EnvironmentBlendMode::OPAQUE);
    if let Ok(mut seen) = shared.0.lock() {
        seen.last = frame;
        seen.frames += 1;
        if !opaque {
            seen.not_opaque += 1;
        }
    }
}

/// Sets the wanted mode and the brightness in the settings.
fn ask(settings: Option<&mut SettingsRes>, mode: EnvironmentMode, brightness: f32) {
    if let Some(s) = settings {
        let e = &mut s.0.environment;
        e.mode = match mode {
            EnvironmentMode::Passthrough => "passthrough",
            EnvironmentMode::Vr => "vr",
        }
        .to_owned();
        e.passthrough_brightness = brightness;
    }
}

/// One frame of the test.
#[cfg_attr(test, allow(dead_code))]
fn run(
    mut test: ResMut<Dv6>,
    state: Option<Res<XrState>>,
    env: Res<Environment>,
    shared: Res<SeenShared>,
    mut settings: Option<ResMut<SettingsRes>>,
) {
    let seen = shared.get();
    let now = Instant::now();
    match test.phase {
        Phase::Done => {}
        Phase::Start => {
            let running = state.as_deref() == Some(&XrState::Running);
            if running && !env.passthrough_available {
                test.fail("passthrough unavailable on this runtime".to_owned());
            } else if running && seen.last.created {
                test.parts.push("layer created paused".to_owned());
                ask(settings.as_deref_mut(), EnvironmentMode::Passthrough, 0.0);
                test.enter(0, EnvironmentMode::Passthrough);
            } else if now - test.since > START_TIMEOUT {
                test.fail(format!(
                    "no running session with the layer in {} s (running {running})",
                    START_TIMEOUT.as_secs()
                ));
            }
        }
        Phase::Switch {
            cycle,
            mode,
            reached,
        } => {
            let holds = seen.last.holds(mode);
            match reached {
                None if holds => {
                    test.phase = Phase::Switch {
                        cycle,
                        mode,
                        reached: Some(now),
                    };
                }
                None if now - test.since > REACH_TIMEOUT => {
                    test.fail(format!(
                        "cycle {cycle}: {mode:?} not reached in {} s: {:?}",
                        REACH_TIMEOUT.as_secs(),
                        seen.last
                    ));
                }
                None => {}
                Some(at) => {
                    if !holds {
                        test.broken += 1;
                        test.first_broken.get_or_insert(seen.last);
                    }
                    if now - at >= HOLD {
                        test.parts.push(describe(cycle, mode, &seen.last));
                        match (cycle, mode) {
                            (_, EnvironmentMode::Passthrough) => {
                                ask(settings.as_deref_mut(), EnvironmentMode::Vr, 0.0);
                                test.enter(cycle, EnvironmentMode::Vr);
                            }
                            (0, EnvironmentMode::Vr) => {
                                ask(
                                    settings.as_deref_mut(),
                                    EnvironmentMode::Passthrough,
                                    BRIGHTNESS,
                                );
                                test.enter(1, EnvironmentMode::Passthrough);
                            }
                            (_, EnvironmentMode::Vr) => test.finish(&seen),
                        }
                    }
                }
            }
        }
    }
}

/// One held mode's report.
fn describe(cycle: u32, mode: EnvironmentMode, f: &Frame) -> String {
    match mode {
        EnvironmentMode::Passthrough => format!(
            "cycle {cycle} passthrough: provider first of {}, resumed, held {} s",
            f.len,
            HOLD.as_secs()
        ),
        EnvironmentMode::Vr => format!(
            "cycle {cycle} vr: projection layer alone, paused, held {} s",
            HOLD.as_secs()
        ),
    }
}

impl Dv6 {
    fn enter(&mut self, cycle: u32, mode: EnvironmentMode) {
        self.phase = Phase::Switch {
            cycle,
            mode,
            reached: None,
        };
        self.since = Instant::now();
    }

    fn fail(&mut self, why: String) {
        self.pass = false;
        self.parts.push(why);
        self.done();
    }

    /// The last clauses: no frame broke a held mode, OPAQUE throughout, the calls
    /// accepted, the second cycle's style sent (Q-337).
    fn finish(&mut self, seen: &Seen) {
        if self.broken > 0 {
            self.pass = false;
            self.parts.push(format!(
                "{} frames broke a held mode, first {:?}",
                self.broken, self.first_broken
            ));
        }
        self.parts.push(format!(
            "blend OPAQUE in {} of {} frames",
            seen.frames - seen.not_opaque,
            seen.frames
        ));
        if seen.not_opaque > 0 || seen.frames == 0 {
            self.pass = false;
        }
        let want = PassthroughStyle {
            brightness: BRIGHTNESS,
            ..PassthroughStyle::default()
        };
        match seen.last.style {
            Some((s, true)) if s == want => {
                let (b, c, sat) = style_map(s);
                self.parts.push(format!(
                    "style applied (brightness {b}, contrast {c}, saturation {sat})"
                ));
            }
            other => {
                self.pass = false;
                self.parts.push(format!("style not applied: {other:?}"));
            }
        }
        self.parts
            .push(format!("{} call failures", seen.last.failures));
        if seen.last.failures > 0 {
            self.pass = false;
        }
        self.done();
    }

    fn done(&mut self) {
        let word = if self.pass { "PASS" } else { "FAIL" };
        // §3.4, Q-122: the verdict line, as `ftgsxr-quest`'s runner prints it.
        info!("ftgsxr-test: {NAME} {word} {}", self.parts.join("; "));
        self.phase = Phase::Done;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q-346's two states, each with the mode asked for.
    #[test]
    fn frame_holds_q346s_state_for_each_mode() {
        let pt = Frame {
            mode: EnvironmentMode::Passthrough,
            created: true,
            first: true,
            in_list: true,
            len: 2,
            resumed: true,
            ..Frame::default()
        };
        assert!(pt.holds(EnvironmentMode::Passthrough));
        assert!(!pt.holds(EnvironmentMode::Vr));
        assert!(
            !Frame {
                resumed: false,
                ..pt
            }
            .holds(EnvironmentMode::Passthrough),
            "a paused layer"
        );
        assert!(
            !Frame { first: false, ..pt }.holds(EnvironmentMode::Passthrough),
            "not first"
        );
        let vr = Frame {
            mode: EnvironmentMode::Vr,
            created: true,
            len: 1,
            ..Frame::default()
        };
        assert!(vr.holds(EnvironmentMode::Vr));
        assert!(
            !Frame {
                in_list: true,
                len: 2,
                ..vr
            }
            .holds(EnvironmentMode::Vr),
            "the provider left in the list"
        );
        assert!(
            !Frame {
                resumed: true,
                ..vr
            }
            .holds(EnvironmentMode::Vr),
            "a resumed layer"
        );
    }
}
