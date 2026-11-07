// SPDX-License-Identifier: Apache-2.0

//! Dv-7 on the device (§3.4, §1.9 Depth occlusion, E13), as Q-346 reads it without a
//! person. It lives here, not in `ftgsxr-quest`, because it reads the render world's
//! crate-private provider and the frames the drive gave; the quest runner starts it by
//! spawning an entity named [`TRIGGER`] (Q-308), and it prints the verdict line
//! `ftgsxr-test: dv7 PASS|FAIL <detail>` itself (§3.4, Q-122).
//!
//! Once the session runs, the test sets passthrough mode and occlusion on (the driver
//! granted USE\_SCENE beforehand), waits for the first frame that has two depth frames,
//! then counts for [`COUNT`] the frames that had two with finite planes and distinct
//! cameras; it passes when every counted frame had them, then turns occlusion off and
//! passes when the provider stops, with no failed runtime call in the session. The detail
//! carries PRD #565's spike checks: the swapchain's size and length (check 1: the format
//! and layer count are the registry's D16 and 2, which the conversion relies on; a mismatch is a
//! validation error in the log) and whether each view's GPU texture kept its id across the
//! counted frames (check 2).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::app::{App, PostUpdate};
use bevy::log::info;
use bevy::prelude::{
    Commands, IntoScheduleConfigs, Name, Query, Res, ResMut, Resource, not, resource_exists,
};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::TextureId;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy_mod_xr::session::XrState;
use ftgsxr_app::SettingsRes;
use ftgsxr_core::UVec2;
use ftgsxr_render::SplatOcclusion;

use super::{Control, DepthRequest, Provider};
use crate::passthrough::Environment;

/// The name of the entity the quest runner spawns to start the test (Q-308).
// Q-308
pub(crate) const TRIGGER: &str = "ftgsxr-test: dv7";
/// The test's name in the verdict line.
const NAME: &str = "dv7";
/// How long the frames are counted once the first arrived (Q-346: 3 s).
const COUNT: Duration = Duration::from_secs(3);
/// How long the first depth frame may take after the request.
const FIRST_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the session may take to run.
const START_TIMEOUT: Duration = Duration::from_secs(60);
/// How long the provider may take to stop.
const STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// One render frame as the drive left it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Frame {
    /// The provider is started.
    running: bool,
    /// The frames given this frame.
    frames: usize,
    /// Both frames' planes are usable (`planes_ok`).
    planes: bool,
    /// The two frames' cameras differ.
    distinct: bool,
    /// The left frame's age.
    age: u32,
    /// The left frame's planes.
    near: f32,
    far: f32,
    /// The swapchain's size and length.
    swapchain: Option<(UVec2, usize)>,
    /// Each view's prepared GPU texture.
    textures: [Option<TextureId>; 2],
    /// The provider's failed calls in the session.
    failures: u32,
}

impl Frame {
    /// Q-346: two frames with finite planes and distinct cameras.
    fn good(&self) -> bool {
        self.frames == 2 && self.planes && self.distinct
    }
}

/// Whether a frame's planes are usable: a finite positive near, and a far that is
/// finite, the registry's infinite form (below near), or +∞, which the Quest 3 reports
/// and E13's decode reads as the infinite far plane (Q-340, Q-T570.4).
// Q-T570.4
fn planes_ok(near: f32, far: f32) -> bool {
    near.is_finite() && near > 0.0 && (far.is_finite() || far == f32::INFINITY)
}

/// What the render world saw while counting.
#[derive(Clone, Debug, Default)]
pub(crate) struct Seen {
    last: Frame,
    /// Counting, as the main world asks.
    counting: bool,
    counted: u64,
    bad: u64,
    first_bad: Option<Frame>,
    first_textures: Option<[Option<TextureId>; 2]>,
    texture_changes: u64,
    max_age: u32,
}

/// `Seen`, shared by the two worlds.
#[derive(Resource, Clone, Default)]
pub(crate) struct SeenShared(Arc<Mutex<Seen>>);

impl SeenShared {
    fn get(&self) -> Seen {
        self.0.lock().map(|s| s.clone()).unwrap_or_default()
    }

    fn count(&self, on: bool) {
        if let Ok(mut s) = self.0.lock() {
            s.counting = on;
        }
    }
}

/// The test's phase.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    /// Until the session runs.
    Start,
    /// Asked; until the first good frame.
    First,
    /// Counting since `at`.
    Counting {
        at: Instant,
    },
    /// Occlusion off; until the provider stops.
    Stopping,
    Done,
}

/// The test's state, inserted when the quest runner's entity appears.
#[derive(Resource, Debug)]
pub(crate) struct Dv7 {
    phase: Phase,
    since: Instant,
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
            detect.run_if(not(resource_exists::<Dv7>)),
            run.run_if(resource_exists::<Dv7>),
        ),
    );
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.insert_resource(shared).add_systems(
            Render,
            observe
                .in_set(RenderSystems::PrepareResources)
                .after(super::drive),
        );
    }
}

/// Starts the test when the quest runner's entity appears (Q-308).
#[cfg_attr(test, allow(dead_code))]
fn detect(names: Query<&Name>, mut commands: Commands) {
    if names.iter().any(|n| n.as_str() == TRIGGER) {
        info!("dv7: started");
        commands.insert_resource(Dv7 {
            phase: Phase::Start,
            since: Instant::now(),
            parts: Vec::new(),
            pass: true,
        });
    }
}

/// The frame the drive left: its frames, the provider's state, the views' textures.
fn frame_of(
    provider: &Provider,
    occlusion: Option<&SplatOcclusion>,
    request: Option<&DepthRequest>,
    images: &RenderAssets<GpuImage>,
) -> Frame {
    let depth = provider.depth.as_deref();
    let frames = occlusion.map_or(&[][..], |o| &o.frames[..]);
    let textures = request
        .and_then(|r| r.images.as_ref())
        .map_or([None, None], |[l, r]| {
            [
                images.get(l).map(|g| g.texture.id()),
                images.get(r).map(|g| g.texture.id()),
            ]
        });
    Frame {
        running: depth.is_some_and(|d| d.running()),
        frames: frames.len(),
        planes: !frames.is_empty() && frames.iter().all(|f| planes_ok(f.near, f.far)),
        distinct: frames.len() == 2 && frames[0].camera != frames[1].camera,
        age: frames.first().map_or(0, |f| f.age_frames),
        near: frames.first().map_or(0.0, |f| f.near),
        far: frames.first().map_or(0.0, |f| f.far),
        swapchain: depth.and_then(|d| d.swapchain()),
        textures,
        failures: depth.map_or(0, |d| d.failures()),
    }
}

/// One render frame, after the drive.
#[cfg_attr(test, allow(dead_code))]
fn observe(
    provider: Res<Provider>,
    occlusion: Option<Res<SplatOcclusion>>,
    request: Option<Res<DepthRequest>>,
    images: Res<RenderAssets<GpuImage>>,
    shared: Res<SeenShared>,
) {
    let f = frame_of(&provider, occlusion.as_deref(), request.as_deref(), &images);
    if let Ok(mut seen) = shared.0.lock() {
        seen.last = f;
        if seen.counting {
            seen.counted += 1;
            seen.max_age = seen.max_age.max(f.age);
            if !f.good() {
                seen.bad += 1;
                seen.first_bad.get_or_insert(f);
            }
            match seen.first_textures {
                None => seen.first_textures = Some(f.textures),
                Some(t) if t != f.textures => seen.texture_changes += 1,
                Some(_) => {}
            }
        }
    }
}

/// Sets the wanted mode and occlusion in the settings.
fn ask(settings: Option<&mut SettingsRes>, on: bool) {
    if let Some(s) = settings {
        let e = &mut s.0.environment;
        if on {
            e.mode = "passthrough".to_owned();
        }
        e.occlusion = on;
    }
}

/// One frame of the test.
#[cfg_attr(test, allow(dead_code))]
fn run(
    mut test: ResMut<Dv7>,
    state: Option<Res<XrState>>,
    env: Res<Environment>,
    control: Res<Control>,
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
            } else if running && !control.extension {
                test.fail("XR_META_environment_depth not enabled".to_owned());
            } else if running {
                ask(settings.as_deref_mut(), true);
                test.enter(Phase::First);
            } else if now - test.since > START_TIMEOUT {
                test.fail(format!(
                    "no running session in {} s",
                    START_TIMEOUT.as_secs()
                ));
            }
        }
        Phase::First => {
            if seen.last.good() {
                let ms = (now - test.since).as_millis();
                test.parts.push(format!("first frame after {ms} ms"));
                shared.count(true);
                test.enter(Phase::Counting { at: now });
            } else if now - test.since > FIRST_TIMEOUT {
                test.fail(format!(
                    "no depth frame in {} s (flow {:?}, mode {:?}): {:?}",
                    FIRST_TIMEOUT.as_secs(),
                    control.flow,
                    env.mode,
                    seen.last
                ));
            }
        }
        Phase::Counting { at } => {
            if now - at >= COUNT {
                shared.count(false);
                test.counted(&seen);
                ask(settings.as_deref_mut(), false);
                test.enter(Phase::Stopping);
            }
        }
        Phase::Stopping => {
            if !seen.last.running {
                test.parts.push("provider stopped".to_owned());
                test.finish(&seen);
            } else if now - test.since > STOP_TIMEOUT {
                test.fail(format!(
                    "the provider still runs after {} s",
                    STOP_TIMEOUT.as_secs()
                ));
            }
        }
    }
}

impl Dv7 {
    fn enter(&mut self, phase: Phase) {
        self.phase = phase;
        self.since = Instant::now();
    }

    fn fail(&mut self, why: String) {
        self.pass = false;
        self.parts.push(why);
        self.done();
    }

    /// The counted frames' clauses and the spike checks' observations.
    fn counted(&mut self, seen: &Seen) {
        let f = seen.last;
        self.parts.push(format!(
            "{} of {} frames had two frames with finite planes and distinct cameras",
            seen.counted - seen.bad,
            seen.counted
        ));
        if seen.bad > 0 || seen.counted == 0 {
            self.pass = false;
            self.parts
                .push(format!("first bad frame {:?}", seen.first_bad));
        }
        self.parts.push(format!(
            "near {} far {} max age {} frames",
            f.near, f.far, seen.max_age
        ));
        // Spike check 1: the swapchain the conversion reads (registry: D16, two layers).
        match f.swapchain {
            Some((size, n)) => self.parts.push(format!(
                "swapchain {} x {}, {n} images, D16 x 2 layers per the registry",
                size.x, size.y
            )),
            None => {
                self.pass = false;
                self.parts.push("no swapchain".to_owned());
            }
        }
        // Spike check 2: the handles' GPU textures kept across frames.
        let stable = seen.texture_changes == 0 && f.textures.iter().all(Option::is_some);
        self.parts
            .push(format!("views' GPU textures kept across frames: {stable}"));
        if !stable {
            self.pass = false;
        }
    }

    /// The last clause: no failed runtime call in the session.
    fn finish(&mut self, seen: &Seen) {
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

    /// Q-346: a good frame has two frames with finite planes and distinct cameras.
    #[test]
    fn frame_good_needs_two_frames_finite_planes_and_distinct_cameras() {
        let good = Frame {
            running: true,
            frames: 2,
            planes: true,
            distinct: true,
            ..Frame::default()
        };
        assert!(good.good());
        assert!(!Frame { frames: 1, ..good }.good());
        assert!(
            !Frame {
                planes: false,
                ..good
            }
            .good()
        );
        assert!(
            !Frame {
                distinct: false,
                ..good
            }
            .good()
        );
    }

    /// E13's planes: a finite positive near; a finite far, the infinite form written as a
    /// far below near, or +∞ (Q-T570.4); never NaN.
    #[test]
    fn planes_ok_takes_finite_planes_and_the_infinite_form() {
        assert!(planes_ok(0.1, 20.0));
        assert!(planes_ok(0.1, 0.0), "far < near: an infinite far plane");
        assert!(!planes_ok(0.0, 20.0));
        assert!(!planes_ok(f32::NAN, 20.0));
        assert!(
            planes_ok(0.1, f32::INFINITY),
            "+∞: the Quest 3's infinite far plane"
        );
        assert!(!planes_ok(0.1, f32::NAN));
        assert!(!planes_ok(0.1, f32::NEG_INFINITY));
    }
}
