// SPDX-License-Identifier: Apache-2.0

//! §1.10 Soundtracks' spatial clause (`cpu`, the stub output, #457): on the headset (the
//! `Headset` marker, Q-266) with `spatial` on, a soundtrack's gain is the soundtrack gain
//! × min(1, 1/d) and its pan the sine of the emitter's horizontal azimuth from the
//! listener's forward, positive right (Q-268); the listener is the midpoint of the splat
//! cameras (the two eyes), the emitter the instance's posed box center (Q-281).
//! Without the marker, or with `spatial` off, pan 0 and no attenuation (the desktop
//! plays stereo).

use crate::support;

use bevy::prelude::*;
use ftgsxr_app::{
    AudioState, Headset, LibraryCommand, LoadedModel, ModelRef, SortMode, Soundtrack, SplatCamera,
    StubSink,
};
use ftgsxr_render::{OutputSpace, SplatAsset, SplatModel};
use ftgsxr_testkit::fixtures_dir;
use support::TestApp;

const TOL: f32 = 1e-5;

/// The soundtrack volume, other than 1 so that the product is checked.
const VOLUME: f32 = 0.8;

/// The app (on the headset when `headset`) with P1 loaded beside a soundtrack, its sink
/// decoded and its asset ready, and the soundtrack volume at `VOLUME`.
#[track_caller]
fn with_soundtrack(name: &str, headset: bool) -> TestApp {
    let mut t = TestApp::with(name, support::Setup::default(), |app| {
        if headset {
            app.insert_resource(Headset);
        }
    });
    let dir = t.dir.join("beside");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("P1.safetensors");
    std::fs::copy(fixtures_dir().join("probes/P1.safetensors"), &model).unwrap();
    // The stub decodes any non-empty file.
    std::fs::write(dir.join("P1.wav"), b"RIFF").unwrap();
    t.command(LibraryCommand::LoadRef(ModelRef::File(model)));
    let audio = t.audio.clone();
    let ok = t.until(600, |app| {
        let ready = app
            .world_mut()
            .query_filtered::<&SplatModel, With<Soundtrack>>()
            .iter(app.world())
            .any(|m| app.world().resource::<Assets<SplatAsset>>().contains(&m.0));
        ready && audio.sinks().len() == 1
    });
    assert!(ok, "P1 loads with its soundtrack");
    t.app.world_mut().resource_mut::<AudioState>().soundtrack = VOLUME;
    t.frames(1);
    t
}

/// The instance's posed box center in world space.
fn emitter(t: &mut TestApp) -> Vec3 {
    let w = t.app.world_mut();
    let (g, m) = w
        .query_filtered::<(&GlobalTransform, &SplatModel), With<LoadedModel>>()
        .single(w)
        .unwrap();
    let b = w.resource::<Assets<SplatAsset>>().get(&m.0).unwrap().bounds;
    g.transform_point(b.center())
}

/// Two eyes 64 mm apart about `at`, looking along −z.
fn eyes(t: &mut TestApp, at: Vec3) {
    for dx in [-0.032, 0.032] {
        t.app.world_mut().spawn((
            Camera3d::default(),
            SplatCamera {
                sort: SortMode::PerView,
                output: OutputSpace::LinearOver { background: None },
            },
            Transform::from_translation(at + Vec3::new(dx, 0.0, 0.0)),
        ));
    }
}

fn sink(t: &TestApp) -> StubSink {
    t.audio.sinks().into_values().next().unwrap()
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < TOL
}

#[test]
fn spatial_two_metres_right_halves_the_gain_and_pans_right() {
    let mut t = with_soundtrack("right", true);
    let c = emitter(&mut t);
    eyes(&mut t, c - Vec3::new(2.0, 0.0, 0.0));
    t.frames(2);
    let s = sink(&t);
    assert!(close(s.gain, 0.5 * VOLUME), "gain {}", s.gain);
    assert!(close(s.pan, 1.0), "pan {}", s.pan);
}

#[test]
fn spatial_half_a_metre_ahead_is_full_gain_and_centered() {
    let mut t = with_soundtrack("ahead", true);
    let c = emitter(&mut t);
    eyes(&mut t, c + Vec3::new(0.0, 0.0, 0.5));
    t.frames(2);
    let s = sink(&t);
    assert!(close(s.gain, VOLUME), "gain {}", s.gain);
    assert!(close(s.pan, 0.0), "pan {}", s.pan);
}

#[test]
fn spatial_without_the_marker_is_stereo() {
    let mut t = with_soundtrack("desktop", false);
    let c = emitter(&mut t);
    eyes(&mut t, c - Vec3::new(2.0, 0.0, 0.0));
    t.frames(2);
    let s = sink(&t);
    assert!(close(s.gain, VOLUME), "gain {}", s.gain);
    assert!(close(s.pan, 0.0), "pan {}", s.pan);
}

#[test]
fn spatial_off_on_the_headset_is_stereo_again() {
    let mut t = with_soundtrack("off", true);
    let c = emitter(&mut t);
    eyes(&mut t, c - Vec3::new(2.0, 0.0, 0.0));
    t.frames(2);
    assert!(close(sink(&t).pan, 1.0));
    t.app.world_mut().resource_mut::<AudioState>().spatial = false;
    t.frames(1);
    let s = sink(&t);
    assert!(close(s.gain, VOLUME), "gain {}", s.gain);
    assert!(close(s.pan, 0.0), "pan {}", s.pan);
}
