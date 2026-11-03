// SPDX-License-Identifier: Apache-2.0

//! I-5's clock half (§1.6 Clock and Per-instance time, E9, D-32, `cpu`, 10⁻⁶ s): play
//! advances every `SplatTime` by speed · Δt; the loop modes; a frame step; an offset; the
//! extrapolating flag. The probes P1 and P2 span τ ∈ [0, 1.9] s, so r = 1 and τ = T − o
//! under hold. The audio clause (the stub sink follows the clock) is #281's.

mod support;

use bevy::prelude::*;
use ftgsxr_app::{
    Intent, LibraryCommand, LoadedModel, ModelRef, Playback, PlaybackCommand, SplatTime, UiCommand,
};
use ftgsxr_core::{EndRule, LoopMode, PlayState};
use ftgsxr_testkit::fixtures_dir;
use support::{FRAME, TestApp};

const TOL: f64 = 1e-6;

/// The test app with the given probes loaded (their `LoadedModel` and `SplatTime` present).
fn loaded(name: &str, probes: &[&str]) -> (TestApp, Vec<Entity>) {
    let mut t = TestApp::new(name);
    for p in probes {
        let path = fixtures_dir().join(format!("probes/{p}.safetensors"));
        t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    }
    let n = probes.len();
    let ok = t.until(600, |app| {
        app.world_mut()
            .query::<(&LoadedModel, &SplatTime)>()
            .iter(app.world())
            .count()
            == n
    });
    assert!(ok, "the probes load");
    let mut es = Vec::new();
    for p in probes {
        let mut q = t.app.world_mut().query::<(Entity, &LoadedModel)>();
        let e = q
            .iter(t.app.world())
            .find(|(_, m)| m.name == *p)
            .map(|(e, _)| e)
            .unwrap();
        es.push(e);
    }
    (t, es)
}

fn intent(t: &mut TestApp, c: PlaybackCommand) {
    t.app.world_mut().write_message(Intent::Playback(c));
}

fn clock_t(t: &TestApp) -> f64 {
    t.app.world().resource::<Playback>().clock.time()
}

fn splat_time(t: &TestApp, e: Entity) -> SplatTime {
    *t.app.world().get::<SplatTime>(e).unwrap()
}

/// Seeks to `t_s` and runs one frame (paused, so the clock stays there).
fn seek(t: &mut TestApp, t_s: f64) {
    intent(t, PlaybackCommand::Pause);
    intent(t, PlaybackCommand::Seek(t_s));
    t.frames(1);
}

#[test]
fn i05_play_advances_every_splat_time_by_speed_dt() {
    let (mut t, es) = loaded("i05-play", &["P1", "P2"]);
    let dt = FRAME.as_secs_f64();
    assert!((t.app.world().resource::<Playback>().clock.length() - 1.9).abs() < TOL);
    seek(&mut t, 0.25);
    intent(&mut t, PlaybackCommand::Speed(2.0));
    intent(&mut t, PlaybackCommand::Play);
    t.frames(1);
    let t0 = clock_t(&t);
    t.frames(10);
    let t1 = clock_t(&t);
    assert!((t1 - t0 - 2.0 * 10.0 * dt).abs() < TOL, "{t0} → {t1}");
    for e in &es {
        let s = splat_time(&t, *e);
        assert!((s.tau - t1).abs() < TOL);
        assert!(!s.extrapolating);
    }
    intent(&mut t, PlaybackCommand::Toggle);
    t.frames(5);
    assert_eq!(
        t.app.world().resource::<Playback>().clock.state(),
        PlayState::Paused
    );
    assert!((clock_t(&t) - t1).abs() < TOL, "paused holds");
}

#[test]
fn i05_loop_modes_wrap_reflect_and_stop() {
    let (mut t, es) = loaded("i05-loop", &["P1"]);
    let dt = FRAME.as_secs_f64();
    let length = 1.9;
    // Loop: from 1.9 − 3Δt, six frames forward wrap to 3Δt.
    seek(&mut t, length - 3.0 * dt);
    intent(&mut t, PlaybackCommand::Play);
    t.frames(6);
    assert!(
        (clock_t(&t) - 3.0 * dt).abs() < TOL,
        "loop wraps: {}",
        clock_t(&t)
    );
    assert!((splat_time(&t, es[0]).tau - 3.0 * dt).abs() < TOL);
    // Ping-pong: the same six frames reflect to 1.9 − 3Δt, and the speed turns negative.
    intent(&mut t, PlaybackCommand::Loop(LoopMode::PingPong));
    seek(&mut t, length - 3.0 * dt);
    intent(&mut t, PlaybackCommand::Play);
    t.frames(6);
    assert!(
        (clock_t(&t) - (length - 3.0 * dt)).abs() < TOL,
        "ping-pong reflects"
    );
    assert!(t.app.world().resource::<Playback>().clock.speed() < 0.0);
    // Once (through the cycle from ping-pong): stops at the end and pauses.
    t.app
        .world_mut()
        .write_message(Intent::Ui(UiCommand::CycleLoop));
    intent(&mut t, PlaybackCommand::Speed(1.0));
    seek(&mut t, length - 3.0 * dt);
    intent(&mut t, PlaybackCommand::Play);
    t.frames(6);
    assert!((clock_t(&t) - length).abs() < TOL, "once stops at T_len");
    assert_eq!(
        t.app.world().resource::<Playback>().clock.state(),
        PlayState::Paused
    );
    // A range [0.5, 1.5] in loop: from 1.5 − 3Δt, six frames wrap to 0.5 + 3Δt.
    t.app
        .world_mut()
        .write_message(Intent::Ui(UiCommand::CycleLoop));
    intent(&mut t, PlaybackCommand::Range(Some((0.5, 1.5))));
    seek(&mut t, 1.5 - 3.0 * dt);
    intent(&mut t, PlaybackCommand::Play);
    t.frames(6);
    assert!((clock_t(&t) - (0.5 + 3.0 * dt)).abs() < TOL, "range loop");
}

#[test]
fn i05_a_frame_step_moves_by_delta_frame_while_paused() {
    let (mut t, _) = loaded("i05-step", &["P1"]);
    seek(&mut t, 1.0);
    // D-32: P1 has one primitive, so no median spacing: the default 1/30 s.
    intent(&mut t, PlaybackCommand::StepFrames(3));
    t.frames(1);
    assert!((clock_t(&t) - (1.0 + 3.0 / 30.0)).abs() < TOL);
    intent(&mut t, PlaybackCommand::StepFrames(-1));
    t.frames(1);
    assert!((clock_t(&t) - (1.0 + 2.0 / 30.0)).abs() < TOL);
    // §1.6: only while paused.
    intent(&mut t, PlaybackCommand::Play);
    intent(&mut t, PlaybackCommand::StepFrames(30));
    t.frames(1);
    assert!(clock_t(&t) < 1.2, "no step while playing");
}

#[test]
fn i05_an_offset_shifts_tau_and_extrapolate_flags() {
    let (mut t, es) = loaded("i05-offset", &["P1", "P2"]);
    intent(
        &mut t,
        PlaybackCommand::Offset {
            instance: es[1],
            offset_s: 0.5,
        },
    );
    seek(&mut t, 0.3);
    // T_len = max(1.9, 0.5 + 1.9) (E9).
    assert!((t.app.world().resource::<Playback>().clock.length() - 2.4).abs() < TOL);
    assert!((splat_time(&t, es[0]).tau - 0.3).abs() < TOL);
    // Hold clamps 0.3 − 0.5 to τ₀ = 0, unflagged.
    assert_eq!(
        splat_time(&t, es[1]),
        SplatTime {
            tau: 0.0,
            extrapolating: false
        }
    );
    intent(
        &mut t,
        PlaybackCommand::EndRule {
            instance: es[1],
            rule: EndRule::Extrapolate,
        },
    );
    t.frames(1);
    let s = splat_time(&t, es[1]);
    assert!((s.tau - (-0.2)).abs() < TOL);
    assert!(s.extrapolating);
    // Inside the span the flag clears.
    seek(&mut t, 1.0);
    let s = splat_time(&t, es[1]);
    assert!((s.tau - 0.5).abs() < TOL);
    assert!(!s.extrapolating);
}

#[test]
fn i05_single_timestamp_instances_do_not_count_toward_t_len() {
    let (mut t, es) = loaded("i05-single", &["R2_single_timestamp"]);
    assert_eq!(t.app.world().resource::<Playback>().clock.length(), 0.0);
    intent(&mut t, PlaybackCommand::Play);
    t.frames(5);
    assert_eq!(clock_t(&t), 0.0, "the idle clock");
    assert_eq!(splat_time(&t, es[0]).tau, 0.95);
    let (mut t, es) = loaded("i05-mixed", &["R2_single_timestamp", "P1"]);
    intent(
        &mut t,
        PlaybackCommand::Offset {
            instance: es[0],
            offset_s: 10.0,
        },
    );
    t.frames(1);
    assert!((t.app.world().resource::<Playback>().clock.length() - 1.9).abs() < TOL);
    // Unloading P1 leaves only the single timestamp: T_len = 0.
    t.command(LibraryCommand::Unload(es[1]));
    t.frames(2);
    assert_eq!(t.app.world().resource::<Playback>().clock.length(), 0.0);
}

#[test]
fn i05_a_speed_outside_e9_is_refused() {
    let (mut t, _) = loaded("i05-speed", &["P1"]);
    intent(&mut t, PlaybackCommand::Speed(-2.0));
    t.frames(1);
    for bad in [0.0, 4.5, -5.0, f64::NAN] {
        intent(&mut t, PlaybackCommand::Speed(bad));
        t.frames(1);
        assert_eq!(t.app.world().resource::<Playback>().clock.speed(), -2.0);
    }
}

// Q-174
#[test]
fn i05_an_exchange_model_steps_by_its_median_spacing() {
    // D-32: four distinct time centers 0.1 s apart (r = 1 for seconds, span 1.9 s): the
    // median spacing 0.1 s lies within [1/480 s, 1.9 s], so Δ_frame = 0.1 s.
    let mut model = ftgsxr_testkit::random_model(7, 4, 0);
    model.time_center = vec![0.4, 0.5, 0.6, 0.7];
    let mut t = TestApp::new("i05-median");
    let path = t.dir.join("spaced.safetensors");
    std::fs::create_dir_all(&t.dir).unwrap();
    std::fs::write(&path, ftgsxr_testkit::exchange_bytes(&model)).unwrap();
    t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    assert!(t.until(600, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1
    }));
    let m = {
        let mut q = t.app.world_mut().query::<&LoadedModel>();
        q.single(t.app.world()).unwrap().clone()
    };
    assert!((m.frame_step_s - 0.1).abs() < TOL, "{}", m.frame_step_s);
    t.frames(2);
    seek(&mut t, 1.0);
    intent(&mut t, PlaybackCommand::StepFrames(3));
    t.frames(1);
    assert!((clock_t(&t) - 1.3).abs() < TOL, "{}", clock_t(&t));
}
