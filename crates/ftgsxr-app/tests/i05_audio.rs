// SPDX-License-Identifier: Apache-2.0

//! I-5's audio clause (§1.6 Audio, §1.10 Soundtracks, D-22, `cpu`, 10⁻⁶ s): the stub
//! audio sink's position follows the clock at speed 1 and mutes otherwise, seeking to the
//! clock's position when playback at speed 1 resumes. The probe P1 spans τ ∈ [0, 1.9] s
//! (r = 1), so the soundtrack's position is T − o. The stub has no clock: each frame the
//! test moves its playing sinks by the frame's Δt, as a device would.
//!
//! The clause for a scrub while playing (silent until it ends, Q-190, Q-245, #416) is
//! `audio`'s own test, since the scrub state is crate-private; this file pins that a
//! one-off seek while playing is not a scrub.

use crate::support;

use ftgsxr_app::{
    Intent, LibraryCommand, ModelRef, Playback, PlaybackCommand, Soundtrack, StubSink,
};
use ftgsxr_testkit::fixtures_dir;
use support::{FRAME, TestApp};

const TOL: f64 = 1e-6;

/// The app with P1 loaded from a scratch copy beside `P1.wav`, its sink in the stub.
fn with_soundtrack(name: &str) -> TestApp {
    let mut t = TestApp::new(name);
    let dir = t.dir.join("beside");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("P1.safetensors");
    std::fs::copy(fixtures_dir().join("probes/P1.safetensors"), &model).unwrap();
    // The stub decodes any non-empty file.
    std::fs::write(dir.join("P1.wav"), b"RIFF").unwrap();
    t.command(LibraryCommand::LoadRef(ModelRef::File(model)));
    let audio = t.audio.clone();
    let ok = t.until(600, |app| {
        let tracks = app
            .world_mut()
            .query::<&Soundtrack>()
            .iter(app.world())
            .count();
        tracks == 1 && audio.sinks().len() == 1
    });
    assert!(ok, "P1 loads with its soundtrack");
    t
}

fn sink(t: &TestApp) -> StubSink {
    t.audio.sinks().into_values().next().unwrap()
}

fn clock_t(t: &TestApp) -> f64 {
    t.app.world().resource::<Playback>().clock.time()
}

fn intent(t: &mut TestApp, c: PlaybackCommand) {
    t.app.world_mut().write_message(Intent::Playback(c));
}

/// One frame: the stub plays Δt, then the app runs.
fn frame(t: &mut TestApp) {
    t.audio.advance(FRAME.as_secs_f64());
    t.app.update();
}

/// `n` frames, the sink following the clock after each.
fn follows(t: &mut TestApp, n: usize) {
    for _ in 0..n {
        frame(t);
        let s = sink(t);
        assert!(s.playing, "plays at speed 1");
        assert!(
            (s.position - clock_t(t)).abs() < TOL,
            "{} vs T {}",
            s.position,
            clock_t(t)
        );
    }
}

/// `n` frames, the sink paused at one position.
fn silent(t: &mut TestApp, n: usize) {
    frame(t);
    let at = sink(t).position;
    for _ in 0..n {
        frame(t);
        let s = sink(t);
        assert!(!s.playing, "silent");
        assert_eq!(s.position, at, "held while silent");
    }
}

#[test]
fn i05_audio_follows_the_clock_at_speed_one() {
    let mut t = with_soundtrack("i05-audio-follow");
    assert!(!sink(&t).playing, "paused clock, silent sink");
    intent(&mut t, PlaybackCommand::Seek(0.3));
    intent(&mut t, PlaybackCommand::Play);
    follows(&mut t, 30);
    // Past the end, the loop wraps the clock and the sink with it.
    follows(&mut t, 100);
}

#[test]
fn i05_audio_mutes_otherwise_and_seeks_on_resume() {
    let mut t = with_soundtrack("i05-audio-mute");
    intent(&mut t, PlaybackCommand::Play);
    follows(&mut t, 10);
    // Another speed: silent while the clock runs on.
    intent(&mut t, PlaybackCommand::Speed(2.0));
    silent(&mut t, 10);
    intent(&mut t, PlaybackCommand::Speed(1.0));
    follows(&mut t, 5);
    // Reverse.
    intent(&mut t, PlaybackCommand::Speed(-1.0));
    silent(&mut t, 10);
    intent(&mut t, PlaybackCommand::Speed(1.0));
    follows(&mut t, 5);
    // Paused, then scrubbed: silent; at speed 1 again, the clock's position.
    intent(&mut t, PlaybackCommand::Pause);
    silent(&mut t, 5);
    intent(&mut t, PlaybackCommand::Seek(1.2));
    silent(&mut t, 5);
    intent(&mut t, PlaybackCommand::Play);
    frame(&mut t);
    assert!((sink(&t).position - clock_t(&t)).abs() < TOL);
    assert!((clock_t(&t) - 1.2 - FRAME.as_secs_f64()).abs() < TOL);
    follows(&mut t, 5);
}

// Q-190, Q-245: a seek while playing at speed 1 (the start and end keys, a click on the
// scrub bar) moves the sink to the clock's position and leaves it playing.
#[test]
fn i05_audio_a_one_off_seek_while_playing_keeps_playing() {
    let mut t = with_soundtrack("i05-audio-one-off-seek");
    intent(&mut t, PlaybackCommand::Play);
    follows(&mut t, 5);
    intent(&mut t, PlaybackCommand::Seek(1.2));
    frame(&mut t);
    let s = sink(&t);
    assert!(s.playing, "plays on after the seek");
    assert!((s.position - clock_t(&t)).abs() < TOL);
    assert!((clock_t(&t) - 1.2 - FRAME.as_secs_f64()).abs() < TOL);
    follows(&mut t, 5);
}

#[test]
fn i05_audio_soundtrack_time_zero_is_tau0_under_an_offset() {
    let mut t = with_soundtrack("i05-audio-offset");
    let e = t
        .app
        .world_mut()
        .query_filtered::<bevy::prelude::Entity, bevy::prelude::With<Soundtrack>>()
        .single(t.app.world())
        .unwrap();
    intent(
        &mut t,
        PlaybackCommand::Offset {
            instance: e,
            offset_s: 0.5,
        },
    );
    intent(&mut t, PlaybackCommand::Seek(0.2));
    intent(&mut t, PlaybackCommand::Play);
    // T < o: before τ₀, silent.
    silent(&mut t, 5);
    intent(&mut t, PlaybackCommand::Seek(0.9));
    for _ in 0..10 {
        frame(&mut t);
        let s = sink(&t);
        assert!(s.playing);
        assert!((s.position - (clock_t(&t) - 0.5)).abs() < TOL);
    }
}
