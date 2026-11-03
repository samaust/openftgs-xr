// SPDX-License-Identifier: Apache-2.0

//! I-18 (§1.10 Sounds and volumes, `cpu`, 10⁻⁶) with the stub output: mute silences
//! every sink; each volume scales its channel's gain; the per-sound switches. The
//! persistence clause (the values in `Settings.audio`) is `persist`'s (#285).

mod support;

use ftgsxr_app::{
    AudioState, Feature, Intent, LibraryCommand, ModelRef, PlayUiSound, PlaybackCommand,
    Soundtrack, UiSound, VolumeChannel,
};
use ftgsxr_testkit::fixtures_dir;
use support::TestApp;

const TOL: f32 = 1e-6;

const ALL: [UiSound; 8] = [
    UiSound::Hover,
    UiSound::Select,
    UiSound::GrabBegin,
    UiSound::GrabEnd,
    UiSound::SnapTurn,
    UiSound::Click,
    UiSound::Error,
    UiSound::Confirm,
];

/// The app with its models directory present, so the library's listing raises no E-IO
/// toast, whose error sound would join the sounds counted here (#286).
fn app(name: &str) -> TestApp {
    TestApp::with(name, support::Setup::default(), |_| {
        std::fs::create_dir_all(support::scratch(name).join("models")).unwrap();
    })
}

fn play(t: &mut TestApp, sound: UiSound) -> Vec<(UiSound, f32)> {
    t.app.world_mut().write_message(PlayUiSound(sound));
    t.frames(1);
    t.audio.take_played()
}

fn volume(t: &mut TestApp, channel: VolumeChannel, value: f32) {
    t.app
        .world_mut()
        .write_message(Intent::Volume { channel, value });
}

/// P1 with a soundtrack, playing at speed 1.
fn with_soundtrack(name: &str) -> TestApp {
    let mut t = app(name);
    let dir = t.dir.join("beside");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("P1.safetensors");
    std::fs::copy(fixtures_dir().join("probes/P1.safetensors"), &model).unwrap();
    std::fs::write(dir.join("P1.ogg"), b"OggS").unwrap();
    t.command(LibraryCommand::LoadRef(ModelRef::File(model)));
    let audio = t.audio.clone();
    let ok = t.until(600, |app| {
        app.world_mut()
            .query::<&Soundtrack>()
            .iter(app.world())
            .count()
            == 1
            && audio.sinks().len() == 1
    });
    assert!(ok, "P1 loads with its soundtrack");
    t.app
        .world_mut()
        .write_message(Intent::Playback(PlaybackCommand::Play));
    t.frames(1);
    t
}

fn sink_gain(t: &TestApp) -> f32 {
    t.audio.sinks().into_values().next().unwrap().gain
}

#[test]
fn i18_the_clips_load_at_startup_and_play_at_full_gain() {
    let mut t = app("i18-start");
    t.frames(1);
    assert_eq!(t.audio.loaded(), ALL.to_vec());
    for sound in ALL {
        assert_eq!(play(&mut t, sound), vec![(sound, 1.0)]);
    }
}

#[test]
fn i18_each_volume_scales_its_channel() {
    let mut t = with_soundtrack("i18-volumes");
    assert!((sink_gain(&t) - 1.0).abs() < TOL);
    volume(&mut t, VolumeChannel::Ui, 0.4);
    let played = play(&mut t, UiSound::Click);
    assert_eq!(played.len(), 1);
    assert!((played[0].1 - 0.4).abs() < TOL);
    assert!(
        (sink_gain(&t) - 1.0).abs() < TOL,
        "UI volume leaves the soundtrack"
    );
    volume(&mut t, VolumeChannel::Soundtrack, 0.25);
    t.frames(1);
    assert!((sink_gain(&t) - 0.25).abs() < TOL);
    assert!(
        (play(&mut t, UiSound::Click)[0].1 - 0.4).abs() < TOL,
        "and back"
    );
    volume(&mut t, VolumeChannel::Master, 0.5);
    t.frames(1);
    assert!((sink_gain(&t) - 0.125).abs() < TOL);
    assert!((play(&mut t, UiSound::Select)[0].1 - 0.2).abs() < TOL);
    // Out of range clamps to [0, 1].
    volume(&mut t, VolumeChannel::Master, 3.0);
    t.frames(1);
    assert_eq!(t.app.world().resource::<AudioState>().master, 1.0);
}

#[test]
fn i18_mute_silences_every_sink() {
    let mut t = with_soundtrack("i18-mute");
    volume(&mut t, VolumeChannel::Master, 0.5);
    t.app
        .world_mut()
        .write_message(Intent::Toggle(Feature::Sound));
    t.frames(1);
    assert!(t.app.world().resource::<AudioState>().muted);
    assert_eq!(sink_gain(&t), 0.0);
    for sound in ALL {
        assert!(play(&mut t, sound).is_empty(), "{sound:?} muted");
    }
    t.app
        .world_mut()
        .write_message(Intent::Toggle(Feature::Sound));
    t.frames(1);
    assert!(
        (sink_gain(&t) - 0.5).abs() < TOL,
        "unmute restores the gain"
    );
    assert!((play(&mut t, UiSound::Hover)[0].1 - 0.5).abs() < TOL);
}

#[test]
fn i18_each_switch_silences_its_sounds() {
    let mut t = app("i18-switches");
    t.frames(1);
    type Switch = fn(&mut AudioState) -> &mut bool;
    let switches: [(Switch, &[UiSound]); 7] = [
        (|s| &mut s.sounds.hover, &[UiSound::Hover]),
        (|s| &mut s.sounds.select, &[UiSound::Select]),
        (
            |s| &mut s.sounds.grab,
            &[UiSound::GrabBegin, UiSound::GrabEnd],
        ),
        (|s| &mut s.sounds.snap_turn, &[UiSound::SnapTurn]),
        (|s| &mut s.sounds.click, &[UiSound::Click]),
        (|s| &mut s.sounds.error, &[UiSound::Error]),
        (|s| &mut s.sounds.confirm, &[UiSound::Confirm]),
    ];
    for (switch, off) in switches {
        *switch(&mut t.app.world_mut().resource_mut::<AudioState>()) = false;
        for sound in ALL {
            let played = play(&mut t, sound);
            if off.contains(&sound) {
                assert!(played.is_empty(), "{sound:?} switched off");
            } else {
                assert_eq!(played, vec![(sound, 1.0)], "{sound:?} still on");
            }
        }
        *switch(&mut t.app.world_mut().resource_mut::<AudioState>()) = true;
    }
}
