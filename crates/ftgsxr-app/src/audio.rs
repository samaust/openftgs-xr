// SPDX-License-Identifier: Apache-2.0

//! Sounds, volumes and soundtracks (§2.1 `ftgsxr_app::audio`, §1.10, §1.6 Audio, D-22):
//! the volumes and the mute in `AudioState`, the UI sounds synthesized at startup and
//! played on `PlayUiSound`, and a `Soundtrack` for every instance whose `LoadedModel`
//! names one, decoded off the main thread and kept in step with the clock. Everything
//! reaches the system's output through the `AudioOutput` port (Q-149, Q-187): the kira
//! adapter on the desktop and the Quest, `StubAudioOutput` in the tests. Stereo only;
//! spatialization is M-5's.

use std::collections::HashMap;
use std::f32::consts::TAU;
use std::sync::{Arc, Mutex, MutexGuard};

use bevy::app::{App, Plugin, PostUpdate, Startup, Update};
use bevy::ecs::prelude::*;
use bevy::log::warn;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use ftgsxr_core::PlayState;
use ftgsxr_io::{IoError, ModelBytes, ModelRef, SoundSwitches, open_ref};

use crate::intent::{Feature, Intent, VolumeChannel};
use crate::library::LoadedModel;
use crate::platform::PlatformRes;
use crate::playback::{Playback, Scrubbing};
use crate::ui::{DockHooks, Tab, ToastRequest, UiCue};

/// The volumes, the mute and the per-sound switches (§1.10). The volumes are gains in
/// \[0, 1\], as `Intent::Volume` gives them; `persist` maps `Settings.audio`'s percent to
/// and from them (Q-188).
// Q-188
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct AudioState {
    /// The master mute ("sound").
    pub muted: bool,
    /// The master gain.
    pub master: f32,
    /// The UI sounds' gain.
    pub ui: f32,
    /// The soundtracks' gain.
    pub soundtrack: f32,
    /// Spatialized soundtracks on the headset (M-5); the desktop plays stereo.
    pub spatial: bool,
    /// The per-sound switches.
    pub sounds: SoundSwitches,
}

impl Default for AudioState {
    /// Q-50's defaults: every volume at 100 %, `spatial` on, every sound on, not muted.
    fn default() -> Self {
        AudioState {
            muted: false,
            master: 1.0,
            ui: 1.0,
            soundtrack: 1.0,
            spatial: true,
            sounds: SoundSwitches::default(),
        }
    }
}

impl AudioState {
    /// The UI sounds' effective gain: 0 when muted, else master × UI.
    fn ui_gain(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.master * self.ui
        }
    }

    /// The soundtracks' effective gain: 0 when muted, else master × soundtrack.
    fn soundtrack_gain(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.master * self.soundtrack
        }
    }

    /// Whether `sound`'s switch is on (grab start and end share one).
    fn enabled(&self, sound: UiSound) -> bool {
        let s = &self.sounds;
        match sound {
            UiSound::Hover => s.hover,
            UiSound::Select => s.select,
            UiSound::GrabBegin | UiSound::GrabEnd => s.grab,
            UiSound::SnapTurn => s.snap_turn,
            UiSound::Click => s.click,
            UiSound::Error => s.error,
            UiSound::Confirm => s.confirm,
        }
    }
}

/// A UI sound (§1.10).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiSound {
    /// A soft tick on hover.
    Hover,
    /// Selection.
    Select,
    /// A grab starts.
    GrabBegin,
    /// A grab ends.
    GrabEnd,
    /// A snap turn.
    SnapTurn,
    /// A UI click.
    Click,
    /// An error (two tones).
    Error,
    /// A confirmation ("pose saved").
    Confirm,
}

impl UiSound {
    /// Every UI sound.
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
}

/// Plays a UI sound, under its switch, the UI volume and the mute (§1.10).
#[derive(Message, Event, Clone, Copy, Debug, PartialEq)]
pub struct PlayUiSound(pub UiSound);

/// An instance's soundtrack, synchronized by §1.6: inserted by `audio` on every instance
/// whose `LoadedModel.audio` is `Some`.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Soundtrack {
    /// The audio file.
    pub source: ModelRef,
}

/// A soundtrack's sink in an `AudioOutput` (Q-187).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SinkId(pub u64);

/// The port to the system's audio output (Q-149, Q-187). Gains are linear, in
/// \[0, 1\]; positions are seconds from the sound's start.
// Q-149, Q-187
pub trait AudioOutput: Send + Sync {
    /// `None` when the output plays; the reason when there is no output device (E-AUDIO),
    /// in which case every call below does nothing.
    fn unavailable(&self) -> Option<String>;
    /// Keeps a UI sound's clip: mono samples at `sample_rate` Hz.
    fn load_ui_sound(&self, sound: UiSound, sample_rate: u32, samples: Arc<[f32]>);
    /// Plays a UI sound's clip once at `gain`.
    fn play_ui_sound(&self, sound: UiSound, gain: f32);
    /// Decodes an audio file (`ogg`, `wav`, `flac` or `mp3`) into a new sink, paused at 0
    /// with gain 0; its sink and its duration in seconds. Blocking: `audio` calls it on
    /// the compute task pool.
    fn add_soundtrack(&self, bytes: &[u8]) -> Result<(SinkId, f64), String>;
    /// Stops and forgets a sink.
    fn remove(&self, sink: SinkId);
    /// Moves a sink to `position_s`.
    fn seek(&self, sink: SinkId, position_s: f64);
    /// Pauses a sink.
    fn pause(&self, sink: SinkId);
    /// Resumes a sink.
    fn resume(&self, sink: SinkId);
    /// A sink's position, `None` for an unknown sink.
    fn position(&self, sink: SinkId) -> Option<f64>;
    /// Sets a sink's gain.
    fn set_gain(&self, sink: SinkId, gain: f32);
}

/// The audio output in use. `AppPlugin` inserts the kira adapter unless the resource
/// exists before it builds (Q-149).
#[derive(Resource, Clone)]
pub struct AudioOutputRes(pub Arc<dyn AudioOutput>);

/// One sink of `StubAudioOutput`, as the tests read it.
#[derive(Clone, Debug, PartialEq)]
pub struct StubSink {
    /// The position, seconds.
    pub position: f64,
    /// Whether it plays.
    pub playing: bool,
    /// Its gain.
    pub gain: f32,
    /// The size of the file it was made from.
    pub bytes: usize,
}

#[derive(Default)]
struct StubInner {
    next: u64,
    sinks: HashMap<SinkId, StubSink>,
    loaded: Vec<UiSound>,
    played: Vec<(UiSound, f32)>,
}

/// An `AudioOutput` without a device, for the tests (§3.3's "stub audio output", Q-149):
/// it records the UI sounds loaded and played and keeps each sink's state. It has no
/// clock: `advance` moves the playing sinks by a test's Δt. Any non-empty file decodes,
/// with the duration given at construction; an empty one fails to decode.
pub struct StubAudioOutput {
    inner: Mutex<StubInner>,
    duration_s: f64,
    unavailable: Option<String>,
}

impl Default for StubAudioOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl StubAudioOutput {
    /// A stub whose soundtracks last an hour.
    pub fn new() -> StubAudioOutput {
        Self::with_duration(3600.0)
    }

    /// A stub whose soundtracks last `duration_s`.
    pub fn with_duration(duration_s: f64) -> StubAudioOutput {
        StubAudioOutput {
            inner: Mutex::new(StubInner::default()),
            duration_s,
            unavailable: None,
        }
    }

    /// A stub that reports no output device.
    pub fn without_device() -> StubAudioOutput {
        StubAudioOutput {
            unavailable: Some("no output device".to_owned()),
            ..Self::new()
        }
    }

    fn lock(&self) -> MutexGuard<'_, StubInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Moves every playing sink by `dt_s`, stopping it at its end.
    pub fn advance(&self, dt_s: f64) {
        let end = self.duration_s;
        for s in self.lock().sinks.values_mut().filter(|s| s.playing) {
            s.position += dt_s;
            if s.position >= end {
                s.position = end;
                s.playing = false;
            }
        }
    }

    /// Every sink, by id.
    pub fn sinks(&self) -> HashMap<SinkId, StubSink> {
        self.lock().sinks.clone()
    }

    /// The UI sounds loaded, in order.
    pub fn loaded(&self) -> Vec<UiSound> {
        self.lock().loaded.clone()
    }

    /// The UI sounds played with their gains, in order, drained.
    pub fn take_played(&self) -> Vec<(UiSound, f32)> {
        std::mem::take(&mut self.lock().played)
    }
}

impl AudioOutput for StubAudioOutput {
    fn unavailable(&self) -> Option<String> {
        self.unavailable.clone()
    }

    fn load_ui_sound(&self, sound: UiSound, _: u32, _: Arc<[f32]>) {
        if self.unavailable.is_none() {
            self.lock().loaded.push(sound);
        }
    }

    fn play_ui_sound(&self, sound: UiSound, gain: f32) {
        if self.unavailable.is_none() {
            self.lock().played.push((sound, gain));
        }
    }

    fn add_soundtrack(&self, bytes: &[u8]) -> Result<(SinkId, f64), String> {
        if let Some(why) = &self.unavailable {
            return Err(why.clone());
        }
        if bytes.is_empty() {
            return Err("an empty file".to_owned());
        }
        let mut inner = self.lock();
        let id = SinkId(inner.next);
        inner.next += 1;
        inner.sinks.insert(
            id,
            StubSink {
                position: 0.0,
                playing: false,
                gain: 0.0,
                bytes: bytes.len(),
            },
        );
        Ok((id, self.duration_s))
    }

    fn remove(&self, sink: SinkId) {
        self.lock().sinks.remove(&sink);
    }

    fn seek(&self, sink: SinkId, position_s: f64) {
        if let Some(s) = self.lock().sinks.get_mut(&sink) {
            s.position = position_s;
        }
    }

    fn pause(&self, sink: SinkId) {
        if let Some(s) = self.lock().sinks.get_mut(&sink) {
            s.playing = false;
        }
    }

    fn resume(&self, sink: SinkId) {
        if let Some(s) = self.lock().sinks.get_mut(&sink) {
            s.playing = true;
        }
    }

    fn position(&self, sink: SinkId) -> Option<f64> {
        self.lock().sinks.get(&sink).map(|s| s.position)
    }

    fn set_gain(&self, sink: SinkId, gain: f32) {
        if let Some(s) = self.lock().sinks.get_mut(&sink) {
            s.gain = gain;
        }
    }
}

/// A failure `ui` raises as a toast: `E-AUDIO: <user text>` (§1.15) and the instance it
/// concerns, `None` for the output device (Q-189).
// Q-189
#[derive(Message, Clone, Debug, PartialEq)]
pub(crate) struct AudioFailed {
    pub(crate) instance: Option<Entity>,
    pub(crate) text: String,
}

/// §1.15's E-AUDIO line.
const E_AUDIO: &str = "E-AUDIO: Could not play audio file";

/// How far a playing sink may drift from the clock before it is moved, seconds
/// (Q-190).
// Q-190
const DRIFT_S: f64 = 0.05;

/// The UI clips' sample rate, Hz.
const RATE: u32 = 48_000;

/// What `audio` knows of a soundtrack's sink.
enum Slot {
    /// Being read and decoded.
    Decoding(ModelRef, Task<Result<(SinkId, f64), String>>),
    /// In the output: its sink, duration and whether `audio` last left it playing.
    Ready {
        source: ModelRef,
        sink: SinkId,
        duration_s: f64,
        playing: bool,
    },
    /// Failed (E-AUDIO): the instance plays no sound.
    Failed(ModelRef),
}

impl Slot {
    fn source(&self) -> &ModelRef {
        match self {
            Slot::Decoding(s, _) | Slot::Ready { source: s, .. } | Slot::Failed(s) => s,
        }
    }
}

/// The soundtracks' slots by instance.
#[derive(Resource, Default)]
struct Slots(HashMap<Entity, Slot>);

/// The audio systems, added by `AppPlugin` after the playback plugin.
pub(crate) struct AudioPlugin;

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<AudioOutputRes>() {
            let output = default_output();
            app.insert_resource(AudioOutputRes(output));
        }
        // §1.7 Panels (D-21), §1.10: the Scene tab's audio, the Settings tab's `[audio]`.
        {
            let mut hooks = app
                .init_resource::<DockHooks>()
                .world_mut()
                .resource_mut::<DockHooks>();
            hooks.add(Tab::Scene, scene_section);
            hooks.add(Tab::Settings, settings_section);
        }
        app.add_message::<PlayUiSound>()
            .add_message::<AudioFailed>()
            .add_message::<UiCue>()
            .add_message::<ToastRequest>()
            .init_resource::<AudioState>()
            .init_resource::<Scrubbing>()
            .init_resource::<Slots>()
            .add_systems(Startup, start)
            .add_systems(
                Update,
                (ui_bridge, run_intents, play_ui_sounds, attach, decode).chain(),
            )
            // After playback's `advance` (in `Update`): the clock of this frame; after
            // egui's pass: this frame's scrub (Q-246).
            .add_systems(
                PostUpdate,
                (apply_gains, follow_clock)
                    .chain()
                    .after(bevy_egui::EguiPostUpdateSet::EndPass),
            );
    }
}

/// A gain in \[0, 1\] edited as §1.10's percent.
fn percent(ui: &mut bevy_egui::egui::Ui, label: &str, gain: &mut f32) -> bool {
    let mut p = *gain * 100.0;
    let changed = ui
        .add(
            bevy_egui::egui::Slider::new(&mut p, 0.0..=100.0)
                .text(label)
                .suffix(" %"),
        )
        .changed();
    if changed {
        *gain = p / 100.0;
    }
    changed
}

/// Edits a copy of `AudioState` and writes it back only when it changed.
fn edit_state(world: &mut World, draw: impl FnOnce(&mut AudioState)) {
    let Some(a) = world.get_resource::<AudioState>() else {
        return;
    };
    let mut next = a.clone();
    draw(&mut next);
    if next != *world.resource::<AudioState>() {
        *world.resource_mut::<AudioState>() = next;
    }
}

// Q-218
/// The Scene tab's audio (§1.7 Panels, §1.10): the mute, the soundtracks' volume and each
/// instance's soundtrack.
fn scene_section(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    ui.strong("Audio");
    edit_state(world, |a| {
        ui.checkbox(&mut a.muted, "mute");
        percent(ui, "soundtrack", &mut a.soundtrack);
    });
    let mut q = world.query::<(&LoadedModel, &Soundtrack)>();
    for (m, t) in q.iter(world) {
        let file = match &t.source {
            ModelRef::File(p) => p.file_name().map_or_else(
                || p.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
            ModelRef::Http(u) => u.to_string(),
            ModelRef::Document(d) => d.clone(),
        };
        ui.label(format!("{}: {file}", m.name));
    }
}

// Q-218
/// The Settings tab's `[audio]` table (§1.10, §1.11), in percent (Q-188).
fn settings_section(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    bevy_egui::egui::CollapsingHeader::new("[audio]").show(ui, |ui| {
        edit_state(world, |a| {
            ui.checkbox(&mut a.muted, "muted");
            percent(ui, "master", &mut a.master);
            percent(ui, "ui", &mut a.ui);
            percent(ui, "soundtrack", &mut a.soundtrack);
            ui.checkbox(&mut a.spatial, "spatial");
            let s = &mut a.sounds;
            for (on, label) in [
                (&mut s.hover, "sounds.hover"),
                (&mut s.select, "sounds.select"),
                (&mut s.grab, "sounds.grab"),
                (&mut s.snap_turn, "sounds.snap_turn"),
                (&mut s.click, "sounds.click"),
                (&mut s.error, "sounds.error"),
                (&mut s.confirm, "sounds.confirm"),
            ] {
                ui.checkbox(on, label);
            }
        });
    });
}

/// ui's sound cues become `PlayUiSound`, and E-AUDIO becomes ui's toast: `ui` comes
/// before `audio` in §2.1's order, so the mapping lives here (PRD #275 "Dependency
/// direction").
// Q-189
fn ui_bridge(
    mut cues: MessageReader<UiCue>,
    mut failed: MessageReader<AudioFailed>,
    mut sounds: MessageWriter<PlayUiSound>,
    mut toasts: MessageWriter<ToastRequest>,
) {
    for c in cues.read() {
        sounds.write(PlayUiSound(match c {
            UiCue::Hover => UiSound::Hover,
            UiCue::Select => UiSound::Select,
            UiCue::Click => UiSound::Click,
            UiCue::Error => UiSound::Error,
            UiCue::Confirm => UiSound::Confirm,
        }));
    }
    for f in failed.read() {
        toasts.write(ToastRequest::error(f.text.clone()));
    }
}

/// The kira adapter, on the desktop and, since M-5, on Android (PRD #452's spike check 1,
/// #454).
fn default_output() -> Arc<dyn AudioOutput> {
    Arc::new(kira_output::KiraOutput::new())
}

/// Reports a missing device and loads the UI clips.
fn start(output: Res<AudioOutputRes>, mut failed: MessageWriter<AudioFailed>) {
    if let Some(why) = output.0.unavailable() {
        warn!("no audio output: {why}");
        failed.write(AudioFailed {
            instance: None,
            text: E_AUDIO.to_owned(),
        });
        return;
    }
    for sound in UiSound::ALL {
        output
            .0
            .load_ui_sound(sound, RATE, synthesize(sound, RATE).into());
    }
}

/// `Intent::Volume` and `Intent::Toggle(Feature::Sound)` (§1.10, Appendix D.1).
fn run_intents(mut reader: MessageReader<Intent>, mut state: ResMut<AudioState>) {
    for intent in reader.read() {
        match *intent {
            Intent::Toggle(Feature::Sound) => state.muted = !state.muted,
            Intent::Volume { channel, value } => {
                if !value.is_finite() {
                    warn!("volume {value} refused");
                    continue;
                }
                let value = value.clamp(0.0, 1.0);
                match channel {
                    VolumeChannel::Master => state.master = value,
                    VolumeChannel::Ui => state.ui = value,
                    VolumeChannel::Soundtrack => state.soundtrack = value,
                }
            }
            _ => {}
        }
    }
}

/// Plays each `PlayUiSound` whose switch is on, at the UI gain; nothing when muted.
fn play_ui_sounds(
    mut reader: MessageReader<PlayUiSound>,
    state: Res<AudioState>,
    output: Res<AudioOutputRes>,
) {
    let gain = state.ui_gain();
    for PlayUiSound(sound) in reader.read() {
        if gain > 0.0 && state.enabled(*sound) {
            output.0.play_ui_sound(*sound, gain);
        }
    }
}

/// Inserts, replaces or removes each instance's `Soundtrack` from its `LoadedModel.audio`.
fn attach(
    models: Query<(Entity, &LoadedModel, Option<&Soundtrack>), Changed<LoadedModel>>,
    mut commands: Commands,
) {
    for (e, m, current) in &models {
        match (&m.audio, current) {
            (Some(source), Some(s)) if s.source == *source => {}
            (Some(source), _) => {
                commands.entity(e).insert(Soundtrack {
                    source: source.clone(),
                });
            }
            (None, Some(_)) => {
                commands.entity(e).remove::<Soundtrack>();
            }
            (None, None) => {}
        }
    }
}

/// Starts a decode for every new soundtrack, collects finished ones and forgets the sinks
/// of removed ones.
fn decode(
    tracks: Query<(Entity, &Soundtrack)>,
    mut removed: RemovedComponents<Soundtrack>,
    mut slots: ResMut<Slots>,
    output: Res<AudioOutputRes>,
    platform: Option<Res<PlatformRes>>,
    mut failed: MessageWriter<AudioFailed>,
) {
    for e in removed.read() {
        if !tracks.contains(e)
            && let Some(Slot::Ready { sink, .. }) = slots.0.remove(&e)
        {
            output.0.remove(sink);
        }
    }
    // A missing device was reported once at startup.
    if output.0.unavailable().is_some() {
        return;
    }
    for (e, track) in &tracks {
        let fresh = slots.0.get(&e).is_none_or(|s| *s.source() != track.source);
        if !fresh {
            continue;
        }
        if let Some(Slot::Ready { sink, .. }) = slots.0.remove(&e) {
            output.0.remove(sink);
        }
        let (source, out) = (track.source.clone(), output.0.clone());
        let provider = platform.as_ref().and_then(|p| p.0.document_provider());
        let task = AsyncComputeTaskPool::get().spawn(async move {
            match open_ref(&source, None, provider.as_deref()) {
                Ok(ModelBytes::Owned(bytes)) => out.add_soundtrack(&bytes),
                Ok(ModelBytes::Checkpoint(dir)) => Err(format!("{} is a directory", dir.display())),
                Err(err) => Err(io_text(&err)),
            }
        });
        slots
            .0
            .insert(e, Slot::Decoding(track.source.clone(), task));
    }
    for (e, slot) in slots.0.iter_mut() {
        let Slot::Decoding(source, task) = slot else {
            continue;
        };
        let Some(result) = block_on(poll_once(task)) else {
            continue;
        };
        let source = source.clone();
        *slot = match result {
            Ok((sink, duration_s)) => Slot::Ready {
                source,
                sink,
                duration_s,
                playing: false,
            },
            Err(why) => {
                warn!("soundtrack {source:?}: {why}");
                failed.write(AudioFailed {
                    instance: Some(*e),
                    text: E_AUDIO.to_owned(),
                });
                Slot::Failed(source)
            }
        };
    }
}

fn io_text(e: &IoError) -> String {
    format!("{}: {}", e.code(), e.user_text())
}

/// The soundtracks' gain on every change of `AudioState` (an intent, or `persist`).
fn apply_gains(state: Res<AudioState>, slots: Res<Slots>, output: Res<AudioOutputRes>) {
    if !state.is_changed() {
        return;
    }
    let gain = state.soundtrack_gain();
    for slot in slots.0.values() {
        if let Slot::Ready { sink, .. } = slot {
            output.0.set_gain(*sink, gain);
        }
    }
}

/// §1.6 Audio, D-22: soundtrack time 0 is τ₀,m, so the position is (τ_m − τ₀,m)/r_m =
/// T − o_m; a sink plays only while the clock plays at speed 1, no scrub is in progress
/// and T − o_m lies within the soundtrack and the model's span, and is moved to the
/// clock's position when it starts again (a scrub's end included), or when it drifts by
/// more than `DRIFT_S` (a one-off seek, a loop's wrap).
// Q-190, Q-246
fn follow_clock(
    playback: Res<Playback>,
    scrubbing: Res<Scrubbing>,
    state: Res<AudioState>,
    models: Query<&LoadedModel>,
    mut slots: ResMut<Slots>,
    output: Res<AudioOutputRes>,
) {
    let clock = &playback.clock;
    let forward = clock.state() == PlayState::Playing && clock.speed() == 1.0 && !scrubbing.0;
    let t = clock.time();
    for (e, slot) in slots.0.iter_mut() {
        let Slot::Ready {
            sink,
            duration_s,
            playing,
            ..
        } = slot
        else {
            continue;
        };
        let Ok(m) = models.get(*e) else {
            continue;
        };
        let p = t - m.time.offset_s;
        let span_s = m.time.span.span() / m.time.rate;
        let audible = forward && p >= 0.0 && p <= span_s.min(*duration_s);
        match (audible, *playing) {
            (true, false) => {
                output.0.seek(*sink, p);
                output.0.set_gain(*sink, state.soundtrack_gain());
                output.0.resume(*sink);
            }
            (true, true) => {
                let drift = output
                    .0
                    .position(*sink)
                    .is_none_or(|q| (q - p).abs() > DRIFT_S);
                if drift {
                    output.0.seek(*sink, p);
                }
            }
            (false, true) => output.0.pause(*sink),
            (false, false) => {}
        }
        *playing = audible;
    }
}

/// A UI sound's clip, mono at `rate` Hz, peak at most 0.5 (Q-191): sine tones under
/// an exponential decay, a pitch sweep for the grabs, two tones for the error and the
/// confirmation.
// Q-191
fn synthesize(sound: UiSound, rate: u32) -> Vec<f32> {
    // (start Hz, end Hz, seconds, amplitude) per tone, played in sequence.
    let tones: &[(f32, f32, f32, f32)] = match sound {
        UiSound::Hover => &[(2000.0, 2000.0, 0.015, 0.15)],
        UiSound::Select => &[(880.0, 880.0, 0.06, 0.4)],
        UiSound::GrabBegin => &[(440.0, 660.0, 0.08, 0.4)],
        UiSound::GrabEnd => &[(660.0, 440.0, 0.08, 0.4)],
        UiSound::SnapTurn => &[(300.0, 300.0, 0.05, 0.4)],
        UiSound::Click => &[(1500.0, 1500.0, 0.025, 0.3)],
        UiSound::Error => &[(400.0, 400.0, 0.12, 0.5), (300.0, 300.0, 0.12, 0.5)],
        UiSound::Confirm => &[(660.0, 660.0, 0.09, 0.4), (990.0, 990.0, 0.09, 0.4)],
    };
    let r = rate as f32;
    let mut out = Vec::new();
    for &(f0, f1, len, amp) in tones {
        let n = (len * r).round() as usize;
        let mut phase = 0.0f32;
        for i in 0..n {
            let u = i as f32 / n as f32;
            let f = f0 + (f1 - f0) * u;
            phase = (phase + TAU * f / r) % TAU;
            // A 2 ms attack, then a decay to e⁻⁵ at the tone's end.
            let attack = (i as f32 / (0.002 * r)).min(1.0);
            out.push(amp * attack * (-5.0 * u).exp() * phase.sin());
        }
    }
    out
}

/// The kira adapter (Q-187): one `kira::AudioManager` on the default (cpal) backend;
/// a missing device leaves it absent and the output unavailable.
// Q-187
mod kira_output {
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::sync::{Arc, Mutex, MutexGuard};
    use std::time::Duration;

    use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle, StaticSoundSettings};
    use kira::{AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Frame, Tween};

    use super::{AudioOutput, SinkId, UiSound};

    /// No ramp: seeks, pauses and gains take effect at once.
    const NOW: Tween = Tween {
        start_time: kira::StartTime::Immediate,
        duration: Duration::ZERO,
        easing: kira::Easing::Linear,
    };

    /// A linear gain as kira's decibels, silence at or below 10⁻³ (−60 dB).
    fn decibels(gain: f32) -> Decibels {
        if gain <= 1e-3 {
            Decibels::SILENCE
        } else {
            Decibels(20.0 * gain.log10())
        }
    }

    struct Inner {
        manager: AudioManager<DefaultBackend>,
        clips: HashMap<UiSound, StaticSoundData>,
        sinks: HashMap<SinkId, StaticSoundHandle>,
        next: u64,
    }

    pub(super) struct KiraOutput {
        inner: Option<Mutex<Inner>>,
        unavailable: Option<String>,
    }

    impl KiraOutput {
        pub(super) fn new() -> KiraOutput {
            match AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()) {
                Ok(manager) => KiraOutput {
                    inner: Some(Mutex::new(Inner {
                        manager,
                        clips: HashMap::new(),
                        sinks: HashMap::new(),
                        next: 0,
                    })),
                    unavailable: None,
                },
                Err(e) => KiraOutput {
                    inner: None,
                    unavailable: Some(e.to_string()),
                },
            }
        }

        fn lock(&self) -> Option<MutexGuard<'_, Inner>> {
            self.inner
                .as_ref()
                .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
        }

        fn with_sink(&self, sink: SinkId, f: impl FnOnce(&mut StaticSoundHandle)) {
            if let Some(mut inner) = self.lock()
                && let Some(h) = inner.sinks.get_mut(&sink)
            {
                f(h);
            }
        }
    }

    impl AudioOutput for KiraOutput {
        fn unavailable(&self) -> Option<String> {
            self.unavailable.clone()
        }

        fn load_ui_sound(&self, sound: UiSound, sample_rate: u32, samples: Arc<[f32]>) {
            if let Some(mut inner) = self.lock() {
                let frames: Arc<[Frame]> = samples.iter().map(|&s| Frame::from_mono(s)).collect();
                let data = StaticSoundData {
                    sample_rate,
                    frames,
                    settings: StaticSoundSettings::default(),
                    slice: None,
                };
                inner.clips.insert(sound, data);
            }
        }

        fn play_ui_sound(&self, sound: UiSound, gain: f32) {
            if let Some(mut inner) = self.lock()
                && let Some(data) = inner.clips.get(&sound).cloned()
                && let Err(e) = inner.manager.play(data.volume(decibels(gain)))
            {
                bevy::log::warn!("UI sound {sound:?}: {e}");
            }
        }

        fn add_soundtrack(&self, bytes: &[u8]) -> Result<(SinkId, f64), String> {
            if let Some(why) = &self.unavailable {
                return Err(why.clone());
            }
            // Decoded here, on the caller's (compute) thread, outside the lock.
            let data = StaticSoundData::from_cursor(Cursor::new(bytes.to_vec()))
                .map_err(|e| e.to_string())?;
            let duration_s = data.duration().as_secs_f64();
            let mut inner = self.lock().ok_or("no output device")?;
            let mut handle = inner
                .manager
                .play(data.volume(Decibels::SILENCE))
                .map_err(|e| e.to_string())?;
            handle.pause(NOW);
            let id = SinkId(inner.next);
            inner.next += 1;
            inner.sinks.insert(id, handle);
            Ok((id, duration_s))
        }

        fn remove(&self, sink: SinkId) {
            if let Some(mut inner) = self.lock()
                && let Some(mut h) = inner.sinks.remove(&sink)
            {
                h.stop(NOW);
            }
        }

        fn seek(&self, sink: SinkId, position_s: f64) {
            self.with_sink(sink, |h| h.seek_to(position_s));
        }

        fn pause(&self, sink: SinkId) {
            self.with_sink(sink, |h| h.pause(NOW));
        }

        fn resume(&self, sink: SinkId) {
            self.with_sink(sink, |h| h.resume(NOW));
        }

        fn position(&self, sink: SinkId) -> Option<f64> {
            self.lock()?.sinks.get(&sink).map(|h| h.position())
        }

        fn set_gain(&self, sink: SinkId, gain: f32) {
            self.with_sink(sink, |h| h.set_volume(decibels(gain), NOW));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Draws a dock section once on a headless egui context (#286).
    fn draw_once(world: &mut World, hook: fn(&mut World, &mut bevy_egui::egui::Ui)) {
        let ctx = bevy_egui::egui::Context::default();
        let mut out = ctx.run_ui(bevy_egui::egui::RawInput::default(), |ui| hook(world, ui));
        out.textures_delta.clear();
    }

    // Q-218
    #[test]
    fn the_audio_sections_draw_without_changing_the_state() {
        let mut world = World::new();
        world.insert_resource(AudioState::default());
        world.clear_trackers();
        draw_once(&mut world, scene_section);
        draw_once(&mut world, settings_section);
        assert!(!world.resource_ref::<AudioState>().is_changed());
    }

    #[test]
    fn every_clip_is_short_bounded_and_finite() {
        for sound in UiSound::ALL {
            let clip = synthesize(sound, RATE);
            assert!(!clip.is_empty(), "{sound:?}");
            assert!(clip.len() <= RATE as usize / 4, "{sound:?} under 0.25 s");
            assert!(
                clip.iter().all(|s| s.is_finite() && s.abs() <= 0.5),
                "{sound:?} bounded"
            );
        }
    }

    #[test]
    fn the_error_is_two_tones() {
        let clip = synthesize(UiSound::Error, RATE);
        let half = clip.len() / 2;
        // Each tone decays to e⁻⁵ of its peak, so the middle is near silence and the
        // second tone starts loud again.
        let near = |a: usize, b: usize| clip[a..b].iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(near(half - 50, half) < 0.01);
        assert!(near(half + 100, half + 1000) > 0.2);
    }

    /// A headless app with the audio plugin alone on `output`.
    fn app(output: Arc<StubAudioOutput>) -> App {
        let mut app = App::new();
        app.add_plugins(bevy::MinimalPlugins)
            .add_message::<Intent>()
            .init_resource::<Playback>()
            .insert_resource(AudioOutputRes(output))
            .add_plugins(AudioPlugin);
        app
    }

    fn failures(app: &mut App) -> Vec<AudioFailed> {
        let mut cursor = app.world().resource::<Messages<AudioFailed>>().get_cursor();
        let messages = app.world().resource::<Messages<AudioFailed>>();
        cursor.read(messages).cloned().collect()
    }

    #[test]
    fn without_a_device_e_audio_and_silence() {
        let out = Arc::new(StubAudioOutput::without_device());
        let mut app = app(out.clone());
        app.update();
        assert_eq!(
            failures(&mut app),
            vec![AudioFailed {
                instance: None,
                text: "E-AUDIO: Could not play audio file".to_owned()
            }]
        );
        app.world_mut().write_message(PlayUiSound(UiSound::Click));
        app.update();
        assert!(out.loaded().is_empty() && out.take_played().is_empty());
    }

    #[test]
    fn an_undecodable_soundtrack_is_e_audio_for_its_instance() {
        let out = Arc::new(StubAudioOutput::new());
        let mut app = app(out.clone());
        let dir = std::env::temp_dir().join(format!("ftgsxr-audio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let empty = dir.join("empty.wav");
        std::fs::write(&empty, b"").unwrap();
        let e = app
            .world_mut()
            .spawn(Soundtrack {
                source: ModelRef::File(empty),
            })
            .id();
        let mut seen = Vec::new();
        for _ in 0..200 {
            app.update();
            seen.extend(failures(&mut app));
            if !seen.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].instance, Some(e));
        assert!(seen[0].text.starts_with("E-AUDIO: "));
        assert!(out.sinks().is_empty());
    }

    /// P1 loaded with a soundtrack beside it on `out`'s stub; the instance's sink.
    fn with_soundtrack(app: &mut App, out: &StubAudioOutput, name: &str) -> SinkId {
        let dir = std::env::temp_dir().join(format!("ftgsxr-audio-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = ftgsxr_testkit::fixtures_dir().join("probes/P1.safetensors");
        let bytes = std::fs::read(&path).unwrap();
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let info = ftgsxr_core::read_info(&bytes[8..8 + n], bytes.len() as u64).unwrap();
        // The stub decodes any non-empty file.
        let wav = dir.join("P1.wav");
        std::fs::write(&wav, b"RIFF").unwrap();
        app.world_mut().spawn(LoadedModel {
            source: ModelRef::File(path),
            identity: "0".repeat(64),
            time: ftgsxr_core::ModelTime::from_span(&info.time),
            info,
            name: "P1".into(),
            reduced: false,
            audio: Some(ModelRef::File(wav)),
            frame_step_s: 1.0 / 30.0,
        });
        for _ in 0..200 {
            app.update();
            if out.sinks().len() == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let _ = std::fs::remove_dir_all(&dir);
        let sinks = out.sinks();
        assert_eq!(sinks.len(), 1, "P1's soundtrack decodes");
        *sinks.keys().next().unwrap()
    }

    // Q-190, Q-246 (#416): while `Scrubbing` holds, a sink is silent with the clock at
    // speed 1, and plays at the clock's position once it ends. `Scrubbing` is
    // crate-private, so this is a private test; `tests/i05_audio.rs` pins the rest of
    // I-5's audio clause.
    #[test]
    fn follow_clock_is_silent_while_a_scrub_is_in_progress() {
        const TOL: f64 = 1e-6;
        let out = Arc::new(StubAudioOutput::new());
        let mut app = app(out.clone());
        let sink = with_soundtrack(&mut app, &out, "scrub");
        {
            let mut p = app.world_mut().resource_mut::<Playback>();
            p.clock.seek(0.3);
            p.clock.set_state(PlayState::Playing);
        }
        app.update();
        let s = out.sinks()[&sink].clone();
        assert!(s.playing, "plays at speed 1");
        // A scrub: seeks over several frames while the clock plays at speed 1.
        app.world_mut().resource_mut::<Scrubbing>().0 = true;
        for t in [0.5, 0.7, 0.9, 1.1] {
            app.world_mut().resource_mut::<Playback>().clock.seek(t);
            app.update();
            assert!(
                !out.sinks()[&sink].playing,
                "silent while scrubbing, T = {t}"
            );
        }
        // The scrub ends: the sink plays from the clock's position.
        app.world_mut().resource_mut::<Scrubbing>().0 = false;
        app.update();
        let s = out.sinks()[&sink].clone();
        let t = app.world().resource::<Playback>().clock.time();
        assert!(s.playing, "plays again after the scrub");
        assert!((s.position - t).abs() < TOL, "{} vs T {t}", s.position);
    }

    #[test]
    fn the_effective_gains() {
        let mut s = AudioState {
            master: 0.5,
            ui: 0.4,
            soundtrack: 0.8,
            ..AudioState::default()
        };
        assert!((s.ui_gain() - 0.2).abs() < 1e-6);
        assert!((s.soundtrack_gain() - 0.4).abs() < 1e-6);
        s.muted = true;
        assert_eq!((s.ui_gain(), s.soundtrack_gain()), (0.0, 0.0));
    }
}
