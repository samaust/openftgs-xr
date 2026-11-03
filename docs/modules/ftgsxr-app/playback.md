# Playback and audio in `ftgsxr-app`

How `ftgsxr_app::playback` meets §1.6's clock and per-instance time (E9, D-32) and how `ftgsxr_app::audio` meets §1.10's sounds and §1.6's soundtracks (D-22). The spec items are §1.6, §1.10, E9, D-22, D-32 and §2.1 `ftgsxr_app::playback` and `ftgsxr_app::audio`; this doc explains them and never overrides them. Built in M-4 (#280, #281).

## The clock

`Playback` holds the one clock T. A private `PlaybackPlugin`, after `LibraryPlugin`, runs one chain in `Update`: the playback intents are applied, T_len follows the loaded instances, the clock advances by the frame's Δt, and every instance's `SplatTime` is written from its `ModelTime` (E9).

- **Speed and loop.** `SpeedUp` and `SpeedDown` step through a fixed preset list (`speed_preset`); the loop mode last set is kept in a crate-private `LoopSetting`, which `CycleLoop` steps from and `persist` saves (Q-176). The range last set is kept in `RangeSetting`, since `Clock` does not report it (Q-204).
- **Offsets.** Any finite offset is set; a non-finite one is refused (Q-177).
- **Frame steps.** A step moves by Δ_frame of the crate-private `StepTarget` instance when it is timed, else by the smallest Δ_frame over the timed instances, else D-32's default (Q-175). Δ_frame is `LoadedModel.frame_step_s` (Q-174). Nothing fills `StepTarget` from `manipulate`'s `Selected` yet, so a step uses the smallest Δ_frame; this is a gap, reported to the owner, not a decision.
- **Warnings.** A refused speed or offset, and an unknown time unit (once per instance as it loads), are a crate-private `PlaybackWarning` that `ui` turns into a warning toast (Q-217).

Nothing in `playback` calls `audio`: the dependency runs one way (PRD #275's "Dependency direction"), and the soundtrack follows `Playback` and `SplatTime`.

## Audio

A private `AudioPlugin`, after `UiPlugin`, holds `AudioState` (the gains and the mute) and reaches the system's output only through the `AudioOutput` port (Q-149, Q-187): a kira adapter on the desktop, compiled out on Android, and `StubAudioOutput`, which records the calls, in the tests. `AppPlugin` inserts the kira adapter unless `AudioOutputRes` is already there.

- **Gains.** Each in [0, 1]; the effective gain is master times the channel's, 0 when muted; `persist` converts the settings' percent (Q-188).
- **UI sounds.** The clips are synthesized at startup, one per `UiSound` (Q-191), loaded into the output and played on `PlayUiSound`, unless switched off, muted or at zero gain. `ui` asks for them through its `UiCue`, which `audio` maps to its sounds.
- **Soundtracks.** For every instance whose `LoadedModel` names a soundtrack, the file is decoded on the compute pool (`add_soundtrack`) into a paused sink. In `PostUpdate` the gains are applied and each sink follows the clock: it plays only while the clock plays at speed 1 within the soundtrack's span, and is moved to the clock's position when it starts or drifts too far (Q-190).
- **Failures.** E-AUDIO is a crate-private `AudioFailed`: once at startup when the output is unavailable, once per instance whose soundtrack fails, which then plays nothing (Q-189); `audio` itself turns it into `ui`'s `ToastRequest`, since it comes after `ui` in §2.1's order.
- **Dock.** The Scene tab's volumes and the Settings tab's `[audio]` table are `audio`'s own sections, added through `ui`'s `DockHooks` (Q-218).

Stereo only; spatial audio is M-5's.

## Q-n rows relied on

Playback: Q-174, Q-175, Q-176, Q-177, Q-204, Q-217; with Q-19 from earlier. Audio: Q-149, Q-187, Q-188, Q-189, Q-190, Q-191, Q-218; with Q-50 (the defaults).
