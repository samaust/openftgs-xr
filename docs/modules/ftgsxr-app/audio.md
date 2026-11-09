# Sounds and soundtracks in `ftgsxr-app`

How `ftgsxr_app::audio` meets the volumes, the UI sounds and the soundtracks (§1.10, §1.6 Audio, D-22): the output port and its two adapters, the UI sounds, the soundtracks kept in step with the clock, and their spatialization on the headset. The spec items are those and §2.1's `ftgsxr_app::audio`; this doc explains them and never overrides them. Built in M-4; the panning, Android's output and the spatial mix in M-5 (#453, #454, #457).

## The port

Everything reaches the system's output through the `AudioOutput` trait (Q-149, Q-187): UI clips loaded once and played at a gain, soundtrack sinks decoded from a file and then sought, paused, resumed, read back and given a gain and a stereo pan. `set_panning` is M-5's addition (Q-268): −1 the left speaker only, 0 centered, 1 the right only. `AppPlugin` inserts the kira adapter unless an `AudioOutputRes` exists before it builds, which is how the tests put in theirs.

- **The kira adapter.** One `kira::AudioManager` on its default cpal backend, on the desktop and, since M-5, on the Quest, where cpal links Android's AAudio from NDK platform 32 (`ftgsxr-quest/android.md`, #454). A missing device leaves the output unavailable: every call does nothing and E-AUDIO is raised once. The pan is clamped to kira's range, a non-finite one centered (Q-280).
- **`StubAudioOutput`.** §3.3's stub audio output: no device and no clock (a test's `advance` moves the playing sinks), each sink's position, playing state, gain and pan kept in a `StubSink` for the tests to read; `pan` is 0 until set and records each `set_panning` as given, an unknown sink ignored (Q-280).

## Volumes and UI sounds

`AudioState` holds the mute, the master, UI and soundtrack gains in \[0, 1\], `spatial` and the per-sound switches, Q-50's defaults; `persist` maps the settings' percent to and from them (Q-188), and `Intent::Volume` and `Toggle(Feature::Sound)` change them. A UI sound plays at master × UI, nothing when muted or when its switch is off. The clips are synthesized at startup, short sine tones under a decay, the player's own (Q-191).

`ui` comes before `audio` in §2.1's order, so `ui` raises `UiCue`s and `audio`'s bridge turns them into `PlayUiSound`, and audio's failures into ui's toasts (Q-189). The cues map to hover, select, click, error and confirmation. On the headset the main panel's and the wrist menu's clicks and hovers arrive the same way (`ui.md`). `audio` reads the rest itself (it comes after `manipulate` and `rig`): grab start when a hand joins an instance's `Grabbed` (it appears, or turns two-handed) and grab end when one leaves it (a hand released, or the component gone), the edges the haptics take (Q-270, `ftgsxr-xr/xr_input.md`), and the snap turn on `Intent::SnapTurn`; each at most once a frame, a squeeze on nothing sounding nothing. E18's trim handles sound the same way from `TrimHandles`' edges: the hover when a hand comes to hover a handle, the grab start and end when a hand starts or ends a drag (Q-379).

## Soundtracks

An instance whose `LoadedModel` names a soundtrack gets a `Soundtrack`, decoded on the compute task pool into a sink; a failure is E-AUDIO's toast for that instance. Each frame in PostUpdate, after playback's clock and egui's pass, the sinks follow §1.6 and D-22: soundtrack time 0 is τ₀,m, a sink plays only while the clock plays forward at speed 1 with no scrub in progress and its position inside the soundtrack and the model's span, and it is moved to the clock when it starts again or drifts past Q-190's tolerance (Q-246 for the scrub). The plain gain is master × soundtrack, reapplied on each change of `AudioState`.

## Spatialization

`spatialize` runs last in that chain, after the transform propagation, so it reads this frame's poses (§1.10 Soundtracks, Q-268, Q-281). With the `Headset` marker (Q-266) and `spatial` on, every ready sink is set each frame:

- **The listener** is the midpoint of the world-space translations of every active `SplatCamera` (the two eyes on the headset), its forward the sum of their forwards (Q-281).
- **The emitter** is the instance's `GlobalTransform` applied to B\_m's center (`SplatBounds.aabb`), so it follows the content group and the instance's pose.
- **The mix.** The gain is the plain gain times min(1, 1/d) over the 3D distance d; the pan is the sine of the emitter's horizontal azimuth from the listener's forward, positive right, clamped to \[−1, 1\]: both the forward and the offset are projected on the floor and normalized, and the pan is 0 when either projection is zero (an emitter straight above or below, a forward straight up or down) (Q-268, Q-281).

Without a camera, or for an instance without its boxes yet (`SplatBounds`, trimmed while a trim is on, E17), the sink gets pan 0 and the plain gain. Without the marker (the desktop, `play --xr`) or with `spatial` off the soundtracks play stereo: `spatialize` writes the stereo values once on leaving the spatial mode and nothing otherwise, the plain gain being `apply_gains`' and `follow_clock`'s.

## Q-n rows relied on

`audio`: Q-149, Q-187, Q-270, Q-188, Q-189, Q-190, Q-191, Q-218, Q-246, Q-266, Q-268, Q-280, Q-281; with Q-50 (the defaults).
