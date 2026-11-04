# The session in `ftgsxr-xr`

How `ftgsxr_xr` runs the OpenXR session: how `XrPlugin` composes with Bevy's plugins, the reference-space chain, the eye cameras, the frame-loop hooks, the session's states and its recovery, and the sleep and wake that Dv-13 observes (M-3); the rig binding, the re-placement on a reference-space change, and locomotion's comfort options on the headset (M-5, #461). The spec items are §1.8 Session and spaces and Locomotion, §1.14 Frames, §1.15 (E-XR-RUNTIME, E-XR-LOST), §2.1 `XrPlugin`, `XrError` and `PlayerRig`, §2.4 Frame loop hooks, D-14, D-17, D-18, D-24 and E6; this doc explains them and never overrides them. The display settings are `display.md`'s subject, the controllers and hands `tracking.md`'s, the intents `xr_input.md`'s and the panel quads `panel.md`'s.

## Who runs the session

bevy\_mod\_openxr 0.6.0 runs the session: its init, session, events and render plugins create the instance, the session and the swapchain, run the frame loop and spawn one camera per eye. The module adds what the player needs on top and replaces one plugin of the crate, the STAGE-only reference space.

`XrPlugin` must build before Bevy's `RenderPlugin`, which the crate's init plugin replaces, so it is added through `add_xr(plugins, xr)` (Q-120): the group's `RenderPlugin` is disabled, `XrPlugin` goes before it, and a private tail plugin goes at the end, after `PipelinedRenderingPlugin`, which the crate's render plugin looks for. The tail holds the crate's events plugin, then `session`, then the crate's render, camera, action and spatial plugins, and after them `tracking` (which adds to the binding plugin's schedule), `xr_input` (which reads `tracking`'s `XrHands`), `panel` (after `xr_input`, §2.1's module order) and `display` (which uses the events plugin and `session`'s re-creation), in that order. The crate's hand-tracking and overlay plugins are left out: `tracking` locates the hands itself. `ftgsxr-quest`, `play --xr` (M-4) and the `xr` tests build their groups this way, with or without winit.

`XrPlugin` itself:

- asks for the extensions of Appendix E.1 that the built milestones use, the hand extensions only with `hands` (`tracking.md`); the crate drops those the runtime lacks with a warning;
- configures an OPAQUE session (D-14) with the swapchain format `Rgba8UnormSrgb` (Q-91);
- on Android, adds the instance extension `VK_KHR_get_surface_capabilities2` through the crate's `OxrManualGraphicsConfig`, which the Quest runtime's surface extension requires (Q-139);
- logs E-XR-RUNTIME once when the instance could not be created, as `session` does when the session could not be created: a warning on the desktop, which continues without XR, an error on the Quest, which exits 2 (§1.13, §1.15);
- inserts `Environment` in VR mode, with `passthrough_available` true when XR\_FB\_passthrough was enabled. `passthrough` is only logged until M-6; `hands` is true for `play --xr` and the Quest.

## The vendored crates

Four crates are vendored under `vendor/` and used through `[patch.crates-io]`, each with its changes marked in place, with attribution:

- **bevy\_mod\_openxr 0.6.0** (Q-139, D-30): on Android, the Vulkan instance targets 1.3, the Quest 3 device's version, instead of 1.1, so the features and SPIR-V version wgpu enables are ones the instance allows; and the HAL instance's flags drop wgpu's `VALIDATION_INDIRECT_CALL`, the Android rule of D-30 kept in this one place.
- **wgpu-hal 29.0.4** (D-30): a public `flags()` getter on the Vulkan instance.
- **wgpu-core 29.0.4** (D-30): `Instance::from_hal_instance` keeps a Vulkan HAL instance's flags instead of taking the defaults, which had made the removal above ineffective. A second change serves the desktop window, not the headset (D-52, #401): `Surface::present` first clears and submits a surface texture that no submission used since its acquire, which Bevy presents while its pipelines compile, so the image reaches the present layout and its acquire semaphore is waited on; R-16 stays at zero messages for windowed `play`. The XR swapchain is bevy\_mod\_openxr's and doesn't go through it.
- **bevy\_render 0.19.1** (D-53, #409): it too serves the desktop window, not the headset. A window surface reconfigured when it is found out of date, or when its window's size or present mode changed, takes the surface's current extent instead of a size the window has already left, so a resize creates the window swapchain at the window's size. The XR session presents no window surface, so its swapchain doesn't go through it.

With the first three, the XR path's device has no indirect-call validation on the Quest, and its indirect draws draw (D-30, `ftgsxr-render/pipeline.md`). The wgpu patches are platform-neutral; the copies are dropped once a wgpu release that Bevy's pin takes carries both the flags fix and D-52's. bevy\_render's change applies on Linux and Android only, and its copy is dropped once a Bevy release that the workspace takes fixes it (D-53).

## The reference-space chain

At each session creation, `session` creates the primary reference space by D-24's chain: the first of the settings' `[xr] reference_space` names (`stage`, `local-floor`, `local`) that the runtime enumerates, LOCAL\_FLOOR only with XR\_EXT\_local\_floor enabled, and LOCAL when none is, even when the list leaves `local` out (Q-49). Under LOCAL the tracking root is raised 1.5 m and a warning says so, so the eyes, the hands and the content share a floor-level frame, which the docs and the HUD call stage space. The settings come from `SettingsRes` when present, else `Settings::default()` (Q-127), which is how I-17 excludes STAGE before M-4. The space is removed in both worlds before the session is destroyed, with the record of which type it was.

## The rig binding

At each session creation `session` makes the crate's tracking root a child of `PlayerRig` (§1.8 Locomotion, §2.1 `PlayerRig`), so E12's motion of the rig, a teleport and the seated offset carry the eyes, whose views the crate composes with the root's global transform, and the hands, which `tracking` locates through the same transform. The crate spawns the root once and keeps it across sessions, so the binding holds through a re-creation; it is checked again at each session (PRD #452's spike check 4, Q-124). Without a rig (an app without `AppPlugin`, as the `xr` tests' plain apps) the root stays unparented. The LOCAL fallback's lift stays on the root itself, under the rig (I-17).

## The re-placement

A ReferenceSpaceChangePending event for the primary space's type (the user recentered the headset) re-places the content group so the models keep their place in the room, not their coordinates (§1.8 Session and spaces, Q-264). With M the tracking root's global pose (the rig and the LOCAL lift, scale dropped) and T the event's `poseInPreviousSpace` projected on its twist about +y (Q-254), the group's new pose is M T⁻¹ M⁻¹ composed with its old one (Q-298), which `replaced_group` computes; with M the identity it is Q-264's T⁻¹ composed with the group. The group's translation and rotation are written; the rig is untouched. An invalid pose changes nothing, an event for another space type is only logged, and the change applies when the event arrives, not at its `changeTime` (Q-298).

## Comfort

`session`'s `comfort` adds locomotion's comfort options (§1.8 Locomotion, Comfort and accessibility):

- **The seated offset.** Each change of the wanted offset is added to `PlayerRig`'s translation y, so E12's motion, the teleport and the recenter, which keep or read the rig's height, carry it. The wanted offset is `[locomotion] height_offset_m` while `Locomotion.enabled` (VR mode) and 0 while it is false (passthrough mode, where the stage must stay the real room's), so switching modes removes and restores it (Q-299). Only the headset applies it: without the `Headset` marker (`play --xr`) the wanted offset is always 0 (Q-T452.9).
- **The vignette.** While a frame's intents move the rig smoothly (a `Locomote` with a move, or with a turn under smooth turn), the vignette ramps up, and back down after, by Q-269's ramps; off with `Locomotion.vignette` false. Each eye gets a child quad ahead of it, past the eye's near plane (Q-87), sized each frame from the eye's located field of view so the radial texture covers Q-269's outer share of each half field of view (Q-300). The texture is generated, black with a smooth radial alpha; the material is unlit and blended, drawn in the transparent pass after the splats.
- **The teleport fade.** In the frame `rig` applies `Intent::Teleport` the rig has already jumped, so the fade starts black and clears linearly over §1.8's 150 ms, on a second black quad in front of the vignette (Q-301). None while locomotion is off, since the teleport moves nothing then (Q-282).
- **Reduced motion.** Under `[locomotion] reduced_motion` both ramps stay at 0.

Each eye renders layer 0 and a layer of its own (1 plus its index), and its two quads are on its own layer only, so neither eye sees the other's quads. The quads exist where the mesh, material and image stores do (the player's app, not every `xr` test); the ramps run either way, and the eyes and their quads go with each session.

## The eye cameras

The crate spawns one camera per eye at every session, with an identity projection matrix, and recomputes it every frame from the located view and its `near`. `session` readies each new eye in two steps:

1. **When it appears** (PreUpdate): the projection's near plane becomes 0.01 m, the engine's (Q-87; the crate's is 0.1), set inside the crate's `XrProjection` so the located matrix is never reset; tonemapping off (D-6); and the camera order set to the eye's index, so the left eye is the adapter's first splat camera (Q-125).
2. **Once located** (Last): the eye becomes a `SplatCamera` with `XrPlugin.sort` (Q-121) and `OutputSpace::LinearOver` over the VR-mode background (Q-49, Q-127); from then on `ftgsxr-app`'s view following keeps its background on `[environment] background`, or a replayed request's (Q-234). It waits until the camera's computed clip mapping, the one the render world extracts, is a perspective one at 0.01 m: the projection alone can be a frame ahead of the computed matrix, and a `SplatCamera` extracted with the identity is warned as "not a perspective projection" (#224). E6's camera of each eye then comes from that matrix and the eye's view transform.

In M-3 both builds pass `SortMode::PerView`: each eye prepares, sorts and draws on its own (D-4 per view). `SharedCenter` is accepted with a warning that it draws nothing until M-8 (R-9), when D-4's default returns (Q-121).

## The frame-loop hooks as built

§2.4's hooks are the crate's: `xr_wait_frame` in `XrFirst`, which paces the app; the frame begun and the images acquired in the render world's `PreRender`; released and the frame ended in `PostRender`; one Bevy camera per eye, each into one array layer of the swapchain. The player's prepare and draw nodes run in each eye's 3D pass (`ftgsxr-render/pipeline.md`). The render-world resource of §2.4 that lets the second eye only draw serves shared-center mode and is not built yet: in per-view mode both eyes prepare (R-9, M-8).

On the Quest no Bevy window exists; winit's runner is set to poll (reactive with no wait), so the frames are paced by xrWaitFrame alone (`ftgsxr-quest/android.md`).

## States, loss and re-creation

`session` logs every `XrStateChanged`. LOSS\_PENDING, which the crate reports as `Exiting { should_restart: true }`, is E-XR-LOST, logged as a warning since §1.15 lists it as recoverable (§1.13), and marks the session for re-creation. The crate destroys a session but never creates one again, so once the old session is destroyed `session` asks for a new one. `display`'s render-scale change uses the same mark (Q-124, `display.md`). `XrError`'s codes and user texts are Q-119's.

## Sleep and wake

The lifecycle Dv-13 checks on the Quest (#229, D-18): when the headset sleeps, Android pauses the activity, and while it is paused the runner runs no update, so the app sees nothing of the sleep at the time. When the headset wakes, the states of the sleep arrive together: the session leaves Running and returns to it. The runner's `dv13` therefore needs only that the session left Running and came back, and prints the states it saw. The host driver puts the headset to sleep with the proximity sensor reported off and Android's sleep key, and wakes it with the wake key and the sensor reported on, so the run holds whether or not someone wears the headset.
