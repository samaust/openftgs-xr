# Controllers and hands in `ftgsxr-xr`

How `ftgsxr_xr`'s `tracking` turns the controllers and the tracked hands into `XrHands`: the action set `player`, the three interaction profiles and their suggested bindings, the aim and grip poses and the head (M-3), and the hands (M-5, #459, #463): the permission, the trackers, the aim chain, the per-side ownership, the frequency hint, the drawn meshes and Dv-8's device side. The spec items are §1.8 Controllers and Hands, Appendix D.2, D.3, Appendix E.1, D-25, D-44 and §2.1 `XrHands`, `TrackedHand` and `HandSource`; the mapping from D.2's actions to `TrackedHand`'s fields is Q-128's, a tracked hand's Q-262's. This doc explains them and never overrides them.

## The action set `player`

One action set, `player`, is created once per instance, holding every action of Appendix D.2: `select`, `grab`, `move`, `turn`, `push_pull`, `menu`, `play_pause`, `reset_pose`, `fit`, `mode`, the `aim` and `grip` poses and the `haptic` output. Every action has the subaction paths of both hands (Q-128). The haptic action was created in M-3, before anything pulsed, because a set cannot grow after it is attached; `xr_input`'s pulses go through it (`xr_input.md`).

The set's life follows OpenXR's rules, which bevy\_mod\_openxr's action plugins run on schedules of their own:

- the suggested bindings are sent **once per instance**: the crate runs its binding schedule at every session creation, and suggesting after the first attach is an error;
- the set is **attached at each session creation**, since attachment is per session;
- it is **synced each frame** in PreUpdate, before the crate's sync set, while the session runs.

If the instance is absent or the set cannot be created, `XrHands` stays at its default and nothing else runs.

## The profiles

Appendix D.2's bindings, as written (Q-128), for three profiles:

- `/interaction_profiles/meta/touch_controller_plus`, only with XR\_META\_touch\_controller\_plus enabled (Appendix E.1), and `/interaction_profiles/oculus/touch_controller`: the same Touch column, `move` on the left thumbstick, `turn` on the right thumbstick's x, `push_pull` on the thumbstick's y, `fit` and `mode` on X and Y, `play_pause` and `reset_pose` on A and B, `menu` on the left menu button, `select` and `grab` on the trigger and squeeze values, and the poses and haptic output on both hands;
- `/interaction_profiles/khr/simple_controller`: `select` on select, `menu` on the left hand, the poses and the haptic output on both.

The runtime picks the profile the connected controllers match.

## From actions to `TrackedHand`

`XrHands` is written in PostUpdate, after the crate locates the views; the action spaces were located in PreUpdate. Per hand (Q-128):

- the hand is `Some` while its `aim` action is active for that hand, and `None` otherwise; `tracked` is whether the aim space's position and orientation are both tracked;
- `stick` is `move` on the left, (`turn`, `push_pull`) on the right; `primary` is X (`fit`) on the left and A (`play_pause`) on the right; `secondary` Y (`mode`) and B (`reset_pose`); `menu` is `menu`; `trigger` is `select`'s value and `squeeze` `grab`'s; `pinch` and `pinch_middle` are 0; `source` is `HandSource::Controller`.

The semantic actions are thus carried in raw fields; reading them as intents, with their thresholds, is `xr_input`'s (`xr_input.md`).

## The frame of the poses

The aim and grip action spaces are spawned per session and despawned with it. They are located in the primary reference space (`session.md`), then the tracking root's global transform is applied, so the poses (scale 1) are in the eyes' frame: stage space, through the player rig the tracking root is bound to (`session.md`) and the LOCAL fallback's 1.5 m. A tracked hand's poses take the same path. `head` is the midpoint of the two located views and the slerp of their orientations at 0.5, in the same frame. `left_handed` is `[locomotion] left_handed` (Q-49) from `SettingsRes`, false without it. `XrHands` is written in PostUpdate after the transform propagation, so the rig's motion of this frame is in it; `xr_input` reads it in the next frame's PreUpdate.

`HudData.head_tracked` is written beside the head: true while the session runs with two located views, false once the session is destroyed (Q-274, Q-289), since the crate keeps the last views and drops their tracking flags. Q-274's two hand fields are not built: `HandSource` lives in this crate, which `ftgsxr-app` cannot name, and the row waits for the owner's answer.

D-44's four pulses (`Pulse` and its `HOVER`, `GRAB` and `TICK`) are declared here beside `Actions::pulse`, which plays one on a hand's `haptic` output, so `xr_input`'s haptics and Dv-8 below both take them downward in §2.1's order.

## Hands

`XrPlugin.hands` asks for XR\_EXT\_hand\_tracking, XR\_FB\_hand\_tracking\_aim and XR\_META\_simultaneous\_hands\_and\_controllers through the crate's extension set, and for XR\_META\_hand\_tracking\_frequency\_hint by name, since no pinned openxr release carries it (Appendix E.1, Q-275). Each one the runtime lacks logs E-XR-UNSUPPORTED once at plugin build, as a warning without a toast: hands are not an explicit request (Q-134, Q-161). Monado enables the first and lacks the other three. bevy\_mod\_openxr's hand-tracking plugin stays out of the composition: `tracking` locates the hands itself, so one call gives the joints and the aim state together.

- **The permission.** At each session creation, with `hands` and XR\_EXT\_hand\_tracking enabled, `tracking` asks the platform port for `Permission::HandTracking` the first time and polls the answer each frame (Q-276). The answer is kept for the process: a session re-created after E-XR-LOST or a render-scale change creates its trackers again without asking, a grant that arrives while the session runs creates them in that session, and a denial stays (Q-287). A denial leaves hands off and raises E-XR-PERMISSION's warning and its toast with the feature named, "hands" (Q-261, Q-221). An app without `PlatformRes` (the `xr` tests, which run `XrPlugin` without `AppPlugin`) counts the permission as granted, as `LinuxPlatform` grants every one (Q-287). A crate-private `AskHandsAgain` message asks again after a denial, but nothing outside the tests writes it: the Comfort tab's "ask again" button of Q-276 is in `ftgsxr-app`, which cannot name this crate, and waits for the owner's answer (Q-294).
- **The trackers.** One `HandTracker` per side, created with the session's permission and destroyed with the session. At creation, with XR\_META\_simultaneous\_hands\_and\_controllers enabled, the simultaneous mode is resumed (PRD #452's spike check 5); without it the runtime's own switching between hands and controllers stands. The hint's function is loaded then too (below).
- **The aim chain.** Each frame, one `xrLocateHandJointsEXT` per hand at the crate's frame time (the predicted display time, plus one period when rendering is pipelined) in the primary reference space, with `XrHandTrackingAimStateFB` chained into the locations when XR\_FB\_hand\_tracking\_aim is enabled. A locate failure is a warning once until a locate succeeds again. The openxr crate's own locate sets no `next` chain, so this raw call is the module's, with its SAFETY comment (§2.6).
- **From a hand to `TrackedHand`.** Q-262: `aim` is the aim state's pose, `grip` the wrist joint's, `pinch` and `pinch_middle` the index and middle strengths, `menu` the MENU\_PRESSED flag, `tracked` the VALID flag with the wrist located; the controller fields are zero; `source` is `HandSource::Hand`; the hand is `Some` while its locate reports it active. Without the aim extension (Monado) `aim` is the wrist's pose, the strengths 0 and `tracked` false (Q-285), so the hand casts no ray and begins nothing.
- **Ownership.** Per side, `SideOwnership` decides between the side's controller and hand each frame, before `XrHands` is written: a device alone owns the side; with both active, the one whose input last changed owns it (§1.8 Hands, Q-262). The controller's buttons, trigger, squeeze and stick compare with the previous frame, as does the hand's MENU\_PRESSED edge; the aim motion and the pinch strengths compare with a reference, the device's state when it became active or at its last counted input, so slow drift past the thresholds counts. Becoming active is not input, and when both change in one frame, or appear together, the owner stays, the controller when there was none (Q-286). The owning device's `TrackedHand` is the side's.
- **The frequency hint.** `xrSetHandTrackingFrequencyHintMETA` is loaded through `xrGetInstanceProcAddr` when its extension is enabled, and the hint is sent when it changes: high while a hand owns a side, the default otherwise, and always on the first frame, since the manifest's HIGH meta-data may have started the runtime high (§1.8 Hands, §2.5, Q-275). A failed call is a warning, not retried each frame.

## The meshes

In VR mode `meshes` draws each side's owner, children of the tracking root so their transforms are reference-space poses (§1.8 Controllers and Hands, Q-288): a hand as capsules between its joints with the joint radii, made the first frame both joints of a bone are located; a controller as the player's generated low-polygon box with a ring at the grip pose. Both share one unlit neutral grey material; nothing is drawn in passthrough mode, and no binary asset ships. `tracking` comes before `passthrough` in §2.1's order, so it does not read `Environment`: the meshes follow a crate-private `DevicesDrawn` flag, which `passthrough::devices_drawn` sets from the mode each frame before the meshes are placed (`plugin` orders it).

## Dv-8's device side

Dv-8's clauses need the trackers, the raw locates and the haptic output, which are crate-private, so they live in a private module `tracking::dv8`, built for Android and the host tests and registered on Android only (Q-308). It stays inert until an entity named `ftgsxr-test: dv8` exists, which the quest runner spawns (`ftgsxr-quest/android.md`), and prints the verdict line itself in §3.4's format. Its clauses, as Q-272 reads the row without a person: the joints, counted from `tracking`'s own locates over its window rather than by a second locate, with whether a hand was seen, the aim state's flags, the simultaneous resume and the wrist's rate seen under each hint state (Q-309, which bounds the tracking rate from above); the pulse, a grab start on the active controller's side (a `Grabbed` entity for that hand, despawned after), which `xr_input`'s haptics must turn into D-44's grab pulse with the call succeeding, read from `HapticLog`, "no controller" when none is active; the pinch, a synthetic `XrHands` pinch written after `tracking` that `xr_input` turns into `Select` the next frame. The host test of the pinch clause runs `xr_input`'s plugin, so it sits in `xr_input`'s tests, which may name `tracking::dv8`; `tracking` itself imports nothing of `xr_input`.
