# Controllers in `ftgsxr-xr`

How `ftgsxr_xr`'s `tracking` turns the controllers into `XrHands` in M-3: the action set `player`, the three interaction profiles and their suggested bindings, the aim and grip poses, the head, and what M-5 adds. The spec items are §1.8 Controllers, Appendix D.2, Appendix E.1 and §2.1 `XrHands`, `TrackedHand` and `HandSource`; the mapping from D.2's actions to `TrackedHand`'s fields is Q-128's. This doc explains them and never overrides them.

## The action set `player`

One action set, `player`, is created once per instance, holding every action of Appendix D.2: `select`, `grab`, `move`, `turn`, `push_pull`, `menu`, `play_pause`, `reset_pose`, `fit`, `mode`, the `aim` and `grip` poses and the `haptic` output. Every action has the subaction paths of both hands (Q-128). The haptic action is created now, though M-3 sends no pulse, because a set cannot grow after it is attached.

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

The semantic actions are thus carried in raw fields; reading them as intents, with their thresholds, is `xr_input`'s (M-5).

## The frame of the poses

The aim and grip action spaces are spawned per session and despawned with it. They are located in the primary reference space (`session.md`), then the tracking root's transform is applied, so the poses (scale 1) are in the eyes' frame, stage space, the LOCAL fallback's 1.5 m included. `head` is the midpoint of the two located views and the slerp of their orientations at 0.5, in the same frame. `left_handed` is `[locomotion] left_handed` (Q-49) from `SettingsRes`, false without it.

## What M-5 adds

Hand tracking (Appendix D.3, `HandSource::Hand`, the pinch strengths), haptic pulses, the thresholds and intents of `xr_input`, and the use of `XrPlugin.hands`, which M-3 only logs. bevy\_mod\_openxr's hand-tracking plugin is left out of the composition until then (`session.md`).
