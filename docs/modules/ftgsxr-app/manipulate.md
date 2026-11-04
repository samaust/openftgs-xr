# Manipulation and the rig in `ftgsxr-app`

How `ftgsxr_app::manipulate` meets picking, selection, grabs and the pose store (E10, E11, §1.5 Poses, §1.7 Selection and manipulation, §1.8 Rays, selection and grabbing, D-11, D-42, D-43), and how `ftgsxr_app::rig` meets the player rig, the content group, locomotion and the desktop camera (E12, §1.7 Cameras, §1.8 Locomotion, §1.14 Frames, D-17, D-44). The spec items are those and §2.1's entries for both modules; this doc explains them and never overrides them. Built in M-4 (#282, #283).

## `manipulate`

A private `ManipulatePlugin`, after `PlaybackPlugin`. One system reads the frame's intents in order and keeps each hand's last ray and pose, so a `Select` picks from that hand's ray at its point in the order (Q-182).

- **Hover and pick.** After the intents, each hand's last ray is picked over every instance with a box, hidden ones included, with `pick_instance`; an instance hit by two hands carries the nearer hit (`Hovered`, Q-182).
- **Grabs.** One- and two-hand grabs follow E10 and E11. Push and pull move along the hand's ray in proportion to the hit distance at the grab's start; any pose change outside the grab (push and pull, nudge, turn, fit, reset, snap to floor) begins the grab again from the new pose (Q-183). `Turn` rotates about stage +y through the instance's origin; `Nudge` moves in stage space (Q-184).
- **Fit and presets.** When `LoadedModel` appears without a saved pose, the up-axis preset comes from the first `ftgs-scene` manifest beside a local model, else Y-up, and a checkpoint's spatial scale gives L_f (Q-181). The fit's head on the desktop is the lowest-order active camera, the identity pose with none (Q-180).
- **Gizmo.** The desktop's gizmo handles sit at the posed box's center along the stage axes, sized from the viewer's distance; only the mouse uses them, and a handle hit on `Select` starts a drag instead of a pick (Q-186). The hovered and selected boxes and the gizmo are drawn with Bevy's gizmo lines (§1.4 Selection and debug views).
- **Frames.** An instance's `Transform` is its pose relative to its parent, `ContentGroup`; its stage pose is that pose followed by the group's transform (§1.14). Picking, hover, grabs (keep upright included, about stage +y, Q-246), push and pull, turn, nudge, fit, snap to floor, the gizmo and the drawn boxes all work on the stage pose, since the rays, the head and the floor are in stage space, and write the result back relative to the group through the group's inverse (#419).
- **Poses.** "Save pose" stores the pose relative to the content group, and "Reset pose" and a saved pose applied on load write it back as it is (Q-245). This module changes `Poses` in memory only; `persist` writes `poses.json` (Q-205).
- **Defaults.** `ManipulateSettings::default()` and an empty `Poses`; `AppPlugin` inserts each when absent, so `persist` or a caller may insert them first (Q-185).

## `rig`

A private `RigPlugin`, after `ManipulatePlugin`. `PlayerRig` and `ContentGroup` are spawned in every app; the desktop camera, a child of the rig, and the floor grid, an opaque unlit line mesh at y = 0, only where a primary window exists (Q-194).

- **Locomotion.** `apply_locomotion` is a pure function of the rig's and the head's stage poses and one intent: smooth movement in the head's yaw frame, smooth or snap turns about the vertical through the head (Q-195). `stick_response` is E12's response curve on a raw stick value, pure.
- **The desktop camera.** A crate-private `CameraController` holds the camera mode, the orbit and nine bookmark slots; it reads `Intent::Camera`, with the units and limits of Q-196. `desktop_input` builds the intents against those units, and `persist` saves the mode, the orbit and the bookmarks (Q-206).
- **Jumps.** A crate-private `CameraJump` moves the fly camera to a training camera's pose and field of view (Q-219, `ui.md`).

## Q-n rows relied on

`manipulate`: Q-180, Q-181, Q-182, Q-183, Q-184, Q-185, Q-186, Q-245, Q-246; with Q-52 (the pose shapes). `rig`: Q-194, Q-195, Q-196, Q-219; with Q-87 (the camera's depth).
