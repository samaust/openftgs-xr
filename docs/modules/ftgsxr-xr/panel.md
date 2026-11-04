# The panel quads in `ftgsxr-xr`

How `ftgsxr_xr::panel` places the main panel and the wrist menu on the headset: the pure placement functions, the quads and their life across sessions, the summon and hide, the move by the title bar and the two-hand resize, and the wrist menu on the palm. Built in M-5 (#462). The spec items are §1.8 User interface, Rays, selection and grabbing, and Comfort and accessibility, E11, D-21 and §2.1's `panel` entry (`MainPanel`, `PanelFollow`, `WristMenu`); this doc explains them and never overrides them. What the quads show is `ftgsxr-app/ui.md`'s subject, the rays that reach them `xr_input.md`'s.

## The placement functions

§2.1 gives `panel` no functions; Q-265 adds three pure ones, which U-37 tests (`cpu`):

- `summon_pose(head)`: the quad's center ahead of the head along its horizontal forward and below the eyes by §1.8's distances, upright, its +z toward the head. The horizontal forward is the fit's (Q-17), so a head looking straight up or down still has one.
- `wrist_menu_pose(grip, head)`: Q-265's offset from the grip along the palm's outward normal, the quad facing the head's position and upright (+y stage up made perpendicular to the facing; +x stage x when the facing is vertical).
- `wrist_menu_visible(grip, head)`: whether the palm's outward normal, dotted with the unit vector from the grip to the head, exceeds Q-265's threshold.

The two wrist functions take a *palm frame*, not a raw grip: a pose at the grip whose +x is the palm's outward normal, since that normal is a different grip axis per hand and per source (Q-305). The systems build it with `palm_frame`: the left controller's grip as it is, the right controller's turned 180° about its y, and a tracked hand's wrist pose turned −90° about its z on either side.

## The quads

`PanelPlugin` follows `xr_input` in the XR tail (`session.md`). With the `Headset` marker (Q-266), the first `XrSessionCreated` spawns both quads, each a unit `Rectangle` scaled to its size with an unlit `StandardMaterial` showing a new render-target image, its texture's top-left at the quad's top-left (Q-210), and both hidden (Q-307):

- the main panel: `WorldPanel`, `MainPanel { follow: Summon }` and `PanelQuad` at §1.8's default size and resolution;
- the wrist menu: `WorldPanel`, `WristMenu` and `PanelQuad` at Q-265's size and resolution.

It then sets `UiSurface::Panel` to the main panel's entity and sizes, which turns on `ui`'s panel mode. A later session keeps the quads and the surface, so a session re-created after E-XR-LOST or a render-scale change (Q-124) changes nothing here. Without the marker (`play --xr`) nothing is spawned and the dock stays. The quads are top-level entities, world-locked, not children of `PlayerRig`: locomotion moves the user past them. `PanelQuad` is declared in `xr_input`, which precedes `panel` in §2.1's order and needs the sizes for its ray test; `panel` keeps it and `UiSurface::Panel` in step with the quad.

## Summon and hide

`Toggle(Feature::Panel)` (the menu button, the system gesture or the wrist menu's panel button) shows a hidden main panel and hides a visible one, dropping any grab on it (§1.8). `PanelFollow` says where a hidden panel reappears (Q-307): `Summon` at `summon_pose` of `XrHands.head`, `WorldLocked` where it was left. The head is the views' head, never a camera. The panel spawns as `Summon` and nothing changes it, so every summon follows the head; between summons it stays where it is.

## Move and resize

The main panel is grabbed through `PanelPointer`'s grab presses, which `xr_input` sends while a ray is on the quad (Q-306):

- a grab press on the title bar, Q-306's top strip of the texture, while the panel is visible and the hand tracked, begins E11's one-hand grab at the hand's aim pose; the panel then follows that hand with E11's pose;
- while one hand holds it, the other hand's grab press anywhere on the panel makes it a two-hand grab, E11 with "keep upright", whose scale is the panel's width, kept in §1.8's range, the height following the 4:3 aspect;
- a `GrabEnd` of either hand returns to a one-hand grab by the other, a release of the one hand frees it; the panel follows only tracked hands.

Each hand that joins or leaves the grab sends `xr_input`'s crate-private `PanelGrabEdge`, which the haptics turn into D-44's grab pulse (Q-T452.4).

During a two-hand grab the quad scales live: the transform's scale, `PanelQuad.size_m` and `UiSurface::Panel.size_m` follow each frame, each written only when it changes. When the two-hand grab ends, the image is re-created at the new width's resolution (Q-306's pixels per metre, rounded), and `WorldPanel.image`, the material, `PanelQuad.size_px` and `UiSurface::Panel.size_px` are updated, so `ui` gets a new panel camera for the new image (`ftgsxr-app/ui.md`). A hand's pinch is `select` and `grab` at once (Appendix D.3), so on the title bar `xr_input` withholds the pinch's select from the `PanelPointer`: the pinch moves the panel and clicks no tab (Q-T452.6). The strip's extent is `xr_input`'s `TITLE_FRACTION`, and `PanelQuad.title_bar` marks the main panel's quad (the wrist menu has none).

## The wrist menu

Each frame `place_wrist` reads the left hand's `TrackedHand`, the right's in left-handed mode (§1.8 Comfort), builds its palm frame and, while the hand is tracked and `wrist_menu_visible` holds, shows the quad at `wrist_menu_pose`; otherwise the quad is hidden. Its buttons and their intents are `ui`'s (Q-291).
