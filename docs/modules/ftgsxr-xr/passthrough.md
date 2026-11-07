# Passthrough in `ftgsxr-xr`

How `ftgsxr_xr::passthrough` meets §1.9's environment modes: the mode switch, the eyes' output per mode, the passthrough style, the passthrough layer in the render world, the contact shadows and Dv-6's device side (M-6, #568, #569). The spec items are §1.9 Environment modes and Floor, shadows and anchors, §1.4 Backgrounds and Depth and occlusion, §1.11 `[environment]`, §1.15 E-XR-UNSUPPORTED, §2.1 `XrPlugin`, `Environment`, `EnvironmentMode` and `PassthroughStyle`, §3.4 Dv-6, R-18, D-14 and Appendix E.1; this doc explains them and never overrides them. The environment depth is `depth.md`'s subject, the anchors `anchors.md`'s and the app-side controls `ftgsxr-app/ui.md`'s and `ftgsxr-app/persist.md`'s.

## What `XrPlugin.passthrough` builds

The switch and the layer are built only with `XrPlugin.passthrough`, which the Quest and `play --xr` set (§2.1). Without it nothing is created and the environment mode stays VR, M-3's behaviour, which the `xr` tests keep except the one that tests passthrough. With it, the module runs wherever the runtime offers XR\_FB\_passthrough: on Monado, which does not, nothing is created and `passthrough_available` stays false. `depth` is built under the same flag, since the Depth API needs passthrough (Appendix E.2). The contact shadows are a plugin of their own (below), added by `XrPlugin`'s tail whatever the flag.

## The switch

The switch runs in the main world while a session exists, in one chain: the toggle, the effective mode and the style, then the eyes (Q-335).

- **One wanted mode.** `[environment] mode` is the wanted mode and the one value `ftgsxr-app` reads; `persist` gates it by the `Headset` marker, so the desktop and `play --xr` stay in VR mode whatever the setting says (§1.4 Backgrounds, `ftgsxr-app/persist.md`).
- **The toggle.** On the headset, `Intent::Toggle(Feature::Passthrough)`, from the wrist menu's button or the Environment tab's mode switch, flips the setting; nothing else consumes it, and without the XR crate the control is absent anyway (Q-335, Q-336).
- **The effective mode.** Each frame `Environment.mode` is written as passthrough when the setting asks for it, the marker exists and `passthrough_available`, VR otherwise; without a session it is VR. The style is written beside it from the three `passthrough_*` settings.
- **A mode the device cannot show.** When the setting asks for passthrough on a headset whose runtime lacks the extension, the setting is written back to `vr` and E-XR-UNSUPPORTED is logged as a warning each time, with its toast once per session (Q-161, Q-335), so the setting never holds a mode the device cannot show. The `xr` test `xr_passthrough_unsupported.rs` runs this clause on Monado beside I-15.
- **`XrFeatures.passthrough`.** `Available` on the headset while a session exists and the runtime offers the extension, `Absent` otherwise (without a session too, as occlusion), so the passthrough controls show only there (Q-336, §1.9 "On the desktop the Environment tab shows the VR-mode controls only").

## The eyes' output per mode

Each eye's `SplatCamera.output` keeps the form `LinearOver` and follows the mode: no background in passthrough mode, the background in VR mode (Q-94, Q-101, Q-234). Only the form is changed: the background itself stays `rig`'s view following, or a replayed request's override, and an eye made later, over the settings' background (Q-127), is caught the next frame. With no background the resolve writes its passthrough form, the depth-gated mesh branch blends the layer over the meshes and the transparent clear stays, so the view target carries the layer's alpha to the compositor (D-14, `ftgsxr-render/pipeline.md`).

## The style

The three controls in \[−1, 1\] are mapped linearly onto the extension's brightness, contrast and saturation, 0 at the extension's neutral values at both ends (Q-337), and sent to the layer at each change. The structure is sent alone in the style's chain, at full opacity and with no edges, since the registry page makes the chained color maps mutually exclusive. A private test pins the map's fixed points.

## The layer

The layer lives in the render world, where bevy\_mod\_openxr keeps its layer list, and the main world's mode and style reach it as an extracted request (`layer`).

- **Owned through the raw table.** At the session's first render frame the passthrough feature and its RECONSTRUCTION layer are created paused, through the raw `fb_passthrough` function table, and the module owns both (`Owned`), destroying the layer then the feature before the session goes. This departs from PRD #565's plan, which used the vendored crate's helper and its layer provider: openxr 0.21.1's `Passthrough::start` calls the pause function instead of the start function, and its `Passthrough` gives no raw handle to start it with, so a feature made through it could never be started again once paused. The crate's `OxrFbPassthroughPlugin` is not used either, since it starts passthrough at creation and submits the layer unconditionally. Owning the feature and the layer through the raw table, and pausing both in VR mode, is the owner's decision during the build (#568, merge c6764d0); `ftgsxr-xr` still turns on the crate's `fb_passthrough` feature, a feature of a §2.3 row.
- **Passthrough mode.** The feature is started, the layer resumed and the module's provider inserted first in the layer list, under the projection layer, which already blends with its texture's alpha. The session stays OPAQUE (D-14).
- **VR mode.** The provider is removed from the list, the layer paused and the feature paused too, so the headset's cameras are off while passthrough is not shown (owner, #568).
- **The record.** The list holds trait objects, so `Record` keeps the address of the provider inserted here, with whether the layer is resumed and the feature started, the style last sent and whether the runtime accepted it, and the calls that failed in the session; Dv-6 reads it (Q-346). A failed pause, resume or style call is counted and logged.

## The contact shadows

`ContactShadowPlugin` (Q-342) is public so that R-18 can reach it without an OpenXR instance (§2.7). For each visible model instance, while `Environment.mode` is passthrough, the `Headset` marker exists and `[environment] contact_shadows` is on, it keeps one shadow entity under the instance: §1.9's disc, read as an ellipse at y = 0 with the posed box's half extents in x and z, its alpha §1.9's opacity inside and falling by smoothstep to 0 over §1.9's feather at the rim, or over its whole radius when that is smaller. Nothing is drawn in VR mode or on the desktop.

- **The pass.** The shadow is drawn in the main opaque pass with depth, in its alpha-mask phase, which Bevy draws after the opaque phase, so the tracked-hand gauntlets' depth (`tracking.md`) is written first and hides it whatever the bin order (Q-364), as premultiplied black, by a small custom material whose blend keeps the target's color and takes the larger alpha, so overlapping footprints join instead of one feather cutting the other's body. The splats then composite over it through the resolve's mesh branch (Q-101), and the compositor blends the view target over the room (D-14). An opaque-pass write is the one order that puts the shadow under the splats (§1.4 Depth and occlusion) while its alpha still reaches the compositor. The shadow's depth hides splats below the floor, which the environment depth would hide too.
- **The geometry.** The footprint is B\_m's eight corners posed by the instance's global transform, the content group's included; the shadow entities are roots, placed after the transforms' propagation, so the poses are the frame's own while the mode and the instances' visibility may be a frame old.
- **The shader.** The material's WGSL is an inline string in `passthrough/shadow.rs`, inserted into the shader store under a fixed handle, not a `.wgsl` file (Q-354): U-35 has every `.wgsl` file referenced by `ftgsxr-render`, and a file in this crate would fail it as written. The string keeps its SPDX line.
- **Where it draws.** The material exists where Bevy's `PbrPlugin` does, in the player's plugin groups; in an app without it the plugin draws nothing.
- **R-18** (`tests/r18_contact_shadow.rs`, `gpu`) adds the plugin on its own: the switch runs only while a session exists, so the test inserts `Environment` in passthrough mode, the `Headset` marker and the settings itself, renders through an image-target `SplatCamera` of `LinearOver` with no background and reads the target's alpha along the footprint.

## Dv-6 on the device

Dv-6's clauses live here, not in `ftgsxr-quest`, since they read `layer`'s crate-private record and the render world's layer list; they stay inert until the quest runner spawns the entity `ftgsxr-test: dv6`, and print the verdict line themselves (Q-308, Q-122, `ftgsxr-quest/android.md`). As Q-346 reads the row without a person: once the session runs and the layer exists, the test flips `[environment] mode` to passthrough and back twice, the second time with a brightness change so a style is sent. It passes when, in passthrough mode, the list holds this module's provider first of two with the layer resumed and the feature started, and in VR mode the projection layer alone with the layer and the feature paused, each mode held for a while with no frame breaking it; the blend mode OPAQUE in every frame; and every pause, resume and style call accepted. PRD #565's spike check 3, a compositor error over the switches, is the host driver's scan of the app's log. Dv-6 passed on the Quest 3 with the feature and the layer paused in VR mode (#568).

## Q-n rows relied on

Q-335, Q-336, Q-337, Q-342, Q-346, Q-354; with Q-94, Q-101, Q-122, Q-127, Q-161, Q-234 and Q-308.
