# The trim's path, from the intent to the cull and the trimmed box

How an instance's trim (E17) gets from the panel to the cull, and how its trimmed box gets back to everything that reads an instance's boxes. It spans three crates: `ftgsxr-app` holds the intent and the panel, `ftgsxr-render` the component, the cull and the bounds task, and `ftgsxr-core` the trim's own type and the trimmed box's computation. The spec items are E17, D-56, §1.4 stages 1, 2 and 8 and the frame skip, §1.5 Instances and Poses, §1.7 Trim, §1.11 and §2.1's trim signatures (revision 45). This doc explains them and never overrides them.

## One place for the trim

The trim lives in one component, `SplatTrim`, on the instance. No `SplatTrim` means off. Everything reads that component: the panel, Save pose, the scene writer, the extraction for the cull, and the bounds task. A trim whose mode is off may still carry shapes (Reset trim leaves the initial shapes there, Q-374). The bounds task and the cull treat such a trim as no trim.

- **Writing it.** Only `manipulate`'s handling of `InstanceAction::SetTrim` and `ResetTrim` writes the component. `SetTrim` ignores a trim that is on with shapes `Trim::is_valid` refuses. A saved pose's trim is inserted when that pose is applied at load. A scene model's trim replaces it, and a scene model without one removes it (Q-374).
- **The panel.** `ui/trim_row.rs` draws the Scene tab's trim row and sends `SetTrim` with the whole edited trim; it never writes the component. The slider ranges and the initial shapes come from the asset's untrimmed model box, never from `SplatBounds`, so they don't move while the user trims.

## The cull

`render.rs` extracts `SplatTrim` into each instance's `ModelDraw.trim`. The trim is part of the prepare key, so a change of trim reruns stages 1–4 (§1.4's frame skip). `passes.rs` writes the trim into the instance block after the centers' radius: the mode in the former pad word, then the shapes as whole vec4 words, as #226 asks of Adreno.

In `cull.wgsl`, both cull entry points run `trim_keeps` on the moved center in model space. It runs after E3's test and after `mark_active`, so a trimmed row still counts as active and a row E3 skips is never counted as trimmed (Q-373). The trimmed count is gated by the same flag as the active and non-finite counts, so an instance drawn as both splats and centers counts each row once. `size` copies it into the statistics block (`pipeline.md`).

`HeadlessRenderer` and the headless commands build their `ModelDraw`s with the default trim, which is off (§2.1).

## The trimmed box

`bounds.rs`' `update_bounds` runs in `SplatPlugin`'s `Update` chain after the loads. A private `BoundsState` remembers, for each instance, the asset and the trim its boxes were asked for:

- A new asset, or a trim turned off, gets the asset's model and content boxes at once.
- A trim turned on spawns `ftgsxr_core::trimmed_bounds` on the async compute task pool over the asset's packed arrays, the uploaded values E17 asks for. The instance keeps its previous boxes until the result arrives.
- A newer trim replaces the `BoundsState`, which drops the earlier task, so a stale result is never applied (E17).
- A result of `None` gives the trim's own box for both boxes, with `kept_none` set; the panel's notice reads that flag.

`trimmed_bounds` (in `layout.rs`) shares E10's counted rows and fallback with `Model::bounds`, and the percentile box with `Model::content_bounds` (`percentile_box`). It clips each counted row's visible interval to the trim: slab clipping for a box, a quadratic for a sphere. Both shapes are convex and the motion is linear, so the kept part is a single interval.

## Who reads `SplatBounds`

Every per-instance use of the model box reads `SplatBounds` (§1.5 Poses):

- the fit and snap to floor, through the fit box;
- picking, on the desktop and in XR, and the ray ends;
- the selection outline and the gizmo;
- orbit, frame selected and recenter;
- the contact shadow and the soundtrack's position.

Only the automatic fit at load still reads the asset's boxes, since an instance without a saved pose has no trim, and so does a thumbnail (§1.5). The selected instance's trim shape is drawn by `manipulate`'s `draw`, in its own color (Q-374).
