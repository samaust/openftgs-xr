# The splat draw and the headless renderer in `ftgsxr-render`

How `ftgsxr_render::gpu` turns a view's sorted records into an image: stage 5 of §1.4's per-frame pipeline (the splat part of `SplatPasses::encode_draw`, `splat.wgsl`) and `HeadlessRenderer`, the adapter §1.7's `render` command and the conformance tests R-1, R-2 and R-19 render through. The spec items are §1.4 stage 5 and its paragraphs on several models, stereo, depth and occlusion and backgrounds, §1.7 `render`, §1.12's bounded deviations, E6, E7, E13, D-3, D-5, D-6, D-7, D-35, D-39, D-45 and D-48; this doc explains them and never overrides them.

## The splat draw

`encode_draw` issues one indirect instanced draw of four-vertex strips into the caller's render pass. Its arguments are the first four counter words, which stage 3 completes from the clamped visible count, so the GPU sizes the draw as it sizes the sort (D-34). Instance i draws the record whose slot the sort left at position i of the first payload buffer (`sort.md`), so the records arrive far to near (E8) and the hardware blend `One, OneMinusSrcAlpha` on color and alpha accumulates E7's over operator into the layer (D-3, D-48). The caller clears the layer to (0, 0, 0, 0) before the pass.

The vertex stage reads the per-view record (the cull's layout, `cull.wgsl`) and places the quad at μ′ ± r in the camera's pixels, row 0 at the top of the image, so the read back needs no flip. The fragment stage evaluates E7 at the pixel's sample point: the negative-form skip, then α = min(α_max, σ_eff·e^(−form/2)) with the α_skip skip, then E13's weight, the same order as `ftgsxr_core::alpha_at`. The record carries the premultiplied color c·σ_eff with σ_eff as its fourth half float, so the fragment writes the record's color scaled by α/σ_eff, which is c·α.

Two inputs the draw needs are not in `encode_draw`'s arguments (Q-87):

- **E7's constants per model** (D-39). `encode_prepare` writes α_max, α_skip and the pixel center offset of each instance's model into a small buffer the frame buffers keep, and the fragment stage reads it through the record's instance word. Several models with different conventions in one view therefore keep each model's constants, while still sharing one record buffer and one sort (D-5).
- **The depth mapping.** With a depth format the splats test against the pass's opaque depth and write none (§1.4 Depth and occlusion); the record's camera depth is mapped as reverse Z with an infinite far plane at the engine's near plane, the provisional convention the Bevy adapter is to check.

E13's weight is the WGSL mirror of `ftgsxr_core::occlusion_weight` (Q-63), including the hard step at δ ≤ 0. Every M-2 tier has occlusion off, so no environment depth is sampled and the weight is 1; the `occlusion` argument is read from M-6 on (R-12). In shared-center mode the draw records nothing yet: the per-eye projection of stage-space records is M-8's (R-9). The pass belongs to the caller, so `encode_draw` writes no timestamps; whoever begins the pass gives it the draw's two timestamps of Q-81.

## The headless renderer

`HeadlessRenderer` owns its device (Q-74), the passes of its tier compiled for a 16-bit float layer and no depth (D-35), the uploaded models and one set of frame buffers. `load` applies §1.2's load options with the request's degree as the SH cap (Q-86: a degree above L keeps L; the `render` command checks the request) and keeps the packed arrays, so that a lost device can be rebuilt.

`render` draws one view per call, sorted per view (§1.4 Stereo), every instance at its pose as splats at its loaded degree. In the `conformance` tier the capacity is the total N of the call's instances, so a conformance render never overflows (Q-86); the frame buffers are remade only when that total changes. One command buffer runs stages 1–5 and copies the layer into a staging buffer, whose rows are padded to wgpu's copy alignment. The CPU then composites the layer over the request's background, C + (1 − A)·c_bg per channel with alpha A (E7), unclamped (§1.7), and refuses a non-finite value as `RenderError::NonFinite`, which the command turns into exit 3.

A lost device is noticed through the device-lost callback, a failed poll or a failed mapping. The renderer then makes one recovery attempt in its lifetime (E-GPU-LOST): a new device and passes, every loaded model re-uploaded from its packed arrays, the render retried. A second loss is `RenderError::DeviceLost`.

## How the conformance tests use it

R-1 (`tests/r01_probes.rs`) and R-2 (`tests/r02_additional_probes.rs`) share `tests/probe_support`. For each committed probe they load the model at the request's degree, render every item at its timestamp from its camera over the request's background at the identity pose, and compare each item with openftgs's reference render under E15's tier. Each probe's minimum PSNR goes to standard error, which is how R-1's report is given. R-2's white-background probe shows that only the request's background reaches the composite, never the model's `render_conventions` background.

R-19 (`tests/r19_transmittance_bound.rs`) renders the transmittance probe of Q-10 and checks, pixel by pixel, the bound E7 states for D-7, with T taken from the reference's alpha and c the far primitive's color, plus one half-float step of the values compared (Q-88): the layer (D-35) and the record's color and α (§1.4 stage 2) are binary16, whose step near 1 is larger than the bound where T is small.
