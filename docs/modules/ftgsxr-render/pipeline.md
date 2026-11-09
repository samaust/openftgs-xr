# The per-frame pipeline in `ftgsxr-render`

How `ftgsxr_render::gpu` runs the eight stages of §1.4's per-frame pipeline for one view: which `SplatPasses` call encodes each stage, which buffers and textures it reads and writes, and how the statistics come back two frames later. The spec items are §1.4 (the per-frame pipeline, several models, stereo, depth and occlusion, backgrounds), E1–E8, E13, E14, D-3–D-6, D-14, D-33–D-35, §2.4 and, for the M-4 section, §1.12, E16 and §1.2 load option 1, and for the M-6 section §1.9 Depth occlusion, D-45, §2.1 `DepthFrame`, `SplatOcclusion` and `OcclusionInput`, and §2.6; the sort itself is `sort.md`'s subject and the headless renderer `conformance.md`'s. This doc explains them and never overrides them.

## The calls and the passes they encode

| Stage | Call | Encoded as | Shader |
| --- | --- | --- | --- |
| 1 Prepare | `encode_prepare` | CPU work, uniform buffers made at mapping | none |
| 2 Cull | `encode_prepare` | the "prepare" compute pass, one dispatch per instance and kind | `cull.wgsl` (`cull_splats`, `cull_centers`) |
| 3 Size | `encode_prepare` | the same compute pass, one workgroup | `cull.wgsl` (`size`) |
| 4 Sort | `encode_prepare` | a compute pass of its own, every dispatch indirect | `sort_count.wgsl`, `sort_scan.wgsl`, `sort_scatter.wgsl` |
| 5 Draw splats | `encode_draw` | in the caller's render pass on the layer | `occlusion.wgsl` prepended to `splat.wgsl` |
| 6 Draw centers | `encode_draw` | the same render pass, after the splats (Q-96) | `centers.wgsl` |
| 7 Resolve | `encode_resolve` | in the caller's render pass on the view target | `resolve.wgsl` |
| 8 Statistics | `encode_stats` | copies on the caller's encoder | none |

The caller owns the encoders and the render passes: it clears the layer to (0, 0, 0, 0), attaches the opaque depth when the passes were made with a depth format, and sets the draw pass's timestamp writes (Q-87). `SplatPasses::new` compiles every pipeline once per (layer format, depth format, tier); the resolve is compiled for one view-target format, `Rgba8UnormSrgb`, with no depth attachment (Q-91). A headless render runs stages 1–5 and reads the layer back instead of resolving (§1.7, `conformance.md`).

## The buffers

One `FrameBuffers` set per view (per-view mode) holds, sized once per tier and capacity C (§1.12, §2.4 rule 4, Q-23):

- **records**: C slots in the tier's record layout. Splat records fill slots from 0; center records follow them, from slot min(splats, C) (Q-78).
- **keys and payloads**, two pairs: the sort's (k̄, slot) pairs, sorted into the first pair (`sort.md`).
- **histograms**: the sort's per-block digit counts (Q-85).
- **counters**: the splat draw's indirect arguments (words 0–3), the centers draw's (words 4–7), the overflow flag, the rows that passed E3's test (Q-79), the non-finite count (Q-118) and the rows trimmed (E17, Q-373).
- **sort dispatch**: stage 3's indirect dispatch arguments for the sort (Q-85).
- **statistics**: the 80-byte block of stage 8 (§1.4, revision 13): the splat and center counts, the overflow flag, C and the count of primitives culled for a non-finite projection (E6) in words 0–4, the count of rows trimmed (E17, revision 45) in word 5, two pad words, then the six 64-bit timestamps from byte 32; and four staging buffers it is copied into (Q-81).
- **draw constants**: one block per instance, written by `encode_prepare` and kept for `encode_draw` (Q-87): E7's constants of the instance's model (D-39), then the centers' colormap, ρ and scales (Q-92, Q-93).

Each `GpuModel` holds its structure-of-arrays buffers, its uniform block of E6 and E7 constants, its temporal index and the 99th percentile of its speeds (Q-93). The cull binds every field's buffer whole except the SH array's, which it binds as up to eight SH ranges of W words each, W·4 bytes being the device's largest storage binding rounded down to its storage offset alignment (at least 64 bytes); the uniform block carries W, and the slots a model doesn't need repeat its last range, so a model whose SH array fits one binding takes the same path (Q-377).

## Stage by stage

**1 Prepare.** On the CPU, per instance: τ\_m − τ₀ in float32 (computed in float64), the pose's rotation, translation and scale, E4's eye in model space, E14's gains with the flag that applies them only when they differ from the identity (Q-80), the dispatch ranges of the temporal index at τ\_m (E3), the SH degree and ρ. Splat dispatches are listed before every centers dispatch (Q-78). The frame block holds the cull views, C and the sort pose. Nothing is read from the GPU (D-34).

**2 Cull.** Reads the model buffers and uniform blocks, the instance block and the frame block; writes records, keys, payloads and counters. A visible splat takes a slot by an atomic increment of the splat count; a slot at or beyond C is dropped and raises the overflow flag. A center takes slot min(splats, C) + j by its own count (Q-78). A row culled in every view with a non-finite value in at least one view's projection (E6), or a μ′ beyond float32's range, decided on x_c before μ′ is computed, adds one to the counters' non-finite word, once per row and instance: an instance drawn as splats and centers counts it in the dispatch that counts its rows (Q-118, Q-79).

**3 Size.** Reads and rewrites the counters: clamps the splat count to C and the center count to what remains, completes both indirect draws (four vertices, first instance 0) and raises the overflow flag when either was clamped. Writes the sort dispatch arguments and the counts, flag, C and non-finite count of the statistics block, its pad and timestamps zeroed.

**4 Sort.** Reads the counters and the sort dispatch arguments; sorts the splats' pairs through the histograms into the first key and payload pair (`sort.md`). Centers are not sorted.

**5 Draw splats.** Reads the records, the first payload buffer (draw order), the draw constants and a view block; one indirect draw from counter words 0–3, blended `One, OneMinusSrcAlpha` into the layer, depth-tested without depth write when a depth format was given (Q-87). With an `OcclusionInput` each fragment's α is weighed by E13 against the environment depth (below; `conformance.md`). In shared-center mode nothing is drawn until the per-eye projection of R-9 (Q-87).

**6 Draw centers.** Reads the records, the counters (bound read-only: instance i draws slot `visible_splats + i`, where `visible_splats` is word 1 after stage 3), the draw constants and the view block; one indirect draw from counter words 4–7. Each disc is the pixels whose sample point lies within ρ camera pixels of the record's μ′, written opaque in the instance's colormap color (Q-92; the time-center map spans \[τ₀, τ₁\], whose τ₁ `GpuArrays` carries, and a single-timestamp model maps every center to its low end, Q-97), with depth written when depth-tested. In `Both` mode a disc replaces its splat's pixels (Q-96).

**7 Resolve.** Reads the layer through a linear sampler at each target pixel's center (Q-95) and a block holding the background. Without a background it writes the layer un-premultiplied, decoded from sRGB and re-premultiplied, blended over the target, so the target keeps the layer's alpha for the compositor (D-14). With one it decodes E7's C + (1 − A)·c\_bg, formed in the file's space, and writes it opaque (Q-94). The sRGB target's encode on store then shows the file's values (D-6, R-11).

**8 Statistics.** Resolves the frame's timestamps into the statistics block, then copies it into the next of four staging buffers, mapped when the encoder's submission completes.

## The two-frame statistics

Stage 8 never waits (P-5). The four staging buffers form a ring: the n-th `encode_stats` (from 0) copies frame n's block into buffer n mod 4 and asks for its mapping on submission. A buffer still waiting for its mapping skips that frame's copy and keeps the frame it holds; a mapped buffer nobody returned is unmapped and reused. `FrameStats::poll` polls the device without blocking and returns the newest mapped block of frame n − 2 or earlier, once, unmapping and freeing the older mapped ones it passes over, and `None` otherwise; it never returns a block of frame n − 1 or n, so on a GPU that keeps pace the block is frame n − 2's, and on one running behind a block that maps late still reaches the HUD (Q-81). The counts it reports are the clamped ones, those the draws use. R-6 checks the arrival two frames late under a watchdog on the frame loop.

## The Bevy adapter's render nodes

`ftgsxr_render::bevy` (§2.1, §2.4 Frame loop hooks) runs the same calls inside each `SplatCamera`'s 3D pass. Bevy 0.19's render passes are systems of the per-camera `Core3d` schedule, so the adapter's two nodes are systems there too, in `Core3dSystems::MainPass` (Q-100):

| Node | Runs | Encodes |
| --- | --- | --- |
| prepare | before `main_opaque_pass_3d` | `encode_prepare` on the context's encoder, unless the frame is skipped (below) |
| draw | after `main_opaque_pass_3d`, before `main_transparent_pass_3d` | `encode_draw` in a pass that loads the view's opaque depth (`Depth32Float`, reverse Z, Q-87), then `encode_resolve` for `LinearOver`, then `encode_stats` |

With `OutputSpace::FileEncoded` the draw writes straight into the view target, whose format the passes are compiled for: the target then holds the layer of a headless render, exactly on a 16-bit float target and rounded on another target that is not sRGB. On an sRGB target the view is not drawn (warned once), since the hardware would encode the file's values on store and decode them for blending, the linear-light misreading D-6 and R-11 rule out (Q-100). With `LinearOver` the draw writes a per-view 16-bit float layer, cleared transparent, and the resolve composites it onto an `Rgba8UnormSrgb` view target (Q-91, Q-94). Each splat camera has its own `FrameBuffers`, of capacity C from the tier (`SumOfModels`: the drawn instances' total N, Q-86); the passes are shared by every view of a (layer format, sort mode). Adding `SplatCamera` turns MSAA and tonemapping off and clears the camera transparent (D-6). E6's camera of the view comes from Bevy's clip-from-view matrix and view transform, so splats and meshes see one view.

A `SplatCamera` must cover its whole render target in M-2: no viewport, or one equal to the target (§2.4 rule 4, "the layer sized from the swapchain or window"; Q-91, Q-100). The adapter sizes a view's E6 camera and its `LinearOver` layer from the camera's viewport, but the depth attachment it tests against is the target's, and the `FileEncoded` draw scales its sample point by the camera's size over the target's. A sub-viewport would therefore give a `LinearOver` view a layer and a depth attachment of different sizes, and a `FileEncoded` view a shifted sample point. Nothing warns or refuses at runtime. Viewports (thumbnails, split views) are M-4's to support.

The main world loads on the async compute task pool (`SplatLoad`, Q-99), uploads in chunks of at most 64 MB a frame, and reports `SplatEvent::Uploaded`, with the model's static bytes and the SH degree it was uploaded at (L′, min(the file's degree, the governor's `sh_cap` at the load), Q-241), or `Failed`; the render world extracts the visible instances whose models are on the GPU. The adapter's device is Bevy's, with Bevy's error handlers kept (Q-98).

## The governor's knobs, the render scale and the depth-gated resolve (M-4)

**The knobs.** `SplatTierState` is the render plugin's copy of the tier that the quality governor steps (E16, `ftgsxr-app/governor.md`): the render scale, the SH cap, the sort mode and the foveation, inserted from the plugin's tier and extracted each frame. The SH cap acts without a reload: every instance is drawn at the smallest of its loaded degree, the cap and `SplatShLimit`'s extra cap (a replayed request's, Q-235; `None` adds none, and loads never read it), and the prepare key follows it, while the next loads take it as their load option (§1.2 load option 1). The sort mode and the foveation are carried for the headset (M-8).

**The running tier.** `SplatTierState.tier` is the running tier, and replacing it, with `SplatTierState::of` the new tier, switches the tier while the app runs (Q-239): the plugin re-creates its `SplatDevice` on Bevy's device with the new tier's memory budget, so every view's passes and frame buffers are made again for the new sort key, record layout and capacity, and the render world takes the new tier. A model packed with other half-precision fields than the running tier's is not uploaded, and a load that started under another tier starts again with the new tier's options. The app makes the switch in `PreUpdate`, before every load and upload of the frame (Q-239). The plugin re-reads no file: the app reloads its instances (`ftgsxr-app/library.md`, Q-238).

**The layer at the render scale.** With `LinearOver`, the layer is sized by `layer_size(view, render_scale)`, and E6's camera targets the layer's size; `FileEncoded` draws into the view target unscaled (Q-164). A render pass's attachments share one size, so below scale 1 a per-view depth at the layer's size is filled each frame, after the opaque pass and before the layer's draw, by a small full-screen pass (`depth_copy.wgsl`) that reads the view's depth, and the layer's draw tests against that copy with Q-87's fixed-function test (Q-164). The view's depth is made readable for it when `SplatCamera` is added, through `Camera3d`'s depth usages. Each layer texel takes the depth of the view pixel under its center, point-sampled, so every copied depth is a real surface's (Q-171). At scale 1 no copy is made.

**The depth-gated resolve.** When `SplatPasses::new` receives a depth format, the resolve is two pipelines from `resolve.wgsl`, both without depth write, and the adapter's resolve pass attaches the view's own depth (Q-101). Where the view's depth still holds its clear value, no opaque geometry was drawn and the background branch writes the background form of stage 7, opaque; where opaque geometry (the floor grid, controllers, hands) wrote depth, the mesh branch blends the layer over it in the passthrough form, so a splat in front of a mesh still shows over it and the mesh is never erased. The two branches compare `Equal` and `Less` against the stored depth. Without a depth format (the headless renderer, R-11, the benchmarks) the single pipeline stays.

## Stereo

On the headset each eye is its own `SplatCamera` (`ftgsxr-xr/session.md`), so per-view mode (D-4 per view, which M-3's builds use, Q-121) runs the whole pipeline once per eye: each eye has its own `FrameBuffers` and statistics block, prepares, sorts and draws into its own array layer of the swapchain, and the passes are shared. Since both eyes yield a `FrameStats` per frame, `SplatStats` publishes the first splat camera's, the live one of lowest camera order with ties to the lower render entity; the XR crate orders the eyes by index, so it is the left eye, and a primitive counts once in the visible and non-finite counts, as in shared-center mode; the per-eye blocks are not summed (Q-125). `SplatEvent::Overflow` still fires for each view's frame that dropped records (D-34). The draws stay indirect on the Quest at C = 1,000,000 (D-30): wgpu's indirect-call validation, which empties every indirect draw on the Quest 3's Adreno, is off on Android for the headless device and the XR path's, through the vendored bevy\_mod\_openxr, wgpu-core and wgpu-hal (Q-139), and with it Dv-2 holds 72 Hz in both eyes; D-30's full-capacity fallback, measured at 40.9 Hz, is not used.

## The occlusion path (M-6)

E13's environment depth reaches stage 5 through `encode_draw`'s `occlusion` argument (D-45, R-12; #566).

**One splat pipeline.** `SplatPasses` keeps one splat pipeline, whose bind group gains the depth view, a non-filtering sampler and E13's part of the view block (PRD #565, "One splat pipeline"). Without an `OcclusionInput` the block's flag is 0 and a 1 × 1 placeholder depth, made once in `SplatPasses::new`, is bound, so the headless renderer and every earlier test draw as before. The depth view may be any single-sample 2D view that samples as an unfilterable float, a depth format, `R16Unorm` or `R32Float` (the headset's `DepthFrame.image`, which `ftgsxr_xr::depth` converts from the runtime's 16-bit depth image, Q-360), its first channel the stored depth, read with `textureLoad` at the 2 × 2 nearest texels of the projection, over which the weight is filtered (Q-340, #601); the non-filtering sampler stays bound but unread. `occlusion.wgsl` holds E13's two functions, the stored depth to metres and the weight, and is prepended to `splat.wgsl` when the pipeline is built. The view block carries, with the flag, the mode and δ, both cameras' intrinsics, the depth image's size and planes, and the view's camera space to the depth camera's, composed in float64 on the CPU. What the fragment does with them is `conformance.md`'s.

**Which frame a view draws against.** The render world holds `SplatOcclusion` (Q-339): `ftgsxr_xr::depth` writes it there each frame, after its acquire (`ftgsxr-xr/depth.md`). The adapter's draw node gives the i-th live splat camera in camera order, Q-125's order with ties to the lower render entity, so the left eye first, `frames[i]` as its `OcclusionInput`, its image as Bevy prepared it with the frame's camera and planes; a camera past the end of `frames`, and every camera while the running tier's `occlusion` flag is false, gets none and draws unoccluded (Q-338), so a frame that has not arrived leaves its eye unoccluded rather than wrongly occluded. `tests/bevy_adapter_m6.rs` checks the i-th camera, a camera without a frame, empty frames and the tier's flag.

**The import helper.** `import_vulkan_texture` in `gpu` turns a Vulkan image made outside wgpu, the runtime's depth swapchain image, into a `wgpu::Texture` of the given size, layer count and format, through wgpu-hal, the `unsafe` §2.6 allows in this crate (Q-339). Its memory stays the image's owner's; the caller vouches for the image's device, shape, usages and life. The texture is tracked by wgpu from an uninitialized state, which Q-356 accepts for the runtime's images; on the headset it is only sampled, by `depth`'s conversion pass, since the runtime's images lack the transfer-source usage a copy needs (Q-360). It panics on a device that is not Vulkan. `tests/import_vulkan_texture.rs` stands in for the runtime with a two-layer 16-bit depth image of its own device.

**R-12's tier.** No named tier can run R-12 (the conformance and desktop tiers have occlusion off, and the Quest tiers draw no splats before M-8), so R-12 renders with the conformance tier and `occlusion: true` (Q-347), its synthetic depth images written as an OpenXR projection writes them (Q-340), through the display test support's occlusion input.

## Where the frame skips stages 1–4

§1.4 and §2.4 rule 2: when nothing that stages 1–4 read has changed, they are skipped and the previous frame's records, sorted payloads, counters and draw constants are drawn again, identical by construction. The adapter compares, per view, the view's `ViewParams` (E6's camera and the target size) and each drawn instance's GPU model, pose, τ (bit for bit), render mode, color transform and degree with the previous frame's; a new frame buffer set (another capacity, sort mode or layer format) always prepares, while a change of the output alone, such as a `LinearOver` background, keeps the set and does not (§2.4 rule 4, #201). The draw, the resolve and the statistics copy run every frame: stage 3's statistics block holds the skipped frame's counts already. The headless renderer prepares every render.

## What waits on later milestones

The shared-center vertex projection of splats and centers is R-9's (Q-87); the debug overlays are carried but not drawn (PRD #76); the adapter applies no `RenderLayers` or foveation yet (the render scale and the SH cap are applied since M-4, above), and a shared-center camera prepares per view and draws no splats (Q-87).
