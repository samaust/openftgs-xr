# The per-frame pipeline in `ftgsxr-render`

How `ftgsxr_render::gpu` runs the eight stages of §1.4's per-frame pipeline for one view: which `SplatPasses` call encodes each stage, which buffers and textures it reads and writes, and how the statistics come back two frames later. The spec items are §1.4 (the per-frame pipeline, several models, stereo, depth and occlusion, backgrounds), E1–E8, E13, E14, D-3–D-6, D-14, D-33–D-35 and §2.4; the sort itself is `sort.md`'s subject and the headless renderer `conformance.md`'s. This doc explains them and never overrides them.

## The calls and the passes they encode

| Stage | Call | Encoded as | Shader |
| --- | --- | --- | --- |
| 1 Prepare | `encode_prepare` | CPU work, uniform buffers made at mapping | none |
| 2 Cull | `encode_prepare` | the "prepare" compute pass, one dispatch per instance and kind | `cull.wgsl` (`cull_splats`, `cull_centers`) |
| 3 Size | `encode_prepare` | the same compute pass, one workgroup | `cull.wgsl` (`size`) |
| 4 Sort | `encode_prepare` | a compute pass of its own, every dispatch indirect | `sort_count.wgsl`, `sort_scan.wgsl`, `sort_scatter.wgsl` |
| 5 Draw splats | `encode_draw` | in the caller's render pass on the layer | `splat.wgsl` |
| 6 Draw centers | `encode_draw` | the same render pass, after the splats (Q-96) | `centers.wgsl` |
| 7 Resolve | `encode_resolve` | in the caller's render pass on the view target | `resolve.wgsl` |
| 8 Statistics | `encode_stats` | copies on the caller's encoder | none |

The caller owns the encoders and the render passes: it clears the layer to (0, 0, 0, 0), attaches the opaque depth when the passes were made with a depth format, and sets the draw pass's timestamp writes (Q-87). `SplatPasses::new` compiles every pipeline once per (layer format, depth format, tier); the resolve is compiled for one view-target format, `Rgba8UnormSrgb`, with no depth attachment (Q-91). A headless render runs stages 1–5 and reads the layer back instead of resolving (§1.7, `conformance.md`).

## The buffers

One `FrameBuffers` set per view (per-view mode) holds, sized once per tier and capacity C (§1.12, §2.4 rule 4, Q-23):

- **records**: C slots in the tier's record layout. Splat records fill slots from 0; center records follow them, from slot min(splats, C) (Q-78).
- **keys and payloads**, two pairs: the sort's (k̄, slot) pairs, sorted into the first pair (`sort.md`).
- **histograms**: the sort's per-block digit counts (Q-85).
- **counters**: the splat draw's indirect arguments (words 0–3), the centers draw's (words 4–7), the overflow flag, and the rows that passed E3's test (Q-79).
- **sort dispatch**: stage 3's indirect dispatch arguments for the sort (Q-85).
- **statistics**: the 64-byte block of stage 8, and three staging buffers it is copied into (Q-81).
- **draw constants**: one block per instance, written by `encode_prepare` and kept for `encode_draw` (Q-87): E7's constants of the instance's model (D-39), then the centers' colormap, ρ and scales (Q-92, Q-93).

Each `GpuModel` holds its structure-of-arrays buffers, its uniform block of E6 and E7 constants, its temporal index and the 99th percentile of its speeds (Q-93).

## Stage by stage

**1 Prepare.** On the CPU, per instance: τ\_m − τ₀ in float32 (computed in float64), the pose's rotation, translation and scale, E4's eye in model space, E14's gains with the flag that applies them only when they differ from the identity (Q-80), the dispatch ranges of the temporal index at τ\_m (E3), the SH degree and ρ. Splat dispatches are listed before every centers dispatch (Q-78). The frame block holds the cull views, C and the sort pose. Nothing is read from the GPU (D-34).

**2 Cull.** Reads the model buffers and uniform blocks, the instance block and the frame block; writes records, keys, payloads and counters. A visible splat takes a slot by an atomic increment of the splat count; a slot at or beyond C is dropped and raises the overflow flag. A center takes slot min(splats, C) + j by its own count (Q-78).

**3 Size.** Reads and rewrites the counters: clamps the splat count to C and the center count to what remains, completes both indirect draws (four vertices, first instance 0) and raises the overflow flag when either was clamped. Writes the sort dispatch arguments and the counts, flag and C of the statistics block, its timestamps zeroed.

**4 Sort.** Reads the counters and the sort dispatch arguments; sorts the splats' pairs through the histograms into the first key and payload pair (`sort.md`). Centers are not sorted.

**5 Draw splats.** Reads the records, the first payload buffer (draw order), the draw constants and a view block; one indirect draw from counter words 0–3, blended `One, OneMinusSrcAlpha` into the layer, depth-tested without depth write when a depth format was given (Q-87). In shared-center mode nothing is drawn until the per-eye projection of R-9 (Q-87).

**6 Draw centers.** Reads the records, the counters (bound read-only: instance i draws slot `visible_splats + i`, where `visible_splats` is word 1 after stage 3), the draw constants and the view block; one indirect draw from counter words 4–7. Each disc is the pixels whose sample point lies within ρ camera pixels of the record's μ′, written opaque in the instance's colormap color (Q-92; the time-center map spans \[τ₀, τ₁\], whose τ₁ `GpuArrays` carries, and a single-timestamp model maps every center to its low end, Q-97), with depth written when depth-tested. In `Both` mode a disc replaces its splat's pixels (Q-96).

**7 Resolve.** Reads the layer through a linear sampler at each target pixel's center (Q-95) and a block holding the background. Without a background it writes the layer un-premultiplied, decoded from sRGB and re-premultiplied, blended over the target, so the target keeps the layer's alpha for the compositor (D-14). With one it decodes E7's C + (1 − A)·c\_bg, formed in the file's space, and writes it opaque (Q-94). The sRGB target's encode on store then shows the file's values (D-6, R-11).

**8 Statistics.** Resolves the frame's timestamps into the statistics block, then copies it into the next of three staging buffers, mapped when the encoder's submission completes.

## The two-frame statistics

Stage 8 never waits (P-5). The three staging buffers form a ring: the n-th `encode_stats` (from 0) copies frame n's block into buffer n mod 3 and asks for its mapping on submission. A buffer still waiting for its mapping skips that frame's copy; a mapped buffer nobody polled is unmapped and reused. `FrameStats::poll` polls the device without blocking and returns frame n − 2's block, once, when its buffer is mapped, and `None` otherwise (Q-81). The counts it reports are the clamped ones, those the draws use. R-6 checks the arrival two frames late under a watchdog on the frame loop.

## The Bevy adapter's render nodes

`ftgsxr_render::bevy` (§2.1, §2.4 Frame loop hooks) runs the same calls inside each `SplatCamera`'s 3D pass. Bevy 0.19's render passes are systems of the per-camera `Core3d` schedule, so the adapter's two nodes are systems there too, in `Core3dSystems::MainPass` (Q-100):

| Node | Runs | Encodes |
| --- | --- | --- |
| prepare | before `main_opaque_pass_3d` | `encode_prepare` on the context's encoder, unless the frame is skipped (below) |
| draw | after `main_opaque_pass_3d`, before `main_transparent_pass_3d` | `encode_draw` in a pass that loads the view's opaque depth (`Depth32Float`, reverse Z, Q-87), then `encode_resolve` for `LinearOver`, then `encode_stats` |

With `OutputSpace::FileEncoded` the draw writes straight into the view target, whose format the passes are compiled for: the target then holds the layer of a headless render, exactly on a 16-bit float target and rounded on another target that is not sRGB. On an sRGB target the view is not drawn (warned once), since the hardware would encode the file's values on store and decode them for blending, the linear-light misreading D-6 and R-11 rule out (Q-100). With `LinearOver` the draw writes a per-view 16-bit float layer, cleared transparent, and the resolve composites it onto an `Rgba8UnormSrgb` view target (Q-91, Q-94). Each splat camera has its own `FrameBuffers`, of capacity C from the tier (`SumOfModels`: the drawn instances' total N, Q-86); the passes are shared by every view of a (layer format, sort mode). Adding `SplatCamera` turns MSAA and tonemapping off and clears the camera transparent (D-6). E6's camera of the view comes from Bevy's clip-from-view matrix and view transform, so splats and meshes see one view.

The main world loads on the async compute task pool (`SplatLoad`, Q-99), uploads in chunks of at most 64 MB a frame, and reports `SplatEvent::Uploaded` or `Failed`; the render world extracts the visible instances whose models are on the GPU. The adapter's device is Bevy's, with Bevy's error handlers kept (Q-98).

## Where the frame skips stages 1–4

§1.4 and §2.4 rule 2: when nothing that stages 1–4 read has changed, they are skipped and the previous frame's records, sorted payloads, counters and draw constants are drawn again, identical by construction. The adapter compares, per view, the view's `ViewParams` (E6's camera and the target size) and each drawn instance's GPU model, pose, τ (bit for bit), render mode, color transform and degree with the previous frame's; a new frame buffer set (another capacity, sort mode or layer format) always prepares. The draw, the resolve and the statistics copy run every frame: stage 3's statistics block holds the skipped frame's counts already. The headless renderer prepares every render.

## What waits on later milestones

E13's environment depth is not sampled yet (`occlusion` is unread, R-12); the shared-center vertex projection of splats and centers is R-9's (Q-87); the debug overlays are carried but not drawn (PRD #76); the adapter applies no `RenderLayers`, render scale or foveation yet, and a shared-center camera prepares per view and draws no splats (Q-87).
