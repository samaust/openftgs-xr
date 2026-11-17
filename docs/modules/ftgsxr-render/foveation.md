# The layer's foveation in `ftgsxr-render`

How `ftgsxr_render::bevy` draws each eye's layer pass with a fragment density map of its own (§1.12 Headset display, D-58, §1.4 stages 5 and 6, §2.4 rule 3). The spec items are those, §2.3's note for D-58 and E6; the rows relied on are Q-457, Q-458, Q-459, Q-460, Q-461, Q-164 and Q-462. This doc explains them and never overrides them; the pass itself is in `pipeline.md`.

## `SplatFoveation`

The public component `SplatFoveation(pub Foveation)`, re-exported from `ftgsxr_render`, is the seam that carries a camera's effective level to its layer pass (Q-458). It has no `Default`: its absence is the unfoveated case. A camera without it, or with it at `Off`, draws its layer exactly as without foveation, with no map attached, on every device (Q-457). A camera at any other level draws with its map where the device has `wgpu::Features::FRAGMENT_DENSITY_MAP`, and with no map elsewhere. The adapter extracts it with the `SplatCamera`s each frame.

`ftgsxr_xr::display`'s `carry_to_eyes` inserts it: it puts it on each eye, whatever the device's support, and keeps it equal to `DisplaySettings.foveation` each frame before extraction; the desktop window's camera never has it.

## The map builder

The private `density_map(level, layer size, clip_from_view, texel)` builds the map's `Rg8Unorm` texels, row-major from the framebuffer's origin:

- **No map at off** (Q-457): it returns none.
- **The grid** is the device's minimum density texel size, per axis, read once per device through `density_map_support`'s `min_texel` (Q-460, Q-461), and the map is ⌈layer ÷ texel⌉ texels on each axis, which covers the layer and meets Vulkan's minimum-size rule. A device that reports no texel size gets no map.
- **The optical axis and the view angles** come from the camera's extracted projection, read at the layer's size with the same mapping as `view_camera`'s E6 intrinsics (Q-459): the axis is the pixel the view's forward axis projects to, off center on the Quest's asymmetric eyes, so a map sized to a smaller layer stays centered at every render scale.
- **The pattern** is §1.12's concentric regions per level, dynamic taking high's. Each texel's angle is taken at its center, clamped into the layer, and a density is stored as round(255 d) on both axes (Q-462).

## When a map is re-made

Each view keeps its map with the (level, layer size, projection) it was made for, and it is re-made in the render world's prepare when any of the three changes (Q-457). The new map is uploaded with the queue before the frame's passes, as the Quest 3's lack of `fragmentDensityMapDynamic` requires. A replaced map is simply dropped: wgpu-core tracks the map's view and keeps it alive while a pass that reads it is in flight (§2.3's note for D-58).

## The route's vendored changes

Route P1 (D-58, §2.3's note): the vendored wgpu-types adds the feature and the texture usage, wgpu the pipeline flag and the command-encoder entry point, wgpu-core the tracking and the pipeline-against-pass check, wgpu-hal the Vulkan attachment and the compatible passes, and bevy\_render passes the flag unset. The adapter begins the map pass on `RenderContext::command_encoder()` and wraps it with `TrackedRenderPass::new`.

## Where it is checked

The builder's `cpu` tests (`density_map_*` in `bevy/foveation.rs`) pin the axis-centered region under an asymmetric frustum at render scales 1.0 and 0.6, no map at off, dynamic as high, the map's size and §1.12's radii. `gpu_bevy_adapter_draws_unfoveated_without_splat_foveation_or_at_off` pins Q-458's no-map case. The map path runs only on a device with the extension, so Dv-14, Dv-2 and A-13 check it on the Quest 3 (#810).
