# The environment depth in `ftgsxr-xr`

How `ftgsxr_xr::depth` gives the renderer the environment depth for E13: the depth provider's control in the main world and its drive in the render world, the adapter over XR\_META\_environment\_depth with its acquire and copy, the depth camera's frame, the permission, and Dv-7's device side (M-6, #570). The spec items are §1.9 Depth occlusion, E13, D-16, D-45, §1.13 HUD, §1.15 E-XR-PERMISSION and E-XR-UNSUPPORTED, §2.1 `EnvironmentDepth`, `DepthFrame` and `SplatOcclusion`, §2.6 (`unsafe`), §3.4 Dv-7 and Appendix E.1 and E.2; this doc explains them and never overrides them. How the draw samples the depth is `ftgsxr-render/pipeline.md`'s and `conformance.md`'s subject.

## Two halves, two worlds

The Depth API's acquire must fall between the frame's begin and end (Appendix E.2), which bevy\_mod\_openxr issues in the render world, so the module is split:

- **The control** (main world, `PostUpdate`) decides whether the provider runs, owns the permission flow, makes the views' images, and publishes what the render world reported: `XrFeatures.occlusion`, `HudData.depth_image` and the main world's `SplatOcclusion`. It hands the render world a `DepthRequest` (run, hand removal, soft or hard and δ, the images), extracted each frame.
- **The drive** (render world, after the frame's begin and Bevy's prepared assets, before the camera passes) holds the provider in a crate-private resource, makes the adapter once per session, starts and stops it as asked, sends hand removal at the start and on change (D-16), calls `acquire` twice and writes the frame's `SplatOcclusion` into the render world, where the render adapter's draw node reads it (Q-339). It reports back through a shared `Report`: the swapchain's size and length, whether the runtime refused the provider as unsupported, and the frame's depth frames.

The main world's `SplatOcclusion` is therefore the previous frame's: the HUD's resolution and age are one frame late, which is all they serve (Q-339). Built with `XrPlugin.passthrough` only (`passthrough.md`).

## When the provider runs

The provider runs while three inputs allow it: the effective mode is passthrough, `[environment] occlusion` is on, and the running tier's `occlusion` flag is true (Q-338). The tier's flag is the permission and the setting the user's switch; with any of the three off the provider is stopped and the render world's `SplatOcclusion.frames` is empty, so the draw gets no depth and w\_occ is 1. A private test covers the condition over its three inputs.

## The permission

`Permission::Scene` is asked through the platform port the first time occlusion turns on in passthrough mode (Q-348), and the provider runs once it is granted; the shim answers at once for an install-time permission. A denial gives Q-261's toast, writes `[environment] occlusion` off, sets `XrFeatures.occlusion` to `Denied` and lists the permission in `DeniedPermissions` (Q-355). While denied, the setting turning on again (the Environment tab's "Ask again", its checkbox, or a settings change) or `AskPermissionAgain(Permission::Scene)` from the Comfort tab asks once more (Q-322, Q-355), so one click asks again and the switch shows what the feature does. The flow is kept across sessions, as the hands' and the anchors' are.

`XrFeatures.occlusion` is `Absent` where the instance did not enable the extension, no session exists, or the runtime refused the provider as unsupported, `Denied` after a denial, `Available` otherwise (Q-336). The occlusion control shows whenever the feature is not `Absent`, whatever the tier says (Q-338).

## One acquire, two views

§2.1's `acquire` returns one `DepthFrame`, while one runtime acquire gives one image with two views. `acquire` is therefore called once per view per frame (Q-349): the first call of a frame makes the runtime's acquire and returns the left view's frame, the second the right view's from the same image, and further calls return `None` until the next frame. `Views` holds that sequence and the drive's `frames_of` makes the two calls, so `SplatOcclusion.frames` is left then right, the order the render adapter's views take (Q-125).

What a frame gives without a newer image (Q-357): the last frame's two `DepthFrame`s again, `age_frames` one higher, their copied images still holding that image; nothing before the first image, and nothing after a failed acquire. An image is new when the runtime's capture time differs from the last, or is not reported; images Bevy has not prepared yet count as no newer image. A start the runtime refuses is logged (E-XR-UNSUPPORTED as a warning, which also makes the feature `Absent`) and not retried until the request goes off; a refused hand-removal call is a warning and the provider runs with the system's default.

## The images and the copy

`DepthFrame.image` is a Bevy `Image` handle, while the runtime's depth images are Vulkan images of its own swapchain (Q-339):

- **The views' images.** Once the swapchain's size is known, the control makes one Bevy image per view at that size and the swapchain's format, with copy-destination and texture-binding usage, and makes them again when a new session's swapchain has another size; the frames are cleared with the session.
- **The adapter** (`meta`) creates the runtime's provider at the first `start`, after the permission is granted, since the runtime may refuse it before, with its one swapchain, whose images are two-layer arrays (left, right) in the registry's 16-bit depth format; it lives for the session. The detail of Dv-7 carries the swapchain's size and length for PRD #565's spike check 1.
- **The import.** Each swapchain image is imported into wgpu once, through `ftgsxr_render::import_vulkan_texture`, the helper of `ftgsxr_render::gpu` that keeps §2.6's wgpu-hal `unsafe` in the render crate (Q-339).
- **The copy.** Each frame whose acquire returns an image, each layer is copied, before the camera passes, into its view's image's GPU texture, so the handle keeps its `GpuImage` and the render adapter binds its view as `OcclusionInput.depth` (PRD #565's spike check 2, which Dv-7 reports).
- **The layout.** The registry hands the acquired image over in the shader-read layout and wants it back so, while wgpu tracks an imported texture from an uninitialized state and copies from the transfer-source layout; no usage the registry names includes transfer source either. Q-356 accepts this: the first copy of each swapchain image may read discarded texels, and wgpu's barriers name another layout than the runtime keeps. Dv-7's log and A-5 tell whether the driver honours it; a raw barrier would need `ash`, which §2.3 does not list.

## The depth camera's frame

The acquire names the primary reference space at the frame's predicted display time, so each view's pose is in that space. `depth_camera` composes it with the tracking root's global transform, as the eyes' views and `tracking`'s hands are, before E6's camera is built from the stage pose and the view's field of view at the image's size (Q-341). The depth camera then shares the splats' stage space however the rig moved; a private test checks the composition under a translated root.

## Dv-7 on the device

Dv-7's clauses live here, since they read the render world's crate-private provider and the frames the drive gave; they stay inert until the quest runner spawns the entity `ftgsxr-test: dv7` and print the verdict line themselves (Q-308, Q-122). As Q-346 reads the row without a person: the host driver grants USE\_SCENE first; once the session runs, the test sets passthrough mode and occlusion on, waits for the first frame with two depth frames, then counts for Q-346's window the frames that had two with finite planes and distinct cameras, and passes when every counted frame had them; it then turns occlusion off and passes when the provider stops with no failed runtime call in the session. The detail reports the swapchain's size and length (spike check 1) and whether each view's GPU texture kept its id across the counted frames (spike check 2).

## Q-n rows relied on

Q-336, Q-338, Q-339, Q-341, Q-345, Q-346, Q-348, Q-349, Q-355, Q-356, Q-357; with Q-122, Q-125, Q-134, Q-161, Q-261, Q-308 and Q-322.
