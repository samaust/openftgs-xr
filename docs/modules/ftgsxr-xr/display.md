# The headset display in `ftgsxr-xr`

How `ftgsxr_xr`'s `display` applies `DisplaySettings` on the headset: the refresh rate, fixed foveation and the render scale, the three extensions it calls through the openxr crate's raw function tables, the session re-creation a render-scale change takes, and what happens on a runtime without them. The spec items are §1.12 Headset display, §2.1 `DisplaySettings`, D-27, Appendix E.1 and E.2, and Dv-12; this doc explains them and never overrides them.

## Why raw function tables

bevy\_mod\_openxr 0.6.0 sizes and formats the swapchain once, at session creation, and has no refresh rate or foveation (the refresh rate is in an open pull request, Appendix E.1). `display` therefore calls XR\_FB\_display\_refresh\_rate, XR\_FB\_foveation with its configuration, Vulkan and swapchain-update-state companions, and XR\_META\_recommended\_layer\_resolution itself, through the openxr crate's loaded function tables, each `unsafe` block with its SAFETY comment (§2.6; Q-124). `XrPlugin` asks for all of them; whether each table is loaded is checked before every call.

## The settings

`DisplaySettings` is a resource made from `[xr]` (Q-49) at plugin build, from `SettingsRes` or `Settings::default()` (Q-127): the wanted refresh rate (`None` keeps the runtime's), the runtime's rates, and the foveation and render scale, the settings' own or else the tier's (`quest-balanced` on Android, `desktop` elsewhere, when the tier name is unknown). A change to the resource is a request; `display` acts on each changed field once, and only while the session is Running.

- **Refresh rate.** Once each session has begun, the runtime's rates go into `available_rates`; a wanted rate from that list is requested, one outside it is not requested and stays. The runtime's change event is logged as `display refresh rate <from> -> <to> Hz`, which is when the rate took effect.
- **Foveation.** A profile at the setting's level is created in the main world and handed to the render world, where the crate keeps the swapchain, and applied there once per profile with xrUpdateSwapchainFB: the crate creates the swapchain without a `next` chain, so the update is the only path (Q-124). The levels map as Q-131 gives: `Off`, `Low`, `Medium`, `High` to NONE, LOW, MEDIUM, HIGH with dynamic off, and `Dynamic` to HIGH with the dynamic flag, so the runtime varies the level up to HIGH. The result is logged as `foveation <level> applied to the swapchain`, or an error. Each new session has a new swapchain, so the profile is applied again; it is dropped with its session.
- **Render scale.** See the re-creation path below.

`Foveation` is `ftgsxr_render`'s type; `ftgsxr_xr` re-exports it so `ftgsxr-quest`, whose band excludes `ftgsxr-render`, can set the field (Q-133).

## The re-creation path

A render-scale change of 0.1 or more from the scale the live swapchain was made with (§1.12, D-27) re-creates the swapchain the only way the crate allows: `display` sets the session configuration's resolution to the new size, marks the session for re-creation (`session.md`) and requests the session's exit; the crate ends and destroys it, and `session` creates it again, the frame loop paused meanwhile. The crate leaks the old swapchain's images; that is logged. The new size is the base size times the scale, rounded and clamped to the maximum image rect (Q-132), where the base is:

- XR\_META\_recommended\_layer\_resolution's recommendation for a projection layer with null swapchain handles (the extension's case for allocating a new swapchain) at the latest predicted display time, when the extension is enabled and the result is valid;
- otherwise the view configuration's recommended eye-buffer size (Appendix E.2).

The first session's swapchain, made before any frame state exists, uses the view configuration's size times the settings' scale, set at plugin build when the scale is not 1. Dv-12 sets the scale to 0.8 and the driver reads the new size from the log.

## Without the extensions

On a runtime that lacks an extension (Monado, I-15), `available_rates` stays empty, and a request for that control logs E-XR-UNSUPPORTED once per request, as a warning `<code>: <user text>` under `ftgsxr_xr::display` (Q-134; §1.15: the control is hidden, so not an error), and the setting is kept for a later runtime (Q-124). Foveation is reported only when it was asked for, explicitly in the settings or by a change, not for the tier's default. An explicit request also triggers a `Toast` for ui, the first time in each session for that extension; later requests in the same session log the warning only (Q-161). The render scale still works without XR\_META\_recommended\_layer\_resolution, from the view configuration's size, with E-XR-UNSUPPORTED logged for that extension.
