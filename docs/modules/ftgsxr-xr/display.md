# The headset display in `ftgsxr-xr`

How `ftgsxr_xr`'s `display` applies `DisplaySettings` on the headset: the refresh rate, fixed foveation and the render scale, the three extensions it calls through the openxr crate's raw function tables, the session re-creation a render-scale change takes, and what happens on a runtime without them. The spec items are §1.12 Headset display (as revision 56 words it), §2.1 `DisplaySettings` and `HudData`, E16's foveation knob, D-27, Appendix E.1 and E.2, and Dv-12; this doc explains them and never overrides them.

## Why raw function tables

bevy\_mod\_openxr 0.6.0 sizes and formats the swapchain once, at session creation, and has no refresh rate or foveation (the refresh rate is in an open pull request, Appendix E.1). `display` therefore calls XR\_FB\_display\_refresh\_rate, XR\_FB\_foveation with its configuration, Vulkan and swapchain-update-state companions, and XR\_META\_recommended\_layer\_resolution itself, through the openxr crate's loaded function tables, each `unsafe` block with its SAFETY comment (§2.6; Q-124). `XrPlugin` asks for all of them; whether each table is loaded is checked before every call.

## The settings

`DisplaySettings` is a resource made from `[xr]` (Q-49) at plugin build, from `SettingsRes` or `Settings::default()` (Q-127): the wanted refresh rate (`None` keeps the runtime's), the runtime's rates, the foveation, the setting's own or else the tier's (`quest-balanced` on Android, `desktop` elsewhere, when the tier name is unknown), and the render scale, `[xr] render_scale` or 1.0 when unset. A change to the resource is a request; `display` acts on each changed field once, and only while the session is Running.

**The scale's source.** The swapchain follows `[xr] render_scale` alone (§1.12 at revision 56). The tier's render scale, which the quality governor steps, never reaches `display`: it sizes the splat layer within each eye's image, as in the desktop's window, so the two multiply (`ftgsxr-render`'s adapter, Q-164).

**Following the settings and the tier** (Q-404, Q-T729.1). A private `follow` system runs each frame, whatever the session's state, before the changes are applied. It writes a field of `DisplaySettings` only when its source has moved from the value it last applied: `refresh_rate` and `render_scale` (1.0 when unset) from `[xr]`; `foveation` from `[xr] foveation` when it names a level, else from the running tier's `SplatTierState.foveation`. That is the foveation knob's path: a governor step (E16) or a live tier switch changes `SplatTierState`, `follow` copies it into `DisplaySettings`, and the change is applied to the swapchain as below. An explicit `[xr] foveation` holds against the tier, and the governor leaves the knob alone while it is set (`ftgsxr-app/governor.md`). A direct write to `DisplaySettings` (Dv-12's runner, the `xr` lane's tests) holds until its source moves.

- **Refresh rate.** Once each session has begun, the runtime's rates go into `available_rates`; a wanted rate from that list is requested, one outside it is not requested and stays. The runtime's change event is logged as `display refresh rate <from> -> <to> Hz`, which is when the rate took effect.
- **Foveation.** A profile at the setting's level is created in the main world and handed to the render world, where the crate keeps the swapchain, and applied there once per profile with xrUpdateSwapchainFB: the crate creates the swapchain without a `next` chain, so the update is the only path (Q-124). The levels map as Q-131 gives: `Off`, `Low`, `Medium`, `High` to NONE, LOW, MEDIUM, HIGH with dynamic off, and `Dynamic` to HIGH with the dynamic flag, so the runtime varies the level up to HIGH. The result is logged as `foveation <level> applied to the swapchain`, or an error. Each new session has a new swapchain, so the profile is applied again; it is dropped with its session.
- **Render scale.** See the re-creation path below.

A foveation that is the running tier's while `[xr] foveation` is unset is not a request: on a runtime without XR\_FB\_foveation it is not attempted and nothing is logged (Q-161, Q-T729.1).

`Foveation` is `ftgsxr_render`'s type; `ftgsxr_xr` re-exports it so `ftgsxr-quest`, whose band excludes `ftgsxr-render`, can set the field (Q-133).

## The re-creation path

A render-scale change of 0.1 or more from the scale the live swapchain was made with (§1.12, D-27) re-creates the swapchain the only way the crate allows: `display` sets the session configuration's resolution to the new size, marks the session for re-creation (`session.md`) and requests the session's exit; the crate ends and destroys it, and `session` creates it again, the frame loop paused meanwhile. The crate leaks the old swapchain's images; that is logged. The new size is the base size times the scale, rounded and clamped to the maximum image rect (Q-132), where the base is:

- XR\_META\_recommended\_layer\_resolution's recommendation for a projection layer with null swapchain handles (the extension's case for allocating a new swapchain) at the latest predicted display time, when the extension is enabled and the result is valid;
- otherwise the view configuration's recommended eye-buffer size (Appendix E.2).

The first session's swapchain, made before any frame state exists, uses the view configuration's size times the settings' scale, set at plugin build when the scale is not 1. The two bases differ on the Quest 3 by a few per cent, so one `[xr] render_scale` names two sizes; the launch is not special, and every creation and re-creation logs `swapchain WxH at render scale s (§1.12); the view configuration recommends RxR` (Q-412). Since revision 56 only a change of `[xr] render_scale` re-creates, so the governor never meets the second base. Dv-12 sets the scale to 0.8, then 1.0, and the driver reads both sizes from the log; spike S3 runs `quest-low` and reads the swapchain at the recommended size beside the adapter's `splat layer` line at 0.8 of it.

## `XrDisplay`

`XrPlugin` inserts `ftgsxr_app`'s `XrDisplay`, whose presence puts the governor on the headset's ladder (Q-404, `ftgsxr-app/governor.md`), and `display` writes it each frame while the session is Running:

- `refresh_hz`: the runtime's current rate, read once the session has begun and at each refresh-rate change event (XR\_FB\_display\_refresh\_rate); without the extension, the reciprocal of the frame state's predicted display period; `None` while no session runs, a re-creation included.
- `available_rates`: `DisplaySettings.available_rates`, the runtime's list (empty without the extension), which the Display tab's refresh-rate picker offers (Q-405); the picker writes `[xr] refresh_rate`, which `follow` carries here.
- `foveation`: whether the runtime offers XR\_FB\_foveation with its swapchain-update companion.
- `swapchain`: the live swapchain's size per eye.

The rate is mirrored into `HudData.refresh_hz`, the HUD's field on both builds (Q-159), which decides nothing.

## Without the extensions

On a runtime that lacks an extension (Monado, I-15), `available_rates` stays empty, and a request for that control logs E-XR-UNSUPPORTED once per request, as a warning `<code>: <user text>` under `ftgsxr_xr::display` (Q-134; §1.15: the control is hidden, so not an error), and the setting is kept for a later runtime (Q-124). Foveation is reported only when it was asked for, explicitly in the settings or by a direct change to `DisplaySettings`, not for the running tier's value (its default, a governor step, a live tier switch). An explicit request also triggers a `Toast` for ui, the first time in each session for that extension; later requests in the same session log the warning only (Q-161). The render scale still works without XR\_META\_recommended\_layer\_resolution, from the view configuration's size, with E-XR-UNSUPPORTED logged for that extension.
