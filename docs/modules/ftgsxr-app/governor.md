# The quality governor in `ftgsxr-app`

How `ftgsxr_app::governor` meets E16 and D-47: `Governor`, `GovernorCounters`, `Knob`, the two ladders and `govern`. The spec items are E16, D-47, §1.12 and §2.1 `ftgsxr_app::governor`, with U-22, I-12 and R-17 as its tests; this doc explains them and never overrides them. Built in M-4 (#288).

## The pure step

`govern(gov, counters, state, ladder, e)` is a pure function over a ladder of knobs given as data (Q-153), so U-22 runs E16's full ladder without a GPU. `GovernorCounters` holds the over and under streaks. An e above the band counts over and restarts under, below the band the reverse, inside it restarts both; a streak that reaches its length steps once and restarts. A step down takes the first knob of the ladder that can move, a step up the last; a knob steps down to E16's floor and back up only to the running tier's own value, the render scale in exact tenths, and `Dynamic` foveation is left alone. While `enabled` is false `govern` steps nothing and restarts both streaks (Q-178).

- **Ladders.** `DESKTOP_LADDER` is the render scale then the SH cap; `HEADSET_LADDER` is E16's full order (Q-153). Foveation and the sort mode are off the desktop's ladder: foveation is the headset's, and a one-view desktop has no shared sort to switch to.

## The system

A private `GovernorPlugin`, added last by `AppPlugin`, feeds `govern` each new statistics block's GPU time against the frame budget and writes the render layer's `SplatTierState` (`pipeline.md`). A frame counts only when `SplatStats` changed and holds a block; a frame without statistics counts to neither streak (Q-153, Q-178). On the desktop f_budget is the primary window's monitor refresh rate, else a 60 Hz default (Q-153); `Governor.budget_override_ms`, when set, replaces it on either machine and counts frames even without a rate (R-17, Q-416).

**On the headset** (M-8, #723, Q-404). The ladder and the budget follow `ui`'s `XrDisplay`, which only the XR crate inserts (`ui.md`), read each frame so the order the plugins are added in does not matter. While it exists the system steps `HEADSET_LADDER` and takes f_budget from `XrDisplay.refresh_hz` (at 72 Hz 12.39 ms, at 90 Hz 9.61 ms); while that is `None`, before a session runs, no frame counts to either streak. Without the resource, the desktop rule above stays.

- **The pinned and skipped knob.** The foveation knob is off the ladder, leaving render scale, SH cap and sort mode, while `[xr] foveation` is set in the settings (the XR crate's `display` applies that value, so the governor never fights it) and where `XrDisplay.foveation` is false (the runtime offers none). `Dynamic` foveation stays on the ladder but never moves, so a step passes over it (Q-178). `govern` itself is unchanged: the system passes it the ladder as data.
- **The render scale** the governor steps sizes the splat layer within the eye buffer; it never reaches the swapchain, whose size follows `[xr] render_scale` (§1.12).

`Governor.enabled` is set at startup and whenever `SettingsRes` or the running tier changes: the settings' governor switch, and the running tier being the build's default; any other tier, from `--tier`, the settings or the Display tab, counts as chosen by hand and turns it off (Q-179, Q-240). The Display tab's knobs and governor switch are this module's dock section (Q-218).

**The live tier switch.** The Display tab's tier picker shows the running tier, and the Settings tab's `[general] tier` the saved one; choosing a tier in either is a pick, a crate-private `TierPick` message that `persist` writes along with `[general] tier` (saved by Q-207's debounce). The last pick of a frame, when `QualityTier::by_name` knows it and it is not the running tier, becomes the running tier, `SplatTierState::of` the new tier with its knobs at the tier's own values, in `PreUpdate` before the library's reload; the render plugin and the library follow (`ftgsxr-render/pipeline.md`, `library.md`, Q-238, Q-239). A pick switches even when it equals the saved tier, so the saved tier can be picked back after a `--tier` override; nothing else switches, so the override holds until the user picks a tier (Q-240, #402).

## On the device

R-17 runs on the headset as well as on the desktop's GPU (M-8, #730, Q-431). The device check needs E16's full ladder with a refresh rate from a real runtime: `quest-balanced`, whose knobs start at render scale 1.0, medium foveation, SH cap 2 and shared-center sort. With `budget_override_ms` at 0.1 the knobs step down in this order:

1. The render scale, from 1.0 to 0.6 in four steps.
2. The foveation, from medium to high.
3. The SH cap, from 2 to 1.

Each step comes after 10 blocks over budget. The sort mode never steps, because the tier already sorts shared-center. At 1,000 the knobs step back up in the reverse order, each after 120 blocks under budget.

**How the check runs.** The work is split between three parts:

- **The quest runner** `r17` (`ftgsxr-quest`) loads the probe and keeps it ahead of the left eye. It holds the governor off until the session runs. Then it spawns the marker `ftgsxr-test: r17` (Q-308).
- **`ftgsxr_xr`'s private `display::r17`** does the rest. It reads `SplatTierState`, which the quest crate cannot name. It runs in `Last`, after this module's `Update`, and counts blocks with the governor's own rule, so each step's count is exact. After a warm-up with the governor off, it turns the governor on with the override in the same frame, so both streaks start at zero. It logs each knob change with its block count.
- **The host driver** `dv_r17_governor.rs` reads those lines and checks the foveation and swapchain lines from `display` (`ftgsxr-xr/display.md`). A foveation step reaches the swapchain through `display`'s `follow` (`foveation High applied`, then `Medium`). The render scale only sizes the splat layer, so no swapchain is re-created during the run (§1.12 at revision 56).

**What can stop the run.** The device keeps its saved settings between tests, and three of them can break the check. `[general] governor = false` makes `follow_settings` turn the governor off at the first step. A saved `[xr] foveation`, or a runtime without foveation, takes the foveation knob off the ladder. The run checks all three before it starts and fails with the reason. It never writes the settings, because `persist` would save the change on the device.

**GPU time.** The check depends on the device's GPU times (f_gpu). Spike S1 found timestamp queries on the XR device, so f_gpu needs no fallback (#718). The warm-up fails the run if a block's GPU time is not over 1.05 × 0.1 ms. For a heavier model than P1, set `FTGSXR_R17_ROWS`.

## Q-n rows relied on

Q-153, Q-178, Q-179, Q-218, Q-238, Q-239, Q-240, Q-308, Q-404, Q-416, Q-431.
