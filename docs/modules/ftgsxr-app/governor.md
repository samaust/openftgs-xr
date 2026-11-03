# The quality governor in `ftgsxr-app`

How `ftgsxr_app::governor` meets E16 and D-47: `Governor`, `GovernorCounters`, `Knob`, the two ladders and `govern`. The spec items are E16, D-47, §1.12 and §2.1 `ftgsxr_app::governor`, with U-22, I-12 and R-17 as its tests; this doc explains them and never overrides them. Built in M-4 (#288).

## The pure step

`govern(gov, counters, state, ladder, e)` is a pure function over a ladder of knobs given as data (Q-153), so U-22 runs E16's full ladder without a GPU. `GovernorCounters` holds the over and under streaks. An e above the band counts over and restarts under, below the band the reverse, inside it restarts both; a streak that reaches its length steps once and restarts. A step down takes the first knob of the ladder that can move, a step up the last; a knob steps down to E16's floor and back up only to the running tier's own value, the render scale in exact tenths, and `Dynamic` foveation is left alone. While `enabled` is false `govern` steps nothing and restarts both streaks (Q-178).

- **Ladders.** `DESKTOP_LADDER` is the render scale then the SH cap; `HEADSET_LADDER` is E16's full order, for M-8 (Q-153). Foveation and the sort mode are off the desktop's ladder: foveation is the headset's, and a one-view desktop has no shared sort to switch to.

## The system

A private `GovernorPlugin`, added last by `AppPlugin`, feeds `govern` each new statistics block's GPU time against the frame budget and writes the render layer's `SplatTierState` (`pipeline.md`). A frame counts only when `SplatStats` changed and holds a block; a frame without statistics counts to neither streak (Q-153, Q-178). f_budget is the primary window's monitor refresh rate, else a 60 Hz default (Q-153).

`Governor.enabled` is set at startup and whenever `SettingsRes` or the running tier changes: the settings' governor switch, and the running tier being the build's default; any other tier, from `--tier`, the settings or the Display tab, counts as chosen by hand and turns it off (Q-179, Q-235). The Display tab's knobs and governor switch are this module's dock section (Q-218).

**The live tier switch.** When `[general] tier` changes while the app runs (the Display or Settings tab's choice, Q-218) to a tier `QualityTier::by_name` knows, the module writes `SplatTierState::of` the new tier, its knobs at the tier's own values; the render plugin and the library follow (`ftgsxr-render/pipeline.md`, `library.md`, Q-233, Q-234). Only a change of the setting switches: a `--tier` override at startup that differs from the settings holds until the user picks a tier (Q-235).

## Q-n rows relied on

Q-153, Q-178, Q-179, Q-218, Q-233, Q-234, Q-235.
