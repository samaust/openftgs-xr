# Time in `ftgsxr-core`

How the core carries time from a model file to the GPU: the three quantities, the three types that hold them, the invariants between those types, and the single-timestamp model as it runs through every module that reads time. The spec items are E9, §1.6, D-32 and D-9, with §1.2 (load option 3), §1.14 Time and openftgs §1.4 Time; this doc explains them and never overrides them.

## Three times

- **Raw timestamp τ.** What a model file carries: every time center, τ₀ and τ₁, in the model's own `time_unit` (§1.2, openftgs §1.4 Time). The CPU copy of a model keeps it in float64 (§2.1 `Model`), and every CPU evaluation of E1 and E2 (`motion`) takes τ in that unit.
- **Rebased GPU time.** On the GPU a time center is stored as τ − τ₀ in float32, and each instance's uniform block carries τ_m − τ₀ (§1.2 load option 3, §1.4 Prepare). Rebasing keeps a raw timestamp far from zero (an epoch, say) precise; `layout` does it once at packing, and U-7 pins the error it leaves.
- **Playback clock T.** The player's one time source, in seconds since its zero (E9, §1.14 Time). It knows nothing about units or models: `Clock` holds T, the speed, the loop mode, the playback range and T_len, and the caller advances it by each frame's Δt.

E9 is the only bridge between the clock's seconds and a model's raw time; `ModelTime::tau_at` is its implementation, and nothing else converts.

## Who holds what

- **`TimeSpan`** (`types`) is the model's own: τ₀, τ₁ and the unit, as read from the file. It answers the training time span T_τ, whether the model is a single-timestamp model, and E9's rate r for the unit, refusing a unit it does not know (D-32).
- **`ModelTime`** (`clock`) is one model instance's mapping: a copy of the span, the offset o_m in seconds, the rate r_m and the end rule. `from_span` gives the default mapping; an unknown unit gets rate 1 and a warning there, not in `TimeSpan`, so the span's refusal stays the testable signal and the warning is emitted once per instance (PRD #1's decision on warnings).
- **`Clock`** (`clock`) is the scene's: one per player, shared by every instance (§1.6).

Invariants between them:

- `ModelTime.span` is the span of the model the instance shows; `frame_step_s` reads T_τ and r_m from the mapping and only the time centers and the kind from the model (Q-20).
- Each instance's `length_s` is its end on the clock; the caller sets the clock's T_len to the largest of them (E9). The clock never reads a `ModelTime`.
- T stays in [0, T_len]; while playing in loop or ping-pong it also stays within the playback range, and `speed()` is always the signed rate at which T moves, so a ping-pong reflection shows as a sign change (Q-19).
- `tau_at` under hold returns a τ_m inside the span and never flags it; under extrapolate it returns E9's value unchanged and flags it exactly when it lies outside the span (Q-20).

## Inside `clock`

`Clock::advance` maps T back into the active interval in closed form rather than by repeated steps: loop by a Euclidean remainder over the half-open interval, ping-pong by a triangle wave whose reflection count decides the speed's sign, once by stopping at an end and pausing. A long frame or a seek far from the range therefore costs the same as a short one, and the proptest variant of U-14 checks the fold against E9's unwrapped time. The active interval is the playback range in loop and ping-pong and the whole clock in once, the plain reading of §1.6 (Q-19). A clock whose T_len is zero is idle (E9).

`ModelTime::frame_step_s` follows §1.6's chain for Δ_frame in order (D-32): the bounded median spacing of the distinct time centers, then a checkpoint's frame interval Δt_f carried by `ModelKind::Checkpoint`, then the default. The median is the statistical median over the gaps between distinct centers, the same convention openftgs uses for the Δt_f the checkpoint carries (Q-20). A checkpoint hint that comes out zero or non-finite, as for a single-timestamp checkpoint, falls through to the default (Q-20).

## The single-timestamp model

A model with τ₀ = τ₁ is static: no motion, a temporal opacity of one at every τ, and no effect from the clock (§1.2, D-9; openftgs D-70, D-71). Each module that reads time handles it at its own seam:

- **`clock`.** `tau_at` returns τ₀ whatever the clock time and end rule, unflagged, so the instance never shows as extrapolating; its T_τ/r_m term in E9's T_len is zero, and when every instance is a single-timestamp model the clock is idle (E9). Its frame step is the default, there being one distinct time center (D-32), and §1.6 shows no time controls for it.
- **`motion`.** E1 leaves positions where they are, since §1.2's reader rules require zero velocities and time centers equal to τ₀; E2's temporal opacity takes the single-timestamp flag and returns one at any τ (§2.1 `temporal_opacity`, D-9, U-7).
- **`index`.** Every row is active at every τ, so the dispatch range at any τ must cover every row (E3, D-10, U-8's superset property); the index's margins are in units of T′, which §1.2 sets to one for a single timestamp.
- **`layout`.** Rebasing gives every time center zero and the instance's τ_m − τ₀ is zero (§1.2 load option 3), so the GPU path sees the same static model the CPU does. `GpuArrays` carries τ₁ = τ₀, and the time-center colormap's span is then zero: every center maps to its low end (Q-97).

## Q-n rows relied on

- **Q-19** the clock's behaviours E9 and §1.6 leave open: the initial state, the half-open loop, the signed speed after a reflection, once and the range, out-of-domain arguments.
- **Q-20** the frame step's median convention and its zero checkpoint hint, and when `tau_at` flags extrapolation.
