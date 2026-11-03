// SPDX-License-Identifier: Apache-2.0

//! The quality governor (§2.1 `ftgsxr_app::governor`, E16, D-47, §1.12): a pure step over
//! a ladder of knobs given as data (Q-153), and the system that feeds it each new
//! statistics block's GPU time against the frame budget and writes `SplatTierState`.

use bevy::app::{App, Plugin, Update};
use bevy::ecs::prelude::*;
use bevy::window::{Monitor, OnMonitor, PrimaryWindow};
use ftgsxr_render::{Foveation, SortMode, SplatStats, SplatTierState};

use crate::persist::SettingsRes;

/// The governor's switch and thresholds (§2.1, E16): it steps a knob down after
/// `down_frames` consecutive frames over budget and up after `up_frames` under;
/// `budget_override_ms`, when set, replaces f_budget (R-17).
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct Governor {
    /// Whether the governor steps the knobs: false while a tier is chosen by hand or
    /// `Settings.general.governor` is off (§1.12).
    pub enabled: bool,
    /// The frames over 1.05 that step a knob down (E16: 10).
    pub down_frames: u32,
    /// The frames under 0.8 that step a knob up (E16: 120).
    pub up_frames: u32,
    /// f_budget in milliseconds in place of the display's (R-17).
    pub budget_override_ms: Option<f32>,
}

impl Default for Governor {
    // Q-178
    fn default() -> Self {
        Governor {
            enabled: true,
            down_frames: 10,
            up_frames: 120,
            budget_override_ms: None,
        }
    }
}

/// One of E16's knobs (Q-153).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Knob {
    /// The render scale, by 0.1 from 1.0 down to 0.6.
    RenderScale,
    /// The fixed foveation level, one level up to high (headset only).
    Foveation,
    /// The SH degree cap, by one down to 1.
    ShCap,
    /// The sort mode, per view to shared center.
    SortMode,
}

/// The desktop's ladder (Q-153): render scale, then the SH cap.
pub const DESKTOP_LADDER: [Knob; 2] = [Knob::RenderScale, Knob::ShCap]; // Q-178

/// E16's full ladder, the headset's (M-8, Q-153).
pub const HEADSET_LADDER: [Knob; 4] = [
    Knob::RenderScale,
    Knob::Foveation,
    Knob::ShCap,
    Knob::SortMode,
]; // Q-178

/// The two streaks E16 counts (Q-153).
// Q-178
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GovernorCounters {
    /// Consecutive frames with e > 1.05.
    pub over: u32,
    /// Consecutive frames with e < 0.8.
    pub under: u32,
}

/// The render scale's floor (E16).
const SCALE_FLOOR_TENTHS: i32 = 6;
/// The SH cap's floor (E16).
const SH_FLOOR: u8 = 1;

/// A render scale in tenths, so steps stay exact.
fn tenths(scale: f32) -> i32 {
    (scale * 10.0).round() as i32
}

/// The foveation's rank in step-down order; `None` for `Dynamic`, which the governor
/// leaves alone (Q-178).
fn fov_rank(f: Foveation) -> Option<u8> {
    match f {
        Foveation::Off => Some(0),
        Foveation::Low => Some(1),
        Foveation::Medium => Some(2),
        Foveation::High => Some(3),
        Foveation::Dynamic => None,
    }
}

fn fov_of(rank: u8) -> Foveation {
    match rank {
        0 => Foveation::Off,
        1 => Foveation::Low,
        2 => Foveation::Medium,
        _ => Foveation::High,
    }
}

fn sort_rank(m: SortMode) -> u8 {
    match m {
        SortMode::PerView => 0,
        SortMode::SharedCenter => 1,
    }
}

/// Steps `knob` one notch down (`down`) or up, between E16's floor and the tier's own
/// value (Q-178); whether it moved.
fn step(state: &mut SplatTierState, knob: Knob, down: bool) -> bool {
    let tier = &state.tier;
    match knob {
        Knob::RenderScale => {
            let (now, top) = (tenths(state.render_scale), tenths(tier.render_scale));
            let next = if down { now - 1 } else { now + 1 };
            if (down && next < SCALE_FLOOR_TENTHS.min(top)) || (!down && next > top) {
                return false;
            }
            state.render_scale = next as f32 / 10.0;
        }
        Knob::Foveation => {
            let (Some(now), Some(top)) = (fov_rank(state.foveation), fov_rank(tier.foveation))
            else {
                return false;
            };
            if (down && now >= 3) || (!down && now <= top) {
                return false;
            }
            state.foveation = fov_of(if down { now + 1 } else { now - 1 });
        }
        Knob::ShCap => {
            // The tier's own cap, as `SplatTierState` starts from it (3 keeps every degree).
            let top = tier.layout.sh_cap.unwrap_or(3);
            if (down && state.sh_cap <= SH_FLOOR) || (!down && state.sh_cap >= top) {
                return false;
            }
            state.sh_cap = if down {
                state.sh_cap - 1
            } else {
                state.sh_cap + 1
            };
        }
        Knob::SortMode => {
            let (now, top) = (sort_rank(state.sort_mode), sort_rank(tier.sort_mode));
            if (down && now >= 1) || (!down && now <= top) {
                return false;
            }
            state.sort_mode = if down {
                SortMode::SharedCenter
            } else {
                SortMode::PerView
            };
        }
    }
    true
}

/// E16's step for one frame's e (Q-153): counts the streaks and, when one reaches its
/// threshold, steps the first knob of `ladder` that can go down, or the last that can go
/// up, and restarts that streak. Returns the knob stepped and whether it went down;
/// `None` when nothing stepped or the governor is off (its streaks then restart).
pub fn govern(
    gov: &Governor,
    counters: &mut GovernorCounters,
    state: &mut SplatTierState,
    ladder: &[Knob],
    e: f32,
) -> Option<(Knob, bool)> {
    if !gov.enabled {
        *counters = GovernorCounters::default();
        return None;
    }
    if e > 1.05 {
        counters.over += 1;
        counters.under = 0;
    } else if e < 0.8 {
        counters.under += 1;
        counters.over = 0;
    } else {
        *counters = GovernorCounters::default();
        return None;
    }
    if counters.over >= gov.down_frames {
        counters.over = 0;
        return ladder
            .iter()
            .find(|k| step(state, **k, true))
            .map(|k| (*k, true));
    }
    if counters.under >= gov.up_frames {
        counters.under = 0;
        return ladder
            .iter()
            .rev()
            .find(|k| step(state, **k, false))
            .map(|k| (*k, false));
    }
    None
}

/// f_budget in milliseconds (E16): one refresh period less 1.5 ms.
fn budget_ms(refresh_hz: f32) -> f32 {
    1000.0 / refresh_hz - 1.5
}

/// The refresh rate f_budget uses without an override (Q-153).
const FALLBACK_HZ: f32 = 60.0;

/// The build's ladder and the streaks (private state of the system).
#[derive(Resource)]
struct GovernorState {
    ladder: &'static [Knob],
    counters: GovernorCounters,
}

/// Adds `Governor`, its state and its systems (added by `AppPlugin`).
pub(crate) struct GovernorPlugin;

impl Plugin for GovernorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Governor>()
            .insert_resource(GovernorState {
                // The headset's ladder joins with M-8 (Q-153).
                ladder: &DESKTOP_LADDER,
                counters: GovernorCounters::default(),
            })
            .add_systems(Update, (follow_settings, run_governor).chain());
    }
}

/// `Governor.enabled` from the settings and the running tier, whenever the settings
/// change (§1.12).
// Q-179
fn follow_settings(
    settings: Option<Res<SettingsRes>>,
    tier: Res<SplatTierState>,
    mut gov: ResMut<Governor>,
) {
    let Some(settings) = settings else { return };
    if !settings.is_changed() {
        return;
    }
    let general = &settings.0.general;
    let default_tier = ftgsxr_io::GeneralSettings::default().tier;
    let enabled = general.governor && tier.tier.name == default_tier;
    if gov.enabled != enabled {
        gov.enabled = enabled;
    }
}

/// Each new statistics block's e into `govern` (E16); a frame without a new block counts
/// to neither streak (Q-153, Q-178).
fn run_governor(
    stats: Res<SplatStats>,
    gov: Res<Governor>,
    mut st: ResMut<GovernorState>,
    mut tier: ResMut<SplatTierState>,
    primary: Query<Option<&OnMonitor>, With<PrimaryWindow>>,
    monitors: Query<&Monitor>,
) {
    if !stats.is_changed() {
        return;
    }
    let Some(frame) = stats.0 else { return };
    let budget = gov.budget_override_ms.unwrap_or_else(|| {
        let hz = primary
            .iter()
            .flatten()
            .next()
            .and_then(|m| monitors.get(m.0).ok())
            .and_then(|m| m.refresh_rate_millihertz)
            .map(|mhz| mhz as f32 / 1000.0)
            .unwrap_or(FALLBACK_HZ);
        budget_ms(hz)
    });
    let e = frame.gpu_times_ms.total / budget;
    let GovernorState { ladder, counters } = &mut *st;
    // Write `SplatTierState` only on a step, so its change detection means a change.
    let mut copy = tier.clone();
    if let Some((knob, down)) = govern(&gov, counters, &mut copy, ladder, e) {
        bevy::log::info!(?knob, down, "governor step");
        *tier = copy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_one_period_less_1_5_ms() {
        assert!((budget_ms(60.0) - (1000.0 / 60.0 - 1.5)).abs() < 1e-5);
        assert!((budget_ms(90.0) - 9.611_111).abs() < 1e-4);
    }
}
