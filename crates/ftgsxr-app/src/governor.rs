// SPDX-License-Identifier: Apache-2.0

//! The quality governor (§2.1 `ftgsxr_app::governor`, E16, D-47, §1.12): a pure step over
//! a ladder of knobs given as data (Q-153), and the system that feeds it each new
//! statistics block's GPU time against the frame budget and writes `SplatTierState`.

use bevy::app::{App, Plugin, PreUpdate, Update};
use bevy::ecs::prelude::*;
use bevy::window::{Monitor, OnMonitor, PrimaryWindow};
use ftgsxr_render::{Foveation, SortMode, SplatStats, SplatTierState};

use crate::library::TierReload;
use crate::persist::{SettingsRes, TierPick};
use crate::ui::{DockHooks, Tab};

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
        // §1.7 Panels (D-21): the Display tab's tier, governor and render scale.
        app.init_resource::<DockHooks>()
            .world_mut()
            .resource_mut::<DockHooks>()
            .add(Tab::Display, display_section);
        app.init_resource::<Governor>()
            .insert_resource(GovernorState {
                // The headset's ladder joins with M-8 (Q-153).
                ladder: &DESKTOP_LADDER,
                counters: GovernorCounters::default(),
            })
            .add_message::<TierPick>()
            .add_systems(Update, (follow_settings, run_governor).chain())
            // Q-238, Q-240: the switch lands before `Update` and before the library's
            // reload, so no load or upload of the frame sees the old tier.
            .add_systems(PreUpdate, follow_tier.before(TierReload));
    }
}

/// The live tier switch (§1.7 Panels, §1.12): the last tier the user picked this frame,
/// when `by_name` knows it and it is not the running one, becomes the running tier, its
/// knobs at its own values, even when it equals the saved `[general] tier`. Only a pick
/// switches: neither the first frame nor a difference between the settings and the
/// running tier does, so a `--tier` override at startup holds until the user picks a tier
/// (Q-240).
// Q-240, Q-239
fn follow_tier(mut picks: MessageReader<TierPick>, mut tier: ResMut<SplatTierState>) {
    let Some(TierPick(wanted)) = picks.read().last() else {
        return;
    };
    if tier.tier.name == *wanted {
        return;
    }
    match ftgsxr_render::QualityTier::by_name(wanted) {
        Some(next) => {
            bevy::log::info!("quality tier {} -> {}", tier.tier.name, next.name);
            *tier = SplatTierState::of(&next);
        }
        None => bevy::log::warn!("quality tier {wanted}: no such tier; the tier stays"),
    }
}

/// `Governor.enabled` from the settings and the running tier, whenever either changes
/// (§1.12), so a pick of the build's default tier with `[general] governor` on turns the
/// governor back on (Q-240).
// Q-179, Q-240
fn follow_settings(
    settings: Option<Res<SettingsRes>>,
    tier: Res<SplatTierState>,
    mut gov: ResMut<Governor>,
) {
    let Some(settings) = settings else { return };
    if !settings.is_changed() && !tier.is_changed() {
        return;
    }
    let general = &settings.0.general;
    let default_tier = ftgsxr_io::GeneralSettings::default().tier;
    let enabled = general.governor && tier.tier.name == default_tier;
    if gov.enabled != enabled {
        gov.enabled = enabled;
    }
}

/// The Display tab's section (§1.7 Panels): the running tier and its knobs, the
/// governor's switch (`[general] governor`, which `follow_settings` maps), and the render
/// scale by hand while the governor is off, in tenths between E16's floor and the tier's
/// own (Q-218). The tier picker is `persist`'s, whose pick `follow_tier` switches to
/// (Q-240).
// Q-218, Q-240
fn display_section(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    use bevy_egui::egui;
    let enabled = world.get_resource::<Governor>().is_some_and(|g| g.enabled);
    if let Some(t) = world.get_resource::<SplatTierState>() {
        ui.label(format!(
            "tier {} · render scale {:.1} · SH ≤ {} · sort {:?} · foveation {:?}",
            t.tier.name, t.render_scale, t.sh_cap, t.sort_mode, t.foveation
        ));
    }
    if let Some(s) = world.get_resource::<SettingsRes>() {
        let mut on = s.0.general.governor;
        if ui.checkbox(&mut on, "governor").changed() {
            world.resource_mut::<SettingsRes>().0.general.governor = on;
        }
    }
    if !enabled && let Some(t) = world.get_resource::<SplatTierState>() {
        let top = tenths(t.tier.render_scale).max(SCALE_FLOOR_TENTHS);
        let mut now = tenths(t.render_scale).clamp(SCALE_FLOOR_TENTHS, top);
        let r = ui.add(
            egui::Slider::new(&mut now, SCALE_FLOOR_TENTHS..=top).text("render scale (tenths)"),
        );
        if r.changed() {
            let scale = now as f32 / 10.0;
            let mut t = world.resource_mut::<SplatTierState>();
            if t.render_scale != scale {
                t.render_scale = scale;
            }
        }
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
    use ftgsxr_render::QualityTier;

    /// Draws a dock section once on a headless egui context (#286).
    fn draw_once(world: &mut World, hook: fn(&mut World, &mut bevy_egui::egui::Ui)) {
        let ctx = bevy_egui::egui::Context::default();
        let mut out = ctx.run_ui(bevy_egui::egui::RawInput::default(), |ui| hook(world, ui));
        out.textures_delta.clear();
    }

    // Q-218
    #[test]
    fn the_display_section_draws_without_changing_the_tier() {
        let mut world = World::new();
        world.insert_resource(Governor::default());
        world.insert_resource(SettingsRes::default());
        let tier = ftgsxr_render::QualityTier::desktop();
        world.insert_resource(SplatTierState {
            render_scale: tier.render_scale,
            sh_cap: 3,
            sort_mode: tier.sort_mode,
            foveation: tier.foveation,
            tier,
        });
        world.clear_trackers();
        draw_once(&mut world, display_section);
        assert!(!world.resource_ref::<SplatTierState>().is_changed());
        assert!(!world.resource_ref::<SettingsRes>().is_changed());
    }

    /// The tier switch's systems on their own (Q-240): `[general] tier` saved as `saved`,
    /// `running` as `--tier` would set it, the governor at its defaults.
    fn tier_app(saved: &str, running: &QualityTier) -> App {
        let mut app = App::new();
        let mut settings = SettingsRes::default();
        settings.0.general.tier = saved.into();
        app.add_message::<TierPick>()
            .insert_resource(settings)
            .insert_resource(SplatTierState::of(running))
            .insert_resource(Governor::default())
            .add_systems(PreUpdate, follow_tier)
            .add_systems(Update, follow_settings);
        app
    }

    /// A pick as the Display or Settings tab's tier picker makes it.
    fn pick(app: &mut App, tier: &str) {
        crate::persist::pick_tier(app.world_mut(), tier.into());
    }

    fn running(app: &App) -> String {
        app.world().resource::<SplatTierState>().tier.name.clone()
    }

    fn saved(app: &App) -> String {
        app.world().resource::<SettingsRes>().0.general.tier.clone()
    }

    fn governing(app: &App) -> bool {
        app.world().resource::<Governor>().enabled
    }

    fn frames(app: &mut App, n: usize) {
        for _ in 0..n {
            app.update();
        }
    }

    // Q-240, Q-179 (moved from `tests/live_tier.rs`: the pick is crate-private, #402)
    #[test]
    fn live_tier_switch_follows_the_settings() {
        let mut app = tier_app("desktop", &QualityTier::desktop());
        frames(&mut app, 2);
        assert_eq!(running(&app), "desktop");
        assert!(governing(&app));
        pick(&mut app, "quest-low");
        frames(&mut app, 2);
        // §1.12's table: `quest-low` runs at render scale 0.8, SH cap 1; the knobs start at
        // the new tier's own values.
        let s = app.world().resource::<SplatTierState>().clone();
        assert_eq!(s.tier, QualityTier::quest_low());
        assert_eq!((s.render_scale, s.sh_cap), (0.8, 1));
        // The pick is saved too (Q-207's debounce writes `SettingsRes`).
        assert_eq!(saved(&app), "quest-low");
        // Q-179: a tier chosen by hand turns the governor off.
        assert!(!governing(&app));
        // Back to the build's default with `[general] governor` on: back on.
        pick(&mut app, "desktop");
        frames(&mut app, 2);
        assert_eq!(running(&app), "desktop");
        assert!(governing(&app));
        // A name no tier has leaves the running tier alone, and is not saved.
        pick(&mut app, "no-such-tier");
        frames(&mut app, 2);
        assert_eq!(running(&app), "desktop");
        assert_eq!(saved(&app), "desktop");
    }

    // Q-240 (#402): under a `--tier` override, a pick of the saved tier switches to it.
    #[test]
    fn a_pick_of_the_saved_tier_ends_a_startup_override() {
        let mut app = tier_app("desktop", &QualityTier::quest_high());
        frames(&mut app, 5);
        // Nothing switches at the first frame or because the settings differ.
        assert_eq!(running(&app), "quest-high");
        assert!(!governing(&app));
        pick(&mut app, "desktop");
        frames(&mut app, 2);
        assert_eq!(running(&app), "desktop");
        assert_eq!(saved(&app), "desktop");
        // Q-179: the build's default with `[general] governor` on.
        assert!(governing(&app));
    }

    // Q-240: only a pick switches; a write of `[general] tier` alone does not.
    #[test]
    fn a_settings_write_alone_leaves_the_running_tier() {
        let mut app = tier_app("desktop", &QualityTier::desktop());
        frames(&mut app, 2);
        app.world_mut().resource_mut::<SettingsRes>().0.general.tier = "quest-low".into();
        frames(&mut app, 2);
        assert_eq!(running(&app), "desktop");
    }

    // Q-240: picking the tier already running changes nothing, its knobs included.
    #[test]
    fn a_pick_of_the_running_tier_changes_nothing() {
        let mut app = tier_app("desktop", &QualityTier::quest_high());
        frames(&mut app, 2);
        app.world_mut()
            .resource_mut::<SplatTierState>()
            .render_scale = 0.7;
        pick(&mut app, "quest-high");
        frames(&mut app, 2);
        let s = app.world().resource::<SplatTierState>();
        assert_eq!((s.tier.name.as_str(), s.render_scale), ("quest-high", 0.7));
    }

    #[test]
    fn budget_is_one_period_less_1_5_ms() {
        assert!((budget_ms(60.0) - (1000.0 / 60.0 - 1.5)).abs() < 1e-5);
        assert!((budget_ms(90.0) - 9.611_111).abs() < 1e-4);
    }
}
