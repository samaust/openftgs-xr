// SPDX-License-Identifier: Apache-2.0

//! R-17 on the device (§3.2, E16, D-47, §1.12; #730), as Q-431 reads it. It lives
//! here, not in `ftgsxr-quest`, because it reads `SplatTierState`, which the quest runner
//! cannot name (§2.1's band table); the runner loads the probe, keeps it ahead of the
//! left eye and starts the test by spawning an entity named [`TRIGGER`] (Q-308). The test
//! prints the verdict line `ftgsxr-test: r17 PASS|FAIL <detail>` itself (§3.4, Q-122).
//!
//! In `Last`, after the governor's `Update`, so a step is seen in the frame it is made:
//! the governor is held off for [`WARM_BLOCKS`] statistics blocks (a block is a frame in
//! which `SplatStats` changed and holds one, the governor's own count, Q-153); then, with
//! the knobs at the tier's own values and every warm-up block's GPU time over 1.05 ×
//! 0.1 ms, it is turned on with `budget_override_ms` at 0.1 ms. Each knob change is
//! logged with the blocks since the last one; [`HOLD_DOWN`] blocks without a change end
//! the way down, the override goes to 1,000 ms, and [`HOLD_UP`] blocks without a change
//! end the way up. The host driver judges the lines; the verdict is PASS once the
//! sequence completes, FAIL when it cannot start or finish.
// Q-431, Q-308

use std::time::{Duration, Instant};

use bevy::app::{App, Last};
use bevy::log::info;
use bevy::prelude::{
    Commands, DetectChanges, IntoScheduleConfigs, Name, Query, Res, ResMut, Resource, not,
    resource_exists,
};
use ftgsxr_app::{Governor, SettingsRes, XrDisplay};
use ftgsxr_render::{Foveation, SortMode, SplatStats, SplatTierState};

/// The name of the entity the quest runner spawns to start the test (Q-308).
// Q-308
pub(crate) const TRIGGER: &str = "ftgsxr-test: r17";
/// The test's name in the verdict line.
const NAME: &str = "r17";
/// The tier the test runs in (#730).
const TIER: &str = "quest-balanced";
/// The blocks the governor is held off for before the override.
const WARM_BLOCKS: u32 = 30;
/// The override on the way down and up (R-17), milliseconds.
const DOWN_MS: f32 = 0.1;
const UP_MS: f32 = 1000.0;
/// E16's thresholds on e.
const OVER: f32 = 1.05;
const UNDER: f32 = 0.8;
/// The blocks without a change that end the way down: four of E16's 10-frame streaks, so
/// a sort-mode step would have come.
const HOLD_DOWN: u32 = 40;
/// The blocks without a change that end the way up: two of E16's 120-frame streaks.
const HOLD_UP: u32 = 240;
/// How long the whole sequence may take once started.
const TIMEOUT: Duration = Duration::from_secs(180);

/// The governor's four knobs as `SplatTierState` holds them, the render scale in tenths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Knobs {
    scale_tenths: i32,
    foveation: Foveation,
    sh_cap: u8,
    sort: SortMode,
}

impl Knobs {
    /// The running knobs.
    fn of(s: &SplatTierState) -> Knobs {
        Knobs {
            scale_tenths: (s.render_scale * 10.0).round() as i32,
            foveation: s.foveation,
            sh_cap: s.sh_cap,
            sort: s.sort_mode,
        }
    }

    /// The running tier's own values (`SplatTierState::of`).
    fn tier_own(s: &SplatTierState) -> Knobs {
        Knobs::of(&SplatTierState::of(&s.tier))
    }

    fn scale(&self) -> String {
        format!("{}.{}", self.scale_tenths / 10, self.scale_tenths % 10)
    }

    /// `<Knob> <from> -> <to>` for each knob that differs (E16's names, as `Knob`'s).
    fn change(&self, to: &Knobs) -> String {
        let mut parts = Vec::new();
        if self.scale_tenths != to.scale_tenths {
            parts.push(format!("RenderScale {} -> {}", self.scale(), to.scale()));
        }
        if self.foveation != to.foveation {
            parts.push(format!(
                "Foveation {:?} -> {:?}",
                self.foveation, to.foveation
            ));
        }
        if self.sh_cap != to.sh_cap {
            parts.push(format!("ShCap {} -> {}", self.sh_cap, to.sh_cap));
        }
        if self.sort != to.sort {
            parts.push(format!("SortMode {:?} -> {:?}", self.sort, to.sort));
        }
        parts.join(", ")
    }

    fn describe(&self) -> String {
        format!(
            "render scale {}, foveation {:?}, SH cap {}, sort {:?}",
            self.scale(),
            self.foveation,
            self.sh_cap,
            self.sort
        )
    }
}

/// The test's phase.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    /// The governor held off; the warm-up blocks' totals gathered.
    Warm,
    /// The override at 0.1 ms.
    Down,
    /// The override at 1,000 ms.
    Up,
    Done,
}

/// What a frame asks of the system.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Out {
    /// A line to log.
    Log(String),
    /// Turn the governor on with this override, from the next frame.
    Budget(f32),
    /// The verdict line.
    Verdict(bool, String),
}

/// The sequence, fed one frame at a time (private tests drive it without a device).
#[derive(Debug)]
pub(crate) struct Run {
    phase: Phase,
    warm: Vec<f32>,
    last: Option<Knobs>,
    /// Blocks since the phase began or the last change.
    since: u32,
    /// The GPU totals' range since then.
    min: f32,
    max: f32,
    /// Changes in this phase.
    steps: u32,
    /// Blocks of this phase whose e was not on the phase's side of its threshold.
    outside: u32,
    /// The steps per phase, for the verdict.
    down_steps: u32,
}

impl Run {
    pub(crate) fn new() -> Run {
        Run {
            phase: Phase::Warm,
            warm: Vec::new(),
            last: None,
            since: 0,
            min: f32::INFINITY,
            max: 0.0,
            steps: 0,
            outside: 0,
            down_steps: 0,
        }
    }

    fn restart_range(&mut self) {
        self.since = 0;
        self.min = f32::INFINITY;
        self.max = 0.0;
    }

    /// One frame: `block` is the new statistics block's GPU total in ms, if any; `knobs`
    /// the running knobs and `own` the tier's own values.
    pub(crate) fn frame(&mut self, block: Option<f32>, knobs: Knobs, own: Knobs) -> Vec<Out> {
        let mut out = Vec::new();
        match self.phase {
            Phase::Warm => {
                if let Some(t) = block {
                    self.warm.push(t);
                }
                if self.warm.len() as u32 >= WARM_BLOCKS {
                    let min = self.warm.iter().copied().fold(f32::INFINITY, f32::min);
                    let mut sorted = self.warm.clone();
                    sorted.sort_by(f32::total_cmp);
                    let median = sorted[sorted.len() / 2];
                    let line = format!(
                        "warm-up {} blocks, GPU total median {median} ms, min {min} ms; knobs {}",
                        self.warm.len(),
                        knobs.describe()
                    );
                    out.push(Out::Log(format!("r17: {line}")));
                    if knobs != own {
                        self.phase = Phase::Done;
                        let detail = format!(
                            "the knobs are not the tier's own ({}): {line}",
                            own.describe()
                        );
                        out.push(Out::Verdict(false, detail));
                    } else if min <= OVER * DOWN_MS {
                        self.phase = Phase::Done;
                        let detail = format!(
                            "the probe's GPU time is not over {} ms: {line}",
                            OVER * DOWN_MS
                        );
                        out.push(Out::Verdict(false, detail));
                    } else {
                        self.phase = Phase::Down;
                        self.last = Some(knobs);
                        self.restart_range();
                        out.push(Out::Log(format!("r17: budget {DOWN_MS} ms")));
                        out.push(Out::Budget(DOWN_MS));
                    }
                }
            }
            Phase::Down | Phase::Up => {
                let down = self.phase == Phase::Down;
                if let Some(t) = block {
                    self.since += 1;
                    self.min = self.min.min(t);
                    self.max = self.max.max(t);
                    let budget = if down { DOWN_MS } else { UP_MS };
                    let e = t / budget;
                    if (down && e <= OVER) || (!down && e >= UNDER) {
                        self.outside += 1;
                    }
                }
                let last = self.last.unwrap_or(knobs);
                if knobs != last {
                    self.steps += 1;
                    out.push(Out::Log(format!(
                        "r17: step {} {} {} after {} blocks, GPU total {}..{} ms",
                        self.steps,
                        if down { "down" } else { "up" },
                        last.change(&knobs),
                        self.since,
                        self.min,
                        self.max
                    )));
                    self.last = Some(knobs);
                    self.restart_range();
                } else if down && self.since >= HOLD_DOWN {
                    out.push(Out::Log(format!(
                        "r17: down held {} blocks without a step after {} steps, {} blocks not over; knobs {}; budget {UP_MS} ms",
                        self.since,
                        self.steps,
                        self.outside,
                        knobs.describe()
                    )));
                    out.push(Out::Budget(UP_MS));
                    self.down_steps = self.steps;
                    self.phase = Phase::Up;
                    self.steps = 0;
                    self.outside = 0;
                    self.restart_range();
                } else if !down && self.since >= HOLD_UP {
                    out.push(Out::Log(format!(
                        "r17: up held {} blocks without a step after {} steps, {} blocks not under; knobs {}",
                        self.since,
                        self.steps,
                        self.outside,
                        knobs.describe()
                    )));
                    self.phase = Phase::Done;
                    out.push(Out::Verdict(
                        true,
                        format!("{} steps down, {} up", self.down_steps, self.steps),
                    ));
                }
            }
            Phase::Done => {}
        }
        out
    }

    fn governing(&self) -> bool {
        matches!(self.phase, Phase::Down | Phase::Up)
    }

    fn done(&self) -> bool {
        self.phase == Phase::Done
    }
}

/// The test's state, inserted when the quest runner's entity appears.
#[derive(Resource)]
struct R17 {
    run: Run,
    started: Instant,
}

/// Adds the test's systems: inert until an entity named [`TRIGGER`] exists.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn add(app: &mut App) {
    app.add_systems(
        Last,
        (
            detect.run_if(not(resource_exists::<R17>)),
            run.run_if(resource_exists::<R17>),
        )
            .chain(),
    );
}

#[cfg_attr(test, allow(dead_code))]
fn detect(names: Query<&Name>, mut commands: Commands) {
    if names.iter().any(|n| n.as_str() == TRIGGER) {
        info!("r17: started");
        commands.insert_resource(R17 {
            run: Run::new(),
            started: Instant::now(),
        });
    }
}

/// Why the run cannot show E16's full headset ladder, from the settings, the runtime and
/// the tier; `None` when it can. The settings are not written: `persist` would save them.
fn precondition(
    settings: Option<&SettingsRes>,
    display: Option<&XrDisplay>,
    tier: &SplatTierState,
) -> Option<String> {
    if tier.tier.name != TIER {
        return Some(format!(
            "the running tier is {}, not {TIER}",
            tier.tier.name
        ));
    }
    if let Some(s) = settings {
        if !s.0.general.governor {
            return Some("the settings turn the governor off ([general] governor)".to_owned());
        }
        if let Some(f) = &s.0.xr.foveation {
            return Some(format!("the settings pin [xr] foveation at {f:?}"));
        }
    }
    match display {
        None => Some("no XrDisplay: not on the headset's ladder".to_owned()),
        Some(d) if !d.foveation => Some("the runtime offers no foveation".to_owned()),
        Some(_) => None,
    }
}

#[cfg_attr(test, allow(dead_code))]
fn run(
    mut test: ResMut<R17>,
    stats: Res<SplatStats>,
    tier: Res<SplatTierState>,
    mut gov: ResMut<Governor>,
    display: Option<Res<XrDisplay>>,
    settings: Option<Res<SettingsRes>>,
) {
    if test.run.done() {
        return;
    }
    let verdict = |pass: bool, detail: &str| {
        let word = if pass { "PASS" } else { "FAIL" };
        info!("ftgsxr-test: {NAME} {word} {detail}");
    };
    if test.started.elapsed() > TIMEOUT {
        verdict(false, &format!("timed out: {:?}", test.run));
        test.run.phase = Phase::Done;
        return;
    }
    if test.run.governing() {
        if !gov.enabled {
            verdict(false, "the governor was turned off during the run");
            test.run.phase = Phase::Done;
            return;
        }
    } else if gov.enabled || gov.budget_override_ms.is_some() {
        // Held off until the override, so its streaks start at zero (E16, Q-178).
        gov.enabled = false;
        gov.budget_override_ms = None;
    }
    let block = stats
        .is_changed()
        .then_some(stats.0)
        .flatten()
        .map(|s| s.gpu_times_ms.total);
    let knobs = Knobs::of(&tier);
    let own = Knobs::tier_own(&tier);
    let starting = test.run.phase == Phase::Warm;
    for o in test.run.frame(block, knobs, own) {
        match o {
            Out::Log(l) => info!("{l}"),
            Out::Budget(ms) => {
                if starting
                    && let Some(why) = precondition(settings.as_deref(), display.as_deref(), &tier)
                {
                    verdict(false, &why);
                    test.run.phase = Phase::Done;
                    return;
                }
                gov.enabled = true;
                gov.budget_override_ms = Some(ms);
            }
            Out::Verdict(pass, detail) => verdict(pass, &detail),
        }
    }
    if test.run.done() && gov.budget_override_ms.is_some() {
        gov.budget_override_ms = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWN: Knobs = Knobs {
        scale_tenths: 10,
        foveation: Foveation::Medium,
        sh_cap: 2,
        sort: SortMode::SharedCenter,
    };

    /// Feeds `n` frames each with a block of `total` ms at `knobs`; the outputs.
    fn feed(r: &mut Run, n: u32, total: f32, knobs: Knobs) -> Vec<Out> {
        (0..n)
            .flat_map(|_| r.frame(Some(total), knobs, OWN))
            .collect()
    }

    fn logs(o: &[Out]) -> Vec<String> {
        o.iter()
            .filter_map(|o| match o {
                Out::Log(l) => Some(l.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn run_warms_up_then_sets_the_override_and_logs_each_change_with_its_blocks() {
        let mut r = Run::new();
        let o = feed(&mut r, WARM_BLOCKS - 1, 0.5, OWN);
        assert!(!o.iter().any(|o| matches!(o, Out::Budget(_))));
        let o = feed(&mut r, 1, 0.5, OWN);
        assert!(o.contains(&Out::Budget(0.1)), "{o:?}");
        // Ten blocks over, the tenth with the governor's step.
        feed(&mut r, 9, 0.5, OWN);
        let step = Knobs {
            scale_tenths: 9,
            ..OWN
        };
        let o = feed(&mut r, 1, 0.5, step);
        assert_eq!(
            logs(&o),
            ["r17: step 1 down RenderScale 1.0 -> 0.9 after 10 blocks, GPU total 0.5..0.5 ms"]
        );
        // A frame without a block counts nothing.
        assert!(r.frame(None, step, OWN).is_empty());
    }

    #[test]
    fn run_holds_then_goes_up_then_passes() {
        let mut r = Run::new();
        feed(&mut r, WARM_BLOCKS, 0.5, OWN);
        let floor = Knobs {
            scale_tenths: 6,
            foveation: Foveation::High,
            sh_cap: 1,
            ..OWN
        };
        let o = feed(&mut r, 1, 0.5, floor);
        assert!(logs(&o)[0].contains(
            "RenderScale 1.0 -> 0.6, Foveation Medium -> High, ShCap 2 -> 1 after 1 blocks"
        ));
        let o = feed(&mut r, HOLD_DOWN - 1, 0.5, floor);
        assert!(o.is_empty());
        let o = feed(&mut r, 1, 0.5, floor);
        assert!(o.contains(&Out::Budget(1000.0)), "{o:?}");
        assert!(logs(&o)[0].starts_with(
            "r17: down held 40 blocks without a step after 1 steps, 0 blocks not over"
        ));
        let o = feed(&mut r, 120, 0.5, OWN);
        assert!(logs(&o)[0].starts_with("r17: step 1 up"), "{o:?}");
        let o = feed(&mut r, HOLD_UP, 0.5, OWN);
        assert_eq!(
            o.last(),
            Some(&Out::Verdict(true, "1 steps down, 1 up".to_owned()))
        );
        assert!(r.done());
    }

    #[test]
    fn run_fails_a_probe_too_light_for_the_override_or_knobs_off_the_tier() {
        let mut r = Run::new();
        let o = feed(&mut r, WARM_BLOCKS, 0.1, OWN);
        assert!(matches!(o.last(), Some(Out::Verdict(false, d)) if d.contains("not over")));
        let mut r = Run::new();
        let off = Knobs { sh_cap: 1, ..OWN };
        let o = feed(&mut r, WARM_BLOCKS, 0.5, off);
        assert!(matches!(o.last(), Some(Out::Verdict(false, d)) if d.contains("tier's own")));
    }

    #[test]
    fn run_counts_the_blocks_outside_the_phases_side() {
        let mut r = Run::new();
        feed(&mut r, WARM_BLOCKS, 0.5, OWN);
        feed(&mut r, 5, 0.1, OWN);
        let o = feed(&mut r, HOLD_DOWN - 5, 0.5, OWN);
        assert!(logs(&o)[0].contains("5 blocks not over"), "{o:?}");
    }

    #[test]
    fn precondition_names_a_pinned_foveation_and_a_governor_turned_off() {
        let tier = SplatTierState::of(&ftgsxr_render::QualityTier::quest_balanced());
        let display = XrDisplay {
            foveation: true,
            ..XrDisplay::default()
        };
        let mut s = SettingsRes::default();
        assert_eq!(precondition(Some(&s), Some(&display), &tier), None);
        s.0.xr.foveation = Some("low".to_owned());
        assert!(precondition(Some(&s), Some(&display), &tier).is_some());
        let mut s = SettingsRes::default();
        s.0.general.governor = false;
        assert!(precondition(Some(&s), Some(&display), &tier).is_some());
        let no_fov = XrDisplay::default();
        assert!(precondition(None, Some(&no_fov), &tier).is_some());
        let desktop = SplatTierState::of(&ftgsxr_render::QualityTier::desktop());
        assert!(precondition(None, Some(&display), &desktop).is_some());
    }
}
