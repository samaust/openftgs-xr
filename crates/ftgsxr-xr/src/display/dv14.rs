// SPDX-License-Identifier: Apache-2.0

//! Dv-14's device side (§3.4 Dv-14, §1.12 Headset display, D-58, E16's floor; #810), on
//! Q-308's pattern and Q-454's setup: inert until the quest runner spawns an entity
//! named [`TRIGGER`]. It holds the governor off and, at render scale 1.0 and then E16's
//! floor of 0.6 (written into `SplatTierState.render_scale`, Q-133), runs four rounds of
//! one window at off and one at high through `DisplaySettings.foveation`, the order
//! alternating from round to round. Each window is 90 distinct statistics blocks
//! (Q-125) after 2 s to settle (4 s after a change of scale), logged as one line
//! `dv14: window <scale> <level> r<round> draw median …` with spike S-F1's summary,
//! whose `draw` spans both eyes' splat passes under `quest-balanced`'s shared center
//! (Q-403). It judges nothing: the verdict is PASS once the sequence completes, FAIL
//! only on a timeout; the host judges the margin (Q-454, Q-122).
// Q-454, Q-308, Q-457

use std::time::{Duration, Instant};

use bevy::app::{App, Last};
use bevy::log::info;
use bevy::prelude::{
    DetectChanges, IntoScheduleConfigs, Name, Query, Res, ResMut, Resource, not, resource_exists,
};
use ftgsxr_app::Governor;
use ftgsxr_render::{Foveation, SplatStats, SplatTierState};

use super::DisplaySettings;
use super::sf1::{Block, summary};

/// The name of the entity the quest runner spawns (Q-308).
pub(crate) const TRIGGER: &str = "ftgsxr-test: dv14";
const NAME: &str = "dv14";
/// Render scale 1.0 and E16's floor (Q-454).
const SCALES: [f32; 2] = [1.0, 0.6];
/// Rounds per scale (Q-454).
const ROUNDS: usize = 4;
/// Blocks per window (Q-454).
const BLOCKS: usize = 90;
const SETTLE: Duration = Duration::from_secs(2);
const SETTLE_SCALE: Duration = Duration::from_secs(4);
const WINDOW_TIMEOUT: Duration = Duration::from_secs(60);

/// One window of the plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Window {
    pub(crate) scale: f32,
    pub(crate) level: Foveation,
    pub(crate) round: usize,
    pub(crate) settle: Duration,
}

/// Q-454's windows in order: at each scale, rounds of off then high, high first in
/// odd rounds.
pub(crate) fn plan() -> Vec<Window> {
    let mut out = Vec::new();
    for scale in SCALES {
        for round in 0..ROUNDS {
            let pair = if round % 2 == 0 {
                [Foveation::Off, Foveation::High]
            } else {
                [Foveation::High, Foveation::Off]
            };
            for level in pair {
                let settle = if out.last().is_none_or(|w: &Window| w.scale != scale) {
                    SETTLE_SCALE
                } else {
                    SETTLE
                };
                out.push(Window {
                    scale,
                    level,
                    round,
                    settle,
                });
            }
        }
    }
    out
}

fn label(w: &Window) -> String {
    format!("{:.1} {:?} r{}", w.scale, w.level, w.round)
}

/// What a frame asks of the system.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Out {
    Log(String),
    Set(f32, Foveation),
    Verdict(bool, String),
}

/// The sequence, fed one frame at a time with the time since its phase began.
#[derive(Debug)]
pub(crate) struct Run {
    plan: Vec<Window>,
    at: usize,
    measuring: bool,
    blocks: Vec<Block>,
}

impl Run {
    /// The run and its first request.
    pub(crate) fn start() -> (Run, Vec<Out>) {
        let plan = plan();
        let first = Out::Set(plan[0].scale, plan[0].level);
        let run = Run {
            plan,
            at: 0,
            measuring: false,
            blocks: Vec::new(),
        };
        (run, vec![first])
    }

    /// The window under way, if any.
    pub(crate) fn current(&self) -> Option<Window> {
        self.plan.get(self.at).copied()
    }

    /// One frame; returns the requests and whether the phase changed.
    pub(crate) fn frame(&mut self, since: Duration, block: Option<Block>) -> (Vec<Out>, bool) {
        let Some(w) = self.current() else {
            return (Vec::new(), false);
        };
        if !self.measuring {
            if since < w.settle {
                return (Vec::new(), false);
            }
            self.measuring = true;
            self.blocks.clear();
            return (Vec::new(), true);
        }
        if let Some(b) = block {
            self.blocks.push(b);
        }
        let full = self.blocks.len() >= BLOCKS;
        if !full && since <= WINDOW_TIMEOUT {
            return (Vec::new(), false);
        }
        let line = summary(&label(&w), &self.blocks, since);
        let mut out = vec![Out::Log(format!("dv14: window {line}"))];
        if !full {
            out.push(Out::Verdict(false, format!("too few blocks: {line}")));
            self.at = self.plan.len();
            return (out, true);
        }
        self.measuring = false;
        self.at += 1;
        match self.current() {
            Some(n) => out.push(Out::Set(n.scale, n.level)),
            None => out.push(Out::Verdict(true, format!("{} windows", self.plan.len()))),
        }
        (out, true)
    }
}

#[derive(Resource)]
struct Dv14 {
    run: Run,
    since: Instant,
}

/// Adds the systems: inert until an entity named [`TRIGGER`] exists.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn add(app: &mut App) {
    app.add_systems(
        Last,
        (
            detect.run_if(not(resource_exists::<Dv14>)),
            run.run_if(resource_exists::<Dv14>),
        )
            .chain(),
    );
}

#[cfg_attr(test, allow(dead_code))]
fn detect(
    names: Query<&Name>,
    mut tier: Option<ResMut<SplatTierState>>,
    mut display: Option<ResMut<DisplaySettings>>,
    mut commands: bevy::prelude::Commands,
) {
    if !names.iter().any(|n| n.as_str() == TRIGGER) {
        return;
    }
    let (run, out) = Run::start();
    apply(&out, tier.as_deref_mut(), display.as_deref_mut());
    commands.insert_resource(Dv14 {
        run,
        since: Instant::now(),
    });
}

#[cfg_attr(test, allow(dead_code))]
fn apply(out: &[Out], tier: Option<&mut SplatTierState>, display: Option<&mut DisplaySettings>) {
    let Some((s, l)) = out.iter().find_map(|o| match o {
        Out::Set(s, l) => Some((*s, *l)),
        _ => None,
    }) else {
        return;
    };
    info!("dv14: render scale {s}, foveation {l:?}");
    if let Some(t) = tier
        && t.render_scale != s
    {
        t.render_scale = s;
    }
    if let Some(d) = display
        && d.foveation != l
    {
        d.foveation = l;
    }
}

#[cfg_attr(test, allow(dead_code))]
fn run(
    mut test: ResMut<Dv14>,
    stats: Res<SplatStats>,
    mut tier: Option<ResMut<SplatTierState>>,
    mut display: Option<ResMut<DisplaySettings>>,
    gov: Option<ResMut<Governor>>,
) {
    let Some(w) = test.run.current() else {
        return;
    };
    // Held off, so no knob moves under the measurement (E16).
    if let Some(mut g) = gov
        && (g.enabled || g.budget_override_ms.is_some())
    {
        g.enabled = false;
        g.budget_override_ms = None;
    }
    // The scale and level held every frame, whatever else writes them; written only
    // when they differ, since a write marks the tier state changed and `follow_settings`
    // then turns the governor back on (#810).
    if let Some(t) = tier.as_mut()
        && t.render_scale != w.scale
    {
        t.render_scale = w.scale;
    }
    if let Some(d) = display.as_mut()
        && d.foveation != w.level
    {
        d.foveation = w.level;
    }
    let block = stats
        .is_changed()
        .then_some(stats.0)
        .flatten()
        .map(|s| Block {
            cull: s.gpu_times_ms.cull,
            sort: s.gpu_times_ms.sort,
            draw: s.gpu_times_ms.draw,
            total: s.gpu_times_ms.total,
            visible: s.visible_splats,
            overflow: s.overflow,
        });
    let since = test.since.elapsed();
    let (out, changed) = test.run.frame(since, block);
    if out.iter().any(|o| matches!(o, Out::Set(..))) {
        apply(&out, tier.as_deref_mut(), display.as_deref_mut());
    }
    for o in out {
        match o {
            Out::Log(l) => info!("{l}"),
            Out::Verdict(pass, detail) => {
                let word = if pass { "PASS" } else { "FAIL" };
                info!("ftgsxr-test: {NAME} {word} {detail}");
            }
            Out::Set(..) => {}
        }
    }
    if changed {
        test.since = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: Block = Block {
        cull: 0.1,
        sort: 0.2,
        draw: 5.0,
        total: 5.5,
        visible: 100,
        overflow: false,
    };

    #[test]
    fn plan_alternates_off_and_high_over_four_rounds_at_each_scale() {
        let p = plan();
        assert_eq!(p.len(), 16);
        let at = |i: usize| (p[i].scale, p[i].level, p[i].round);
        assert_eq!(at(0), (1.0, Foveation::Off, 0));
        assert_eq!(at(1), (1.0, Foveation::High, 0));
        assert_eq!(at(2), (1.0, Foveation::High, 1));
        assert_eq!(at(3), (1.0, Foveation::Off, 1));
        assert_eq!(at(8), (0.6, Foveation::Off, 0));
        assert_eq!(at(15), (0.6, Foveation::Off, 3));
        assert_eq!(p[0].settle, SETTLE_SCALE);
        assert_eq!(p[8].settle, SETTLE_SCALE);
        assert_eq!(p[1].settle, SETTLE);
        assert_eq!(p[9].settle, SETTLE);
    }

    #[test]
    fn run_logs_each_window_and_passes_at_the_end() {
        let (mut r, first) = Run::start();
        assert_eq!(first, [Out::Set(1.0, Foveation::Off)]);
        let mut logs = Vec::new();
        let mut last = Vec::new();
        for w in plan() {
            assert!(
                r.frame(w.settle - Duration::from_millis(1), Some(B))
                    .0
                    .is_empty()
            );
            assert!(r.frame(w.settle, None).1);
            for _ in 0..BLOCKS {
                last = r.frame(Duration::from_millis(1), Some(B)).0;
            }
            logs.push(last[0].clone());
        }
        assert_eq!(
            logs[1],
            Out::Log(
                "dv14: window 1.0 High r0 draw median 5 ms, p10 5 p90 5 min 5 max 5, total \
                 median 5.5 cull 0.1 sort 0.2 over 90 blocks in 0.00 s, visible 100..100, \
                 overflow false"
                    .into()
            )
        );
        assert!(matches!(&last[1], Out::Verdict(true, _)));
        assert_eq!(r.current(), None);
    }

    /// Frames on which `run` marked the tier state or the display settings changed.
    #[derive(bevy::prelude::Resource, Default)]
    struct Marked(u32);

    fn mark(tier: Res<SplatTierState>, display: Res<DisplaySettings>, mut marked: ResMut<Marked>) {
        if tier.is_changed() || display.is_changed() {
            marked.0 += 1;
        }
    }

    /// Writing the tier state on a frame that changes nothing marks it changed, which
    /// turns the governor back on through `follow_settings` (#810): `run` must not.
    #[test]
    fn run_leaves_the_tier_state_and_display_unchanged_while_holding() {
        let mut app = App::new();
        let mut tier = SplatTierState::of(&ftgsxr_render::QualityTier::quest_balanced());
        tier.render_scale = 1.0;
        let display = DisplaySettings {
            refresh_rate: None,
            available_rates: Vec::new(),
            foveation: Foveation::Off,
            render_scale: 1.0,
        };
        let (run_state, _) = Run::start();
        app.insert_resource(tier)
            .insert_resource(display)
            .insert_resource(Governor {
                enabled: false,
                ..Governor::default()
            })
            .init_resource::<SplatStats>()
            .init_resource::<Marked>()
            .insert_resource(Dv14 {
                run: run_state,
                since: Instant::now(),
            })
            .add_systems(Last, (run, mark).chain());
        app.update();
        app.world_mut().resource_mut::<Marked>().0 = 0;
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(app.world().resource::<Marked>().0, 0);
    }

    #[test]
    fn run_fails_a_window_without_its_blocks() {
        let (mut r, _) = Run::start();
        r.frame(SETTLE_SCALE, None);
        let (o, _) = r.frame(WINDOW_TIMEOUT + Duration::from_millis(1), Some(B));
        assert!(matches!(&o[1], Out::Verdict(false, d) if d.contains("too few blocks")));
        assert_eq!(r.current(), None);
    }
}
