// SPDX-License-Identifier: Apache-2.0

//! Spike S-F1's device side (PRD #804, #805; §1.12 Headset display, D-58, E16's floor,
//! Appendix E), on Q-308's pattern: inert until the quest runner spawns an entity named
//! [`TRIGGER`]. It then logs the XR device's density-map features and properties once
//! (Q-T805.1), holds the governor off, and at render scale 1.0 and then E16's floor
//! (written into `SplatTierState.render_scale`, which the runner cannot name: Q-133)
//! steps `DisplaySettings.foveation` through off, low, medium and high. Each level
//! settles, then its first-eye statistics blocks (Q-125) are gathered beside counters
//! window `sf1 <scale> <level>` (Q-420) and summarized in one line. It judges nothing:
//! the verdict is PASS once the sequence completes, FAIL only on a timeout; the host's
//! `dv` test reads the lines (Q-454, Q-122).
// Q-454, Q-308, Q-420

use std::time::{Duration, Instant};

use bevy::app::{App, Last};
use bevy::log::info;
use bevy::prelude::{
    Commands, DetectChanges, Entity, IntoScheduleConfigs, Name, Query, Res, ResMut, Resource, not,
    resource_exists,
};
use bevy::render::renderer::{RenderAdapter, RenderDevice};
use ftgsxr_app::Governor;
use ftgsxr_render::{Foveation, SplatStats, SplatTierState, density_map_support};

use super::DisplaySettings;

/// The name of the entity the quest runner spawns to start the spike (Q-308).
// Q-308
pub(crate) const TRIGGER: &str = "ftgsxr-test: sf1";
/// The spike's name in the verdict line.
const NAME: &str = "sf1";
/// Render scale 1.0 and E16's floor.
const SCALES: [f32; 2] = [1.0, 0.6];
const LEVELS: [Foveation; 4] = [
    Foveation::Off,
    Foveation::Low,
    Foveation::Medium,
    Foveation::High,
];
/// How long a new level is left before its blocks count (Q-420).
const SETTLE: Duration = Duration::from_secs(2);
/// After a change of render scale, which re-makes the layer, a longer settle.
const SETTLE_SCALE: Duration = Duration::from_secs(4);
/// Blocks per window (Q-420).
const BLOCKS: usize = 120;
/// How long a window's blocks may take.
const WINDOW_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the last counters window is left to log before the verdict (Q-420).
const CLOSING: Duration = Duration::from_millis(500);
/// The counters window marker's prefix (`display::metrics`, Q-420).
const METRICS: &str = "ftgsxr-test: metrics ";

/// One statistics block's figures, ms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Block {
    pub(crate) cull: f32,
    pub(crate) sort: f32,
    pub(crate) draw: f32,
    pub(crate) total: f32,
    pub(crate) visible: u32,
    pub(crate) overflow: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Settle(usize, usize),
    Measure(usize, usize),
    Closing,
    Done,
}

/// What a frame asks of the system.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Out {
    Log(String),
    /// Set the tier's render scale.
    Scale(f32),
    /// Set `DisplaySettings.foveation`.
    Level(Foveation),
    /// Open the counters window of this label, or close the open one.
    Window(Option<String>),
    Verdict(bool, String),
}

/// The sequence, fed one frame at a time with the time since its phase began.
#[derive(Debug)]
pub(crate) struct Run {
    phase: Phase,
    blocks: Vec<Block>,
    done: Vec<String>,
}

/// The value at quantile `q` of `v` (the upper median at 0.5, Q-420).
fn quantile(v: &[f32], q: f32) -> f32 {
    let mut v = v.to_vec();
    v.sort_by(f32::total_cmp);
    let i = ((v.len() as f32 * q) as usize).min(v.len().saturating_sub(1));
    v.get(i).copied().unwrap_or(0.0)
}

fn label(s: usize, l: usize) -> String {
    format!("{:.1} {:?}", SCALES[s], LEVELS[l])
}

impl Run {
    /// The run, and the first frame's requests: scale 1.0 at off.
    pub(crate) fn start() -> (Run, Vec<Out>) {
        let run = Run {
            phase: Phase::Settle(0, 0),
            blocks: Vec::new(),
            done: Vec::new(),
        };
        (run, vec![Out::Scale(SCALES[0]), Out::Level(LEVELS[0])])
    }

    /// The scale the run holds now.
    pub(crate) fn scale(&self) -> Option<f32> {
        match self.phase {
            Phase::Settle(s, _) | Phase::Measure(s, _) => Some(SCALES[s]),
            _ => None,
        }
    }

    pub(crate) fn done(&self) -> bool {
        self.phase == Phase::Done
    }

    /// One frame: `since` the time since the phase began, `block` a new first-eye block.
    /// Returns the requests and whether the phase changed (its clock restarts).
    pub(crate) fn frame(&mut self, since: Duration, block: Option<Block>) -> (Vec<Out>, bool) {
        let mut out = Vec::new();
        let next = match self.phase {
            Phase::Settle(s, l) => {
                let settle = if l == 0 { SETTLE_SCALE } else { SETTLE };
                (since >= settle).then(|| {
                    self.blocks.clear();
                    out.push(Out::Window(Some(format!("sf1 {}", label(s, l)))));
                    Phase::Measure(s, l)
                })
            }
            Phase::Measure(s, l) => {
                if let Some(b) = block {
                    self.blocks.push(b);
                }
                let full = self.blocks.len() >= BLOCKS;
                (full || since > WINDOW_TIMEOUT).then(|| {
                    let line = self.summary(s, l, since);
                    out.push(Out::Log(format!("sf1: scale {line}")));
                    out.push(Out::Window(None));
                    if !full {
                        out.push(Out::Verdict(false, format!("too few blocks: {line}")));
                        return Phase::Done;
                    }
                    self.done.push(label(s, l));
                    if l + 1 < LEVELS.len() {
                        out.push(Out::Level(LEVELS[l + 1]));
                        Phase::Settle(s, l + 1)
                    } else if s + 1 < SCALES.len() {
                        out.push(Out::Scale(SCALES[s + 1]));
                        out.push(Out::Level(LEVELS[0]));
                        Phase::Settle(s + 1, 0)
                    } else {
                        Phase::Closing
                    }
                })
            }
            Phase::Closing => (since >= CLOSING).then(|| {
                out.push(Out::Verdict(true, format!("at {}", self.done.join(", "))));
                Phase::Done
            }),
            Phase::Done => None,
        };
        let changed = next.is_some();
        if let Some(p) = next {
            self.phase = p;
        }
        (out, changed)
    }

    /// `<scale> <level> draw median <ms> ms, p10 <ms> p90 <ms> min <ms> max <ms>, total
    /// median <ms> cull <ms> sort <ms> over <n> blocks in <s> s, visible <a>..<b>,
    /// overflow <bool>`.
    fn summary(&self, s: usize, l: usize, since: Duration) -> String {
        let b = &self.blocks;
        let of = |f: fn(&Block) -> f32| b.iter().map(f).collect::<Vec<_>>();
        let draw = of(|b| b.draw);
        format!(
            "{} draw median {} ms, p10 {} p90 {} min {} max {}, total median {} cull {} sort {} \
             over {} blocks in {:.2} s, visible {}..{}, overflow {}",
            label(s, l),
            quantile(&draw, 0.5),
            quantile(&draw, 0.1),
            quantile(&draw, 0.9),
            quantile(&draw, 0.0),
            quantile(&draw, 1.0),
            quantile(&of(|b| b.total), 0.5),
            quantile(&of(|b| b.cull), 0.5),
            quantile(&of(|b| b.sort), 0.5),
            b.len(),
            since.as_secs_f64(),
            b.iter().map(|b| b.visible).min().unwrap_or(0),
            b.iter().map(|b| b.visible).max().unwrap_or(0),
            b.iter().any(|b| b.overflow),
        )
    }
}

/// The spike's state, inserted when the runner's entity appears.
#[derive(Resource)]
struct Sf1 {
    run: Run,
    since: Instant,
    window: Option<Entity>,
}

/// Adds the spike's systems: inert until an entity named [`TRIGGER`] exists.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn add(app: &mut App) {
    app.add_systems(
        Last,
        (
            detect.run_if(not(resource_exists::<Sf1>)),
            run.run_if(resource_exists::<Sf1>),
        )
            .chain(),
    );
}

#[cfg_attr(test, allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn detect(
    names: Query<&Name>,
    adapter: Option<Res<RenderAdapter>>,
    device: Option<Res<RenderDevice>>,
    mut tier: Option<ResMut<SplatTierState>>,
    mut display: Option<ResMut<DisplaySettings>>,
    mut commands: Commands,
) {
    if !names.iter().any(|n| n.as_str() == TRIGGER) {
        return;
    }
    // Q-T805.1: the device's features and properties, once.
    let support = adapter
        .as_deref()
        .zip(device.as_deref())
        .and_then(|(a, d)| density_map_support(a, d.wgpu_device()));
    match support {
        Some(s) => info!("sf1: density map: {s}"),
        None => info!("sf1: density map: no Vulkan adapter"),
    }
    let (run, out) = Run::start();
    apply(&out, tier.as_deref_mut(), display.as_deref_mut());
    commands.insert_resource(Sf1 {
        run,
        since: Instant::now(),
        window: None,
    });
}

/// The scale and level requests; the rest are the caller's.
fn apply(out: &[Out], tier: Option<&mut SplatTierState>, display: Option<&mut DisplaySettings>) {
    for o in out {
        match o {
            Out::Scale(s) => {
                if let Some(t) = tier.as_deref() {
                    info!("sf1: render scale {s} (was {})", t.render_scale);
                }
            }
            Out::Level(l) => info!("sf1: foveation {l:?}"),
            _ => {}
        }
    }
    if let Some(t) = tier
        && let Some(s) = out.iter().rev().find_map(|o| match o {
            Out::Scale(s) => Some(*s),
            _ => None,
        })
    {
        t.render_scale = s;
    }
    if let Some(d) = display
        && let Some(l) = out.iter().rev().find_map(|o| match o {
            Out::Level(l) => Some(*l),
            _ => None,
        })
    {
        d.foveation = l;
    }
}

#[cfg_attr(test, allow(dead_code))]
fn run(
    mut test: ResMut<Sf1>,
    stats: Res<SplatStats>,
    mut tier: Option<ResMut<SplatTierState>>,
    mut display: Option<ResMut<DisplaySettings>>,
    gov: Option<ResMut<Governor>>,
    mut commands: Commands,
) {
    if test.run.done() {
        return;
    }
    // Held off, so no knob moves under the measurement (E16).
    if let Some(mut g) = gov
        && (g.enabled || g.budget_override_ms.is_some())
    {
        g.enabled = false;
        g.budget_override_ms = None;
    }
    // The scale held every frame, whatever else writes the tier state.
    if let (Some(s), Some(t)) = (test.run.scale(), tier.as_deref_mut())
        && t.render_scale != s
    {
        t.render_scale = s;
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
    apply(&out, tier.as_deref_mut(), display.as_deref_mut());
    for o in out {
        match o {
            Out::Log(l) => info!("{l}"),
            Out::Window(Some(label)) => {
                test.window = Some(commands.spawn(Name::new(format!("{METRICS}{label}"))).id());
            }
            Out::Window(None) => {
                if let Some(w) = test.window.take() {
                    commands.entity(w).despawn();
                }
            }
            Out::Verdict(pass, detail) => {
                let word = if pass { "PASS" } else { "FAIL" };
                info!("ftgsxr-test: {NAME} {word} {detail}");
            }
            Out::Scale(_) | Out::Level(_) => {}
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

    /// Runs one window to its end from its settle, returning its requests.
    fn window(r: &mut Run, settle: Duration) -> Vec<Out> {
        let mut out = r.frame(settle, None).0;
        for i in 0..BLOCKS {
            out.extend(r.frame(Duration::from_millis(i as u64), Some(B)).0);
        }
        out
    }

    #[test]
    fn run_steps_four_levels_at_each_scale_and_passes() {
        let (mut r, first) = Run::start();
        assert_eq!(first, [Out::Scale(1.0), Out::Level(Foveation::Off)]);
        let mut all = Vec::new();
        for i in 0..8 {
            let settle = if i % 4 == 0 { SETTLE_SCALE } else { SETTLE };
            all.extend(window(&mut r, settle));
        }
        let logs: Vec<_> = all
            .iter()
            .filter_map(|o| match o {
                Out::Log(l) => Some(l.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(logs.len(), 8);
        assert!(logs[0].starts_with(
            "sf1: scale 1.0 Off draw median 5 ms, p10 5 p90 5 min 5 max 5, total median 5.5"
        ));
        assert!(logs[7].starts_with("sf1: scale 0.6 High draw median"));
        let windows: Vec<_> = all
            .iter()
            .filter_map(|o| match o {
                Out::Window(Some(l)) => Some(l.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(windows[3], "sf1 1.0 High");
        assert_eq!(windows[4], "sf1 0.6 Off");
        assert!(all.contains(&Out::Scale(0.6)));
        assert_eq!(r.scale(), None);
        let (o, _) = r.frame(CLOSING, None);
        assert!(matches!(&o[..], [Out::Verdict(true, _)]), "{o:?}");
        assert!(r.done());
    }

    #[test]
    fn run_waits_out_the_settle_before_counting() {
        let (mut r, _) = Run::start();
        let (o, changed) = r.frame(SETTLE, Some(B));
        assert!(o.is_empty() && !changed, "the first settle is the scale's");
        let (o, changed) = r.frame(SETTLE_SCALE, None);
        assert_eq!(o, [Out::Window(Some("sf1 1.0 Off".into()))]);
        assert!(changed);
    }

    #[test]
    fn run_fails_a_window_without_its_blocks() {
        let (mut r, _) = Run::start();
        r.frame(SETTLE_SCALE, None);
        let (o, _) = r.frame(WINDOW_TIMEOUT + Duration::from_millis(1), Some(B));
        assert!(
            o.iter()
                .any(|o| matches!(o, Out::Verdict(false, d) if d.contains("too few blocks")))
        );
        assert!(r.done());
    }

    #[test]
    fn quantile_takes_the_upper_median_and_the_ends() {
        let v = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(quantile(&v, 0.5), 3.0);
        assert_eq!(quantile(&v, 0.0), 1.0);
        assert_eq!(quantile(&v, 1.0), 4.0);
        assert_eq!(quantile(&[], 0.5), 0.0);
    }
}
