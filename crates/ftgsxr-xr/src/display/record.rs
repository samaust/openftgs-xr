// SPDX-License-Identifier: Apache-2.0

//! A-10's record (§3.6 A-10, §1.13 Logs, Appendix E.1; Q-411): what the owner's
//! twenty minutes are judged from while the HUD's drawing is M-9's.
//!
//! - XR\_EXT\_performance\_settings is enabled for its notifications only (no performance
//!   level is ever requested). Each session logs `perf settings:
//!   XR_EXT_performance_settings enabled` (or `absent`), and each notification is a line
//!   `perf settings: <domain> <sub-domain> <from> -> <to>` (`cpu`/`gpu`;
//!   `compositing`/`rendering`/`thermal`; `normal`/`warning`/`impaired`), a warning when
//!   it enters a level above normal.
//! - Once a second (§1.13's rate limit), a line `perf record: second <i> frames <n> worst
//!   <ms> ms tier <name> scale <s> sh <c> sort <mode> foveation <level> loaded <n> dropped
//!   <d>`. The seconds are bins of the frames' predicted display times, counted from the
//!   first frame (one clock across sessions, so a second without a presented frame is a
//!   missing index); `frames` counts the frames the runtime asked to render; `worst` is
//!   the longest step between consecutive such frames' display times ending in the
//!   second; the knobs are the governor's (`SplatTierState`); `loaded` is `HudData`'s
//!   loaded primitives; `dropped` is the runtime's cumulative
//!   `compositor/dropped_frame_count` (XR\_META\_performance\_metrics, spike S1), `-`
//!   where the runtime has none.
// Q-411, Q-T735.1

use bevy::app::{App, Update};
use bevy::log::{info, warn};
use bevy::prelude::{Res, ResMut, Resource};
use bevy_mod_openxr::poll_events::{OxrEventHandlerExt, OxrEventIn};
use bevy_mod_openxr::resources::{OxrFrameState, OxrInstance};
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_xr::session::{XrSessionCreated, XrState};
use ftgsxr_app::HudData;
use ftgsxr_render::SplatTierState;
use openxr::sys;

use super::metrics::{self, Metrics};

/// One second of the display's clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bin {
    /// Seconds since the first presented frame's display time.
    pub(crate) index: u64,
    /// The presented frames whose display time falls in the second.
    pub(crate) frames: u32,
    /// The longest step to a frame of the second from the frame before it, in ns.
    pub(crate) worst_ns: i64,
}

/// The frames' display times binned by second.
#[derive(Debug, Default)]
pub(crate) struct Bins {
    origin: Option<i64>,
    last: Option<i64>,
    current: Option<Bin>,
}

const SECOND_NS: i64 = 1_000_000_000;

impl Bins {
    /// Takes a presented frame's predicted display time (ns); returns the second it
    /// finishes, if any. A time not after the last one is the same frame and is ignored.
    pub(crate) fn frame(&mut self, t: i64) -> Option<Bin> {
        if self.last.is_some_and(|l| t <= l) {
            return None;
        }
        let origin = *self.origin.get_or_insert(t);
        let step = self.last.map_or(0, |l| t - l);
        self.last = Some(t);
        let index = u64::try_from((t - origin) / SECOND_NS).unwrap_or(0);
        let finished = match self.current {
            Some(c) if c.index == index => None,
            other => {
                self.current = Some(Bin {
                    index,
                    frames: 0,
                    worst_ns: 0,
                });
                other
            }
        };
        let c = self
            .current
            .as_mut()
            .expect("invariant: a bin was just set");
        c.frames += 1;
        c.worst_ns = c.worst_ns.max(step);
        finished
    }
}

/// The record line of `bin`.
pub(crate) fn record_line(
    bin: &Bin,
    tier: Option<&SplatTierState>,
    loaded: u64,
    dropped: Option<f64>,
) -> String {
    let knobs = tier.map_or_else(
        || "tier - scale - sh - sort - foveation -".to_owned(),
        |t| {
            format!(
                "tier {} scale {} sh {} sort {:?} foveation {:?}",
                t.tier.name, t.render_scale, t.sh_cap, t.sort_mode, t.foveation
            )
        },
    );
    let dropped = dropped.map_or_else(|| "-".to_owned(), |d| format!("{d:.0}"));
    format!(
        "perf record: second {} frames {} worst {:.1} ms {knobs} loaded {loaded} dropped {dropped}",
        bin.index,
        bin.frames,
        bin.worst_ns as f64 / 1e6
    )
}

fn domain(d: sys::PerfSettingsDomainEXT) -> String {
    match d {
        sys::PerfSettingsDomainEXT::CPU => "cpu".to_owned(),
        sys::PerfSettingsDomainEXT::GPU => "gpu".to_owned(),
        other => format!("domain-{}", other.into_raw()),
    }
}

fn sub_domain(d: sys::PerfSettingsSubDomainEXT) -> String {
    match d {
        sys::PerfSettingsSubDomainEXT::COMPOSITING => "compositing".to_owned(),
        sys::PerfSettingsSubDomainEXT::RENDERING => "rendering".to_owned(),
        sys::PerfSettingsSubDomainEXT::THERMAL => "thermal".to_owned(),
        other => format!("sub-domain-{}", other.into_raw()),
    }
}

fn level(l: sys::PerfSettingsNotificationLevelEXT) -> String {
    match l {
        sys::PerfSettingsNotificationLevelEXT::NORMAL => "normal".to_owned(),
        sys::PerfSettingsNotificationLevelEXT::WARNING => "warning".to_owned(),
        sys::PerfSettingsNotificationLevelEXT::IMPAIRED => "impaired".to_owned(),
        other => format!("level-{}", other.into_raw()),
    }
}

/// A notification's line, and whether it enters a level above normal.
pub(crate) fn notification_line(
    d: sys::PerfSettingsDomainEXT,
    s: sys::PerfSettingsSubDomainEXT,
    from: sys::PerfSettingsNotificationLevelEXT,
    to: sys::PerfSettingsNotificationLevelEXT,
) -> (String, bool) {
    (
        format!(
            "perf settings: {} {} {} -> {}",
            domain(d),
            sub_domain(s),
            level(from),
            level(to)
        ),
        to != sys::PerfSettingsNotificationLevelEXT::NORMAL,
    )
}

#[derive(Resource, Default)]
struct Record {
    bins: Bins,
}

/// Adds the record's systems (`display`'s plugin).
pub(crate) fn add(app: &mut App) {
    app.init_resource::<Record>()
        .add_systems(XrSessionCreated, report_extension)
        .add_systems(Update, record)
        .add_oxr_event_handler(log_notification);
}

/// Each session says whether the notifications can arrive.
fn report_extension(instance: Option<Res<OxrInstance>>) {
    let on = instance.is_some_and(|i| i.exts().ext_performance_settings.is_some());
    info!(
        "perf settings: XR_EXT_performance_settings {}",
        if on { "enabled" } else { "absent" }
    );
}

fn log_notification(event: OxrEventIn) {
    if let openxr::Event::PerfSettingsEXT(e) = *event {
        let (line, raised) =
            notification_line(e.domain(), e.sub_domain(), e.from_level(), e.to_level());
        if raised {
            warn!("{line}");
        } else {
            info!("{line}");
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: one parameter per resource it reads"
)]
fn record(
    mut r: ResMut<Record>,
    mut m: ResMut<Metrics>,
    frame: Option<Res<OxrFrameState>>,
    state: Option<Res<XrState>>,
    instance: Option<Res<OxrInstance>>,
    session: Option<Res<OxrSession>>,
    tier: Option<Res<SplatTierState>>,
    hud: Option<Res<HudData>>,
) {
    if state.as_deref() != Some(&XrState::Running) {
        return;
    }
    let Some(frame) = frame.filter(|f| f.0.should_render) else {
        return;
    };
    let Some(bin) = r.bins.frame(frame.0.predicted_display_time.as_nanos()) else {
        return;
    };
    let dropped = match (instance, session) {
        (Some(i), Some(s)) => metrics::dropped_frames(&mut m, &i, &s),
        _ => None,
    };
    info!(
        "{}",
        record_line(&bin, tier.as_deref(), hud.map_or(0, |h| h.loaded), dropped)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ftgsxr_render::QualityTier;

    const MS: i64 = 1_000_000;

    /// 72 Hz for two seconds: 72 frames in each, the step 13.9 ms; the first second
    /// finishes at the first frame of the next.
    #[test]
    fn bins_count_the_frames_of_each_display_second() {
        let mut b = Bins::default();
        // 72 Hz, the period rounded up, as a period of whole nanoseconds at least 1/72 s.
        let period = SECOND_NS / 72 + 1;
        let mut out = Vec::new();
        for k in 0..=144 {
            out.extend(b.frame(5 * SECOND_NS + k * period + 7));
        }
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].index, out[0].frames), (0, 72));
        assert_eq!((out[1].index, out[1].frames), (1, 72));
        assert_eq!(out[1].worst_ns, period);
    }

    /// A late frame's step is the second's worst; a repeated time is the same frame; a
    /// second without a frame has no bin, so its index is missing.
    #[test]
    fn bins_keep_the_worst_step_and_skip_empty_seconds() {
        let mut b = Bins::default();
        assert_eq!(b.frame(0), None);
        assert_eq!(b.frame(14 * MS), None);
        assert_eq!(b.frame(14 * MS), None);
        assert_eq!(b.frame(70 * MS), None);
        let first = b.frame(2_500 * MS).expect("second 0 ends");
        assert_eq!(
            first,
            Bin {
                index: 0,
                frames: 3,
                worst_ns: 56 * MS
            }
        );
        let second = b.frame(3_000 * MS).expect("second 2 ends");
        assert_eq!(
            (second.index, second.frames, second.worst_ns),
            (2, 1, 2_430 * MS)
        );
    }

    /// The line A-10's verdict reads (`tests/a10_sustained_use.rs`).
    #[test]
    fn record_line_names_the_frames_the_knobs_and_the_counter() {
        let tier = SplatTierState::of(&QualityTier::quest_balanced());
        let bin = Bin {
            index: 12,
            frames: 72,
            worst_ns: 13_888_889,
        };
        assert_eq!(
            record_line(&bin, Some(&tier), 1_000_000, Some(437.0)),
            "perf record: second 12 frames 72 worst 13.9 ms tier quest-balanced scale 1 sh 2 \
             sort SharedCenter foveation Medium loaded 1000000 dropped 437"
        );
        assert!(
            record_line(&bin, None, 0, None)
                .ends_with("tier - scale - sh - sort - foveation - loaded 0 dropped -")
        );
    }

    #[test]
    fn notification_line_names_the_domain_and_the_levels() {
        use sys::PerfSettingsNotificationLevelEXT as L;
        let (line, raised) = notification_line(
            sys::PerfSettingsDomainEXT::GPU,
            sys::PerfSettingsSubDomainEXT::THERMAL,
            L::NORMAL,
            L::WARNING,
        );
        assert_eq!(line, "perf settings: gpu thermal normal -> warning");
        assert!(raised);
        let (line, raised) = notification_line(
            sys::PerfSettingsDomainEXT::CPU,
            sys::PerfSettingsSubDomainEXT::RENDERING,
            L::IMPAIRED,
            L::NORMAL,
        );
        assert_eq!(line, "perf settings: cpu rendering impaired -> normal");
        assert!(!raised);
    }
}
