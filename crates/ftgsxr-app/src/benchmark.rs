// SPDX-License-Identifier: Apache-2.0

//! The benchmark files and their shared pieces (§3.5, Q-67, Q-103): the `ftgsxr-bench`
//! file's results, budgets and text, the machine fields, the synthetic models and the
//! JSON reader and writer, shared by `headless`'s `bench` (before the session, Q-406)
//! and the session benchmarks `AppPlugin` runs (`session`, Q-406, Q-407). Private to
//! the crate and ordered before `plugin`, so `plugin` reaches it without depending on
//! `headless`, which comes after it (Q-223).

use std::fs;

use bevy::log::warn;
use ftgsxr_core::{Mat3, Quat, Vec3};

pub(crate) mod json;
pub(crate) mod session;
pub(crate) mod synthetic;

use json::Json;

/// How a budget's value meets its limit (Q-103).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rule {
    /// value ≥ limit; higher is better.
    AtLeast,
    /// value ≤ limit; lower is better.
    AtMost,
    /// |value / limit − 1| ≤ 1 %; lower is better.
    Within1Percent,
}

impl Rule {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Rule::AtLeast => "at_least",
            Rule::AtMost => "at_most",
            Rule::Within1Percent => "within_1_percent",
        }
    }

    /// The rule a benchmark file names (`name`'s inverse, Q-440).
    pub(crate) fn from_name(name: &str) -> Option<Rule> {
        [Rule::AtLeast, Rule::AtMost, Rule::Within1Percent]
            .into_iter()
            .find(|r| r.name() == name)
    }

    pub(crate) fn passes(self, value: f64, limit: f64) -> bool {
        match self {
            Rule::AtLeast => value >= limit,
            Rule::AtMost => value <= limit,
            Rule::Within1Percent => (value / limit - 1.0).abs() <= 0.01,
        }
    }

    /// Whether `value` is more than 10 % worse than `baseline` (§3.5). A
    /// `within_1_percent` budget (B-6) never is: it is deterministic, and its own 1 %
    /// budget is the check (Q-103).
    pub(crate) fn regressed(self, value: f64, baseline: f64) -> bool {
        match self {
            Rule::AtLeast => value < 0.9 * baseline,
            Rule::AtMost => value > 1.1 * baseline,
            Rule::Within1Percent => false,
        }
    }
}

/// One measured figure: `{setup, measure, value, unit}`, or a size the device refused,
/// `value` null and the reason in `skipped` (Q-413, Q-417).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Figure {
    pub(crate) setup: Json,
    pub(crate) measure: &'static str,
    pub(crate) value: f64,
    pub(crate) unit: &'static str,
    pub(crate) skipped: Option<String>,
}

impl Figure {
    // Q-417
    pub(crate) fn to_json(&self) -> Json {
        let value = match self.skipped {
            Some(_) => Json::Null,
            None => Json::Num(self.value),
        };
        let mut json = Json::obj([
            ("setup", self.setup.clone()),
            ("measure", Json::str(self.measure)),
            ("value", value),
            ("unit", Json::str(self.unit)),
        ]);
        if let (Json::Obj(members), Some(reason)) = (&mut json, &self.skipped) {
            members.push(("skipped".to_owned(), Json::str(reason.clone())));
        }
        json
    }
}

/// One budgeted figure: `{id, setup, measure, rule, limit, value, pass}`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Budget {
    pub(crate) id: &'static str,
    pub(crate) setup: Json,
    pub(crate) measure: &'static str,
    pub(crate) rule: Rule,
    pub(crate) limit: f64,
    pub(crate) value: f64,
}

impl Budget {
    pub(crate) fn pass(&self) -> bool {
        self.rule.passes(self.value, self.limit)
    }

    pub(crate) fn to_json(&self) -> Json {
        Json::obj([
            ("id", Json::str(self.id)),
            ("setup", self.setup.clone()),
            ("measure", Json::str(self.measure)),
            ("rule", Json::str(self.rule.name())),
            ("limit", Json::Num(self.limit)),
            ("value", Json::Num(self.value)),
            ("pass", Json::Bool(self.pass())),
        ])
    }
}

/// The benchmark file's text, two-space indented with a final newline (Q-67, Q-103).
// Q-67, Q-103
pub(crate) fn report_json(
    name: &str,
    gpu: &str,
    profile: &str,
    measured: &Measured,
    baseline: Json,
    notes: Vec<String>,
    headset: Option<&str>,
) -> String {
    let file = Json::obj([
        ("format", Json::str("ftgsxr-bench")),
        ("version", Json::Num(1.0)),
        ("bench", Json::str(name)),
        ("machine", machine(gpu, headset)),
        (
            "versions",
            Json::obj([
                ("ftgsxr", Json::str(env!("CARGO_PKG_VERSION"))),
                ("bevy", Json::str(BEVY_VERSION)),
                ("wgpu", Json::str(WGPU_VERSION)),
            ]),
        ),
        ("profile", Json::str(profile)),
        (
            "results",
            Json::Arr(measured.figures.iter().map(Figure::to_json).collect()),
        ),
        (
            "budgets",
            Json::Arr(measured.budgets.iter().map(Budget::to_json).collect()),
        ),
        ("baseline", baseline),
        (
            "notes",
            Json::Arr(notes.into_iter().map(Json::str).collect()),
        ),
    ]);
    let mut json = file.pretty();
    json.push('\n');
    json
}

/// The pinned Bevy (§2.3, the workspace's `[workspace.dependencies]`), which no runtime
/// API reports; `versions_match_the_lock_file` keeps it true.
pub(crate) const BEVY_VERSION: &str = "0.19.1";
/// The wgpu in the build (Bevy's pin, Q-73), as `BEVY_VERSION`.
pub(crate) const WGPU_VERSION: &str = "29.0.4";

/// The machine fields (§3.5, Q-103): on the headset (Q-406) `headset` is its build
/// fingerprint, after the others, and a field the device cannot give is `unknown`
/// (Q-417), as `driver` is without NVIDIA's file.
// Q-103, Q-406, Q-417
pub(crate) fn machine(gpu: &str, headset: Option<&str>) -> Json {
    let first_line = |path: &str| {
        fs::read_to_string(path)
            .ok()
            .and_then(|t| t.lines().next().map(|l| l.trim().to_owned()))
    };
    let driver = first_line("/proc/driver/nvidia/version").unwrap_or_else(|| "unknown".into());
    let os_name = fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|t| {
            t.lines()
                .find_map(|l| l.strip_prefix("PRETTY_NAME="))
                .map(|v| v.trim_matches('"').to_owned())
        })
        .unwrap_or_else(|| "unknown".into());
    let os = match first_line("/proc/sys/kernel/osrelease") {
        Some(kernel) => format!("{os_name} (kernel {kernel})"),
        None => os_name,
    };
    let cpu = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".into());
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut fields = Json::obj([
        ("gpu", Json::str(gpu)),
        ("driver", Json::str(driver)),
        ("os", Json::str(os)),
        ("cpu", Json::str(cpu)),
        ("threads", Json::Num(threads as f64)),
    ]);
    if let (Json::Obj(members), Some(h)) = (&mut fields, headset) {
        members.push(("headset".to_owned(), Json::str(h)));
    }
    fields
}

/// A benchmark's figures, budgets and notes.
#[derive(Default)]
pub(crate) struct Measured {
    pub(crate) figures: Vec<Figure>,
    pub(crate) budgets: Vec<Budget>,
    pub(crate) notes: Vec<&'static str>,
}

impl Measured {
    pub(crate) fn figure(
        &mut self,
        setup: &Json,
        measure: &'static str,
        value: f64,
        unit: &'static str,
    ) {
        self.figures.push(Figure {
            setup: setup.clone(),
            measure,
            value,
            unit,
            skipped: None,
        });
    }

    /// A size the device refused (Q-413): one result, without a value or a budget.
    // Q-413, Q-417
    pub(crate) fn skip(
        &mut self,
        setup: &Json,
        measure: &'static str,
        unit: &'static str,
        reason: String,
    ) {
        warn!("ftgsxr bench: {} skipped: {reason}", setup.compact());
        self.figures.push(Figure {
            setup: setup.clone(),
            measure,
            value: 0.0,
            unit,
            skipped: Some(reason),
        });
    }
}

/// The stage-space rotation of a camera (looking along −z with y up) that looks along
/// `forward` with y up.
pub(crate) fn look_at(forward: Vec3) -> Quat {
    let back = -forward.normalize();
    let right = Vec3::Y.cross(back).normalize();
    let up = back.cross(right);
    Quat::from_mat3(&Mat3::from_cols(right, up, back))
}

/// The median of `values` (sorted in place); 0 when empty.
pub(crate) fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}
