// SPDX-License-Identifier: Apache-2.0

//! The Scene tab's trim row (§1.7 Trim, E17): off, box or sphere, the box's six limits
//! named through the orientation preset or the sphere's center and radius, as sliders
//! with numeric fields, "Reset trim", and the notice when the trim keeps nothing.

use bevy::ecs::prelude::*;
use bevy_egui::egui;
use ftgsxr_core::{Aabb, Preset, Trim, TrimMode};
use ftgsxr_render::SplatBounds;

use super::dock::{Out, button};
use crate::intent::{InstanceAction, Intent};

/// What the row shows of one instance: its trim, its boxes and its untrimmed B_m.
#[derive(Clone, Copy)]
pub(super) struct TrimInfo {
    pub(super) trim: Option<Trim>,
    pub(super) bounds: Option<SplatBounds>,
    pub(super) untrimmed: Aabb,
}

/// The six box limits in the panel's order left, right, bottom, top, back, front, each
/// named and bound to (axis, max) of the model's frame through the preset (E17's table);
/// a custom preset names the limits x, y and z min and max.
pub(super) fn trim_limit_labels(preset: Preset) -> [(&'static str, usize, bool); 6] {
    match preset {
        Preset::YUp => [
            ("left", 0, false),
            ("right", 0, true),
            ("bottom", 1, false),
            ("top", 1, true),
            ("back", 2, false),
            ("front", 2, true),
        ],
        Preset::ZUp => [
            ("left", 0, false),
            ("right", 0, true),
            ("bottom", 2, false),
            ("top", 2, true),
            ("back", 1, true),
            ("front", 1, false),
        ],
        Preset::NegYUp => [
            ("left", 0, false),
            ("right", 0, true),
            ("bottom", 1, true),
            ("top", 1, false),
            ("back", 2, true),
            ("front", 2, false),
        ],
        Preset::Custom(_) => [
            ("x min", 0, false),
            ("x max", 0, true),
            ("y min", 1, false),
            ("y max", 1, true),
            ("z min", 2, false),
            ("z max", 2, true),
        ],
    }
}

/// The limit's model-frame name, "x min" … "z max".
fn axis_name(axis: usize, max: bool) -> String {
    format!(
        "{} {}",
        ["x", "y", "z"][axis],
        if max { "max" } else { "min" }
    )
}

/// §1.7's slider ranges from the untrimmed B_m: each axis extended by 10 % of the longest
/// side beyond each face; the radius from 10⁻³ of that side to 1.1 times the half
/// diagonal.
pub(super) fn slider_ranges(b: &Aabb) -> ([(f32, f32); 3], (f32, f32)) {
    let l = b.longest_side();
    let pad = 0.1 * l;
    let axes = std::array::from_fn(|a| (b.min[a] - pad, b.max[a] + pad));
    (axes, (1e-3 * l, 1.1 * 0.5 * b.size().length()))
}

/// One slider with its numeric field: the slider spans `range`, the field takes any
/// finite value (`SliderClamping::Never`). Whether the value changed.
fn slider(ui: &mut egui::Ui, label: &str, v: &mut f32, range: (f32, f32)) -> bool {
    let before = *v;
    let changed = ui
        .add(
            egui::Slider::new(v, range.0..=range.1)
                .clamping(egui::SliderClamping::Never)
                .text(label),
        )
        .changed();
    if !v.is_finite() {
        *v = before;
        return false;
    }
    changed
}

/// The trim row of instance `e`, oriented by `preset`; its edits go out as
/// `InstanceAction::SetTrim` and `ResetTrim`.
pub(super) fn trim_row(
    ui: &mut egui::Ui,
    e: Entity,
    preset: Preset,
    info: TrimInfo,
    out: &mut Vec<Out>,
    hv: &mut Option<egui::Id>,
) {
    // E17: the initial shapes until a trim exists; switching modes keeps both shapes.
    let now = info.trim.unwrap_or_else(|| Trim::initial(&info.untrimmed));
    let mut next = now;
    ui.horizontal(|ui| {
        ui.label("trim");
        for (label, mode) in [
            ("off", TrimMode::Off),
            ("box", TrimMode::Box),
            ("sphere", TrimMode::Sphere),
        ] {
            if ui.selectable_label(now.mode == mode, label).clicked() {
                next.mode = mode;
            }
        }
        if button(ui, "Reset trim", out, hv) {
            out.push(Out::Intent(Intent::Instance {
                instance: e,
                action: InstanceAction::ResetTrim,
            }));
        }
    });
    let (axes, radius) = slider_ranges(&info.untrimmed);
    match now.mode {
        TrimMode::Off => {}
        TrimMode::Box => {
            ui.label("model units");
            for (label, axis, max) in trim_limit_labels(preset) {
                let mut v = if max {
                    now.box_max[axis]
                } else {
                    now.box_min[axis]
                };
                let text = if label.contains(' ') {
                    label.to_string()
                } else {
                    format!("{label} ({})", axis_name(axis, max))
                };
                if slider(ui, &text, &mut v, axes[axis]) {
                    next.set_box_limit(axis, max, v);
                }
            }
        }
        TrimMode::Sphere => {
            ui.label("model units");
            let mut c = now.center;
            for (a, name) in ["center x", "center y", "center z"].into_iter().enumerate() {
                if slider(ui, name, &mut c[a], axes[a]) {
                    next.center = c;
                }
            }
            let mut r = now.radius;
            if slider(ui, "radius", &mut r, radius) {
                // §1.7: the radius stays at least 10⁻³ of the longest side.
                next.radius = r.max(radius.0);
            }
        }
    }
    if now.mode != TrimMode::Off && info.bounds.is_some_and(|b| b.kept_none) {
        ui.colored_label(ui.visuals().warn_fg_color, "Trim keeps no primitives");
    }
    if next != now {
        out.push(Out::Intent(Intent::Instance {
            instance: e,
            action: InstanceAction::SetTrim(next),
        }));
    }
}
