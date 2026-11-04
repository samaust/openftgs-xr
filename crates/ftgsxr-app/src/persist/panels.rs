// SPDX-License-Identifier: Apache-2.0

//! `persist`'s sections of ui's dock (§1.7 Panels, D-21, §1.11): the Display tab's
//! background, centers radius and colormap, the Environment tab's background and grid,
//! and the Settings tab's tables. Each edits `SettingsRes`, the source of truth that
//! `sync_settings` maps into the world and the debounce saves (Q-207); `[audio]` is
//! `audio`'s section and the tier and governor are `governor`'s.

use bevy::ecs::prelude::*;
use bevy_egui::egui;
use ftgsxr_core::LoopMode;
use ftgsxr_io::Settings;
use ftgsxr_render::SplatTierState;

use super::{SettingsRes, pick_tier};
use crate::library::LoadedModel;
use crate::manipulate::Selected;

/// The values `ftgsxr-io` accepts for the string keys (§1.11).
const TIERS: &[&str] = &[
    "conformance",
    "desktop",
    "quest-high",
    "quest-balanced",
    "quest-low",
];
const LOG_LEVELS: &[&str] = &["error", "warn", "info", "debug", "trace"];
const RENDER_MODES: &[&str] = &["splats", "centers", "both", "hidden"];
const COLORMAPS: &[&str] = &["dc", "time", "speed", "opacity"];
const ENVIRONMENT_MODES: &[&str] = &["vr", "passthrough"];
const LIGHTING_MODES: &[&str] = &["off", "manual", "auto-global", "auto-local"];
const FOVEATION: &[&str] = &["off", "low", "medium", "high", "dynamic"];
const SPEEDS: [f64; 12] = [
    -4.0, -2.0, -1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0,
];

/// Draws `draw` on a copy of the settings and writes it back only when it changed, so
/// an untouched panel never marks `SettingsRes` changed.
pub(crate) fn edit(
    world: &mut World,
    ui: &mut egui::Ui,
    draw: impl FnOnce(&mut Settings, &mut egui::Ui),
) {
    let Some(s) = world.get_resource::<SettingsRes>() else {
        return;
    };
    let mut next = s.0.clone();
    draw(&mut next, ui);
    if next != world.resource::<SettingsRes>().0 {
        world.resource_mut::<SettingsRes>().0 = next;
    }
}

pub(crate) fn num(ui: &mut egui::Ui, label: &str, v: &mut f32, lo: f32, hi: f32, step: f64) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(v).range(lo..=hi).speed(step));
    });
}

pub(crate) fn choice(ui: &mut egui::Ui, label: &str, v: &mut String, options: &[&str]) {
    egui::ComboBox::from_label(label)
        .selected_text(v.clone())
        .show_ui(ui, |ui| {
            for o in options {
                ui.selectable_value(v, (*o).to_string(), *o);
            }
        });
}

fn rgb(ui: &mut egui::Ui, label: &str, c: &mut [f32; 3]) {
    ui.horizontal(|ui| {
        ui.label(label);
        for v in c.iter_mut() {
            ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.01));
        }
    });
}

/// The background the selected instance (else the first) was trained against
/// (`render_conventions.background`, §1.4 Backgrounds), when one is loaded.
fn training_background(world: &mut World) -> Option<[f32; 3]> {
    let mut q = world.query::<(&LoadedModel, Option<&Selected>)>();
    let all: Vec<_> = q
        .iter(world)
        .map(|(m, s)| (m.info.conventions.background, s.is_some()))
        .collect();
    all.iter()
        .find(|(_, s)| *s)
        .or(all.first())
        .map(|(b, _)| *b)
}

/// The background's color and its training preset (§1.4 Backgrounds).
fn background(ui: &mut egui::Ui, c: &mut [f32; 3], training: Option<[f32; 3]>) {
    rgb(ui, "background", c);
    if let Some(t) = training
        && ui.button("training background").clicked()
    {
        *c = t;
    }
}

/// The Display tab: the quality tier, the centers' radius and colormap (on every instance
/// drawn with centers), new instances' mode, the overlays and the background. The tier
/// picker shows the running tier (`SplatTierState`), which a `--tier` override may make
/// differ from the saved one; a pick switches to it and saves it (Q-240).
// Q-240
pub(crate) fn display_section(world: &mut World, ui: &mut egui::Ui) {
    let training = training_background(world);
    if let Some(running) = world
        .get_resource::<SplatTierState>()
        .map(|t| t.tier.name.clone())
    {
        let mut picked = running.clone();
        choice(ui, "quality tier", &mut picked, TIERS);
        // egui reports no change for the option already selected: picking the running
        // tier changes nothing.
        if picked != running {
            pick_tier(world, picked);
        }
    }
    edit(world, ui, |s, ui| {
        let d = &mut s.display;
        choice(ui, "mode of new instances", &mut d.mode, RENDER_MODES);
        num(
            ui,
            "centers radius (px)",
            &mut d.centers_radius,
            1.0,
            8.0,
            0.1,
        );
        choice(ui, "colormap", &mut d.colormap, COLORMAPS);
        background(ui, &mut s.environment.background, training);
    });
}

/// The Environment tab on the desktop: background and grid (§1.7 Panels).
pub(crate) fn environment_section(world: &mut World, ui: &mut egui::Ui) {
    let training = training_background(world);
    edit(world, ui, |s, ui| {
        background(ui, &mut s.environment.background, training);
        ui.checkbox(&mut s.display.grid, "floor grid");
    });
}

fn loop_name(m: LoopMode) -> &'static str {
    match m {
        LoopMode::Loop => "loop",
        LoopMode::PingPong => "ping-pong",
        LoopMode::Once => "once",
    }
}

/// The Settings tab: §1.11's tables but `[audio]` (audio's section). `[general] tier`
/// shows the saved tier, and choosing another is a pick, which switches the running tier
/// as well (Q-240); `log_level` takes effect at the next start (Q-218).
// Q-218, Q-240
pub(crate) fn settings_tables(world: &mut World, ui: &mut egui::Ui) {
    let mut picked = None;
    edit(world, ui, |s, ui| {
        egui::CollapsingHeader::new("[general]").show(ui, |ui| {
            let g = &mut s.general;
            let saved = g.tier.clone();
            choice(ui, "tier", &mut g.tier, TIERS);
            if g.tier != saved {
                picked = Some(g.tier.clone());
            }
            ui.checkbox(&mut g.governor, "governor");
            num(ui, "ui_scale", &mut g.ui_scale, 0.8, 1.5, 0.01);
            choice(ui, "log_level (next start)", &mut g.log_level, LOG_LEVELS);
        });
        egui::CollapsingHeader::new("[[sources]]").show(ui, |ui| {
            for src in &s.sources {
                ui.label(format!("{} · {} · {}", src.kind, src.name, src.location));
            }
            ui.label("added and removed in the Library tab");
        });
        egui::CollapsingHeader::new("[playback]").show(ui, |ui| {
            let p = &mut s.playback;
            egui::ComboBox::from_label("speed")
                .selected_text(format!("{}", p.speed))
                .show_ui(ui, |ui| {
                    for v in SPEEDS {
                        ui.selectable_value(&mut p.speed, v, format!("{v}"));
                    }
                });
            egui::ComboBox::from_label("loop")
                .selected_text(loop_name(p.loop_mode))
                .show_ui(ui, |ui| {
                    for m in [LoopMode::Loop, LoopMode::PingPong, LoopMode::Once] {
                        ui.selectable_value(&mut p.loop_mode, m, loop_name(m));
                    }
                });
        });
        egui::CollapsingHeader::new("[display]").show(ui, |ui| {
            let d = &mut s.display;
            choice(ui, "mode", &mut d.mode, RENDER_MODES);
            num(ui, "centers_radius", &mut d.centers_radius, 1.0, 8.0, 0.1);
            choice(ui, "colormap", &mut d.colormap, COLORMAPS);
            ui.checkbox(&mut d.grid, "grid");
            ui.checkbox(&mut d.overlays.boxes, "overlays.boxes");
            ui.checkbox(&mut d.overlays.velocities, "overlays.velocities");
            ui.horizontal(|ui| {
                ui.label("overlays.velocity_every");
                ui.add(egui::DragValue::new(&mut d.overlays.velocity_every).range(1..=1000));
            });
            ui.checkbox(&mut d.overlays.time_tint, "overlays.time_tint");
        });
        egui::CollapsingHeader::new("[environment]").show(ui, |ui| {
            let e = &mut s.environment;
            choice(ui, "mode", &mut e.mode, ENVIRONMENT_MODES);
            rgb(ui, "background", &mut e.background);
            num(
                ui,
                "passthrough_brightness",
                &mut e.passthrough_brightness,
                -1.0,
                1.0,
                0.01,
            );
            num(
                ui,
                "passthrough_contrast",
                &mut e.passthrough_contrast,
                -1.0,
                1.0,
                0.01,
            );
            num(
                ui,
                "passthrough_saturation",
                &mut e.passthrough_saturation,
                -1.0,
                1.0,
                0.01,
            );
            ui.checkbox(&mut e.occlusion, "occlusion");
            ui.checkbox(&mut e.occlusion_soft, "occlusion_soft");
            num(
                ui,
                "occlusion_delta_m",
                &mut e.occlusion_delta_m,
                0.0,
                0.2,
                0.001,
            );
            ui.checkbox(&mut e.hand_removal, "hand_removal");
            choice(ui, "lighting_mode", &mut e.lighting_mode, LIGHTING_MODES);
            let m = &mut e.lighting_manual;
            num(
                ui,
                "lighting_manual.exposure_ev",
                &mut m.exposure_ev,
                -4.0,
                4.0,
                0.05,
            );
            num(
                ui,
                "lighting_manual.temperature",
                &mut m.temperature,
                -1.0,
                1.0,
                0.01,
            );
            num(ui, "lighting_manual.tint", &mut m.tint, -1.0, 1.0, 0.01);
            num(ui, "lighting_gamma", &mut e.lighting_gamma, 0.0, 1.0, 0.01);
            ui.checkbox(&mut e.contact_shadows, "contact_shadows");
        });
        egui::CollapsingHeader::new("[locomotion]").show(ui, |ui| {
            let l = &mut s.locomotion;
            num(ui, "speed_mps", &mut l.speed_mps, 1.0, 4.0, 0.05);
            ui.checkbox(&mut l.snap_turn, "snap_turn");
            num(ui, "snap_deg", &mut l.snap_deg, 1.0, 180.0, 1.0);
            num(ui, "smooth_dps", &mut l.smooth_dps, 1.0, 360.0, 1.0);
            ui.checkbox(&mut l.vignette, "vignette");
            ui.checkbox(&mut l.teleport, "teleport");
            num(
                ui,
                "height_offset_m",
                &mut l.height_offset_m,
                -0.5,
                0.5,
                0.01,
            );
            ui.checkbox(&mut l.left_handed, "left_handed");
            ui.checkbox(&mut l.reduced_motion, "reduced_motion");
        });
        egui::CollapsingHeader::new("[desktop]").show(ui, |ui| {
            let d = &mut s.desktop;
            num(ui, "fov_deg", &mut d.fov_deg, 40.0, 110.0, 0.5);
            num(ui, "fly_speed_mps", &mut d.fly_speed_mps, 0.1, 20.0, 0.05);
            egui::Grid::new("ftgsxr-keys").show(ui, |ui| {
                for (action, key) in &mut d.keys {
                    ui.label(action.as_str());
                    ui.text_edit_singleline(key);
                    ui.end_row();
                }
            });
        });
        egui::CollapsingHeader::new("[xr]").show(ui, |ui| {
            let x = &mut s.xr;
            let mut fov = x.foveation.clone().unwrap_or_else(|| "runtime".into());
            let mut options = vec!["runtime"];
            options.extend_from_slice(FOVEATION);
            choice(ui, "foveation", &mut fov, &options);
            x.foveation = (fov != "runtime").then_some(fov);
            ui.label(format!(
                "refresh_rate {} · render_scale {} · reference_space {}",
                x.refresh_rate.map_or("runtime".into(), |r| r.to_string()),
                x.render_scale.map_or("tier".into(), |r| r.to_string()),
                x.reference_space.join(", ")
            ));
        });
        egui::CollapsingHeader::new("[cache]").show(ui, |ui| {
            let mut gib = s.cache.max_bytes as f64 / f64::from(1u32 << 30);
            ui.horizontal(|ui| {
                ui.label("max_bytes");
                if ui
                    .add(
                        egui::DragValue::new(&mut gib)
                            .range(0.0..=4096.0)
                            .suffix(" GiB"),
                    )
                    .changed()
                {
                    s.cache.max_bytes = (gib * f64::from(1u32 << 30)) as u64;
                }
            });
        });
    });
    if let Some(tier) = picked {
        pick_tier(world, tier);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Draws `hook` once on a headless egui context.
    pub(crate) fn draw_once(world: &mut World, hook: fn(&mut World, &mut egui::Ui)) {
        let ctx = egui::Context::default();
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| hook(world, ui));
        out.textures_delta.clear();
    }

    #[test]
    fn an_untouched_section_leaves_the_settings_unchanged() {
        let mut world = World::new();
        world.insert_resource(SettingsRes::default());
        world.clear_trackers();
        for hook in [display_section, environment_section, settings_tables] {
            draw_once(&mut world, hook);
        }
        assert!(!world.resource_ref::<SettingsRes>().is_changed());
        assert_eq!(world.resource::<SettingsRes>().0, Settings::default());
    }

    /// Each text `shapes` draw, with its position.
    fn texts(shapes: &[egui::epaint::ClippedShape]) -> Vec<(String, egui::Pos2)> {
        fn walk(s: &egui::Shape, out: &mut Vec<(String, egui::Pos2)>) {
            match s {
                egui::Shape::Text(t) => out.push((t.galley.text().to_string(), t.pos)),
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        out
    }

    /// The Display tab under `--tier quest-high` with `desktop` saved (Q-240).
    fn overridden() -> World {
        let mut world = World::new();
        world.insert_resource(SettingsRes::default());
        world.insert_resource(SplatTierState::of(&ftgsxr_render::QualityTier::quest_high()));
        world.init_resource::<Messages<super::super::TierPick>>();
        world.clear_trackers();
        world
    }

    /// One frame of the Display tab with `events`; the texts drawn.
    fn frame(
        ctx: &egui::Context,
        world: &mut World,
        events: Vec<egui::Event>,
    ) -> Vec<(String, egui::Pos2)> {
        frame_of(ctx, world, events, display_section)
    }

    /// One frame of `hook` with `events`; the texts drawn.
    fn frame_of(
        ctx: &egui::Context,
        world: &mut World,
        events: Vec<egui::Event>,
        hook: fn(&mut World, &mut egui::Ui),
    ) -> Vec<(String, egui::Pos2)> {
        let input = egui::RawInput {
            events,
            ..egui::RawInput::default()
        };
        let mut out = ctx.run_ui(input, |ui| hook(world, ui));
        out.textures_delta.clear();
        texts(&out.shapes)
    }

    /// A click a little inside a text drawn at `at`.
    fn click(at: egui::Pos2) -> Vec<Vec<egui::Event>> {
        let p = at + egui::vec2(4.0, 4.0);
        let button = |pressed| egui::Event::PointerButton {
            pos: p,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        vec![
            vec![egui::Event::PointerMoved(p)],
            vec![button(true)],
            vec![button(false)],
        ]
    }

    fn at(texts: &[(String, egui::Pos2)], label: &str) -> egui::Pos2 {
        texts
            .iter()
            .filter(|(t, _)| t == label)
            .map(|(_, p)| *p)
            .next_back()
            .unwrap_or_else(|| panic!("{label} is drawn: {texts:?}"))
    }

    // Q-240 (#402): the picker's selected text is the running tier, not the saved one.
    #[test]
    fn the_tier_picker_shows_the_running_tier() {
        let mut world = overridden();
        let ctx = egui::Context::default();
        let drawn = frame(&ctx, &mut world, Vec::new());
        assert!(drawn.iter().any(|(t, _)| t == "quest-high"), "{drawn:?}");
        assert!(!drawn.iter().any(|(t, _)| t == "desktop"), "{drawn:?}");
        assert!(!world.resource_ref::<SettingsRes>().is_changed());
    }

    // Q-240 (#402): choosing the saved tier in the picker under an override is a pick; it
    // leaves the saved tier as it is (already `desktop`).
    #[test]
    fn choosing_the_saved_tier_under_an_override_picks_it() {
        let mut world = overridden();
        let ctx = egui::Context::default();
        let closed = frame(&ctx, &mut world, Vec::new());
        for events in click(at(&closed, "quest-high")) {
            frame(&ctx, &mut world, events);
        }
        // The open popup lists the tiers.
        let open = frame(&ctx, &mut world, Vec::new());
        for events in click(at(&open, "desktop")) {
            frame(&ctx, &mut world, events);
        }
        let picks: Vec<_> = world
            .resource_mut::<Messages<super::super::TierPick>>()
            .drain()
            .collect();
        assert_eq!(picks, vec![super::super::TierPick("desktop".into())]);
        assert_eq!(world.resource::<SettingsRes>().0.general.tier, "desktop");
    }

    // Q-218, Q-240: the Settings tab's `[general] tier` shows the saved tier, and choosing
    // another is a pick, saved and switched to.
    #[test]
    fn choosing_a_tier_in_the_settings_tab_picks_it() {
        let mut world = overridden();
        let ctx = egui::Context::default();
        let step = |at: egui::Pos2, world: &mut World| {
            for events in click(at) {
                frame_of(&ctx, world, events, settings_tables);
            }
            frame_of(&ctx, world, Vec::new(), settings_tables)
        };
        let closed = frame_of(&ctx, &mut world, Vec::new(), settings_tables);
        let general = step(at(&closed, "[general]"), &mut world);
        let open = step(at(&general, "desktop"), &mut world);
        step(at(&open, "quest-low"), &mut world);
        let picks: Vec<_> = world
            .resource_mut::<Messages<super::super::TierPick>>()
            .drain()
            .collect();
        assert_eq!(picks, vec![super::super::TierPick("quest-low".into())]);
        assert_eq!(world.resource::<SettingsRes>().0.general.tier, "quest-low");
    }
}
