// SPDX-License-Identifier: Apache-2.0

//! The screen's UI (§1.7 Panels): the bar above the viewport, the dock on the left with
//! its seven tabs, the toast line and the confirmation dialogs. Drawn by one exclusive
//! system, which collects what the user did and sends it after the frame.

use bevy::camera::visibility::Visibility;
use bevy::ecs::prelude::*;
use bevy::math::{EulerRot, Quat};
use bevy::transform::components::Transform;
use bevy_egui::{EguiContext, PrimaryEguiContext, egui};
use ftgsxr_core::{EndRule, LoopMode, PlayState, Preset};
use ftgsxr_io::{EntryKind, SourceSpec, Url};
use ftgsxr_render::{Colormap, RenderMode, SplatMode, SplatTime};

use super::cameras::{self, JumpToCamera, TrainingCameras, TrainingView};
use super::{
    Confirm, DockHooks, InstanceLighting, RecentScenes, SceneLoadConfirmed, Tab, ToastLevel, UiCue,
    UiState,
};
use crate::intent::{InstanceAction, Intent, OverlayToggle, PlaybackCommand, UiCommand};
use crate::library::{EntryState, Library, LibraryCommand, LoadedModel, LocateEntry};
use crate::manipulate::{FIT_RANGE_M, PoseExtras, SetPoseExtras};
use crate::platform::PickKind;
use crate::playback::{Playback, RangeSetting, Scrubbing};

/// What the user did this frame.
enum Out {
    Intent(Intent),
    Library(LibraryCommand),
    Locate(usize),
    Answer(bool),
    Cue(UiCue),
    Jump(JumpToCamera),
    Toast(String),
}

/// §1.5's render modes, as the Scene and Display tabs offer them; the centers'
/// parameters are placeholders that `persist` replaces with `[display]`'s (Q-218).
const MODES: [(&str, RenderMode); 4] = [
    ("Splats", RenderMode::Splats),
    (
        "Centers",
        RenderMode::Centers {
            pixel_radius: 3.0,
            colormap: Colormap::Dc,
        },
    ),
    (
        "Both",
        RenderMode::Both {
            pixel_radius: 3.0,
            colormap: Colormap::Dc,
        },
    ),
    ("Hidden", RenderMode::Hidden),
];

/// One instance's render mode row (§1.5): `current` shown selected; the mode chosen,
/// when it is another. The row edits the mode only, never the instance's visibility.
fn mode_row(ui: &mut egui::Ui, current: RenderMode) -> Option<RenderMode> {
    let mut chosen = None;
    ui.horizontal(|ui| {
        for (label, m) in MODES {
            let on = std::mem::discriminant(&current) == std::mem::discriminant(&m);
            if ui.selectable_label(on, label).clicked() && !on {
                chosen = Some(m);
            }
        }
    });
    chosen
}

/// §1.6's speed presets.
const SPEEDS: [f64; 6] = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0];

/// The dock, the bar, the toast line and the dialogs on the primary context.
pub(super) fn draw(
    world: &mut World,
    mut font: Local<Option<bool>>,
    mut hovered: Local<Option<egui::Id>>,
) {
    let Some(ctx) = world
        .query_filtered::<&mut EguiContext, With<PrimaryEguiContext>>()
        .iter_mut(world)
        .next()
        .map(|mut c| c.get_mut().clone())
    else {
        return;
    };
    // Q-211
    let has_font = *font.get_or_insert_with(|| super::load_system_font(&ctx));
    {
        let mut state = world.resource_mut::<UiState>();
        if state.hidden_frames > 0 {
            state.hidden_frames -= 1;
            return;
        }
        if !state.visible || !has_font {
            return;
        }
    }
    let mut out = Vec::new();
    let mut now_hovered = None;
    // bevy_egui 0.42's panels show inside a root `Ui` over the viewport.
    let mut root = egui::Ui::new(
        ctx.clone(),
        "ftgsxr-viewport".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );
    bar(world, &mut root, &mut out, &mut now_hovered);
    toast_line(world, &mut root);
    dock(world, &mut root, &mut out, &mut now_hovered);
    dialog(world, &ctx, &mut out);
    if now_hovered.is_some() && now_hovered != *hovered {
        out.push(Out::Cue(UiCue::Hover));
    }
    *hovered = now_hovered;
    for o in out {
        match o {
            Out::Intent(i) => {
                world.write_message(i);
            }
            Out::Library(c) => {
                world.write_message(c);
            }
            Out::Locate(i) => {
                world.write_message(LocateEntry(i));
            }
            Out::Cue(c) => {
                world.write_message(c);
            }
            Out::Jump(j) => {
                world.write_message(j);
            }
            Out::Toast(text) => {
                world.write_message(super::ToastRequest::error(text));
            }
            Out::Answer(yes) => {
                let a = world.resource_mut::<UiState>().answer(yes);
                if let Some(i) = a.intent {
                    world.write_message(i);
                }
                if let Some(p) = a.scene {
                    world.write_message(SceneLoadConfirmed(p));
                }
                world.write_message(a.cue);
            }
        }
    }
}

/// A button that reports its hover and its click's sound.
fn button(
    ui: &mut egui::Ui,
    text: &str,
    out: &mut Vec<Out>,
    hovered: &mut Option<egui::Id>,
) -> bool {
    let r = ui.button(text);
    if r.hovered() {
        *hovered = Some(r.id);
    }
    let clicked = r.clicked();
    if clicked {
        out.push(Out::Cue(UiCue::Click));
    }
    clicked
}

/// The bar above the viewport: the dock's toggle, play and pause, the time readout.
fn bar(world: &mut World, ctx: &mut egui::Ui, out: &mut Vec<Out>, hv: &mut Option<egui::Id>) {
    let (t, len, playing) = world
        .get_resource::<Playback>()
        .map_or((0.0, 0.0, false), |p| {
            (
                p.clock.time(),
                p.clock.length(),
                p.clock.state() == PlayState::Playing,
            )
        });
    let readout = selected_readout(world, t);
    egui::Panel::top("ftgsxr-bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let open = world.resource::<UiState>().dock_open;
            if button(ui, if open { "◀" } else { "▶" }, out, hv) {
                world.resource_mut::<UiState>().dock_open = !open;
            }
            if button(ui, if playing { "Pause" } else { "Play" }, out, hv) {
                out.push(Out::Intent(Intent::Playback(PlaybackCommand::Toggle)));
            }
            if button(ui, "−1", out, hv) {
                out.push(Out::Intent(Intent::Playback(PlaybackCommand::StepFrames(
                    -1,
                ))));
            }
            if button(ui, "+1", out, hv) {
                out.push(Out::Intent(Intent::Playback(PlaybackCommand::StepFrames(
                    1,
                ))));
            }
            ui.label(format!("T = {t:.3} s / {len:.3} s"));
            if let Some((text, italic)) = readout {
                let mut r = egui::RichText::new(text);
                if italic {
                    r = r.italics();
                }
                ui.label(r);
            }
        });
    });
}

/// The selected instance's time in its unit (three decimals) and frame number (§1.6),
/// in italics while extrapolating.
fn selected_readout(world: &mut World, t: f64) -> Option<(String, bool)> {
    let mut q = world.query_filtered::<(&LoadedModel, Option<&ftgsxr_render::SplatTime>), With<crate::manipulate::Selected>>();
    let (m, st) = q.iter(world).next()?;
    let tau = st.map_or(0.0, |s| s.tau);
    let frame = if m.frame_step_s > 0.0 {
        (t / m.frame_step_s).floor() as i64
    } else {
        0
    };
    Some((
        format!(
            "{}: τ = {tau:.3} {}, frame {frame}",
            m.name, m.info.time.time_unit
        ),
        st.is_some_and(|s| s.extrapolating),
    ))
}

/// The toast line at the bottom of the window.
fn toast_line(world: &mut World, ctx: &mut egui::Ui) {
    let line = world.resource::<UiState>().line.clone();
    if line.is_empty() {
        return;
    }
    egui::Panel::bottom("ftgsxr-toasts").show(ctx, |ui| {
        for (t, _) in line.iter().rev().take(3) {
            let color = match t.level {
                ToastLevel::Info => ui.visuals().text_color(),
                ToastLevel::Warning => ui.visuals().warn_fg_color,
                ToastLevel::Error => ui.visuals().error_fg_color,
            };
            ui.colored_label(color, &t.text);
        }
    });
}

/// §1.7 Panels: the dock's narrowest width, in points.
const DOCK_MIN_WIDTH: f32 = 160.0;
/// §1.7 Panels: the dock's widest, as a share of the window's width.
const DOCK_MAX_SHARE: f32 = 0.6;

/// The dock's panel at `width` points, kept between 160 points and 60 % of a window
/// `window` points wide; below 267 points the 160-point floor wins (§1.7 Panels, Q-251).
// Q-251
fn dock_panel(width: f32, window: f32) -> egui::Panel {
    let max = (DOCK_MAX_SHARE * window).max(DOCK_MIN_WIDTH);
    egui::Panel::left("ftgsxr-dock")
        .resizable(true)
        .default_size(width)
        .size_range(DOCK_MIN_WIDTH..=max)
}

/// The dock and its tabs; its width is kept in `UiState`, and `persist` saves it with
/// whether it is open (§1.7 Panels, Q-252).
fn dock(world: &mut World, ctx: &mut egui::Ui, out: &mut Vec<Out>, hv: &mut Option<egui::Id>) {
    let (open, width, tab) = {
        let s = world.resource::<UiState>();
        (s.dock_open, s.dock_width, s.tab)
    };
    if !open {
        return;
    }
    let window = ctx.ctx().viewport_rect().width();
    let r = dock_panel(width, window).show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            for t in Tab::ALL {
                let r = ui.selectable_label(t == tab, format!("{t:?}"));
                if r.hovered() {
                    *hv = Some(r.id);
                }
                if r.clicked() {
                    world.resource_mut::<UiState>().tab = t;
                    out.push(Out::Cue(UiCue::Click));
                }
            }
        });
        ui.separator();
        egui::ScrollArea::vertical().show(ui, |ui| {
            match tab {
                Tab::Library => {
                    library_tab(world, ui, out, hv);
                    request_replay(world, ui, out, hv);
                }
                Tab::Scene => scene_tab(world, ui, out, hv),
                Tab::Playback => playback_tab(world, ui, out, hv),
                Tab::Display => display_tab(world, ui, out, hv),
                // §1.1, §1.7 Panels, §1.15 E-XR-UNSUPPORTED: no passthrough control on
                // the desktop (absent, not disabled); `persist` adds the background
                // and the grid.
                Tab::Environment => {}
                Tab::Settings => {
                    ui.label("Settings (§1.11)");
                }
                Tab::Help => {
                    ui.label(format!("openftgs-xr {}", env!("CARGO_PKG_VERSION")));
                    ui.label("F1 Help · F2 hide the UI · F3 HUD · F11 fullscreen");
                }
            }
            let hooks = world
                .resource::<DockHooks>()
                .0
                .get(&tab)
                .cloned()
                .unwrap_or_default();
            for h in hooks {
                ui.separator();
                h(world, ui);
            }
        });
    });
    world.resource_mut::<UiState>().dock_width = r.response.rect.width();
}

/// The source a Library tab entry names: an `http(s)` URL is an HTTP index, an absolute
/// path a local folder; anything else none. Its label is the URL's host or the folder's
/// name.
pub(crate) fn source_from_input(text: &str) -> Option<SourceSpec> {
    let text = text.trim();
    if let Ok(url) = Url::parse(text)
        && matches!(url.scheme(), "http" | "https")
    {
        return Some(SourceSpec {
            kind: "http".into(),
            name: url.host_str().unwrap_or(text).to_owned(),
            location: text.to_owned(),
            flavor: None,
        });
    }
    let path = std::path::Path::new(text);
    if path.is_absolute() {
        let name = path
            .file_name()
            .map_or_else(|| text.to_owned(), |n| n.to_string_lossy().into_owned());
        return Some(SourceSpec {
            kind: "dir".into(),
            name,
            location: text.to_owned(),
            flavor: None,
        });
    }
    None
}

fn kind_index(k: EntryKind) -> Option<usize> {
    match k {
        EntryKind::Exchange => Some(0),
        EntryKind::Checkpoint => Some(1),
        EntryKind::Ply => Some(2),
        EntryKind::Dir => Some(3),
        EntryKind::Other => None,
    }
}

/// Library: the sources, the filters, the entries with their states.
fn library_tab(
    world: &mut World,
    ui: &mut egui::Ui,
    out: &mut Vec<Out>,
    hv: &mut Option<egui::Id>,
) {
    let Some(library) = world.get_resource::<Library>() else {
        return;
    };
    let sources: Vec<String> = library
        .sources
        .iter()
        .map(|(s, _)| format!("{s:?}"))
        .collect();
    let entries = library.entries.clone();
    for (i, s) in sources.iter().enumerate() {
        ui.horizontal(|ui| {
            ui.label(s);
            if button(ui, "Refresh", out, hv) {
                out.push(Out::Library(LibraryCommand::Refresh(i)));
            }
            // The platform's models directory (the first, Q-49) stays.
            if i > 0 && button(ui, "Remove", out, hv) {
                out.push(Out::Library(LibraryCommand::RemoveSource(i)));
            }
        });
    }
    // §1.5 Sources, §1.7 Panels: an HTTP index URL or a local folder (Q-160).
    ui.horizontal(|ui| {
        let mut state = world.resource_mut::<UiState>();
        ui.add(
            egui::TextEdit::singleline(&mut state.new_source)
                .hint_text("https://… index or /path/to/folder"),
        );
        let spec = source_from_input(&state.new_source);
        if ui
            .add_enabled(spec.is_some(), egui::Button::new("Add source"))
            .clicked()
            && let Some(spec) = spec
        {
            state.new_source.clear();
            out.push(Out::Cue(UiCue::Click));
            out.push(Out::Library(LibraryCommand::AddSource(spec)));
        }
    });
    ui.horizontal(|ui| {
        if button(ui, "Open files…", out, hv) {
            out.push(Out::Library(LibraryCommand::Pick(PickKind::Files)));
        }
        if button(ui, "Add folder…", out, hv) {
            out.push(Out::Library(LibraryCommand::Pick(PickKind::Folder)));
        }
    });
    let mut state = world.resource_mut::<UiState>();
    ui.horizontal(|ui| {
        ui.label("Name");
        ui.text_edit_singleline(&mut state.filter_name);
    });
    ui.horizontal(|ui| {
        for (i, k) in ["ftgs-if", "checkpoint", "PLY", "folders"]
            .iter()
            .enumerate()
        {
            ui.checkbox(&mut state.filter_kinds[i], *k);
        }
    });
    ui.checkbox(&mut state.reduce, "Reduce to budget");
    let (name, kinds, reduce) = (
        state.filter_name.to_lowercase(),
        state.filter_kinds,
        state.reduce,
    );
    ui.separator();
    for (i, e) in entries.iter().enumerate() {
        if !e.entry.name.to_lowercase().contains(&name)
            || kind_index(e.entry.kind).is_some_and(|k| !kinds[k])
        {
            continue;
        }
        ui.horizontal(|ui| {
            ui.label(&e.entry.name);
            match &e.state {
                EntryState::Listed => {
                    if button(ui, "Load", out, hv) {
                        out.push(Out::Library(LibraryCommand::Load { entry: i, reduce }));
                    }
                }
                EntryState::Fetching(p) => {
                    let f = p.total.map_or(0.0, |t| p.done as f32 / t.max(1) as f32);
                    ui.add(egui::ProgressBar::new(f).desired_width(80.0));
                    if button(ui, "Cancel", out, hv) {
                        out.push(Out::Library(LibraryCommand::Cancel(i)));
                    }
                }
                EntryState::Loading => {
                    ui.label("loading");
                }
                EntryState::Loaded(_) => {
                    ui.label("loaded");
                }
                EntryState::Failed(text) => {
                    ui.colored_label(ui.visuals().error_fg_color, text);
                    if button(ui, "Retry", out, hv) {
                        out.push(Out::Library(LibraryCommand::Load { entry: i, reduce }));
                    }
                }
                EntryState::Missing => {
                    ui.label("missing");
                    // Q-225: only a file of the entry's identity is accepted.
                    if button(ui, "Locate…", out, hv) {
                        out.push(Out::Locate(i));
                    }
                }
            }
        });
    }
}

/// Scene: the instances, their visibility, mode, pose, time offset and end rule; save
/// and load.
fn scene_tab(world: &mut World, ui: &mut egui::Ui, out: &mut Vec<Out>, hv: &mut Option<egui::Id>) {
    ui.horizontal(|ui| {
        if button(ui, "Save scene", out, hv) {
            out.push(Out::Intent(Intent::Ui(UiCommand::SaveScene)));
        }
        if button(ui, "Load scene…", out, hv) {
            world.resource_mut::<UiState>().confirm = Some(Confirm::LoadScene(None));
        }
    });
    let mut q = world.query::<(
        Entity,
        &mut LoadedModel,
        &mut Transform,
        Option<&Visibility>,
        Option<&SplatMode>,
        Option<&InstanceLighting>,
        Option<&PoseExtras>,
        Option<&SplatTime>,
    )>();
    let mut ask = None;
    let mut lightings = Vec::new();
    let mut extras_out = Vec::new();
    for (e, mut m, mut tf, vis, mode, light, extras, time) in q.iter_mut(world) {
        ui.separator();
        // §1.5: the instance's name is editable; §1.6: an "extrapolating" badge.
        ui.horizontal(|ui| {
            let mut name = m.name.clone();
            if ui.text_edit_singleline(&mut name).changed() && name != m.name {
                m.name = name;
            }
            if m.reduced {
                ui.label("(reduced)");
            }
            if time.is_some_and(|t| t.extrapolating) {
                ui.colored_label(ui.visuals().warn_fg_color, "extrapolating");
            }
        });
        let mut visible = vis != Some(&Visibility::Hidden);
        if ui.checkbox(&mut visible, "Visible").changed() {
            out.push(Out::Intent(Intent::Instance {
                instance: e,
                action: InstanceAction::SetVisible(visible),
            }));
        }
        let current = mode.map_or(RenderMode::Splats, |m| m.0);
        if let Some(m) = mode_row(ui, current) {
            out.push(Out::Intent(Intent::Instance {
                instance: e,
                action: InstanceAction::SetMode(m),
            }));
        }
        // §1.7: metres, yaw-pitch-roll in degrees, a factor.
        let mut t = tf.translation;
        let (y, p, r) = tf.rotation.to_euler(EulerRot::YXZ);
        let mut ypr = [y.to_degrees(), p.to_degrees(), r.to_degrees()];
        let mut s = tf.scale.x;
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("t");
            for v in [&mut t.x, &mut t.y, &mut t.z] {
                changed |= ui
                    .add(egui::DragValue::new(v).speed(0.01).suffix(" m"))
                    .changed();
            }
        });
        ui.horizontal(|ui| {
            ui.label("ypr");
            for v in &mut ypr {
                changed |= ui
                    .add(egui::DragValue::new(v).speed(0.5).suffix("°"))
                    .changed();
            }
        });
        ui.horizontal(|ui| {
            ui.label("scale");
            changed |= ui
                .add(egui::DragValue::new(&mut s).speed(0.01).range(1e-3..=1e3))
                .changed();
        });
        if changed {
            tf.translation = t;
            tf.rotation = Quat::from_euler(
                EulerRot::YXZ,
                ypr[0].to_radians(),
                ypr[1].to_radians(),
                ypr[2].to_radians(),
            );
            tf.scale = bevy::math::Vec3::splat(s);
        }
        let mut offset = m.time.offset_s;
        // §1.6: a single-timestamp instance has no time controls.
        let timed = !m.time.span.is_single();
        if timed {
            ui.horizontal(|ui| {
                ui.label("offset");
                if ui
                    .add(egui::DragValue::new(&mut offset).speed(0.01).suffix(" s"))
                    .changed()
                {
                    out.push(Out::Intent(Intent::Playback(PlaybackCommand::Offset {
                        instance: e,
                        offset_s: offset,
                    })));
                }
                let hold = m.time.end_rule == EndRule::Hold;
                if ui.selectable_label(hold, "hold").clicked() {
                    out.push(Out::Intent(Intent::Playback(PlaybackCommand::EndRule {
                        instance: e,
                        rule: EndRule::Hold,
                    })));
                }
                if ui.selectable_label(!hold, "extrapolate").clicked() {
                    out.push(Out::Intent(Intent::Playback(PlaybackCommand::EndRule {
                        instance: e,
                        rule: EndRule::Extrapolate,
                    })));
                }
            });
        }
        // E14's color transform as its manual form (Q-218, Q-250).
        let now = light.map(|c| c.0.clone()).unwrap_or_default();
        let mut l = now.clone();
        ui.horizontal(|ui| {
            ui.label("exposure EV");
            ui.add(
                egui::DragValue::new(&mut l.exposure_ev)
                    .range(-4.0..=4.0)
                    .speed(0.01),
            );
            ui.label("temperature");
            ui.add(
                egui::DragValue::new(&mut l.temperature)
                    .range(-1.0..=1.0)
                    .speed(0.01),
            );
            ui.label("tint");
            ui.add(
                egui::DragValue::new(&mut l.tint)
                    .range(-1.0..=1.0)
                    .speed(0.01),
            );
            if ui.small_button("reset").clicked() {
                l = Default::default();
            }
        });
        if l != now {
            lightings.push((e, l));
        }
        // §1.5 Poses: the orientation preset and L_fit the fit uses.
        let x = extras.copied().unwrap_or(PoseExtras {
            preset: Preset::YUp,
            fit_size_m: 1.0,
        });
        let mut next = x;
        ui.horizontal(|ui| {
            ui.label("up");
            for (label, p) in [
                ("Y-up", Preset::YUp),
                ("Z-up", Preset::ZUp),
                ("−Y-up", Preset::NegYUp),
                ("custom", Preset::Custom(tf.rotation)),
            ] {
                let on = std::mem::discriminant(&x.preset) == std::mem::discriminant(&p);
                // "custom" takes the instance's current rotation.
                if ui.selectable_label(on, label).clicked() {
                    next.preset = p;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("fit size");
            ui.add(
                egui::DragValue::new(&mut next.fit_size_m)
                    .range(FIT_RANGE_M.0..=FIT_RANGE_M.1)
                    .speed(0.01)
                    .suffix(" m"),
            );
        });
        if next != x {
            extras_out.push(SetPoseExtras {
                instance: e,
                preset: next.preset,
                fit_size_m: next.fit_size_m,
            });
        }
        ui.horizontal_wrapped(|ui| {
            for (label, action) in [
                ("Select", InstanceAction::Select),
                ("Fit", InstanceAction::Fit),
                ("Snap to floor", InstanceAction::SnapToFloor),
                ("Save pose", InstanceAction::SavePose),
                ("Reset pose", InstanceAction::ResetPose),
                // §1.5: "the panel's second button" forgets the saved pose.
                ("Forget pose", InstanceAction::ForgetPose),
            ] {
                if button(ui, label, out, hv) {
                    out.push(Out::Intent(Intent::Instance {
                        instance: e,
                        action,
                    }));
                }
            }
            if button(ui, "Unload", out, hv) {
                // Q-199: asked first.
                ask = Some(e);
            }
        });
    }
    for x in extras_out {
        world.write_message(x);
    }
    training_cameras(world, ui, out, hv);
    for (e, l) in lightings {
        if let Ok(mut ec) = world.get_entity_mut(e) {
            ec.insert(InstanceLighting::bundle(l));
        }
    }
    if let Some(e) = ask {
        world.resource_mut::<UiState>().confirm = Some(Confirm::Unload(e));
    }
}

/// §1.7 Training cameras: each instance's manifest images by split, a jump per image,
/// the frustums toggle.
fn training_cameras(
    world: &mut World,
    ui: &mut egui::Ui,
    out: &mut Vec<Out>,
    hv: &mut Option<egui::Id>,
) {
    let mut q = world.query::<(Entity, &LoadedModel, &TrainingCameras)>();
    let sets: Vec<(Entity, String, TrainingCameras)> = q
        .iter(world)
        .map(|(e, m, t)| (e, m.name.clone(), t.clone()))
        .collect();
    if sets.is_empty() {
        return;
    }
    let mut view = world.resource::<TrainingView>().clone();
    ui.separator();
    ui.strong("Training cameras");
    ui.checkbox(&mut view.frustums, "frustums");
    ui.horizontal(|ui| {
        for (label, split) in [
            ("all", None),
            ("train", Some("train")),
            ("test", Some("test")),
        ] {
            let split = split.map(str::to_string);
            if ui.selectable_label(view.split == split, label).clicked() {
                view.split = split;
            }
        }
    });
    for (e, name, tc) in sets {
        egui::CollapsingHeader::new(format!("{name}: {}", tc.path.display()))
            .id_salt(("ftgsxr-cams", e))
            .show(ui, |ui| {
                for img in &tc.manifest.images {
                    if view.split.as_ref().is_some_and(|s| *s != img.split) {
                        continue;
                    }
                    let Some(camera) = img.camera_override.or_else(|| {
                        tc.manifest
                            .cameras
                            .iter()
                            .find(|(n, _)| *n == img.camera)
                            .map(|(_, c)| *c)
                    }) else {
                        continue;
                    };
                    let label = format!(
                        "{} · {} · {:.3} {}",
                        img.camera, img.split, img.timestamp, tc.manifest.time_unit
                    );
                    if button(ui, &label, out, hv) {
                        out.push(Out::Jump(JumpToCamera {
                            instance: e,
                            camera,
                            timestamp: img.timestamp,
                            time_unit: tc.manifest.time_unit.clone(),
                        }));
                    }
                }
            });
    }
    let mut res = world.resource_mut::<TrainingView>();
    if res.frustums != view.frustums || res.split != view.split {
        res.frustums = view.frustums;
        res.split = view.split;
    }
}

/// §1.7: a `ftgs-render-request` found in a directory source, replayed item by item on
/// the selected instance, else the first loaded one (Q-219).
fn request_replay(
    world: &mut World,
    ui: &mut egui::Ui,
    out: &mut Vec<Out>,
    hv: &mut Option<egui::Id>,
) {
    let view = world.resource::<TrainingView>().clone();
    if view.requests.is_empty() && view.replay.is_none() {
        return;
    }
    ui.separator();
    ui.strong("Render requests");
    let mut opened = None;
    for p in &view.requests {
        let name = p.file_name().map_or_else(
            || p.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        if button(ui, &format!("Open {name}"), out, hv) {
            opened = Some(p.clone());
        }
    }
    let mut replay = view.replay.clone();
    if let Some(p) = opened {
        match cameras::open_request(&p) {
            Ok(r) => replay = Some(r),
            Err(text) => out.push(Out::Toast(text)),
        }
    }
    let (mut step, mut close) = (None, false);
    // Q-219: the instance the replay steps on, found once for the label and for stepping.
    let target = replay.as_ref().and_then(|_| replay_target(world));
    if let Some(r) = &replay {
        let n = r.request.items.len();
        ui.horizontal(|ui| {
            if button(ui, "◀", out, hv) {
                step = Some((r.item + n - 1) % n);
            }
            let item = &r.request.items[r.item];
            ui.label(format!(
                "{} / {n} · {} · {:.3} {}",
                r.item + 1,
                item.id,
                item.timestamp,
                r.request.time_unit
            ));
            if button(ui, "▶", out, hv) {
                step = Some((r.item + 1) % n);
            }
            if button(ui, "Show", out, hv) {
                step = Some(r.item);
            }
            if button(ui, "Close", out, hv) {
                close = true;
            }
        });
        // Q-233, Q-234: what the open request changes in the view.
        let mut cams = world.query::<&ftgsxr_render::SplatCamera>();
        let passthrough = cams
            .iter(world)
            .any(|c| c.output == ftgsxr_render::OutputSpace::LinearOver { background: None });
        let governor = world
            .get_resource::<ftgsxr_render::SplatTierState>()
            .map(|t| t.sh_cap);
        let loaded = replay_loaded_degree(world, target);
        for line in replay_lines(&r.request, governor, loaded, passthrough) {
            ui.label(line);
        }
    }
    if close {
        replay = None;
        step = None;
    }
    if let (Some(r), Some(i)) = (&mut replay, step) {
        r.item = i;
        if let Some(instance) = target {
            let item = &r.request.items[i];
            out.push(Out::Jump(JumpToCamera {
                instance,
                camera: item.camera,
                timestamp: item.timestamp,
                time_unit: r.request.time_unit.clone(),
            }));
        }
    }
    let key = |r: &Option<cameras::Replay>| r.as_ref().map(|r| (r.path.clone(), r.item));
    if key(&view.replay) != key(&replay) {
        world.resource_mut::<TrainingView>().replay = replay;
    }
}

/// The instance a replay steps on: the selected one, else the first loaded (Q-219).
fn replay_target(world: &mut World) -> Option<Entity> {
    let mut sel =
        world.query_filtered::<Entity, (With<LoadedModel>, With<crate::manipulate::Selected>)>();
    let s = sel.iter(world).next();
    s.or_else(|| {
        let mut all = world.query_filtered::<Entity, With<LoadedModel>>();
        all.iter(world).next()
    })
}

/// The degree the replayed instance was last uploaded at (`SplatEvent::Uploaded`'s
/// `sh_degree`), `None` with no instance or before its first upload (Q-241).
// Q-241
fn replay_loaded_degree(world: &World, target: Option<Entity>) -> Option<u8> {
    let uploads = world.get_resource::<super::Uploads>()?;
    uploads.0.get(&target?).map(|u| u.sh_degree)
}

/// The replay row's lines on what the open request changes in the view (Q-233, Q-234):
/// its background, whether a passthrough camera leaves it out, and, when the request
/// names a degree above the governor's cap, the degree the replayed instance is drawn
/// at, min(its loaded degree, the cap, the request's degree). `loaded` is the degree
/// that instance was uploaded at (Q-241), `None` with no instance uploaded.
// Q-233, Q-234, Q-241
fn replay_lines(
    request: &ftgsxr_core::RenderRequest,
    governor_cap: Option<u8>,
    loaded: Option<u8>,
    passthrough: bool,
) -> Vec<String> {
    let [red, green, blue] = request.background;
    let mut lines = vec![format!(
        "background: request ({red:.3}, {green:.3}, {blue:.3})"
    )];
    if passthrough {
        lines.push("background not applied in passthrough".into());
    }
    if let (Some(d), Some(g), Some(l)) = (request.sh_degree, governor_cap, loaded)
        && g < d
    {
        let drawn = l.min(g).min(d);
        lines.push(format!(
            "SH degree {drawn} drawn (request {d}, governor cap {g})"
        ));
    }
    lines
}

/// Playback: §1.6's controls.
fn playback_tab(
    world: &mut World,
    ui: &mut egui::Ui,
    out: &mut Vec<Out>,
    hv: &mut Option<egui::Id>,
) {
    let Some(p) = world.get_resource::<Playback>() else {
        return;
    };
    let (mut t, len, speed) = (p.clock.time(), p.clock.length(), p.clock.speed());
    let bar = ui.add(egui::Slider::new(&mut t, 0.0..=len.max(0.0)).suffix(" s"));
    if bar.changed() {
        out.push(Out::Intent(Intent::Playback(PlaybackCommand::Seek(t))));
    }
    // Q-246: a scrub lasts while the bar is dragged and egui has decided the press is a
    // drag, not the start of a click (moved past the click distance or held past the
    // click duration); the slider senses drags only, so `dragged` alone holds from the
    // press.
    let decided = ui.input(|i| i.pointer.is_decidedly_dragging());
    world.resource_mut::<Scrubbing>().0 = bar.dragged() && decided;
    ui.horizontal(|ui| {
        for s in SPEEDS.iter().rev().map(|s| -s).chain(SPEEDS) {
            if ui.selectable_label(speed == s, format!("{s}")).clicked() {
                out.push(Out::Intent(Intent::Playback(PlaybackCommand::Speed(s))));
            }
        }
    });
    ui.horizontal(|ui| {
        for (label, m) in [
            ("loop", LoopMode::Loop),
            ("ping-pong", LoopMode::PingPong),
            ("once", LoopMode::Once),
        ] {
            if button(ui, label, out, hv) {
                out.push(Out::Intent(Intent::Playback(PlaybackCommand::Loop(m))));
            }
        }
    });
    // §1.6: [T_a, T_b] settable at both ends from the scrub position.
    let (a, b) = world
        .get_resource::<RangeSetting>()
        .and_then(|r| r.0)
        .unwrap_or((0.0, len));
    ui.horizontal(|ui| {
        ui.label(format!("range [{a:.3}, {b:.3}] s"));
        if button(ui, "T_a here", out, hv) {
            out.push(Out::Intent(Intent::Playback(PlaybackCommand::Range(Some(
                (t, b.max(t)),
            )))));
        }
        if button(ui, "T_b here", out, hv) {
            out.push(Out::Intent(Intent::Playback(PlaybackCommand::Range(Some(
                (a.min(t), t),
            )))));
        }
        if button(ui, "Clear range", out, hv) {
            out.push(Out::Intent(Intent::Playback(PlaybackCommand::Range(None))));
        }
    });
}

/// Display: the selected instances' render mode and the overlays; `persist` adds the
/// centers' radius and colormap and the background, `governor` the tier's knobs.
fn display_tab(
    world: &mut World,
    ui: &mut egui::Ui,
    out: &mut Vec<Out>,
    hv: &mut Option<egui::Id>,
) {
    let selected: Vec<Entity> = world
        .query_filtered::<Entity, (With<LoadedModel>, With<crate::manipulate::Selected>)>()
        .iter(world)
        .collect();
    if !selected.is_empty() {
        ui.horizontal(|ui| {
            ui.label("selected:");
            for (label, m) in MODES {
                if button(ui, label, out, hv) {
                    for &e in &selected {
                        out.push(Out::Intent(Intent::Instance {
                            instance: e,
                            action: InstanceAction::SetMode(m),
                        }));
                    }
                }
            }
        });
    }
    ui.horizontal(|ui| {
        for (label, o) in [
            ("Boxes", OverlayToggle::Boxes),
            ("Velocities", OverlayToggle::Velocities),
            ("Time tint", OverlayToggle::TimeTint),
        ] {
            if button(ui, label, out, hv) {
                out.push(Out::Intent(Intent::Overlay(o)));
            }
        }
    });
}

/// The open question as a dialog.
fn dialog(world: &mut World, ctx: &egui::Context, out: &mut Vec<Out>) {
    let Some(q) = world.resource::<UiState>().confirm.clone() else {
        return;
    };
    let scenes = match &q {
        Confirm::LoadScene(_) => load_choices(world),
        Confirm::Unload(_) => Vec::new(),
    };
    egui::Window::new("Confirm")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            match &q {
                Confirm::Unload(e) => {
                    let name = world
                        .get::<LoadedModel>(*e)
                        .map_or(String::new(), |m| m.name.clone());
                    ui.label(format!("Unload {name}?"));
                }
                Confirm::LoadScene(chosen) => {
                    ui.label("Replace the current scene by:");
                    for p in &scenes {
                        let label = p
                            .file_name()
                            .map_or(String::new(), |n| n.to_string_lossy().into_owned());
                        if ui
                            .selectable_label(chosen.as_ref() == Some(p), label)
                            .clicked()
                        {
                            world.resource_mut::<UiState>().confirm =
                                Some(Confirm::LoadScene(Some(p.clone())));
                        }
                    }
                    if scenes.is_empty() {
                        ui.label("No saved scene");
                    }
                }
            }
            ui.horizontal(|ui| {
                if ui.button("Yes").clicked() {
                    out.push(Out::Answer(true));
                }
                if ui.button("No").clicked() {
                    out.push(Out::Answer(false));
                }
            });
        });
}

/// The scenes the load dialog offers (§1.5 Scene files, Q-200, #417): the eight most
/// recent, as `persist` keeps them in `RecentScenes`; none without it.
// Q-200
fn load_choices(world: &World) -> Vec<std::path::PathBuf> {
    world
        .get_resource::<RecentScenes>()
        .map(|r| r.0.clone())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{
        Colormap, Entity, InstanceAction, Intent, LoadedModel, Out, RenderMode, SplatMode,
        Transform, Visibility, World, display_tab, dock_panel, egui, load_choices, playback_tab,
        replay_lines, replay_loaded_degree, scene_tab, source_from_input,
    };
    use crate::SplatEvent;
    use crate::manipulate::Selected;
    use crate::ui::{ToastRequest, Uploads, splat_toasts};
    use bevy::app::App;
    use ftgsxr_core::RenderRequest;
    use ftgsxr_core::{ModelTime, read_info};
    use ftgsxr_io::ModelRef;
    use ftgsxr_testkit::fixtures_dir;

    fn request(sh_degree: Option<u8>) -> RenderRequest {
        RenderRequest {
            time_unit: "seconds".into(),
            background: [0.25, 0.5, 1.0],
            sh_degree,
            items: Vec::new(),
            sha256: [0; 32],
        }
    }

    /// The width a dock saved at `width` is drawn at in a window `window` points wide.
    fn drawn_width(width: f32, window: f32) -> f32 {
        let ctx = egui::Context::default();
        let input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(window, 600.0),
            )),
            ..Default::default()
        };
        let mut drawn = 0.0;
        // Two frames: the second draws from the panel's remembered state.
        for _ in 0..2 {
            let mut out = ctx.run_ui(input(), |ui| {
                let w = ui.ctx().viewport_rect().width();
                drawn = dock_panel(width, w)
                    // Content that fills the panel's width, as the dock's tabs do.
                    .show(ui, |ui| {
                        ui.allocate_space(egui::vec2(ui.available_width(), 1.0));
                    })
                    .response
                    .rect
                    .width();
            });
            out.textures_delta.clear();
        }
        drawn
    }

    // §1.7 Panels, Q-251 (#428): a saved width is drawn within [160, 60 % of the
    // window] points, and the 160-point floor wins in a window under 267 points.
    #[test]
    fn dock_panel_draws_a_saved_width_outside_the_range_clamped() {
        assert_eq!(drawn_width(50.0, 1000.0), 160.0);
        assert_eq!(drawn_width(900.0, 1000.0), 600.0);
        assert_eq!(drawn_width(320.0, 1000.0), 320.0);
        assert_eq!(drawn_width(320.0, 200.0), 160.0);
    }

    fn sh_line(lines: &[String]) -> Option<&String> {
        lines.iter().find(|l| l.starts_with("SH degree"))
    }

    // Q-233 (#388): the SH line names the degree the replayed instance is drawn at,
    // min(its loaded degree, the governor's cap, the request's degree), and shows only
    // when the request names a degree above the governor's cap.
    #[test]
    fn the_sh_line_names_the_degree_drawn_when_the_governor_caps_the_request() {
        let lines = replay_lines(&request(Some(3)), Some(2), Some(1), false);
        assert_eq!(
            sh_line(&lines).map(String::as_str),
            Some("SH degree 1 drawn (request 3, governor cap 2)")
        );
        let lines = replay_lines(&request(Some(3)), Some(2), Some(3), false);
        assert_eq!(
            sh_line(&lines).map(String::as_str),
            Some("SH degree 2 drawn (request 3, governor cap 2)")
        );
        assert_eq!(
            sh_line(&replay_lines(&request(Some(2)), Some(3), Some(3), false)),
            None
        );
        assert_eq!(
            sh_line(&replay_lines(&request(Some(2)), Some(2), Some(3), false)),
            None
        );
        assert_eq!(
            sh_line(&replay_lines(&request(None), Some(1), Some(3), false)),
            None
        );
        // No instance loaded: no SH line.
        assert_eq!(
            sh_line(&replay_lines(&request(Some(3)), Some(2), None, false)),
            None
        );
    }

    // Q-241 (#403): the row's loaded degree is the one the instance was uploaded at,
    // `SplatEvent::Uploaded`'s `sh_degree` (§2, revision 17), not its file's: a degree-3
    // model uploaded under a governor's cap of 1, with the cap now 2 and a request of
    // degree 3, is drawn at 1 (Q-241's detecting test); a later upload (a reload, Q-238)
    // replaces it, and before any upload the row has no degree.
    #[test]
    fn replay_loaded_degree_is_the_latest_uploaded_degree() {
        let mut app = App::new();
        app.add_message::<ToastRequest>()
            .init_resource::<Uploads>()
            .add_observer(splat_toasts);
        let e = app.world_mut().spawn_empty().id();
        assert_eq!(
            replay_loaded_degree(app.world(), Some(e)),
            None,
            "no upload yet"
        );
        app.world_mut().trigger(SplatEvent::Uploaded {
            entity: e,
            bytes: 0,
            sh_degree: 1,
        });
        let loaded = replay_loaded_degree(app.world(), Some(e));
        assert_eq!(loaded, Some(1));
        assert_eq!(
            sh_line(&replay_lines(&request(Some(3)), Some(2), loaded, false)).map(String::as_str),
            Some("SH degree 1 drawn (request 3, governor cap 2)")
        );
        app.world_mut().trigger(SplatEvent::Uploaded {
            entity: e,
            bytes: 0,
            sh_degree: 2,
        });
        assert_eq!(
            replay_loaded_degree(app.world(), Some(e)),
            Some(2),
            "a reload's upload"
        );
        assert_eq!(replay_loaded_degree(app.world(), None), None, "no instance");
    }

    // Q-234 (#388): the row names the request's background, and says it is not applied
    // only while a passthrough camera exists.
    #[test]
    fn the_background_lines_name_the_request_and_passthrough() {
        let lines = replay_lines(&request(None), Some(3), Some(3), false);
        assert_eq!(lines, ["background: request (0.250, 0.500, 1.000)"]);
        let lines = replay_lines(&request(None), Some(3), Some(3), true);
        assert_eq!(
            lines,
            [
                "background: request (0.250, 0.500, 1.000)",
                "background not applied in passthrough"
            ]
        );
    }

    /// The instance actions `out` holds, by instance.
    fn actions(out: &[Out]) -> Vec<(Entity, InstanceAction)> {
        out.iter()
            .filter_map(|o| match o {
                Out::Intent(Intent::Instance { instance, action }) => Some((*instance, *action)),
                _ => None,
            })
            .collect()
    }

    /// An instance of the P1 fixture in `mode`, visible, optionally selected.
    fn instance(world: &mut World, mode: RenderMode, selected: bool) -> Entity {
        let path = fixtures_dir().join("probes/P1.safetensors");
        let bytes = std::fs::read(&path).unwrap();
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let info = read_info(&bytes[8..8 + n], bytes.len() as u64).unwrap();
        let m = LoadedModel {
            source: ModelRef::File(path),
            identity: "0".repeat(64),
            time: ModelTime::from_span(&info.time),
            info,
            name: "P1".into(),
            reduced: false,
            audio: None,
            frame_step_s: 1.0 / 30.0,
        };
        let mut ec = world.spawn((m, Transform::IDENTITY, Visibility::Visible, SplatMode(mode)));
        if selected {
            ec.insert(Selected);
        }
        ec.id()
    }

    type Tab = fn(&mut World, &mut egui::Ui, &mut Vec<Out>, &mut Option<egui::Id>);

    /// One frame of `tab` with `events`: the texts drawn and what the user did.
    fn frame(
        ctx: &egui::Context,
        world: &mut World,
        events: Vec<egui::Event>,
        tab: Tab,
    ) -> (Vec<(String, egui::Pos2)>, Vec<Out>) {
        let input = egui::RawInput {
            events,
            ..egui::RawInput::default()
        };
        let mut out = Vec::new();
        let mut hv = None;
        let mut full = ctx.run_ui(input, |ui| tab(world, ui, &mut out, &mut hv));
        full.textures_delta.clear();
        let mut texts = Vec::new();
        fn walk(s: &egui::Shape, texts: &mut Vec<(String, egui::Pos2)>) {
            match s {
                egui::Shape::Text(t) => texts.push((t.galley.text().to_string(), t.pos)),
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, texts)),
                _ => {}
            }
        }
        full.shapes.iter().for_each(|c| walk(&c.shape, &mut texts));
        (texts, out)
    }

    /// Draws `tab`, clicks the text `label`: what the user did over the click's frames.
    fn click(world: &mut World, tab: Tab, label: &str) -> Vec<Out> {
        let ctx = egui::Context::default();
        let (texts, _) = frame(&ctx, world, Vec::new(), tab);
        let at = texts
            .iter()
            .find(|(t, _)| t == label)
            .map(|(_, p)| *p + egui::vec2(4.0, 4.0))
            .unwrap_or_else(|| panic!("{label} is drawn: {texts:?}"));
        let press = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut out = Vec::new();
        for events in [
            vec![egui::Event::PointerMoved(at)],
            vec![press(true)],
            vec![press(false)],
        ] {
            out.extend(frame(&ctx, world, events, tab).1);
        }
        out
    }

    /// Whether the text `label` sits on a selected selectable label: a rectangle filled
    /// with the style's selection color under it.
    fn shown_selected(world: &mut World, tab: Tab, label: &str) -> bool {
        let ctx = egui::Context::default();
        let input = egui::RawInput::default();
        let (mut out, mut hv) = (Vec::new(), None);
        let mut full = ctx.run_ui(input, |ui| tab(world, ui, &mut out, &mut hv));
        full.textures_delta.clear();
        let fill = ctx.global_style().visuals.selection.bg_fill;
        let (mut at, mut rects) = (None, Vec::new());
        fn walk(
            s: &egui::Shape,
            label: &str,
            at: &mut Option<egui::Pos2>,
            rects: &mut Vec<(egui::Rect, egui::Color32)>,
        ) {
            match s {
                egui::Shape::Text(t) if t.galley.text() == label => *at = Some(t.pos),
                egui::Shape::Rect(r) => rects.push((r.rect, r.fill)),
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, label, at, rects)),
                _ => {}
            }
        }
        for c in &full.shapes {
            walk(&c.shape, label, &mut at, &mut rects);
        }
        let at = at.unwrap_or_else(|| panic!("{label} is drawn")) + egui::vec2(4.0, 4.0);
        rects.iter().any(|(r, f)| *f == fill && r.contains(at))
    }

    // #426: §1.5's render modes include hidden, and §1.7 Panels puts the mode in the
    // Scene tab beside visibility: the tab offers Hidden, which sets the mode alone.
    #[test]
    fn scene_tab_offers_hidden_and_it_leaves_visibility_alone() {
        let mut world = World::new();
        let e = instance(&mut world, RenderMode::Splats, false);
        assert!(shown_selected(&mut world, scene_tab, "Splats"));
        assert!(!shown_selected(&mut world, scene_tab, "Hidden"));
        let out = click(&mut world, scene_tab, "Hidden");
        assert_eq!(
            actions(&out),
            vec![(e, InstanceAction::SetMode(RenderMode::Hidden))]
        );
    }

    // #426: an instance in Hidden mode shows Hidden as its current mode, and choosing it
    // again sends nothing; another mode takes it out of Hidden.
    #[test]
    fn scene_tab_shows_hidden_as_the_current_mode() {
        let mut world = World::new();
        let e = instance(&mut world, RenderMode::Hidden, false);
        assert!(shown_selected(&mut world, scene_tab, "Hidden"));
        assert!(!shown_selected(&mut world, scene_tab, "Splats"));
        assert_eq!(actions(&click(&mut world, scene_tab, "Hidden")), vec![]);
        assert_eq!(
            actions(&click(&mut world, scene_tab, "Splats")),
            vec![(e, InstanceAction::SetMode(RenderMode::Splats))]
        );
    }

    // #426: §1.5 lists visibility apart from the render mode: the Visible checkbox of an
    // instance in Hidden mode sets its visibility alone.
    #[test]
    fn scene_tab_visible_leaves_the_hidden_mode_alone() {
        let mut world = World::new();
        let e = instance(&mut world, RenderMode::Hidden, false);
        assert_eq!(
            actions(&click(&mut world, scene_tab, "Visible")),
            vec![(e, InstanceAction::SetVisible(false))]
        );
    }

    // #426, Q-218: the Display tab sets the selected instances' render mode, Hidden
    // included, and leaves the others alone.
    #[test]
    fn display_tab_offers_hidden_for_the_selected_instances() {
        let mut world = World::new();
        let a = instance(&mut world, RenderMode::Splats, true);
        let b = instance(
            &mut world,
            RenderMode::Both {
                pixel_radius: 3.0,
                colormap: Colormap::Dc,
            },
            true,
        );
        instance(&mut world, RenderMode::Splats, false);
        let mut got = actions(&click(&mut world, display_tab, "Hidden"));
        got.sort_by_key(|(e, _)| *e);
        let mut want = vec![
            (a, InstanceAction::SetMode(RenderMode::Hidden)),
            (b, InstanceAction::SetMode(RenderMode::Hidden)),
        ];
        want.sort_by_key(|(e, _)| *e);
        assert_eq!(got, want);
    }

    #[test]
    fn a_url_is_an_http_source_and_an_absolute_path_a_folder() {
        let s = source_from_input(" https://example.org/models/ ").unwrap();
        assert_eq!(
            (s.kind.as_str(), s.name.as_str(), s.location.as_str()),
            ("http", "example.org", "https://example.org/models/")
        );
        let s = source_from_input("/data/scans").unwrap();
        assert_eq!((s.kind.as_str(), s.name.as_str()), ("dir", "scans"));
        assert!(source_from_input("relative/dir").is_none());
        assert!(source_from_input("ftp://x/").is_none());
        assert!(source_from_input("").is_none());
    }

    /// The playback tab drawn on `ctx` for one frame of `events` at `time_s`; the
    /// intents it wrote and whether `Scrubbing` held after it.
    fn playback_frame(
        world: &mut bevy::ecs::world::World,
        ctx: &bevy_egui::egui::Context,
        time_s: f64,
        events: Vec<bevy_egui::egui::Event>,
    ) -> (Vec<crate::intent::Intent>, bool) {
        use bevy_egui::egui;
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            time: Some(time_s),
            events,
            ..Default::default()
        };
        let mut out = Vec::new();
        let mut hv = None;
        let mut full = ctx.run_ui(input, |ui| playback_tab(world, ui, &mut out, &mut hv));
        full.textures_delta.clear();
        let intents = out
            .into_iter()
            .filter_map(|o| match o {
                super::Out::Intent(i) => Some(i),
                _ => None,
            })
            .collect();
        (intents, world.resource::<crate::playback::Scrubbing>().0)
    }

    // Q-246 (#416): a drag of the scrub bar is a scrub, from egui's drag decision (moved
    // past the click distance, or held past the click duration) to the release; a press
    // released without a drag is a one-off seek.
    #[test]
    fn playback_tab_a_drag_of_the_scrub_bar_is_a_scrub_and_a_click_is_not() {
        use crate::intent::{Intent, PlaybackCommand};
        use bevy_egui::egui::{Event, Modifiers, PointerButton, pos2};
        let press = |x: f32, pressed: bool| Event::PointerButton {
            pos: pos2(x, 15.0),
            button: PointerButton::Primary,
            pressed,
            modifiers: Modifiers::NONE,
        };
        let moved = |x: f32| Event::PointerMoved(pos2(x, 15.0));
        let seek = |i: &[Intent]| {
            i.iter()
                .any(|i| matches!(i, Intent::Playback(PlaybackCommand::Seek(_))))
        };
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(crate::playback::Playback {
            clock: ftgsxr_core::Clock::new(2.0),
        });
        world.init_resource::<crate::playback::Scrubbing>();
        world.init_resource::<crate::playback::RangeSetting>();
        let ctx = bevy_egui::egui::Context::default();
        assert!(!playback_frame(&mut world, &ctx, 0.0, vec![moved(30.0)]).1);
        // A click: pressed and released in place.
        let (_, held) = playback_frame(&mut world, &ctx, 0.02, vec![press(30.0, true)]);
        assert!(!held, "a press is not yet a scrub");
        let (_, held) = playback_frame(&mut world, &ctx, 0.04, vec![press(30.0, false)]);
        assert!(!held, "a click is a one-off seek");
        // A drag: pressed, moved along the bar, released.
        playback_frame(&mut world, &ctx, 0.06, vec![press(30.0, true)]);
        let (i, held) = playback_frame(&mut world, &ctx, 0.08, vec![moved(80.0)]);
        assert!(seek(&i), "the drag moves the scrub bar");
        assert!(held, "a drag is a scrub");
        let (_, held) = playback_frame(&mut world, &ctx, 0.10, vec![moved(90.0)]);
        assert!(held, "the scrub lasts while the drag does");
        let (_, held) = playback_frame(&mut world, &ctx, 0.12, vec![press(90.0, false)]);
        assert!(!held, "the release ends the scrub");
        // A press held in place past egui's click duration is a scrub too.
        playback_frame(&mut world, &ctx, 0.20, vec![press(30.0, true)]);
        let (_, held) = playback_frame(&mut world, &ctx, 1.20, vec![]);
        assert!(held, "a long press is a scrub");
        let (_, held) = playback_frame(&mut world, &ctx, 1.22, vec![press(30.0, false)]);
        assert!(!held);
    }

    // §1.5 Scene files, Q-200 (#417): the dialog offers exactly `RecentScenes`, in its
    // order, and nothing without it.
    #[test]
    fn load_choices_are_the_recent_scenes() {
        let mut world = bevy::ecs::world::World::new();
        assert!(load_choices(&world).is_empty());
        let recent = vec![
            std::path::PathBuf::from("/c/scenes/b.ftgsxr-scene.json"),
            std::path::PathBuf::from("/c/scenes/a.ftgsxr-scene.json"),
        ];
        world.insert_resource(crate::ui::RecentScenes(recent.clone()));
        assert_eq!(load_choices(&world), recent);
    }
}
