// SPDX-License-Identifier: Apache-2.0

//! The screen's UI (§1.7 Panels): the bar above the viewport, the dock on the left with
//! its seven tabs, the toast line and the confirmation dialogs. Drawn by one exclusive
//! system, which collects what the user did and sends it after the frame.

use std::path::PathBuf;

use bevy::camera::visibility::Visibility;
use bevy::ecs::prelude::*;
use bevy::math::{EulerRot, Quat};
use bevy::transform::components::Transform;
use bevy_egui::{EguiContext, PrimaryEguiContext, egui};
use ftgsxr_core::{EndRule, LoopMode, PlayState};
use ftgsxr_io::EntryKind;
use ftgsxr_render::{ColorTransform, Colormap, RenderMode, SplatMode, SplatTransform};

use super::cameras::{self, JumpToCamera, TrainingCameras, TrainingView};
use super::{Confirm, DockHooks, SceneLoadConfirmed, Tab, ToastLevel, UiCue, UiState};
use crate::intent::{InstanceAction, Intent, OverlayToggle, PlaybackCommand, UiCommand};
use crate::library::{EntryState, Library, LibraryCommand, LoadedModel, LocateEntry};
use crate::platform::{PickKind, PlatformRes};
use crate::playback::Playback;

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

/// The dock and its tabs; its width is kept in `UiState`.
fn dock(world: &mut World, ctx: &mut egui::Ui, out: &mut Vec<Out>, hv: &mut Option<egui::Id>) {
    let (open, width, tab) = {
        let s = world.resource::<UiState>();
        (s.dock_open, s.dock_width, s.tab)
    };
    if !open {
        return;
    }
    let r = egui::Panel::left("ftgsxr-dock")
        .resizable(true)
        .default_size(width)
        .show(ctx, |ui| {
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
                    Tab::Environment => environment_tab(ui),
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
        });
    }
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
                    // Q-T275.3: only a file of the entry's identity is accepted.
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
        &LoadedModel,
        &mut Transform,
        Option<&Visibility>,
        Option<&SplatMode>,
        Option<&SplatTransform>,
    )>();
    let mut ask = None;
    let mut gains = Vec::new();
    for (e, m, mut tf, vis, mode, color) in q.iter_mut(world) {
        ui.separator();
        ui.strong(if m.reduced {
            format!("{} (reduced)", m.name)
        } else {
            m.name.clone()
        });
        let mut visible = vis != Some(&Visibility::Hidden);
        if ui.checkbox(&mut visible, "Visible").changed() {
            out.push(Out::Intent(Intent::Instance {
                instance: e,
                action: InstanceAction::SetVisible(visible),
            }));
        }
        let current = mode.map_or(RenderMode::Splats, |m| m.0);
        ui.horizontal(|ui| {
            for (label, m) in [
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
            ] {
                let on = std::mem::discriminant(&current) == std::mem::discriminant(&m);
                if ui.selectable_label(on, label).clicked() && !on {
                    out.push(Out::Intent(Intent::Instance {
                        instance: e,
                        action: InstanceAction::SetMode(m),
                    }));
                }
            }
        });
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
        // E14's color transform, per channel (Q-218).
        let gain = color.map_or(ColorTransform::IDENTITY.gain, |c| c.0.gain);
        let mut g = gain.to_array();
        ui.horizontal(|ui| {
            ui.label("color gain");
            for v in &mut g {
                ui.add(egui::DragValue::new(v).range(0.0..=4.0).speed(0.01));
            }
            if ui.small_button("reset").clicked() {
                g = ColorTransform::IDENTITY.gain.to_array();
            }
        });
        let g = bevy::math::Vec3::from_array(g);
        if g != gain {
            gains.push((e, g));
        }
        ui.horizontal(|ui| {
            for (label, action) in [
                ("Select", InstanceAction::Select),
                ("Fit", InstanceAction::Fit),
                ("Save pose", InstanceAction::SavePose),
                ("Reset pose", InstanceAction::ResetPose),
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
    training_cameras(world, ui, out, hv);
    for (e, gain) in gains {
        if let Ok(mut ec) = world.get_entity_mut(e) {
            ec.insert(SplatTransform(ColorTransform { gain }));
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
    }
    if close {
        replay = None;
        step = None;
    }
    if let (Some(r), Some(i)) = (&mut replay, step) {
        r.item = i;
        let target = {
            let mut sel = world
                .query_filtered::<Entity, (With<LoadedModel>, With<crate::manipulate::Selected>)>();
            let s = sel.iter(world).next();
            s.or_else(|| {
                let mut all = world.query_filtered::<Entity, With<LoadedModel>>();
                all.iter(world).next()
            })
        };
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
    if ui
        .add(egui::Slider::new(&mut t, 0.0..=len.max(0.0)).suffix(" s"))
        .changed()
    {
        out.push(Out::Intent(Intent::Playback(PlaybackCommand::Seek(t))));
    }
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
    ui.horizontal(|ui| {
        if button(ui, "Range from here", out, hv) {
            out.push(Out::Intent(Intent::Playback(PlaybackCommand::Range(Some(
                (t, len),
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
            for (label, m) in [
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
            ] {
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

/// Environment on the desktop: passthrough is unavailable; `persist` adds the
/// background and the grid (they are settings).
fn environment_tab(ui: &mut egui::Ui) {
    // Q-161: no passthrough on the desktop; the control is not offered.
    ui.label("Passthrough: not available on the desktop");
}

/// The open question as a dialog.
fn dialog(world: &mut World, ctx: &egui::Context, out: &mut Vec<Out>) {
    let Some(q) = world.resource::<UiState>().confirm.clone() else {
        return;
    };
    let scenes = match &q {
        Confirm::LoadScene(_) => scene_files(world),
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

/// The scene files under the platform's scenes directory, newest first.
fn scene_files(world: &World) -> Vec<PathBuf> {
    let Some(p) = world.get_resource::<PlatformRes>() else {
        return Vec::new();
    };
    let Ok(dir) = std::fs::read_dir(&p.0.paths().scenes) else {
        return Vec::new();
    };
    let mut files: Vec<_> = dir
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .ends_with(".ftgsxr-scene.json")
        })
        .map(|e| (e.metadata().and_then(|m| m.modified()).ok(), e.path()))
        .collect();
    files.sort_by_key(|f| std::cmp::Reverse(f.0));
    files.into_iter().map(|(_, p)| p).collect()
}
