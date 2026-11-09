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
use ftgsxr_render::{
    Colormap, RenderMode, SplatAsset, SplatBounds, SplatMode, SplatModel, SplatTime, SplatTrim,
};

use super::cameras::{self, JumpToCamera, TrainingCameras, TrainingView};
use super::trim_row::{TrimInfo, trim_row};
use super::{
    Confirm, DockHooks, InstanceLighting, RecentScenes, SceneLoadConfirmed, Tab, ToastLevel, UiCue,
    UiState,
};
use crate::intent::{InstanceAction, Intent, OverlayToggle, PlaybackCommand, UiCommand};
use crate::library::{EntryState, Library, LibraryCommand, LoadedModel, LocateEntry};
use crate::manipulate::{
    FIT_DEFAULT_M, FIT_MAX_M, FIT_MIN_M, ManipulateSettings, PoseExtras, SetPoseExtras,
};
use crate::platform::PickKind;
use crate::playback::{Playback, RangeSetting, Scrubbing};

/// What the user did this frame.
pub(super) enum Out {
    Intent(Intent),
    Library(LibraryCommand),
    Locate(usize),
    // Q-256
    Replace(usize),
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
    send(world, out);
}

/// Sends what the user did, after the frame's drawing.
pub(super) fn send(world: &mut World, out: Vec<Out>) {
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
            Out::Replace(i) => {
                world.write_message(super::ReplaceRequest(i));
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
                if let Some(r) = a.replace {
                    world.write_message(r);
                }
                world.write_message(a.cue);
            }
        }
    }
}

/// A button that reports its hover and its click's sound.
pub(super) fn button(
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
    let (t, len, playing) = clock_state(world);
    let readout = selected_readout(world, t);
    egui::Panel::top("ftgsxr-bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let open = world.resource::<UiState>().dock_open;
            if button(ui, if open { "◀" } else { "▶" }, out, hv) {
                world.resource_mut::<UiState>().dock_open = !open;
            }
            play_items(ui, (t, len, playing), readout, out, hv);
        });
    });
}

/// The clock's playing state: (T, the length, playing).
fn clock_state(world: &World) -> (f64, f64, bool) {
    world
        .get_resource::<Playback>()
        .map_or((0.0, 0.0, false), |p| {
            (
                p.clock.time(),
                p.clock.length(),
                p.clock.state() == PlayState::Playing,
            )
        })
}

/// The play controls and the time readout: the screen's bar and the main panel's
/// Playback tab (§1.7 Panels, §1.8 User interface).
fn play_items(
    ui: &mut egui::Ui,
    (t, len, playing): (f64, f64, bool),
    readout: Option<(String, bool)>,
    out: &mut Vec<Out>,
    hv: &mut Option<egui::Id>,
) {
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

/// The toast line at the bottom of the window or the main panel.
pub(super) fn toast_line(world: &mut World, ctx: &mut egui::Ui) {
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
/// A saved width is at least 160 already (`load_settings`), so for it only the window's
/// bound applies; the floor bounds a drag.
// Q-251
fn dock_panel(width: f32, window: f32) -> egui::Panel {
    egui::Panel::left(DOCK_ID)
        .resizable(true)
        .default_size(width)
        .size_range(dock_range(window))
}

/// The dock's panel id.
const DOCK_ID: &str = "ftgsxr-dock";

/// §1.7 Panels, Q-251: the dock's widths in a window `window` points wide.
fn dock_range(window: f32) -> std::ops::RangeInclusive<f32> {
    DOCK_MIN_WIDTH..=(DOCK_MAX_SHARE * window).max(DOCK_MIN_WIDTH)
}

/// Draws the dock's panel at its saved `width` clamped to the window's range, and
/// returns the width the user dragged it to, if any (Q-252). egui's remembered size is
/// dropped first, so a window narrowed and widened again draws the saved width once
/// more; the drawn width differs from the clamped one (by half a point or more, beyond
/// egui's 1/32-point rounding) only while the handle is dragged.
// Q-252
fn show_dock_panel<R>(
    ui: &mut egui::Ui,
    width: f32,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> (f32, Option<f32>) {
    let window = ui.ctx().viewport_rect().width();
    ui.ctx()
        .data_mut(|d| d.remove::<egui::containers::panel::PanelState>(egui::Id::new(DOCK_ID)));
    let range = dock_range(window);
    let shown = width.clamp(*range.start(), *range.end());
    let drawn = dock_panel(width, window)
        .show(ui, add_contents)
        .response
        .rect
        .width();
    (drawn, ((drawn - shown).abs() >= 0.5).then_some(drawn))
}

/// The dock and its tabs; its width is kept in `UiState`, changed only by a drag of its
/// handle, and `persist` saves it with whether it is open (§1.7 Panels, Q-252).
fn dock(world: &mut World, ctx: &mut egui::Ui, out: &mut Vec<Out>, hv: &mut Option<egui::Id>) {
    let (open, width, tab) = {
        let s = world.resource::<UiState>();
        (s.dock_open, s.dock_width, s.tab)
    };
    if !open {
        return;
    }
    let (_, dragged) = show_dock_panel(ctx, width, |ui| {
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
        egui::ScrollArea::vertical().show(ui, |ui| tab_body(world, ui, tab, out, hv, false));
    });
    if let Some(w) = dragged {
        world.resource_mut::<UiState>().dock_width = w;
    }
}

/// A tab's content on the screen's dock or, with `panel`, on the main panel (§1.8 User
/// interface): there the play controls join the Playback tab, the training cameras and
/// the request replay are sub-pages opened by a button, and the mouse-only items (Help's
/// keyboard shortcuts) are left out.
// Q-294
pub(super) fn tab_body(
    world: &mut World,
    ui: &mut egui::Ui,
    tab: Tab,
    out: &mut Vec<Out>,
    hv: &mut Option<egui::Id>,
    panel: bool,
) {
    match tab {
        Tab::Library => {
            library_tab(world, ui, out, hv);
            if !panel {
                request_replay(world, ui, out, hv);
            } else if has_requests(world) && button(ui, "Render requests…", out, hv) {
                world.resource_mut::<UiState>().page = Some(super::Page::Replay);
            }
        }
        Tab::Scene => {
            scene_tab(world, ui, out, hv);
            if !panel {
                training_cameras(world, ui, out, hv);
            } else if has_training_cameras(world) && button(ui, "Training cameras…", out, hv) {
                world.resource_mut::<UiState>().page = Some(super::Page::Cameras);
            }
        }
        Tab::Playback => {
            if panel {
                let clock = clock_state(world);
                let readout = selected_readout(world, clock.0);
                ui.horizontal_wrapped(|ui| play_items(ui, clock, readout, out, hv));
            }
            playback_tab(world, ui, out, hv);
        }
        Tab::Display => display_tab(world, ui, out, hv),
        // §1.1, §1.7 Panels, §1.15 E-XR-UNSUPPORTED: no passthrough control on
        // the desktop (absent, not disabled); `persist` adds the background
        // and the grid. On the main panel, §1.8's recenter (D-17, D-24), which has
        // no desktop binding (Appendix D.1).
        // Q-311
        Tab::Environment => {
            if panel && button(ui, "Recenter", out, hv) {
                out.push(Out::Intent(Intent::Recenter));
            }
        }
        Tab::Settings => {
            ui.label("Settings (§1.11)");
        }
        Tab::Help => {
            ui.label(format!("openftgs-xr {}", env!("CARGO_PKG_VERSION")));
            if !panel {
                ui.label("F1 Help · F2 hide the UI · F3 HUD · F11 fullscreen");
            }
        }
        // §1.8 Comfort and accessibility: `persist`'s section.
        Tab::Comfort => {}
    }
    let hooks = world.resource::<DockHooks>().sections(tab, panel);
    for h in hooks {
        ui.separator();
        h(world, ui);
    }
}

/// Whether an instance offers training cameras (the Scene tab's sub-page).
fn has_training_cameras(world: &mut World) -> bool {
    let mut q = world.query::<(&LoadedModel, &TrainingCameras)>();
    q.iter(world).next().is_some()
}

/// Whether a render request can be opened or is open (the Library tab's sub-page).
fn has_requests(world: &World) -> bool {
    world
        .get_resource::<TrainingView>()
        .is_some_and(|v| !v.requests.is_empty() || v.replay.is_some())
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

/// The kind filters' labels, in `UiState.filter_kinds` order (`kind_index`).
const KINDS: [&str; 4] = ["ftgs-if", "checkpoint", "PLY", "folders"];

/// The width a button labelled `text` takes in a row of `ui`, with the spacing after it.
fn button_width(ui: &egui::Ui, text: &str) -> f32 {
    let galley = egui::WidgetText::from(text).into_galley(
        ui,
        Some(egui::TextWrapMode::Extend),
        f32::INFINITY,
        egui::TextStyle::Button,
    );
    galley.size().x + 2.0 * ui.spacing().button_padding.x + ui.spacing().item_spacing.x
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
    // The dock's or panel's width, before a row wider than it widens this `Ui` (#620).
    let width = ui.available_width();
    let sources: Vec<String> = library
        .sources
        .iter()
        .map(|(s, _)| format!("{s:?}"))
        .collect();
    let entries = library.entries.clone();
    // #627: every row wraps, or sizes its field, to the tab's width, down to §1.7's
    // narrowest dock: a control past the edge can't be used.
    for (i, s) in sources.iter().enumerate() {
        ui.horizontal_wrapped(|ui| {
            ui.add(egui::Label::new(s).wrap());
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
    ui.horizontal_wrapped(|ui| {
        let mut state = world.resource_mut::<UiState>();
        // The field fills what "Add source" leaves of the row, or the whole row with the
        // button on the next where that would leave the field too narrow.
        let add = "Add source";
        let add_width = button_width(ui, add);
        let row = ui.available_width();
        let field = if row - add_width >= row / 2.0 {
            row - add_width
        } else {
            row
        };
        ui.add(
            egui::TextEdit::singleline(&mut state.new_source)
                .hint_text("https://… index or /path/to/folder")
                .desired_width(field),
        );
        let spec = source_from_input(&state.new_source);
        if ui
            .add_enabled(spec.is_some(), egui::Button::new(add))
            .clicked()
            && let Some(spec) = spec
        {
            state.new_source.clear();
            out.push(Out::Cue(UiCue::Click));
            out.push(Out::Library(LibraryCommand::AddSource(spec)));
        }
    });
    ui.horizontal_wrapped(|ui| {
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
        let rest = ui.available_width();
        ui.add(egui::TextEdit::singleline(&mut state.filter_name).desired_width(rest));
    });
    ui.horizontal_wrapped(|ui| {
        for (i, k) in KINDS.iter().enumerate() {
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
        ui.horizontal_wrapped(|ui| {
            ui.add(egui::Label::new(&e.entry.name).wrap());
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
                EntryState::Failed(_) => {
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
                    // §1.5 Library (revision 26), Q-256: a file of any identity, once
                    // confirmed.
                    if button(ui, "Replace with…", out, hv) {
                        out.push(Out::Replace(i));
                    }
                }
            }
        });
        // #620: the message on its own line below, wrapped to the tab's width, which the
        // row above would run it past (§1.5 Library, §1.15).
        if let EntryState::Failed(text) = &e.state {
            let text = egui::RichText::new(text).color(ui.visuals().error_fg_color);
            ui.scope(|ui| {
                ui.set_max_width(width);
                ui.add(egui::Label::new(text).wrap());
            });
        }
    }
}

/// The "Fit to" toggle's label: the fit box it stands at (§1.5 Poses, Q-371).
fn fit_to_label(content: bool) -> &'static str {
    if content {
        "Fit to: content"
    } else {
        "Fit to: whole model"
    }
}

/// Scene: the instances, their visibility, mode, pose, time offset and end rule, color
/// transform and trim (§1.7 Trim, E17); save and load.
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
    // E17: each loaded instance's trim, boxes and untrimmed B_m, for its trim row.
    let mut trims = std::collections::HashMap::new();
    let mut tq = world.query::<(
        Entity,
        &SplatModel,
        Option<&SplatTrim>,
        Option<&SplatBounds>,
    )>();
    if let Some(assets) = world.get_resource::<bevy::asset::Assets<SplatAsset>>() {
        for (e, m, trim, bounds) in tq.iter(world) {
            if let Some(a) = assets.get(&m.0) {
                let info = TrimInfo {
                    trim: trim.map(|t| t.0),
                    bounds: bounds.copied(),
                    untrimmed: a.bounds,
                };
                trims.insert(e, info);
            }
        }
    }
    let mut ask = None;
    let mut lightings = Vec::new();
    let mut extras_out = Vec::new();
    // §1.5 Poses: "Fit to" is one global setting, shown and flipped beside every
    // instance's fit (Q-371).
    let fit_content = world
        .get_resource::<ManipulateSettings>()
        .is_none_or(|s| s.fit_to_content);
    let mut flip_fit_to = false;
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
            fit_size_m: FIT_DEFAULT_M,
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
            // From 0.25 m to 10⁸ m (§1.5), a typed "inf" taken as 10⁸ m; each pixel dragged
            // changes the size by 1 %, a logarithmic drag, fine near 0.25 m and still usable
            // at tens of metres (#621, #630).
            let speed = f64::from(next.fit_size_m) * 0.01;
            ui.add(
                egui::DragValue::new(&mut next.fit_size_m)
                    .range(FIT_MIN_M..=FIT_MAX_M)
                    .speed(speed)
                    .suffix(" m"),
            );
        });
        // §1.7 Trim: the trim row, labelled through the preset (E17).
        if let Some(info) = trims.get(&e) {
            trim_row(ui, e, x.preset, *info, out, hv);
        }
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
                // Q-371
                if action == InstanceAction::Fit && button(ui, fit_to_label(fit_content), out, hv) {
                    flip_fit_to = true;
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
    if flip_fit_to && let Some(mut s) = world.get_resource_mut::<ManipulateSettings>() {
        s.fit_to_content = !fit_content;
    }
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
pub(super) fn training_cameras(
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
pub(super) fn request_replay(
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
    // Q-246: a scrub lasts from the press on the bar to the release, whether the
    // pointer moves or not (a click included); the slider senses drags only, so
    // `dragged` holds from the press.
    world.resource_mut::<Scrubbing>().0 = bar.dragged();
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
    if world.resource::<UiState>().confirm.is_none() {
        return;
    }
    egui::Window::new("Confirm")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| confirm_body(world, ui, out));
}

/// The open question and its Yes and No: the screen's dialog, the main panel's
/// sub-page (§1.8 User interface).
pub(super) fn confirm_body(world: &mut World, ui: &mut egui::Ui, out: &mut Vec<Out>) {
    let Some(q) = world.resource::<UiState>().confirm.clone() else {
        return;
    };
    let scenes = match &q {
        Confirm::LoadScene(_) => load_choices(world),
        Confirm::Unload(_) | Confirm::Replace { .. } => Vec::new(),
    };
    {
        {
            match &q {
                Confirm::Unload(e) => {
                    let name = world
                        .get::<LoadedModel>(*e)
                        .map_or(String::new(), |m| m.name.clone());
                    ui.label(format!("Unload {name}?"));
                }
                Confirm::Replace { model, file, .. } => {
                    ui.label(Confirm::replace_text(model, file));
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
        }
    }
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
        Transform, Visibility, World, display_tab, egui, library_tab, load_choices, playback_tab,
        replay_lines, replay_loaded_degree, scene_tab, show_dock_panel, source_from_input,
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

    /// A dock in a test context: its saved width (`UiState.dock_width`), changed as
    /// `dock` changes it, by the width a drag draws (Q-252).
    struct Dock {
        ctx: egui::Context,
        saved: f32,
    }

    impl Dock {
        fn new(saved: f32) -> Self {
            Dock {
                ctx: egui::Context::default(),
                saved,
            }
        }

        /// One frame in a window `window` points wide, with `events`; the drawn width.
        fn frame(&mut self, window: f32, events: Vec<egui::Event>) -> f32 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(window, 600.0),
                )),
                events,
                ..Default::default()
            };
            let (mut drawn, mut dragged) = (0.0, None);
            let mut out = self.ctx.run_ui(input, |ui| {
                (drawn, dragged) = show_dock_panel(ui, self.saved, |ui| {
                    // Content that fills the panel's width, as the dock's tabs do.
                    ui.allocate_space(egui::vec2(ui.available_width(), 1.0));
                });
            });
            out.textures_delta.clear();
            if let Some(w) = dragged {
                self.saved = w;
            }
            drawn
        }

        /// Two frames, the second drawn from the first's state; the drawn width.
        fn settle(&mut self, window: f32) -> f32 {
            self.frame(window, Vec::new());
            self.frame(window, Vec::new())
        }
    }

    /// The width a dock saved at `width` is drawn at in a window `window` points wide.
    fn drawn_width(width: f32, window: f32) -> f32 {
        Dock::new(width).settle(window)
    }

    // §1.7 Panels, Q-251 (#440): `load_settings` already takes a width below 160 as
    // the default, so drawing applies only the window's bound, 60 % of its width, and
    // the 160-point floor wins in a window under 267 points. The 50-point case is
    // `dock_range`'s floor, which still bounds every width drawn.
    #[test]
    fn dock_panel_draws_a_saved_width_outside_the_range_clamped() {
        assert_eq!(drawn_width(50.0, 1000.0), 160.0);
        assert_eq!(drawn_width(900.0, 1000.0), 600.0);
        assert_eq!(drawn_width(320.0, 1000.0), 320.0);
        assert_eq!(drawn_width(320.0, 200.0), 160.0);
    }

    // §1.7 Panels, Q-252 (owner, #441): a window narrowed below the saved width
    // clamps only the drawing; the saved width is kept, and drawn again once the
    // window is wide enough.
    #[test]
    fn a_narrowed_window_keeps_the_saved_dock_width() {
        let mut dock = Dock::new(600.0);
        assert_eq!(dock.settle(1200.0), 600.0);
        assert_eq!(dock.settle(800.0), 480.0);
        assert_eq!(dock.saved, 600.0);
        assert_eq!(dock.settle(1200.0), 600.0);
        assert_eq!(dock.saved, 600.0);
        // A window under 267 points, where the floor wins, keeps it too.
        assert_eq!(dock.settle(200.0), 160.0);
        assert_eq!(dock.settle(1200.0), 600.0);
        assert_eq!(dock.saved, 600.0);
    }

    // §1.7 Panels, Q-252 (owner, #441): a larger UI scale narrows the window in points
    // (`ui_scale_follows_settings` sets egui's zoom, and bevy_egui gives egui the
    // window's pixels divided by the zoom), which clamps only the drawing.
    #[test]
    fn a_larger_ui_scale_keeps_the_saved_dock_width() {
        let pixels = 1200.0;
        let mut dock = Dock::new(600.0);
        assert_eq!(dock.settle(pixels / 1.0), 600.0);
        assert_eq!(dock.settle(pixels / 2.0), 360.0);
        assert_eq!(dock.saved, 600.0);
        assert_eq!(dock.settle(pixels / 1.0), 600.0);
        assert_eq!(dock.saved, 600.0);
    }

    // §1.7 Panels, Q-252 (owner, #441): a drag of the dock's handle is the one change
    // of its saved width, kept within the window's range.
    #[test]
    fn a_drag_of_the_handle_sets_the_saved_dock_width() {
        let mut dock = Dock::new(320.0);
        assert_eq!(dock.settle(1000.0), 320.0);
        let at = |x: f32| egui::Pos2::new(x, 300.0);
        let button = |x: f32, pressed: bool| egui::Event::PointerButton {
            pos: at(x),
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        dock.frame(1000.0, vec![egui::Event::PointerMoved(at(320.0))]);
        dock.frame(1000.0, vec![button(320.0, true)]);
        for x in [340.0, 380.0, 400.0] {
            dock.frame(1000.0, vec![egui::Event::PointerMoved(at(x))]);
        }
        dock.frame(1000.0, vec![button(400.0, false)]);
        assert_eq!(dock.saved, 400.0);
        assert_eq!(dock.settle(1000.0), 400.0);
        assert_eq!(dock.saved, 400.0);
        // A drag past 60 % of the window saves the widest drawn.
        dock.frame(1000.0, vec![egui::Event::PointerMoved(at(400.0))]);
        dock.frame(1000.0, vec![button(400.0, true)]);
        for x in [450.0, 700.0, 900.0] {
            dock.frame(1000.0, vec![egui::Event::PointerMoved(at(x))]);
        }
        dock.frame(1000.0, vec![button(900.0, false)]);
        assert_eq!(dock.saved, 600.0);
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

    /// An instance of P1 with its asset loaded, at `preset`, with `trim` and `bounds`.
    fn trimmed_instance(
        world: &mut World,
        preset: ftgsxr_core::Preset,
        trim: Option<ftgsxr_core::Trim>,
        kept_none: bool,
    ) -> (Entity, ftgsxr_core::Aabb) {
        use bevy::asset::Assets;
        use ftgsxr_render::{SplatAsset, SplatBounds, SplatModel, SplatTrim, load_splat_asset};
        let e = instance(world, RenderMode::Splats, false);
        let bytes = std::fs::read(fixtures_dir().join("probes/P1.safetensors")).unwrap();
        let asset = load_splat_asset(
            &ftgsxr_io::ModelBytes::Owned(bytes),
            None,
            &ftgsxr_core::GpuLayoutOptions::default(),
            None,
        )
        .unwrap();
        let b = asset.bounds;
        if !world.contains_resource::<Assets<SplatAsset>>() {
            world.insert_resource(Assets::<SplatAsset>::default());
        }
        let h = world.resource_mut::<Assets<SplatAsset>>().add(asset);
        let mut ec = world.entity_mut(e);
        ec.insert((
            SplatModel(h),
            crate::manipulate::PoseExtras {
                preset,
                fit_size_m: 1.0,
            },
            SplatBounds {
                aabb: b,
                content: b,
                kept_none,
            },
        ));
        if let Some(t) = trim {
            ec.insert(SplatTrim(t));
        }
        (e, b)
    }

    fn texts(world: &mut World) -> Vec<String> {
        let ctx = egui::Context::default();
        frame(&ctx, world, Vec::new(), scene_tab)
            .0
            .into_iter()
            .map(|(t, _)| t)
            .collect()
    }

    // Private (I-23's panel cases, which have no public seam, Q-375; §1.7 Trim, E17's
    // table): the box limits are named through the orientation preset, and as x, y and z
    // min and max under a custom one.
    #[test]
    fn scene_tab_trim_labels_follow_the_preset() {
        use ftgsxr_core::{Preset, Quat, Trim, TrimMode};
        let cases = [
            (
                Preset::YUp,
                [
                    "left (x min)",
                    "right (x max)",
                    "bottom (y min)",
                    "top (y max)",
                    "back (z min)",
                    "front (z max)",
                ],
            ),
            (
                Preset::ZUp,
                [
                    "left (x min)",
                    "right (x max)",
                    "bottom (z min)",
                    "top (z max)",
                    "back (y max)",
                    "front (y min)",
                ],
            ),
            (
                Preset::NegYUp,
                [
                    "left (x min)",
                    "right (x max)",
                    "bottom (y max)",
                    "top (y min)",
                    "back (z max)",
                    "front (z min)",
                ],
            ),
            (
                Preset::Custom(Quat::from_rotation_x(0.3)),
                ["x min", "x max", "y min", "y max", "z min", "z max"],
            ),
        ];
        for (preset, labels) in cases {
            let mut world = World::new();
            let b = ftgsxr_core::Aabb {
                min: bevy::math::Vec3::splat(-1.0),
                max: bevy::math::Vec3::splat(1.0),
            };
            let trim = Trim {
                mode: TrimMode::Box,
                ..Trim::initial(&b)
            };
            trimmed_instance(&mut world, preset, Some(trim), false);
            let drawn = texts(&mut world);
            for l in labels {
                assert!(drawn.iter().any(|t| t == l), "{preset:?}: {l} in {drawn:?}");
            }
            assert!(drawn.iter().any(|t| t == "model units"));
        }
    }

    // Private (I-23's panel cases, Q-375; E17, §1.7): turning the box on from no trim
    // sends the initial shapes, the box the untrimmed B_m; "Reset trim" sends ResetTrim;
    // the notice shows when nothing is kept.
    #[test]
    fn scene_tab_trim_turns_on_resets_and_warns() {
        use ftgsxr_core::{Preset, Trim, TrimMode};
        let mut world = World::new();
        let (e, b) = trimmed_instance(&mut world, Preset::YUp, None, false);
        let want = Trim {
            mode: TrimMode::Box,
            ..Trim::initial(&b)
        };
        assert_eq!(
            actions(&click(&mut world, scene_tab, "box")),
            vec![(e, InstanceAction::SetTrim(want))]
        );
        assert_eq!(
            actions(&click(&mut world, scene_tab, "Reset trim")),
            vec![(e, InstanceAction::ResetTrim)]
        );
        assert!(
            !texts(&mut world)
                .iter()
                .any(|t| t == "Trim keeps no primitives")
        );
        let mut world = World::new();
        trimmed_instance(&mut world, Preset::YUp, Some(want), true);
        assert!(
            texts(&mut world)
                .iter()
                .any(|t| t == "Trim keeps no primitives")
        );
    }

    // §1.5 Poses (revision 44, Q-371): a "Fit to" toggle beside the fit action, in the
    // Scene tab the desktop dock and the XR panel share, flips the one global fit box
    // and sends no instance action.
    #[test]
    fn scene_tab_fit_to_toggles_the_global_fit_box() {
        let mut world = World::new();
        world.insert_resource(crate::manipulate::ManipulateSettings::default());
        instance(&mut world, RenderMode::Splats, false);
        assert_eq!(
            actions(&click(&mut world, scene_tab, "Fit to: content")),
            vec![]
        );
        assert!(
            !world
                .resource::<crate::manipulate::ManipulateSettings>()
                .fit_to_content
        );
        click(&mut world, scene_tab, "Fit to: whole model");
        assert!(
            world
                .resource::<crate::manipulate::ManipulateSettings>()
                .fit_to_content
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

    // Q-246 (#416, #436): a press on the scrub bar is a scrub from the press to the
    // release, whether the pointer then moves or not: a click is a scrub while pressed.
    #[test]
    fn playback_tab_a_press_on_the_scrub_bar_is_a_scrub_until_the_release() {
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
        let (_, held) = playback_frame(&mut world, &ctx, 0.0, vec![moved(30.0)]);
        assert!(!held, "a hover is not a scrub");
        // A click: pressed and released in place.
        let (_, held) = playback_frame(&mut world, &ctx, 0.02, vec![press(30.0, true)]);
        assert!(held, "a press is a scrub, with no move");
        let (_, held) = playback_frame(&mut world, &ctx, 0.04, vec![]);
        assert!(held, "the scrub lasts while the press does");
        let (_, held) = playback_frame(&mut world, &ctx, 0.06, vec![press(30.0, false)]);
        assert!(!held, "the release of a click ends the scrub");
        // A drag: pressed, moved along the bar, released.
        let (_, held) = playback_frame(&mut world, &ctx, 0.08, vec![press(30.0, true)]);
        assert!(held, "a drag is a scrub from its press");
        let (i, held) = playback_frame(&mut world, &ctx, 0.10, vec![moved(80.0)]);
        assert!(seek(&i), "the drag moves the scrub bar");
        assert!(held, "the scrub lasts while the drag does");
        let (_, held) = playback_frame(&mut world, &ctx, 0.12, vec![moved(90.0)]);
        assert!(held, "the scrub lasts while the drag does");
        let (_, held) = playback_frame(&mut world, &ctx, 0.14, vec![press(90.0, false)]);
        assert!(!held, "the release ends the scrub");
        let (_, held) = playback_frame(&mut world, &ctx, 0.16, vec![]);
        assert!(!held, "nothing pressed, no scrub");
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

    // #620: a failed entry's message (§1.5 Library, "failed (with the message)"; the
    // §1.15 texts) is drawn whole inside a dock at §1.7's narrowest width, with the
    // entry's name and its Retry button.
    const LONG: &str = "E-MEM-BUDGET: Needs 162 MB, budget 134 MB; reduce to budget or \
                        choose a tier (N = 3000000)";

    /// A world whose library holds one entry, `kitchen`, in `state`.
    fn library_with(state: crate::library::EntryState) -> World {
        let mut world = World::new();
        world.init_resource::<crate::ui::UiState>();
        let mut library = crate::library::Library::default();
        library.entries.push(crate::library::LibraryEntry {
            source: 0,
            entry: ftgsxr_io::Entry {
                name: "kitchen".into(),
                kind: ftgsxr_io::EntryKind::Exchange,
                location: "/m/kitchen.safetensors".into(),
                size: None,
                mtime: None,
                sha256: None,
                thumbnail: None,
                audio: None,
                scene: None,
            },
            info: None,
            identity: None,
            thumbnail: None,
            state,
        });
        world.insert_resource(library);
        world
    }

    /// A context with the system font the app installs (Q-211), so texts have widths.
    fn font_context() -> egui::Context {
        let ctx = egui::Context::default();
        assert!(crate::ui::load_system_font(&ctx), "a system font");
        ctx
    }

    /// One frame of the Library tab in a column `width` points wide, with `events`: each
    /// text drawn with its rectangle, and what the user did.
    fn library_frame(
        ctx: &egui::Context,
        world: &mut World,
        width: f32,
        events: Vec<egui::Event>,
    ) -> (Vec<(String, egui::Rect)>, Vec<Out>) {
        let (texts, _, out) = library_frame_shapes(ctx, world, width, events);
        (texts, out)
    }

    /// `library_frame`, with every painted shape's visual bounds as well (the frames of
    /// text fields and checkboxes, which draw no text of their own).
    fn library_frame_shapes(
        ctx: &egui::Context,
        world: &mut World,
        width: f32,
        events: Vec<egui::Event>,
    ) -> (Vec<(String, egui::Rect)>, Vec<egui::Rect>, Vec<Out>) {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1000.0, 1000.0),
            )),
            events,
            ..egui::RawInput::default()
        };
        let (mut out, mut hv) = (Vec::new(), None);
        let column = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, 1000.0));
        let mut full = ctx.run_ui(input, |ui| {
            ui.scope_builder(egui::UiBuilder::new().max_rect(column), |ui| {
                library_tab(world, ui, &mut out, &mut hv)
            });
        });
        full.textures_delta.clear();
        let (mut texts, mut rects) = (Vec::new(), Vec::new());
        fn walk(
            s: &egui::Shape,
            texts: &mut Vec<(String, egui::Rect)>,
            rects: &mut Vec<egui::Rect>,
        ) {
            match s {
                egui::Shape::Text(t) => texts.push((
                    t.galley.text().to_string(),
                    t.galley.rect.translate(t.pos.to_vec2()),
                )),
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, texts, rects)),
                _ => {}
            }
            // A shape that paints nothing (an empty mesh or text) has no finite bounds.
            let r = s.visual_bounding_rect();
            if !matches!(s, egui::Shape::Vec(_)) && r.is_finite() {
                rects.push(r);
            }
        }
        full.shapes
            .iter()
            .for_each(|c| walk(&c.shape, &mut texts, &mut rects));
        (texts, rects, out)
    }

    /// The texts drawn by two frames of the Library tab `width` points wide (the second
    /// laid out from the first's state).
    fn library_texts(world: &mut World, width: f32) -> Vec<(String, egui::Rect)> {
        let ctx = font_context();
        library_frame(&ctx, world, width, Vec::new());
        library_frame(&ctx, world, width, Vec::new()).0
    }

    /// Draws the Library tab `width` points wide and clicks `label`: what the user did.
    fn library_click(world: &mut World, width: f32, label: &str) -> Vec<Out> {
        let ctx = font_context();
        library_frame(&ctx, world, width, Vec::new());
        let (texts, _) = library_frame(&ctx, world, width, Vec::new());
        let at = texts
            .iter()
            .find(|(t, _)| t == label)
            .map(|(_, r)| r.center())
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
            out.extend(library_frame(&ctx, world, width, events).1);
        }
        out
    }

    fn loads(out: &[Out]) -> Vec<crate::library::LibraryCommand> {
        out.iter()
            .filter_map(|o| match o {
                Out::Library(c @ crate::library::LibraryCommand::Load { .. }) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn library_tab_wraps_a_long_failure_text_inside_the_narrowest_dock() {
        let width = 160.0;
        let mut world = library_with(crate::library::EntryState::Failed(LONG.into()));
        let texts = library_texts(&mut world, width);
        for label in [LONG, "kitchen", "Retry"] {
            assert!(
                texts.iter().any(|(t, _)| t == label),
                "{label} is drawn: {texts:?}"
            );
        }
        // The entry's own row (#627's tests check the whole tab). Half a point of slack
        // for the layout's rounding of positions.
        for (t, r) in texts
            .iter()
            .filter(|(t, _)| [LONG, "kitchen", "Retry"].contains(&&**t))
        {
            assert!(r.min.x >= -0.5 && r.max.x <= width + 0.5, "{t:?} at {r:?}");
        }
    }

    #[test]
    fn library_tab_retry_of_a_long_failure_loads_the_entry_with_the_reduce_flag() {
        for reduce in [false, true] {
            let mut world = library_with(crate::library::EntryState::Failed(LONG.into()));
            world.resource_mut::<crate::ui::UiState>().reduce = reduce;
            let out = library_click(&mut world, 160.0, "Retry");
            assert_eq!(
                loads(&out),
                vec![crate::library::LibraryCommand::Load { entry: 0, reduce }]
            );
        }
    }

    #[test]
    fn library_tab_shows_a_short_failure_with_the_name_and_retry() {
        let text = "E-IO: Could not read /m/kitchen.safetensors";
        let mut world = library_with(crate::library::EntryState::Failed(text.into()));
        let texts = library_texts(&mut world, 320.0);
        for label in [text, "kitchen", "Retry"] {
            assert!(
                texts.iter().any(|(t, _)| t == label),
                "{label} is drawn: {texts:?}"
            );
        }
        let out = library_click(&mut world, 320.0, "Retry");
        assert_eq!(
            loads(&out),
            vec![crate::library::LibraryCommand::Load {
                entry: 0,
                reduce: false
            }]
        );
    }

    // #627: every control of the Library tab inside the tab's width, from §1.7's
    // narrowest dock to its default and the main panel's (§1.8).

    /// A world whose library holds two local sources with long paths (the second with
    /// its Remove button, Q-49) and one entry, `kitchen`, in `state`.
    fn library_with_source(state: crate::library::EntryState) -> World {
        let mut world = library_with(state);
        for root in [
            "/home/someone/models/openftgs-synth-kitchen-and-courtyard",
            "/home/someone/more-models/openftgs-synth-studio",
        ] {
            let spec = source_from_input(root).expect("an absolute path is a source");
            let source: std::sync::Arc<dyn ftgsxr_io::ModelSource> =
                std::sync::Arc::new(ftgsxr_io::LocalDir { root: root.into() });
            world
                .resource_mut::<crate::library::Library>()
                .sources
                .push((spec, source));
        }
        world
    }

    /// The entry states, each with the texts its row draws.
    fn entry_states() -> Vec<crate::library::EntryState> {
        use crate::library::EntryState;
        vec![
            EntryState::Listed,
            EntryState::Fetching(ftgsxr_io::Progress {
                done: 1,
                total: Some(4),
            }),
            EntryState::Loading,
            EntryState::Loaded(Entity::PLACEHOLDER),
            EntryState::Failed(LONG.into()),
            EntryState::Missing,
        ]
    }

    /// Every text and every painted shape of two frames of the Library tab, drawn in a
    /// column `width` points wide with `ctx`, lies inside the column (half a point of
    /// slack for the rounding of positions); `labels` are all drawn.
    fn assert_library_inside(ctx: &egui::Context, world: &mut World, width: f32, labels: &[&str]) {
        library_frame_shapes(ctx, world, width, Vec::new());
        let (texts, rects, _) = library_frame_shapes(ctx, world, width, Vec::new());
        for label in labels {
            assert!(
                texts.iter().any(|(t, _)| t == label),
                "{label} is drawn at {width}: {texts:?}"
            );
        }
        for (t, r) in &texts {
            assert!(
                r.min.x >= -0.5 && r.max.x <= width + 0.5,
                "{t:?} at {r:?} in a column {width} wide"
            );
        }
        for r in &rects {
            assert!(
                r.min.x >= -0.5 && r.max.x <= width + 0.5,
                "a shape at {r:?} in a column {width} wide"
            );
        }
    }

    /// The controls every state draws: the source row, the new source's row, the
    /// pickers, the filters.
    const CONTROLS: [&str; 12] = [
        "Refresh",
        "Remove",
        "https://… index or /path/to/folder",
        "Add source",
        "Open files…",
        "Add folder…",
        "Name",
        "ftgs-if",
        "checkpoint",
        "PLY",
        "folders",
        "Reduce to budget",
    ];

    #[test]
    fn library_tab_keeps_every_control_inside_the_narrowest_and_the_default_dock() {
        for width in [160.0, 320.0] {
            for state in entry_states() {
                let mut world = library_with_source(state);
                assert_library_inside(&font_context(), &mut world, width, &CONTROLS);
            }
        }
    }

    #[test]
    fn library_tab_keeps_a_missing_entry_s_buttons_inside_the_narrowest_dock() {
        let mut world = library_with_source(crate::library::EntryState::Missing);
        let labels = ["kitchen", "missing", "Locate…", "Replace with…"];
        assert_library_inside(&font_context(), &mut world, 160.0, &labels);
    }

    #[test]
    fn library_tab_keeps_every_control_inside_the_main_panel() {
        // §1.8: the 1024-pixel texture at the panel's zoom, at each end of §1.11's
        // `ui_scale` range and its default, less the central panel's margins, with the
        // panel's style (Q-267, Q-293); rounded down to a whole point, where the
        // separator's half-point stroke lands on the edge.
        let ctx = font_context();
        ctx.all_styles_mut(crate::ui::panel::panel_style);
        let margins = egui::Frame::central_panel(&ctx.global_style())
            .total_margin()
            .sum()
            .x;
        for ui_scale in [0.8, 1.0, 1.5] {
            let width = (1024.0 / crate::ui::panel::panel_zoom(ui_scale) - margins).floor();
            for state in entry_states() {
                let mut world = library_with_source(state);
                assert_library_inside(&ctx, &mut world, width, &CONTROLS);
            }
        }
    }

    #[test]
    fn library_tab_adds_a_typed_source_and_toggles_the_kinds_in_the_narrowest_dock() {
        let mut world = library_with_source(crate::library::EntryState::Listed);
        // Click the source field (its hint text), then type into it.
        let ctx = font_context();
        library_frame(&ctx, &mut world, 160.0, Vec::new());
        let (texts, _) = library_frame(&ctx, &mut world, 160.0, Vec::new());
        let hint = "https://… index or /path/to/folder";
        let at = texts
            .iter()
            .find(|(t, _)| t == hint)
            .map(|(_, r)| r.center())
            .expect("the source field is drawn");
        let press = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        for events in [
            vec![egui::Event::PointerMoved(at)],
            vec![press(true)],
            vec![press(false)],
            vec![egui::Event::Text("/m/more".into())],
        ] {
            library_frame(&ctx, &mut world, 160.0, events);
        }
        assert_eq!(world.resource::<crate::ui::UiState>().new_source, "/m/more");
        let out = library_click(&mut world, 160.0, "Add source");
        let added: Vec<_> = out
            .iter()
            .filter_map(|o| match o {
                Out::Library(crate::library::LibraryCommand::AddSource(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(added, vec![source_from_input("/m/more").unwrap()]);
        for (i, kind) in super::KINDS.iter().enumerate() {
            let before = world.resource::<crate::ui::UiState>().filter_kinds;
            library_click(&mut world, 160.0, kind);
            let after = world.resource::<crate::ui::UiState>().filter_kinds;
            assert_eq!(after[i], !before[i], "{kind} toggles");
            for j in (0..4).filter(|&j| j != i) {
                assert_eq!(after[j], before[j], "{kind} leaves the others");
            }
        }
    }
}
