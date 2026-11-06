// SPDX-License-Identifier: Apache-2.0

//! Panel mode (§1.8 User interface, D-21): on the headset (the `Headset` marker, Q-266)
//! the interface draws on the main panel's texture and the wrist menu's, each through an
//! image-target camera with its own egui context and schedule, instead of the window.
//!
//! The main panel shows the desktop's tabs without the mouse-only items, plus Comfort: a
//! title bar at the top (Q-306), a tab row, the tab's content, the toast line at the
//! bottom, and a Back button at the top of every sub-page (the confirmation dialogs, the
//! training cameras, the request replay); nothing draws a hover tooltip. The wrist menu shows five buttons. Both take
//! the pointer from `PanelPointer` (Q-263, Q-279).

use bevy::camera::visibility::{RenderLayers, Visibility};
use bevy::camera::{Camera, Camera3d, RenderTarget};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::prelude::*;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::math::Vec2;
use bevy_egui::{EguiContext, EguiSchedule, egui};
use ftgsxr_core::PlayState;

use super::dock::{self, Out};
use super::{Page, Tab, UiCue, UiState, UiSurface, WorldPanel, WristMenu};
use crate::intent::{Feature, InstanceAction, Intent, PlaybackCommand};
use crate::library::LoadedModel;
use crate::manipulate::Selected;
use crate::platform::Headset;
use crate::playback::Playback;

/// The main panel's egui pass.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PanelPass;

/// The wrist menu's egui pass.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct WristPass;

/// An image-target camera `ui` spawned for a quad: its egui context draws the main panel,
/// or the wrist menu when `wrist`.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PanelCamera {
    /// The quad (`WorldPanel`) whose image the camera renders to.
    pub(crate) quad: Entity,
    /// Whether the quad is the wrist menu.
    pub(crate) wrist: bool,
}

/// The panel contexts' zoom factor at `ui_scale` 1 (§1.8 User interface).
// Q-267
pub(crate) const PANEL_ZOOM: f32 = 1.75;

/// The smallest target on the main panel at the default size, in its pixels (§1.8: 22 px
/// at 1024 × 768).
// Q-267
pub(crate) const MIN_TARGET_PX: f32 = 22.0;

/// The smallest text on the main panel at the default size, in its pixels (§1.8: 24 px).
// Q-293
pub(crate) const MIN_TEXT_PX: f32 = 24.0;

/// The panel contexts' zoom factor at `ui_scale` (§1.11, 0.8 to 1.5).
// Q-267
pub(crate) fn panel_zoom(ui_scale: f32) -> f32 {
    PANEL_ZOOM * ui_scale
}

/// The panels' style: every text style at least `MIN_TEXT_PX` and every target at least
/// `MIN_TARGET_PX` high at the default zoom (egui 0.36's body text is 13 points, 22.75 px
/// at 1.75), both scaled by `ui_scale` with the zoom.
// Q-267, Q-293
pub(crate) fn panel_style(style: &mut egui::Style) {
    for font in style.text_styles.values_mut() {
        font.size = font.size.max(MIN_TEXT_PX / PANEL_ZOOM);
    }
    let y = &mut style.spacing.interact_size.y;
    *y = y.max(MIN_TARGET_PX / PANEL_ZOOM);
}

/// The panel camera's order: before the eyes, so the panels drawn this frame are seen
/// this frame (PRD #452's spike check 3).
const PANEL_ORDER: isize = -1;

/// Follows the surfaces (§1.8 User interface, Q-266): with the `Headset` marker, one
/// image-target camera for `UiSurface::Panel`'s quad and one for the first wrist menu
/// quad, each rendering to its quad's `WorldPanel.image`; a camera whose quad, image or
/// surface is gone is despawned, and a re-created image (a resize) gets a new camera.
/// Each camera has tonemapping off (#549).
// Q-294
pub(super) fn follow_surfaces(
    mut commands: Commands,
    headset: Option<Res<Headset>>,
    surface: Res<UiSurface>,
    panels: Query<&WorldPanel>,
    wrists: Query<(Entity, &WorldPanel), With<WristMenu>>,
    cameras: Query<(Entity, &PanelCamera, &RenderTarget)>,
) {
    let mut wanted = Vec::new();
    if headset.is_some() {
        if let UiSurface::Panel { entity, .. } = *surface
            && let Ok(p) = panels.get(entity)
        {
            wanted.push((entity, p.image.clone(), false));
        }
        // One wrist menu: each context needs its own schedule.
        if let Some((e, p)) = wrists.iter().next() {
            wanted.push((e, p.image.clone(), true));
        }
    }
    for (camera, pc, target) in &cameras {
        let kept = wanted.iter().position(|(quad, image, wrist)| {
            *quad == pc.quad
                && *wrist == pc.wrist
                && matches!(target, RenderTarget::Image(i) if i.handle == *image)
        });
        match kept {
            Some(i) => {
                wanted.remove(i);
            }
            None => commands.entity(camera).despawn(),
        }
    }
    for (quad, image, wrist) in wanted {
        let schedule = if wrist {
            EguiSchedule::new(WristPass)
        } else {
            EguiSchedule::new(PanelPass)
        };
        commands.spawn((
            Camera3d::default(),
            Camera {
                order: PANEL_ORDER,
                ..Default::default()
            },
            // #549: Bevy specializes a tonemapping pipeline for every view, and its
            // default, `TonyMcMapface`, needs the `tonemapping_luts` feature. The panel
            // cameras are not HDR, so the pass never runs: no pixel changes.
            Tonemapping::None,
            RenderTarget::Image(image.into()),
            RenderLayers::none(),
            schedule,
            PanelCamera { quad, wrist },
        ));
    }
}

/// Idles the panel cameras whose quads are hidden (#269): a camera is active exactly when
/// its quad's `Visibility` is not `Hidden` (a quad without one counts as shown), so a
/// hidden panel renders no view and its pass draws nothing. Run in `PostUpdate` before
/// egui's pass, after `Update`'s summon and wrist-menu rules, so a quad that shows draws
/// in that frame.
// Q-324
pub(super) fn idle_hidden(
    mut cameras: Query<(&PanelCamera, &mut Camera)>,
    quads: Query<&Visibility>,
) {
    for (panel, mut camera) in &mut cameras {
        let shown = quads
            .get(panel.quad)
            .map_or(true, |v| *v != Visibility::Hidden);
        if camera.is_active != shown {
            camera.is_active = shown;
        }
    }
}

/// The context of the main panel's (or the wrist menu's) camera, and whether the camera
/// is active (Q-324: an idle camera's pass draws nothing, but its font is installed, so
/// the first frame a quad shows has it).
fn context(world: &mut World, wrist: bool) -> Option<(Entity, egui::Context, bool)> {
    world
        .query::<(Entity, &PanelCamera, &Camera, &mut EguiContext)>()
        .iter_mut(world)
        .find(|(_, p, _, _)| p.wrist == wrist)
        .map(|(e, _, cam, mut c)| (e, c.get_mut().clone(), cam.is_active))
}

/// Installs the system font in a new context once (Q-211); whether it has one.
fn font_for(font: &mut Option<(Entity, bool)>, entity: Entity, ctx: &egui::Context) -> bool {
    if font.is_none_or(|(e, _)| e != entity) {
        *font = Some((entity, super::load_system_font(ctx)));
    }
    font.is_some_and(|(_, f)| f)
}

/// A root `Ui` over the whole texture.
fn root(ctx: &egui::Context, id: &'static str) -> egui::Ui {
    egui::Ui::new(
        ctx.clone(),
        id.into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    )
}

/// The main panel (§1.8 User interface) on its context, run in `PanelPass`.
pub(super) fn draw_panel(
    world: &mut World,
    mut font: Local<Option<(Entity, bool)>>,
    mut hovered: Local<Option<egui::Id>>,
) {
    let Some((entity, ctx, active)) = context(world, false) else {
        return;
    };
    // Q-324: the font first, so a summoned panel's first frame has it.
    if !font_for(&mut font, entity, &ctx) || !active {
        return;
    }
    let mut ui = root(&ctx, "ftgsxr-panel");
    panel_style(ui.style_mut());
    let mut out = Vec::new();
    let mut now_hovered = None;
    panel_layout(world, &mut ui, &mut out, &mut now_hovered);
    if now_hovered.is_some() && now_hovered != *hovered {
        out.push(Out::Cue(UiCue::Hover));
    }
    *hovered = now_hovered;
    dock::send(world, out);
}

/// The main panel's title bar as a share of its texture's height: the strip
/// `ftgsxr_xr`'s panel grab tests (its own copy of the fraction, since it depends on
/// this crate), so the bar drawn and the strip grabbed coincide.
// Q-306
const TITLE_FRACTION: f32 = 1.0 / 16.0;

/// The main panel's layout: the title bar at the top, then the tab row (or a sub-page's
/// Back button), the toast line at the bottom, the tab's content or the sub-page between.
// Q-294, Q-306, Q-328
pub(super) fn panel_layout(
    world: &mut World,
    ui: &mut egui::Ui,
    out: &mut Vec<Out>,
    hv: &mut Option<egui::Id>,
) {
    // Q-333
    super::touch::touch_options(ui.ctx());
    super::touch::touch_look(ui);
    let (tab, page, asking) = {
        let s = world.resource::<UiState>();
        (s.tab, s.page, s.confirm.is_some())
    };
    // Q-328: no text and no widget, so a pinch on it only grabs (Q-315); a grip mark,
    // painted, not a glyph, shows where to grab.
    let view = ui.ctx().viewport_rect();
    let bar = TITLE_FRACTION * view.height();
    egui::Panel::top("ftgsxr-panel-title")
        .exact_size(bar)
        .frame(egui::Frame::NONE.fill(ui.visuals().faint_bg_color))
        .show(ui, |ui| {
            let grip = egui::Rect::from_center_size(
                egui::pos2(view.center().x, view.min.y + bar / 2.0),
                egui::vec2(view.width() / 4.0, bar / 6.0),
            );
            ui.painter().rect_filled(
                grip,
                // epaint clamps a radius to half the rect's shorter side: fully rounded ends.
                egui::CornerRadius::same(u8::MAX),
                ui.visuals().weak_text_color(),
            );
        });
    // egui hits the widget nearest the pointer within `interact_radius` of its rect: the
    // tab row's top margin keeps every widget farther than that from the bar.
    let mut row = egui::Frame::side_top_panel(ui.style());
    row.inner_margin.top = row
        .inner_margin
        .top
        .max((ui.style().interaction.interact_radius + 1.0).ceil() as i8);
    egui::Panel::top("ftgsxr-panel-top")
        .frame(row)
        .show(ui, |ui| {
            if asking || page.is_some() {
                if dock::button(ui, "◀ Back", out, hv) {
                    if asking {
                        // Back is No: the question is dropped (Q-199, Q-200, Q-256).
                        out.push(Out::Answer(false));
                    } else {
                        world.resource_mut::<UiState>().page = None;
                    }
                }
            } else {
                ui.horizontal_wrapped(|ui| {
                    for t in Tab::PANEL {
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
            }
        });
    dock::toast_line(world, ui);
    egui::CentralPanel::default().show(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            if asking {
                dock::confirm_body(world, ui, out);
            } else {
                match page {
                    Some(Page::Cameras) => dock::training_cameras(world, ui, out, hv),
                    Some(Page::Replay) => dock::request_replay(world, ui, out, hv),
                    None => dock::tab_body(world, ui, tab, out, hv, true),
                }
            }
        });
    });
}

/// The wrist menu's buttons (§1.8 User interface). The passthrough button is hidden
/// until M-6 builds passthrough (§1.15 E-XR-UNSUPPORTED: "the control is hidden",
/// Q-292), so M-5 shows four.
// Q-292
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WristButton {
    PlayPause,
    Panel,
    ResetPose,
    Fit,
}

impl WristButton {
    /// §1.8's order, left to right, without the hidden passthrough button.
    // Q-291, Q-292
    pub(crate) const ALL: [WristButton; 4] = [
        WristButton::PlayPause,
        WristButton::Panel,
        WristButton::ResetPose,
        WristButton::Fit,
    ];

    /// The button's label, the play button's following the clock.
    pub(crate) fn label(self, playing: bool) -> &'static str {
        match self {
            WristButton::PlayPause if playing => "Pause",
            WristButton::PlayPause => "Play",
            WristButton::Panel => "Panel",
            WristButton::ResetPose => "Reset\npose",
            WristButton::Fit => "Fit",
        }
    }
}

/// The wrist menu's button side as a share of its texture's width: 3 cm on the 0.16 m
/// quad (Q-265).
// Q-291
const WRIST_BUTTON_SHARE: f32 = 0.03 / 0.16;

/// The wrist menu's buttons in a texture of `size_px` pixels: the shown buttons (four in
/// M-5) as squares in a row, each 3 cm of the quad's 0.16 m (120 px of 640) or the
/// texture's height if smaller, spaced evenly and centred vertically (Q-291, Q-292).
// Q-291, Q-292
pub(crate) fn wrist_buttons(size_px: Vec2) -> [(WristButton, egui::Rect); 4] {
    let n = WristButton::ALL.len() as f32;
    let side = (WRIST_BUTTON_SHARE * size_px.x).min(size_px.y);
    let gap = (size_px.x - n * side) / (n + 1.0);
    let top = (size_px.y - side) / 2.0;
    WristButton::ALL.map(|b| {
        let i = WristButton::ALL.iter().position(|&x| x == b).unwrap_or(0) as f32;
        let min = egui::pos2(gap + i * (side + gap), top);
        (b, egui::Rect::from_min_size(min, egui::vec2(side, side)))
    })
}

/// Draws the wrist menu's buttons over `ui`'s whole texture; the one clicked, if any. Its
/// buttons follow the touch press (Q-332, Q-333).
pub(crate) fn wrist_ui(ui: &mut egui::Ui, playing: bool) -> Option<WristButton> {
    // Q-333
    super::touch::touch_options(ui.ctx());
    super::touch::touch_look(ui);
    let ppp = ui.ctx().pixels_per_point();
    let view = ui.ctx().viewport_rect();
    ui.painter().rect_filled(view, 0.0, ui.visuals().panel_fill);
    let size = view.size() * ppp;
    let mut clicked = None;
    for (b, r) in wrist_buttons(Vec2::new(size.x, size.y)) {
        let rect = egui::Rect::from_min_max(
            (r.min.to_vec2() / ppp).to_pos2(),
            (r.max.to_vec2() / ppp).to_pos2(),
        );
        if ui.put(rect, egui::Button::new(b.label(playing))).clicked() {
            clicked = Some(b);
        }
    }
    clicked
}

/// A wrist button's intents (§1.8 User interface): play or pause the clock, the main
/// panel, and the selected instances' reset pose and fit (none with nothing selected).
pub(crate) fn wrist_intents(b: WristButton, selected: &[Entity]) -> Vec<Intent> {
    let each = |action| {
        selected
            .iter()
            .map(|&instance| Intent::Instance { instance, action })
            .collect()
    };
    match b {
        WristButton::PlayPause => vec![Intent::Playback(PlaybackCommand::Toggle)],
        WristButton::Panel => vec![Intent::Toggle(Feature::Panel)],
        WristButton::ResetPose => each(InstanceAction::ResetPose),
        WristButton::Fit => each(InstanceAction::Fit),
    }
}

/// The wrist menu on its context, run in `WristPass`: a pressed button's intents.
pub(super) fn draw_wrist(world: &mut World, mut font: Local<Option<(Entity, bool)>>) {
    let Some((entity, ctx, active)) = context(world, true) else {
        return;
    };
    // Q-324: the font first, so the wrist menu's first frame has it.
    if !font_for(&mut font, entity, &ctx) || !active {
        return;
    }
    let playing = world
        .get_resource::<Playback>()
        .is_some_and(|p| p.clock.state() == PlayState::Playing);
    let mut ui = root(&ctx, "ftgsxr-wrist");
    panel_style(ui.style_mut());
    let Some(b) = wrist_ui(&mut ui, playing) else {
        return;
    };
    let selected: Vec<Entity> = world
        .query_filtered::<Entity, (With<LoadedModel>, With<Selected>)>()
        .iter(world)
        .collect();
    for i in wrist_intents(b, &selected) {
        world.write_message(i);
    }
    world.write_message(UiCue::Click);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::pointer_events;

    /// One frame of the wrist menu at the zoom of panel mode on a 640 × 200 px texture,
    /// with `events`: the texts drawn (in pixels) and the button clicked.
    fn wrist_frame(
        ctx: &egui::Context,
        events: Vec<egui::Event>,
    ) -> (Vec<(String, Vec2)>, Option<WristButton>) {
        wrist_frame_at(ctx, events, PANEL_ZOOM)
    }

    /// `wrist_frame` at the zoom `ppp` (`panel_zoom` of a `ui_scale`).
    fn wrist_frame_at(
        ctx: &egui::Context,
        events: Vec<egui::Event>,
        ppp: f32,
    ) -> (Vec<(String, Vec2)>, Option<WristButton>) {
        wrist_frame_timed(ctx, events, ppp, 0.0)
    }

    /// `wrist_frame_at` at `time` seconds.
    fn wrist_frame_timed(
        ctx: &egui::Context,
        events: Vec<egui::Event>,
        ppp: f32,
        time: f64,
    ) -> (Vec<(String, Vec2)>, Option<WristButton>) {
        let (texts, clicked, _) = wrist_frame_fills(ctx, events, ppp, time);
        (texts, clicked)
    }

    /// `wrist_frame_timed`, with the filled rects painted (in points) and their fills.
    #[allow(clippy::type_complexity)]
    fn wrist_frame_fills(
        ctx: &egui::Context,
        events: Vec<egui::Event>,
        ppp: f32,
        time: f64,
    ) -> (
        Vec<(String, Vec2)>,
        Option<WristButton>,
        Vec<(egui::Rect, egui::Color32)>,
    ) {
        let input = egui::RawInput {
            time: Some(time),
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(640.0 / ppp, 200.0 / ppp),
            )),
            events,
            ..Default::default()
        };
        ctx.set_zoom_factor(ppp);
        let mut clicked = None;
        let mut full = ctx.run_ui(input, |ui| clicked = wrist_ui(ui, false));
        full.textures_delta.clear();
        let mut texts = Vec::new();
        fn walk(s: &egui::Shape, ppp: f32, texts: &mut Vec<(String, Vec2)>) {
            match s {
                egui::Shape::Text(t) => {
                    let c = t.pos + t.galley.size() / 2.0;
                    texts.push((t.galley.text().to_string(), Vec2::new(c.x, c.y) * ppp));
                }
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, ppp, texts)),
                _ => {}
            }
        }
        full.shapes
            .iter()
            .for_each(|c| walk(&c.shape, ppp, &mut texts));
        let fills = full
            .shapes
            .iter()
            .filter_map(|c| match &c.shape {
                egui::Shape::Rect(r) => Some((r.rect, r.fill)),
                _ => None,
            })
            .collect();
        (texts, clicked, fills)
    }

    /// The panel cameras (§1.8 User interface, #549, `cpu`): with the `Headset` marker,
    /// a shown main panel and a wrist menu, each camera `follow_surfaces` spawns has
    /// tonemapping off.
    #[test]
    fn follow_surfaces_spawns_panel_cameras_with_tonemapping_off() {
        use bevy::asset::Handle;
        use bevy::ecs::system::RunSystemOnce;
        use bevy::math::UVec2;

        let mut world = World::new();
        world.insert_resource(Headset);
        let quad = world
            .spawn(WorldPanel {
                image: Handle::default(),
            })
            .id();
        world.spawn((
            WorldPanel {
                image: Handle::default(),
            },
            WristMenu,
        ));
        world.insert_resource(UiSurface::Panel {
            entity: quad,
            size_px: UVec2::new(1024, 768),
            size_m: Vec2::new(0.56, 0.42),
        });
        world
            .run_system_once(follow_surfaces)
            .expect("follow_surfaces runs");
        let mut cameras = world.query::<(&PanelCamera, Option<&Tonemapping>)>();
        let found: Vec<_> = cameras
            .iter(&world)
            .map(|(pc, t)| (pc.wrist, t.copied()))
            .collect();
        assert_eq!(found.len(), 2, "a main-panel and a wrist-menu camera");
        for (wrist, tonemapping) in found {
            assert_eq!(
                tonemapping,
                Some(Tonemapping::None),
                "the {} camera's tonemapping",
                if wrist { "wrist menu" } else { "main panel" }
            );
        }
    }

    /// The wrist menu (§1.8 User interface, `cpu`): a `PanelPointer` press and release at
    /// each button's pixel, converted to points as the panel pointer does (Q-290),
    /// clicks that button, and the button gives its intent: play or pause, passthrough,
    /// the main panel, the selected instance's reset pose and fit.
    #[test]
    fn wrist_ui_each_button_gives_its_intent_from_a_press_at_its_pixel() {
        let selected = Entity::from_raw_u32(7).expect("an entity");
        let expected = [
            ("Play", Intent::Playback(PlaybackCommand::Toggle)),
            ("Panel", Intent::Toggle(Feature::Panel)),
            (
                "Reset\npose",
                Intent::Instance {
                    instance: selected,
                    action: InstanceAction::ResetPose,
                },
            ),
            (
                "Fit",
                Intent::Instance {
                    instance: selected,
                    action: InstanceAction::Fit,
                },
            ),
        ];
        for (label, intent) in expected {
            let ctx = egui::Context::default();
            // The second frame, once the context knows its screen.
            wrist_frame(&ctx, Vec::new());
            let (texts, _) = wrist_frame(&ctx, Vec::new());
            let pixel = texts
                .iter()
                .find(|(t, _)| t == label)
                .map(|(_, p)| *p)
                .unwrap_or_else(|| panic!("{label} is drawn: {texts:?}"));
            assert!(
                pixel.x < 640.0 && pixel.y < 200.0,
                "{label} on the texture: {pixel}"
            );
            let mut last = None;
            let mut clicked = Vec::new();
            for select in [vec![], vec![true], vec![false], vec![]] {
                let at = super::super::panel_point(pixel, PANEL_ZOOM);
                let events = pointer_events(last, Some(at), &select, 0.0);
                last = Some(at);
                clicked.extend(wrist_frame(&ctx, events).1);
            }
            assert_eq!(
                clicked.len(),
                1,
                "{label}: one click, {clicked:?} at {pixel}"
            );
            assert_eq!(
                wrist_intents(clicked[0], &[selected]),
                vec![intent],
                "{label}"
            );
        }
    }

    /// With nothing selected, reset pose and fit give no intent.
    #[test]
    fn wrist_intents_reset_and_fit_need_a_selection() {
        assert!(wrist_intents(WristButton::ResetPose, &[]).is_empty());
        assert!(wrist_intents(WristButton::Fit, &[]).is_empty());
    }

    /// The four shown buttons (the passthrough button hidden until M-6, Q-292) are 3 cm
    /// squares (120 px of the 640 × 200 texture), inside it, apart and evenly spaced
    /// (Q-265, Q-291), above Q-267's 60 px.
    #[test]
    fn wrist_buttons_are_four_120_px_squares_in_a_row() {
        let b = wrist_buttons(Vec2::new(640.0, 200.0));
        assert_eq!(b.map(|(w, _)| w), WristButton::ALL);
        // (640 − 4 × 120) / 5 = 32 px around each.
        assert!((b[0].1.min.x - 32.0).abs() < 1e-3 && (b[3].1.max.x - 608.0).abs() < 1e-3);
        let tex = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(640.0, 200.0));
        for (i, (_, r)) in b.iter().enumerate() {
            assert!((r.width() - 120.0).abs() < 1e-3 && (r.height() - 120.0).abs() < 1e-3);
            assert!(tex.contains_rect(*r), "{r:?}");
            if i > 0 {
                assert!(r.min.x > b[i - 1].1.max.x);
            }
        }
    }

    /// One frame of the wrist menu as the panel pointer drives it (Q-333): `select`'s
    /// changes at `pixel` (in the texture's pixels, after a move there), or the pointer
    /// leaving the quad with `None`, rewritten for the touch press, at `time` seconds.
    fn wrist_touch_frame(
        ctx: &egui::Context,
        last: &mut Option<Vec2>,
        pixel: Option<Vec2>,
        select: &[bool],
        ppp: f32,
        time: f64,
    ) -> Option<WristButton> {
        let at = pixel.map(|p| super::super::panel_point(p, ppp));
        let events = super::super::touch::touch_events(ctx, pointer_events(*last, at, select, 0.0));
        *last = at;
        wrist_frame_timed(ctx, events, ppp, time).1
    }

    /// The pixel of the wrist button labelled `label` at the zoom `ppp`, after two frames
    /// that lay the menu out.
    fn wrist_button_pixel(ctx: &egui::Context, label: &str, ppp: f32) -> Vec2 {
        wrist_frame_at(ctx, Vec::new(), ppp);
        let (texts, _) = wrist_frame_at(ctx, Vec::new(), ppp);
        texts
            .iter()
            .find(|(t, _)| t == label)
            .map(|(_, p)| *p)
            .unwrap_or_else(|| panic!("the {label} button is drawn"))
    }

    /// Q-333 (#561, `cpu`): a press on a wrist button whose hit slides 50 px (1.25 cm on
    /// the 0.16 m quad, past Q-332's former 1 cm, inside the 120 px button) before the
    /// release clicks it once, at `ui_scale` 1 and 1.5.
    #[test]
    fn touch_events_a_wrist_press_slid_inside_its_button_clicks_on_release() {
        for scale in [1.0, 1.5] {
            let zoom = panel_zoom(scale);
            let ctx = egui::Context::default();
            let pixel = wrist_button_pixel(&ctx, "Panel", zoom);
            let mut last = None;
            let mut clicked = Vec::new();
            for (i, (dx, select)) in [
                (0.0, vec![]),
                (0.0, vec![true]),
                (25.0, vec![]),
                (50.0, vec![]),
                (50.0, vec![false]),
                (50.0, vec![]),
            ]
            .into_iter()
            .enumerate()
            {
                let at = pixel + Vec2::new(dx, 0.0);
                let t = 0.1 * i as f64;
                clicked.extend(wrist_touch_frame(
                    &ctx,
                    &mut last,
                    Some(at),
                    &select,
                    zoom,
                    t,
                ));
            }
            assert_eq!(clicked, vec![WristButton::Panel], "ui_scale {scale}");
        }
    }

    /// Q-333 (#561, `cpu`): the touch press's cases on the "Panel" wrist button, at
    /// `ui_scale` 1 and 1.5, each frame a hit (in pixels from the button's centre, `None`
    /// off the quad), `select`'s changes there and the time: a press held 2 s clicks once;
    /// slid 2 px (within egui's own tolerance) once, not twice; slid 200 px out of the
    /// 120 px button and released there, never; slid out and back in, or off the quad
    /// and back, then released inside, once; slid 4 px out over the button's edge from
    /// a press near it and released there, never.
    #[test]
    fn touch_events_a_wrist_press_clicks_on_a_release_inside_and_only_there() {
        type Frame = (Option<f32>, Vec<bool>, f64);
        let f = |dx: Option<f32>, select: &[bool], t: f64| -> Frame { (dx, select.to_vec(), t) };
        let cases: [(&str, Vec<Frame>, usize); 6] = [
            (
                "held 2 s",
                vec![
                    f(Some(0.0), &[], 0.0),
                    f(Some(0.0), &[true], 0.1),
                    f(Some(0.0), &[], 1.0),
                    f(Some(0.0), &[false], 2.1),
                ],
                1,
            ),
            (
                "slid 2 px",
                vec![
                    f(Some(0.0), &[], 0.0),
                    f(Some(0.0), &[true], 0.1),
                    f(Some(2.0), &[false], 0.2),
                ],
                1,
            ),
            (
                "released outside",
                vec![
                    f(Some(0.0), &[], 0.0),
                    f(Some(0.0), &[true], 0.1),
                    f(Some(200.0), &[], 0.2),
                    f(Some(200.0), &[false], 0.3),
                ],
                0,
            ),
            (
                "out and back",
                vec![
                    f(Some(0.0), &[], 0.0),
                    f(Some(0.0), &[true], 0.1),
                    f(Some(200.0), &[], 0.2),
                    f(Some(10.0), &[], 0.3),
                    f(Some(10.0), &[false], 0.4),
                ],
                1,
            ),
            (
                "off the quad and back",
                vec![
                    f(Some(0.0), &[], 0.0),
                    f(Some(0.0), &[true], 0.1),
                    f(None, &[], 0.2),
                    f(Some(0.0), &[], 0.3),
                    f(Some(0.0), &[false], 0.4),
                ],
                1,
            ),
            (
                "just over the edge",
                vec![
                    f(Some(57.0), &[], 0.0),
                    f(Some(57.0), &[true], 0.1),
                    f(Some(61.0), &[false], 0.2),
                ],
                0,
            ),
        ];
        for scale in [1.0, 1.5] {
            let zoom = panel_zoom(scale);
            for (name, frames, clicks) in &cases {
                let ctx = egui::Context::default();
                let pixel = wrist_button_pixel(&ctx, "Panel", zoom);
                let mut last = None;
                let mut clicked = Vec::new();
                for (dx, select, t) in frames.iter().cloned().chain([(Some(0.0), vec![], 3.0)]) {
                    let at = dx.map(|dx| pixel + Vec2::new(dx, 0.0));
                    clicked.extend(wrist_touch_frame(&ctx, &mut last, at, &select, zoom, t));
                }
                assert_eq!(
                    clicked,
                    vec![WristButton::Panel; *clicks],
                    "ui_scale {scale}, {name}"
                );
            }
        }
    }

    /// Q-333 (#561, `cpu`): during one press on the "Panel" wrist button, the button
    /// paints with egui's pressed (active) fill while the hit is inside it and not while
    /// it is outside, both once the hit has moved past egui's own click tolerance (50 px
    /// in, 200 px out, 10 px back in) and before (4 px out over its edge).
    #[test]
    fn touch_look_a_wrist_button_looks_pressed_only_while_the_hit_is_inside() {
        let zoom = PANEL_ZOOM;
        let (_, rect_px) = wrist_buttons(Vec2::new(640.0, 200.0))[1];
        let rect = egui::Rect::from_min_max(
            (rect_px.min.to_vec2() / zoom).to_pos2(),
            (rect_px.max.to_vec2() / zoom).to_pos2(),
        );
        let pressed_fill = egui::Style::default().visuals.widgets.active.weak_bg_fill;
        for (start, moves) in [
            (0.0, vec![(50.0, true), (200.0, false), (10.0, true)]),
            (57.0, vec![(61.0, false)]),
        ] {
            let ctx = egui::Context::default();
            let pixel = wrist_button_pixel(&ctx, "Panel", zoom);
            let mut last = None;
            let mut t = 0.0;
            let mut frame = |dx: f32, select: &[bool]| {
                t += 0.1;
                let at = super::super::panel_point(pixel + Vec2::new(dx, 0.0), zoom);
                let events = super::super::touch::touch_events(
                    &ctx,
                    pointer_events(last, Some(at), select, 0.0),
                );
                last = Some(at);
                let (_, _, fills) = wrist_frame_fills(&ctx, events, zoom, t);
                fills
                    .into_iter()
                    .find(|(r, _)| (r.center() - rect.center()).length() < 1.0)
                    .map(|(_, f)| f)
                    .expect("the Panel button paints its rect")
            };
            frame(start, &[]);
            frame(start, &[true]);
            for (dx, inside) in moves {
                frame(dx, &[]);
                let fill = frame(dx, &[]);
                assert_eq!(fill == pressed_fill, inside, "from {start} px, at {dx} px");
            }
        }
    }

    /// §1.8 at the default `ui_scale` (Q-267, Q-293): every text style is at least
    /// 24 px and every target at least 22 px on the texture, and a larger scale keeps
    /// them larger.
    #[test]
    fn panel_style_meets_the_text_and_target_sizes_at_the_default_scale() {
        let mut style = egui::Style::default();
        panel_style(&mut style);
        for scale in [1.0, 1.5] {
            let zoom = panel_zoom(scale);
            for (s, font) in &style.text_styles {
                assert!(
                    font.size * zoom >= 24.0 - 1e-3,
                    "{s:?}: {}",
                    font.size * zoom
                );
            }
            assert!(style.spacing.interact_size.y * zoom >= 22.0 - 1e-3);
        }
        // egui's own larger sizes are kept.
        let heading = egui::Style::default().text_styles[&egui::TextStyle::Heading].size;
        assert_eq!(
            style.text_styles[&egui::TextStyle::Heading].size,
            heading.max(24.0 / 1.75)
        );
    }

    /// The texts of one main panel frame over a 1024 × 768 point screen, and what the
    /// user did.
    fn panel_frame(
        ctx: &egui::Context,
        world: &mut World,
        events: Vec<egui::Event>,
    ) -> (Vec<(String, egui::Pos2)>, Vec<Out>) {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            events,
            ..Default::default()
        };
        let (mut out, mut hv) = (Vec::new(), None);
        let mut full = ctx.run_ui(input, |ui| panel_layout(world, ui, &mut out, &mut hv));
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

    fn panel_world(tab: Tab) -> World {
        let mut world = World::new();
        world.insert_resource(UiState {
            tab,
            ..Default::default()
        });
        world.init_resource::<super::super::DockHooks>();
        world
    }

    /// The main panel's tab row has Comfort, and its Help leaves out the keyboard
    /// shortcuts and the screen-only sections (§1.8 User interface).
    #[test]
    fn panel_layout_has_comfort_and_no_keyboard_shortcuts() {
        let mut world = panel_world(Tab::Help);
        fn keys(_: &mut World, ui: &mut egui::Ui) {
            ui.label("screen-only keys");
        }
        world
            .resource_mut::<super::super::DockHooks>()
            .add_screen_only(Tab::Help, keys);
        let ctx = egui::Context::default();
        panel_frame(&ctx, &mut world, Vec::new());
        let (texts, _) = panel_frame(&ctx, &mut world, Vec::new());
        let has = |label: &str| texts.iter().any(|(t, _)| t.contains(label));
        assert!(has("Comfort"), "{texts:?}");
        assert!(has("openftgs-xr"), "{texts:?}");
        assert!(!has("F1 Help"), "{texts:?}");
        assert!(!has("screen-only keys"), "{texts:?}");
    }

    /// A question is a sub-page whose Back button at the top answers no (§1.8 User
    /// interface); a sub-page's Back returns to its tab.
    #[test]
    fn panel_layout_back_answers_no_and_closes_a_sub_page() {
        let mut world = panel_world(Tab::Scene);
        world.resource_mut::<UiState>().confirm = Some(super::super::Confirm::LoadScene(None));
        let click = |world: &mut World, label: &str| {
            let ctx = egui::Context::default();
            panel_frame(&ctx, world, Vec::new());
            let (texts, _) = panel_frame(&ctx, world, Vec::new());
            let at = texts
                .iter()
                .find(|(t, _)| t == label)
                .map(|(_, p)| *p + egui::vec2(4.0, 4.0))
                .unwrap_or_else(|| panic!("{label} is drawn: {texts:?}"));
            let mut out = Vec::new();
            for select in [vec![], vec![true], vec![false]] {
                let events = pointer_events(
                    Some(Vec2::new(at.x, at.y)),
                    Some(Vec2::new(at.x, at.y)),
                    &select,
                    0.0,
                );
                out.extend(panel_frame(&ctx, world, events).1);
            }
            out
        };
        let out = click(&mut world, "◀ Back");
        assert!(out.iter().any(|o| matches!(o, Out::Answer(false))));
        world.resource_mut::<UiState>().confirm = None;
        world.resource_mut::<UiState>().page = Some(Page::Cameras);
        click(&mut world, "◀ Back");
        assert_eq!(world.resource::<UiState>().page, None);
        assert_eq!(world.resource::<UiState>().tab, Tab::Scene);
    }

    /// The main panel's Environment tab has §1.8's Recenter (Q-311): a click writes
    /// `Intent::Recenter` with the click's sound.
    #[test]
    fn panel_environment_tab_recenters() {
        let mut world = panel_world(Tab::Environment);
        let ctx = egui::Context::default();
        panel_frame(&ctx, &mut world, Vec::new());
        let (texts, _) = panel_frame(&ctx, &mut world, Vec::new());
        let at = texts
            .iter()
            .find(|(t, _)| t == "Recenter")
            .map(|(_, p)| *p + egui::vec2(4.0, 4.0))
            .unwrap_or_else(|| panic!("Recenter is drawn: {texts:?}"));
        let mut out = Vec::new();
        for select in [vec![], vec![true], vec![false]] {
            let at = Vec2::new(at.x, at.y);
            let events = pointer_events(Some(at), Some(at), &select, 0.0);
            out.extend(panel_frame(&ctx, &mut world, events).1);
        }
        assert!(
            out.iter()
                .any(|o| matches!(o, Out::Intent(Intent::Recenter))),
            "a recenter"
        );
        assert!(out.iter().any(|o| matches!(o, Out::Cue(UiCue::Click))));
    }

    /// The textures #542's tests draw the main panel on: (width px, height px, `ui_scale`),
    /// the default 1024 × 768 at §1.11's smallest, default and largest scales, and the
    /// 1.2 m panel's 2195 × 1646 (Q-306).
    const TEXTURES: [(f32, f32, f32); 4] = [
        (1024.0, 768.0, 0.8),
        (1024.0, 768.0, 1.0),
        (1024.0, 768.0, 1.5),
        (2195.0, 1646.0, 1.0),
    ];

    /// One frame of the main panel as `draw_panel` draws it on a `w` × `h` px texture at
    /// `zoom`, with `events` in the context's points: the shapes painted (in points), what
    /// the user did, and the widget hovered.
    fn texture_shapes(
        ctx: &egui::Context,
        world: &mut World,
        (w, h, zoom): (f32, f32, f32),
        events: Vec<egui::Event>,
    ) -> (Vec<egui::epaint::ClippedShape>, Vec<Out>, Option<egui::Id>) {
        texture_shapes_at(ctx, world, (w, h, zoom), events, None)
    }

    /// `texture_shapes` at `time` seconds (egui's own clock when `None`).
    fn texture_shapes_at(
        ctx: &egui::Context,
        world: &mut World,
        (w, h, zoom): (f32, f32, f32),
        events: Vec<egui::Event>,
        time: Option<f64>,
    ) -> (Vec<egui::epaint::ClippedShape>, Vec<Out>, Option<egui::Id>) {
        ctx.set_zoom_factor(zoom);
        let input = egui::RawInput {
            time,
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(w / zoom, h / zoom),
            )),
            events,
            ..Default::default()
        };
        let (mut out, mut hv) = (Vec::new(), None);
        let mut full = ctx.run_ui(input, |ui| {
            panel_style(ui.style_mut());
            panel_layout(world, ui, &mut out, &mut hv)
        });
        full.textures_delta.clear();
        (full.shapes, out, hv)
    }

    /// A shape's rect from the context's points to the texture's pixels at `zoom`.
    fn to_px(r: egui::Rect, zoom: f32) -> egui::Rect {
        egui::Rect::from_min_max(
            (r.min.to_vec2() * zoom).to_pos2(),
            (r.max.to_vec2() * zoom).to_pos2(),
        )
    }

    /// `texture_shapes`'s frame, with the texts drawn, each with its galley's rect in the
    /// texture's pixels, in place of the shapes.
    fn texture_frame(
        ctx: &egui::Context,
        world: &mut World,
        tex: (f32, f32, f32),
        events: Vec<egui::Event>,
    ) -> (Vec<(String, egui::Rect)>, Vec<Out>, Option<egui::Id>) {
        let zoom = tex.2;
        let (shapes, out, hv) = texture_shapes(ctx, world, tex, events);
        let mut texts = Vec::new();
        fn walk(s: &egui::Shape, zoom: f32, texts: &mut Vec<(String, egui::Rect)>) {
            match s {
                egui::Shape::Text(t) => {
                    let r = egui::Rect::from_min_size(t.pos, t.galley.size());
                    texts.push((t.galley.text().to_string(), to_px(r, zoom)));
                }
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, zoom, texts)),
                _ => {}
            }
        }
        shapes.iter().for_each(|c| walk(&c.shape, zoom, &mut texts));
        (texts, out, hv)
    }

    /// Q-306's answer and Q-328 (#542): the title bar, the top 1/16 of the texture, is
    /// drawn above the tab row and a sub-page's Back button at every `ui_scale` and size:
    /// no text lies in it, and a pointer anywhere in it, pressed and released (a trigger's
    /// select, which `xr_input` passes on there), hovers and clicks nothing.
    #[test]
    fn the_title_bar_holds_no_widget_and_no_text_above_the_tab_row_and_back() {
        for page in [None, Some(Page::Cameras)] {
            for (w, h, scale) in TEXTURES {
                let tex = (w, h, panel_zoom(scale));
                let bar = h / 16.0;
                let mut world = panel_world(Tab::Scene);
                world.resource_mut::<UiState>().page = page;
                let ctx = egui::Context::default();
                texture_frame(&ctx, &mut world, tex, Vec::new());
                let (texts, _, _) = texture_frame(&ctx, &mut world, tex, Vec::new());
                let first = if page.is_some() {
                    "◀ Back"
                } else {
                    "Library"
                };
                assert!(
                    texts.iter().any(|(t, _)| t == first),
                    "{first} is drawn at {tex:?}: {texts:?}"
                );
                for (t, r) in &texts {
                    assert!(
                        r.min.y >= bar,
                        "{t:?} at {r:?} below the {bar} px bar, {tex:?}"
                    );
                }
                for y in [1.0, bar / 2.0, bar - 0.5] {
                    for x in (0..16).map(|i| (i as f32 + 0.5) * w / 16.0) {
                        let at = super::super::panel_point(Vec2::new(x, y), tex.2);
                        for select in [vec![true], vec![false]] {
                            let events = pointer_events(Some(at), Some(at), &select, 0.0);
                            let (_, out, hv) = texture_frame(&ctx, &mut world, tex, events);
                            assert_eq!(hv, None, "a widget hovered at ({x}, {y}), {tex:?}");
                            assert!(
                                !out.iter().any(|o| matches!(o, Out::Cue(UiCue::Click))),
                                "a click at ({x}, {y}), {tex:?}"
                            );
                        }
                    }
                }
                let s = world.resource::<UiState>();
                assert_eq!((s.tab, s.page), (Tab::Scene, page), "{tex:?}");
            }
        }
    }

    /// Q-328 (#544, #547): the title bar shows a grip mark, a filled shape painted in it,
    /// centred across the texture, whose fill stands out from the bar's own, at every
    /// `ui_scale` and size, on a tab and on a sub-page.
    #[test]
    fn the_title_bar_paints_a_centred_grip_that_stands_out_from_its_fill() {
        let mut style = egui::Style::default();
        panel_style(&mut style);
        let bar_fill = style.visuals.faint_bg_color;
        for page in [None, Some(Page::Cameras)] {
            for (w, h, scale) in TEXTURES {
                let tex = (w, h, panel_zoom(scale));
                let bar = h / 16.0;
                let mut world = panel_world(Tab::Scene);
                world.resource_mut::<UiState>().page = page;
                let ctx = egui::Context::default();
                texture_shapes(&ctx, &mut world, tex, Vec::new());
                let (shapes, _, _) = texture_shapes(&ctx, &mut world, tex, Vec::new());
                let mut rects = Vec::new();
                fn walk(s: &egui::Shape, zoom: f32, rects: &mut Vec<(egui::Rect, egui::Color32)>) {
                    match s {
                        egui::Shape::Rect(r) => rects.push((to_px(r.rect, zoom), r.fill)),
                        egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, zoom, rects)),
                        _ => {}
                    }
                }
                shapes
                    .iter()
                    .for_each(|c| walk(&c.shape, tex.2, &mut rects));
                // A mark within the bar, narrower than the bar (so not the bar's own fill).
                let grip = rects.iter().find(|(r, f)| {
                    *f != bar_fill
                        && *f != egui::Color32::TRANSPARENT
                        && r.min.y >= 0.0
                        && r.max.y <= bar
                        && r.height() > 0.0
                        && r.width() > 0.0
                        && r.width() < w / 2.0
                        && (r.center().x - w / 2.0).abs() <= 1.0
                });
                assert!(
                    grip.is_some(),
                    "no centred grip in the {bar} px bar at {:?}: {rects:?}",
                    (w, h, scale, page)
                );
            }
        }
    }

    /// §1.8 Hands ("a pinch on the panel clicks", #542): below the title bar, a press and
    /// release at a tab label's pixel (a pinch's select, which `xr_input` passes on off
    /// the bar, Q-315) opens that tab at every `ui_scale` and size, and one at Back
    /// leaves the sub-page.
    #[test]
    fn a_press_at_a_tab_or_back_below_the_title_bar_clicks_it() {
        for (w, h, scale) in TEXTURES {
            let tex = (w, h, panel_zoom(scale));
            let mut world = panel_world(Tab::Scene);
            let ctx = egui::Context::default();
            let press = |world: &mut World, label: &str| {
                texture_frame(&ctx, world, tex, Vec::new());
                let (texts, _, _) = texture_frame(&ctx, world, tex, Vec::new());
                let c = texts
                    .iter()
                    .find(|(t, _)| t == label)
                    .map(|(_, r)| r.center())
                    .unwrap_or_else(|| panic!("{label} is drawn at {tex:?}: {texts:?}"));
                assert!(c.y > h / 16.0, "{label} below the bar at {tex:?}");
                let at = super::super::panel_point(Vec2::new(c.x, c.y), tex.2);
                for select in [vec![], vec![true], vec![false]] {
                    let events = pointer_events(Some(at), Some(at), &select, 0.0);
                    texture_frame(&ctx, world, tex, events);
                }
            };
            press(&mut world, "Playback");
            assert_eq!(world.resource::<UiState>().tab, Tab::Playback, "{tex:?}");
            world.resource_mut::<UiState>().page = Some(Page::Cameras);
            press(&mut world, "◀ Back");
            assert_eq!(world.resource::<UiState>().page, None, "{tex:?}");
        }
    }

    /// What the touch tests' probe section saw: the items clicked and the slider's value.
    #[derive(Resource, Default)]
    struct TouchProbe {
        clicks: Vec<usize>,
        value: f32,
    }

    /// A section under the main panel's tab: a slider, a 120-point list that scrolls by
    /// dragging its content (which egui does only on a touch screen by default) of 20
    /// "Drag item" buttons, then 40 "Item" buttons, so the panel's own list scrolls; each
    /// records itself in `TouchProbe` (an "Item" by its index, a "Drag item" by 100 more).
    fn touch_probe_section(world: &mut World, ui: &mut egui::Ui) {
        let mut probe = world.resource_mut::<TouchProbe>();
        ui.add(egui::Slider::new(&mut probe.value, 0.0..=1.0));
        egui::ScrollArea::vertical()
            .id_salt("touch-probe-drag")
            .max_height(120.0)
            .scroll_source(egui::scroll_area::ScrollSource::ALL)
            .show(ui, |ui| {
                for i in 0..20 {
                    if ui.button(format!("Drag item {i}")).clicked() {
                        probe.clicks.push(100 + i);
                    }
                }
            });
        for i in 0..40 {
            if ui.button(format!("Item {i}")).clicked() {
                probe.clicks.push(i);
            }
        }
    }

    /// The main panel on a 1024 × 768 px texture at the panel zoom as the panel pointer
    /// drives it (Q-333): from two frames that lay it out, each frame of `frames` (a hit
    /// in pixels from the centre of the text `label`, `select`'s changes there, the time)
    /// rewritten for the touch press; what the user did.
    fn touch_panel(world: &mut World, label: &str, frames: &[(Vec2, Vec<bool>, f64)]) -> Vec<Out> {
        let tex = (1024.0, 768.0, PANEL_ZOOM);
        let ctx = egui::Context::default();
        // Labels as wide as their text, so a slide can stay inside one.
        assert!(super::super::load_system_font(&ctx), "a system font");
        texture_frame(&ctx, world, tex, Vec::new());
        let (texts, _, _) = texture_frame(&ctx, world, tex, Vec::new());
        let c = texts
            .iter()
            .find(|(t, _)| t == label)
            .map(|(_, r)| r.center())
            .unwrap_or_else(|| panic!("{label} is drawn: {texts:?}"));
        let (mut last, mut outs) = (None, Vec::new());
        for (d, select, t) in frames {
            let at = super::super::panel_point(Vec2::new(c.x, c.y) + *d, tex.2);
            let events = super::super::touch::touch_events(
                &ctx,
                pointer_events(last, Some(at), select, 0.0),
            );
            last = Some(at);
            outs.extend(texture_shapes_at(&ctx, world, tex, events, Some(*t)).1);
        }
        outs
    }

    /// A press, then the moves (in pixels), then the release where the last move left
    /// it, 0.1 s apart from t = 1 s.
    fn touch_frames(moves: &[Vec2]) -> Vec<(Vec2, Vec<bool>, f64)> {
        let mut frames = vec![(Vec2::ZERO, vec![], 1.0), (Vec2::ZERO, vec![true], 1.1)];
        for (i, &d) in moves.iter().enumerate() {
            frames.push((d, vec![], 1.2 + 0.1 * i as f64));
        }
        let end = moves.last().copied().unwrap_or(Vec2::ZERO);
        frames.push((end, vec![false], 1.2 + 0.1 * moves.len() as f64));
        frames.push((end, vec![], 1.3 + 0.1 * moves.len() as f64));
        frames
    }

    /// Q-333 (#561, `cpu`): a press on the main panel's "Playback" tab whose hit slides
    /// 20 px (past egui's own tolerance of 6 points, 10.5 px at the panel zoom, inside the
    /// tab) before the release switches to that tab.
    #[test]
    fn touch_events_a_tab_press_slid_inside_the_tab_clicks_it() {
        let mut world = panel_world(Tab::Scene);
        touch_panel(
            &mut world,
            "Playback",
            &touch_frames(&[Vec2::new(10.0, 0.0), Vec2::new(20.0, 0.0)]),
        );
        assert_eq!(world.resource::<UiState>().tab, Tab::Playback);
    }

    /// Q-333 (#561, `cpu`): a button in the main panel's list, which scrolls by the stick
    /// and its scroll bar but not by dragging its content (egui's default off a touch
    /// screen), clicks on a release inside: pressed and released in place, held 2 s, or
    /// slid 60 px up onto another button and back. A button in a list that scrolls by
    /// dragging its content doesn't click when the press drags the list 60 px and comes
    /// back, as a scroll takes the touch from the button under it; and a slider still
    /// drags (a press moved 200 px right raises its value).
    #[test]
    fn touch_events_list_buttons_click_inside_unless_a_drag_scrolls_and_sliders_drag() {
        let world_with_probe = || {
            let mut world = panel_world(Tab::Help);
            world.init_resource::<TouchProbe>();
            world
                .resource_mut::<super::super::DockHooks>()
                .add(Tab::Help, touch_probe_section);
            world
        };
        let up = Vec2::new(0.0, -60.0);
        let there_and_back = touch_frames(&[up * 0.5, up, up * 0.5, Vec2::ZERO]);
        let mut held = touch_frames(&[]);
        for frame in &mut held[2..] {
            frame.2 += 2.0;
        }
        let cases = [
            ("in place", "Item 3", touch_frames(&[]), vec![3]),
            ("held 2 s", "Item 3", held, vec![3]),
            ("out and back", "Item 3", there_and_back.clone(), vec![3]),
            ("dragged list", "Drag item 1", there_and_back, vec![]),
        ];
        for (name, label, frames, clicks) in cases {
            let mut world = world_with_probe();
            touch_panel(&mut world, label, &frames);
            assert_eq!(world.resource::<TouchProbe>().clicks, clicks, "{name}");
        }
        let mut world = world_with_probe();
        // The slider's rail, left of its value's text.
        let rail = Vec2::new(-120.0, 0.0);
        let frames: Vec<_> = touch_frames(&[Vec2::new(100.0, 0.0), Vec2::new(200.0, 0.0)])
            .into_iter()
            .map(|(d, select, t)| (d + rail, select, t))
            .collect();
        touch_panel(&mut world, "0.000", &frames);
        assert!(
            world.resource::<TouchProbe>().value > 0.0,
            "the slider drags"
        );
    }
}
