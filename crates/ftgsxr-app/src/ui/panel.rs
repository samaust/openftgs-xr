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
                egui::CornerRadius::same((grip.height() / 2.0) as u8),
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

/// Draws the wrist menu's buttons over `ui`'s whole texture; the one clicked, if any.
pub(crate) fn wrist_ui(ui: &mut egui::Ui, playing: bool) -> Option<WristButton> {
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
        let ppp = PANEL_ZOOM;
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(640.0 / ppp, 200.0 / ppp),
            )),
            events,
            ..Default::default()
        };
        ctx.set_zoom_factor(PANEL_ZOOM);
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
        (texts, clicked)
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
    /// `zoom`, with `events` in the context's points: the texts drawn, each with its
    /// galley's rect in the texture's pixels, what the user did, and the widget hovered.
    fn texture_frame(
        ctx: &egui::Context,
        world: &mut World,
        (w, h, zoom): (f32, f32, f32),
        events: Vec<egui::Event>,
    ) -> (Vec<(String, egui::Rect)>, Vec<Out>, Option<egui::Id>) {
        ctx.set_zoom_factor(zoom);
        let input = egui::RawInput {
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
        let mut texts = Vec::new();
        fn walk(s: &egui::Shape, zoom: f32, texts: &mut Vec<(String, egui::Rect)>) {
            match s {
                egui::Shape::Text(t) => {
                    let r = egui::Rect::from_min_size(t.pos, t.galley.size());
                    let px = egui::Rect::from_min_max(
                        (r.min.to_vec2() * zoom).to_pos2(),
                        (r.max.to_vec2() * zoom).to_pos2(),
                    );
                    texts.push((t.galley.text().to_string(), px));
                }
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, zoom, texts)),
                _ => {}
            }
        }
        full.shapes
            .iter()
            .for_each(|c| walk(&c.shape, zoom, &mut texts));
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
        for page in [None, Some(Page::Cameras)] {
            for (w, h, scale) in TEXTURES {
                let zoom = panel_zoom(scale);
                let mut world = panel_world(Tab::Scene);
                world.resource_mut::<UiState>().page = page;
                let ctx = egui::Context::default();
                ctx.set_zoom_factor(zoom);
                let input = || egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(w / zoom, h / zoom),
                    )),
                    ..Default::default()
                };
                let (mut out, mut hv, mut fill) = (Vec::new(), None, None);
                let mut frame = |fill: &mut Option<egui::Color32>| {
                    ctx.run_ui(input(), |ui| {
                        panel_style(ui.style_mut());
                        *fill = Some(ui.visuals().faint_bg_color);
                        panel_layout(&mut world, ui, &mut out, &mut hv)
                    })
                };
                frame(&mut fill).textures_delta.clear();
                let mut full = frame(&mut fill);
                full.textures_delta.clear();
                let fill = fill.unwrap();
                let bar = h / 16.0;
                let mut rects = Vec::new();
                fn walk(s: &egui::Shape, zoom: f32, rects: &mut Vec<(egui::Rect, egui::Color32)>) {
                    match s {
                        egui::Shape::Rect(r) => rects.push((
                            egui::Rect::from_min_max(
                                (r.rect.min.to_vec2() * zoom).to_pos2(),
                                (r.rect.max.to_vec2() * zoom).to_pos2(),
                            ),
                            r.fill,
                        )),
                        egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, zoom, rects)),
                        _ => {}
                    }
                }
                full.shapes
                    .iter()
                    .for_each(|c| walk(&c.shape, zoom, &mut rects));
                let grip = rects.iter().find(|(r, f)| {
                    *f != fill
                        && *f != egui::Color32::TRANSPARENT
                        && r.min.y >= 0.0
                        && r.max.y <= bar
                        && r.width() > 0.0
                        && r.height() > 0.0
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
}
