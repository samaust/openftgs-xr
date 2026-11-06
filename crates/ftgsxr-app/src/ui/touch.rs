// SPDX-License-Identifier: Apache-2.0

//! The panels' touch press (§1.8 User interface, Q-333): on the XR panels a button
//! behaves as on a touch interface. A press is *armed* when egui's hit at the press is a
//! widget that senses clicks and not drags, with no widget that drags under the pointer:
//! the button looks pressed while the hit is inside it, a release inside clicks it however
//! far the hit moved or however long `select` was held, and a release outside cancels
//! the press. The panels' lists scroll by the stick and their scroll bars, not by
//! dragging their content (egui's default off a touch screen), so their buttons are armed
//! too. A press with a widget that drags under it (a list that scrolls by dragging its
//! content) isn't armed: it clicks as egui does, so a drag of the list takes the press
//! from the button under it.
//!
//! egui (0.36) clicks the widget a press began on when the release comes within
//! `max_click_dist` of the press, wherever the release is, and gives up the click once
//! the pointer moves farther or leaves; it shows the widget pressed while the button is
//! held, wherever the pointer is. So `touch_events` rewrites the frame's pointer events
//! before they reach the context, from the context's last pass: while the hit is inside
//! the pressed widget egui sees the pointer at the press, so its click and pressed look
//! hold however far the hit wanders inside; a hit back inside after egui gave the click
//! up gives egui a new press there; a release outside whose egui click would still come
//! is preceded by a move past the tolerance. `touch_look` gives the idle look to the
//! pressed widget while the hit is outside it. The armed press lives in the context's
//! temporary memory.

use bevy_egui::egui;

/// The armed press of a panel's context, from its press to its release.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Armed {
    /// The rect of the widget pressed, in the context's points.
    rect: egui::Rect,
    /// Where the press was, inside `rect`: egui sees the pointer there while the hit is
    /// inside `rect`.
    origin: egui::Pos2,
    /// Whether the hit is inside `rect`.
    inside: bool,
    /// Whether egui's own click is lost: the hit moved past `max_click_dist` from the
    /// press, or left the quad, since egui's last press.
    lost: bool,
}

/// The key of a context's armed press in its temporary memory.
fn key() -> egui::Id {
    egui::Id::new("ftgsxr-touch-press")
}

/// Sets a panel context's input options for the touch press: no time limit on a click
/// (egui's `max_click_duration`, 0.8 s by default), so a long press still clicks. It
/// holds from the context's next pass.
// Q-333
pub(crate) fn touch_options(ctx: &egui::Context) {
    if ctx.options(|o| o.input_options.max_click_duration) != f64::INFINITY {
        ctx.options_mut(|o| o.input_options.max_click_duration = f64::INFINITY);
    }
}

/// The frame's pointer events for a panel's context, rewritten for the touch press from
/// the context's last pass (see the module's documentation).
// Q-333
pub(crate) fn touch_events(ctx: &egui::Context, events: Vec<egui::Event>) -> Vec<egui::Event> {
    let mut armed = arm(ctx);
    let max_dist = ctx.options(|o| o.input_options.max_click_dist);
    let mut out = Vec::with_capacity(events.len() + 3);
    let primary = |pos, pressed, modifiers| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers,
    };
    for event in events {
        match event {
            egui::Event::PointerMoved(p) => match &mut armed {
                Some(a) => {
                    a.inside = a.rect.contains(p);
                    if a.inside {
                        out.push(egui::Event::PointerMoved(a.origin));
                        if a.lost {
                            // Back inside after egui gave the click up: a new press.
                            out.push(primary(a.origin, false, egui::Modifiers::NONE));
                            out.push(primary(a.origin, true, egui::Modifiers::NONE));
                            a.lost = false;
                        }
                    } else {
                        a.lost |= a.origin.distance(p) > max_dist;
                        out.push(event);
                    }
                }
                None => out.push(event),
            },
            egui::Event::PointerGone => {
                if let Some(a) = &mut armed {
                    a.inside = false;
                    a.lost = true;
                }
                out.push(event);
            }
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers,
            } => match armed.take() {
                Some(a) if !pressed => {
                    if a.rect.contains(pos) && !a.lost {
                        out.push(primary(a.origin, false, modifiers));
                    } else {
                        if !a.lost {
                            // egui would still click: a move past its tolerance first.
                            let far = a.origin + egui::vec2(2.0 * max_dist + 1.0, 0.0);
                            out.push(egui::Event::PointerMoved(far));
                        }
                        out.push(egui::Event::PointerMoved(pos));
                        out.push(primary(pos, false, modifiers));
                    }
                }
                _ => out.push(primary(pos, pressed, modifiers)),
            },
            other => out.push(other),
        }
    }
    ctx.data_mut(|d| d.insert_temp(key(), armed));
    out
}

/// The context's armed press before this frame's events: the one kept from earlier
/// frames, or, after a pass that took a press, the press egui's hit puts on a widget that
/// senses clicks and not drags with no widget that drags under it; none once the button
/// is up.
fn arm(ctx: &egui::Context) -> Option<Armed> {
    let kept: Option<Armed> = ctx.data(|d| d.get_temp(key())).flatten();
    let (down, origin, could_click, latest) = ctx.input(|i| {
        (
            i.pointer.primary_down(),
            i.pointer.press_origin(),
            i.pointer.could_any_button_be_click(),
            i.pointer.latest_pos(),
        )
    });
    if !down {
        return None;
    }
    let lost = !could_click || latest.is_none();
    if let Some(mut a) = kept {
        a.lost |= lost;
        return Some(a);
    }
    let (click, drag) = ctx.viewport(|v| (v.hits.click, v.hits.drag));
    let (w, origin) = (click?, origin?);
    let armable = w.sense.senses_click()
        && !w.sense.senses_drag()
        && drag.is_none()
        && w.rect.contains(origin);
    armable.then_some(Armed {
        rect: w.rect,
        origin,
        inside: latest.is_some_and(|p| w.rect.contains(p)),
        lost,
    })
}

/// The look of an armed press for the pass about to draw on `ui`: while the hit is
/// outside the pressed widget and egui's click still holds, the pressed look (which egui
/// gives that widget then) is the idle one. Inside, egui's own pressed look holds.
// Q-333
pub(crate) fn touch_look(ui: &mut egui::Ui) {
    let armed = ui
        .ctx()
        .data(|d| d.get_temp::<Option<Armed>>(key()))
        .flatten();
    if armed.is_some_and(|a| !a.inside && !a.lost) {
        let widgets = &mut ui.visuals_mut().widgets;
        widgets.active = widgets.inactive;
    }
}
