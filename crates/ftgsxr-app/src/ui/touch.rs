// SPDX-License-Identifier: Apache-2.0

//! The panels' touch press (§1.8 User interface, Q-333): on the XR panels a button
//! behaves as on a touch interface. A press is *armed* when egui's hit at the press is a
//! widget that senses clicks and not drags, with no widget that drags under the pointer:
//! the button looks pressed while the hit is inside it, a release inside clicks it however
//! far the hit moved or however long `select` was held, and a release outside cancels
//! the press, as does the hit leaving the quad or a stick scroll. The panels' lists
//! scroll by the stick and their scroll bars, not by dragging their content (egui's
//! default off a touch screen), so their buttons are armed too. A press with a widget
//! that drags under it (a slider, a scroll bar, a list that scrolls by dragging its
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
//! up gives egui a new press there; a cancel, or a release outside whose egui click
//! would still come, is preceded by a move past the tolerance. `touch_pass` gives the
//! idle look to the pressed widget while the hit is outside it. The armed press lives in
//! the context's temporary memory.

use bevy_egui::egui;

/// The armed press of a panel's context, from its press to its release or cancel.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Armed {
    /// The rect of the widget pressed, in the context's points.
    rect: egui::Rect,
    /// Where the press was, on the widget egui hit there (inside `rect` or within egui's
    /// `interact_radius` of it): egui sees the pointer there while the hit is inside
    /// `rect`.
    origin: egui::Pos2,
    /// The hit's last position, as the frame's events gave it.
    hit: egui::Pos2,
    /// Whether the hit is inside `rect`.
    inside: bool,
    /// Whether egui's own click is lost: the hit moved past `max_click_dist` from the
    /// press since egui's last press.
    lost: bool,
}

/// What a context's touch press is between frames.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Press {
    /// An armed press.
    Armed(Armed),
    /// A press cancelled before its release (a stick scroll): its release clicks nothing,
    /// and no press is armed until the next one.
    Cancelled,
}

/// The id of a context's touch press in its temporary memory.
fn press_id() -> egui::Id {
    egui::Id::new("ftgsxr-touch-press")
}

/// The panels' pass setup for the touch press, before `ui` draws (Q-333): no time limit
/// on a click (egui's `max_click_duration`, 0.8 s by default), so a long press still
/// clicks, holding from the context's next pass; and while an armed press's hit is
/// outside its widget with egui's click still held, the pressed look (which egui gives
/// that widget then) is the idle one. Inside, egui's own pressed look holds.
// Q-333
pub(crate) fn touch_pass(ui: &mut egui::Ui) {
    let ctx = ui.ctx().clone();
    if ctx.options(|o| o.input_options.max_click_duration) != f64::INFINITY {
        ctx.options_mut(|o| o.input_options.max_click_duration = f64::INFINITY);
    }
    if let Some(Press::Armed(a)) = stored(&ctx)
        && !a.inside
        && !a.lost
    {
        let widgets = &mut ui.visuals_mut().widgets;
        widgets.active = widgets.inactive;
    }
}

/// Whether the context's touch press is armed before this frame's events, so that its
/// ray leaving the quad ends it with egui's release (Q-334).
// Q-333, Q-334
pub(crate) fn armed(ctx: &egui::Context) -> bool {
    matches!(start(ctx), Some(Press::Armed(_)))
}

/// The context's stored touch press.
fn stored(ctx: &egui::Context) -> Option<Press> {
    ctx.data(|d| d.get_temp::<Option<Press>>(press_id()))
        .flatten()
}

/// The frame's pointer events for a panel's context, rewritten for the touch press from
/// the context's last pass (see the module's documentation).
// Q-333
pub(crate) fn touch_events(ctx: &egui::Context, events: Vec<egui::Event>) -> Vec<egui::Event> {
    let mut press = start(ctx);
    let max_dist = ctx.options(|o| o.input_options.max_click_dist);
    // A point past egui's click tolerance from the press: a move there ends its click.
    let past_tolerance = |a: &Armed| a.origin + egui::vec2(2.0 * max_dist + 1.0, 0.0);
    let primary = |pos, pressed, modifiers| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers,
    };
    // Ends an armed press's click in egui, the pointer back at the hit.
    let cancel = |a: &Armed, out: &mut Vec<egui::Event>| {
        if !a.lost {
            out.push(egui::Event::PointerMoved(past_tolerance(a)));
        }
        out.push(egui::Event::PointerMoved(a.hit));
    };
    let mut out = Vec::with_capacity(events.len() + 3);
    for event in events {
        match (event, &mut press) {
            (egui::Event::PointerMoved(p), Some(Press::Armed(a))) => {
                a.hit = p;
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
                    out.push(egui::Event::PointerMoved(p));
                }
            }
            (egui::Event::PointerGone, Some(Press::Armed(a))) => {
                // Off the quad the release never comes here: the press ends now.
                cancel(a, &mut out);
                out.push(primary(a.hit, false, egui::Modifiers::NONE));
                out.push(egui::Event::PointerGone);
                press = None;
            }
            (event @ egui::Event::MouseWheel { .. }, Some(Press::Armed(a))) => {
                // A stick scroll moves the list under the press: it cancels it.
                cancel(a, &mut out);
                out.push(event);
                press = Some(Press::Cancelled);
            }
            (
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers,
                },
                current,
            ) => match current.take() {
                Some(Press::Armed(a)) if !pressed => {
                    if a.rect.contains(pos) && !a.lost {
                        out.push(primary(a.origin, false, modifiers));
                    } else {
                        cancel(&Armed { hit: pos, ..a }, &mut out);
                        out.push(primary(pos, false, modifiers));
                    }
                }
                _ => out.push(primary(pos, pressed, modifiers)),
            },
            (other, _) => out.push(other),
        }
    }
    ctx.data_mut(|d| d.insert_temp(press_id(), press));
    out
}

/// The context's touch press before this frame's events: the one kept from earlier
/// frames while egui's button is down, or, after the pass that took a press, an armed
/// press when egui's hit there is a widget that senses clicks and not drags with no
/// widget that drags under it.
fn start(ctx: &egui::Context) -> Option<Press> {
    let (down, pressed, origin, latest) = ctx.input(|i| {
        (
            i.pointer.primary_down(),
            i.pointer.primary_pressed(),
            i.pointer.press_origin(),
            i.pointer.latest_pos(),
        )
    });
    if !down {
        return None;
    }
    if let Some(kept) = stored(ctx) {
        return Some(kept);
    }
    if !pressed {
        return None;
    }
    let (click, drag) = ctx.viewport(|v| (v.hits.click, v.hits.drag));
    let (w, origin) = (click?, origin?);
    let armable = w.sense.senses_click() && !w.sense.senses_drag() && drag.is_none();
    let hit = latest.unwrap_or(origin);
    armable.then_some(Press::Armed(Armed {
        rect: w.rect,
        origin,
        hit,
        inside: w.rect.contains(hit),
        lost: false,
    }))
}
