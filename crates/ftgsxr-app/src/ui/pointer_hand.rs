// SPDX-License-Identifier: Apache-2.0

//! The panels' pointer hand (§1.8 User interface, I-11, Q-334): egui (0.36) tracks one
//! pointer and one press per context, so each panel context follows one hand at a time,
//! and only that hand's hit, presses and stick scroll reach it.
//!
//! - While the pointer hand holds `select` pressed on the panel, from its press to its
//!   release, it keeps the pointer: the other hand's moves, presses and scroll are
//!   dropped. Its ray leaving the panel ends the press (Q-333): the touch press cancels
//!   an armed one with egui's release, and any other press (a slider's drag) gets the
//!   release at its last hit; the context sees the pointer leave, and the other hand's
//!   ray, if on the panel, takes the pointer the next frame (that frame's hit, press
//!   included, is dropped, so a slider's drag doesn't end at the other's hit).
//! - Otherwise a hand's press on the panel makes it the pointer hand: the context gets
//!   that hand's move to its hit, then its press, so the press lands on the widget under
//!   its own ray, and takes the scroll from that frame on. When both hands press in the
//!   same frame, the pointer hand keeps it.
//! - With no press held, the pointer hand stays the one that pressed last while its ray
//!   is on the panel; when its ray leaves, the other hand's ray on the panel becomes the
//!   pointer, with a move to its hit and no `PointerGone`. The first hand to reach a
//!   panel nobody points at is its pointer hand.
//!
//! A release reaches the context only for a press it got: a hand that becomes the pointer
//! hand while holding a press the context never saw (dropped while the other hand held
//! its own) has that release dropped.

use bevy::math::Vec2;
use bevy_egui::egui;

use super::pointer_events;
use crate::intent::Hand;

/// One hand's ray on a panel this frame, in its context's points.
#[derive(Clone, Debug)]
pub(super) struct HandHit {
    /// The hand.
    pub(super) hand: Hand,
    /// The hit, in the context's points.
    pub(super) at: Vec2,
    /// The select input's presses and releases this frame, in order.
    pub(super) select: Vec<bool>,
    /// The stick's scroll, in the context's points.
    pub(super) scroll: f32,
}

/// A panel context's pointer between frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ContextPointer {
    /// The pointer hand.
    pub(super) hand: Hand,
    /// Its last hit, in the context's points.
    pub(super) at: Vec2,
    /// Whether the context holds a press of that hand: given and not yet released.
    pub(super) held: bool,
}

/// One frame's pointer events for a context, from its pointer before the frame (`prev`)
/// and the frame's hits on its panel in arrival order, and its pointer after the frame;
/// `armed` says whether the context's touch press is armed, so that the press ends with
/// egui's release when the ray leaves (Q-333) without a second one here.
// Q-334
pub(super) fn hand_events(
    prev: Option<ContextPointer>,
    hits: &[HandHit],
    armed: bool,
) -> (Vec<egui::Event>, Option<ContextPointer>) {
    let mut out = Vec::new();
    let mut cur = None;
    if let Some(p) = prev {
        match hits.iter().find(|h| h.hand == p.hand) {
            Some(h) => {
                // A press of the other hand that will take the pointer this frame takes
                // the scroll too: the pointer hand's is dropped.
                let taken = !h.select.last().copied().unwrap_or(p.held)
                    && hits
                        .iter()
                        .any(|o| o.hand != p.hand && o.select.contains(&true));
                let scroll = if taken { 0.0 } else { h.scroll };
                let h = HandHit {
                    scroll,
                    ..h.clone()
                };
                cur = Some(forward(&mut out, Some(p.at), &h, p.held));
            }
            None if p.held => {
                // The ray left during the press: the press ends, released here unless
                // the touch press releases it.
                if !armed {
                    out.push(egui::Event::PointerButton {
                        pos: egui::pos2(p.at.x, p.at.y),
                        button: egui::PointerButton::Primary,
                        pressed: false,
                        modifiers: egui::Modifiers::NONE,
                    });
                }
                out.push(egui::Event::PointerGone);
                return (out, None);
            }
            None => {}
        }
    }
    if !cur.is_some_and(|c: ContextPointer| c.held) {
        let pressing = hits
            .iter()
            .find(|h| Some(h.hand) != cur.map(|c| c.hand) && h.select.contains(&true));
        let next = pressing.or_else(|| cur.is_none().then(|| hits.first()).flatten());
        if let Some(h) = next {
            let last = cur.or(prev).map(|c| c.at);
            cur = Some(forward(&mut out, last, h, false));
        }
    }
    if cur.is_none() && prev.is_some() {
        out.push(egui::Event::PointerGone);
    }
    (out, cur)
}

/// Appends `hit`'s events, from the context's last position `last`, to `out`; the
/// context's pointer after them, from whether it held the hand's press before (`held`).
/// Releases before the first press are dropped when the context holds no press.
fn forward(
    out: &mut Vec<egui::Event>,
    last: Option<Vec2>,
    hit: &HandHit,
    held: bool,
) -> ContextPointer {
    let skip = if held {
        0
    } else {
        hit.select.iter().take_while(|p| !**p).count()
    };
    let select = &hit.select[skip..];
    out.extend(pointer_events(last, Some(hit.at), select, hit.scroll));
    ContextPointer {
        hand: hit.hand,
        at: hit.at,
        held: select.last().copied().unwrap_or(held),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: Vec2 = Vec2::new(100.0, 50.0);
    const R: Vec2 = Vec2::new(400.0, 300.0);

    fn hit(hand: Hand, at: Vec2, select: &[bool], scroll: f32) -> HandHit {
        HandHit {
            hand,
            at,
            select: select.to_vec(),
            scroll,
        }
    }

    fn moved(at: Vec2) -> egui::Event {
        egui::Event::PointerMoved(egui::pos2(at.x, at.y))
    }

    fn button(at: Vec2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos: egui::pos2(at.x, at.y),
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        }
    }

    /// Runs `frames` from no pointer, `armed` false; each frame's events.
    fn run(frames: &[Vec<HandHit>]) -> Vec<Vec<egui::Event>> {
        let mut prev = None;
        frames
            .iter()
            .map(|hits| {
                let (events, next) = hand_events(prev, hits, false);
                prev = next;
                events
            })
            .collect()
    }

    /// Q-334 (#563): with both rays on the panel, a hand's press moves the pointer to its
    /// own hit first, whichever hand comes last in the frame.
    #[test]
    fn hand_events_a_press_moves_the_pointer_to_the_pressing_hand_first() {
        for (a, b) in [(Hand::Left, Hand::Right), (Hand::Right, Hand::Left)] {
            let ev = run(&[
                vec![hit(a, L, &[], 0.0), hit(b, R, &[], 0.0)],
                vec![hit(a, L, &[true], 0.0), hit(b, R, &[], 0.0)],
                vec![hit(b, R, &[], 0.0), hit(a, L, &[false], 0.0)],
            ]);
            assert_eq!(ev[0], vec![moved(L)], "the first hand points");
            assert_eq!(ev[1], vec![moved(L), button(L, true)]);
            assert_eq!(ev[2], vec![moved(L), button(L, false)]);
            let ev = run(&[
                vec![hit(a, L, &[], 0.0), hit(b, R, &[], 0.0)],
                vec![hit(a, L, &[], 0.0), hit(b, R, &[true], 0.0)],
            ]);
            assert_eq!(ev[1], vec![moved(L), moved(R), button(R, true)]);
        }
    }

    /// Q-334 (#563): while the left hand holds `select`, the right hand's moves, presses
    /// and scroll don't reach the context; the left's release does, at its own hit.
    #[test]
    fn hand_events_the_holding_hand_keeps_the_pointer() {
        let ev = run(&[
            vec![
                hit(Hand::Left, L, &[true], 0.0),
                hit(Hand::Right, R, &[], 0.0),
            ],
            vec![
                hit(Hand::Left, L, &[], 0.0),
                hit(Hand::Right, R + Vec2::X, &[true], 4.0),
            ],
            vec![
                hit(Hand::Right, R, &[false], 0.0),
                hit(Hand::Left, L, &[false], 0.0),
            ],
        ]);
        assert_eq!(ev[0], vec![moved(L), button(L, true)]);
        assert_eq!(ev[1], vec![moved(L)]);
        assert_eq!(ev[2], vec![moved(L), button(L, false)]);
    }

    /// Q-334 (#563): with no press held, the pointer hand's ray leaving hands the pointer
    /// to the other's ray with a move and no `PointerGone`; the pointer leaves once no
    /// ray is on the panel.
    #[test]
    fn hand_events_a_leaving_pointer_hand_hands_over_without_pointer_gone() {
        let ev = run(&[
            vec![
                hit(Hand::Left, L, &[true, false], 0.0),
                hit(Hand::Right, R, &[], 0.0),
            ],
            vec![hit(Hand::Right, R, &[], 0.0)],
            vec![hit(Hand::Right, R + Vec2::Y, &[], 0.0)],
            vec![],
            vec![],
        ]);
        assert_eq!(ev[1], vec![moved(R)]);
        assert_eq!(ev[2], vec![moved(R + Vec2::Y)]);
        assert_eq!(ev[3], vec![egui::Event::PointerGone]);
        assert!(ev[4].is_empty());
    }

    /// Q-334 (#563): the stick scroll comes from the pointer hand only.
    #[test]
    fn hand_events_only_the_pointer_hands_scroll_reaches_the_context() {
        let ev = run(&[
            vec![
                hit(Hand::Right, R, &[true, false], 0.0),
                hit(Hand::Left, L, &[], 0.0),
            ],
            vec![hit(Hand::Left, L, &[], 9.0), hit(Hand::Right, R, &[], 0.0)],
            vec![hit(Hand::Left, L, &[], 0.0), hit(Hand::Right, R, &[], 3.0)],
        ]);
        assert_eq!(ev[1], vec![moved(R)]);
        assert!(matches!(ev[2][1], egui::Event::MouseWheel { delta, .. } if delta.y == 3.0));
        // The frame the left's press takes the pointer, the right's scroll is dropped.
        let ev = run(&[
            vec![hit(Hand::Right, R, &[], 0.0)],
            vec![
                hit(Hand::Left, L, &[true], 0.0),
                hit(Hand::Right, R, &[], 3.0),
            ],
        ]);
        assert_eq!(ev[1], vec![moved(R), moved(L), button(L, true)]);
    }

    /// Q-334 (#563): the pointer hand's ray leaving during its press ends the press: a
    /// release at its last hit unless the touch press is armed (which releases it), then
    /// `PointerGone`; the other ray takes the pointer the next frame, and the release of
    /// a press the context never got is dropped.
    #[test]
    fn hand_events_a_press_ends_when_its_ray_leaves_and_unseen_releases_drop() {
        let start = ContextPointer {
            hand: Hand::Left,
            at: L,
            held: true,
        };
        let right = [hit(Hand::Right, R, &[true], 0.0)];
        let (ev, next) = hand_events(Some(start), &right, false);
        assert_eq!(ev, vec![button(L, false), egui::Event::PointerGone]);
        assert_eq!(next, None);
        let (ev, _) = hand_events(Some(start), &right, true);
        assert_eq!(ev, vec![egui::Event::PointerGone]);
        let (ev, next) = hand_events(None, &[hit(Hand::Right, R, &[false], 0.0)], false);
        assert_eq!(ev, vec![moved(R)]);
        assert!(!next.unwrap().held);
    }

    /// Q-334 (#563): a release of the pointer hand and the other hand's press in the
    /// same frame hand the pointer over within the frame; when both press, the pointer
    /// hand keeps it.
    #[test]
    fn hand_events_a_release_and_the_other_hands_press_hand_over_in_one_frame() {
        let held = ContextPointer {
            hand: Hand::Left,
            at: L,
            held: true,
        };
        let (ev, next) = hand_events(
            Some(held),
            &[
                hit(Hand::Right, R, &[true], 0.0),
                hit(Hand::Left, L, &[false], 0.0),
            ],
            false,
        );
        assert_eq!(
            ev,
            vec![moved(L), button(L, false), moved(R), button(R, true)]
        );
        assert_eq!(next.map(|n| (n.hand, n.held)), Some((Hand::Right, true)));
        let free = ContextPointer {
            held: false,
            ..held
        };
        let (ev, next) = hand_events(
            Some(free),
            &[
                hit(Hand::Right, R, &[true], 0.0),
                hit(Hand::Left, L, &[true], 0.0),
            ],
            false,
        );
        assert_eq!(ev, vec![moved(L), button(L, true)]);
        assert_eq!(next.map(|n| n.hand), Some(Hand::Left));
    }
}
