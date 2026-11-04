// SPDX-License-Identifier: Apache-2.0

//! I-7 (`cpu`, §1.7 Selection and manipulation): a synthetic cursor over an instance
//! selects it on a left click, through the mouse ray from the active camera (E10);
//! a click over nothing deselects.

use crate::support;

use bevy::prelude::*;
use ftgsxr_app::Selected;
use support::TestApp;
use support::desktop::{button, cursor, load_p1, windowed};

fn click(t: &mut TestApp, at: Vec2) {
    cursor(t, Some(at));
    t.frames(1);
    button(t, MouseButton::Left, true);
    t.frames(2);
    button(t, MouseButton::Left, false);
    t.frames(2);
}

fn selected(t: &mut TestApp) -> Vec<Entity> {
    let w = t.app.world_mut();
    w.query_filtered::<Entity, With<Selected>>()
        .iter(w)
        .collect()
}

#[test]
fn i07_a_click_over_an_instance_selects_it_and_over_nothing_deselects() {
    let mut t = windowed("i07_select");
    let e = load_p1(&mut t);
    assert!(selected(&mut t).is_empty());
    // The reset view looks at the origin, where P1 sits: the window's centre.
    click(&mut t, Vec2::new(640.0, 360.0));
    assert_eq!(selected(&mut t), vec![e]);
    // The top left corner sees the sky above the instance.
    click(&mut t, Vec2::new(4.0, 4.0));
    assert!(selected(&mut t).is_empty());
}
