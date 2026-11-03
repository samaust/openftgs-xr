// SPDX-License-Identifier: Apache-2.0

//! The overlay toggles (`cpu`, #275; source of truth Appendix D.1 "Toggle boxes,
//! velocity vectors: B, V", §1.4 overlays, §1.7 Panels' Display tab): an
//! `Intent::Overlay` flips the `[display.overlays]` setting and every instance's
//! `SplatOverlays` follows; a second one flips it back (Q-207).

mod support;

use bevy::prelude::*;
use ftgsxr_app::{Intent, LibraryCommand, LoadedModel, ModelRef, OverlayToggle, SettingsRes};
use ftgsxr_render::SplatOverlays;
use ftgsxr_testkit::fixtures_dir;
use support::TestApp;

fn overlays(t: &mut TestApp) -> Vec<ftgsxr_render::Overlays> {
    let w = t.app.world_mut();
    w.query_filtered::<&SplatOverlays, With<LoadedModel>>()
        .iter(w)
        .map(|o| o.0)
        .collect()
}

#[test]
fn b_v_and_the_time_tint_toggle_every_instances_overlays() {
    let mut t = TestApp::new("overlay_toggles");
    t.command(LibraryCommand::LoadRef(ModelRef::File(
        fixtures_dir().join("probes/P1.safetensors"),
    )));
    assert!(
        t.until(600, |app| app
            .world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1),
        "P1 loads"
    );
    t.frames(2);
    let before = overlays(&mut t);
    assert_eq!(before.len(), 1);
    assert!(!before[0].boxes && before[0].velocities.is_none() && !before[0].time_tint);
    for o in [
        OverlayToggle::Boxes,
        OverlayToggle::Velocities,
        OverlayToggle::TimeTint,
    ] {
        t.app.world_mut().write_message(Intent::Overlay(o));
    }
    t.frames(2);
    let on = overlays(&mut t)[0];
    assert!(on.boxes && on.velocities.is_some() && on.time_tint, "{on:?}");
    let s = &t.app.world().resource::<SettingsRes>().0.display.overlays;
    assert!(s.boxes && s.velocities && s.time_tint);
    t.app
        .world_mut()
        .write_message(Intent::Overlay(OverlayToggle::Boxes));
    t.frames(2);
    let off = overlays(&mut t)[0];
    assert!(!off.boxes && off.velocities.is_some(), "{off:?}");
}
