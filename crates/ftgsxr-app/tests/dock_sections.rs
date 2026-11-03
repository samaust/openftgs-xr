// SPDX-License-Identifier: Apache-2.0

//! The dock's settings sections (§1.7 Panels, D-21, #286): an instance switched to
//! centers takes `[display]`'s radius and colormap (Q-T286.7).

mod support;

use ftgsxr_app::SettingsRes;
use ftgsxr_render::{Colormap, RenderMode, SplatMode};
use support::TestApp;

// Q-T286.7
#[test]
fn an_instance_switched_to_centers_takes_the_display_settings() {
    let mut t = TestApp::new("dock-centers");
    t.frames(2);
    {
        let mut s = t.app.world_mut().resource_mut::<SettingsRes>();
        s.0.display.centers_radius = 5.0;
        s.0.display.colormap = "speed".into();
    }
    t.frames(2);
    let e = t
        .app
        .world_mut()
        .spawn(SplatMode(RenderMode::Centers {
            pixel_radius: 3.0,
            colormap: Colormap::Dc,
        }))
        .id();
    t.frames(2);
    assert_eq!(
        t.app.world().get::<SplatMode>(e).unwrap().0,
        RenderMode::Centers {
            pixel_radius: 5.0,
            colormap: Colormap::Speed,
        }
    );
}
