// SPDX-License-Identifier: Apache-2.0

//! The passthrough layer, style and mode switching (§1.9, D-14). M-3 (#223) brings the
//! declarations I-15 reads; the layer, the style's color map and the switch are M-6's.

use bevy::prelude::{Res, ResMut, Resource};

use crate::tracking::DevicesDrawn;

/// The environment mode (§1.9): VR, the projection layer alone and opaque, or passthrough.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EnvironmentMode {
    /// The projection layer alone, opaque, over §1.4's background.
    #[default]
    Vr,
    /// A passthrough layer under the projection layer (D-14).
    Passthrough,
}

/// The passthrough's appearance (§1.9), each control in \[−1, 1\].
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PassthroughStyle {
    /// Brightness, in \[−1, 1\].
    pub brightness: f32,
    /// Contrast, in \[−1, 1\].
    pub contrast: f32,
    /// Saturation, in \[−1, 1\].
    pub saturation: f32,
}

/// The environment's state (§1.9, §2.1): the mode, whether the runtime offers
/// passthrough (XR\_FB\_passthrough enabled on the instance; false on the desktop), and
/// the style.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Environment {
    /// The current mode; `Vr` in M-3.
    pub mode: EnvironmentMode,
    /// Whether XR\_FB\_passthrough was enabled on the instance.
    pub passthrough_available: bool,
    /// The passthrough style.
    pub style: PassthroughStyle,
}

/// The mode check of `tracking`'s hand and controller meshes (§1.8 Controllers and
/// Hands: drawn in VR mode, not in passthrough mode): `DevicesDrawn` follows
/// `Environment.mode`, false without an `Environment`.
pub(crate) fn devices_drawn(env: Option<Res<Environment>>, mut drawn: ResMut<DevicesDrawn>) {
    let want = DevicesDrawn(env.is_some_and(|e| e.mode == EnvironmentMode::Vr));
    if *drawn != want {
        *drawn = want;
    }
}

#[cfg(test)]
mod tests {
    use bevy::app::{App, Update};

    use super::*;

    /// §1.8: the meshes are drawn in VR mode only, and not without an `Environment`.
    #[test]
    fn devices_drawn_follows_the_environment_mode() {
        let mut app = App::new();
        app.init_resource::<DevicesDrawn>()
            .add_systems(Update, devices_drawn);
        app.update();
        assert_eq!(*app.world().resource::<DevicesDrawn>(), DevicesDrawn(false));
        app.insert_resource(Environment {
            mode: EnvironmentMode::Vr,
            passthrough_available: false,
            style: PassthroughStyle::default(),
        });
        app.update();
        assert_eq!(*app.world().resource::<DevicesDrawn>(), DevicesDrawn(true));
        app.world_mut().resource_mut::<Environment>().mode = EnvironmentMode::Passthrough;
        app.update();
        assert_eq!(*app.world().resource::<DevicesDrawn>(), DevicesDrawn(false));
    }
}
