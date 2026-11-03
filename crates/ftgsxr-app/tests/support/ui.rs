// SPDX-License-Identifier: Apache-2.0

//! The toasts' drain (#286): every `Toast` the app raised since `record_toasts`.

use bevy::app::{App, Last};
use bevy::ecs::prelude::*;
use ftgsxr_app::Toast;

/// The toasts raised so far.
#[derive(Resource, Default)]
pub struct ToastLog(pub Vec<Toast>);

fn record(mut toasts: MessageReader<Toast>, mut log: ResMut<ToastLog>) {
    log.0.extend(toasts.read().cloned());
}

/// Starts recording the app's toasts into `ToastLog`.
pub fn record_toasts(app: &mut App) {
    app.init_resource::<ToastLog>().add_systems(Last, record);
}

/// The toasts recorded so far.
pub fn toasts(app: &App) -> Vec<Toast> {
    app.world().resource::<ToastLog>().0.clone()
}

/// Whether a toast starting with `prefix` was raised.
pub fn toasted(app: &App, prefix: &str) -> bool {
    toasts(app).iter().any(|t| t.text.starts_with(prefix))
}
