// SPDX-License-Identifier: Apache-2.0

//! An instance's boxes (§2.1 `SplatTrim`, `SplatBounds`): E10's B_m and B_c of its asset,
//! or, with a trim on, E17's trimmed boxes, computed from the uploaded arrays on the async
//! compute task pool whenever the trim changes (§2.4 Threads, D-56).

use ::bevy::asset::{AssetId, Assets};
use ::bevy::ecs::prelude::*;
use ::bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use ftgsxr_core::{Aabb, Trim, TrimMode, trimmed_bounds};

use super::{SplatAsset, SplatModel};

/// The instance's trim (E17), in model space; an instance without one is untrimmed.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SplatTrim(pub Trim);

/// The instance's B_m and B_c in model space (§1.5 Poses): E10's boxes of its asset, or
/// E17's trimmed boxes while a trim is on, the shape's own box for both with `kept_none`
/// set when the trim keeps no row. The plugin inserts it once the asset is loaded and
/// keeps the previous boxes until a trim's result arrives (E17).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SplatBounds {
    /// B_m: what picking, the selection box, orbit and the rest use (§1.5).
    pub aabb: Aabb,
    /// B_c: the fit box under `fit_to = "content"` (§1.5).
    pub content: Aabb,
    /// Whether the trim keeps no row (§1.7's "Trim keeps no primitives").
    pub kept_none: bool,
}

/// What an instance's `SplatBounds` were last asked for: the asset and the trim, with the
/// trimmed boxes' task while it runs. Replacing the task drops the old one, so a result
/// for a trim that has since changed is discarded (E17).
#[derive(Component)]
pub(crate) struct BoundsState {
    asset: AssetId<SplatAsset>,
    trim: Trim,
    pending: Option<Task<Option<(Aabb, Aabb)>>>,
}

/// The trim that decides the boxes: off whatever its shapes when its mode is off.
fn effective(trim: Option<&SplatTrim>) -> Trim {
    match trim {
        Some(t) if t.0.mode != TrimMode::Off => t.0,
        _ => Trim::default(),
    }
}

/// The untrimmed boxes of `asset`.
fn untrimmed(asset: &SplatAsset) -> SplatBounds {
    SplatBounds {
        aabb: asset.bounds,
        content: asset.content_bounds,
        kept_none: false,
    }
}

/// The boxes of a trimmed result: the trimmed boxes, or the shape's box for both.
fn from_result(result: Option<(Aabb, Aabb)>, trim: &Trim, asset: &SplatAsset) -> SplatBounds {
    match (result, trim.shape_bounds()) {
        (Some((aabb, content)), _) => SplatBounds {
            aabb,
            content,
            kept_none: false,
        },
        (None, Some(shape)) => SplatBounds {
            aabb: shape,
            content: shape,
            kept_none: true,
        },
        (None, None) => untrimmed(asset),
    }
}

type BoundsQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static SplatModel,
        Option<&'static SplatTrim>,
        Option<&'static mut BoundsState>,
        Option<&'static SplatBounds>,
    ),
>;

/// Keeps each instance's `SplatBounds` (E17): the untrimmed boxes at once for a new asset
/// or a trim turned off; a task for a trim that is on, whose result replaces the boxes when
/// it arrives.
// Q-373
pub(crate) fn update_bounds(
    mut commands: Commands,
    assets: Res<Assets<SplatAsset>>,
    mut instances: BoundsQuery,
) {
    for (entity, model, trim, state, bounds) in &mut instances {
        let id = model.0.id();
        let Some(asset) = assets.get(id) else {
            continue;
        };
        let trim = effective(trim);
        let mut state = match state {
            Some(s) if s.asset == id && s.trim == trim => s,
            stale => {
                let same_asset = stale.as_ref().is_some_and(|s| s.asset == id);
                let pending = (trim.mode != TrimMode::Off).then(|| {
                    let arrays = std::sync::Arc::clone(&asset.packed.arrays);
                    let alpha_skip = asset.packed.conventions.alpha_skip;
                    AsyncComputeTaskPool::get()
                        .spawn(async move { trimmed_bounds(&arrays, alpha_skip, &trim) })
                });
                // A trim turned off, or a new asset, gives the untrimmed boxes now; a
                // trim on keeps the previous boxes of the same asset until its result.
                if pending.is_none() || !same_asset || bounds.is_none() {
                    commands.entity(entity).insert(untrimmed(asset));
                }
                commands.entity(entity).insert(BoundsState {
                    asset: id,
                    trim,
                    pending,
                });
                continue;
            }
        };
        if let Some(task) = state.pending.as_mut()
            && let Some(result) = check_ready(task)
        {
            state.pending = None;
            commands
                .entity(entity)
                .insert(from_result(result, &trim, asset));
        }
    }
}
