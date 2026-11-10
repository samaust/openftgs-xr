// SPDX-License-Identifier: Apache-2.0

//! `LayerProbe` (§2.1, E14's exposure match): the main world's one-shot read back of the
//! layer probe through Bevy's `gpu_readback`, as Q-388's depth readback. While `request`
//! is set and none is in flight, a `ShaderBuffer` of the probe's size and a `Readback` of
//! it are spawned; the render world's draw node encodes the probe of the first splat
//! camera's layer (the left eye, Q-125) into it in that frame, and Bevy copies it after
//! the frame's render passes. The `Readback` is removed the frame after, so it is copied
//! once, and its `ReadbackComplete` writes `result` and clears `request`. Nothing waits
//! for the GPU (D-34, §2.4 rule 1). A read back that the probe did not write (no
//! `LinearOver` layer that frame) lacks the probe's mark and leaves `request` set, so the
//! next frame asks again.

use ::bevy::asset::{Assets, Handle, RenderAssetUsages};
use ::bevy::ecs::prelude::*;
use ::bevy::log::warn;
use ::bevy::render::Extract;
use ::bevy::render::gpu_readback::{Readback, ReadbackComplete};
use ::bevy::render::storage::ShaderBuffer;

use crate::gpu::LayerSums;
use crate::gpu::probe::PROBE_BYTES;

/// Frames after which a readback that has not returned is dropped and asked again.
const READBACK_FRAMES_MAX: u32 = 8;

/// E14's exposure match: while `request` is set, the left eye's layer is probed once after
/// the splat pass and read back without waiting; `result` arrives one or two frames later
/// and clears `request`.
#[derive(Resource, Default, Debug)]
pub struct LayerProbe {
    /// Set to ask for one probe; cleared when `result` is written.
    pub request: bool,
    /// The newest probe's sums.
    pub result: Option<LayerSums>,
}

/// The readback in flight: its entity, its buffer and the frames since it was spawned.
#[derive(Resource, Default)]
pub(crate) struct ProbeReadback {
    pending: Option<(Entity, Handle<ShaderBuffer>, u32)>,
}

/// Render world: the buffer the draw node probes into this frame, if any.
#[derive(Resource, Default)]
pub(crate) struct ProbeTarget(pub(crate) Option<Handle<ShaderBuffer>>);

/// Main world (`Update`): spawns the one-shot readback on a request, removes its
/// `Readback` the frame after and drops one that never returned.
pub(crate) fn request(
    probe: Option<Res<LayerProbe>>,
    mut rb: ResMut<ProbeReadback>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut commands: Commands,
) {
    if let Some((entity, _, frames)) = rb.pending.as_mut() {
        *frames += 1;
        let (entity, frames) = (*entity, *frames);
        if frames == 1 {
            commands.entity(entity).remove::<Readback>();
        } else if frames > READBACK_FRAMES_MAX {
            warn!("layer probe: the readback did not return in {frames} frames; asked again");
            commands.entity(entity).despawn();
            rb.pending = None;
        }
        return;
    }
    if !probe.is_some_and(|p| p.request) {
        return;
    }
    // Zeros: without the probe's mark, a copy the probe did not write is told apart.
    let buffer = buffers.add(ShaderBuffer::new(
        &[0; PROBE_BYTES as usize],
        RenderAssetUsages::RENDER_WORLD,
    ));
    let entity = commands
        .spawn(Readback::buffer(buffer.clone()))
        .observe(complete)
        .id();
    rb.pending = Some((entity, buffer, 0));
}

/// The sums into `LayerProbe.result`, clearing `request`; a copy without the mark
/// leaves the request for the next frame.
fn complete(
    event: On<ReadbackComplete>,
    mut rb: ResMut<ProbeReadback>,
    probe: Option<ResMut<LayerProbe>>,
    mut commands: Commands,
) {
    let entity = event.event().entity;
    commands.entity(entity).despawn();
    if !matches!(rb.pending, Some((e, ..)) if e == entity) {
        return;
    }
    rb.pending = None;
    let (Some(mut probe), Some(sums)) = (probe, LayerSums::from_bytes(&event.data)) else {
        return;
    };
    probe.result = Some(sums);
    probe.request = false;
}

/// Extract: the buffer to probe into, in the frame its readback was spawned.
pub(crate) fn extract_target(mut target: ResMut<ProbeTarget>, rb: Extract<Res<ProbeReadback>>) {
    target.0 = match &rb.pending {
        Some((_, buffer, 0)) => Some(buffer.clone()),
        _ => None,
    };
}
