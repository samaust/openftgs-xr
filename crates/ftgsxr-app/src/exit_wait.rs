// SPDX-License-Identifier: Apache-2.0

//! The exit-time wait for in-flight pipeline compiles (#376, §1.7 Command line, §2.4
//! Threads, Q-233). With bevy's `multi_threaded`, `PipelineCache` compiles pipelines on the
//! async compute pool; a process that exits while one compiles leaves the worker inside
//! the GPU driver, whose exit-time teardown can crash it. On an `AppExit` the app holds the
//! exit until a render frame that ran after it reports no pipeline queued or compiling, at
//! most `CAP` of wall time, then lets it through, in the same `Last` run as the settings'
//! save at exit, which so sees it once.
//!
//! With pipelined rendering the render world lives on its own thread, so its report
//! reaches the main world through a handle both worlds hold. A compile first queued in the
//! render frame that runs after the release can still race the exit; pipelines queue when
//! something first draws, which a quiet frame doesn't.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use bevy::app::{App, AppExit, Last};
use bevy::ecs::message::{MessageCursor, Messages};
use bevy::ecs::prelude::{IntoScheduleConfigs, Local, Res, Resource, World};
use bevy::log::warn;
use bevy::render::render_resource::{CachedPipelineState, PipelineCache, PipelineDescriptor};
use bevy::render::{Render, RenderApp, RenderSystems};

/// Q-233: the longest an exit waits for in-flight compiles.
// Q-233
pub(crate) const CAP: Duration = Duration::from_secs(2);

/// The render world's latest report.
#[derive(Default)]
struct Report {
    /// The render frames reported so far.
    frame: u64,
    /// The pipelines queued or compiling at that frame, by label.
    pending: Vec<String>,
}

/// The handle both worlds hold: the render world writes, the main world reads.
#[derive(Resource, Clone, Default)]
struct PendingPipelines(Arc<Mutex<Report>>);

impl PendingPipelines {
    fn lock(&self) -> std::sync::MutexGuard<'_, Report> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Adds the wait when the render app exists (`AppPlugin` is built after `RenderPlugin`,
/// before pipelined rendering moves the render app to its thread); an app without one
/// exits as before.
pub(crate) fn add(app: &mut App) {
    let shared = PendingPipelines::default();
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render
        .insert_resource(shared.clone())
        .add_systems(Render, report_pending.in_set(RenderSystems::Cleanup));
    app.insert_resource(shared)
        .add_systems(Last, hold_exit.before(crate::persist::save_at_exit));
}

/// Render world: the pipelines queued or compiling after this frame's queue processing.
fn report_pending(cache: Res<PipelineCache>, shared: Res<PendingPipelines>) {
    let pending: Vec<String> = cache
        .pipelines()
        .filter(|p| {
            matches!(
                p.state,
                CachedPipelineState::Queued | CachedPipelineState::Creating(_)
            )
        })
        .map(|p| label(&p.descriptor))
        .collect();
    let mut report = shared.lock();
    report.frame += 1;
    report.pending = pending;
}

fn label(d: &PipelineDescriptor) -> String {
    let l = match d {
        PipelineDescriptor::RenderPipelineDescriptor(d) => d.label.as_deref(),
        PipelineDescriptor::ComputePipelineDescriptor(d) => d.label.as_deref(),
    };
    l.unwrap_or("unlabelled").to_owned()
}

/// What the main world holds.
#[derive(Default)]
struct Held {
    /// The exit held, an error over a success as `App::should_exit` chooses.
    exit: Option<AppExit>,
    /// When it was first held, and the render frames reported then.
    since: Option<(Instant, u64)>,
    /// Once released, every later exit passes.
    released: bool,
}

/// Main world, in `Last` before the settings' save at exit: holds a new `AppExit` (taking
/// it out of the messages, so neither the save nor the runner sees it) until a render
/// frame after it reports nothing pending, or `CAP` passes; then writes it back.
fn hold_exit(world: &mut World, mut cursor: Local<MessageCursor<AppExit>>, mut held: Local<Held>) {
    if held.released {
        return;
    }
    let Some(messages) = world.get_resource::<Messages<AppExit>>() else {
        return;
    };
    for e in cursor.read(messages) {
        held.exit = match (held.exit.take(), e) {
            (Some(prev), _) if prev.is_error() => Some(prev),
            _ => Some(e.clone()),
        };
    }
    let Some(exit) = held.exit.clone() else {
        return;
    };
    let shared = world.resource::<PendingPipelines>().clone();
    let (frame, pending) = {
        let r = shared.lock();
        (r.frame, r.pending.clone())
    };
    let (start, at) = *held.since.get_or_insert((Instant::now(), frame));
    let quiet = frame > at && pending.is_empty();
    let late = start.elapsed() >= CAP;
    let mut messages = world.resource_mut::<Messages<AppExit>>();
    messages.clear();
    if quiet || late {
        if !quiet {
            warn!(
                "exiting with {} render pipelines still queued or compiling after {CAP:?}: {}",
                pending.len(),
                pending.join(", ")
            );
        }
        messages.write(exit);
        held.released = true;
    }
}
