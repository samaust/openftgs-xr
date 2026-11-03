// SPDX-License-Identifier: Apache-2.0

//! The exit-time wait for in-flight pipeline compiles (#376, §1.7 Command line, §2.4
//! Threads, Q-236). With bevy's `multi_threaded`, `PipelineCache` compiles pipelines on the
//! async compute pool; a process that exits while one compiles leaves the worker inside
//! the GPU driver, whose exit-time teardown can crash it. On an `AppExit` the app holds the
//! exit until a render frame that ran after it reports no pipeline queued or compiling, at
//! most `CAP` of wall time, then lets it through. The hold runs in `Last`, in bevy's
//! `ExitSystems` after the systems that write the window's exit there, so that exit is held
//! too; still inside the set, so before bevy_winit's `despawn_windows`, which drops every
//! window once it sees an exit (dropped while the frames go on, they hang the render
//! thread); and before the settings' save at exit, so the save sees the exit once, when it
//! passes.
//!
//! With pipelined rendering the render world lives on its own thread, so its report
//! reaches the main world through a handle both worlds hold. A compile first queued in the
//! render frame that runs after the release can still race the exit; pipelines queue when
//! something first draws, which a quiet frame doesn't.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use bevy::app::{App, AppExit, Last};
use bevy::ecs::message::{MessageCursor, Messages};
use bevy::ecs::prelude::{IntoScheduleConfigs, Local, Res, Resource, World};
use bevy::log::warn;
use bevy::render::render_resource::{CachedPipelineState, PipelineCache, PipelineDescriptor};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::window::{ExitSystems, exit_on_all_closed, exit_on_primary_closed};

/// Q-236: the longest an exit waits for in-flight compiles.
// Q-236
const CAP: Duration = Duration::from_secs(2);

/// The render world's latest report.
#[derive(Default)]
struct Report {
    /// How many render frames have reported.
    frames_reported: u64,
    /// The pipelines queued or compiling at the latest of them, by label.
    pending: Vec<String>,
}

/// The handle both worlds hold: the render world writes, the main world reads.
#[derive(Resource, Clone, Default)]
struct PendingPipelines(Arc<Mutex<Report>>);

impl PendingPipelines {
    fn lock(&self) -> MutexGuard<'_, Report> {
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
    app.insert_resource(shared).add_systems(
        Last,
        hold_exit
            .in_set(ExitSystems)
            .after(exit_on_all_closed)
            .after(exit_on_primary_closed)
            .before(crate::persist::save_at_exit),
    );
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
    report.frames_reported += 1;
    report.pending = pending;
}

/// A pipeline's label for the cap's warning.
fn label(descriptor: &PipelineDescriptor) -> String {
    let label = match descriptor {
        PipelineDescriptor::RenderPipelineDescriptor(render) => render.label.as_deref(),
        PipelineDescriptor::ComputePipelineDescriptor(compute) => compute.label.as_deref(),
    };
    label.unwrap_or("unlabelled").to_owned()
}

/// When an exit was first held.
#[derive(Clone, Copy)]
struct HeldSince {
    /// The wall time, for `CAP`.
    started: Instant,
    /// The render frames reported by then: a quiet report must come from a later one.
    frames_reported: u64,
}

/// What the main world holds.
#[derive(Default)]
struct Held {
    /// The exit held, an error over a success as `App::should_exit` chooses.
    exit: Option<AppExit>,
    /// When it was first held.
    since: Option<HeldSince>,
    /// Once released, every later exit passes.
    released: bool,
}

/// Main world, in `Last`'s `ExitSystems` after the window's exit systems, before the
/// settings' save at exit: holds a new `AppExit` (taking
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
    let (frames_reported, pending) = {
        let report = shared.lock();
        (report.frames_reported, report.pending.clone())
    };
    let since = *held.since.get_or_insert(HeldSince {
        started: Instant::now(),
        frames_reported,
    });
    let quiet = frames_reported > since.frames_reported && pending.is_empty();
    let late = since.started.elapsed() >= CAP;
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
