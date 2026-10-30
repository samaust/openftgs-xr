// SPDX-License-Identifier: Apache-2.0

//! The main world's part of the adapter: the device the passes share with Bevy (Q-74,
//! Q-T119.1), loads on the async compute task pool (§2.4 Threads), uploads of at most
//! 64 MB per frame (§1.5, §2.4 rule 1), `SplatEvent::Uploaded` and `Failed`, and D-6's
//! settings on a new splat camera.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use ::bevy::asset::{AssetEvent, AssetId, Assets};
use ::bevy::camera::{Camera, ClearColorConfig};
use ::bevy::color::Color;
use ::bevy::core_pipeline::tonemapping::Tonemapping;
use ::bevy::ecs::prelude::*;
use ::bevy::log::{info, warn};
use ::bevy::render::renderer::{RenderAdapter, RenderAdapterInfo, RenderDevice, RenderQueue};
use ::bevy::render::view::Msaa;
use ::bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use ftgsxr_core::ModelKind;
use ftgsxr_io::ModelBytes;

use super::{SplatAsset, SplatCamera, SplatEvent, SplatModel, SplatTierState, load_splat_asset};
use crate::gpu::device::{BudgetSource, MemoryBudget, SplatDevice, largest_device_local_heap};
use crate::gpu::model::ModelUpload;
use crate::gpu::tier::DESKTOP_FALLBACK_BUDGET;
use crate::gpu::{GpuModel, RenderError};

/// The bytes uploaded per frame at most (§1.5, §2.4 rule 1: "at most 64 MB"), in MB of
/// 10⁶ bytes as §1.15's texts count them (Q-74).
pub(crate) const UPLOAD_BYTES_PER_FRAME: u64 = 64_000_000;

/// A model to load into the entity (Q-T119.2): the plugin reads, validates and packs it
/// on the async compute task pool with the tier's load options and the governor's SH cap
/// (`load_splat_asset`), then adds the `SplatAsset` and inserts `SplatModel` on the
/// entity; a failure is `SplatEvent::Failed` on the entity, which gets no `SplatModel`.
#[derive(Component, Debug)]
pub struct SplatLoad {
    /// The model's bytes, or a checkpoint's directory (D-51).
    pub bytes: ModelBytes,
    /// The kind, when the source knows it; sniffed from the bytes otherwise.
    pub kind_hint: Option<ModelKind>,
    /// §1.2 load option 4's explicit reduction to that many rows (D-13).
    pub reduce_to: Option<usize>,
}

/// A load in progress on the async compute task pool.
#[derive(Component)]
pub(crate) struct LoadTask(Task<Result<SplatAsset, RenderError>>);

/// The device the passes run on: Bevy's, with the memory budget (Q-66, Q-86).
#[derive(Resource, Clone)]
pub(crate) struct SplatGpu(pub(crate) Arc<SplatDevice>);

/// The models on the GPU, those uploading and those that failed, by asset.
#[derive(Resource, Default)]
pub(crate) struct GpuModels {
    pub(crate) ready: HashMap<AssetId<SplatAsset>, Arc<GpuModel>>,
    uploading: VecDeque<(AssetId<SplatAsset>, ModelUpload)>,
    failed: HashMap<AssetId<SplatAsset>, String>,
}

impl GpuModels {
    fn knows(&self, id: AssetId<SplatAsset>) -> bool {
        self.ready.contains_key(&id)
            || self.failed.contains_key(&id)
            || self.uploading.iter().any(|(i, _)| *i == id)
    }

    fn forget(&mut self, id: AssetId<SplatAsset>) {
        self.ready.remove(&id);
        self.failed.remove(&id);
        self.uploading.retain(|(i, _)| *i != id);
    }
}

/// The asset an entity's `Uploaded` or `Failed` was reported for, so each is reported
/// once per entity and asset.
#[derive(Component)]
pub(crate) struct Reported(AssetId<SplatAsset>);

/// `<code>: <user text>` (§1.15, Q-64), the text of `SplatEvent::Failed`.
fn error_text(e: &RenderError) -> String {
    format!("{}: {}", e.code(), e.user_text())
}

/// The device budget the plugin checks uploads against: §1.12's desktop rule (Q-66) for
/// the tiers that hold the 8 GiB placeholder, the tier's own figure otherwise (Q-86).
fn budget(tier_bytes: u64, adapter: Option<&wgpu::Adapter>) -> MemoryBudget {
    // Q-66, Q-86
    if tier_bytes != DESKTOP_FALLBACK_BUDGET {
        return MemoryBudget {
            bytes: tier_bytes,
            source: BudgetSource::Tier,
        };
    }
    MemoryBudget::desktop(adapter.and_then(largest_device_local_heap))
}

/// Makes the `SplatDevice` from Bevy's device once the renderer exists, and again when
/// Bevy re-creates it; models of an old device are uploaded again.
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: one parameter per resource it reads"
)]
pub(crate) fn ensure_device(
    mut commands: Commands,
    device: Option<Res<RenderDevice>>,
    queue: Option<Res<RenderQueue>>,
    adapter: Option<Res<RenderAdapter>>,
    info: Option<Res<RenderAdapterInfo>>,
    gpu: Option<Res<SplatGpu>>,
    tier: Res<SplatTierState>,
    mut models: ResMut<GpuModels>,
    reported: Query<Entity, With<Reported>>,
) {
    let (Some(device), Some(queue), Some(info)) = (device, queue, info) else {
        return;
    };
    if gpu.is_some() && !device.is_changed() {
        return;
    }
    let budget = budget(
        tier.tier.memory_budget_bytes,
        adapter.as_ref().map(|a| &**a.0),
    );
    match budget.source {
        BudgetSource::AdapterHeap { heap_bytes } => info!(
            "splat memory budget {} bytes: 75 % of the {heap_bytes}-byte device-local heap",
            budget.bytes
        ),
        BudgetSource::Fallback => warn!(
            "splat memory budget {} bytes: the fallback, the adapter's heap could not be queried",
            budget.bytes
        ),
        BudgetSource::Tier => info!("splat memory budget {} bytes: the tier's", budget.bytes),
    }
    // Q-T119.1: Bevy's device keeps Bevy's error handlers.
    let dev = SplatDevice::from_bevy(
        device.wgpu_device().clone(),
        (**queue.0).clone(),
        (*info.0).clone(),
        budget,
    );
    commands.insert_resource(SplatGpu(Arc::new(dev)));
    *models = GpuModels::default();
    for entity in &reported {
        commands.entity(entity).remove::<Reported>();
    }
}

/// D-6 on a new splat camera (Q-T119.3): MSAA off, no tonemapping, cleared transparent.
pub(crate) fn make_camera_follow_d6(
    add: On<Add, SplatCamera>,
    mut cameras: Query<&mut Camera>,
    mut commands: Commands,
) {
    let entity = add.entity;
    commands
        .entity(entity)
        .insert((Msaa::Off, Tonemapping::None));
    if let Ok(mut camera) = cameras.get_mut(entity) {
        camera.clear_color = ClearColorConfig::Custom(Color::NONE);
    }
}

/// Starts each new `SplatLoad` as one task on the async compute task pool (§2.4 Threads).
pub(crate) fn start_loads(
    mut commands: Commands,
    loads: Query<Entity, (With<SplatLoad>, Without<LoadTask>)>,
    tier: Res<SplatTierState>,
) {
    for entity in &loads {
        let opts = tier.load_options();
        commands.queue(move |world: &mut World| {
            let Some(load) = world.entity_mut(entity).take::<SplatLoad>() else {
                return;
            };
            let task = AsyncComputeTaskPool::get().spawn(async move {
                load_splat_asset(&load.bytes, load.kind_hint, &opts, load.reduce_to)
            });
            world.entity_mut(entity).insert(LoadTask(task));
        });
    }
}

/// Adds each finished load's asset and its `SplatModel`, or reports its failure.
pub(crate) fn finish_loads(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut LoadTask)>,
    mut assets: ResMut<Assets<SplatAsset>>,
) {
    for (entity, mut task) in &mut tasks {
        let Some(result) = check_ready(&mut task.0) else {
            continue;
        };
        let mut e = commands.entity(entity);
        e.remove::<LoadTask>();
        match result {
            Ok(asset) => {
                e.insert(SplatModel(assets.add(asset)));
            }
            Err(error) => {
                warn!("splat load failed: {}", error_text(&error));
                commands.trigger(SplatEvent::Failed {
                    entity,
                    error: error_text(&error),
                });
            }
        }
    }
}

/// Begins the upload of every instance's model not on the GPU yet, writes at most
/// `UPLOAD_BYTES_PER_FRAME` of the uploads in order, and reports each instance's
/// `Uploaded` or `Failed` once (§1.5: the instance appears when the upload completes).
pub(crate) fn upload_models(
    mut commands: Commands,
    gpu: Option<Res<SplatGpu>>,
    mut models: ResMut<GpuModels>,
    assets: Res<Assets<SplatAsset>>,
    mut events: MessageReader<AssetEvent<SplatAsset>>,
    instances: Query<(Entity, &SplatModel, Option<&Reported>)>,
) {
    for event in events.read() {
        match *event {
            AssetEvent::Modified { id }
            | AssetEvent::Removed { id }
            | AssetEvent::Unused { id } => {
                models.forget(id);
            }
            _ => {}
        }
    }
    let Some(gpu) = gpu else {
        return;
    };
    let dev = &gpu.0;
    for (_, model, _) in &instances {
        let id = model.0.id();
        if models.knows(id) {
            continue;
        }
        let Some(asset) = assets.get(id) else {
            continue;
        };
        let p = &asset.packed;
        match ModelUpload::begin(
            dev,
            Arc::clone(&p.arrays),
            p.index.clone(),
            &p.conventions,
            p.speed_p99,
        ) {
            Ok(upload) => models.uploading.push_back((id, upload)),
            Err(e) => {
                models.failed.insert(id, error_text(&e));
            }
        }
    }
    let mut budget = UPLOAD_BYTES_PER_FRAME;
    while let Some((_, upload)) = models.uploading.front_mut() {
        if !upload.step(dev, &mut budget) {
            break;
        }
        let (id, upload) = models.uploading.pop_front().expect("the front");
        models.ready.insert(id, Arc::new(upload.finish()));
    }
    for (entity, model, reported) in &instances {
        let id = model.0.id();
        if reported.is_some_and(|r| r.0 == id) {
            continue;
        }
        let event = if let Some(m) = models.ready.get(&id) {
            SplatEvent::Uploaded {
                entity,
                bytes: m.memory_bytes(),
            }
        } else if let Some(error) = models.failed.get(&id) {
            warn!("splat upload failed: {error}");
            SplatEvent::Failed {
                entity,
                error: error.clone(),
            }
        } else {
            continue;
        };
        commands.entity(entity).insert(Reported(id));
        commands.trigger(event);
    }
}
