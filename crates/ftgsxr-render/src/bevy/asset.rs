// SPDX-License-Identifier: Apache-2.0

//! `SplatAsset` and `load_splat_asset` (§2.1): the CPU part of a load (§1.5 Loading and
//! unloading): read and validate (§1.2), the explicit reduction (D-13), the GPU packing
//! with the load options (§1.2) and the temporal ordering (E3, D-10).

use std::path::Path;
use std::sync::Arc;

use ::bevy::asset::Asset;
use ::bevy::log::warn;
use ::bevy::reflect::TypePath;
use ftgsxr_core::{
    Aabb, CoreError, GpuArrays, GpuLayoutOptions, Model, ModelInfo, ModelKind, RenderConventions,
    TemporalIndex, pack_for_gpu, read_checkpoint, read_exchange, read_ply, reduce_to_budget,
};
use ftgsxr_io::ModelBytes;

use crate::gpu::RenderError;
use crate::gpu::model::packed_speed_p99;

/// The samples E10's box takes over the span: τ₀ + k T_τ/8, k = 0 … 8.
const BOX_SAMPLES: usize = 8;

/// A model packed for the GPU, shared by every instance of the asset and kept until the
/// asset is dropped, so the upload can be made again.
pub(crate) struct Packed {
    pub(crate) arrays: Arc<GpuArrays>,
    pub(crate) index: TemporalIndex,
    pub(crate) conventions: RenderConventions,
    /// The 99th percentile of the packed speeds (Q-93), computed off the main thread.
    pub(crate) speed_p99: f32,
}

/// A loaded model, ready to upload (§2.1): its header information, E10's box B_m in model
/// space, its identity (D-40), and the packed arrays and temporal index.
#[derive(Asset, TypePath)]
pub struct SplatAsset {
    /// What the file's header tells, with the size of the file the identity hashes (Q-76).
    pub info: ModelInfo,
    /// E10's B_m, in model space.
    pub bounds: Aabb,
    /// The SHA-256 of the file's bytes, of `params.safetensors` for a checkpoint (D-40).
    pub identity: [u8; 32],
    pub(crate) packed: Arc<Packed>,
}

/// The CPU part of a load (read, validate, pack), runnable in a task pool;
/// `ModelBytes::Checkpoint` is read by path (D-51).
///
/// `Owned` bytes are read as `kind_hint` says, or, without a hint, as a PLY snapshot when
/// they start with the PLY magic `ply` and as an exchange file otherwise; a checkpoint
/// hint on owned bytes is E-FILE-FORMAT (a checkpoint is a directory, D-51). With
/// `reduce_to`, the model keeps its `reduce_to` rows of highest importance (§1.2 load
/// option 4, D-13). `opts` are the tier's load options (§1.2 options 1 and 2); the
/// memory check against the budget is the upload's (§2.4 rule 4). A convention rendered
/// as openftgs's instead is logged as a warning (E6, D-39, Q-110).
pub fn load_splat_asset(
    bytes: &ModelBytes,
    kind_hint: Option<ModelKind>,
    opts: &GpuLayoutOptions,
    reduce_to: Option<usize>,
) -> Result<SplatAsset, RenderError> {
    let (model, file_bytes) = read_model(bytes, kind_hint).map_err(RenderError::Core)?;
    // D-39, Q-110
    for w in model.conventions.warnings() {
        warn!("{w}");
    }
    let model = match reduce_to {
        Some(budget) if budget < model.len() => {
            select_rows(&model, &reduce_to_budget(&model, budget))
        }
        _ => model,
    };
    let info = ModelInfo {
        n: model.len() as u64,
        sh_degree: model.sh_degree,
        time: model.time.clone(),
        producer: model.producer.clone(),
        conventions: model.conventions.clone(),
        kind: model.kind,
        file_bytes,
    };
    let bounds = model.bounds(BOX_SAMPLES);
    let (index, order) = TemporalIndex::build(&model);
    let arrays = pack_for_gpu(&model, &order, opts);
    let speed_p99 = packed_speed_p99(&arrays);
    Ok(SplatAsset {
        info,
        bounds,
        identity: model.identity,
        packed: Arc::new(Packed {
            arrays: Arc::new(arrays),
            index,
            conventions: model.conventions,
            speed_p99,
        }),
    })
}

/// The model and the size of the file its identity hashes (Q-76).
fn read_model(bytes: &ModelBytes, kind_hint: Option<ModelKind>) -> Result<(Model, u64), CoreError> {
    match bytes {
        ModelBytes::Checkpoint(dir) => {
            let model = read_checkpoint(dir)?;
            Ok((model, params_bytes(dir)?))
        }
        ModelBytes::Owned(b) => {
            let model = match kind_hint {
                Some(ModelKind::Ply) => read_ply(b)?,
                Some(ModelKind::ExchangeFile) => read_exchange(b)?,
                Some(ModelKind::Checkpoint { .. }) => {
                    return Err(CoreError::Format(
                        "a checkpoint is a directory and loads by path (D-51), not from bytes"
                            .into(),
                    ));
                }
                None if b.starts_with(b"ply") => read_ply(b)?,
                None => read_exchange(b)?,
            };
            Ok((model, b.len() as u64))
        }
    }
}

/// The size of a checkpoint's `params.safetensors` (Q-76).
fn params_bytes(dir: &Path) -> Result<u64, CoreError> {
    Ok(std::fs::metadata(dir.join("params.safetensors"))
        .map_err(CoreError::Io)?
        .len())
}

/// The rows `rows` of `model` (file order kept), every other field unchanged.
fn select_rows(model: &Model, rows: &[u32]) -> Model {
    let k = (usize::from(model.sh_degree) + 1).pow(2) * 3;
    let pick = |i: &u32| *i as usize;
    Model {
        position: rows.iter().map(|i| model.position[pick(i)]).collect(),
        time_center: rows.iter().map(|i| model.time_center[pick(i)]).collect(),
        duration: rows.iter().map(|i| model.duration[pick(i)]).collect(),
        velocity: rows.iter().map(|i| model.velocity[pick(i)]).collect(),
        scale: rows.iter().map(|i| model.scale[pick(i)]).collect(),
        rotation: rows.iter().map(|i| model.rotation[pick(i)]).collect(),
        opacity: rows.iter().map(|i| model.opacity[pick(i)]).collect(),
        sh: rows
            .iter()
            .flat_map(|i| model.sh[pick(i) * k..(pick(i) + 1) * k].iter().copied())
            .collect(),
        sh_degree: model.sh_degree,
        time: model.time.clone(),
        conventions: model.conventions.clone(),
        producer: model.producer.clone(),
        identity: model.identity,
        kind: model.kind,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-row degree-1 model.
    fn two_rows() -> Model {
        Model {
            position: vec![[0.0; 3], [1.0, 2.0, 3.0]],
            time_center: vec![0.0, 1.0],
            duration: vec![1.0, 1.0],
            velocity: vec![[0.0; 3]; 2],
            scale: vec![[0.1; 3], [0.2; 3]],
            rotation: vec![[1.0, 0.0, 0.0, 0.0]; 2],
            opacity: vec![0.5, 0.25],
            sh: (0..24).map(|v| v as f32).collect(),
            sh_degree: 1,
            time: ftgsxr_core::TimeSpan {
                tau0: 0.0,
                tau1: 1.0,
                time_unit: "s".into(),
            },
            conventions: RenderConventions::default(),
            producer: String::new(),
            identity: [7; 32],
            kind: ModelKind::ExchangeFile,
        }
    }

    #[test]
    fn select_rows_keeps_each_rows_fields_and_its_sh_block() {
        let m = two_rows();
        let s = select_rows(&m, &[1]);
        assert_eq!(s.position, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(s.opacity, vec![0.25]);
        assert_eq!(s.time_center, vec![1.0]);
        assert_eq!(s.sh, (12..24).map(|v| v as f32).collect::<Vec<_>>());
        assert_eq!(s.identity, m.identity);
    }

    #[test]
    fn owned_bytes_with_a_checkpoint_hint_are_a_format_error() {
        let hint = ModelKind::Checkpoint {
            spatial_scale: 1.0,
            frame_dt: 0.1,
        };
        let r = load_splat_asset(
            &ModelBytes::Owned(vec![0; 16]),
            Some(hint),
            &GpuLayoutOptions::default(),
            None,
        );
        match r {
            Err(RenderError::Core(e)) => assert_eq!(e.code(), "E-FILE-FORMAT"),
            Err(e) => panic!("{e:?}"),
            Ok(_) => panic!("loaded"),
        }
    }
}
