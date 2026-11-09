// SPDX-License-Identifier: Apache-2.0

//! `GpuModel`: one model's structure-of-arrays buffers (§2.4 GPU data layout, Q-22), its
//! temporal index (E3, D-10) and its uniform block of E6 and E7 constants (D-39).

use bevy::log::warn;
use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
use ftgsxr_core::{GpuArrays, RenderConventions, SH_BINDINGS, TemporalIndex};

use super::device::{Reservation, SplatDevice};
use super::error::RenderError;
use super::readback::half_to_f32;

/// The model's uniform block: every E6 and E7 constant of its `render_conventions` (D-39,
/// §2.6: never a literal in a shader) and the layout facts the shaders read. Little-endian
/// 32-bit words, 64 bytes, in this order (byte offsets):
///
/// | offset | field | type |
/// | --- | --- | --- |
/// | 0 | `alpha_max` (E7) | f32 |
/// | 4 | `eps2d` (E6) | f32 |
/// | 8 | `near` (E6) | f32 |
/// | 12 | `far` (E6) | f32 |
/// | 16 | `alpha_skip` (E3, E6, E7) | f32 |
/// | 20 | `pixel_center_offset` (E6, E7) | f32 |
/// | 24 | `jacobian_clamp_margin` (E6) | f32 |
/// | 28 | `screen_extent_max` (E6) | f32 |
/// | 32 | N | u32 |
/// | 36 | the loaded SH degree L′ | u32 |
/// | 40 | half fields: bit 0 velocity, 1 scale, 2 rotation, 3 opacity, 4 sh | u32 |
/// | 44 | single timestamp (1) or not (0) | u32 |
/// | 48 | W, the words of one SH range (Q-376) | u32 |
/// | 52 | 0 (padding to 16-byte rows) | 3 × u32 |
///
/// `transmittance_stop` is not applied (D-7), and the background is the view's (E7).
pub(crate) const MODEL_UNIFORM_BYTES: usize = 64; // Q-376

/// The model buffers' extra usage in the crate's unit tests, which read them back.
const TEST_READ: BufferUsages = if cfg!(test) {
    BufferUsages::COPY_SRC
} else {
    BufferUsages::empty()
};

fn model_uniform(
    arrays: &GpuArrays,
    conv: &RenderConventions,
    sh_range_words: u32,
) -> [u8; MODEL_UNIFORM_BYTES] {
    let h = arrays.half;
    let flags = u32::from(h.velocity)
        | u32::from(h.scale) << 1
        | u32::from(h.rotation) << 2
        | u32::from(h.opacity) << 3
        | u32::from(h.sh) << 4;
    let words: [[u8; 4]; 16] = [
        conv.alpha_max.to_le_bytes(),
        conv.eps2d.to_le_bytes(),
        conv.near.to_le_bytes(),
        conv.far.to_le_bytes(),
        conv.alpha_skip.to_le_bytes(),
        conv.pixel_center_offset.to_le_bytes(),
        conv.jacobian_clamp_margin.to_le_bytes(),
        conv.screen_extent_max.to_le_bytes(),
        arrays.n.to_le_bytes(),
        u32::from(arrays.sh_degree).to_le_bytes(),
        flags.to_le_bytes(),
        u32::from(arrays.single).to_le_bytes(),
        sh_range_words.to_le_bytes(),
        [0; 4],
        [0; 4],
        [0; 4],
    ];
    let mut out = [0u8; MODEL_UNIFORM_BYTES];
    out.copy_from_slice(words.as_flattened());
    out
}

/// The buffers of one model, one per `GpuArrays` field (§2.4: no padding between rows; a
/// buffer is padded to whole 32-bit words at its end, Q-22).
pub(crate) struct ModelBuffers {
    pub(crate) position: wgpu::Buffer,
    pub(crate) time_center: wgpu::Buffer,
    pub(crate) duration: wgpu::Buffer,
    pub(crate) velocity: wgpu::Buffer,
    pub(crate) scale: wgpu::Buffer,
    pub(crate) rotation: wgpu::Buffer,
    pub(crate) opacity: wgpu::Buffer,
    pub(crate) sh: wgpu::Buffer,
    pub(crate) uniform: wgpu::Buffer,
}

/// One model on the GPU (§2.1): the SoA buffers of `GpuArrays`, the temporal index, one
/// uniform block.
pub struct GpuModel {
    pub(crate) buffers: ModelBuffers,
    pub(crate) index: TemporalIndex,
    /// τ₀, the origin of the rebased time centers (§1.2 load option 3).
    pub(crate) tau0: f64,
    /// τ₁, the end of the model's span, for the time-center colormap (§1.4 stage 6). // Q-97
    pub(crate) tau1: f64,
    /// L′, the loaded SH degree.
    pub(crate) sh_degree: u8,
    /// The model's E6 and E7 constants (D-39), which the draw reads too.
    pub(crate) conventions: RenderConventions,
    /// The 99th percentile of the rows' speeds ‖velocity‖ in model units per unit of τ,
    /// the scale of the speed colormap (§1.4 stage 6). // Q-93
    pub(crate) speed_p99: f32,
    /// The bytes of one SH range, W · 4, on the model's device (Q-376).
    pub(crate) sh_range_bytes: u64,
    n: u32,
    /// The model's static bytes, reserved on its device until the model is dropped (Q-74).
    reservation: Reservation,
}

/// The bytes `arrays` holds: §1.12's static bytes per primitive times N (Q-23).
fn static_bytes(a: &GpuArrays) -> u64 {
    let f32s = (a.position.len() + a.time_center.len() + a.duration.len()) * 4;
    let packed = a.velocity.len() + a.scale.len() + a.rotation.len() + a.opacity.len() + a.sh.len();
    (f32s + packed) as u64
}

/// The bytes of a field's buffer holding `len` bytes: padded to whole words, at least one
/// (Q-22), the range its storage binding takes.
fn buffer_size(len: usize) -> usize {
    len.div_ceil(4).max(1) * 4
}

/// A storage buffer holding `bytes` padded with zeros to whole words (at least one),
/// written through the mapping at creation.
fn storage_buffer(dev: &SplatDevice, label: &str, mut bytes: Vec<u8>) -> wgpu::Buffer {
    let size = buffer_size(bytes.len());
    bytes.resize(size, 0);
    let buffer = dev.device().create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | TEST_READ,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(&bytes);
    buffer.unmap();
    buffer
}

fn f32_buffer(dev: &SplatDevice, label: &str, values: &[f32]) -> wgpu::Buffer {
    storage_buffer(
        dev,
        label,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn byte_buffer(dev: &SplatDevice, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    storage_buffer(dev, label, bytes.to_vec())
}

/// The rows' speeds ‖velocity‖ as the GPU reads them (float32 or half, Q-22).
fn speeds(arrays: &GpuArrays) -> Vec<f32> {
    let values: Vec<f32> = if arrays.half.velocity {
        arrays
            .velocity
            .as_chunks::<2>()
            .0
            .iter()
            .map(|h| half_to_f32(u16::from_le_bytes(*h)))
            .collect()
    } else {
        arrays
            .velocity
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect()
    };
    values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .collect()
}

/// The 99th percentile of the speeds by nearest rank, the ⌈0.99 N⌉-th smallest; 0 for no
/// rows. // Q-93
fn speed_percentile_99(arrays: &GpuArrays) -> f32 {
    let mut s = speeds(arrays);
    if s.is_empty() {
        return 0.0;
    }
    s.sort_by(f32::total_cmp);
    let rank = (s.len() * 99).div_ceil(100).max(1);
    s[rank - 1]
}

/// The bytes of one SH range on a device of `limits`, W · 4 (Q-376): the largest storage
/// binding rounded down to whole multiples of the storage offset alignment, at least 64
/// bytes, so that every range's offset is aligned.
// Q-376
pub(crate) fn sh_range_bytes(limits: &wgpu::Limits) -> u64 {
    let align = u64::from(limits.min_storage_buffer_offset_alignment).max(64);
    // At most 4 GiB, so W fits the shader's 32-bit word index.
    limits.max_storage_buffer_binding_size.min(1 << 32) / align * align
}

/// The SH ranges `bytes` of SH take at `range` bytes each, at least one (Q-376).
pub(crate) fn sh_ranges(bytes: u64, range: u64) -> u64 {
    bytes.div_ceil(range.max(1)).max(1)
}

/// W, the words of one SH range on `dev`, for the model's uniform (Q-376).
fn sh_range_words(dev: &SplatDevice) -> u32 {
    u32::try_from(sh_range_bytes(dev.limits()) / 4).expect("an SH range of at most 4 GiB")
}

/// `RenderError::Budget` when `arrays` cannot be bound on `dev` (Q-376, Q-369), logged with
/// N, the field, its buffer's bytes and the limit hit: (1) the SH buffer exceeds
/// `SH_BINDINGS` SH ranges, (2) a field's buffer exceeds the device's largest buffer, (3)
/// another field's buffer exceeds one storage binding; each buffer padded to whole words.
// Q-369, Q-376
fn check_bindings(dev: &SplatDevice, arrays: &GpuArrays) -> Result<(), RenderError> {
    let limits = dev.limits();
    let refuse = |name: &str, bytes: u64, budget: u64, what: String| {
        warn!(
            "model of N = {} refused: its {name} buffer takes {bytes} bytes, over {what} \
             (E-MEM-BUDGET)",
            arrays.n
        );
        Err(RenderError::Budget {
            needed: bytes,
            budget,
        })
    };
    let bytes = |i: usize| buffer_size(field(arrays, i).len()) as u64;
    let range = sh_range_bytes(limits);
    let sh_budget = u64::from(SH_BINDINGS) * range;
    if bytes(SH) > sh_budget {
        return refuse(
            FIELD_NAMES[SH],
            bytes(SH),
            sh_budget,
            format!("its {SH_BINDINGS} SH ranges of {range} bytes ({sh_budget} bytes)"),
        );
    }
    let max_buffer = limits.max_buffer_size;
    for (i, name) in FIELD_NAMES.iter().enumerate() {
        if bytes(i) > max_buffer {
            return refuse(
                name,
                bytes(i),
                max_buffer,
                format!("the device's {max_buffer}-byte largest buffer"),
            );
        }
    }
    let binding = limits.max_storage_buffer_binding_size;
    for (i, name) in FIELD_NAMES.iter().enumerate().take(SH) {
        if bytes(i) > binding {
            return refuse(
                name,
                bytes(i),
                binding,
                format!("the device's {binding}-byte storage binding"),
            );
        }
    }
    Ok(())
}

/// `arrays`' static bytes reserved on `dev`'s budget (Q-74), once every field's buffer fits
/// the device's bindings (Q-369, Q-376); a refusal is logged with N, the bytes needed and the budget
/// left (§1.2 load option 4, D-13), since `RenderError::Budget` carries no N.
// Q-117
fn reserve_logged(dev: &SplatDevice, arrays: &GpuArrays) -> Result<Reservation, RenderError> {
    check_bindings(dev, arrays)?;
    dev.reserve(static_bytes(arrays)).inspect_err(|e| {
        if let RenderError::Budget { needed, budget } = e {
            warn!(
                "model of N = {} refused: needs {needed} bytes, budget {budget} bytes (E-MEM-BUDGET)",
                arrays.n
            );
        }
    })
}

impl GpuModel {
    /// Uploads `arrays` (packed by `pack_for_gpu` in the order `index` was built for) with
    /// the model's conventions in its uniform block (D-39).
    ///
    /// The static bytes are reserved on the device's memory budget before anything is
    /// allocated on the GPU (§2.4 rule 4, §1.2 load option 4, D-13), and given back when
    /// the model is dropped (Q-74): a model that does not fit in what the device's live
    /// models leave of the budget is `RenderError::Budget`, its `budget` the bytes left.
    /// So is a model whose SH buffer exceeds `SH_BINDINGS` SH ranges, whose buffer exceeds
    /// the device's largest buffer, or whose other field's buffer exceeds one storage
    /// binding, its `needed` the buffer's bytes and its `budget` the limit hit (Q-369,
    /// Q-376).
    ///
    /// # Panics
    ///
    /// When `index` does not hold `arrays.n` rows.
    pub fn upload(
        dev: &SplatDevice,
        arrays: &GpuArrays,
        index: TemporalIndex,
        conv: &RenderConventions,
    ) -> Result<GpuModel, RenderError> {
        assert_eq!(
            index.len(),
            arrays.n,
            "the temporal index is built for the packed rows"
        );
        // Q-74: the model's static bytes against what the device's live models leave.
        let reservation = reserve_logged(dev, arrays)?;
        let uniform = dev.device().create_buffer(&BufferDescriptor {
            label: Some("ftgsxr model uniform"),
            size: MODEL_UNIFORM_BYTES as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | TEST_READ,
            mapped_at_creation: false,
        });
        dev.queue().write_buffer(
            &uniform,
            0,
            &model_uniform(arrays, conv, sh_range_words(dev)),
        );
        let buffers = ModelBuffers {
            position: f32_buffer(dev, "ftgsxr position", &arrays.position),
            time_center: f32_buffer(dev, "ftgsxr time center", &arrays.time_center),
            duration: f32_buffer(dev, "ftgsxr duration", &arrays.duration),
            velocity: byte_buffer(dev, "ftgsxr velocity", &arrays.velocity),
            scale: byte_buffer(dev, "ftgsxr scale", &arrays.scale),
            rotation: byte_buffer(dev, "ftgsxr rotation", &arrays.rotation),
            opacity: byte_buffer(dev, "ftgsxr opacity", &arrays.opacity),
            sh: byte_buffer(dev, "ftgsxr sh", &arrays.sh),
            uniform,
        };
        Ok(GpuModel {
            buffers,
            index,
            tau0: arrays.tau0,
            tau1: arrays.tau1,
            sh_degree: arrays.sh_degree,
            conventions: conv.clone(),
            speed_p99: speed_percentile_99(arrays),
            sh_range_bytes: sh_range_bytes(dev.limits()),
            n: arrays.n,
            reservation,
        })
    }

    /// The model's static GPU bytes: §1.12's bytes per primitive times N, as `GpuArrays`
    /// holds them (Q-23); the word padding at the ends of half-precision buffers (at most
    /// 2 bytes each, Q-22) and the 64-byte uniform block are not counted.
    pub fn memory_bytes(&self) -> u64 {
        self.reservation.bytes()
    }

    /// N, the rows uploaded.
    pub fn n(&self) -> u32 {
        self.n
    }
}

/// The eight `GpuArrays` fields in `ModelBuffers` order, as the bytes their buffers hold.
enum FieldBytes<'a> {
    /// Float32 values, written little-endian.
    F32(&'a [f32]),
    /// Packed bytes (float32 or half, Q-22).
    Bytes(&'a [u8]),
}

impl FieldBytes<'_> {
    fn len(&self) -> usize {
        match self {
            FieldBytes::F32(v) => v.len() * 4,
            FieldBytes::Bytes(b) => b.len(),
        }
    }

    /// Bytes `start..end`, zero-padded to whole words (`queue.write_buffer`'s alignment).
    fn range(&self, start: usize, end: usize) -> Vec<u8> {
        let mut out = match self {
            FieldBytes::F32(v) => v[start / 4..end.div_ceil(4)]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
            FieldBytes::Bytes(b) => b[start..end].to_vec(),
        };
        out.resize((end - start).div_ceil(4) * 4, 0);
        out
    }
}

const FIELDS: usize = 8;
/// The SH array's index in `field` order.
const SH: usize = FIELDS - 1;

/// The fields' names in `field` order, for the binding refusals' log (Q-369, Q-376).
const FIELD_NAMES: [&str; FIELDS] = [
    "position",
    "time center",
    "duration",
    "velocity",
    "scale",
    "rotation",
    "opacity",
    "sh",
];

fn field(a: &GpuArrays, i: usize) -> FieldBytes<'_> {
    match i {
        0 => FieldBytes::F32(&a.position),
        1 => FieldBytes::F32(&a.time_center),
        2 => FieldBytes::F32(&a.duration),
        3 => FieldBytes::Bytes(&a.velocity),
        4 => FieldBytes::Bytes(&a.scale),
        5 => FieldBytes::Bytes(&a.rotation),
        6 => FieldBytes::Bytes(&a.opacity),
        _ => FieldBytes::Bytes(&a.sh),
    }
}

fn field_buffer(b: &ModelBuffers, i: usize) -> &wgpu::Buffer {
    match i {
        0 => &b.position,
        1 => &b.time_center,
        2 => &b.duration,
        3 => &b.velocity,
        4 => &b.scale,
        5 => &b.rotation,
        6 => &b.opacity,
        _ => &b.sh,
    }
}

/// A model being uploaded in chunks (§1.5 Loading, §2.4 rule 1: at most 64 MB per
/// frame), for the Bevy adapter: the budget is reserved and the buffers made at `begin`,
/// the rows are written by `step` through the queue, a budget of bytes at a time, and the
/// model is usable once `step` reports the last chunk written. The reservation passes to
/// the model at `finish`; an upload dropped before it gives the bytes back (Q-74).
pub(crate) struct ModelUpload {
    arrays: std::sync::Arc<GpuArrays>,
    model: GpuModel,
    /// The field being written and the byte offset reached in it.
    field: usize,
    offset: usize,
}

impl ModelUpload {
    /// The budget reservation of `GpuModel::upload` (Q-74), then the model's buffers, unfilled
    /// (zeros), and its uniform block; `speed_p99` is the 99th percentile of the rows'
    /// speeds, computed with the packing off the main thread (Q-93).
    ///
    /// # Panics
    ///
    /// When `index` does not hold `arrays.n` rows.
    pub(crate) fn begin(
        dev: &SplatDevice,
        arrays: std::sync::Arc<GpuArrays>,
        index: TemporalIndex,
        conv: &RenderConventions,
        speed_p99: f32,
    ) -> Result<ModelUpload, RenderError> {
        assert_eq!(
            index.len(),
            arrays.n,
            "the temporal index is built for the packed rows"
        );
        // Q-74: the model's static bytes against what the device's live models leave.
        let reservation = reserve_logged(dev, &arrays)?;
        let buffer = |label: &str, len: usize| {
            dev.device().create_buffer(&BufferDescriptor {
                label: Some(label),
                size: buffer_size(len) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | TEST_READ,
                mapped_at_creation: false,
            })
        };
        let uniform = dev.device().create_buffer(&BufferDescriptor {
            label: Some("ftgsxr model uniform"),
            size: MODEL_UNIFORM_BYTES as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | TEST_READ,
            mapped_at_creation: false,
        });
        dev.queue().write_buffer(
            &uniform,
            0,
            &model_uniform(&arrays, conv, sh_range_words(dev)),
        );
        let len = |i| field(&arrays, i).len();
        let buffers = ModelBuffers {
            position: buffer("ftgsxr position", len(0)),
            time_center: buffer("ftgsxr time center", len(1)),
            duration: buffer("ftgsxr duration", len(2)),
            velocity: buffer("ftgsxr velocity", len(3)),
            scale: buffer("ftgsxr scale", len(4)),
            rotation: buffer("ftgsxr rotation", len(5)),
            opacity: buffer("ftgsxr opacity", len(6)),
            sh: buffer("ftgsxr sh", len(7)),
            uniform,
        };
        let model = GpuModel {
            buffers,
            index,
            tau0: arrays.tau0,
            tau1: arrays.tau1,
            sh_degree: arrays.sh_degree,
            conventions: conv.clone(),
            speed_p99,
            sh_range_bytes: sh_range_bytes(dev.limits()),
            n: arrays.n,
            reservation,
        };
        Ok(ModelUpload {
            arrays,
            model,
            field: 0,
            offset: 0,
        })
    }

    /// Writes the next chunks, at most `*budget` bytes (rounded down to whole words, the
    /// last chunk of a field padded to one), and takes them off `*budget`; `true` once
    /// every row is written.
    pub(crate) fn step(&mut self, dev: &SplatDevice, budget: &mut u64) -> bool {
        while self.field < FIELDS {
            let data = field(&self.arrays, self.field);
            let left = data.len() - self.offset;
            if left == 0 {
                self.field += 1;
                self.offset = 0;
                continue;
            }
            let room = (*budget / 4 * 4) as usize;
            if room == 0 {
                return false;
            }
            let take = left.min(room);
            let bytes = data.range(self.offset, self.offset + take);
            dev.queue().write_buffer(
                field_buffer(&self.model.buffers, self.field),
                self.offset as u64,
                &bytes,
            );
            *budget = budget.saturating_sub(bytes.len() as u64);
            self.offset += take;
        }
        true
    }

    /// The uploaded model, once `step` returned `true`.
    pub(crate) fn finish(self) -> GpuModel {
        debug_assert_eq!(self.field, FIELDS, "every chunk written");
        self.model
    }
}

/// The 99th percentile of the packed rows' speeds (Q-93), for `ModelUpload::begin`.
pub(crate) fn packed_speed_p99(arrays: &GpuArrays) -> f32 {
    speed_percentile_99(arrays)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ftgsxr_core::HalfFields;

    fn arrays_with_velocity(velocity: Vec<u8>, half: bool) -> GpuArrays {
        GpuArrays {
            n: 0,
            sh_degree: 0,
            tau0: 0.0,
            tau1: 1.0,
            single: false,
            half: HalfFields {
                velocity: half,
                scale: false,
                rotation: false,
                opacity: false,
                sh: false,
            },
            position: vec![],
            time_center: vec![],
            duration: vec![],
            velocity,
            scale: vec![],
            rotation: vec![],
            opacity: vec![],
            sh: vec![],
        }
    }

    /// Arrays of `n` rows, every field's bytes distinct and the half-precision fields of
    /// odd lengths, so a misplaced or dropped chunk shows.
    fn patterned_arrays(n: u32) -> GpuArrays {
        let f = |k: usize, s: f32| (0..k).map(|i| i as f32 * s + 0.25).collect::<Vec<f32>>();
        let b = |k: usize, s: u8| {
            (0..k)
                .map(|i| (i as u8).wrapping_mul(s).wrapping_add(1))
                .collect()
        };
        let n_ = n as usize;
        GpuArrays {
            n,
            sh_degree: 0,
            tau0: 0.0,
            tau1: 1.0,
            single: false,
            half: HalfFields::default(),
            position: f(3 * n_, 1.0),
            time_center: f(n_, 2.0),
            duration: f(n_, 3.0),
            velocity: b(6 * n_ + 2, 3),
            scale: b(6 * n_ + 2, 5),
            rotation: b(8 * n_, 7),
            opacity: b(2 * n_ + 2, 11),
            sh: b(6 * n_ + 2, 13),
        }
    }

    /// The temporal index of a single-timestamp model of `n` rows: every row persistent.
    fn persistent_index(n: u32) -> TemporalIndex {
        let model = ftgsxr_core::Model {
            position: vec![[0.0; 3]; n as usize],
            time_center: vec![0.0; n as usize],
            duration: vec![1.0; n as usize],
            velocity: vec![[0.0; 3]; n as usize],
            scale: vec![[1.0; 3]; n as usize],
            rotation: vec![[1.0, 0.0, 0.0, 0.0]; n as usize],
            opacity: vec![0.5; n as usize],
            sh: vec![0.0; 3 * n as usize],
            sh_degree: 0,
            time: ftgsxr_core::TimeSpan {
                tau0: 0.0,
                tau1: 0.0,
                time_unit: "s".into(),
            },
            conventions: RenderConventions::default(),
            producer: String::new(),
            identity: [0; 32],
            kind: ftgsxr_core::ModelKind::ExchangeFile,
        };
        TemporalIndex::build(&model).0
    }

    fn read_buffer(dev: &SplatDevice, buffer: &wgpu::Buffer) -> Vec<u8> {
        let staging = dev.device().create_buffer(&BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.device().create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        dev.queue().submit([enc.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("mapped"));
        dev.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("polled");
        let bytes = staging.get_mapped_range(..).to_vec();
        staging.unmap();
        bytes
    }

    /// §1.5, §2.4 rule 1: a chunked upload, a budget of 1000 bytes a frame, fills the
    /// buffers byte for byte as the one-call upload does, never writes more than its budget
    /// in a step, and reports done only after the last chunk.
    #[test]
    #[ignore = "gpu"]
    fn gpu_model_upload_in_chunks_equals_the_one_call_upload() {
        let dev = SplatDevice::headless().expect("an adapter");
        let n = 1000;
        let arrays = std::sync::Arc::new(patterned_arrays(n));
        let index = persistent_index(n);
        let conv = RenderConventions::default();
        let whole = GpuModel::upload(&dev, &arrays, index.clone(), &conv).expect("fits");
        let mut chunked =
            ModelUpload::begin(&dev, std::sync::Arc::clone(&arrays), index, &conv, 0.0)
                .expect("fits");
        let total: usize = (0..FIELDS).map(|i| field(&arrays, i).len()).sum();
        let mut steps = 0;
        loop {
            let mut budget = 1000;
            let done = chunked.step(&dev, &mut budget);
            assert!(budget <= 1000);
            steps += 1;
            if done {
                break;
            }
            assert!(budget < 4, "a step stops only when its budget is spent");
        }
        assert!(steps >= total / 1000, "{steps} steps for {total} bytes");
        let chunked = chunked.finish();
        for i in 0..FIELDS {
            assert_eq!(
                read_buffer(&dev, field_buffer(&chunked.buffers, i)),
                read_buffer(&dev, field_buffer(&whole.buffers, i)),
                "field {i}"
            );
        }
        assert_eq!(
            read_buffer(&dev, &chunked.buffers.uniform),
            read_buffer(&dev, &whole.buffers.uniform)
        );
        assert_eq!(dev.take_validation_messages(), Vec::<String>::new());
    }

    /// Q-74 (#174): the chunked upload reserves its bytes at `begin` like `GpuModel::upload`;
    /// an upload dropped before `finish` gives them back, and after `finish` the model
    /// holds them until it is dropped.
    #[test]
    #[ignore = "gpu"]
    fn gpu_model_upload_in_chunks_reserves_the_budget_until_dropped() {
        use crate::gpu::device::{BudgetSource, MemoryBudget};

        let mut dev = SplatDevice::headless().expect("an adapter");
        let n = 64;
        let arrays = std::sync::Arc::new(patterned_arrays(n));
        let index = persistent_index(n);
        let conv = RenderConventions::default();
        // `patterned_arrays(64)`: float32 position, time center and duration take
        // 12 + 4 + 4 bytes a row; velocity, scale and sh 6 a row plus 2, rotation 8 a row,
        // opacity 2 a row plus 2: 48 · 64 + 8 = 3 080 bytes.
        let one = 3080;
        let limit = one * 3 / 2;
        dev.set_memory_budget(MemoryBudget {
            bytes: limit,
            source: BudgetSource::Tier,
        });
        let begin = |dev: &SplatDevice| {
            ModelUpload::begin(
                dev,
                std::sync::Arc::clone(&arrays),
                index.clone(),
                &conv,
                0.0,
            )
        };
        let first = begin(&dev).expect("fits alone");
        assert_eq!(dev.model_bytes_in_use(), one);
        match begin(&dev) {
            Err(RenderError::Budget { needed, budget }) => {
                assert_eq!((needed, budget), (one, limit - one));
            }
            Err(e) => panic!("expected E-MEM-BUDGET, got {e:?}"),
            Ok(_) => panic!("two uploads over the budget together both began"),
        }
        assert_eq!(dev.model_bytes_in_use(), one);
        drop(first);
        assert_eq!(
            dev.model_bytes_in_use(),
            0,
            "an unfinished upload gives its bytes back"
        );
        let mut second = begin(&dev).expect("fits now");
        let mut frame = u64::MAX;
        while !second.step(&dev, &mut frame) {}
        let model = second.finish();
        assert_eq!(
            dev.model_bytes_in_use(),
            one,
            "the model keeps the reservation"
        );
        assert_eq!(model.memory_bytes(), one);
        drop(model);
        assert_eq!(dev.model_bytes_in_use(), 0);
        assert_eq!(dev.take_validation_messages(), Vec::<String>::new());
    }

    /// A writer into a shared buffer, for a captured log.
    #[derive(Clone)]
    struct Captured(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("the log buffer")
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Captured {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().expect("the log buffer").clone()).expect("UTF-8")
        }
    }

    /// The thread's log captured until the guard is dropped.
    fn capture_log() -> (Captured, bevy::log::tracing::subscriber::DefaultGuard) {
        use bevy::log::tracing_subscriber::{self, util::SubscriberInitExt};
        let log = Captured(std::sync::Arc::default());
        let writer = log.clone();
        let guard = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish()
            .set_default();
        (log, guard)
    }

    /// A device of its own, as `SplatDevice::headless` makes one, but whose largest storage
    /// binding is `bytes` (Q-369) and, when given, whose largest buffer is `max_buffer`.
    fn device_with_binding_limit(bytes: u64, max_buffer: Option<u64>) -> SplatDevice {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        desc.backends = wgpu::Backends::VULKAN;
        desc.flags = wgpu::InstanceFlags::from_build_config().with_env();
        let instance = wgpu::Instance::new(desc);
        let adapter =
            bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            }))
            .expect("an adapter");
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: bytes,
            max_buffer_size: max_buffer.unwrap_or(adapter.limits().max_buffer_size),
            ..adapter.limits()
        };
        let (device, queue) =
            bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("ftgsxr binding-limit test"),
                required_limits,
                ..Default::default()
            }))
            .expect("a device");
        SplatDevice::from_wgpu(device, queue, adapter.get_info())
    }

    /// Arrays of `n` degree-3 rows in float32, whose SH array (192 bytes a row) is the
    /// largest.
    fn degree3_arrays(n: u32) -> GpuArrays {
        let n_ = n as usize;
        GpuArrays {
            n,
            sh_degree: 3,
            sh: vec![0; 192 * n_],
            ..patterned_arrays(n)
        }
    }

    /// Binds `size` bytes of `buffer` from `offset` (all of it from there when `None`) as
    /// a read-only storage buffer of a compute bind group.
    fn bind(dev: &SplatDevice, buffer: &wgpu::Buffer, offset: u64, size: Option<u64>) {
        let layout = dev
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let _ = dev.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset,
                    size: size.and_then(std::num::NonZeroU64::new),
                }),
            }],
        });
    }

    /// `r` is E-MEM-BUDGET with `(needed, budget)`.
    fn refused(r: Result<(), RenderError>, figures: (u64, u64)) {
        match r {
            Err(RenderError::Budget { needed, budget }) => assert_eq!((needed, budget), figures),
            Err(e) => panic!("expected E-MEM-BUDGET, got {e:?}"),
            Ok(()) => panic!("a model over the device's bindings uploaded"),
        }
    }

    /// `arrays` refused with `figures` by the one-call and the chunked upload alike.
    fn refused_by_both(dev: &SplatDevice, arrays: GpuArrays, figures: (u64, u64)) {
        let n = arrays.n;
        let arrays = std::sync::Arc::new(arrays);
        let conv = RenderConventions::default();
        refused(
            GpuModel::upload(dev, &arrays, persistent_index(n), &conv).map(|_| ()),
            figures,
        );
        refused(
            ModelUpload::begin(dev, arrays, persistent_index(n), &conv, 0.0).map(|_| ()),
            figures,
        );
    }

    /// Q-376, Q-369 (#617, #642): on a device whose largest storage binding is 64 KiB (an
    /// SH range of 65 536 bytes at a 64-byte alignment), 1 000 degree-3 rows in float32 (a
    /// 192 000-byte SH buffer) upload and their three ranges bind without a validation
    /// message, while one range's bytes past the limit do not bind; 2 731 rows (524 352
    /// bytes, nine ranges) are refused with E-MEM-BUDGET, the SH buffer's bytes and the
    /// eight ranges' 524 288, by the one-call and the chunked upload alike, before anything
    /// is reserved, the refusal logging N, the field and both figures; a non-SH field over
    /// one binding (6 000 rows: a 72 000-byte position buffer, SH in half at degree 0) is
    /// refused with its bytes and the binding; and on a device whose largest buffer is
    /// 256 KiB, 1 500 degree-3 rows (288 000 bytes of SH, within eight ranges) are refused
    /// with the SH buffer's bytes and the 262 144-byte largest buffer.
    #[test]
    #[ignore = "gpu"]
    fn gpu_upload_refuses_a_field_over_the_binding_limit() {
        let (log, _subscriber) = capture_log();
        let limit = 65_536;
        let dev = device_with_binding_limit(limit, None);
        assert_eq!(dev.limits().max_storage_buffer_binding_size, limit);
        assert_eq!(sh_range_bytes(dev.limits()), limit);
        let conv = RenderConventions::default();

        let model = GpuModel::upload(&dev, &degree3_arrays(1000), persistent_index(1000), &conv)
            .expect("three SH ranges upload");
        assert_eq!(model.buffers.sh.size(), 192_000);
        assert_eq!(sh_ranges(192_000, model.sh_range_bytes), 3);
        for k in 0..3 {
            let offset = k * limit;
            bind(
                &dev,
                &model.buffers.sh,
                offset,
                Some(limit.min(192_000 - offset)),
            );
        }
        assert_eq!(dev.take_validation_messages(), Vec::<String>::new());
        // The control: the device does hold the lowered limit.
        bind(&dev, &model.buffers.sh, 0, Some(limit + 256));
        assert!(
            !dev.take_validation_messages().is_empty(),
            "a range over the limit fails to bind"
        );
        drop(model);

        refused_by_both(&dev, degree3_arrays(2731), (524_352, 524_288));
        assert_eq!(dev.model_bytes_in_use(), 0, "nothing reserved");
        let text = log.text();
        assert!(
            text.contains("N = 2731")
                && text.contains("sh buffer")
                && text.contains("524352 bytes")
                && text.contains("524288 bytes"),
            "{text}"
        );

        let position = GpuArrays {
            half: HalfFields {
                sh: true,
                ..HalfFields::default()
            },
            ..patterned_arrays(6000)
        };
        refused_by_both(&dev, position, (72_000, limit));
        assert_eq!(dev.model_bytes_in_use(), 0, "nothing reserved");
        let text = log.text();
        assert!(
            text.contains("N = 6000")
                && text.contains("position buffer")
                && text.contains("72000 bytes")
                && text.contains("65536-byte storage binding"),
            "{text}"
        );
        assert_eq!(dev.take_validation_messages(), Vec::<String>::new());

        let small = device_with_binding_limit(limit, Some(262_144));
        assert_eq!(small.limits().max_buffer_size, 262_144);
        refused_by_both(&small, degree3_arrays(1500), (288_000, 262_144));
        assert_eq!(small.model_bytes_in_use(), 0, "nothing reserved");
        assert!(
            log.text().contains("262144-byte largest buffer"),
            "{}",
            log.text()
        );
        assert_eq!(small.take_validation_messages(), Vec::<String>::new());
    }

    /// Q-117 (#189): an upload over the budget logs N, the bytes needed and the budget,
    /// which `RenderError::Budget` does not carry.
    #[test]
    #[ignore = "gpu"]
    fn gpu_reserve_logged_over_the_budget_logs_n_and_the_bytes() {
        use crate::gpu::device::{BudgetSource, MemoryBudget};

        let (log, _subscriber) = capture_log();

        let mut dev = SplatDevice::headless().expect("an adapter");
        let n = 64;
        let arrays = patterned_arrays(n);
        // 48 · 64 + 8 = 3 080 bytes, as in the chunked test above, against 1 000.
        dev.set_memory_budget(MemoryBudget {
            bytes: 1000,
            source: BudgetSource::Tier,
        });
        match GpuModel::upload(
            &dev,
            &arrays,
            persistent_index(n),
            &RenderConventions::default(),
        ) {
            Err(RenderError::Budget { needed, budget }) => {
                assert_eq!((needed, budget), (3080, 1000))
            }
            other => panic!("expected E-MEM-BUDGET, got {:?}", other.map(|_| ())),
        }
        let text = log.text();
        assert!(
            text.contains("N = 64") && text.contains("3080 bytes") && text.contains("1000 bytes"),
            "{text}"
        );
        assert_eq!(dev.take_validation_messages(), Vec::<String>::new());
    }

    /// Q-93: the nearest-rank 99th percentile of ‖velocity‖, from float32 or half rows.
    #[test]
    fn speed_percentile_is_the_nearest_rank_99th() {
        // Speeds 1..=200 along x: ⌈0.99 · 200⌉ = 198.
        let bytes: Vec<u8> = (1..=200)
            .flat_map(|i| [i as f32, 0.0, 0.0])
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(
            speed_percentile_99(&arrays_with_velocity(bytes, false)),
            198.0
        );
        // One row (3, 4, 0) in halves: speed 5.
        let halves: Vec<u8> = [0x4200u16, 0x4400, 0x0000]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        assert_eq!(
            speed_percentile_99(&arrays_with_velocity(halves, true)),
            5.0
        );
        assert_eq!(
            speed_percentile_99(&arrays_with_velocity(vec![], false)),
            0.0
        );
    }

    #[test]
    fn model_uniform_carries_the_conventions_in_order() {
        let conv = RenderConventions {
            alpha_max: 0.5,
            eps2d: 0.25,
            near: 2.0,
            far: 8.0,
            alpha_skip: 0.125,
            pixel_center_offset: 1.5,
            jacobian_clamp_margin: 0.75,
            screen_extent_max: 3.0,
            ..RenderConventions::default()
        };
        let arrays = GpuArrays {
            n: 7,
            sh_degree: 2,
            tau0: 0.0,
            tau1: 0.0,
            single: true,
            half: HalfFields {
                velocity: true,
                scale: false,
                rotation: true,
                opacity: false,
                sh: true,
            },
            position: vec![],
            time_center: vec![],
            duration: vec![],
            velocity: vec![],
            scale: vec![],
            rotation: vec![],
            opacity: vec![],
            sh: vec![],
        };
        let bytes = model_uniform(&arrays, &conv, 4096);
        let word = |i: usize| <[u8; 4]>::try_from(&bytes[4 * i..4 * i + 4]).unwrap();
        let f = |i| f32::from_le_bytes(word(i));
        let u = |i| u32::from_le_bytes(word(i));
        assert_eq!(
            [f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)],
            [0.5, 0.25, 2.0, 8.0, 0.125, 1.5, 0.75, 3.0]
        );
        assert_eq!([u(8), u(9), u(10), u(11)], [7, 2, 0b10101, 1]);
        // Q-376: W, then padding.
        assert_eq!([u(12), u(13), u(14), u(15)], [4096, 0, 0, 0]);
    }
}
