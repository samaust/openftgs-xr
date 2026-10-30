// SPDX-License-Identifier: Apache-2.0

//! `GpuModel`: one model's structure-of-arrays buffers (§2.4 GPU data layout, Q-22), its
//! temporal index (E3, D-10) and its uniform block of E6 and E7 constants (D-39).

use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
use ftgsxr_core::{GpuArrays, RenderConventions, TemporalIndex};

use super::device::SplatDevice;
use super::error::RenderError;
use super::readback::half_to_f32;

/// The model's uniform block: every E6 and E7 constant of its `render_conventions` (D-39,
/// §2.6: never a literal in a shader) and the layout facts the shaders read. Little-endian
/// 32-bit words, 48 bytes, in this order (byte offsets):
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
///
/// `transmittance_stop` is not applied (D-7), and the background is the view's (E7).
pub(crate) const MODEL_UNIFORM_BYTES: usize = 48;

fn model_uniform(arrays: &GpuArrays, conv: &RenderConventions) -> [u8; MODEL_UNIFORM_BYTES] {
    let h = arrays.half;
    let flags = u32::from(h.velocity)
        | u32::from(h.scale) << 1
        | u32::from(h.rotation) << 2
        | u32::from(h.opacity) << 3
        | u32::from(h.sh) << 4;
    let words: [[u8; 4]; 12] = [
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
    /// L′, the loaded SH degree.
    pub(crate) sh_degree: u8,
    /// The model's E6 and E7 constants (D-39), which the draw reads too.
    pub(crate) conventions: RenderConventions,
    /// The 99th percentile of the rows' speeds ‖velocity‖ in model units per unit of τ,
    /// the scale of the speed colormap (§1.4 stage 6). // Q-93
    pub(crate) speed_p99: f32,
    n: u32,
    bytes: u64,
}

/// The bytes `arrays` holds: §1.12's static bytes per primitive times N (Q-23).
fn static_bytes(a: &GpuArrays) -> u64 {
    let f32s = (a.position.len() + a.time_center.len() + a.duration.len()) * 4;
    let packed = a.velocity.len() + a.scale.len() + a.rotation.len() + a.opacity.len() + a.sh.len();
    (f32s + packed) as u64
}

/// A storage buffer holding `bytes` padded with zeros to whole words (at least one),
/// written through the mapping at creation.
fn storage_buffer(dev: &SplatDevice, label: &str, mut bytes: Vec<u8>) -> wgpu::Buffer {
    let size = bytes.len().div_ceil(4).max(1) * 4;
    bytes.resize(size, 0);
    let buffer = dev.device().create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
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

impl GpuModel {
    /// Uploads `arrays` (packed by `pack_for_gpu` in the order `index` was built for) with
    /// the model's conventions in its uniform block (D-39).
    ///
    /// The static bytes are checked against the device's memory budget before anything is
    /// allocated on the GPU (§2.4 rule 4, §1.2 load option 4, D-13): a model that does not
    /// fit is `RenderError::Budget`.
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
        // Q-74: the model's own static bytes against the device's budget, per model.
        let bytes = static_bytes(arrays);
        let budget = dev.memory_budget().bytes;
        if bytes > budget {
            return Err(RenderError::Budget {
                needed: bytes,
                budget,
            });
        }
        let uniform = dev.device().create_buffer(&BufferDescriptor {
            label: Some("ftgsxr model uniform"),
            size: MODEL_UNIFORM_BYTES as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue()
            .write_buffer(&uniform, 0, &model_uniform(arrays, conv));
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
            sh_degree: arrays.sh_degree,
            conventions: conv.clone(),
            speed_p99: speed_percentile_99(arrays),
            n: arrays.n,
            bytes,
        })
    }

    /// The model's static GPU bytes: §1.12's bytes per primitive times N, as `GpuArrays`
    /// holds them (Q-23); the word padding at the ends of half-precision buffers (at most
    /// 2 bytes each, Q-22) and the 48-byte uniform block are not counted.
    pub fn memory_bytes(&self) -> u64 {
        self.bytes
    }

    /// N, the rows uploaded.
    pub fn n(&self) -> u32 {
        self.n
    }
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
        let bytes = model_uniform(&arrays, &conv);
        let word = |i: usize| <[u8; 4]>::try_from(&bytes[4 * i..4 * i + 4]).unwrap();
        let f = |i| f32::from_le_bytes(word(i));
        let u = |i| u32::from_le_bytes(word(i));
        assert_eq!(
            [f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)],
            [0.5, 0.25, 2.0, 8.0, 0.125, 1.5, 0.75, 3.0]
        );
        assert_eq!([u(8), u(9), u(10), u(11)], [7, 2, 0b10101, 1]);
    }
}
