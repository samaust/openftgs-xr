// SPDX-License-Identifier: Apache-2.0

//! Reading the frame buffers back (Q-69): a blocking test and diagnostics hook that the
//! frame loop never calls (§2.4 rule 1, P-5).

use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
use ftgsxr_core::Vec2;

use super::device::SplatDevice;
use super::frame::{COUNTER_WORDS, FrameBuffers, RecordMode};

/// One per-view record as the cull pass wrote it (§1.4 stage 2), decoded (Q-69).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordView {
    /// μ′ in pixels.
    pub mean_px: Vec2,
    /// The conic Σ′⁻¹ as (a, b, c) (Q-21).
    pub conic: [f32; 3],
    /// The radii (r_x, r_y) in whole pixels.
    pub half_extent: Vec2,
    /// z_c, the camera-space depth (E8's input).
    pub depth: f32,
    /// The premultiplied color c·σ_eff and σ_eff, from their half floats.
    pub color: [f32; 4],
}

/// What a frame's buffers hold after `encode_prepare` (Q-69).
#[derive(Clone, Debug, PartialEq)]
pub struct FrameContents {
    /// Splat records, after stage 3's clamp to C.
    pub visible_splats: u32,
    /// Center records, after the clamp (Q-78).
    pub visible_centers: u32,
    /// The overflow flag (D-34).
    pub overflow: bool,
    /// The rows that passed E3's exact test σ_eff ≥ α_skip in the cull, counted once per
    /// row and instance (Q-79).
    pub active_rows: u32,
    /// The per-view records in slot order; empty in shared-center mode (decoded from M-8).
    pub records: Vec<RecordView>,
    /// The payloads in draw order (the slots, once the sort has run).
    pub sorted: Vec<u32>,
    /// The sort passes' dispatch arguments from stage 3.
    pub sort_dispatch: [u32; 3],
}

/// IEEE binary16 to float32, exactly.
pub(crate) fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    match exp {
        0 => sign * frac * (2.0f32).powi(-24),
        31 if frac == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + frac / 1024.0) * (2.0f32).powi(exp - 15),
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// A per-view record from its 12 words (`cull.wgsl`).
fn decode_view_record(w: &[u32]) -> RecordView {
    let f = |i: usize| f32::from_bits(w[i]);
    let h = |i: usize| {
        [
            half_to_f32((w[i] & 0xffff) as u16),
            half_to_f32((w[i] >> 16) as u16),
        ]
    };
    let [r, g] = h(8);
    let [b, a] = h(9);
    RecordView {
        mean_px: Vec2::new(f(0), f(1)),
        conic: [f(2), f(3), f(4)],
        depth: f(5),
        half_extent: Vec2::new(f(6), f(7)),
        color: [r, g, b, a],
    }
}

/// Copies `[0, size)` of each `(source, size)` into one mappable buffer, submits, waits and
/// returns the bytes of each.
pub(crate) fn read_buffers(dev: &SplatDevice, sources: &[(&wgpu::Buffer, u64)]) -> Vec<Vec<u8>> {
    let total: u64 = sources.iter().map(|(_, s)| s.div_ceil(4) * 4).sum();
    let staging = dev.device().create_buffer(&BufferDescriptor {
        label: Some("ftgsxr read back"),
        size: total.max(4),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = dev
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ftgsxr read back"),
        });
    let mut at = 0;
    let mut spans = Vec::new();
    for &(buffer, size) in sources {
        let size = size.div_ceil(4) * 4;
        if size > 0 {
            enc.copy_buffer_to_buffer(buffer, 0, &staging, at, size);
        }
        spans.push((at, size));
        at += size;
    }
    dev.queue().submit([enc.finish()]);
    // `read_back` returns `FrameContents` (Q-69), with no error to carry a lost device, and
    // is a test and diagnostics hook: a lost device panics here rather than as
    // `RenderError::DeviceLost`.
    staging.map_async(wgpu::MapMode::Read, .., |r| {
        r.expect("invariant: the read-back buffer maps on a live device (Q-69)");
    });
    dev.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("invariant: a live device completes the read back (Q-69)");
    let out = {
        let view = staging.get_mapped_range(..);
        spans
            .iter()
            .map(|&(a, s)| view[a as usize..(a + s) as usize].to_vec())
            .collect()
    };
    staging.unmap();
    out
}

impl FrameBuffers {
    /// The frame's counts, overflow flag, records and payloads, decoded (Q-69). It maps
    /// and waits: a test and diagnostics hook, never called in the frame loop (P-5).
    /// Submit the frame's encoder first.
    ///
    /// # Panics
    ///
    /// When the device is lost: the signature (Q-69) has no error to carry it.
    pub fn read_back(&self, dev: &SplatDevice) -> FrameContents {
        // Q-69
        let set = &self.set;
        let head = read_buffers(
            dev,
            &[(&set.indirect, COUNTER_WORDS * 4), (&set.dispatch, 12)],
        );
        let counters = words(&head[0]);
        let dispatch = words(&head[1]);
        let splats = counters[1];
        let stride = self.mode.bytes();
        let body = read_buffers(
            dev,
            &[
                (&set.records, u64::from(splats) * stride),
                (&set.payloads[0], u64::from(splats) * 4),
            ],
        );
        let records = match self.mode {
            RecordMode::PerView => words(&body[0])
                .as_chunks::<12>()
                .0
                .iter()
                .map(|w| decode_view_record(w))
                .collect(),
            RecordMode::Shared | RecordMode::SharedHalf => Vec::new(),
        };
        FrameContents {
            visible_splats: splats,
            visible_centers: counters[5],
            overflow: counters[8] != 0,
            active_rows: counters[9],
            records,
            sorted: words(&body[1]),
            sort_dispatch: [dispatch[0], dispatch[1], dispatch[2]],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_to_f32_decodes_binary16() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x3555), 0.333_251_95);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(half_to_f32(0x7bff), 65504.0);
        assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
        assert!(half_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn decode_view_record_reads_the_layout_of_cull_wgsl() {
        let mut w = [0u32; 12];
        for (i, v) in [10.5f32, -3.0, 0.25, -0.125, 0.5, 4.0, 7.0, 9.0]
            .iter()
            .enumerate()
        {
            w[i] = v.to_bits();
        }
        // (0.5, 1.0) and (2.0, 0.25) as half pairs, the first in the low bits.
        w[8] = 0x3800 | (0x3c00 << 16);
        w[9] = 0x4000 | (0x3400 << 16);
        let r = decode_view_record(&w);
        assert_eq!(r.mean_px, Vec2::new(10.5, -3.0));
        assert_eq!(r.conic, [0.25, -0.125, 0.5]);
        assert_eq!(r.depth, 4.0);
        assert_eq!(r.half_extent, Vec2::new(7.0, 9.0));
        assert_eq!(r.color, [0.5, 1.0, 2.0, 0.25]);
    }
}
