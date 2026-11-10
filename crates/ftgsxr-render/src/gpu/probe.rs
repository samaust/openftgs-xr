// SPDX-License-Identifier: Apache-2.0

//! The layer probe of E14's exposure match (§2.1 `LayerSums`, `probe.wgsl`, R-21): a compute
//! pass that sums each cell of a 64 × 48 grid over a splat layer, (Σ α r_lin, Σ α g_lin,
//! Σ α b_lin, Σ α), into a small storage buffer, which an adapter reads back (D-34).

use ftgsxr_core::UVec2;

use super::error::RenderError;

/// `probe.wgsl`'s source.
pub(crate) const PROBE_WGSL: &str = include_str!("shaders/probe.wgsl");
/// The grid's columns and rows (E14).
pub(crate) const GRID: UVec2 = UVec2::new(64, 48);
/// The grid's bytes: one `vec4<f32>` per cell.
pub(crate) const CELLS_BYTES: u64 = (GRID.x * GRID.y * 16) as u64;

/// E14's exposure match, the layer probe's result: per cell of the 64 × 48 grid over the
/// layer, row-major from the top-left, (Σ α r_lin, Σ α g_lin, Σ α b_lin, Σ α) over its
/// pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerSums {
    /// The cells, row-major from the top-left.
    pub cells: Vec<[f32; 4]>,
    /// The grid's columns (64).
    pub width: u32,
    /// The grid's rows (48).
    pub height: u32,
    /// The layer's size in pixels.
    pub layer: UVec2,
}

impl LayerSums {
    /// The grid read back as little-endian floats (`CELLS_BYTES` of them), of a layer of
    /// `layer` pixels.
    pub(crate) fn from_bytes(bytes: &[u8], layer: UVec2) -> LayerSums {
        let cells = bytes[..CELLS_BYTES as usize]
            .as_chunks::<16>()
            .0
            .iter()
            .map(|c| {
                std::array::from_fn(|k| {
                    f32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])
                })
            })
            .collect();
        LayerSums {
            cells,
            width: GRID.x,
            height: GRID.y,
            layer,
        }
    }
}

/// The compiled probe pass.
pub(crate) struct ProbePass {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl ProbePass {
    /// Compiles `probe.wgsl`; E-GPU-SHADER when it does not.
    pub(crate) fn new(device: &wgpu::Device) -> Result<ProbePass, RenderError> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ftgsxr probe.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PROBE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ftgsxr probe"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ftgsxr probe"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ftgsxr probe"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = bevy::tasks::block_on(scope.pop()) {
            return Err(RenderError::Shader(e.to_string()));
        }
        Ok(ProbePass { layout, pipeline })
    }

    /// The probe of `layer` (a 16-bit float layer with texture-binding usage) into `cells`
    /// (storage usage, at least `CELLS_BYTES`), every cell written.
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        layer: &wgpu::TextureView,
        cells: &wgpu::Buffer,
    ) {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ftgsxr probe"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(layer),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cells.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ftgsxr layer probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(GRID.x, GRID.y, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_bytes_reads_the_cells_row_major() {
        let mut bytes = vec![0u8; CELLS_BYTES as usize];
        // Cell (1, 0) and the last cell.
        for (k, v) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
            bytes[16 + 4 * k..][..4].copy_from_slice(&v.to_le_bytes());
            bytes[CELLS_BYTES as usize - 16 + 4 * k..][..4].copy_from_slice(&(-v).to_le_bytes());
        }
        let sums = LayerSums::from_bytes(&bytes, UVec2::new(200, 150));
        assert_eq!(sums.cells.len(), 64 * 48);
        assert_eq!(sums.cells[1], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sums.cells[64 * 48 - 1], [-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(
            (sums.width, sums.height, sums.layer),
            (64, 48, UVec2::new(200, 150))
        );
    }
}
