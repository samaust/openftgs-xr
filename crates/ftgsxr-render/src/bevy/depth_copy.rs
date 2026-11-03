// SPDX-License-Identifier: Apache-2.0

//! The view's depth copied at the splat layer's size below render scale 1 (Q-164,
//! `shaders/depth_copy.wgsl`), so the layer's draw keeps Q-87's depth test.

use wgpu::util::DeviceExt;

/// `depth_copy.wgsl`.
const DEPTH_COPY_WGSL: &str = include_str!("shaders/depth_copy.wgsl");

/// The copy's pipeline into `format` (the view's depth format) and its layout.
pub(crate) struct DepthCopy {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl DepthCopy {
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> DepthCopy {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ftgsxr depth_copy.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DEPTH_COPY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ftgsxr depth copy"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ftgsxr depth copy"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ftgsxr depth copy"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_copy"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_copy"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        });
        DepthCopy { layout, pipeline }
    }

    /// One pass writing `view_depth`, resampled, into `target` of `size` (the layer's).
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        view_depth: &wgpu::TextureView,
        target: &wgpu::TextureView,
        size: [u32; 2],
    ) {
        let mut block = [0u8; 16];
        block[..4].copy_from_slice(&size[0].to_le_bytes());
        block[4..8].copy_from_slice(&size[1].to_le_bytes());
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ftgsxr depth copy"),
            contents: &block,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ftgsxr depth copy"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ftgsxr depth copy"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: target,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_copy_wgsl_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(DEPTH_COPY_WGSL)
            .unwrap_or_else(|e| panic!("parses: {}", e.emit_to_string(DEPTH_COPY_WGSL)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validates");
    }
}
