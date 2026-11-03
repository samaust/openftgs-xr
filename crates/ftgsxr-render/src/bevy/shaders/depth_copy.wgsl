// SPDX-License-Identifier: Apache-2.0

// depth_copy.wgsl: the view's depth copied at the splat layer's size below render scale 1
// (Q-164, §1.12), so the layer's draw keeps the depth test against the opaque pass (§1.4
// stage 5, Q-87). One full-screen triangle into a depth target of the layer's size; each
// texel takes the view's depth at the view pixel holding the texel's center (Q-171).

@group(0) @binding(0) var view_depth: texture_depth_2d;

@vertex
fn vs_copy(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}

struct Out {
    @builtin(frag_depth) depth: f32,
}

// Q-171
@fragment
fn fs_copy(@builtin(position) position: vec4<f32>) -> Out {
    let view_size = vec2<f32>(textureDimensions(view_depth));
    // The layer's size is the target's, the view's size scaled; its texel centers are at
    // position.xy, mapped onto the view's pixels.
    let layer_size = vec2<f32>(copy.layer_size);
    let p = min(vec2<u32>(position.xy * view_size / layer_size), vec2<u32>(view_size) - 1u);
    var out: Out;
    out.depth = textureLoad(view_depth, p, 0);
    return out;
}

struct Copy {
    layer_size: vec2<u32>,
}

@group(0) @binding(1) var<uniform> copy: Copy;
