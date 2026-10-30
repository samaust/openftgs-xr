// SPDX-License-Identifier: Apache-2.0

//! The Bevy plugin adapter (§2.1 `ftgsxr_render::bevy`): the `SplatAsset` and its load,
//! the components, resources and events of a scene of splat models, and the two render
//! nodes that run `gpu`'s passes inside each `SplatCamera`'s 3D pass (§1.4, §2.4 Frame
//! loop hooks, D-2, D-4, D-5, D-6).
//!
//! The main world loads (on the async compute task pool, one task per load, §2.4
//! Threads), uploads (at most 64 MB per frame, §1.5, §2.4 rule 1) and reports
//! (`SplatEvent`, `SplatStats`); the render world extracts the instances and cameras,
//! keeps one `FrameBuffers` per view and encodes the passes (`render`).

mod asset;
mod camera;
mod load;
mod render;

use ::bevy::app::{App, First, Plugin, PostUpdate, Update};
use ::bevy::asset::{AssetApp, Handle};
use ::bevy::ecs::prelude::*;
use ::bevy::image::Image;
use ::bevy::render::RenderApp;
use ftgsxr_core::{Camera, GpuLayoutOptions};

pub use asset::{SplatAsset, load_splat_asset};
pub use load::SplatLoad;

use crate::gpu::{
    ColorTransform, Foveation, FrameStats, Overlays, QualityTier, RenderMode, SortMode,
};

/// The plugin adapter (§2.1): adds the asset, the loads and uploads, the statistics and
/// the two render nodes, for the tier `tier` (§1.12).
pub struct SplatPlugin {
    /// The quality tier every camera and load of the app uses (§1.12).
    pub tier: QualityTier,
}

/// One instance of a model (§1.5 Instances), drawn by every `SplatCamera`; with
/// `Transform` (uniform scale: P_m's s_m is the x scale, E5) and `Visibility`.
#[derive(Component, Clone, Debug)]
pub struct SplatModel(pub Handle<SplatAsset>);

/// τ_m, the raw time the instance is evaluated at (E9), and whether the clock is past
/// the model's span (§1.6). An instance without one is drawn at its model's τ₀ (Q-99).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SplatTime {
    /// τ_m in the model's time unit.
    pub tau: f64,
    /// Whether τ_m lies outside \[τ₀, τ₁\] (the HUD's extrapolation flag).
    pub extrapolating: bool,
}

/// The instance's render mode (§1.4); `RenderMode::Splats` without one.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SplatMode(pub RenderMode);

/// The instance's color transform (E14); the identity without one.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SplatTransform(pub ColorTransform);

/// The instance's debug overlays (§1.4); none without one. Carried, not drawn yet (PRD #76).
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct SplatOverlays(pub Overlays);

/// Marks the cameras that draw splats, with the camera's sort mode (D-4) and the space
/// its output is written in (D-6).
///
/// The plugin makes a splat camera follow D-6 when the component is added: MSAA off, no
/// tonemapping, cleared transparent (Q-100).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SplatCamera {
    /// Per view or shared center (D-4).
    pub sort: SortMode,
    /// Where the layer goes (D-6).
    pub output: OutputSpace,
}

/// The space a `SplatCamera`'s splats are written in (D-6, §1.4 stage 7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutputSpace {
    /// The layer through the resolve: decoded to linear light and composited over the view
    /// target, over `background` (file space, Q-94) when given (VR mode), else keeping the
    /// layer's alpha (passthrough, D-14).
    LinearOver {
        /// The file-space background composited first (E7's c_bg), or none.
        background: Option<[f32; 3]>,
    },
    /// No resolve: the splats are blended, file-encoded and premultiplied, straight into
    /// the view target, which then holds the layer of a headless render (§1.7); exact on a
    /// 16-bit float target (Q-100).
    FileEncoded,
}

/// One environment depth image with its camera and planes (E13).
#[derive(Clone, Debug)]
pub struct DepthFrame {
    /// The depth image.
    pub image: Handle<Image>,
    /// The camera the image was taken from.
    pub camera: Camera,
    /// The near plane of its depth values.
    pub near: f32,
    /// The far plane of its depth values.
    pub far: f32,
    /// Frames since the image arrived.
    pub age_frames: u32,
}

/// The environment depth E13 occludes with, one frame per view (the runtime's left and
/// right). Declared for M-6; occlusion stays off until then.
#[derive(Resource, Clone, Debug)]
pub struct SplatOcclusion {
    /// One depth frame per view.
    pub frames: Vec<DepthFrame>,
    /// Soft occlusion (E13's smoothstep) or hard.
    pub soft: bool,
    /// E13's δ in metres.
    pub delta_m: f32,
}

/// The latest statistics of the splat cameras' frames, two frames late (§1.4 stage 8,
/// §1.13); `None` until the first arrives.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct SplatStats(pub Option<FrameStats>);

/// The governor's copy of the tier (E16): the knobs it steps.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct SplatTierState {
    /// The tier the plugin was built with.
    pub tier: QualityTier,
    /// The render scale (§1.12).
    pub render_scale: f32,
    /// The SH degree cap the next loads apply (§1.2 load option 1).
    pub sh_cap: u8,
    /// The sort mode (D-4).
    pub sort_mode: SortMode,
    /// The foveation level (§1.12).
    pub foveation: Foveation,
}

/// What the plugin reports (§1.5, §1.13), triggered for observers.
#[derive(Event, Clone, Debug, PartialEq)]
pub enum SplatEvent {
    /// The instance's model finished uploading; `bytes` are its static GPU bytes (Q-23).
    Uploaded {
        /// The `SplatModel` (or `SplatLoad`) entity.
        entity: Entity,
        /// The model's static GPU bytes.
        bytes: u64,
    },
    /// The instance's load or upload failed; `error` is `<code>: <user text>` (§1.15).
    Failed {
        /// The `SplatModel` (or `SplatLoad`) entity.
        entity: Entity,
        /// The error's code and text.
        error: String,
    },
    /// A frame dropped records for want of capacity (D-34), two frames late.
    Overflow {
        /// The rows that passed the cull, at most C.
        visible: u32,
        /// C.
        capacity: u32,
    },
}

impl SplatTierState {
    fn of(tier: &QualityTier) -> SplatTierState {
        SplatTierState {
            tier: tier.clone(),
            render_scale: tier.render_scale,
            // A cap of 3 keeps every degree (§1.2 load option 1).
            sh_cap: tier.layout.sh_cap.unwrap_or(3),
            sort_mode: tier.sort_mode,
            foveation: tier.foveation,
        }
    }

    /// The load options of the next load: the tier's, with the governor's SH cap.
    pub(crate) fn load_options(&self) -> GpuLayoutOptions {
        GpuLayoutOptions {
            sh_cap: Some(self.sh_cap),
            ..self.tier.layout
        }
    }
}

impl Plugin for SplatPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<SplatAsset>()
            .insert_resource(SplatTierState::of(&self.tier))
            .init_resource::<SplatStats>()
            .init_resource::<load::GpuModels>()
            .init_resource::<render::StatsInbox>()
            .add_observer(load::make_camera_follow_d6)
            .add_systems(First, (load::ensure_device, render::publish_stats))
            .add_systems(Update, (load::start_loads, load::finish_loads).chain())
            .add_systems(PostUpdate, load::upload_models);
        let inbox = app.world().resource::<render::StatsInbox>().clone();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render::build(render_app, self.tier.clone(), inbox);
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render::finish(render_app);
        }
    }
}
