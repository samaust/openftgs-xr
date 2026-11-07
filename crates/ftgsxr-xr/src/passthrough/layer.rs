// SPDX-License-Identifier: Apache-2.0

//! The passthrough layer in the render world, where bevy\_mod\_openxr keeps its layer
//! list (§1.9, D-14; PRD #565's "The passthrough adapter").
//!
//! At the session's first render frame the passthrough and its RECONSTRUCTION layer are
//! created through the crate's `create_passthrough` helper, not its plugin, which would
//! submit the layer unconditionally, and the layer is paused at once. In passthrough mode
//! the layer is resumed and a provider of the crate's `PassthroughLayer` is inserted
//! first in `OxrRenderLayers`, under the projection layer, which keeps its
//! `BLEND_TEXTURE_SOURCE_ALPHA` flag; in VR mode the layer is paused and the provider
//! removed. The list holds trait objects, so `Record` keeps the address of the provider
//! inserted here. The style goes through the raw table's xrPassthroughLayerSetStyleFB at
//! each change (Q-337). The session stays OPAQUE (D-14).
//!
//! openxr 0.21.1's `Passthrough::start` calls xrPassthroughPauseFB, and its
//! `Passthrough` gives no raw handle to call xrPassthroughStartFB with, so the
//! passthrough feature is created running (`IS_RUNNING_AT_CREATION`, which the crate
//! passes to the layer too) and stays running for the session; only the layer is paused
//! and resumed. The layer is paused in the frame it is created, before any frame could
//! submit it.

use bevy::app::App;
use bevy::ecs::world::World;
use bevy::log::{error, info};
use bevy::prelude::{Commands, IntoScheduleConfigs, Res, ResMut, Resource, resource_exists};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy_mod_openxr::features::fb_passthrough::create_passthrough;
use bevy_mod_openxr::layer_builder::{CompositionLayer, LayerProvider, PassthroughLayer};
use bevy_mod_openxr::resources::{OxrPassthrough, OxrPassthroughLayerFB, OxrRenderLayers};
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_xr::session::XrPreDestroySession;
use openxr::sys;

use super::{EnvironmentMode, LayerRequest, PassthroughStyle, style_map};

/// The provider inserted in the layer list: the crate's `PassthroughLayer`, behind a type
/// with a size, so its box has an address of its own to find it by.
struct Provider {
    _tag: u8,
}

impl LayerProvider for Provider {
    fn get<'a>(&'a self, world: &'a World) -> Option<Box<dyn CompositionLayer<'a> + 'a>> {
        PassthroughLayer.get(world)
    }
}

/// The address of a provider in the list, to compare with `Record::inserted`.
fn address(p: &(dyn LayerProvider + Send + Sync)) -> usize {
    std::ptr::from_ref(p).cast::<()>() as usize
}

/// What this module did to the session's layer (§1.9; Q-346's record, which Dv-6 reads).
#[derive(Resource, Debug, Default)]
pub(crate) struct Record {
    /// Creation was attempted in this session.
    tried: bool,
    /// The address of the provider inserted in `OxrRenderLayers`, while it is there.
    inserted: Option<usize>,
    /// Whether the layer is resumed.
    pub(crate) resumed: bool,
    /// The style last sent, and whether the runtime accepted it.
    pub(crate) style: Option<(PassthroughStyle, bool)>,
    /// The pause, resume and style calls that failed in this session.
    pub(crate) failures: u32,
}

impl Record {
    /// Whether `layers` holds this module's provider first, and only it and one more
    /// (the projection layer).
    #[cfg_attr(not(any(test, target_os = "android")), allow(dead_code))]
    pub(crate) fn first_of(&self, layers: &OxrRenderLayers) -> bool {
        self.inserted
            .is_some_and(|a| layers.first().is_some_and(|p| address(p.as_ref()) == a))
    }

    /// Whether `layers` holds this module's provider anywhere.
    pub(crate) fn in_list(&self, layers: &OxrRenderLayers) -> bool {
        self.inserted
            .is_some_and(|a| layers.iter().any(|p| address(p.as_ref()) == a))
    }
}

/// Adds the layer's render-world systems.
pub(super) fn add(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<Record>()
        .add_systems(XrPreDestroySession, destroy)
        .add_systems(
            Render,
            apply
                .in_set(RenderSystems::Prepare)
                .run_if(resource_exists::<OxrSession>)
                .run_if(resource_exists::<OxrRenderLayers>),
        );
}

/// Creates the layer once per session, then follows the request: the mode (resume and
/// insert, or pause and remove) and the style.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply(
    request: Res<LayerRequest>,
    session: Res<OxrSession>,
    passthrough: Option<Res<OxrPassthrough>>,
    layer: Option<Res<OxrPassthroughLayerFB>>,
    mut layers: ResMut<OxrRenderLayers>,
    mut record: ResMut<Record>,
    mut commands: Commands,
) {
    let (Some(_passthrough), Some(layer)) = (passthrough, layer) else {
        if !record.tried {
            record.tried = true;
            create(&session, &mut commands);
        }
        return;
    };
    let table = &session.instance().exts().fb_passthrough;
    if record.style.map(|(s, _)| s) != Some(request.style)
        && let Some(table) = table.as_ref()
    {
        let ok = set_style(table, layer.as_raw(), request.style);
        if !ok {
            record.failures += 1;
        }
        record.style = Some((request.style, ok));
    }
    match request.mode {
        EnvironmentMode::Passthrough => {
            if !record.resumed {
                match layer.resume() {
                    Ok(()) => record.resumed = true,
                    Err(e) => {
                        record.failures += 1;
                        error!("passthrough: resuming the layer failed: {e}");
                        return;
                    }
                }
            }
            insert(&mut layers, &mut record);
        }
        EnvironmentMode::Vr => {
            remove(&mut layers, &mut record);
            if record.resumed {
                pause(&layer, &mut record);
            }
        }
    }
}

/// Creates the passthrough running and its RECONSTRUCTION layer, paused at once (the
/// module's comment says why), as the crate's resources, which its `PassthroughLayer`
/// reads.
fn create(session: &OxrSession, commands: &mut Commands) {
    match create_passthrough(
        session,
        openxr::PassthroughFlagsFB::IS_RUNNING_AT_CREATION,
        openxr::PassthroughLayerPurposeFB::RECONSTRUCTION,
    ) {
        Ok((passthrough, layer)) => {
            if let Err(e) = layer.pause() {
                error!("passthrough: pausing the new layer failed: {e}");
            }
            info!("passthrough: created, the layer paused");
            commands.insert_resource(passthrough);
            commands.insert_resource(layer);
        }
        Err(e) => error!("passthrough: creating the layer failed: {e}"),
    }
}

/// Inserts this module's provider first in `layers`, unless it is there.
fn insert(layers: &mut OxrRenderLayers, record: &mut Record) {
    if record.in_list(layers) {
        return;
    }
    let provider: Box<dyn LayerProvider + Send + Sync> = Box::new(Provider { _tag: 0 });
    record.inserted = Some(address(provider.as_ref()));
    layers.insert(0, provider);
    info!("passthrough: the layer is submitted");
}

/// Removes this module's provider from `layers`, if it is there.
fn remove(layers: &mut OxrRenderLayers, record: &mut Record) {
    if let Some(a) = record.inserted.take() {
        layers.retain(|p| address(p.as_ref()) != a);
        info!("passthrough: the layer is removed");
    }
}

/// Pauses the layer.
fn pause(layer: &OxrPassthroughLayerFB, record: &mut Record) {
    match layer.pause() {
        Ok(()) => record.resumed = false,
        Err(e) => {
            record.failures += 1;
            error!("passthrough: pausing the layer failed: {e}");
        }
    }
}

/// Sends `style` to `layer` (Q-337): full opacity, no edges, and the brightness, contrast
/// and saturation structure alone in the chain (the registry page makes the chained
/// color maps mutually exclusive). Whether the runtime accepted it.
// Q-337
fn set_style(
    table: &openxr::raw::PassthroughFB,
    layer: sys::PassthroughLayerFB,
    style: PassthroughStyle,
) -> bool {
    let (brightness, contrast, saturation) = style_map(style);
    let bcs = sys::PassthroughBrightnessContrastSaturationFB {
        ty: sys::PassthroughBrightnessContrastSaturationFB::TYPE,
        next: std::ptr::null(),
        brightness,
        contrast,
        saturation,
    };
    let info = sys::PassthroughStyleFB {
        ty: sys::PassthroughStyleFB::TYPE,
        next: std::ptr::from_ref(&bcs).cast(),
        texture_opacity_factor: 1.0,
        edge_color: sys::Color4f {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        },
    };
    // SAFETY: `table` is the instance's loaded XR_FB_passthrough table and `layer` a live
    // layer of a session of that instance (the resource is removed before the session is
    // destroyed); `info` and the `bcs` its `next` points to are valid, typed structures
    // that outlive the call, which only reads them.
    let r = unsafe { (table.passthrough_layer_set_style)(layer, &info) };
    if r.into_raw() < 0 {
        error!("passthrough: setting the style failed: {r:?}");
        false
    } else {
        info!(
            "passthrough: style brightness {brightness} contrast {contrast} saturation \
             {saturation}"
        );
        true
    }
}

/// Before the session is destroyed: the provider out of the list, the layer and the
/// passthrough dropped (they destroy their handles on the session), the record cleared.
fn destroy(
    mut layers: Option<ResMut<OxrRenderLayers>>,
    mut record: ResMut<Record>,
    mut commands: Commands,
) {
    if let Some(layers) = layers.as_deref_mut() {
        remove(layers, &mut record);
    }
    commands.remove_resource::<OxrPassthroughLayerFB>();
    commands.remove_resource::<OxrPassthrough>();
    *record = Record::default();
}

#[cfg(test)]
mod tests {
    use bevy_mod_openxr::layer_builder::ProjectionLayer;

    use super::*;

    /// Q-346's record: the provider inserted is found first in the list, once, and its
    /// removal leaves the projection layer alone; a second insert adds nothing.
    #[test]
    fn insert_and_remove_keep_the_record_of_the_provider() {
        let mut layers = OxrRenderLayers(vec![Box::new(ProjectionLayer)]);
        let mut record = Record::default();
        assert!(!record.first_of(&layers) && !record.in_list(&layers));
        insert(&mut layers, &mut record);
        insert(&mut layers, &mut record);
        assert_eq!(layers.len(), 2);
        assert!(record.first_of(&layers));
        // Another provider before it: still in the list, no longer first.
        layers.insert(0, Box::new(Provider { _tag: 1 }));
        assert!(record.in_list(&layers) && !record.first_of(&layers));
        remove(&mut layers, &mut record);
        assert_eq!(layers.len(), 2, "only its own provider is removed");
        layers.remove(0);
        assert_eq!(layers.len(), 1);
        assert!(!record.in_list(&layers));
    }
}
