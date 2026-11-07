// SPDX-License-Identifier: Apache-2.0

//! The passthrough layer in the render world, where bevy\_mod\_openxr keeps its layer
//! list (§1.9, D-14; PRD #565's "The passthrough adapter").
//!
//! At the session's first render frame the passthrough feature and its RECONSTRUCTION
//! layer are created paused through the raw XR\_FB\_passthrough table, and this module
//! owns them (`Owned`). In passthrough mode the feature is started, the layer resumed
//! and this module's provider inserted first in `OxrRenderLayers`, under the projection
//! layer, which keeps its `BLEND_TEXTURE_SOURCE_ALPHA` flag; in VR mode the provider is
//! removed, the layer paused and the feature paused, so the cameras are off when
//! passthrough is not shown (owner, #568). The list holds trait objects, so `Record`
//! keeps the address of the provider inserted here. The style goes through
//! xrPassthroughLayerSetStyleFB at each change (Q-337). The session stays OPAQUE (D-14).
//!
//! Neither the crate's helpers nor openxr 0.21.1's wrappers are used: openxr 0.21.1's
//! `Passthrough::start` calls xrPassthroughPauseFB instead of xrPassthroughStartFB, and
//! its `Passthrough` gives no raw handle to start it with, so a feature created through
//! them could never be started again once paused.

use bevy::app::App;
use bevy::ecs::world::World;
use bevy::log::{error, info};
use bevy::prelude::{Commands, IntoScheduleConfigs, Res, ResMut, Resource, resource_exists};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy_mod_openxr::layer_builder::{CompositionLayer, LayerProvider};
use bevy_mod_openxr::resources::{OxrRenderLayers, OxrSwapchain};
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_xr::session::XrPreDestroySession;
use openxr::sys;
use openxr::sys::Handle as _;

use super::{EnvironmentMode, LayerRequest, PassthroughStyle, style_map};

/// The passthrough feature and its layer, created and owned here, destroyed on drop
/// (before the session, whose clone it holds so the handles never outlive it).
#[derive(Resource)]
pub(crate) struct Owned {
    table: openxr::raw::PassthroughFB,
    feature: sys::PassthroughFB,
    layer: sys::PassthroughLayerFB,
    _session: OxrSession,
}

impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: `layer` and `feature` were created by this table on the session `Owned`
        // holds, which is alive; each is destroyed once, here, the layer before its
        // feature, and nothing uses them afterwards (`Owned` is gone, and the provider
        // holding the layer's handle is removed from the list before `Owned` is).
        unsafe {
            (self.table.destroy_passthrough_layer)(self.layer);
            (self.table.destroy_passthrough)(self.feature);
        }
    }
}

/// `r` as a result.
fn check(r: sys::Result) -> Result<(), sys::Result> {
    if r.into_raw() < 0 { Err(r) } else { Ok(()) }
}

/// The passthrough composition layer submitted for `layer`, flagged as bevy\_mod\_openxr's
/// `PassthroughLayer` is (blend with the source alpha).
struct RawLayer {
    inner: sys::CompositionLayerPassthroughFB,
}

// SAFETY: `header` returns the layer's own structure, whose first fields (`ty`, `next`,
// `flags`, `space`) are XrCompositionLayerBaseHeader's, as OpenXR's layer structures all
// begin; the structure is `repr(C)`, owned by `self` and lives as long as the reference
// returned. It uses no swapchain.
unsafe impl<'a> CompositionLayer<'a> for RawLayer {
    fn swapchain(&self) -> Option<&'a OxrSwapchain> {
        None
    }

    fn header(&self) -> &sys::CompositionLayerBaseHeader {
        // SAFETY: see the impl's comment: the base header is a layout prefix of
        // `CompositionLayerPassthroughFB`, and the reference borrows `self`.
        unsafe { &*std::ptr::from_ref(&self.inner).cast::<sys::CompositionLayerBaseHeader>() }
    }
}

/// The provider inserted in the layer list, holding the layer's handle (a type with a
/// size, so its box has an address of its own to find it by).
struct Provider {
    layer: sys::PassthroughLayerFB,
}

impl LayerProvider for Provider {
    fn get<'a>(&'a self, _world: &'a World) -> Option<Box<dyn CompositionLayer<'a> + 'a>> {
        Some(Box::new(RawLayer {
            inner: sys::CompositionLayerPassthroughFB {
                ty: sys::CompositionLayerPassthroughFB::TYPE,
                next: std::ptr::null(),
                flags: sys::CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA,
                space: sys::Space::NULL,
                layer_handle: self.layer,
            },
        }))
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
    /// Whether the passthrough feature is started.
    pub(crate) running: bool,
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

/// Creates the feature and the layer once per session, then follows the request: the
/// mode (start, resume and insert, or remove, pause and pause) and the style.
pub(super) fn apply(
    request: Res<LayerRequest>,
    session: Res<OxrSession>,
    owned: Option<Res<Owned>>,
    mut layers: ResMut<OxrRenderLayers>,
    mut record: ResMut<Record>,
    mut commands: Commands,
) {
    let Some(owned) = owned else {
        // Nothing is created where the runtime lacks the extension (I-15 on Monado).
        if !record.tried
            && let Some(table) = session.instance().exts().fb_passthrough
        {
            record.tried = true;
            create(&session, table, &mut commands);
        }
        return;
    };
    if record.style.map(|(s, _)| s) != Some(request.style) {
        let ok = set_style(&owned.table, owned.layer, request.style);
        if !ok {
            record.failures += 1;
        }
        record.style = Some((request.style, ok));
    }
    match request.mode {
        EnvironmentMode::Passthrough => {
            if !record.running {
                // SAFETY: `feature` is a live handle of this table (`Owned` holds it).
                let r = unsafe { (owned.table.passthrough_start)(owned.feature) };
                if !call(r, "starting the passthrough", &mut record) {
                    return;
                }
                record.running = true;
            }
            if !record.resumed {
                // SAFETY: `layer` is a live handle of this table (`Owned` holds it).
                let r = unsafe { (owned.table.passthrough_layer_resume)(owned.layer) };
                if !call(r, "resuming the layer", &mut record) {
                    return;
                }
                record.resumed = true;
            }
            insert(&mut layers, &mut record, owned.layer);
        }
        EnvironmentMode::Vr => {
            remove(&mut layers, &mut record);
            if record.resumed {
                // SAFETY: `layer` is a live handle of this table (`Owned` holds it).
                let r = unsafe { (owned.table.passthrough_layer_pause)(owned.layer) };
                if call(r, "pausing the layer", &mut record) {
                    record.resumed = false;
                }
            }
            if record.running && !record.resumed {
                // SAFETY: `feature` is a live handle of this table (`Owned` holds it).
                let r = unsafe { (owned.table.passthrough_pause)(owned.feature) };
                if call(r, "pausing the passthrough", &mut record) {
                    record.running = false;
                }
            }
        }
    }
}

/// Whether the call that returned `r` succeeded; a failure is counted and logged.
fn call(r: sys::Result, what: &str, record: &mut Record) -> bool {
    match check(r) {
        Ok(()) => true,
        Err(e) => {
            record.failures += 1;
            error!("passthrough: {what} failed: {e:?}");
            false
        }
    }
}

/// Creates the passthrough feature and its RECONSTRUCTION layer, both paused (no
/// `IS_RUNNING_AT_CREATION`), as `Owned`.
fn create(session: &OxrSession, table: openxr::raw::PassthroughFB, commands: &mut Commands) {
    let info = sys::PassthroughCreateInfoFB {
        ty: sys::PassthroughCreateInfoFB::TYPE,
        next: std::ptr::null(),
        flags: sys::PassthroughFlagsFB::EMPTY,
    };
    let mut feature = sys::PassthroughFB::NULL;
    // SAFETY: `table` is the session's instance's loaded XR_FB_passthrough table, the
    // session is live (the resource exists), `info` is a valid, typed structure read
    // during the call and `feature` a valid place for the handle it writes.
    let r = unsafe { (table.create_passthrough)(session.as_raw(), &info, &mut feature) };
    if let Err(e) = check(r) {
        error!("passthrough: creating the passthrough failed: {e:?}");
        return;
    }
    let info = sys::PassthroughLayerCreateInfoFB {
        ty: sys::PassthroughLayerCreateInfoFB::TYPE,
        next: std::ptr::null(),
        passthrough: feature,
        flags: sys::PassthroughFlagsFB::EMPTY,
        purpose: sys::PassthroughLayerPurposeFB::RECONSTRUCTION,
    };
    let mut layer = sys::PassthroughLayerFB::NULL;
    // SAFETY: as above; `feature` is the live handle just created on this session.
    let r = unsafe { (table.create_passthrough_layer)(session.as_raw(), &info, &mut layer) };
    if let Err(e) = check(r) {
        error!("passthrough: creating the layer failed: {e:?}");
        // SAFETY: `feature` was created above and is used by nothing else.
        unsafe { (table.destroy_passthrough)(feature) };
        return;
    }
    info!("passthrough: created paused");
    commands.insert_resource(Owned {
        table,
        feature,
        layer,
        _session: session.clone(),
    });
}

/// Inserts this module's provider first in `layers`, unless it is there.
fn insert(layers: &mut OxrRenderLayers, record: &mut Record, layer: sys::PassthroughLayerFB) {
    if record.in_list(layers) {
        return;
    }
    let provider: Box<dyn LayerProvider + Send + Sync> = Box::new(Provider { layer });
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
    // layer of a session of that instance (`Owned` holds it and is removed before the
    // session is destroyed); `info` and the `bcs` its `next` points to are valid, typed structures
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

/// Before the session is destroyed: the provider out of the list, then `Owned` dropped
/// (it destroys the layer and the feature), the record cleared.
fn destroy(
    mut layers: Option<ResMut<OxrRenderLayers>>,
    mut record: ResMut<Record>,
    mut commands: Commands,
) {
    if let Some(layers) = layers.as_deref_mut() {
        remove(layers, &mut record);
    }
    commands.remove_resource::<Owned>();
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
        let layer = sys::PassthroughLayerFB::from_raw(7);
        insert(&mut layers, &mut record, layer);
        insert(&mut layers, &mut record, layer);
        assert_eq!(layers.len(), 2);
        assert!(record.first_of(&layers));
        // Another provider before it: still in the list, no longer first.
        layers.insert(0, Box::new(Provider { layer }));
        assert!(record.in_list(&layers) && !record.first_of(&layers));
        remove(&mut layers, &mut record);
        assert_eq!(layers.len(), 2, "only its own provider is removed");
        layers.remove(0);
        assert_eq!(layers.len(), 1);
        assert!(!record.in_list(&layers));
    }
}
