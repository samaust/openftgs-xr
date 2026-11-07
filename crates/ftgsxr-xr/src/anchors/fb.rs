// SPDX-License-Identifier: Apache-2.0

//! The `AnchorStore` adapter over XR\_FB\_spatial\_entity, XR\_FB\_spatial\_entity\_query
//! and XR\_META\_spatial\_entity\_persistence (§1.9, D-26, Appendix E.1; source-log rows
//! 346 to 350). Every call is asynchronous and completes through an event matched
//! by its request id (Q-343): the create gives the anchor's space and UUID, the query a
//! saved anchor's space, the save and the erase their results. A component the next call
//! needs (STORABLE before the save, LOCATABLE before the locate) is checked and enabled
//! when off, since the documentation leaves their initial state open (source-log row 349).

use std::collections::HashMap;
use std::ptr;

use bevy::log::{info, warn};
use ftgsxr_core::{Pose, Quat, Vec3};
use openxr::sys::Handle as _;
use openxr::{AnyGraphics, Event, sys};

use super::{AnchorDriver, AnchorId, AnchorStore, Status};
use crate::error::XrError;

/// How long a query by UUID may search before the runtime completes it (Q-352).
// Q-352
const QUERY_TIMEOUT_NS: i64 = 10_000_000_000;

/// The prefix of a pending creation's id (Q-343).
const REQUEST: &str = "request:";

/// A UUID as text: lowercase hexadecimal, hyphenated 8-4-4-4-12 (Q-351).
// Q-351
pub(crate) fn uuid_text(u: &sys::Uuid) -> String {
    let h: String = u.data.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

/// The UUID of `text`, hyphens ignored, 32 hexadecimal digits (Q-351).
// Q-351
pub(crate) fn parse_uuid(text: &str) -> Option<sys::Uuid> {
    let hex: Vec<u8> = text.bytes().filter(|&b| b != b'-').collect();
    if hex.len() != 32 {
        return None;
    }
    let mut data = [0u8; 16];
    for (i, pair) in hex.chunks(2).enumerate() {
        let s = std::str::from_utf8(pair).ok()?;
        data[i] = u8::from_str_radix(s, 16).ok()?;
    }
    Some(sys::Uuid { data })
}

/// What became of a creation.
#[derive(Clone, Debug)]
enum Creation {
    Pending,
    Done(String),
    Failed(String),
}

/// What became of a query.
#[derive(Clone, Debug)]
enum Query {
    Pending,
    Found,
    NotFound,
    Failed(String),
}

/// An anchor's save.
#[derive(Clone, Debug, PartialEq)]
enum Save {
    /// Not asked.
    No,
    /// Asked; waiting for STORABLE to be enabled.
    Storable,
    /// `xrSaveSpacesMETA` issued.
    Saving,
    Saved,
    Failed(String),
}

/// An anchor whose space this session holds.
struct Anchor {
    space: openxr::Space,
    save: Save,
}

/// A component-status request: the anchor's UUID and the component.
type StatusRequest = (String, sys::SpaceComponentTypeFB);

/// The adapter (§2.1's adapter of `AnchorStore`).
pub(crate) struct FbAnchors {
    session: openxr::Session<AnyGraphics>,
    /// The primary reference space and the frame time poses are given and located in.
    base: sys::Space,
    time: sys::Time,
    creations: HashMap<u64, Creation>,
    queries: HashMap<String, Query>,
    query_requests: HashMap<u64, String>,
    saves: HashMap<u64, String>,
    statuses: HashMap<u64, StatusRequest>,
    erases: HashMap<u64, String>,
    anchors: HashMap<String, Anchor>,
    /// The erase results, by UUID (Dv-9's cleanup).
    erased: HashMap<String, Result<(), String>>,
}

/// Whether `r` is a success code.
fn ok(r: sys::Result) -> bool {
    r.into_raw() >= 0
}

/// An OpenXR pose as a player pose.
fn pose_of(p: &sys::Posef) -> Pose {
    let o = p.orientation;
    Pose {
        translation: Vec3::new(p.position.x, p.position.y, p.position.z),
        rotation: Quat::from_xyzw(o.x, o.y, o.z, o.w).normalize(),
        scale: 1.0,
    }
}

/// A player pose as an OpenXR pose (the scale dropped).
fn posef(p: &Pose) -> sys::Posef {
    let q = p.rotation.normalize();
    sys::Posef {
        orientation: sys::Quaternionf {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        },
        position: sys::Vector3f {
            x: p.translation.x,
            y: p.translation.y,
            z: p.translation.z,
        },
    }
}

impl FbAnchors {
    /// The adapter for `session`, whose instance enabled the three extensions.
    pub(crate) fn new(session: openxr::Session<AnyGraphics>) -> FbAnchors {
        FbAnchors {
            session,
            base: sys::Space::NULL,
            time: sys::Time::from_nanos(0),
            creations: HashMap::new(),
            queries: HashMap::new(),
            query_requests: HashMap::new(),
            saves: HashMap::new(),
            statuses: HashMap::new(),
            erases: HashMap::new(),
            anchors: HashMap::new(),
            erased: HashMap::new(),
        }
    }

    /// The UUID `id` names: itself, or a completed creation's.
    fn resolve(&self, id: &AnchorId) -> Option<String> {
        match id.0.strip_prefix(REQUEST) {
            Some(n) => match self.creations.get(&n.parse().ok()?) {
                Some(Creation::Done(uuid)) => Some(uuid.clone()),
                _ => None,
            },
            None => Some(id.0.clone()),
        }
    }

    /// Whether `component` is enabled on `space`; `None` when the call fails.
    fn enabled(&self, space: sys::Space, component: sys::SpaceComponentTypeFB) -> Option<bool> {
        let fp = self.session.instance().exts().fb_spatial_entity?;
        let mut status = sys::SpaceComponentStatusFB {
            ty: sys::SpaceComponentStatusFB::TYPE,
            next: ptr::null_mut(),
            enabled: false.into(),
            change_pending: false.into(),
        };
        // SAFETY: the table is the runtime's (XR_FB_spatial_entity enabled); `space` is
        // a live spatial-entity space this adapter holds; `status` is the output struct
        // with its TYPE, alive for the call.
        let r = unsafe { (fp.get_space_component_status)(space, component, &mut status) };
        ok(r).then(|| status.enabled.into())
    }

    /// Asks the runtime to enable `component` on the anchor `uuid`; its event follows.
    // Q-352
    fn enable(&mut self, uuid: &str, component: sys::SpaceComponentTypeFB) -> Result<(), String> {
        let fp = self
            .session
            .instance()
            .exts()
            .fb_spatial_entity
            .ok_or("XR_FB_spatial_entity")?;
        let space = self.anchors.get(uuid).ok_or("no space")?.space.as_raw();
        let info = sys::SpaceComponentStatusSetInfoFB {
            ty: sys::SpaceComponentStatusSetInfoFB::TYPE,
            next: ptr::null(),
            component_type: component,
            enabled: true.into(),
            timeout: sys::Duration::from_nanos(0),
        };
        let mut req = sys::AsyncRequestIdFB::from_raw(0);
        // SAFETY: the table is the runtime's; `space` is a live space this adapter holds;
        // `info` is initialised with its TYPE and outlives the call; `req` is written.
        let r = unsafe { (fp.set_space_component_status)(space, &info, &mut req) };
        if r == sys::Result::ERROR_SPACE_COMPONENT_STATUS_ALREADY_SET_FB {
            self.on_enabled(uuid, component);
            return Ok(());
        }
        if !ok(r) {
            return Err(format!("xrSetSpaceComponentStatusFB({component:?}): {r:?}"));
        }
        self.statuses
            .insert(req.into_raw(), (uuid.to_owned(), component));
        Ok(())
    }

    /// The component is on: the step that waited for it follows.
    fn on_enabled(&mut self, uuid: &str, component: sys::SpaceComponentTypeFB) {
        if component == sys::SpaceComponentTypeFB::STORABLE
            && self
                .anchors
                .get(uuid)
                .is_some_and(|a| a.save == Save::Storable)
        {
            self.save(uuid);
        }
    }

    /// Issues `xrSaveSpacesMETA` for the anchor `uuid`.
    fn save(&mut self, uuid: &str) {
        let result = (|| {
            let fp = self
                .session
                .instance()
                .exts()
                .meta_spatial_entity_persistence
                .ok_or("XR_META_spatial_entity_persistence")?;
            let mut space = self.anchors.get(uuid).ok_or("no space")?.space.as_raw();
            let info = sys::SpacesSaveInfoMETA {
                ty: sys::SpacesSaveInfoMETA::TYPE,
                next: ptr::null(),
                space_count: 1,
                spaces: &mut space,
            };
            let mut req = sys::AsyncRequestIdFB::from_raw(0);
            // SAFETY: the table is the runtime's (the extension enabled); the session is
            // alive (held); `info` points at one live space and outlives the call; `req`
            // is written by the runtime.
            let r = unsafe { (fp.save_spaces)(self.session.as_raw(), &info, &mut req) };
            if !ok(r) {
                return Err(format!("xrSaveSpacesMETA: {r:?}"));
            }
            Ok(req.into_raw())
        })();
        match result {
            Ok(req) => {
                self.saves.insert(req, uuid.to_owned());
                self.set_save(uuid, Save::Saving);
            }
            Err(e) => {
                warn!("anchor {uuid}: {e}");
                self.set_save(uuid, Save::Failed(e));
            }
        }
    }

    fn set_save(&mut self, uuid: &str, save: Save) {
        if let Some(a) = self.anchors.get_mut(uuid) {
            a.save = save;
        }
    }

    /// Takes ownership of a space the runtime returned for the anchor `uuid`.
    fn hold(&mut self, uuid: String, raw: sys::Space) {
        // SAFETY: `raw` is a spatial-anchor space the runtime just returned for this
        // session (a create's or a query's event), not yet owned by anyone; the `Space`
        // destroys it when dropped.
        let space = unsafe { openxr::Space::reference_from_raw(self.session.clone(), raw) };
        self.anchors.insert(
            uuid,
            Anchor {
                space,
                save: Save::No,
            },
        );
    }

    /// The results of the query `req`, retrieved once (two calls).
    fn retrieve(&mut self, req: u64) {
        let Some(uuid) = self.query_requests.get(&req).cloned() else {
            return;
        };
        let Some(fp) = self.session.instance().exts().fb_spatial_entity_query else {
            return;
        };
        let session = self.session.as_raw();
        let mut results = sys::SpaceQueryResultsFB {
            ty: sys::SpaceQueryResultsFB::TYPE,
            next: ptr::null_mut(),
            result_capacity_input: 0,
            result_count_output: 0,
            results: ptr::null_mut(),
        };
        let id = sys::AsyncRequestIdFB::from_raw(req);
        // SAFETY: the table is the runtime's; the session is alive; `results` asks for
        // the count only (capacity 0, null array).
        let r = unsafe { (fp.retrieve_space_query_results)(session, id, &mut results) };
        if !ok(r) {
            self.queries
                .insert(uuid, Query::Failed(format!("retrieve: {r:?}")));
            return;
        }
        let n = results.result_count_output as usize;
        let mut out = vec![
            sys::SpaceQueryResultFB {
                space: sys::Space::NULL,
                uuid: sys::Uuid { data: [0; 16] },
            };
            n
        ];
        results.result_capacity_input = n as u32;
        results.results = out.as_mut_ptr();
        // SAFETY: as above, with `results` pointing at `n` writable elements, the count
        // the first call gave.
        let r = unsafe { (fp.retrieve_space_query_results)(session, id, &mut results) };
        if !ok(r) {
            self.queries
                .insert(uuid, Query::Failed(format!("retrieve: {r:?}")));
            return;
        }
        out.truncate(results.result_count_output as usize);
        for found in out {
            let text = uuid_text(&found.uuid);
            if text != uuid {
                continue;
            }
            self.hold(text.clone(), found.space);
            self.queries.insert(text.clone(), Query::Found);
            if self.enabled(found.space, sys::SpaceComponentTypeFB::LOCATABLE) != Some(true)
                && let Err(e) = self.enable(&text, sys::SpaceComponentTypeFB::LOCATABLE)
            {
                warn!("anchor {text}: {e}");
            }
        }
    }

    /// The anchor `uuid`'s status, from its space and its save.
    fn anchor_status(&self, uuid: &str) -> Option<Status> {
        let a = self.anchors.get(uuid)?;
        let id = AnchorId(uuid.to_owned());
        Some(match &a.save {
            Save::Saved => Status::Persisted(id),
            Save::Failed(e) => Status::Failed(e.clone()),
            _ => Status::Created(id),
        })
    }
}

impl AnchorStore for FbAnchors {
    fn create(&mut self, pose: Pose) -> Result<AnchorId, XrError> {
        let fp = self
            .session
            .instance()
            .exts()
            .fb_spatial_entity
            .ok_or(XrError::Unsupported("XR_FB_spatial_entity"))?;
        if self.base == sys::Space::NULL || self.time.as_nanos() == 0 {
            return Err(XrError::Runtime("anchor: no frame yet".into()));
        }
        let info = sys::SpatialAnchorCreateInfoFB {
            ty: sys::SpatialAnchorCreateInfoFB::TYPE,
            next: ptr::null(),
            space: self.base,
            pose_in_space: posef(&pose),
            time: self.time,
        };
        let mut req = sys::AsyncRequestIdFB::from_raw(0);
        // SAFETY: the table is the runtime's (the extension enabled); the session and the
        // base space are alive (the session's primary reference space, refreshed each
        // frame); `info` is initialised with its TYPE and outlives the call.
        let r = unsafe { (fp.create_spatial_anchor)(self.session.as_raw(), &info, &mut req) };
        if !ok(r) {
            return Err(XrError::Runtime(format!("xrCreateSpatialAnchorFB: {r:?}")));
        }
        let n = req.into_raw();
        self.creations.insert(n, Creation::Pending);
        Ok(AnchorId(format!("{REQUEST}{n}")))
    }

    fn locate(&self, id: &AnchorId) -> Option<Pose> {
        let uuid = self.resolve(id)?;
        let space = self.anchors.get(&uuid)?.space.as_raw();
        if self.base == sys::Space::NULL || self.time.as_nanos() == 0 {
            return None;
        }
        let mut loc = sys::SpaceLocation {
            ty: sys::SpaceLocation::TYPE,
            next: ptr::null_mut(),
            location_flags: sys::SpaceLocationFlags::EMPTY,
            pose: sys::Posef::IDENTITY,
        };
        // SAFETY: the core table is the instance's; `space` is a live anchor space and
        // `base` the session's live primary reference space; `loc` is the output struct
        // with its TYPE, alive for the call.
        let r = unsafe {
            (self.session.instance().fp().locate_space)(space, self.base, self.time, &mut loc)
        };
        let valid =
            sys::SpaceLocationFlags::POSITION_VALID | sys::SpaceLocationFlags::ORIENTATION_VALID;
        (ok(r) && loc.location_flags.contains(valid)).then(|| pose_of(&loc.pose))
    }

    fn persist(&mut self, id: &AnchorId) -> Result<(), XrError> {
        let uuid = self
            .resolve(id)
            .ok_or_else(|| XrError::Runtime(format!("anchor {}: not created", id.0)))?;
        let space = self
            .anchors
            .get(&uuid)
            .ok_or_else(|| XrError::Runtime(format!("anchor {uuid}: no space")))?
            .space
            .as_raw();
        match self.enabled(space, sys::SpaceComponentTypeFB::STORABLE) {
            Some(true) => self.save(&uuid),
            _ => {
                self.set_save(&uuid, Save::Storable);
                self.enable(&uuid, sys::SpaceComponentTypeFB::STORABLE)
                    .map_err(XrError::Runtime)?;
            }
        }
        Ok(())
    }

    fn forget(&mut self, id: &AnchorId) -> Result<(), XrError> {
        let fp = self
            .session
            .instance()
            .exts()
            .meta_spatial_entity_persistence
            .ok_or(XrError::Unsupported("XR_META_spatial_entity_persistence"))?;
        let text = self
            .resolve(id)
            .ok_or_else(|| XrError::Runtime(format!("anchor {}: not created", id.0)))?;
        let mut uuid = parse_uuid(&text)
            .ok_or_else(|| XrError::Runtime(format!("anchor {text}: not a UUID")))?;
        let info = sys::SpacesEraseInfoMETA {
            ty: sys::SpacesEraseInfoMETA::TYPE,
            next: ptr::null(),
            space_count: 0,
            spaces: ptr::null_mut(),
            uuid_count: 1,
            uuids: &mut uuid,
        };
        let mut req = sys::AsyncRequestIdFB::from_raw(0);
        // SAFETY: the table is the runtime's (the extension enabled); the session is
        // alive; `info` names one UUID, alive for the call, and no spaces (source-log row 348: the
        // erase takes UUIDs alone).
        let r = unsafe { (fp.erase_spaces)(self.session.as_raw(), &info, &mut req) };
        if !ok(r) {
            return Err(XrError::Runtime(format!("xrEraseSpacesMETA: {r:?}")));
        }
        self.erases.insert(req.into_raw(), text.clone());
        // The space stays tracked until destroyed (source-log row 349): dropped here.
        self.anchors.remove(&text);
        Ok(())
    }
}

impl AnchorDriver for FbAnchors {
    fn available(&self) -> bool {
        true
    }

    fn ready(&self) -> bool {
        self.base != sys::Space::NULL && self.time.as_nanos() != 0
    }

    fn query(&mut self, uuid: &AnchorId) -> Result<(), XrError> {
        let fp = self
            .session
            .instance()
            .exts()
            .fb_spatial_entity_query
            .ok_or(XrError::Unsupported("XR_FB_spatial_entity_query"))?;
        if self.anchors.contains_key(&uuid.0) {
            self.queries.insert(uuid.0.clone(), Query::Found);
            return Ok(());
        }
        let mut raw = parse_uuid(&uuid.0)
            .ok_or_else(|| XrError::Runtime(format!("anchor {}: not a UUID", uuid.0)))?;
        let filter = sys::SpaceUuidFilterInfoFB {
            ty: sys::SpaceUuidFilterInfoFB::TYPE,
            next: ptr::null(),
            uuid_count: 1,
            uuids: &mut raw,
        };
        let info = sys::SpaceQueryInfoFB {
            ty: sys::SpaceQueryInfoFB::TYPE,
            next: ptr::null(),
            query_action: sys::SpaceQueryActionFB::LOAD,
            max_result_count: 1,
            timeout: sys::Duration::from_nanos(QUERY_TIMEOUT_NS),
            filter: (&raw const filter).cast(),
            exclude_filter: ptr::null(),
        };
        let mut req = sys::AsyncRequestIdFB::from_raw(0);
        // SAFETY: the table is the runtime's (the extension enabled); the session is
        // alive; `info` and the filter it points at are initialised with their TYPEs,
        // the filter's one UUID alive for the call.
        let r =
            unsafe { (fp.query_spaces)(self.session.as_raw(), (&raw const info).cast(), &mut req) };
        if !ok(r) {
            return Err(XrError::Runtime(format!("xrQuerySpacesFB: {r:?}")));
        }
        self.query_requests.insert(req.into_raw(), uuid.0.clone());
        self.queries.insert(uuid.0.clone(), Query::Pending);
        Ok(())
    }

    fn status(&self, id: &AnchorId) -> Status {
        if let Some(n) = id.0.strip_prefix(REQUEST) {
            return match n.parse().ok().and_then(|n: u64| self.creations.get(&n)) {
                None => Status::Failed(format!("unknown request {n}")),
                Some(Creation::Pending) => Status::Pending,
                Some(Creation::Failed(e)) => Status::Failed(e.clone()),
                Some(Creation::Done(uuid)) => self
                    .anchor_status(uuid)
                    .unwrap_or_else(|| Status::Failed("forgotten".into())),
            };
        }
        if let Some(s) = self.anchor_status(&id.0) {
            return s;
        }
        match self.queries.get(&id.0) {
            Some(Query::Pending) => Status::Pending,
            Some(Query::NotFound) => Status::NotFound,
            Some(Query::Failed(e)) => Status::Failed(e.clone()),
            Some(Query::Found) | None => Status::Failed("unknown anchor".into()),
        }
    }

    fn frame(&mut self, time: sys::Time, base: sys::Space) {
        self.time = time;
        self.base = base;
    }

    fn event(&mut self, event: &Event<'_>) {
        match *event {
            Event::SpatialAnchorCreateCompleteFB(e) => {
                let n = e.request_id().into_raw();
                if !self.creations.contains_key(&n) {
                    return;
                }
                if ok(e.result()) {
                    let uuid = uuid_text(&e.uuid());
                    info!("anchor {uuid} created (request {n})");
                    self.hold(uuid.clone(), e.space());
                    self.creations.insert(n, Creation::Done(uuid));
                } else {
                    let msg = format!("xrCreateSpatialAnchorFB completed with {:?}", e.result());
                    warn!("anchor request {n}: {msg}");
                    self.creations.insert(n, Creation::Failed(msg));
                }
            }
            Event::SpaceSetStatusCompleteFB(e) => {
                let Some((uuid, component)) = self.statuses.remove(&e.request_id().into_raw())
                else {
                    return;
                };
                if ok(e.result()) {
                    self.on_enabled(&uuid, component);
                } else {
                    let msg = format!("enabling {component:?}: {:?}", e.result());
                    warn!("anchor {uuid}: {msg}");
                    if component == sys::SpaceComponentTypeFB::STORABLE {
                        self.set_save(&uuid, Save::Failed(msg));
                    }
                }
            }
            Event::SpacesSaveResultMETA(e) => {
                let Some(uuid) = self.saves.remove(&e.request_id().into_raw()) else {
                    return;
                };
                if ok(e.result()) {
                    info!("anchor {uuid} persisted");
                    self.set_save(&uuid, Save::Saved);
                } else {
                    let msg = format!("xrSaveSpacesMETA completed with {:?}", e.result());
                    warn!("anchor {uuid}: {msg}");
                    self.set_save(&uuid, Save::Failed(msg));
                }
            }
            Event::SpaceQueryResultsAvailableFB(e) => self.retrieve(e.request_id().into_raw()),
            Event::SpaceQueryCompleteFB(e) => {
                let Some(uuid) = self.query_requests.remove(&e.request_id().into_raw()) else {
                    return;
                };
                if matches!(self.queries.get(&uuid), Some(Query::Pending)) {
                    let q = if ok(e.result()) {
                        Query::NotFound
                    } else {
                        Query::Failed(format!("xrQuerySpacesFB completed with {:?}", e.result()))
                    };
                    info!("anchor {uuid}: query complete, {q:?}");
                    self.queries.insert(uuid, q);
                }
            }
            Event::SpacesEraseResultMETA(e) => {
                let Some(uuid) = self.erases.remove(&e.request_id().into_raw()) else {
                    return;
                };
                let r = if ok(e.result()) {
                    info!("anchor {uuid} erased");
                    Ok(())
                } else {
                    Err(format!("xrEraseSpacesMETA completed with {:?}", e.result()))
                };
                self.erased.insert(uuid, r);
            }
            _ => {}
        }
    }

    fn erased(&self, uuid: &str) -> Option<Result<(), String>> {
        self.erased.get(uuid).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q-351: a UUID's text is lowercase, hyphenated 8-4-4-4-12, and parses back.
    #[test]
    fn uuid_text_round_trips_through_parse_uuid() {
        let u = sys::Uuid {
            data: [
                0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                0xcd, 0xef,
            ],
        };
        let t = uuid_text(&u);
        assert_eq!(t, "12345678-9abc-def0-0123-456789abcdef");
        assert_eq!(parse_uuid(&t).map(|p| p.data), Some(u.data));
        assert!(parse_uuid("not a uuid").is_none());
        assert!(parse_uuid("request:3").is_none());
    }
}
