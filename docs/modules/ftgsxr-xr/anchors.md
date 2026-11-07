# Spatial anchors in `ftgsxr-xr`

How `ftgsxr_xr::anchors` keeps a scene in its place in the room: the `AnchorStore` port and its adapter over the spatial-entity extensions, `NoAnchors`, the XR side of the scene anchor's handshake with `persist`, the frames of the anchors' poses, the content group's placement on a located anchor, the forget, and Dv-9's device side (M-6, #571). The spec items are §1.9 Floor, shadows and anchors, §1.11 Scene files (`anchor`), §1.13 HUD, §1.14 Frames, §1.15 E-XR-PERMISSION and E-XR-UNSUPPORTED, §2.1 `AnchorId`, `AnchorStore`, `NoAnchors` and `PersistCommand`, §3.4 Dv-9, D-26 and Appendix E.1; this doc explains them and never overrides them. The app side of the handshake is `ftgsxr-app/persist.md`'s subject.

## The port and its asynchronous calls

§2.1's `AnchorStore` is synchronous, while the runtime's calls complete through events a frame or more later (Q-343). The port keeps its signatures and reads them so:

- `create` issues the request and returns at once an `AnchorId` holding the request's id as text, `request:<n>`; when the creation's event arrives the id resolves to the anchor's UUID, which `locate` and `persist` accept as well.
- `locate` is `None` until the anchor's space exists and is located, then the runtime's pose, unprojected: Q-254's projection happens where the content group is placed.
- `persist` saves asynchronously, and `forget` erases the saved anchor.

An `AnchorId`'s UUID is its 16 bytes in order as lowercase hexadecimal, hyphenated 8-4-4-4-12, the form the scene file keeps and logcat shows; it is parsed back with the hyphens ignored and exactly 32 digits (Q-351).

Beyond the port, a crate-private `AnchorDriver` gives the handshake what the events carry: whether the runtime offers anchors, whether the store can create and locate yet, the query of a saved anchor by its UUID, the status of a creation, a save or a query (pending, created, persisted, not found, failed), the frame's time and reference space, and the runtime's events. The adapter, `NoAnchors` and the private tests' fake implement it; no public item injects a store (PRD #565, "Where the ports' objects live").

## The stores

- **The adapter** (`fb`), over XR\_FB\_spatial\_entity and XR\_META\_spatial\_entity\_persistence, with a query by UUID through XR\_META\_spatial\_entity\_discovery or, where the runtime lacks it, XR\_FB\_spatial\_entity\_query, is the session's store where the instance enabled the first two and either of the last two (Q-358). Appendix E.1 lists the FB query for anchors, but Q-358 records that on the Quest 3 its query with a UUID filter returned a validation failure in Dv-9's first run, so the adapter asks for discovery with one UUID filter first; discovery's results are taken as they become available, each once, until its completion event (source-log 364). Every call is matched to its completion event by its request id: the creation gives the anchor's space and UUID, the query or the discovery a saved anchor's space, the save and the erase their results. A component the next call needs, STORABLE before the save and LOCATABLE before the locate, is checked and enabled only when the runtime reports it off, since the documentation leaves their initial state open (Q-352). A query or a discovery that completes without the anchor is a not-found; the FB query, on the fallback path alone, asks the runtime to time out after Q-352's limit (Q-358).
- **`NoAnchors`** is the store on the desktop and on a headset without the extensions, each missing one logged once as E-XR-UNSUPPORTED, a warning with no toast (Q-134, Q-161). Every call errs at once, so a scene save never waits on it (Q-343).

The store is a crate-private main-world resource, `NoAnchors` until a session with the extensions exists; the plugin feeds it the runtime's events and, while the session runs, each frame's predicted time and primary reference space.

## The frames

The poses `create` takes and `locate` returns are in the session's primary reference space (STAGE or its fallback, §1.8), the one the runtime anchors in, so an anchor stays in the room however the rig moved (Q-351). With M the tracking root's world pose, scale dropped on both sides:

- **`creation_pose`**: the anchor for the content group at world pose G is created at M⁻¹ G, so it carries the group's yaw and translation (§1.9: "at the content group's origin").
- **`placed_group`**: a located L places the group at the world pose M L projected on its twist about stage +y, its translation kept and its scale 1 (Q-254), as `session`'s re-placement composes the root (Q-264, `session.md`).

Q-254's three private tests (a 2° tilt, a 30° yaw under a tilt, a 3 cm height) and the handshake's creation test run under a moved and turned root.

## The handshake's XR side

`persist` and `anchors` meet at one public resource of `ftgsxr_app::persist`, `SceneAnchor` (Q-343): `persist` sets `Creating` at a scene save on the headset in passthrough mode and `Locating(uuid)` at the load of a file with an anchor on the headset, and ends the save's wait (`ftgsxr-app/persist.md`). Each frame `handshake` answers:

- **A save** (`Creating`). Permission granted, the store creates an anchor at `creation_pose` of the content group's origin, persists it once created, and sets `Ready(uuid)` once persisted; `persist` then writes the file with that UUID. A store that errs (`NoAnchors`) or a creation, save or permission that fails leaves `None`, which ends `persist`'s wait as a failure (Q-350).
- **A load** (`Locating(uuid)`). The store queries the UUID; once found and located, the group is placed by `placed_group` and the state set to `Ready(uuid)`. An anchor the query does not find sets `Lost(uuid)`, which `persist` toasts, the group staying where the file put it. A found anchor not yet located stays `Locating`, with no time limit; the HUD shows it.
- **The permission.** `Permission::Anchors` is asked through the platform port on the first create or locate (Q-348). A denial gives Q-261's toast, sets `XrFeatures.anchors` to `Denied` and lists the permission in `DeniedPermissions`; during a save it writes `None`, during a load `Lost(uuid)` (Q-352). `AskPermissionAgain(Permission::Anchors)` asks once more.
- **Before the store exists** (the session not running yet, a re-creation): where the instance enabled the extensions, a `Creating` or `Locating` waits for the session's adapter and its first frame rather than failing on the `NoAnchors` standing in; `persist`'s wait still bounds a save (Q-352).
- **A late creation.** A creation still pending when the state leaves `Creating` (the save gave up waiting) is dropped, and its anchor forgotten once it resolves, so no file-less anchor stays in storage; the state stays as `persist` set it (Q-352).
- **The HUD.** `HudData.anchor` is written from the state and what `Ready` came from: `Creating`, `Persisted` (a save's), `Locating`, `Located` (a load's) or `Lost`, `None` without an anchor (Q-345).

`XrFeatures.anchors` is `Available` where the store offers anchors, `Absent` otherwise (Q-336).

## The forget

§1.9 forgets an anchor when its scene file is deleted from the player, but no action deletes a scene file yet; that gap is the owner's (Q-344). Meanwhile an anchor is forgotten when the scene file that named it is saved again with a new one, the old UUID erased once the new one is persisted, and in Dv-9's cleanup. Since `persist` keeps the path's resolution private, `anchors` reads the §2.1 `PersistCommand::SaveScene(p)` and `LoadScene(p)` and remembers, for the process's life, the UUID each `p` last named, a save's persisted anchor or a load's located one (Q-353). A file re-saved under another spelling of its path, or first re-saved after a restart without being loaded, keeps its old anchor in storage.

## Dv-9 on the device

Dv-9's clauses live here, since they need the crate-private store; they stay inert until the quest runner spawns the entity `ftgsxr-test: dv9`, and print the verdict line themselves (Q-308, Q-122). The host driver grants USE\_ANCHOR\_API, then launches the app twice (Q-346):

1. The first launch creates an anchor Q-346's distance ahead of the content group's origin, persists it, and prints its UUID and created pose in the primary reference space in the verdict's detail, with the frames and milliseconds the creation's and the save's events took (PRD #565's spike check 4).
2. The driver passes that part of the detail to the second launch in the intent extra `ftgsxr-anchor`, which the runner turns into a second entity, `ftgsxr-anchor: <detail>`. The second launch finds the UUID through the adapter's query (Q-358) (spike check 5: found after a restart, or not), locates it, and passes when the located position lies within Q-346's distance of the created one and the erase succeeds. Dv-9 passed on the Quest 3 with the discovery query (#571, Q-358).

## Q-n rows relied on

Q-254, Q-336, Q-343, Q-344, Q-345, Q-346, Q-348, Q-350, Q-351, Q-352, Q-353, Q-358; with Q-122, Q-134, Q-161, Q-261, Q-264 and Q-308.
