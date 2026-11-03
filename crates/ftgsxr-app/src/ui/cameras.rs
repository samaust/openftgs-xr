// SPDX-License-Identifier: Apache-2.0

//! Training cameras and requests (§1.7): a `ftgs-scene` manifest beside a model offers
//! its cameras in the Scene tab (the list by split, "jump to camera", the frustums drawn
//! in the instance's frame); a `ftgs-render-request` file found in a directory source is
//! replayed item by item from the Library tab. A jump moves the fly camera to the
//! camera's pose and field of view and seeks the clock to the image's timestamp (E9).
//! Cameras follow openftgs's conventions (E6, §1.14 Frames): OpenCV axes, world-to-camera.

use std::path::{Path, PathBuf};

use bevy::color::Color;
use bevy::ecs::lifecycle::Add;
use bevy::ecs::prelude::*;
use bevy::gizmos::config::GizmoConfigStore;
use bevy::gizmos::gizmos::Gizmos;
use bevy::log::warn;
use bevy::math::{Mat3, Quat, Vec3};
use bevy::transform::components::GlobalTransform;
use ftgsxr_core::{Camera, DMat3, DVec3, Pose, RenderRequest, SceneManifest, read_request};
use ftgsxr_io::ModelRef;
use ftgsxr_render::SplatShLimit;

use super::ToastRequest;
use crate::intent::{Intent, PlaybackCommand};
use crate::library::{IndexScene, Library, LoadedModel};
use crate::manipulate::manifest_beside;
use crate::rig::{BackgroundOverride, CameraJump};

/// The `ftgs-scene` manifest found beside an instance's model (§1.7), with its path.
// Q-219
#[derive(Component, Clone, Debug)]
pub(crate) struct TrainingCameras {
    pub(crate) path: PathBuf,
    pub(crate) manifest: SceneManifest,
}

/// A request to jump to `camera` at `timestamp` (in `time_unit`), in `instance`'s frame
/// (§1.7): the Scene tab's image list and the request replay send it.
// Q-219
#[derive(Message, Clone, Debug, PartialEq)]
pub(crate) struct JumpToCamera {
    pub(crate) instance: Entity,
    pub(crate) camera: Camera,
    pub(crate) timestamp: f64,
    pub(crate) time_unit: String,
}

/// An opened request and the item shown (§1.7).
#[derive(Clone, Debug)]
pub(crate) struct Replay {
    pub(crate) path: PathBuf,
    pub(crate) request: RenderRequest,
    pub(crate) item: usize,
}

/// The training-camera view's state: the frustums toggle, the split filter, the request
/// files the directory sources hold and the request being replayed.
#[derive(Resource, Clone, Debug, Default)]
pub(crate) struct TrainingView {
    pub(crate) frustums: bool,
    /// `None` shows both splits.
    pub(crate) split: Option<String>,
    pub(crate) requests: Vec<PathBuf>,
    pub(crate) replay: Option<Replay>,
}

pub(super) fn build(app: &mut bevy::app::App) {
    use bevy::app::Update;
    app.add_message::<JumpToCamera>()
        .add_message::<CameraJump>()
        .init_resource::<TrainingView>()
        // `SplatPlugin` and `RigPlugin` init them too; an app without them still runs.
        .init_resource::<BackgroundOverride>()
        .init_resource::<SplatShLimit>()
        .add_observer(find_manifest)
        .add_systems(
            Update,
            (
                find_requests,
                follow_replay,
                run_jumps,
                frustums.run_if(resource_exists::<GizmoConfigStore>),
            )
                .in_set(super::UiSet),
        );
}

/// The instance's manifest, as it appears (§1.7): the index entry's `scene`, downloaded
/// with the model, else for a local model "any `*.json` whose `format` is `ftgs-scene`" in
/// its directory (Q-219). An index manifest that does not read as `ftgs-scene` offers no
/// cameras.
fn find_manifest(
    add: On<Add, LoadedModel>,
    models: Query<(&LoadedModel, Option<&IndexScene>)>,
    mut commands: Commands,
) {
    let Ok((m, index)) = models.get(add.entity) else {
        return;
    };
    if let Some(IndexScene(path)) = index {
        let read = std::fs::read(path)
            .map_err(|e| e.to_string())
            .and_then(|b| ftgsxr_core::read_scene_manifest(&b).map_err(|e| e.to_string()));
        match read {
            Ok(manifest) => {
                commands.entity(add.entity).try_insert(TrainingCameras {
                    path: path.clone(),
                    manifest,
                });
            }
            Err(e) => warn!("the scene manifest {}: {e}", path.display()),
        }
        return;
    }
    let ModelRef::File(p) = &m.source else {
        return;
    };
    // A checkpoint is a directory; its manifest sits beside it or in it.
    let dirs = [p.parent().map(Path::to_path_buf), Some(p.clone())];
    let found = dirs
        .into_iter()
        .flatten()
        .filter(|d| d.is_dir())
        .find_map(|d| manifest_beside(&d));
    if let Some((path, manifest)) = found {
        commands
            .entity(add.entity)
            .try_insert(TrainingCameras { path, manifest });
    }
}

/// The `ftgs-render-request` files at the top of each directory source, in name order,
/// found again whenever the library changes (Q-219).
// Q-219
fn find_requests(
    library: Option<Res<Library>>,
    mut scanned: Local<Option<Vec<String>>>,
    mut view: ResMut<TrainingView>,
) {
    let Some(library) = library.filter(|l| l.is_changed()) else {
        return;
    };
    // Only a change of the directory sources rescans, not their entries' states.
    let dirs: Vec<String> = library
        .sources
        .iter()
        .filter(|(s, _)| s.kind == "dir")
        .map(|(s, _)| s.location.clone())
        .collect();
    if scanned.as_ref() == Some(&dirs) {
        return;
    }
    let mut found = Vec::new();
    for location in &dirs {
        let Ok(dir) = std::fs::read_dir(location) else {
            continue;
        };
        let mut jsons: Vec<PathBuf> = dir
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "json"))
            .filter(|p| std::fs::read(p).is_ok_and(|b| read_request(&b).is_ok()))
            .collect();
        jsons.sort();
        found.extend(jsons);
    }
    *scanned = Some(dirs);
    if view.requests != found {
        view.requests = found;
    }
}

/// While a request is open, its background is the view's (`BackgroundOverride`) and its
/// `sh_degree` the extra drawing cap (`SplatShLimit`); without one, neither is set. The
/// governor's `SplatTierState` and the settings are never written.
// Q-230, Q-231, Q-232
fn follow_replay(
    view: Res<TrainingView>,
    mut background: ResMut<BackgroundOverride>,
    mut limit: ResMut<SplatShLimit>,
) {
    let r = view.replay.as_ref().map(|r| &r.request);
    let want = (r.map(|r| r.background), r.and_then(|r| r.sh_degree));
    if background.0 != want.0 {
        background.0 = want.0;
    }
    if limit.0 != want.1 {
        limit.0 = want.1;
    }
}

/// Opens a request file for replay (the Library tab); its failure is a toast.
pub(crate) fn open_request(path: &Path) -> Result<Replay, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("E-IO: Could not read {}: {e}", path.display()))?;
    let request = read_request(&bytes).map_err(|e| format!("{}: {e}", e.code()))?;
    Ok(Replay {
        path: path.to_path_buf(),
        request,
        item: 0,
    })
}

/// The OpenCV-to-engine axis flip, diag(1, −1, −1) (§1.14 Frames).
const FLIP: DVec3 = DVec3::new(1.0, -1.0, -1.0);

/// A camera's stage pose and vertical field of view in degrees: E6's desktop rule
/// inverted (R(q) = R_wᵀ diag(1, −1, −1), p = −R_wᵀ t_w), in model space, then posed by
/// the instance's transform; its uniform scale moves the position and leaves the
/// rotation. φ = 2 atan(H / 2 f_y).
// Q-219
pub(crate) fn camera_pose(camera: &Camera, instance: &GlobalTransform) -> (Pose, f32) {
    let r_w = DMat3::from_mat4(camera.world_to_camera);
    let r = r_w.transpose() * DMat3::from_diagonal(FLIP);
    let model_rot = Quat::from_mat3(&Mat3::from_cols(
        r.x_axis.as_vec3(),
        r.y_axis.as_vec3(),
        r.z_axis.as_vec3(),
    ))
    .normalize();
    let (_, rot, _) = instance.to_scale_rotation_translation();
    let pose = Pose {
        translation: instance.transform_point(camera.center().as_vec3()),
        rotation: (rot * model_rot).normalize(),
        scale: 1.0,
    };
    let fy = camera.k.y_axis.y;
    let fov = 2.0 * (f64::from(camera.height) / (2.0 * fy)).atan();
    (pose, fov.to_degrees() as f32)
}

/// Each jump: the fly camera takes the pose and field of view, the clock pauses at
/// T = o_m + (τ − τ₀)/r_m (E9; a single-timestamp model leaves the clock). A camera whose
/// time unit is not the model's is refused with a warning (Q-219).
fn run_jumps(
    mut jumps: MessageReader<JumpToCamera>,
    models: Query<(&LoadedModel, &GlobalTransform)>,
    mut camera: MessageWriter<CameraJump>,
    mut intents: MessageWriter<Intent>,
    mut toasts: MessageWriter<ToastRequest>,
) {
    for j in jumps.read() {
        let Ok((m, g)) = models.get(j.instance) else {
            continue;
        };
        let span = &m.time.span;
        if j.time_unit != span.time_unit {
            toasts.write(ToastRequest::warning(format!(
                "Camera time unit {}, model {}",
                j.time_unit, span.time_unit
            )));
            continue;
        }
        let (pose, fov_deg) = camera_pose(&j.camera, g);
        camera.write(CameraJump { pose, fov_deg });
        if !span.is_single() {
            let t = m.time.offset_s + (j.timestamp - span.tau0) / m.time.rate;
            intents.write(Intent::Playback(PlaybackCommand::Pause));
            intents.write(Intent::Playback(PlaybackCommand::Seek(t)));
        }
    }
}

const FRUSTUM: Color = Color::srgb(0.3, 0.8, 1.0);
/// The frustums' depth in stage metres (cosmetic, Q-219).
const FRUSTUM_DEPTH_M: f64 = 0.1;

/// Each manifest camera's frustum in its instance's frame (§1.7 toggle): the center to
/// the image corners back-projected through K⁻¹ at a fixed depth, and the image's edges.
fn frustums(
    view: Res<TrainingView>,
    q: Query<(&TrainingCameras, &GlobalTransform)>,
    mut gizmos: Gizmos,
) {
    if !view.frustums {
        return;
    }
    for (tc, g) in &q {
        let scale = f64::from(g.to_scale_rotation_translation().0.x).max(1e-9);
        let depth = FRUSTUM_DEPTH_M / scale;
        for (_, cam) in &tc.manifest.cameras {
            let k_inv = cam.k.inverse();
            let r_t = DMat3::from_mat4(cam.world_to_camera).transpose();
            let c = cam.center();
            let (w, h) = (f64::from(cam.width), f64::from(cam.height));
            let corner = |u: f64, v: f64| {
                let ray = k_inv * DVec3::new(u, v, 1.0);
                g.transform_point((c + r_t * (ray * depth)).as_vec3())
            };
            let pts = [
                corner(0.0, 0.0),
                corner(w, 0.0),
                corner(w, h),
                corner(0.0, h),
            ];
            let eye: Vec3 = g.transform_point(c.as_vec3());
            for i in 0..4 {
                gizmos.line(eye, pts[i], FRUSTUM);
                gizmos.line(pts[i], pts[(i + 1) % 4], FRUSTUM);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::app::App;
    use bevy::transform::components::Transform;
    use ftgsxr_core::{Fov, ModelTime, read_info};
    use ftgsxr_testkit::{fixtures_dir, scratch_dir};

    use super::*;

    fn scratch(name: &str) -> PathBuf {
        scratch_dir(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/tmp"),
            "ftgsxr_app_unit",
            name,
        )
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        a.distance(b) < 1e-4
    }

    // Q-219: the inverse of E6's desktop rule, the renderer's forward map.
    #[test]
    fn a_jump_pose_inverts_e6s_desktop_rule() {
        let p = Vec3::new(0.3, -1.2, 2.5);
        let q = Quat::from_euler(bevy::math::EulerRot::YXZ, 0.7, -0.3, 0.2);
        let cam = Camera::from_stage_pose(
            p,
            q,
            Fov::Vertical {
                radians: 70f32.to_radians(),
            },
            640,
            480,
        );
        let (pose, fov) = camera_pose(&cam, &GlobalTransform::IDENTITY);
        assert!(close(pose.translation, p), "{pose:?}");
        assert!(pose.rotation.angle_between(q) < 1e-4, "{pose:?}");
        assert!((fov - 70.0).abs() < 1e-3, "{fov}");
        // In an instance's frame: its scale moves the position, not the rotation.
        let inst = Transform {
            translation: Vec3::new(1.0, 0.5, -2.0),
            rotation: Quat::from_rotation_y(1.1),
            scale: Vec3::splat(2.0),
        };
        let g = GlobalTransform::from(inst);
        let (posed, _) = camera_pose(&cam, &g);
        assert!(
            close(posed.translation, inst.transform_point(p)),
            "{posed:?}"
        );
        assert!(posed.rotation.angle_between(inst.rotation * q) < 1e-4);
        assert_eq!(posed.scale, 1.0);
    }

    fn app() -> App {
        let mut app = App::new();
        app.add_message::<JumpToCamera>()
            .add_message::<CameraJump>()
            .add_message::<Intent>()
            .add_message::<ToastRequest>()
            .add_observer(find_manifest)
            .add_systems(bevy::app::Update, run_jumps);
        app
    }

    fn model(path: PathBuf) -> LoadedModel {
        let bytes = std::fs::read(&path).unwrap();
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let info = read_info(&bytes[8..8 + n], bytes.len() as u64).unwrap();
        LoadedModel {
            source: ModelRef::File(path),
            identity: "0".repeat(64),
            time: ModelTime::from_span(&info.time),
            info,
            name: "P1".into(),
            reduced: false,
            audio: None,
            frame_step_s: 1.0 / 30.0,
        }
    }

    // Q-219: §1.7, "jump to camera": the fly camera takes the camera's pose and field
    // of view in the instance's frame, the clock the image's timestamp.
    #[test]
    fn a_manifest_beside_the_model_offers_its_cameras_and_a_jump() {
        let dir = scratch("training-cameras");
        let src = fixtures_dir();
        std::fs::copy(
            src.join("probes/P1.safetensors"),
            dir.join("P1.safetensors"),
        )
        .unwrap();
        std::fs::copy(
            src.join("checkpoint/scene/scene.json"),
            dir.join("scene.json"),
        )
        .unwrap();
        let mut app = app();
        let mut m = model(dir.join("P1.safetensors"));
        m.time.offset_s = 0.5;
        let inst = Transform::from_xyz(1.0, 0.0, -1.0).with_scale(Vec3::splat(0.5));
        let e = app.world_mut().spawn((m, GlobalTransform::from(inst))).id();
        app.update();
        let tc = app
            .world()
            .get::<TrainingCameras>(e)
            .expect("the manifest is found")
            .clone();
        assert_eq!(tc.manifest.images.len(), 4);
        let img = &tc.manifest.images[3];
        assert_eq!(img.split, "test");
        let (_, camera) = tc
            .manifest
            .cameras
            .iter()
            .find(|(n, _)| *n == img.camera)
            .unwrap();
        app.world_mut().write_message(JumpToCamera {
            instance: e,
            camera: *camera,
            timestamp: img.timestamp,
            time_unit: tc.manifest.time_unit.clone(),
        });
        app.update();
        let jumps: Vec<CameraJump> = app
            .world_mut()
            .resource_mut::<Messages<CameraJump>>()
            .drain()
            .collect();
        assert_eq!(jumps.len(), 1);
        // OpenCV: the camera looks along +z of its frame, y down (§1.14 Frames).
        let r_t = DMat3::from_mat4(camera.world_to_camera).transpose();
        let fwd = inst.rotation * (r_t * DVec3::Z).as_vec3();
        let up = inst.rotation * (r_t * DVec3::NEG_Y).as_vec3();
        let j = jumps[0];
        assert!(close(
            j.pose.translation,
            inst.transform_point(camera.center().as_vec3())
        ));
        assert!(close(j.pose.rotation * Vec3::NEG_Z, fwd), "{j:?}");
        assert!(close(j.pose.rotation * Vec3::Y, up), "{j:?}");
        let want_fov = 2.0 * (30.0f32 / 80.0).atan().to_degrees();
        assert!((j.fov_deg - want_fov).abs() < 1e-3, "{}", j.fov_deg);
        let intents: Vec<Intent> = app
            .world_mut()
            .resource_mut::<Messages<Intent>>()
            .drain()
            .collect();
        let t = 0.5 + img.timestamp;
        assert_eq!(
            intents,
            vec![
                Intent::Playback(PlaybackCommand::Pause),
                Intent::Playback(PlaybackCommand::Seek(t))
            ]
        );
    }

    // Q-230, Q-231: from Open to Close (or its replacement), the request's background is
    // the override and its degree the extra cap; stepping keeps them, the governor's
    // `SplatTierState` and the settings' background are untouched, and Close clears both.
    #[test]
    fn an_open_request_sets_and_close_clears_the_overrides() {
        use ftgsxr_render::{Foveation, QualityTier, SortMode, SplatShLimit, SplatTierState};
        let mut app = App::new();
        let governor = SplatTierState {
            tier: QualityTier::desktop(),
            render_scale: 1.0,
            sh_cap: 2,
            sort_mode: SortMode::PerView,
            foveation: Foveation::Off,
        };
        app.init_resource::<TrainingView>()
            .init_resource::<BackgroundOverride>()
            .init_resource::<SplatShLimit>()
            .insert_resource(governor.clone())
            .add_systems(bevy::app::Update, follow_replay);
        let overrides = |app: &App| {
            (
                app.world().resource::<BackgroundOverride>().0,
                app.world().resource::<SplatShLimit>().0,
            )
        };
        app.update();
        assert_eq!(overrides(&app), (None, None));
        let mut replay = open_request(&fixtures_dir().join("probes/P1.request.json")).unwrap();
        assert!(replay.request.items.len() > 1);
        replay.request.background = [0.2, 0.3, 0.4];
        replay.request.sh_degree = Some(1);
        app.world_mut().resource_mut::<TrainingView>().replay = Some(replay);
        app.update();
        assert_eq!(overrides(&app), (Some([0.2, 0.3, 0.4]), Some(1)));
        app.world_mut()
            .resource_mut::<TrainingView>()
            .replay
            .as_mut()
            .unwrap()
            .item = 1;
        app.update();
        assert_eq!(overrides(&app), (Some([0.2, 0.3, 0.4]), Some(1)));
        // Another request in its place: P1's own, black and of the model's degree.
        let other = open_request(&fixtures_dir().join("probes/P1.request.json")).unwrap();
        app.world_mut().resource_mut::<TrainingView>().replay = Some(other);
        app.update();
        assert_eq!(overrides(&app), (Some([0.0, 0.0, 0.0]), None));
        app.world_mut().resource_mut::<TrainingView>().replay = None;
        app.update();
        assert_eq!(overrides(&app), (None, None));
        assert_eq!(*app.world().resource::<SplatTierState>(), governor);
    }

    // Q-219: a camera of another time unit is refused with a warning.
    #[test]
    fn a_camera_of_another_time_unit_is_refused() {
        let mut app = app();
        let e = app
            .world_mut()
            .spawn((
                model(fixtures_dir().join("probes/P1.safetensors")),
                GlobalTransform::IDENTITY,
            ))
            .id();
        let camera = Camera::from_stage_pose(
            Vec3::ZERO,
            Quat::IDENTITY,
            Fov::Vertical { radians: 1.0 },
            8,
            8,
        );
        app.world_mut().write_message(JumpToCamera {
            instance: e,
            camera,
            timestamp: 1.0,
            time_unit: "ms".into(),
        });
        app.update();
        let w = app.world_mut();
        assert_eq!(w.resource_mut::<Messages<CameraJump>>().drain().count(), 0);
        let toasts: Vec<ToastRequest> =
            w.resource_mut::<Messages<ToastRequest>>().drain().collect();
        assert_eq!(toasts.len(), 1);
        assert_eq!(toasts[0].toast.text, "Camera time unit ms, model s");
    }

    // §1.7 ("or an index entry names one"), Q-219: an HTTP model whose index entry's
    // `scene` manifest was downloaded offers that manifest's cameras; without it, none.
    #[test]
    fn an_index_scene_offers_its_cameras_to_an_http_model() {
        let mut app = app();
        let manifest_path = fixtures_dir().join("checkpoint/scene/scene.json");
        let want =
            ftgsxr_core::read_scene_manifest(&std::fs::read(&manifest_path).unwrap()).unwrap();
        let mut m = model(fixtures_dir().join("probes/P1.safetensors"));
        m.source = ModelRef::Http(
            ftgsxr_io::Url::parse("http://127.0.0.1:1/models/P1.safetensors").unwrap(),
        );
        let with = app
            .world_mut()
            .spawn((
                IndexScene(manifest_path.clone()),
                m.clone(),
                GlobalTransform::IDENTITY,
            ))
            .id();
        // A downloaded file that is not a `ftgs-scene` manifest (a checkpoint's) offers none.
        let not_a_scene = app
            .world_mut()
            .spawn((
                IndexScene(fixtures_dir().join("checkpoint/tiny/manifest.json")),
                m.clone(),
                GlobalTransform::IDENTITY,
            ))
            .id();
        let without = app.world_mut().spawn((m, GlobalTransform::IDENTITY)).id();
        app.update();
        let tc = app
            .world()
            .get::<TrainingCameras>(with)
            .expect("the index scene's cameras")
            .clone();
        assert_eq!(tc.path, manifest_path);
        assert_eq!(tc.manifest, want);
        assert!(app.world().get::<TrainingCameras>(not_a_scene).is_none());
        assert!(app.world().get::<TrainingCameras>(without).is_none());
        // A jump to one of its cameras moves the fly camera to that camera's center.
        let (_, camera) = tc.manifest.cameras[0].clone();
        app.world_mut().write_message(JumpToCamera {
            instance: with,
            camera,
            timestamp: tc.manifest.images[0].timestamp,
            time_unit: tc.manifest.time_unit.clone(),
        });
        app.update();
        let jumps: Vec<CameraJump> = app
            .world_mut()
            .resource_mut::<Messages<CameraJump>>()
            .drain()
            .collect();
        assert_eq!(jumps.len(), 1);
        assert!(close(jumps[0].pose.translation, camera.center().as_vec3()));
    }
}
