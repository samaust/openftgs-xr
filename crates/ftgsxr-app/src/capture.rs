// SPDX-License-Identifier: Apache-2.0

//! Screenshots (§1.7 Screenshots, §2.1 `ftgsxr_app::capture`): F12's `Intent::Screenshot`
//! becomes a `ScreenshotRequest` for the desktop camera; the UI is hidden for the captured
//! frame, Bevy's screenshot of the camera's render target is taken, the PNG is encoded
//! with `image` at the target's resolution, an iTXt chunk with the player's version, the
//! loaded models' identities and poses, the clock time and the camera is inserted (Q-150),
//! and the file is written atomically to `<screenshots>/<date>-<time>.png` (§1.11, Q-14).

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use bevy::app::{App, Plugin, PostUpdate, Update};
use bevy::camera::{Camera, Projection, RenderTarget};
use bevy::ecs::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::transform::TransformSystems;
use bevy::transform::components::GlobalTransform;
use image::ImageEncoder;
use image::codecs::png::PngEncoder;

use crate::intent::Intent;
use crate::library::LoadedModel;
use crate::manipulate::now_rfc3339;
use crate::platform::PlatformRes;
use crate::playback::Playback;
use crate::rig::{CameraController, DesktopCamera};
use crate::ui::{Toast, ToastLevel, ToastRequest, UiSet, UiState};

/// A screenshot of what `camera` renders (§2.1). Like `Intent`, it derives `Message`
/// (written with a `MessageWriter`) and `Event` (Q-166).
#[derive(Message, Event, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenshotRequest {
    /// The camera whose render target is captured and whose pose the chunk names.
    pub camera: Entity,
}

/// The iTXt chunk's keyword (Q-150).
pub(crate) const KEYWORD: &str = "openftgs-xr";

/// Frames a screenshot may wait for Bevy's capture before it is given up (Q-T287.2).
const EXPIRY_FRAMES: u32 = 120;

/// The screenshot in flight: at most one (Q-T287.2).
#[derive(Resource, Default)]
struct Pending(Option<InFlight>);

struct InFlight {
    camera: Entity,
    /// Bevy's `Screenshot` entity, once spawned (`PostUpdate`).
    shot: Option<Entity>,
    /// The chunk's JSON, gathered in the frame captured.
    text: String,
    frames: u32,
}

/// The module's plugin, added by `AppPlugin` after `AudioPlugin`.
pub(crate) struct CapturePlugin;

impl Plugin for CapturePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<ScreenshotRequest>()
            .add_message::<Intent>()
            .add_message::<ToastRequest>()
            .init_resource::<Pending>()
            .add_systems(
                Update,
                (screenshot_intents, take_requests, expire)
                    .chain()
                    .after(UiSet),
            )
            .add_systems(
                PostUpdate,
                spawn_screenshot.after(TransformSystems::Propagate),
            );
    }
}

/// `Intent::Screenshot` (F12) asks for the desktop camera, else the lowest-order active
/// camera (Q-T287.1, Q-180's head).
// Q-T287.1
fn screenshot_intents(
    mut intents: MessageReader<Intent>,
    mut out: MessageWriter<ScreenshotRequest>,
    desktop: Query<Entity, With<DesktopCamera>>,
    cameras: Query<(Entity, &Camera)>,
) {
    for i in intents.read() {
        if !matches!(i, Intent::Screenshot) {
            continue;
        }
        let camera = desktop.iter().next().or_else(|| {
            cameras
                .iter()
                .filter(|(_, c)| c.is_active)
                .min_by_key(|(_, c)| c.order)
                .map(|(e, _)| e)
        });
        if let Some(camera) = camera {
            out.write(ScreenshotRequest { camera });
        }
    }
}

/// A request starts a screenshot unless one is in flight (Q-T287.2) and hides the UI for
/// this frame, whose egui pass runs in `PostUpdate`.
fn take_requests(
    mut requests: MessageReader<ScreenshotRequest>,
    mut pending: ResMut<Pending>,
    ui: Option<ResMut<UiState>>,
) {
    let Some(r) = requests.read().last().copied() else {
        return;
    };
    if pending.0.is_some() {
        return;
    }
    pending.0 = Some(InFlight {
        camera: r.camera,
        shot: None,
        text: String::new(),
        frames: 0,
    });
    if let Some(mut ui) = ui {
        ui.hide_for_one_frame();
    }
}

/// A screenshot Bevy never answers (a window without a surface) ends with a warning.
// Q-T287.2
fn expire(
    mut commands: Commands,
    mut pending: ResMut<Pending>,
    mut toasts: MessageWriter<ToastRequest>,
) {
    let Some(p) = pending.0.as_mut() else {
        return;
    };
    p.frames += 1;
    if p.frames > EXPIRY_FRAMES {
        if let Some(shot) = p.shot {
            commands.entity(shot).try_despawn();
        }
        pending.0 = None;
        toasts.write(ToastRequest::warning("The screenshot was not captured"));
    }
}

/// After the transforms of the frame captured: the chunk's content, and Bevy's
/// screenshot of the camera's render target (PRD #275's spike check 4 and its fallback:
/// the target is the camera's own, a window on the desktop, an image in the `gpu_` lane).
#[allow(clippy::type_complexity)]
fn spawn_screenshot(
    mut commands: Commands,
    mut pending: ResMut<Pending>,
    cameras: Query<(&GlobalTransform, Option<&Projection>, Option<&RenderTarget>)>,
    models: Query<(Entity, &LoadedModel, &GlobalTransform)>,
    playback: Option<Res<Playback>>,
    controller: Option<Res<CameraController>>,
) {
    let Some(p) = pending.0.as_mut() else {
        return;
    };
    if p.shot.is_some() {
        return;
    }
    let Ok((camera, projection, target)) = cameras.get(p.camera) else {
        pending.0 = None;
        return;
    };
    let mut rows: Vec<_> = models.iter().collect();
    rows.sort_by_key(|r| r.0);
    let meta = Metadata {
        version: env!("CARGO_PKG_VERSION"),
        models: rows
            .into_iter()
            .map(|(_, m, g)| (m.identity.clone(), m.name.clone(), pose_of(g)))
            .collect(),
        clock_t: playback.map_or(0.0, |p| p.clock.time()),
        camera: pose_of(camera),
        fov_deg: match projection {
            Some(Projection::Perspective(p)) => Some(p.fov.to_degrees()),
            _ => None,
        },
        orbit: controller.is_some_and(|c| c.orbit),
    };
    p.text = meta.to_json();
    let target = target.cloned().unwrap_or_default();
    let shot = commands.spawn(Screenshot(target)).observe(captured).id();
    p.shot = Some(shot);
}

/// Bevy's image: encoded, the chunk inserted, written; the toast line says where.
fn captured(
    ev: On<ScreenshotCaptured>,
    mut pending: ResMut<Pending>,
    platform: Option<Res<PlatformRes>>,
    mut toasts: MessageWriter<ToastRequest>,
) {
    let Some(p) = pending.0.take_if(|p| p.shot == Some(ev.entity)) else {
        return;
    };
    let Some(platform) = platform else {
        return;
    };
    let dir = platform.0.paths().screenshots.clone();
    match write_screenshot(&dir, &ev.image, &p.text) {
        Ok(path) => {
            toasts.write(ToastRequest(Toast {
                text: format!("Screenshot saved to {}", path.display()),
                level: ToastLevel::Info,
            }));
        }
        Err((path, e)) => {
            toasts.write(ToastRequest::error(format!(
                "E-IO: Could not write {}: {e}",
                path.display()
            )));
        }
    }
}

/// Encodes `image` as an RGB PNG (Q-T287.3), inserts the chunk and writes the file.
fn write_screenshot(
    dir: &Path,
    image: &bevy::image::Image,
    text: &str,
) -> Result<PathBuf, (PathBuf, String)> {
    let png = encode(image, text).map_err(|e| (dir.to_path_buf(), e))?;
    let path = free_name(dir, &now_rfc3339());
    write_atomic(&path, &png).map_err(|e| (path.clone(), e.to_string()))?;
    Ok(path)
}

fn encode(image: &bevy::image::Image, text: &str) -> Result<Vec<u8>, String> {
    // Q-T287.3: the alpha channel is dropped, as Bevy's `save_to_disk` does.
    let rgb = image
        .clone()
        .try_into_dynamic()
        .map_err(|e| e.to_string())?
        .to_rgb8();
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| e.to_string())?;
    insert_itxt(&png, KEYWORD, text).ok_or_else(|| "not a PNG".to_owned())
}

/// `<dir>/YYYYMMDD-HHMMSS.png` in UTC from `now` (RFC 3339), with `-2`, `-3`, … before
/// the suffix when the name is taken (Q-T287.4).
// Q-T287.4
pub(crate) fn free_name(dir: &Path, now: &str) -> PathBuf {
    let digits: String = now.chars().take(19).filter(char::is_ascii_digit).collect();
    let stem = format!("{}-{}", &digits[..8], &digits[8..]);
    let mut path = dir.join(format!("{stem}.png"));
    let mut k = 2;
    while path.exists() {
        path = dir.join(format!("{stem}-{k}.png"));
        k += 1;
    }
    path
}

/// Writes to a temporary sibling and renames it over `path` (§1.11), creating the
/// directory (Q-14).
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir)?;
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let temporary = dir.join(format!(".{name}.tmp"));
    fs::write(&temporary, bytes)
        .and_then(|()| fs::rename(&temporary, path))
        .inspect_err(|_| {
            let _ = fs::remove_file(&temporary);
        })
}

/// A pose in §1.11's JSON shape, rotation (x, y, z, w) (§1.14).
struct PoseJson {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: f32,
}

// Q-T287.1: stage-space poses (the global transforms).
fn pose_of(g: &GlobalTransform) -> PoseJson {
    let (s, r, t) = g.to_scale_rotation_translation();
    PoseJson {
        translation: t.to_array(),
        rotation: r.to_array(),
        scale: s.x,
    }
}

/// The chunk's content (Q-150).
struct Metadata {
    version: &'static str,
    models: Vec<(String, String, PoseJson)>,
    clock_t: f64,
    camera: PoseJson,
    fov_deg: Option<f32>,
    orbit: bool,
}

impl Metadata {
    fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\"version\":");
        json_str(&mut s, self.version);
        s.push_str(",\"models\":[");
        for (k, (identity, name, pose)) in self.models.iter().enumerate() {
            if k > 0 {
                s.push(',');
            }
            s.push_str("{\"identity\":");
            json_str(&mut s, identity);
            s.push_str(",\"name\":");
            json_str(&mut s, name);
            s.push_str(",\"pose\":");
            json_pose(&mut s, pose);
            s.push('}');
        }
        s.push_str("],\"clock_t\":");
        json_num(&mut s, self.clock_t);
        s.push_str(",\"camera\":{\"pose\":");
        json_pose(&mut s, &self.camera);
        s.push_str(",\"fov_deg\":");
        match self.fov_deg {
            Some(f) => json_num(&mut s, f),
            None => s.push_str("null"),
        }
        s.push_str(",\"mode\":");
        json_str(&mut s, if self.orbit { "orbit" } else { "fly" });
        s.push_str("}}");
        s
    }
}

fn json_pose(s: &mut String, p: &PoseJson) {
    s.push_str("{\"translation\":");
    json_nums(s, &p.translation);
    s.push_str(",\"rotation\":");
    json_nums(s, &p.rotation);
    s.push_str(",\"scale\":");
    json_num(s, p.scale);
    s.push('}');
}

fn json_nums(s: &mut String, v: &[f32]) {
    s.push('[');
    for (k, x) in v.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        json_num(s, *x);
    }
    s.push(']');
}

/// A finite number in Rust's shortest round-trip form; a non-finite one is `null`.
fn json_num<T: Into<f64> + std::fmt::Display + Copy>(s: &mut String, x: T) {
    if x.into().is_finite() {
        let _ = write!(s, "{x}");
    } else {
        s.push_str("null");
    }
}

/// A JSON string with RFC 8259's escapes.
pub(crate) fn json_str(s: &mut String, v: &str) {
    s.push('"');
    for c in v.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(s, "\\u{:04x}", c as u32);
            }
            c => s.push(c),
        }
    }
    s.push('"');
}

/// `png` with an uncompressed iTXt chunk (keyword, no language tag, no translated
/// keyword, UTF-8 text) inserted after IHDR; `None` when `png` does not start with the
/// signature and IHDR (Q-150).
// Q-150
pub(crate) fn insert_itxt(png: &[u8], keyword: &str, text: &str) -> Option<Vec<u8>> {
    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    const IHDR_END: usize = 8 + 4 + 4 + 13 + 4;
    if png.len() < IHDR_END || png[..8] != SIGNATURE || &png[12..16] != b"IHDR" {
        return None;
    }
    let mut data = Vec::with_capacity(keyword.len() + 5 + text.len());
    data.extend_from_slice(keyword.as_bytes());
    data.extend_from_slice(&[0, 0, 0, 0, 0]); // NUL, flag, method, language NUL, translated NUL
    data.extend_from_slice(text.as_bytes());
    let mut out = Vec::with_capacity(png.len() + data.len() + 12);
    out.extend_from_slice(&png[..IHDR_END]);
    out.extend_from_slice(&u32::try_from(data.len()).ok()?.to_be_bytes());
    let mut typed = b"iTXt".to_vec();
    typed.extend_from_slice(&data);
    out.extend_from_slice(&typed);
    out.extend_from_slice(&crc32(&typed).to_be_bytes());
    out.extend_from_slice(&png[IHDR_END..]);
    Some(out)
}

/// The PNG chunk CRC (ISO 3309, reflected polynomial 0xEDB88320).
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_of_the_standard_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn the_chunk_follows_ihdr_and_its_crc_matches_images_own() {
        let mut png = Vec::new();
        PngEncoder::new(&mut png)
            .write_image(&[1, 2, 3, 4, 5, 6], 2, 1, image::ExtendedColorType::Rgb8)
            .unwrap();
        // `image`'s IHDR CRC computed the same way.
        assert_eq!(crc32(&png[12..29]).to_be_bytes(), png[29..33]);
        let out = insert_itxt(&png, KEYWORD, "{\"a\":\"é\"}").unwrap();
        assert_eq!(&out[37..41], b"iTXt");
        let decoded = image::load_from_memory(&out).unwrap().to_rgb8();
        assert_eq!(decoded.as_raw(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn json_strings_are_escaped() {
        let mut s = String::new();
        json_str(&mut s, "a\"b\\c\nd\u{1}é");
        assert_eq!(s, "\"a\\\"b\\\\c\\nd\\u0001é\"");
    }

    #[test]
    fn names_are_the_utc_time_and_never_overwrite() {
        let dir = std::env::temp_dir().join(format!("ftgsxr-capture-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let now = "2026-11-02T13:04:05.123Z";
        let first = free_name(&dir, now);
        assert_eq!(first, dir.join("20261102-130405.png"));
        fs::write(&first, b"x").unwrap();
        assert_eq!(free_name(&dir, now), dir.join("20261102-130405-2.png"));
        let _ = fs::remove_dir_all(&dir);
    }
}
