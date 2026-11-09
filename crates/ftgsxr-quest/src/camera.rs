// SPDX-License-Identifier: Apache-2.0

//! The passthrough camera's thumbnails (§1.9, §2.5 The shim, D-15): the array
//! `pollCameraThumb` returns, parsed and its raw lens converted (Q-386), its lens pose
//! composed with the head's stage pose into `CameraThumb.camera` (Q-387), and
//! `startCamera`'s codes as `start`'s texts (Q-392). Built on the host for its tests, as
//! `tree` is.

use bevy::math::{Quat, Vec3};
use ftgsxr_app::{Camera, CameraThumb, Fov, Permission};
use ftgsxr_xr::XrError;

/// The thumbnail's size (§1.9, §2.5: 64 × 48 RGB, 9216 floats).
pub(crate) const WIDTH: u32 = 64;
/// The thumbnail's height.
pub(crate) const HEIGHT: u32 = 48;
const RGB: usize = (WIDTH * HEIGHT * 3) as usize;

/// One array from the shim, parsed.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Parsed {
    pub(crate) rgb: Vec<f32>,
    pub(crate) at_s: f64,
    pub(crate) lens: Option<Lens>,
}

/// The lens pose relative to the headset, in OpenXR's axes (looking along −z with y up),
/// and the field of view's four angles (left, right, up, down, radians).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Lens {
    pub(crate) position: Vec3,
    pub(crate) rotation: Quat,
    pub(crate) fov: [f32; 4],
}

/// The raw lens characteristics after the timestamp (Q-386): `LENS_POSE_TRANSLATION` (3),
/// `LENS_POSE_ROTATION` (4), fx, fy, cx and cy of `LENS_INTRINSIC_CALIBRATION` (4), the
/// pre-correction active array (left, top, right, bottom) and the stream's size (2).
const RAW_LENS: usize = 17;

/// Parses `pollCameraThumb`'s array (Q-386): 9216 RGB floats, the timestamp as whole
/// seconds and fraction, then 17 floats of raw lens characteristics when known, which
/// `lens` converts. Another length is `Err` naming it, which the caller logs as a warning.
// Q-386
pub(crate) fn parse(array: &[f32]) -> Result<Parsed, String> {
    let n = array.len();
    if n != RGB + 2 && n != RGB + 2 + RAW_LENS {
        return Err(format!(
            "a camera thumbnail of {n} floats, not {} or {}: dropped",
            RGB + 2,
            RGB + 2 + RAW_LENS
        ));
    }
    let at_s = f64::from(array[RGB]) + f64::from(array[RGB + 1]);
    let lens = if n == RGB + 2 + RAW_LENS {
        lens(&array[RGB + 2..])
    } else {
        None
    };
    Ok(Parsed {
        rgb: array[..RGB].to_vec(),
        at_s,
        lens,
    })
}

/// The lens from its raw characteristics (Q-386, D-15: the shim only reads them), or
/// `None` when they give no camera: a value not finite, a zero rotation, a focal length
/// not above zero, or an empty array or stream.
// Q-386
fn lens(raw: &[f32]) -> Option<Lens> {
    if raw.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let position = Vec3::new(raw[0], raw[1], raw[2]);
    // `LENS_POSE_ROTATION` rotates Android's sensor axes (x right, y up, z back on the
    // Quest 3: OpenXR's, as observed on the owner's Quest 3 with `adb shell dumpsys
    // media.camera`, 2026-11-09, #667) into the camera's (x along the long side, y down the
    // short side, z along the optical axis): the camera's orientation is its inverse, and
    // OpenXR's camera (looking along −z with y up) that turned by π about x, the quaternion
    // (1, 0, 0, 0).
    let q = Quat::from_xyzw(raw[3], raw[4], raw[5], raw[6]);
    if q.length_squared() < 1e-12 {
        return None;
    }
    let rotation = (q.conjugate() * Quat::from_xyzw(1.0, 0.0, 0.0, 0.0)).normalize();
    let [fx, fy, cx, cy] = [raw[7], raw[8], raw[9], raw[10]].map(f64::from);
    let [left, top, right, bottom] = [raw[11], raw[12], raw[13], raw[14]].map(f64::from);
    let [stream_w, stream_h] = [raw[15], raw[16]].map(f64::from);
    let (w, h) = (right - left, bottom - top);
    if fx <= 0.0 || fy <= 0.0 || w <= 0.0 || h <= 0.0 || stream_w <= 0.0 || stream_h <= 0.0 {
        return None;
    }
    // The intrinsics are in the pre-correction array's pixels from its top-left; the stream
    // sees the array's centred crop to the stream's aspect, against which the angles are
    // taken (left and down negative).
    let aspect = stream_w / stream_h;
    let (cw, ch) = ((h * aspect).min(w), (w / aspect).min(h));
    let (cx, cy) = (cx - (w - cw) / 2.0, cy - (h - ch) / 2.0);
    let fov = [
        -(cx / fx).atan(),
        ((cw - cx) / fx).atan(),
        (cy / fy).atan(),
        -((ch - cy) / fy).atan(),
    ]
    .map(|a| a as f32);
    Some(Lens {
        position,
        rotation,
        fov,
    })
}

/// The thumbnail `latest` returns (Q-387): the lens pose under the head's stage pose
/// (`head_position`, `head_rotation`, the current frame's `XrHands.head`) as
/// `Camera::from_stage_pose` with `Fov::Angles` at 64 × 48; `None` without a lens.
// Q-387
pub(crate) fn thumb(parsed: Parsed, head_position: Vec3, head_rotation: Quat) -> CameraThumb {
    // The head pose is the frame's, not the capture's (20–40 ms earlier, E.2), and the
    // lens's origin is the headset's gyroscope (`LENS_POSE_REFERENCE` on the Quest 3, as
    // observed on the owner's Quest 3 with `adb shell dumpsys media.camera`, 2026-11-09,
    // #667), not the views' midpoint: both errors are centimetres against E14's Δ = 0.5 m
    // (Q-387).
    let camera = parsed.lens.map(|lens| {
        let [left, right, up, down] = lens.fov;
        Camera::from_stage_pose(
            head_position + head_rotation * lens.position,
            (head_rotation * lens.rotation).normalize(),
            Fov::Angles {
                left,
                right,
                up,
                down,
            },
            WIDTH,
            HEIGHT,
        )
    });
    CameraThumb {
        width: WIDTH,
        height: HEIGHT,
        rgb: parsed.rgb,
        at_s: parsed.at_s,
        camera,
    }
}

/// `CameraFrames::start`'s error when the camera did not open, with its `cause`.
pub(crate) fn not_opened(cause: impl std::fmt::Display) -> String {
    format!("the passthrough camera could not be opened: {cause}")
}

/// `startCamera`'s code as `CameraFrames::start`'s result (Q-392): 0 opening; 1 and 2
/// begin with §1.15's codes; 3 (or an unknown code) is the open's failure, with no code.
// Q-392
pub(crate) fn start_result(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err(format!(
            "{}: no passthrough camera",
            XrError::Unsupported("").code()
        )),
        2 => Err(format!(
            "{}: the camera permissions are not granted",
            XrError::Permission(Permission::Camera).code()
        )),
        _ => Err(not_opened(format!(
            "code {code}, the shim's log has the cause"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array(extra: &[f32]) -> Vec<f32> {
        let mut a: Vec<f32> = (0..RGB).map(|i| (i % 7) as f32 / 7.0).collect();
        a.extend_from_slice(&[1234.0, 0.5]);
        a.extend_from_slice(extra);
        a
    }

    /// Q-386: 9218 floats are a thumbnail with its timestamp and no lens.
    #[test]
    fn parse_without_pose() {
        let a = array(&[]);
        assert_eq!(a.len(), 9218);
        let p = parse(&a).expect("a thumbnail");
        assert_eq!(p.rgb, a[..9216]);
        assert_eq!(p.at_s, 1234.5);
        assert_eq!(p.lens, None);
    }

    /// The 17 raw floats of Q-386: translation, `LENS_POSE_ROTATION` (x, y, z, w), fx, fy,
    /// cx, cy, the pre-correction array (left, top, right, bottom), the stream's size.
    fn raw(rotation: [f32; 4], k: [f32; 4], array: [f32; 4]) -> Vec<f32> {
        let mut r = vec![0.1, -0.2, -0.3];
        r.extend_from_slice(&rotation);
        r.extend_from_slice(&k);
        r.extend_from_slice(&array);
        r.extend_from_slice(&[1280.0, 960.0]);
        r
    }

    fn lens_of(raw: &[f32]) -> Option<Lens> {
        let a = array(raw);
        assert_eq!(a.len(), 9235);
        let p = parse(&a).expect("a thumbnail");
        assert_eq!(p.at_s, 1234.5);
        assert_eq!(p.rgb, a[..9216]);
        p.lens
    }

    fn near(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-6
    }

    fn near_fov(got: [f32; 4], want: [f32; 4]) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?} {want:?}");
        }
    }

    const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
    const CENTRED: [f32; 4] = [640.0, 640.0, 640.0, 480.0];
    const ARRAY_4_3: [f32; 4] = [0.0, 0.0, 1280.0, 960.0];

    /// Q-386: 9235 floats carry the raw characteristics; the translation passes through.
    /// An identity `LENS_POSE_ROTATION` puts the camera's axes on the sensor's (Android's
    /// camera looks along its +z with its y down the image), so in OpenXR's axes the camera
    /// looks along the sensor's +z and its up is the sensor's −y: a turn by π about x.
    #[test]
    fn parse_with_raw_lens_identity_rotation() {
        let lens = lens_of(&raw(IDENTITY, CENTRED, ARRAY_4_3)).expect("a lens");
        assert_eq!(lens.position, Vec3::new(0.1, -0.2, -0.3));
        assert!(
            near(lens.rotation * Vec3::NEG_Z, Vec3::Z),
            "{}",
            lens.rotation
        );
        assert!(
            near(lens.rotation * Vec3::Y, Vec3::NEG_Y),
            "{}",
            lens.rotation
        );
        assert!(lens.rotation.is_normalized());
    }

    /// Q-386: `LENS_POSE_ROTATION` = a turn of +90° about the sensor's y rotates the
    /// sensor's axes into the camera's, so the camera's optical axis (its +z) is the
    /// sensor's R_y(−90°)·z = −x, and its image-down (+y) stays the sensor's +y.
    #[test]
    fn parse_with_raw_lens_rotated() {
        let q = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let lens = lens_of(&raw([q.x, q.y, q.z, q.w], CENTRED, ARRAY_4_3)).expect("a lens");
        assert!(
            near(lens.rotation * Vec3::NEG_Z, Vec3::NEG_X),
            "{}",
            lens.rotation
        );
        assert!(
            near(lens.rotation * Vec3::Y, Vec3::NEG_Y),
            "{}",
            lens.rotation
        );
    }

    /// Q-386: the principal point at the centre of a 4:3 array, f = 640 over 640 × 480
    /// half-sizes: ±atan(1) = ±π/4 across and ±atan(0.75) up and down, each pair of equal
    /// magnitude, left and down negative.
    #[test]
    fn fov_symmetric() {
        let lens = lens_of(&raw(IDENTITY, CENTRED, ARRAY_4_3)).expect("a lens");
        let (h, v) = (std::f32::consts::FRAC_PI_4, 0.75f32.atan());
        near_fov(lens.fov, [-h, h, v, -v]);
        // fy = 480 over the same half-height: all four angles π/4.
        let lens =
            lens_of(&raw(IDENTITY, [640.0, 480.0, 640.0, 480.0], ARRAY_4_3)).expect("a lens");
        let q = std::f32::consts::FRAC_PI_4;
        near_fov(lens.fov, [-q, q, q, -q]);
    }

    /// Q-386: an off-centre principal point (cx = 320 of 1280, f = 640): left −atan(0.5),
    /// right atan(1.5); up and down unchanged, ±atan(0.75).
    #[test]
    fn fov_off_centre() {
        let lens =
            lens_of(&raw(IDENTITY, [640.0, 640.0, 320.0, 480.0], ARRAY_4_3)).expect("a lens");
        let v = 0.75f32.atan();
        near_fov(lens.fov, [-(0.5f32.atan()), 1.5f32.atan(), v, -v]);
    }

    /// Q-386: a 1600 × 960 pre-correction array is cropped to 4:3 on its long side only:
    /// 1280 × 960 from x = 160, so cx = 800 becomes 640 in the crop; f = 800 gives
    /// ±atan(640/800) and ±atan(480/800). The array's left and top do not move the
    /// intrinsics, which are in the array's own pixels.
    #[test]
    fn fov_wide_array_crops_the_long_side() {
        let k = [800.0, 800.0, 800.0, 480.0];
        let (h, v) = (0.8f32.atan(), 0.6f32.atan());
        for array in [[0.0, 0.0, 1600.0, 960.0], [8.0, 4.0, 1608.0, 964.0]] {
            let lens = lens_of(&raw(IDENTITY, k, array)).expect("a lens");
            near_fov(lens.fov, [-h, h, v, -v]);
        }
    }

    /// Q-386: a tall array (960 × 1600) is cropped on its height only: 960 × 720 from
    /// y = 440, so cy = 800 becomes 360; f = 480 gives ±atan(1) across and ±atan(0.75) up
    /// and down.
    #[test]
    fn fov_tall_array_crops_the_height() {
        let k = [480.0, 480.0, 480.0, 800.0];
        let lens = lens_of(&raw(IDENTITY, k, [0.0, 0.0, 960.0, 1600.0])).expect("a lens");
        let (h, v) = (1.0f32.atan(), 0.75f32.atan());
        near_fov(lens.fov, [-h, h, v, -v]);
    }

    /// Q-386: characteristics that give no camera (a focal length not above zero, an empty
    /// array or stream, a value not finite) give a thumbnail without a lens.
    #[test]
    fn unusable_raw_lens_gives_no_lens() {
        let cases = [
            raw(IDENTITY, [0.0, 640.0, 640.0, 480.0], ARRAY_4_3),
            raw(IDENTITY, [640.0, -1.0, 640.0, 480.0], ARRAY_4_3),
            raw(IDENTITY, CENTRED, [0.0, 0.0, 0.0, 960.0]),
            raw([f32::NAN, 0.0, 0.0, 1.0], CENTRED, ARRAY_4_3),
            raw([0.0; 4], CENTRED, ARRAY_4_3),
        ];
        for r in cases {
            assert_eq!(lens_of(&r), None, "{r:?}");
        }
        let mut r = raw(IDENTITY, CENTRED, ARRAY_4_3);
        r[16] = 0.0;
        assert_eq!(lens_of(&r), None);
    }

    /// Q-386: any other length gives no thumbnail and a text naming the length, the old
    /// 9229-float layout included.
    #[test]
    fn parse_rejects_other_lengths() {
        for n in [0, 9216, 9217, 9220, 9229, 9230, 9236] {
            let e = parse(&vec![0.0; n]).expect_err("no thumbnail");
            assert!(e.contains(&n.to_string()), "{e}");
        }
    }

    /// Q-387: the lens pose under a translated head turned 90° to the left gives the
    /// stage-space camera: centre head + R·lens, looking along the head's forward (−x).
    #[test]
    fn thumb_composes_head_and_lens() {
        let (a, b) = (0.5f32.atan(), 0.375f32.atan());
        let parsed = Parsed {
            rgb: vec![0.25; RGB],
            at_s: 7.0,
            lens: Some(Lens {
                position: Vec3::new(0.1, 0.0, -0.2),
                rotation: Quat::IDENTITY,
                fov: [-a, a, b, -b],
            }),
        };
        let head_rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let t = thumb(parsed.clone(), Vec3::new(1.0, 2.0, 3.0), head_rotation);
        assert_eq!((t.width, t.height, t.at_s), (64, 48, 7.0));
        assert_eq!(t.rgb, parsed.rgb);
        let cam = t.camera.expect("a camera");
        let c = cam.center();
        // R_y(90°) takes (0.1, 0, −0.2) to (−0.2, 0, −0.1).
        assert!(
            (c - bevy::math::DVec3::new(0.8, 2.0, 2.9)).length() < 1e-5,
            "{c}"
        );
        // A point 1 m along −x from the centre is on the optical axis, 1 m ahead.
        let ahead = cam.world_to_camera * bevy::math::DVec4::new(-0.2, 2.0, 2.9, 1.0);
        assert!((ahead.truncate() - bevy::math::DVec3::new(0.0, 0.0, 1.0)).length() < 1e-5);
        // tan(a) = 0.5 over 32 px and tan(b) = 0.375 over 24 px: f = 64, centred.
        let k = cam.intrinsics();
        for (got, want) in [(k.fx, 64.0), (k.fy, 64.0), (k.cx, 32.0), (k.cy, 24.0)] {
            assert!((got - want).abs() < 1e-3, "{got} {want}");
        }
    }

    /// Q-387: no lens from the shim, no camera.
    #[test]
    fn thumb_without_lens_has_no_camera() {
        let p = parse(&array(&[])).expect("a thumbnail");
        let t = thumb(p, Vec3::ONE, Quat::IDENTITY);
        assert_eq!(t.camera, None);
        assert_eq!(t.rgb.len(), 9216);
    }

    /// Q-392: code 0 is `Ok`; 1 begins with E-XR-UNSUPPORTED, 2 with E-XR-PERMISSION, and
    /// 3 with no code at all.
    #[test]
    fn start_result_maps_codes() {
        assert_eq!(start_result(0), Ok(()));
        assert_eq!(
            start_result(1),
            Err("E-XR-UNSUPPORTED: no passthrough camera".into())
        );
        assert!(
            start_result(2)
                .unwrap_err()
                .starts_with("E-XR-PERMISSION: ")
        );
        for code in [3, 9] {
            let e = start_result(code).unwrap_err();
            assert!(!e.starts_with("E-"), "{e}");
            assert!(
                e.starts_with("the passthrough camera could not be opened"),
                "{e}"
            );
        }
    }
}
