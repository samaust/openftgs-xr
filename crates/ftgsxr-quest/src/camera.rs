// SPDX-License-Identifier: Apache-2.0

//! The passthrough camera's thumbnails (§1.9, §2.5 The shim, D-15): the array
//! `pollCameraThumb` returns, parsed (Q-386), its lens pose composed with the head's
//! stage pose into `CameraThumb.camera` (Q-387), and `startCamera`'s codes as `start`'s
//! texts (Q-392). Built on the host for its tests, as `tree` is.

use bevy::math::{Quat, Vec3};
use ftgsxr_app::{Camera, CameraThumb, Fov};

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

/// Parses `pollCameraThumb`'s array (Q-386): 9216 RGB floats, the timestamp as whole
/// seconds and fraction, then 11 floats of lens pose and field of view when known. Another
/// length is `Err` naming it, which the caller logs as a warning.
// Q-386
pub(crate) fn parse(array: &[f32]) -> Result<Parsed, String> {
    let n = array.len();
    if n != RGB + 2 && n != RGB + 13 {
        return Err(format!(
            "a camera thumbnail of {n} floats, not {} or {}: dropped",
            RGB + 2,
            RGB + 13
        ));
    }
    let at_s = f64::from(array[RGB]) + f64::from(array[RGB + 1]);
    let lens = (n == RGB + 13).then(|| {
        let l = &array[RGB + 2..];
        Lens {
            position: Vec3::new(l[0], l[1], l[2]),
            rotation: Quat::from_xyzw(l[3], l[4], l[5], l[6]),
            fov: [l[7], l[8], l[9], l[10]],
        }
    });
    Ok(Parsed {
        rgb: array[..RGB].to_vec(),
        at_s,
        lens,
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

/// `startCamera`'s code as `CameraFrames::start`'s result (Q-392): 0 opening; 1 and 2
/// begin with §1.15's codes; 3 (or an unknown code) is the open's failure, with no code.
// Q-392
pub(crate) fn start_result(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err("E-XR-UNSUPPORTED: no passthrough camera".into()),
        2 => Err("E-XR-PERMISSION: the camera permissions are not granted".into()),
        _ => Err(format!(
            "the passthrough camera could not be opened: code {code}, the shim's log has the \
             cause"
        )),
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

    /// Q-386: 9229 floats carry the lens's translation, unit quaternion (x, y, z, w) and
    /// the four angles.
    #[test]
    fn parse_with_pose() {
        let a = array(&[0.1, -0.2, -0.3, 0.0, 0.0, 0.6, 0.8, -0.7, 0.6, 0.5, -0.4]);
        assert_eq!(a.len(), 9229);
        let p = parse(&a).expect("a thumbnail");
        assert_eq!(p.at_s, 1234.5);
        let lens = p.lens.expect("a lens");
        assert_eq!(lens.position, Vec3::new(0.1, -0.2, -0.3));
        assert_eq!(lens.rotation, Quat::from_xyzw(0.0, 0.0, 0.6, 0.8));
        assert_eq!(lens.fov, [-0.7, 0.6, 0.5, -0.4]);
    }

    /// Q-386: any other length gives no thumbnail and a text naming the length.
    #[test]
    fn parse_rejects_other_lengths() {
        for n in [0, 9216, 9217, 9220, 9230] {
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
