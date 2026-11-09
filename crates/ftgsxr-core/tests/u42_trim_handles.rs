// SPDX-License-Identifier: Apache-2.0

//! U-42 (§3.1, `cpu` lane): E18's `Trim::handles` and `Trim::drag_handle`: a
//! hand-worked box's six face points and a sphere's seven points, none when off; each
//! face dragged in and out, past its opposite and back; a radius handle on each side
//! and at the floor; the center moved; a handle that does not match the mode. Exact.

use ftgsxr_core::{Trim, TrimHandle, TrimMode, Vec3};

fn boxed(min: [f32; 3], max: [f32; 3]) -> Trim {
    Trim {
        mode: TrimMode::Box,
        box_min: Vec3::from_array(min),
        box_max: Vec3::from_array(max),
        center: Vec3::new(5.0, 5.0, 5.0),
        radius: 2.0,
    }
}

fn sphere(center: [f32; 3], radius: f32) -> Trim {
    Trim {
        mode: TrimMode::Sphere,
        box_min: Vec3::new(-1.0, -1.0, -1.0),
        box_max: Vec3::ONE,
        center: Vec3::from_array(center),
        radius,
    }
}

/// The handles as a sorted list, so the test does not depend on their order.
fn sorted(mut h: Vec<(TrimHandle, Vec3)>) -> Vec<(TrimHandle, [f32; 3])> {
    let mut v: Vec<_> = h.drain(..).map(|(k, p)| (k, p.to_array())).collect();
    v.sort_by(|a, b| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)));
    v
}

#[test]
fn u42_a_box_has_a_handle_at_each_face_center() {
    // Box [-1, 1] × [0, 4] × [2, 3]: its center is (0, 2, 2.5).
    let t = boxed([-1.0, 0.0, 2.0], [1.0, 4.0, 3.0]);
    let expected = sorted(vec![
        (
            TrimHandle::Face {
                axis: 0,
                max: false,
            },
            Vec3::new(-1.0, 2.0, 2.5),
        ),
        (
            TrimHandle::Face { axis: 0, max: true },
            Vec3::new(1.0, 2.0, 2.5),
        ),
        (
            TrimHandle::Face {
                axis: 1,
                max: false,
            },
            Vec3::new(0.0, 0.0, 2.5),
        ),
        (
            TrimHandle::Face { axis: 1, max: true },
            Vec3::new(0.0, 4.0, 2.5),
        ),
        (
            TrimHandle::Face {
                axis: 2,
                max: false,
            },
            Vec3::new(0.0, 2.0, 2.0),
        ),
        (
            TrimHandle::Face { axis: 2, max: true },
            Vec3::new(0.0, 2.0, 3.0),
        ),
    ]);
    assert_eq!(sorted(t.handles()), expected);
}

#[test]
fn u42_a_sphere_has_its_center_and_six_radius_handles() {
    let t = sphere([1.0, 2.0, 3.0], 0.5);
    let expected = sorted(vec![
        (TrimHandle::Center, Vec3::new(1.0, 2.0, 3.0)),
        (
            TrimHandle::Radius {
                axis: 0,
                positive: false,
            },
            Vec3::new(0.5, 2.0, 3.0),
        ),
        (
            TrimHandle::Radius {
                axis: 0,
                positive: true,
            },
            Vec3::new(1.5, 2.0, 3.0),
        ),
        (
            TrimHandle::Radius {
                axis: 1,
                positive: false,
            },
            Vec3::new(1.0, 1.5, 3.0),
        ),
        (
            TrimHandle::Radius {
                axis: 1,
                positive: true,
            },
            Vec3::new(1.0, 2.5, 3.0),
        ),
        (
            TrimHandle::Radius {
                axis: 2,
                positive: false,
            },
            Vec3::new(1.0, 2.0, 2.5),
        ),
        (
            TrimHandle::Radius {
                axis: 2,
                positive: true,
            },
            Vec3::new(1.0, 2.0, 3.5),
        ),
    ]);
    assert_eq!(sorted(t.handles()), expected);
}

#[test]
fn u42_an_off_trim_has_no_handles() {
    let t = Trim {
        mode: TrimMode::Off,
        ..boxed([0.0; 3], [1.0; 3])
    };
    assert!(t.handles().is_empty());
}

#[test]
fn u42_each_face_dragged_in_and_out_moves_only_its_limit() {
    let t0 = boxed([-1.0, 0.0, 2.0], [1.0, 4.0, 3.0]);
    // A drag point displacement with every component set: a face takes only its own.
    let delta = Vec3::new(0.25, 0.5, -0.125);
    for axis in 0..3 {
        for max in [false, true] {
            for sign in [1.0f32, -1.0] {
                let d = sign * delta;
                let t = t0.drag_handle(TrimHandle::Face { axis, max }, d, 1e-3);
                let mut want = t0;
                if max {
                    want.box_max[axis] += d[axis];
                } else {
                    want.box_min[axis] += d[axis];
                }
                assert_eq!(t, want, "axis {axis}, max {max}, sign {sign}");
            }
        }
    }
}

#[test]
fn u42_a_face_past_its_opposite_carries_it_and_dragged_back_returns_it() {
    let t0 = boxed([-1.0, 0.0, 2.0], [1.0, 4.0, 3.0]);
    let face = TrimHandle::Face { axis: 0, max: true };
    // b_max,x = 1 dragged by −3 lands at −2, past b_min,x = −1, which follows to −2.
    let past = t0.drag_handle(face, Vec3::new(-3.0, 0.0, 0.0), 1e-3);
    assert_eq!(past.box_max, Vec3::new(-2.0, 4.0, 3.0));
    assert_eq!(past.box_min, Vec3::new(-2.0, 0.0, 2.0));
    // Every frame applies to t0: dragged back to −0.5, b_min,x is t0's −1 again.
    let back = t0.drag_handle(face, Vec3::new(-1.5, 0.0, 0.0), 1e-3);
    assert_eq!(back.box_max, Vec3::new(-0.5, 4.0, 3.0));
    assert_eq!(back.box_min, Vec3::new(-1.0, 0.0, 2.0));
    // And the b_min face past b_max carries b_max up.
    let up = t0.drag_handle(
        TrimHandle::Face {
            axis: 2,
            max: false,
        },
        Vec3::new(0.0, 0.0, 2.0),
        1e-3,
    );
    assert_eq!(up.box_min, Vec3::new(-1.0, 0.0, 4.0));
    assert_eq!(up.box_max, Vec3::new(1.0, 4.0, 4.0));
}

#[test]
fn u42_a_radius_handle_follows_its_side_and_stops_at_the_floor() {
    let t0 = sphere([1.0, 2.0, 3.0], 0.5);
    let delta = Vec3::new(0.25, -0.125, 0.375);
    // The + handle takes δ_a, the − handle −δ_a; only its own axis counts.
    let cases = [
        (0, true, 0.75),
        (0, false, 0.25),
        (1, true, 0.375),
        (1, false, 0.625),
        (2, true, 0.875),
        (2, false, 0.125),
    ];
    for (axis, positive, r) in cases {
        let t = t0.drag_handle(TrimHandle::Radius { axis, positive }, delta, 1e-3);
        assert_eq!(
            t,
            Trim { radius: r, ..t0 },
            "axis {axis}, positive {positive}"
        );
    }
    // Pulled through the center and beyond: r stops at the floor ρ.
    let t = t0.drag_handle(
        TrimHandle::Radius {
            axis: 1,
            positive: true,
        },
        Vec3::new(0.0, -2.0, 0.0),
        0.0625,
    );
    assert_eq!(
        t,
        Trim {
            radius: 0.0625,
            ..t0
        }
    );
}

#[test]
fn u42_the_center_moves_by_the_whole_displacement() {
    let t0 = sphere([1.0, 2.0, 3.0], 0.5);
    let t = t0.drag_handle(TrimHandle::Center, Vec3::new(0.25, -0.5, 1.0), 1e-3);
    assert_eq!(
        t,
        Trim {
            center: Vec3::new(1.25, 1.5, 4.0),
            ..t0
        }
    );
}

#[test]
fn u42_a_handle_that_does_not_match_the_mode_changes_nothing() {
    let b = boxed([-1.0, 0.0, 2.0], [1.0, 4.0, 3.0]);
    let s = sphere([1.0, 2.0, 3.0], 0.5);
    let off = Trim {
        mode: TrimMode::Off,
        ..b
    };
    let d = Vec3::new(0.5, 0.5, 0.5);
    assert_eq!(b.drag_handle(TrimHandle::Center, d, 1e-3), b);
    assert_eq!(
        b.drag_handle(
            TrimHandle::Radius {
                axis: 0,
                positive: true
            },
            d,
            1e-3
        ),
        b
    );
    assert_eq!(
        s.drag_handle(TrimHandle::Face { axis: 0, max: true }, d, 1e-3),
        s
    );
    for h in [
        TrimHandle::Face {
            axis: 1,
            max: false,
        },
        TrimHandle::Center,
        TrimHandle::Radius {
            axis: 2,
            positive: false,
        },
    ] {
        assert_eq!(off.drag_handle(h, d, 1e-3), off);
    }
}
