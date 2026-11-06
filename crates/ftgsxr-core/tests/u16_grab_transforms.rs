// SPDX-License-Identifier: Apache-2.0

//! U-16, E11 (`Grab`, `TwoHandGrab`, `cpu` lane, D-43). One hand: with free rotation
//! the model's pose relative to the hand is invariant under 1,000 random hand motions;
//! with keep upright, under the same motions, the grab point follows the hand, the
//! model's up axis is unchanged and its yaw changes by the twist of the hand's turn
//! about ŷ (spec revision 37). Two hands: scale =
//! distance ratio, clamped at 0.01 and 100; the midpoint is followed; a 30° horizontal
//! rotation of the hands yaws the model 30° and nothing else in upright mode; free
//! rotation takes d₀ to d; two grab points on a posed box keep their box-normalized
//! coordinates under free rotation, and their midpoint keeps its own under keep upright
//! (spec revision 34). Tolerance 10⁻⁵.
//!
//! Expected values are checked on points: a pose is compared through where it sends a
//! few model-space points, worked from E5's x ↦ s R(q) x + t, so q and −q compare equal.

use ftgsxr_core::{Grab, Pose, Quat, ScaleLimits, TwoHandGrab, Vec3};

/// The row's tolerance.
const TOL: f32 = 1e-5;

/// A small deterministic generator (xorshift64*), local to the test.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [lo, hi).
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
    fn vec3(&mut self, half: f32) -> Vec3 {
        Vec3::new(
            self.uniform(-half, half),
            self.uniform(-half, half),
            self.uniform(-half, half),
        )
    }
    /// A random unit quaternion: a point of the 4-ball by rejection, normalized.
    fn rotation(&mut self) -> Quat {
        loop {
            let v = [0; 4].map(|_| self.uniform(-1.0, 1.0));
            let n2: f32 = v.iter().map(|c| c * c).sum();
            if (0.01..=1.0).contains(&n2) {
                return Quat::from_array(v).normalize();
            }
        }
    }
    /// A hand pose: a rigid transform within 2 m of the stage origin.
    fn hand(&mut self) -> Pose {
        Pose {
            translation: self.vec3(2.0),
            rotation: self.rotation(),
            scale: 1.0,
        }
    }
}

/// Model-space probe points: the origin and the three unit axes.
const PROBES: [Vec3; 4] = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];

/// E5: x ↦ s R(q) x + t, worked here rather than through `Pose`.
fn apply(p: &Pose, x: Vec3) -> Vec3 {
    p.rotation * x * p.scale + p.translation
}

/// A stage point in a rigid hand's frame: R_hᵀ (y − t_h).
fn in_hand(hand: &Pose, y: Vec3) -> Vec3 {
    hand.rotation.inverse() * (y - hand.translation)
}

fn assert_vec3(got: Vec3, want: Vec3, what: &str) {
    assert!(
        (got - want).abs().max_element() <= TOL,
        "{what}: got {got}, want {want}"
    );
}

#[test]
fn u16_one_hand_keeps_the_model_fixed_in_the_hand() {
    let mut rng = Rng(0x5EED_0016);
    let hand0 = rng.hand();
    let model0 = Pose {
        translation: rng.vec3(2.0),
        rotation: rng.rotation(),
        scale: 1.7,
    };
    let held: Vec<Vec3> = PROBES
        .iter()
        .map(|&x| in_hand(&hand0, apply(&model0, x)))
        .collect();
    let grab = Grab::begin(hand0, model0, rng.vec3(2.0));
    for _ in 0..1000 {
        let hand = rng.hand();
        let model = grab.update(hand, true);
        for (x, want) in PROBES.iter().zip(&held) {
            assert_vec3(in_hand(&hand, apply(&model, *x)), *want, "hand-frame point");
        }
        assert!(
            (model.scale - 1.7).abs() <= TOL,
            "a rigid hand keeps the scale"
        );
    }
    // At the start pose the model has not moved.
    let still = grab.update(hand0, true);
    for x in PROBES {
        assert_vec3(apply(&still, x), apply(&model0, x), "unmoved hand");
    }
}

/// E11's twist about ŷ of a unit rotation q: φ = 2 atan2(q_y, q_w), 0 where both vanish.
fn twist_y(q: Quat) -> f32 {
    if q.y == 0.0 && q.w == 0.0 {
        0.0
    } else {
        2.0 * q.y.atan2(q.w)
    }
}

/// R_ŷ(φ) applied to a vector, worked from its matrix rather than through `Quat`.
fn yaw(phi: f32, v: Vec3) -> Vec3 {
    let (s, c) = phi.sin_cos();
    Vec3::new(c * v.x + s * v.z, v.y, -s * v.x + c * v.z)
}

#[test]
fn u16_one_hand_upright_follows_the_grab_point_and_turns_by_yaw_only() {
    let mut rng = Rng(0x5EED_2016);
    let hand0 = rng.hand();
    // A tilted model: the grab must keep its tilt.
    let model0 = Pose {
        translation: rng.vec3(2.0),
        rotation: rng.rotation(),
        scale: 1.7,
    };
    let g0 = rng.vec3(2.0);
    let g_in_hand = in_hand(&hand0, g0);
    let g_on_model = unapply(&model0, g0);
    let up_in_model = model0.rotation.inverse() * Vec3::Y;
    let grab = Grab::begin(hand0, model0, g0);
    for _ in 0..1000 {
        let hand = rng.hand();
        let model = grab.update(hand, false);
        // g = H H₀⁻¹ g₀, and it stays that point of the model.
        let g = hand.rotation * g_in_hand + hand.translation;
        assert_vec3(apply(&model, g_on_model), g, "grab point");
        // The stage's up axis, seen from the model, is where it was.
        assert_vec3(
            model.rotation.inverse() * Vec3::Y,
            up_in_model,
            "up axis in the model",
        );
        // Each model axis turns by R_ŷ(φ), φ the twist of R_H R_H₀⁻¹ about ŷ.
        let phi = twist_y(hand.rotation * hand0.rotation.inverse());
        for x in [Vec3::X, Vec3::Y, Vec3::Z] {
            assert_vec3(
                model.rotation * x,
                yaw(phi, model0.rotation * x),
                "model axis",
            );
        }
        assert!((model.scale - 1.7).abs() <= TOL, "the scale is kept");
    }
    let still = grab.update(hand0, false);
    for x in PROBES {
        assert_vec3(apply(&still, x), apply(&model0, x), "unmoved hand");
    }
}

#[test]
fn u16_one_hand_upright_a_half_turn_about_a_horizontal_axis_has_no_twist() {
    // q = (1, 0, 0, 0): q_y = q_w = 0, so φ = 0 and the model only follows g.
    let hand0 = Pose {
        translation: Vec3::new(0.0, 1.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: 1.0,
    };
    let model0 = tilted_model();
    let g0 = Vec3::new(0.3, 1.2, -1.5);
    let grab = Grab::begin(hand0, model0, g0);
    let hand = Pose {
        rotation: Quat::from_rotation_x(std::f32::consts::PI),
        ..hand0
    };
    let model = grab.update(hand, false);
    let g = hand.rotation * in_hand(&hand0, g0) + hand.translation;
    for x in PROBES {
        assert_vec3(
            apply(&model, x),
            apply(&model0, x) - g0 + g,
            "translated only",
        );
    }
}

/// E11's limits, written out (D-43).
const LIMITS: ScaleLimits = ScaleLimits {
    min: 0.01,
    max: 100.0,
};

/// A hand at a stage position, with an orientation E11's two-hand rule must ignore.
fn hand_at(p: Vec3, rng: &mut Rng) -> Pose {
    Pose {
        translation: p,
        rotation: rng.rotation(),
        scale: 1.0,
    }
}

#[test]
fn u16_two_hands_scale_by_the_distance_ratio_clamped() {
    let mut rng = Rng(0x5EED_1016);
    let (a0, b0) = (Vec3::new(-0.5, 1.2, -0.4), Vec3::new(0.5, 1.2, -0.4)); // 1 m apart
    let unit = Pose {
        translation: Vec3::new(0.2, 0.9, -1.0),
        rotation: Quat::from_rotation_y(0.3),
        scale: 1.0,
    };
    let grab = TwoHandGrab::begin(hand_at(a0, &mut rng), hand_at(b0, &mut rng), unit);
    // The hands move apart symmetrically about their midpoint (0, 1.2, −0.4).
    let apart = |d: f32| {
        (
            Vec3::new(-d / 2.0, 1.2, -0.4),
            Vec3::new(d / 2.0, 1.2, -0.4),
        )
    };
    for (distance, want) in [
        (1.0, 1.0),
        (2.0, 2.0),
        (0.25, 0.25),
        (0.02, 0.02),
        (0.005, 0.01), // ratio 0.005, clamped at s_min
        (50.0, 50.0),
        (150.0, 100.0), // ratio 150, clamped at s_max
    ] {
        let (a, b) = apart(distance);
        let p = grab.update(hand_at(a, &mut rng), hand_at(b, &mut rng), &LIMITS, false);
        assert!(
            (p.scale - want).abs() <= TOL * want.max(1.0),
            "distance {distance}: scale {} want {want}",
            p.scale
        );
    }
    // The ratio multiplies the start scale s₀, and the clamp applies to the product.
    let big = Pose {
        scale: 40.0,
        ..unit
    };
    let grab = TwoHandGrab::begin(hand_at(a0, &mut rng), hand_at(b0, &mut rng), big);
    let (a, b) = apart(2.0);
    let p = grab.update(hand_at(a, &mut rng), hand_at(b, &mut rng), &LIMITS, false);
    assert!(
        (p.scale - 80.0).abs() <= TOL * 80.0,
        "s₀·2 = 80, got {}",
        p.scale
    );
    let (a, b) = apart(3.0);
    let p = grab.update(hand_at(a, &mut rng), hand_at(b, &mut rng), &LIMITS, false);
    assert!(
        (p.scale - 100.0).abs() <= TOL * 100.0,
        "s₀·3 = 120 → 100, got {}",
        p.scale
    );
}

/// A model pose away from the hands, rotated off every axis, with a non-unit scale.
fn tilted_model() -> Pose {
    Pose {
        translation: Vec3::new(0.7, 0.4, -1.6),
        rotation: Quat::from_rotation_y(0.4)
            * Quat::from_rotation_x(-0.9)
            * Quat::from_rotation_z(0.25),
        scale: 1.5,
    }
}

/// The model-space point that P₀ puts at the stage point y: R₀ᵀ (y − t₀)/s₀.
fn model_point_at(p0: &Pose, y: Vec3) -> Vec3 {
    p0.rotation.inverse() * (y - p0.translation) / p0.scale
}

#[test]
fn u16_two_hands_follow_the_midpoint() {
    let mut rng = Rng(0x5EED_2016);
    let model0 = tilted_model();
    let (a0, b0) = (Vec3::new(-0.3, 1.1, -0.5), Vec3::new(0.4, 1.3, -0.6));
    let m0 = (a0 + b0) / 2.0;
    let pinned = model_point_at(&model0, m0);
    let grab = TwoHandGrab::begin(hand_at(a0, &mut rng), hand_at(b0, &mut rng), model0);
    for _ in 0..100 {
        // The hands move anywhere: a new midpoint, distance and direction.
        let (a, b) = (rng.vec3(2.0), rng.vec3(2.0));
        let m = (a + b) / 2.0;
        for free in [false, true] {
            let p = grab.update(hand_at(a, &mut rng), hand_at(b, &mut rng), &LIMITS, free);
            // The model point that sat at the hands' midpoint sits at the new midpoint.
            assert_vec3(apply(&p, pinned), m, "midpoint");
        }
    }
    // A pure translation of both hands translates the model by the same vector.
    let v = Vec3::new(0.3, -0.2, 1.1);
    let p = grab.update(
        hand_at(a0 + v, &mut rng),
        hand_at(b0 + v, &mut rng),
        &LIMITS,
        false,
    );
    for x in PROBES {
        assert_vec3(apply(&p, x), apply(&model0, x) + v, "translated hands");
    }
}

#[test]
fn u16_upright_30_degree_turn_yaws_30_degrees_and_nothing_else() {
    let mut rng = Rng(0x5EED_3016);
    let model0 = tilted_model();
    // The hand axis is not horizontal: E11 projects it.
    let (a0, b0) = (Vec3::new(-0.35, 1.0, -0.45), Vec3::new(0.35, 1.25, -0.55));
    let m0 = (a0 + b0) / 2.0;
    let grab = TwoHandGrab::begin(hand_at(a0, &mut rng), hand_at(b0, &mut rng), model0);
    for degrees in [30.0f32, -30.0] {
        // Both hands turn by 30° about the vertical line through their midpoint, so the
        // midpoint, the distance and the axis's vertical part are unchanged.
        let (c, s) = (degrees.to_radians().cos(), degrees.to_radians().sin());
        // A turn by φ about +y: (x, z) ↦ (x cos φ + z sin φ, −x sin φ + z cos φ).
        let turn = |p: Vec3| {
            let r = p - m0;
            m0 + Vec3::new(r.x * c + r.z * s, r.y, -r.x * s + r.z * c)
        };
        let p = grab.update(
            hand_at(turn(a0), &mut rng),
            hand_at(turn(b0), &mut rng),
            &LIMITS,
            false,
        );
        assert!((p.scale - model0.scale).abs() <= TOL, "scale unchanged");
        // Every model point is turned the same way about the same line.
        for x in PROBES {
            assert_vec3(apply(&p, x), turn(apply(&model0, x)), "yawed point");
        }
    }
}

#[test]
fn u16_free_rotation_takes_d0_to_d() {
    let mut rng = Rng(0x5EED_4016);
    let model0 = tilted_model();
    let (a0, b0) = (Vec3::new(-0.3, 1.1, -0.5), Vec3::new(0.4, 1.3, -0.6));
    let d0 = (b0 - a0).normalize();
    let grab = TwoHandGrab::begin(hand_at(a0, &mut rng), hand_at(b0, &mut rng), model0);
    for _ in 0..100 {
        let (a, b) = (rng.vec3(2.0), rng.vec3(2.0));
        let d = (b - a).normalize();
        let p = grab.update(hand_at(a, &mut rng), hand_at(b, &mut rng), &LIMITS, true);
        // The rotation applied to the model, R(q′) R(q₀)ᵀ, read off the model's axes,
        // sends d̂₀ to d̂.
        let applied = |v: Vec3| p.rotation * (model0.rotation.inverse() * v);
        assert_vec3(applied(d0), d, "d̂₀ ↦ d̂");
        // Shortest: the axis d̂₀ × d̂ is left in place.
        let axis = d0.cross(d);
        if axis.length() > 1e-3 {
            let axis = axis.normalize();
            assert_vec3(applied(axis), axis, "rotation axis fixed");
        }
    }
}

/// A box B_m for the grab points (spec revision 34): not a cube, off the origin.
const BOX_MIN: Vec3 = Vec3::new(-0.4, -0.1, -0.3);
const BOX_MAX: Vec3 = Vec3::new(0.6, 0.5, 0.2);

/// A model-space point's box-normalized coordinates: 0 at B_m's min corner, 1 at its max.
fn normalized(x: Vec3) -> Vec3 {
    (x - BOX_MIN) / (BOX_MAX - BOX_MIN)
}

/// The model-space point that `p` puts at the stage point y: R(q)ᵀ (y − t)/s, worked
/// here rather than through `Pose::inverse`.
fn unapply(p: &Pose, y: Vec3) -> Vec3 {
    p.rotation.inverse() * (y - p.translation) / p.scale
}

/// The stage point of the box-normalized point `n` under `p`.
fn on_box(p: &Pose, n: Vec3) -> Vec3 {
    apply(p, BOX_MIN + n * (BOX_MAX - BOX_MIN))
}

/// Two grab points on the posed box, given by their box-normalized coordinates, then
/// moved at random: the grab points' box-normalized coordinates under the returned
/// poses, for each motion.
fn grab_point_runs(seed: u64, free_rotation: bool, na: Vec3, nb: Vec3) -> Vec<(Vec3, Vec3, Vec3)> {
    let mut rng = Rng(seed);
    let model0 = tilted_model();
    let (a0, b0) = (on_box(&model0, na), on_box(&model0, nb));
    let grab = TwoHandGrab::begin(hand_at(a0, &mut rng), hand_at(b0, &mut rng), model0);
    (0..100)
        .map(|_| {
            let (a, b) = (rng.vec3(2.0), rng.vec3(2.0));
            let p = grab.update(
                hand_at(a, &mut rng),
                hand_at(b, &mut rng),
                &LIMITS,
                free_rotation,
            );
            (
                normalized(unapply(&p, a)),
                normalized(unapply(&p, b)),
                normalized(unapply(&p, (a + b) * 0.5)),
            )
        })
        .collect()
}

/// Spec revision 34's U-16 clause: two grab points on a posed box keep their
/// box-normalized coordinates under free rotation, and their midpoint keeps its own
/// under keep upright. The random motions stay within the scale limits: the grab
/// points' distance changes by less than 100× either way.
#[test]
fn u16_grab_points_keep_their_box_coordinates() {
    let (na, nb) = (Vec3::new(0.1, 0.8, 1.0), Vec3::new(0.9, 0.3, 0.0));
    let model0 = tilted_model();
    let nm = normalized(unapply(
        &model0,
        (on_box(&model0, na) + on_box(&model0, nb)) * 0.5,
    ));
    for (a, b, m) in grab_point_runs(0x5EED_6016, true, na, nb) {
        assert_vec3(a, na, "free rotation: the first grab point");
        assert_vec3(b, nb, "free rotation: the second grab point");
        assert_vec3(m, nm, "free rotation: the midpoint");
    }
    for (_, _, m) in grab_point_runs(0x5EED_7016, false, na, nb) {
        assert_vec3(m, nm, "keep upright: the midpoint");
    }
}
