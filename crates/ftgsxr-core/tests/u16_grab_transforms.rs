// SPDX-License-Identifier: Apache-2.0

//! U-16, E11 (`Grab`, `TwoHandGrab`, `cpu` lane, D-43). One hand: the model's pose
//! relative to the hand is invariant under 1,000 random hand motions. Two hands: scale =
//! distance ratio, clamped at 0.01 and 100; the midpoint is followed; a 30° horizontal
//! rotation of the hands yaws the model 30° and nothing else in upright mode; free
//! rotation takes d₀ to d. Tolerance 10⁻⁵.
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
    let grab = Grab::begin(hand0, model0);
    for _ in 0..1000 {
        let hand = rng.hand();
        let model = grab.update(hand);
        for (x, want) in PROBES.iter().zip(&held) {
            assert_vec3(in_hand(&hand, apply(&model, *x)), *want, "hand-frame point");
        }
        assert!(
            (model.scale - 1.7).abs() <= TOL,
            "a rigid hand keeps the scale"
        );
    }
    // At the start pose the model has not moved.
    let still = grab.update(hand0);
    for x in PROBES {
        assert_vec3(apply(&still, x), apply(&model0, x), "unmoved hand");
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
