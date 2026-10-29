// SPDX-License-Identifier: Apache-2.0

//! The exchange model in memory and its reader rules (§1.2), its header-only information
//! (§1.5) and the box of E10.

use glam::{DVec3, Vec3};

use super::{CoreError, RenderConventions, TimeSpan};

/// An exchange model in memory (§1.2): physical values in raw time units; row i is
/// primitive i. Every reader produces one, validated (`Model::validate`).
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// \[N, 3\] μx, world units of the camera poses.
    pub position: Vec<[f32; 3]>,
    /// \[N\] τ₀ + μt·T_τ, raw τ.
    pub time_center: Vec<f64>,
    /// \[N\] > 0, units of τ (1 with a single timestamp, openftgs D-70).
    pub duration: Vec<f64>,
    /// \[N, 3\] world units per unit of τ.
    pub velocity: Vec<[f32; 3]>,
    /// \[N, 3\] > 0, world units.
    pub scale: Vec<[f32; 3]>,
    /// \[N, 4\] unit quaternion, file order (w, x, y, z), canonical sign (§1.2, §1.14).
    pub rotation: Vec<[f32; 4]>,
    /// \[N\] in (0, 1).
    pub opacity: Vec<f32>,
    /// N · (L+1)² · 3, coefficient-major, RGB innermost (E4's order).
    pub sh: Vec<f32>,
    /// L ∈ 0..=3.
    pub sh_degree: u8,
    /// τ₀, τ₁ and the unit of τ.
    pub time: TimeSpan,
    /// The `render_conventions` block (D-39).
    pub conventions: RenderConventions,
    /// The producing software, empty when the file names none.
    pub producer: String,
    /// The SHA-256 of the file's bytes (D-40).
    pub identity: [u8; 32],
    /// Which reader produced the model, with a checkpoint's hints.
    pub kind: ModelKind,
}

/// Which of §1.2's three files a model came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelKind {
    /// An exchange model `ftgs-if`.
    ExchangeFile,
    /// A checkpoint `ftgs-checkpoint`, with its placement and frame-step hints (§1.5, §1.6).
    Checkpoint {
        /// The checkpoint's `spatial_scale` S (openftgs D-27).
        spatial_scale: f64,
        /// The checkpoint's `frame_dt` Δt_f in normalized time (openftgs E14).
        frame_dt: f64,
    },
    /// A PLY snapshot (D-41).
    Ply,
}

/// What the file's header alone tells (§1.5 header-only information).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    /// N, the number of primitives.
    pub n: u64,
    /// L.
    pub sh_degree: u8,
    /// τ₀, τ₁ and the unit of τ.
    pub time: TimeSpan,
    /// The producing software, empty when the file names none.
    pub producer: String,
    /// The `render_conventions` block.
    pub conventions: RenderConventions,
    /// Which kind of file it is.
    pub kind: ModelKind,
    /// The file's size in bytes.
    pub file_bytes: u64,
}

/// An axis-aligned box (E10's B_m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// The lower corner.
    pub min: Vec3,
    /// The upper corner.
    pub max: Vec3,
}

impl Aabb {
    /// The box's center, (min + max)/2.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// The box's extent along each axis, max − min.
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// The longest of the three sides (§1.5's fit).
    pub fn longest_side(&self) -> f32 {
        self.size().max_element()
    }
}

/// E1's moved position μx(τ) = position + velocity (τ − time_center), computed in float64
/// and rounded to float32. Private to the core; the `motion` module's public E1 may reuse it.
pub(crate) fn moved_position_f64(
    position: [f32; 3],
    velocity: [f32; 3],
    time_center: f64,
    tau: f64,
) -> Vec3 {
    let dt = tau - time_center;
    let p = DVec3::from_array(position.map(f64::from));
    let v = DVec3::from_array(velocity.map(f64::from));
    (p + v * dt).as_vec3()
}

/// The first non-finite value of a slice, as an E-FILE-VALUE error naming the key and row.
fn check_finite<T: Copy + Into<f64>>(
    key: &str,
    values: impl IntoIterator<Item = T>,
) -> Result<(), CoreError> {
    for (i, x) in values.into_iter().enumerate() {
        let x: f64 = x.into();
        if !x.is_finite() {
            return Err(CoreError::Value(format!("{key}[{i}] = {x}")));
        }
    }
    Ok(())
}

#[expect(
    clippy::len_without_is_empty,
    reason = "§2.1 gives `len` without `is_empty`; a validated model is never empty (N ≥ 1)"
)]
impl Model {
    /// N, the number of primitives.
    pub fn len(&self) -> usize {
        self.position.len()
    }

    /// E10's B_m in model space: the box over μx(τ) at τ = τ₀ + k T_τ/`samples`,
    /// k = 0 … `samples` (τ₀ alone for a single timestamp or `samples` = 0), of every
    /// primitive with opacity > α_skip, expanded by 3 · the largest scale of those
    /// primitives. With no such primitive the box covers every primitive (Q-9).
    // Q-9
    pub fn bounds(&self, samples: usize) -> Aabb {
        let alpha_skip = self.conventions.alpha_skip;
        let visible: Vec<usize> = (0..self.len())
            .filter(|&i| self.opacity[i] > alpha_skip)
            .collect();
        let rows: Vec<usize> = if visible.is_empty() {
            (0..self.len()).collect()
        } else {
            visible
        };
        let steps = if self.is_single_timestamp() {
            0
        } else {
            samples
        };
        let (tau0, span) = (self.time.tau0, self.time.span());
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut largest_scale = 0.0f32;
        for &i in &rows {
            largest_scale = largest_scale.max(Vec3::from_array(self.scale[i]).max_element());
            for k in 0..=steps {
                let tau = if steps == 0 {
                    tau0
                } else {
                    tau0 + k as f64 * span / steps as f64
                };
                let p = moved_position_f64(
                    self.position[i],
                    self.velocity[i],
                    self.time_center[i],
                    tau,
                );
                min = min.min(p);
                max = max.max(p);
            }
        }
        let margin = Vec3::splat(3.0 * largest_scale);
        Aabb {
            min: min - margin,
            max: max + margin,
        }
    }

    /// §1.2's reader rules on the arrays in memory, reporting the first rule broken in
    /// §1.2's order: `sh_degree` in 0–3 (E-FILE-SH); τ₀ and τ₁ finite with τ₀ ≤ τ₁
    /// (E-FILE-VALUE); N ≥ 1 and the same N in every array (E-FILE-SHAPE), the `sh`
    /// length N·(L+1)²·3 (E-FILE-SH); every value finite; durations and scales > 0;
    /// rotation norms > 10⁻⁸; opacities in (0, 1) (E-FILE-VALUE); with a single timestamp
    /// every time center τ₀ and every velocity 0 (E-FILE-SINGLE, openftgs D-71); the
    /// conventions' numbers finite and the background in \[0, 1\] (E-FILE-VALUE, Q-8).
    /// The container rules (format, version, keys, dtypes) are the readers'.
    pub fn validate(&self) -> Result<(), CoreError> {
        let l = self.sh_degree;
        if l > 3 {
            return Err(CoreError::Sh(format!("sh_degree {l}")));
        }
        let (tau0, tau1) = (self.time.tau0, self.time.tau1);
        if !tau0.is_finite() || !tau1.is_finite() || tau1 < tau0 {
            return Err(CoreError::Value(format!("tau0 = {tau0}, tau1 = {tau1}")));
        }

        let n = self.len();
        if n == 0 {
            return Err(CoreError::Shape(
                "position has shape [0, 3], expected N ≥ 1".to_owned(),
            ));
        }
        let lengths = [
            ("time_center", self.time_center.len()),
            ("duration", self.duration.len()),
            ("velocity", self.velocity.len()),
            ("scale", self.scale.len()),
            ("rotation", self.rotation.len()),
            ("opacity", self.opacity.len()),
        ];
        for (key, len) in lengths {
            if len != n {
                return Err(CoreError::Shape(format!(
                    "{key} has {len} rows, expected N = {n}"
                )));
            }
        }
        let k = (usize::from(l) + 1).pow(2);
        if self.sh.len() != n * k * 3 {
            return Err(CoreError::Sh(format!(
                "sh holds {} values, expected [N, {k}, 3] = {} for sh_degree {l}",
                self.sh.len(),
                n * k * 3
            )));
        }

        check_finite("position", self.position.iter().flatten().copied())?;
        check_finite("time_center", self.time_center.iter().copied())?;
        check_finite("duration", self.duration.iter().copied())?;
        check_finite("velocity", self.velocity.iter().flatten().copied())?;
        check_finite("scale", self.scale.iter().flatten().copied())?;
        check_finite("rotation", self.rotation.iter().flatten().copied())?;
        check_finite("opacity", self.opacity.iter().copied())?;
        check_finite("sh", self.sh.iter().copied())?;

        if let Some(i) = self.duration.iter().position(|&d| d <= 0.0) {
            return Err(CoreError::Value(format!(
                "duration[{i}] = {}",
                self.duration[i]
            )));
        }
        if let Some(i) = self.scale.iter().position(|s| s.iter().any(|&x| x <= 0.0)) {
            return Err(CoreError::Value(format!(
                "scale[{i}] = {:?}",
                self.scale[i]
            )));
        }
        if let Some(i) = self
            .rotation
            .iter()
            .position(|q| q.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt() <= 1e-8)
        {
            return Err(CoreError::Value(format!(
                "rotation[{i}] = {:?}",
                self.rotation[i]
            )));
        }
        if let Some(i) = self.opacity.iter().position(|&o| o <= 0.0 || o >= 1.0) {
            return Err(CoreError::Value(format!(
                "opacity[{i}] = {}",
                self.opacity[i]
            )));
        }

        // openftgs D-71
        if self.is_single_timestamp() {
            if let Some(i) = self.time_center.iter().position(|&t| t != tau0) {
                return Err(CoreError::SingleTimestamp(format!(
                    "time_center[{i}] = {} ≠ tau0 = {tau0}",
                    self.time_center[i]
                )));
            }
            if let Some(i) = self
                .velocity
                .iter()
                .position(|v| v.iter().any(|&x| x != 0.0))
            {
                return Err(CoreError::SingleTimestamp(format!(
                    "velocity[{i}] = {:?}",
                    self.velocity[i]
                )));
            }
        }

        self.validate_conventions()
    }

    /// The conventions block's rule (§1.2, openftgs D-84) and its numbers finite (Q-8).
    // Q-8
    fn validate_conventions(&self) -> Result<(), CoreError> {
        let c = &self.conventions;
        let numbers = [
            ("alpha_max", c.alpha_max),
            ("eps2d", c.eps2d),
            ("near", c.near),
            ("far", c.far),
            ("alpha_skip", c.alpha_skip),
            ("transmittance_stop", c.transmittance_stop.unwrap_or(0.0)),
            ("pixel_center_offset", c.pixel_center_offset),
            ("jacobian_clamp_margin", c.jacobian_clamp_margin),
            ("screen_extent_max", c.screen_extent_max),
        ];
        for (key, x) in numbers {
            if !x.is_finite() {
                return Err(CoreError::Value(format!("render_conventions.{key} = {x}")));
            }
        }
        if !c.background.iter().all(|x| (0.0..=1.0).contains(x)) {
            return Err(CoreError::Value(format!(
                "render_conventions.background = {:?}",
                c.background
            )));
        }
        Ok(())
    }

    /// §1.2 load option 1: with a cap L′ < L, drops the bands above L′ (coefficients
    /// (L′+1)² onward of every row) and sets L to L′; a cap at or above L is the identity.
    pub fn cap_sh(&mut self, degree: u8) {
        if degree >= self.sh_degree {
            return;
        }
        let row = (usize::from(self.sh_degree) + 1).pow(2) * 3;
        let kept = (usize::from(degree) + 1).pow(2) * 3;
        self.sh = self
            .sh
            .chunks_exact(row)
            .flat_map(|r| r[..kept].iter().copied())
            .collect();
        self.sh_degree = degree;
    }

    /// Whether τ₀ = τ₁ (D-9).
    pub fn is_single_timestamp(&self) -> bool {
        self.time.is_single()
    }
}
