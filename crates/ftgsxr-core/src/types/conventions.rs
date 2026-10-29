// SPDX-License-Identifier: Apache-2.0

//! The `render_conventions` block of an exchange model (§1.2, openftgs D-73, D-39).

use serde_json::{Map, Value};

use super::CoreError;

/// openftgs's screen-extent rule (E6).
const OPACITY_AWARE: &str = "opacity_aware";
/// openftgs's rasterize mode (E6).
const CLASSIC: &str = "classic";

/// The `render_conventions` block of §1.2 (openftgs D-73): the constants of E3, E6 and E7
/// the player honors (D-39). `Default` is openftgs's values.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderConventions {
    /// α_max of E7 (0.999).
    pub alpha_max: f32,
    /// ε of E6, added to Σ′'s diagonal (0.3).
    pub eps2d: f32,
    /// The near cull of E6 in camera z (0.01).
    pub near: f32,
    /// The far cull of E6 in camera z (10¹⁰).
    pub far: f32,
    /// α_skip of E3, E6 and E7 (1/255).
    pub alpha_skip: f32,
    /// The transmittance stop (10⁻⁴); `None` when the block gives `null`. The player does
    /// not apply it (D-7).
    pub transmittance_stop: Option<f32>,
    /// The pixel-center offset of the continuous convention (0.5).
    pub pixel_center_offset: f32,
    /// The background the model was trained against, RGB in \[0, 1\] (black; openftgs D-84).
    pub background: [f32; 3],
    /// E6's Jacobian clamp margin m (0.15).
    pub jacobian_clamp_margin: f32,
    /// E6's screen-extent rule (`opacity_aware`).
    pub screen_extent_rule: String,
    /// E6's screen-extent cap in standard deviations (3.33).
    pub screen_extent_max: f32,
    /// E7's rasterize mode (`classic`).
    pub rasterize_mode: String,
}

impl Default for RenderConventions {
    /// openftgs's values (§1.2, openftgs D-73), black background.
    fn default() -> Self {
        RenderConventions {
            alpha_max: 0.999,
            eps2d: 0.3,
            near: 0.01,
            far: 1e10,
            alpha_skip: 1.0 / 255.0,
            transmittance_stop: Some(1e-4),
            pixel_center_offset: 0.5,
            background: [0.0; 3],
            jacobian_clamp_margin: 0.15,
            screen_extent_rule: OPACITY_AWARE.to_owned(),
            screen_extent_max: 3.33,
            rasterize_mode: CLASSIC.to_owned(),
        }
    }
}

/// A numeric entry: absent keeps the default; present, it must be a number finite in
/// float32 (E-FILE-VALUE, Q-T4.1).
fn number(map: &Map<String, Value>, key: &str, default: f32) -> Result<f32, CoreError> {
    match map.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_f64()
            .map(|x| x as f32)
            .filter(|x| x.is_finite())
            .ok_or_else(|| CoreError::Value(format!("render_conventions.{key} = {v}"))),
    }
}

/// A string entry: absent keeps the default; present, it must be a string (Q-T4.1).
fn string(map: &Map<String, Value>, key: &str, default: &str) -> Result<String, CoreError> {
    match map.get(key) {
        None => Ok(default.to_owned()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(v) => Err(CoreError::Value(format!("render_conventions.{key} = {v}"))),
    }
}

/// The background: absent is black; present, three numbers in \[0, 1\] (openftgs D-84).
fn background(map: &Map<String, Value>) -> Result<[f32; 3], CoreError> {
    let Some(v) = map.get("background") else {
        return Ok([0.0; 3]);
    };
    let bad = || CoreError::Value(format!("render_conventions.background = {v}"));
    let items = v.as_array().filter(|a| a.len() == 3).ok_or_else(bad)?;
    let mut rgb = [0.0f32; 3];
    for (c, item) in rgb.iter_mut().zip(items) {
        let x = item
            .as_f64()
            .filter(|x| (0.0..=1.0).contains(x))
            .ok_or_else(bad)?;
        *c = x as f32;
    }
    Ok(rgb)
}

impl RenderConventions {
    /// Parses the block (§1.2): it must be a JSON object; unknown keys are ignored; the
    /// background is checked in \[0, 1\]³; missing entries take openftgs's values;
    /// `transmittance_stop` may be `null`. A block that is not a JSON object, or an entry
    /// of the wrong type, is `CoreError::Value` (E-FILE-VALUE, Q-T4.1).
    // Q-T4.1
    pub fn from_json(json: &str) -> Result<Self, CoreError> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| CoreError::Value(format!("render_conventions (not JSON: {e})")))?;
        let Value::Object(map) = value else {
            return Err(CoreError::Value(
                "render_conventions (not a JSON object)".to_owned(),
            ));
        };
        let d = RenderConventions::default();
        let transmittance_stop = match map.get("transmittance_stop") {
            Some(Value::Null) => None,
            _ => Some(number(&map, "transmittance_stop", 1e-4)?),
        };
        Ok(RenderConventions {
            alpha_max: number(&map, "alpha_max", d.alpha_max)?,
            eps2d: number(&map, "eps2d", d.eps2d)?,
            near: number(&map, "near", d.near)?,
            far: number(&map, "far", d.far)?,
            alpha_skip: number(&map, "alpha_skip", d.alpha_skip)?,
            transmittance_stop,
            pixel_center_offset: number(&map, "pixel_center_offset", d.pixel_center_offset)?,
            background: background(&map)?,
            jacobian_clamp_margin: number(&map, "jacobian_clamp_margin", d.jacobian_clamp_margin)?,
            screen_extent_rule: string(&map, "screen_extent_rule", OPACITY_AWARE)?,
            screen_extent_max: number(&map, "screen_extent_max", d.screen_extent_max)?,
            rasterize_mode: string(&map, "rasterize_mode", CLASSIC)?,
        })
    }

    /// The block as compact JSON (no whitespace) with openftgs's keys in openftgs's order
    /// (§1.2, openftgs D-73); numbers are the shortest decimal that reads back to the
    /// same float32, `transmittance_stop` is `null` when `None`.
    pub fn to_json(&self) -> String {
        fn num(x: f32) -> String {
            serde_json::to_string(&x).expect("invariant: a float32 serializes to JSON")
        }
        fn text(s: &str) -> String {
            serde_json::to_string(s).expect("invariant: a string serializes to JSON")
        }
        let stop = self
            .transmittance_stop
            .map_or_else(|| "null".to_owned(), num);
        let [r, g, b] = self.background.map(num);
        format!(
            concat!(
                "{{\"alpha_max\":{},\"eps2d\":{},\"near\":{},\"far\":{},\"alpha_skip\":{},",
                "\"transmittance_stop\":{},\"pixel_center_offset\":{},\"background\":[{},{},{}],",
                "\"jacobian_clamp_margin\":{},\"screen_extent_rule\":{},\"screen_extent_max\":{},",
                "\"rasterize_mode\":{}}}"
            ),
            num(self.alpha_max),
            num(self.eps2d),
            num(self.near),
            num(self.far),
            num(self.alpha_skip),
            stop,
            num(self.pixel_center_offset),
            r,
            g,
            b,
            num(self.jacobian_clamp_margin),
            text(&self.screen_extent_rule),
            num(self.screen_extent_max),
            text(&self.rasterize_mode),
        )
    }

    /// One warning per entry the player renders as openftgs's instead: a
    /// `screen_extent_rule` other than `opacity_aware` or a `rasterize_mode` other than
    /// `classic` (E6, D-39).
    // D-39
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.screen_extent_rule != OPACITY_AWARE {
            out.push(format!(
                "screen_extent_rule {:?} is not supported; rendering as {OPACITY_AWARE:?}",
                self.screen_extent_rule
            ));
        }
        if self.rasterize_mode != CLASSIC {
            out.push(format!(
                "rasterize_mode {:?} is not supported; rendering as {CLASSIC:?}",
                self.rasterize_mode
            ));
        }
        out
    }
}
