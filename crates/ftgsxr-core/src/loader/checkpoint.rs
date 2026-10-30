// SPDX-License-Identifier: Apache-2.0

//! The checkpoint `ftgs-checkpoint` v1 (§1.2, D-8, openftgs §2.2, openftgs
//! `io.save_checkpoint`): `manifest.json` and `params.safetensors`, activated into the
//! exchange form (openftgs §1.1 activations, openftgs E14, openftgs D-70).

use std::path::Path;

use safetensors::tensor::Dtype;
use serde_json::{Map, Value};

use super::container::{Container, check_tensor, f32_le, f32s_le, values};
use super::exchange::finish;
use super::{PARALLEL_ROWS, check_version, hex, map_rows, opacity_from_logit, read_file, sha256};
use crate::types::{CoreError, Model, ModelKind, RenderConventions, TimeSpan};

/// The checkpoint's manifest file.
pub(crate) const MANIFEST: &str = "manifest.json";
/// The checkpoint's parameters file.
const PARAMS: &str = "params.safetensors";
/// The `format` of a checkpoint.
const FORMAT: &str = "ftgs-checkpoint";

/// What the manifest tells the reader (§1.2), checked.
struct Manifest {
    time: TimeSpan,
    spatial_scale: f64,
    frame_dt: f64,
    sh_degree: u8,
    params_sha256: String,
    background: Result<RenderConventions, CoreError>,
    producer: String,
}

/// The entry `key`, or E-FILE-KEYS.
fn required<'a>(map: &'a Map<String, Value>, key: &str) -> Result<&'a Value, CoreError> {
    map.get(key).ok_or_else(|| CoreError::Keys(key.to_owned()))
}

/// A JSON number entry (E-FILE-SHAPE when another JSON type, E-FILE-VALUE when not finite
/// in float64).
fn number(map: &Map<String, Value>, key: &str) -> Result<f64, CoreError> {
    match required(map, key)? {
        Value::Number(n) => n
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| CoreError::Value(format!("{key} = {n}"))),
        other => Err(CoreError::Shape(format!(
            "{key} is {other}, expected a number"
        ))),
    }
}

/// Reads and checks `manifest.json` (§1.2; Q-34): format, version, then `tau0`, `tau1`,
/// `spatial_scale`, `frame_dt`, `sh_degree` and `files["params.safetensors"]`.
// Q-34
fn read_manifest(bytes: &[u8]) -> Result<Manifest, CoreError> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|e| CoreError::Format(format!("{MANIFEST} is not JSON ({e})")))?;
    let Value::Object(map) = value else {
        return Err(CoreError::Format(format!(
            "{MANIFEST} holds no JSON object"
        )));
    };
    match required(&map, "format")? {
        Value::String(s) if s == FORMAT => {}
        other => {
            return Err(CoreError::Format(format!(
                "format {other}, expected {FORMAT:?}"
            )));
        }
    }
    match required(&map, "version")? {
        Value::Number(n) if n.is_u64() || n.is_i64() => check_version(&n.to_string())?,
        Value::String(s) => check_version(s)?,
        other => return Err(CoreError::Version(other.to_string())),
    }
    for key in [
        "tau0",
        "tau1",
        "spatial_scale",
        "frame_dt",
        "sh_degree",
        "files",
    ] {
        required(&map, key)?;
    }
    let (tau0, tau1) = (number(&map, "tau0")?, number(&map, "tau1")?);
    let spatial_scale = number(&map, "spatial_scale")?;
    let frame_dt = number(&map, "frame_dt")?;
    let sh_degree = match required(&map, "sh_degree")? {
        Value::Number(n) if n.is_u64() || n.is_i64() => n
            .as_u64()
            .and_then(|l| u8::try_from(l).ok())
            .filter(|l| *l <= 3)
            .ok_or_else(|| CoreError::Sh(format!("sh_degree {n}")))?,
        other => {
            return Err(CoreError::Shape(format!(
                "sh_degree is {other}, expected an integer"
            )));
        }
    };
    let time_unit = match map.get("time_unit") {
        None | Some(Value::Null) => "s".to_owned(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(CoreError::Shape(format!(
                "time_unit is {other}, expected a string"
            )));
        }
    };
    let files = match required(&map, "files")? {
        Value::Object(files) => files,
        other => {
            return Err(CoreError::Shape(format!(
                "files is {other}, expected an object"
            )));
        }
    };
    let params_sha256 = match files.get(PARAMS) {
        Some(Value::String(s)) => s.to_ascii_lowercase(),
        Some(other) => {
            return Err(CoreError::Shape(format!(
                "files.{PARAMS} is {other}, expected a string"
            )));
        }
        None => return Err(CoreError::Keys(format!("files.{PARAMS}"))),
    };
    // openftgs D-84: config.background is the background the colors were trained against;
    // checked as a `render_conventions` background is (Q-8).
    let background = match map.get("config").and_then(|c| c.get("background")) {
        None | Some(Value::Null) => Ok(RenderConventions::default()),
        Some(bg) => RenderConventions::from_json(&format!("{{\"background\":{bg}}}")),
    };
    // Q-34 (owner, #55): without a version, `openftgs`, as openftgs's own producer string
    // falls back when `save_checkpoint` writes `versions.openftgs` as null.
    let producer = match map.get("versions").and_then(|v| v.get("openftgs")) {
        Some(Value::String(v)) => format!("openftgs {v}"),
        _ => "openftgs".to_owned(),
    };
    Ok(Manifest {
        time: TimeSpan {
            tau0,
            tau1,
            time_unit,
        },
        spatial_scale,
        frame_dt,
        sh_degree,
        params_sha256,
        background,
        producer,
    })
}

/// Reads a checkpoint directory (§1.2, D-8): `manifest.json` and `params.safetensors`, the
/// other files ignored. The manifest is checked first (Q-34); then the SHA-256 of
/// `params.safetensors`, computed beside its decoding, must equal the manifest's `files`
/// entry (E-FILE-CHECKSUM); then the nine float32 tensors of openftgs §1.1 are checked
/// (E-FILE-KEYS, E-FILE-SHAPE, the `shN` coefficient count E-FILE-SH), activated into the
/// exchange form with T′ = T_τ, or 1 when T_τ = 0, and validated by `Model::validate`.
/// `identity` is the SHA-256 of `params.safetensors` (D-40); `kind` carries the manifest's
/// `spatial_scale` and `frame_dt`; the background is the manifest's `config.background`.
pub fn read_checkpoint(dir: &Path) -> Result<Model, CoreError> {
    let manifest = read_manifest(&read_file(&dir.join(MANIFEST))?)?;
    let params = read_file(&dir.join(PARAMS))?;
    let (identity, model) = rayon::join(|| sha256(&params), || activate(&params, &manifest));
    if hex(&identity) != manifest.params_sha256 {
        return Err(CoreError::Checksum(format!(
            "{PARAMS} has SHA-256 {}, the manifest gives {}",
            hex(&identity),
            manifest.params_sha256
        )));
    }
    let mut model = model?;
    model.identity = identity;
    finish(&mut model, manifest.background)?;
    Ok(model)
}

/// Checks `params.safetensors` and activates it (§1.2; openftgs `FreeTimeGaussians.physical`
/// gives each field's precision).
fn activate(bytes: &[u8], m: &Manifest) -> Result<Model, CoreError> {
    let file = Container::parse(bytes)?;
    let header = file.header();
    let names = [
        "means",
        "times",
        "log_durations",
        "velocities",
        "log_scales",
        "quats",
        "opacity_logits",
        "sh0",
        "shN",
    ];
    for key in names {
        if header.info(key).is_none() {
            return Err(CoreError::Keys(key.to_owned()));
        }
    }
    let info = |key: &str| header.info(key).expect("invariant: presence checked above");
    let means = info("means");
    let n = match means.shape[..] {
        [n, 3] if n >= 1 => n,
        _ => {
            return Err(CoreError::Shape(format!(
                "means has shape {:?}, expected [N, 3] with N ≥ 1",
                means.shape
            )));
        }
    };
    let k = (usize::from(m.sh_degree) + 1).pow(2);
    check_tensor("means", means, Dtype::F32, &[n, 3])?;
    check_tensor("times", info("times"), Dtype::F32, &[n])?;
    check_tensor("log_durations", info("log_durations"), Dtype::F32, &[n])?;
    check_tensor("velocities", info("velocities"), Dtype::F32, &[n, 3])?;
    check_tensor("log_scales", info("log_scales"), Dtype::F32, &[n, 3])?;
    check_tensor("quats", info("quats"), Dtype::F32, &[n, 4])?;
    check_tensor("opacity_logits", info("opacity_logits"), Dtype::F32, &[n])?;
    check_tensor("sh0", info("sh0"), Dtype::F32, &[n, 1, 3])?;
    let sh_n = info("shN");
    match sh_n.shape[..] {
        [rows, kk, 3] if rows == n && kk != k - 1 => {
            return Err(CoreError::Sh(format!(
                "shN has shape {:?}, expected [{n}, {}, 3] for sh_degree {}",
                sh_n.shape,
                k - 1,
                m.sh_degree
            )));
        }
        _ => check_tensor("shN", sh_n, Dtype::F32, &[n, k - 1, 3])?,
    }

    let tensor = |key: &str| file.bytes(info(key));
    let (tau0, span) = (m.time.tau0, m.time.span());
    // openftgs D-70: the unit of duration and velocity is T_τ, or 1 with a single timestamp.
    let unit = if span > 0.0 { span } else { 1.0 };
    let times = values(tensor("times"), f32_le);
    let log_durations = values(tensor("log_durations"), f32_le);
    let velocities = values(tensor("velocities"), f32s_le::<3>);
    let log_scales = values(tensor("log_scales"), f32s_le::<3>);
    let logits = values(tensor("opacity_logits"), f32_le);
    let sh0 = values(tensor("sh0"), f32_le);
    let sh_rest = values(tensor("shN"), f32_le);

    let parallel = n >= PARALLEL_ROWS;
    let time_center = map_rows(parallel, &times, |&t| tau0 + f64::from(t) * span);
    let duration = map_rows(parallel, &log_durations, |&d| f64::from(d).exp() * unit);
    let velocity = map_rows(parallel, &velocities, |v| {
        v.map(|x| (f64::from(x) / unit) as f32)
    });
    let scale = map_rows(parallel, &log_scales, |s| s.map(f32::exp));
    let opacity = map_rows(parallel, &logits, |&o| opacity_from_logit(o));
    let per_row = (k - 1) * 3;
    let mut sh = Vec::with_capacity(n * k * 3);
    for i in 0..n {
        sh.extend_from_slice(&sh0[3 * i..3 * i + 3]);
        sh.extend_from_slice(&sh_rest[per_row * i..per_row * (i + 1)]);
    }

    Ok(Model {
        position: values(tensor("means"), f32s_le::<3>),
        time_center,
        duration,
        velocity,
        scale,
        rotation: values(tensor("quats"), f32s_le::<4>),
        opacity,
        sh,
        sh_degree: m.sh_degree,
        time: m.time.clone(),
        conventions: RenderConventions::default(),
        producer: m.producer.clone(),
        identity: [0; 32],
        kind: ModelKind::Checkpoint {
            spatial_scale: m.spatial_scale,
            frame_dt: m.frame_dt,
        },
    })
}
