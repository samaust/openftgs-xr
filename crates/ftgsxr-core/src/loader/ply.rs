// SPDX-License-Identifier: Apache-2.0

//! The PLY snapshot (§1.2, D-41, openftgs `io.write_ply`): a binary little-endian PLY with
//! one `vertex` element of float32 properties, read as a single-timestamp model.

use std::collections::HashMap;

use super::exchange::finish;
use super::{PARALLEL_ROWS, map_rows, opacity_from_logit, sha256};
use crate::types::{CoreError, Model, ModelKind, RenderConventions, TimeSpan};

/// The longest header searched for `end_header` (a header of openftgs's layout at degree 3
/// is about 1 KB).
const MAX_HEADER: usize = 1 << 20;

/// The properties every snapshot holds, besides the `f_rest_i` (§1.2).
const NAMED: [&str; 17] = [
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
    "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
];

/// The checked header: N, each property's column, the `f_rest` count, the producer.
struct Header {
    n: usize,
    columns: HashMap<String, usize>,
    rest: usize,
    producer: String,
    body_start: usize,
}

/// Parses and checks the header (§1.2, D-41; codes Q-35).
// Q-35
fn read_header(bytes: &[u8]) -> Result<Header, CoreError> {
    let format = |why: &str| CoreError::Format(format!("not a PLY snapshot ({why})"));
    let window = &bytes[..bytes.len().min(MAX_HEADER)];
    const END: &[u8] = b"end_header";
    let end = window
        .windows(END.len())
        .position(|w| w == END)
        .ok_or_else(|| format("no end_header"))?;
    let after = end + END.len();
    let body_start = match bytes.get(after..) {
        Some([b'\n', ..]) => after + 1,
        Some([b'\r', b'\n', ..]) => after + 2,
        _ => return Err(format("end_header not ended by a newline")),
    };
    let text = std::str::from_utf8(&bytes[..after]).map_err(|_| format("a header not ASCII"))?;
    let mut lines = text.lines().map(str::trim_end);
    if lines.next() != Some("ply") {
        return Err(format("no ply magic"));
    }

    let mut format_seen = false;
    let mut producer: Option<String> = None;
    let mut n: Option<usize> = None;
    let mut names: Vec<String> = Vec::new();
    for line in lines {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            [] | ["end_header"] => {}
            ["format", "binary_little_endian", "1.0"] if !format_seen => format_seen = true,
            ["format", ..] => return Err(format(&format!("{line:?}"))),
            ["comment", ..] => {
                if producer.is_none() {
                    let text = line.trim_start().strip_prefix("comment").unwrap_or("");
                    producer = Some(text.trim().to_owned());
                }
            }
            ["obj_info", ..] => {}
            ["element", "vertex", count] if n.is_none() => {
                n = Some(count.parse().map_err(|_| format(&format!("{line:?}")))?);
            }
            ["element", ..] => return Err(format(&format!("{line:?}"))),
            ["property", kind, name] if n.is_some() => {
                if !matches!(*kind, "float" | "float32") {
                    return Err(CoreError::Shape(format!(
                        "PLY property {name} has type {kind}, expected float32"
                    )));
                }
                names.push((*name).to_owned());
            }
            ["property", "list", ..] if n.is_some() => {
                return Err(CoreError::Shape(format!(
                    "PLY {line:?}, expected float32 properties"
                )));
            }
            _ => return Err(format(&format!("{line:?}"))),
        }
    }
    if !format_seen {
        return Err(format("no binary_little_endian format line"));
    }
    let n = n.ok_or_else(|| format("no vertex element"))?;

    let mut columns = HashMap::with_capacity(names.len());
    let mut rest = 0usize;
    for (column, name) in names.iter().enumerate() {
        let known = NAMED.contains(&name.as_str())
            || name
                .strip_prefix("f_rest_")
                .is_some_and(|i| !i.is_empty() && i.bytes().all(|b| b.is_ascii_digit()));
        if !known {
            return Err(format(&format!("unknown property {name}")));
        }
        if columns.insert(name.clone(), column).is_some() {
            return Err(format(&format!("property {name} twice")));
        }
        if name.starts_with("f_rest_") {
            rest += 1;
        }
    }
    for name in NAMED {
        if !columns.contains_key(name) {
            return Err(CoreError::Keys(name.to_owned()));
        }
    }
    for i in 0..rest {
        let name = format!("f_rest_{i}");
        if !columns.contains_key(&name) {
            return Err(CoreError::Keys(name));
        }
    }
    Ok(Header {
        n,
        columns,
        rest,
        producer: producer.unwrap_or_default(),
        body_start,
    })
}

/// Reads a PLY snapshot (§1.2, D-41) as a single-timestamp model: τ₀ = τ₁ = 0 s, every
/// time center 0, duration 1 and velocity 0; position (x, y, z); scale exp(scale_i);
/// opacity the sigmoid of the logit, clamped as a checkpoint's; SH from `f_dc_*` and the
/// channel-major `f_rest_*`, K = 1 + (count of `f_rest`)/3, which must be (L+1)² for
/// L ∈ 0–3 (E-FILE-SH); the normals ignored; the background black; the producer the first
/// `comment` line, else empty. Only the binary little-endian format with one `vertex`
/// element of float32 properties named as openftgs's `write_ply` names them (in any order)
/// is read (Q-35). The model is then validated and its rotations normalized with the
/// canonical sign; `identity` is the SHA-256 of `bytes` (D-40).
pub fn read_ply(bytes: &[u8]) -> Result<Model, CoreError> {
    let (identity, model) = rayon::join(|| sha256(bytes), || decode(bytes));
    let mut model = model?;
    model.identity = identity;
    Ok(model)
}

/// The header, the body, the model, `Model::validate`.
fn decode(bytes: &[u8]) -> Result<Model, CoreError> {
    let h = read_header(bytes)?;
    let k = 1 + h.rest / 3;
    let sh_degree = match (h.rest % 3, k) {
        (0, 1) => 0u8,
        (0, 4) => 1,
        (0, 9) => 2,
        (0, 16) => 3,
        _ => {
            return Err(CoreError::Sh(format!(
                "{} f_rest properties, expected 0, 9, 24 or 45",
                h.rest
            )));
        }
    };
    let stride = h.columns.len() * 4;
    let body = &bytes[h.body_start..];
    if Some(body.len()) != h.n.checked_mul(stride) {
        return Err(CoreError::Format(format!(
            "a PLY body of {} bytes, expected {} vertices of {stride} bytes",
            body.len(),
            h.n
        )));
    }
    let rows: Vec<&[u8]> = body.chunks_exact(stride).collect();
    let col = |name: &str| 4 * h.columns[name];
    let get = |row: &[u8], at: usize| {
        f32::from_le_bytes(row[at..at + 4].try_into().expect("invariant: 4 bytes"))
    };
    let at = |names: [&str; 3]| names.map(col);
    let (xyz, scl, dc) = (
        at(["x", "y", "z"]),
        at(["scale_0", "scale_1", "scale_2"]),
        at(["f_dc_0", "f_dc_1", "f_dc_2"]),
    );
    let rot = ["rot_0", "rot_1", "rot_2", "rot_3"].map(col);
    let opacity_at = col("opacity");
    // Channel-major: the K − 1 higher coefficients of red, then of green, then of blue.
    let rest: Vec<usize> = (0..h.rest).map(|i| col(&format!("f_rest_{i}"))).collect();

    let parallel = h.n >= PARALLEL_ROWS;
    let position = map_rows(parallel, &rows, |r| xyz.map(|c| get(r, c)));
    let scale = map_rows(parallel, &rows, |r| {
        scl.map(|c| f64::from(get(r, c)).exp() as f32)
    });
    let rotation = map_rows(parallel, &rows, |r| rot.map(|c| get(r, c)));
    let opacity = map_rows(parallel, &rows, |r| opacity_from_logit(get(r, opacity_at)));
    let sh_rows = map_rows(parallel, &rows, |r| {
        let mut out = Vec::with_capacity(k * 3);
        out.extend(dc.map(|c| get(r, c)));
        for j in 0..k - 1 {
            for channel in 0..3 {
                out.push(get(r, rest[channel * (k - 1) + j]));
            }
        }
        out
    });

    let n = h.n;
    let mut model = Model {
        position,
        time_center: vec![0.0; n],
        duration: vec![1.0; n],
        velocity: vec![[0.0; 3]; n],
        scale,
        rotation,
        opacity,
        sh: sh_rows.concat(),
        sh_degree,
        time: TimeSpan {
            tau0: 0.0,
            tau1: 0.0,
            time_unit: "s".to_owned(),
        },
        conventions: RenderConventions::default(),
        producer: h.producer,
        identity: [0; 32],
        kind: ModelKind::Ply,
    };
    finish(&mut model, Ok(RenderConventions::default()))?;
    Ok(model)
}
