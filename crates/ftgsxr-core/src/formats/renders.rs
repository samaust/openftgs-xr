// SPDX-License-Identifier: Apache-2.0

//! The `ftgs-renders` writer and reader (§1.7, openftgs §2.2, openftgs `io.write_renders`
//! and `io.read_renders`).

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};

use safetensors::tensor::{Dtype, SafeTensors, View};

use super::json::check_version;
use crate::types::CoreError;

/// The renders' `format`.
const FORMAT: &str = "ftgs-renders";
/// The prefix of a color render's key.
const RGB: &str = "rgb/";
/// The prefix of an opacity render's key.
const ALPHA: &str = "alpha/";

/// One render as `ftgs-renders` stores it.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderImage {
    /// W in pixels.
    pub width: u32,
    /// H in pixels.
    pub height: u32,
    /// Row-major \[H, W, 3\] linear RGB, not clamped to 1.
    pub rgb: Vec<f32>,
    /// Row-major \[H, W, 1\] accumulated opacity.
    pub alpha: Vec<f32>,
}

/// A float32 tensor of the file, its values in little-endian bytes.
struct F32Tensor {
    shape: [usize; 3],
    bytes: Vec<u8>,
}

impl F32Tensor {
    fn new(height: u32, width: u32, channels: usize, values: &[f32]) -> F32Tensor {
        F32Tensor {
            shape: [height as usize, width as usize, channels],
            bytes: values.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

impl View for &F32Tensor {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

/// The 64 lowercase hex digits of a digest.
fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Checks one image before it is written: its buffers hold H·W·3 and H·W values, all
/// finite.
fn check_image(id: &str, image: &RenderImage) -> Result<(), CoreError> {
    let n = image.width as usize * image.height as usize;
    if image.rgb.len() != 3 * n || image.alpha.len() != n {
        return Err(CoreError::Shape(format!(
            "render {id:?}: rgb holds {} and alpha {} values, expected {} and {n} for {} × {}",
            image.rgb.len(),
            image.alpha.len(),
            3 * n,
            image.width,
            image.height
        )));
    }
    if !image.rgb.iter().chain(&image.alpha).all(|x| x.is_finite()) {
        return Err(CoreError::Value(format!(
            "render {id:?} (a non-finite value)"
        )));
    }
    Ok(())
}

/// Writes `items` as a `ftgs-renders` v1 file (§1.7, openftgs §2.2): `rgb/<id>` as
/// float32 \[H, W, 3\], unclamped, and `alpha/<id>` as float32 \[H, W, 1\]; the metadata
/// `format`, `version` "1", `producer` and `request_sha256` as 64 lowercase hex digits. The
/// file is written to a temporary sibling renamed over `path`, so a failed write leaves
/// the old file or none. At least one item, non-empty unique ids, buffers that match each
/// size and finite values are required (openftgs's checks; codes of Q-25).
// Q-25
pub fn write_renders(
    path: &std::path::Path,
    items: &[(String, RenderImage)],
    producer: &str,
    request_sha256: [u8; 32],
) -> Result<(), CoreError> {
    if items.is_empty() {
        return Err(CoreError::Value("renders (no item)".to_owned()));
    }
    let mut seen = HashSet::new();
    for (id, image) in items {
        if id.is_empty() {
            return Err(CoreError::Value("render id \"\"".to_owned()));
        }
        if !seen.insert(id.as_str()) {
            return Err(CoreError::Value(format!("render id {id:?} (repeated)")));
        }
        check_image(id, image)?;
    }
    let tensors: Vec<(String, F32Tensor)> = items
        .iter()
        .flat_map(|(id, image)| {
            [
                (
                    format!("{RGB}{id}"),
                    F32Tensor::new(image.height, image.width, 3, &image.rgb),
                ),
                (
                    format!("{ALPHA}{id}"),
                    F32Tensor::new(image.height, image.width, 1, &image.alpha),
                ),
            ]
        })
        .collect();
    let metadata = HashMap::from([
        ("format".to_owned(), FORMAT.to_owned()),
        ("version".to_owned(), "1".to_owned()),
        ("producer".to_owned(), producer.to_owned()),
        ("request_sha256".to_owned(), hex(&request_sha256)),
    ]);
    // safetensors writes through a sibling temporary file and renames it over `path`.
    safetensors::serialize_to_file(
        tensors.iter().map(|(k, t)| (k.as_str(), t)),
        Some(metadata),
        path,
    )
    .map_err(|e| match e {
        safetensors::SafeTensorError::IoError(io) => CoreError::Io(io),
        other => CoreError::Io(std::io::Error::other(other.to_string())),
    })
}

/// One tensor read as float32 values, checked for dtype, shape \[H, W, `channels`\] and
/// finite values.
fn tensor(
    st: &SafeTensors<'_>,
    key: &str,
    channels: usize,
) -> Result<(u32, u32, Vec<f32>), CoreError> {
    let view = st
        .tensor(key)
        .map_err(|_| CoreError::Keys(key.to_owned()))?;
    let shape = view.shape();
    if view.dtype() != Dtype::F32 || shape.len() != 3 || shape[2] != channels {
        return Err(CoreError::Shape(format!(
            "{key} is {:?} {shape:?}, expected F32 [H, W, {channels}]",
            view.dtype()
        )));
    }
    let size =
        |x: usize| u32::try_from(x).map_err(|_| CoreError::Shape(format!("{key} {shape:?}")));
    let (height, width) = (size(shape[0])?, size(shape[1])?);
    let values: Vec<f32> = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if !values.iter().all(|x| x.is_finite()) {
        return Err(CoreError::Value(format!("{key} (a non-finite value)")));
    }
    Ok((width, height, values))
}

/// Reads and checks a `ftgs-renders` file (§1.7, openftgs §2.2, openftgs `io.read_renders`):
/// the `format` and a version of major 1; with `expected_request_sha256`, a
/// `request_sha256` equal to it (else E-FILE-CHECKSUM); at least one render, every
/// `rgb/<id>` paired with its `alpha/<id>`, float32 \[H, W, 3\] and \[H, W, 1\] of one size,
/// finite. Other tensors are ignored. The file keeps no item order, so the renders come
/// back sorted by id. Codes of Q-25.
// Q-25
pub fn read_renders(
    bytes: &[u8],
    expected_request_sha256: Option<[u8; 32]>,
) -> Result<Vec<(String, RenderImage)>, CoreError> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| CoreError::Format(format!("not a safetensors file ({e})")))?;
    let (_, header) = SafeTensors::read_metadata(bytes)
        .map_err(|e| CoreError::Format(format!("not a safetensors file ({e})")))?;
    let meta = header.metadata().clone().unwrap_or_default();
    match meta.get("format") {
        Some(f) if f == FORMAT => {}
        Some(f) => {
            return Err(CoreError::Format(format!(
                "format {f:?}, expected {FORMAT:?}"
            )));
        }
        None => return Err(CoreError::Keys("format".to_owned())),
    }
    check_version(
        meta.get("version")
            .ok_or_else(|| CoreError::Keys("version".to_owned()))?,
    )?;
    if let Some(expected) = expected_request_sha256 {
        let found = meta.get("request_sha256");
        if found.map(String::as_str) != Some(hex(&expected).as_str()) {
            return Err(CoreError::Checksum(format!(
                "renders made for the request {found:?}, expected {}",
                hex(&expected)
            )));
        }
    }
    let names = st.names();
    let rgb: BTreeMap<&str, &str> = names
        .iter()
        .filter_map(|k| k.strip_prefix(RGB).map(|id| (id, *k)))
        .collect();
    let alpha: BTreeMap<&str, &str> = names
        .iter()
        .filter_map(|k| k.strip_prefix(ALPHA).map(|id| (id, *k)))
        .collect();
    if let Some(id) = rgb.keys().find(|id| !alpha.contains_key(*id)) {
        return Err(CoreError::Keys(format!("{ALPHA}{id}")));
    }
    if let Some(id) = alpha.keys().find(|id| !rgb.contains_key(*id)) {
        return Err(CoreError::Keys(format!("{RGB}{id}")));
    }
    if rgb.is_empty() {
        return Err(CoreError::Keys(format!("{RGB}<id>")));
    }
    let mut out = Vec::with_capacity(rgb.len());
    for (id, key) in rgb {
        if id.is_empty() {
            return Err(CoreError::Value(format!("render id \"\" ({key})")));
        }
        let (width, height, rgb) = tensor(&st, key, 3)?;
        let (aw, ah, alpha) = tensor(&st, alpha[id], 1)?;
        if (aw, ah) != (width, height) {
            return Err(CoreError::Shape(format!(
                "{ALPHA}{id} is {ah} × {aw}, expected {height} × {width} as {key}"
            )));
        }
        out.push((
            id.to_owned(),
            RenderImage {
                width,
                height,
                rgb,
                alpha,
            },
        ));
    }
    Ok(out)
}
