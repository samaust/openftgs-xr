// SPDX-License-Identifier: Apache-2.0

//! The safetensors container (§1.2, spec Appendix A row 31): an 8-byte little-endian header
//! length, a UTF-8 JSON header that begins with `{` and may be padded with spaces, then the
//! tensor data at the header's `data_offsets`. Every fault of the container is E-FILE-FORMAT
//! (Q-T12.2); the tensors' dtypes and shapes are the readers' to check.

use std::collections::HashMap;

use bytemuck::AnyBitPattern;
use safetensors::tensor::{Dtype, Metadata, TensorInfo};

use super::{SAFETENSORS_PREFIX, header_len};
use crate::types::CoreError;

/// The largest header §1.2 accepts: a header above 100 MB is refused.
const MAX_HEADER: u64 = 100_000_000;

/// A parsed safetensors file: its header and its data section.
pub(crate) struct Container<'a> {
    header: Metadata,
    data: &'a [u8],
}

/// The JSON header alone (the bytes after the prefix), parsed and its offsets checked to
/// cover a data section of `data_len` bytes without holes (E-FILE-FORMAT otherwise).
// Q-T12.2
pub(crate) fn parse_header(header: &[u8]) -> Result<Metadata, CoreError> {
    if header.len() as u64 > MAX_HEADER {
        return Err(CoreError::Format(format!(
            "a safetensors header of {} bytes, above 100 MB",
            header.len()
        )));
    }
    if header.first() != Some(&b'{') {
        return Err(CoreError::Format(
            "a safetensors header that does not begin with {".to_owned(),
        ));
    }
    serde_json::from_slice::<Metadata>(header)
        .map_err(|e| CoreError::Format(format!("not a safetensors header ({e})")))
}

impl<'a> Container<'a> {
    /// Parses a whole safetensors file (§1.2); any fault is E-FILE-FORMAT.
    // Q-T12.2
    pub(crate) fn parse(bytes: &'a [u8]) -> Result<Container<'a>, CoreError> {
        let prefix: &[u8; SAFETENSORS_PREFIX] = bytes
            .first_chunk()
            .ok_or_else(|| CoreError::Format("a file shorter than 8 bytes".to_owned()))?;
        let n = header_len(prefix);
        if n > MAX_HEADER {
            return Err(CoreError::Format(format!(
                "a safetensors header of {n} bytes, above 100 MB"
            )));
        }
        let start = SAFETENSORS_PREFIX + n as usize;
        let header = bytes.get(SAFETENSORS_PREFIX..start).ok_or_else(|| {
            CoreError::Format(format!(
                "a header of {n} bytes in a file of {}",
                bytes.len()
            ))
        })?;
        let header = parse_header(header)?;
        let data = &bytes[start..];
        if header.data_len() != data.len() {
            return Err(CoreError::Format(format!(
                "tensor data of {} bytes where the header gives {}",
                data.len(),
                header.data_len()
            )));
        }
        Ok(Container { header, data })
    }

    /// The parsed header.
    pub(crate) fn header(&self) -> &Metadata {
        &self.header
    }

    /// The bytes of tensor `name`, which the caller has found in the header.
    pub(crate) fn bytes(&self, info: &TensorInfo) -> &'a [u8] {
        let (s, e) = info.data_offsets;
        &self.data[s..e]
    }
}

/// The header's `__metadata__` map (empty when absent).
pub(crate) fn metadata(header: &Metadata) -> HashMap<String, String> {
    header.metadata().clone().unwrap_or_default()
}

/// The dtype name §1.2's tables use.
pub(crate) fn dtype_name(dtype: Dtype) -> String {
    match dtype {
        Dtype::F32 => "float32".to_owned(),
        Dtype::F64 => "float64".to_owned(),
        other => format!("{other:?}"),
    }
}

/// Checks a tensor's dtype and shape: the dtype must be `dtype` and the shape `shape`
/// (E-FILE-SHAPE otherwise).
pub(crate) fn check_tensor(
    key: &str,
    info: &TensorInfo,
    dtype: Dtype,
    shape: &[usize],
) -> Result<(), CoreError> {
    if info.dtype != dtype {
        return Err(CoreError::Shape(format!(
            "{key} has dtype {}, expected {}",
            dtype_name(info.dtype),
            dtype_name(dtype)
        )));
    }
    if info.shape != shape {
        return Err(CoreError::Shape(format!(
            "{key} has shape {:?}, expected {shape:?}",
            info.shape
        )));
    }
    Ok(())
}

/// Little-endian values as `T` (a float or an array of floats): cast in place when the bytes
/// are aligned for `T` (and the machine is little-endian), copied value by value otherwise;
/// in both cases into an owned vector (§1.2: "no alignment of the data is guaranteed").
/// `from_le` reads one `T` from its `size_of::<T>()` bytes.
pub(crate) fn values<T: AnyBitPattern>(bytes: &[u8], from_le: impl Fn(&[u8]) -> T) -> Vec<T> {
    if cfg!(target_endian = "little")
        && let Ok(cast) = bytemuck::try_cast_slice::<u8, T>(bytes)
    {
        return cast.to_vec();
    }
    bytes.chunks_exact(size_of::<T>()).map(from_le).collect()
}

/// One little-endian float32 from 4 bytes.
pub(crate) fn f32_le(b: &[u8]) -> f32 {
    f32::from_le_bytes(b.try_into().expect("invariant: chunks of 4 bytes"))
}

/// One little-endian float64 from 8 bytes.
pub(crate) fn f64_le(b: &[u8]) -> f64 {
    f64::from_le_bytes(b.try_into().expect("invariant: chunks of 8 bytes"))
}

/// `K` little-endian float32 values from `4 K` bytes.
pub(crate) fn f32s_le<const K: usize>(b: &[u8]) -> [f32; K] {
    std::array::from_fn(|i| f32_le(&b[4 * i..4 * i + 4]))
}
