// SPDX-License-Identifier: Apache-2.0

//! A safetensors container writer for tests (Q-1): it writes §1.2's container bytes
//! directly, so U-2 and U-3 can make files a conforming writer refuses to produce.

/// The data section's alignment the header's padding reaches by default, as the
/// safetensors writer pads its header (spec Appendix A row 31).
const ALIGN: usize = 8;

/// One tensor entry: name, dtype string, shape and its bytes as given.
#[derive(Clone, Debug)]
struct Entry {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// Builds safetensors container bytes (§1.2): an 8-byte little-endian header length, a
/// compact UTF-8 JSON header padded with spaces, then the tensors' bytes. Nothing is
/// validated: any dtype string, any shape and any byte count are written as given, so
/// invalid files can be made.
///
/// The header holds `__metadata__` first (omitted when no metadata was given), then the
/// tensors in call order, each `{"dtype", "shape", "data_offsets"}`; the data holds their
/// bytes in the same order, back to back from offset 0, so `data_offsets` always cover the
/// data without holes. By default the header is padded with the fewest spaces that start
/// the data at a multiple of 8 bytes in the file.
///
/// ```
/// use ftgsxr_testkit::SafetensorsBuilder;
/// let bytes = SafetensorsBuilder::new()
///     .metadata("format", "ftgs-if")
///     .tensor("opacity", "F32", &[1], 0.5f32.to_le_bytes().to_vec())
///     .build();
/// assert_eq!(bytes.len() % 8, 4);
/// ```
#[derive(Clone, Debug, Default)]
pub struct SafetensorsBuilder {
    metadata: Vec<(String, String)>,
    tensors: Vec<Entry>,
    pad: usize,
    misalign: usize,
}

impl SafetensorsBuilder {
    /// An empty container: no metadata, no tensors.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `key: value` to `__metadata__`, in call order (a repeated key is written twice).
    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.push((key.to_owned(), value.to_owned()));
        self
    }

    /// Adds a tensor entry with the dtype string (`"F32"`, `"F64"`, or anything else),
    /// the shape and the bytes as given; its data follows the previous tensor's.
    pub fn tensor(mut self, name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>) -> Self {
        self.tensors.push(Entry {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape: shape.to_vec(),
            bytes,
        });
        self
    }

    /// Appends `spaces` spaces to the JSON header (§1.2: "may be padded with spaces"),
    /// before the default alignment padding; calls add up.
    pub fn pad_header(mut self, spaces: usize) -> Self {
        self.pad += spaces;
        self
    }

    /// Pads the header with `bytes` mod 8 more spaces, so the data section starts that
    /// many bytes past a multiple of 8 in the file and the tensors' data is unaligned
    /// (§1.2's copy path); `data_offsets` stay contiguous from 0. The last call wins.
    pub fn misalign_data(mut self, bytes: usize) -> Self {
        self.misalign = bytes % ALIGN;
        self
    }

    /// The container's bytes.
    pub fn build(self) -> Vec<u8> {
        let mut header = String::from("{");
        let mut first = true;
        let mut sep = |header: &mut String| {
            if !first {
                header.push(',');
            }
            first = false;
        };
        if !self.metadata.is_empty() {
            sep(&mut header);
            header.push_str("\"__metadata__\":{");
            for (i, (key, value)) in self.metadata.iter().enumerate() {
                if i > 0 {
                    header.push(',');
                }
                header.push_str(&json_string(key));
                header.push(':');
                header.push_str(&json_string(value));
            }
            header.push('}');
        }
        let mut offset = 0usize;
        for t in &self.tensors {
            sep(&mut header);
            let shape: Vec<String> = t.shape.iter().map(usize::to_string).collect();
            let end = offset + t.bytes.len();
            header.push_str(&format!(
                "{}:{{\"dtype\":{},\"shape\":[{}],\"data_offsets\":[{offset},{end}]}}",
                json_string(&t.name),
                json_string(&t.dtype),
                shape.join(","),
            ));
            offset = end;
        }
        header.push('}');
        header.push_str(&" ".repeat(self.pad));
        let start = 8 + header.len();
        let align = (self.misalign + ALIGN - start % ALIGN) % ALIGN;
        header.push_str(&" ".repeat(align));

        let mut out = Vec::with_capacity(8 + header.len() + offset);
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for t in self.tensors {
            out.extend_from_slice(&t.bytes);
        }
        out
    }
}

/// `s` as a JSON string literal.
fn json_string(s: &str) -> String {
    serde_json::to_string(s).expect("invariant: a string serializes to JSON")
}
