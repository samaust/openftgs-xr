// SPDX-License-Identifier: Apache-2.0

//! The safetensors container writer of the testkit (Q-1; §1.2's container, spec
//! Appendix A row 31): the bytes it builds are read back here by hand, from §1.2's
//! description of the container, not through any reader of the workspace.

use ftgsxr_testkit::SafetensorsBuilder;
use serde_json::{Value, json};

/// Splits a container into (header length, header text, data) per §1.2: an 8-byte
/// little-endian length, the JSON header, then the data.
fn split(bytes: &[u8]) -> (usize, &str, &[u8]) {
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + len]).expect("the header is UTF-8");
    (len, header, &bytes[8 + len..])
}

fn two_tensors() -> SafetensorsBuilder {
    SafetensorsBuilder::new()
        .metadata("format", "ftgs-if")
        .metadata("quoted", r#"{"a":[1,2],"b":"x\"y"}"#)
        .tensor("first", "F32", &[2], vec![1, 2, 3, 4, 5, 6, 7, 8])
        .tensor("second", "F64", &[1, 1], vec![9; 8])
}

#[test]
fn builder_writes_the_container_of_section_1_2() {
    let bytes = two_tensors().build();
    let (len, header, data) = split(&bytes);
    assert!(header.starts_with('{'), "{header:?}");
    assert_eq!(
        (8 + len) % 8,
        0,
        "the data starts 8-byte aligned by default"
    );
    let v: Value = serde_json::from_str(header).expect("the header is JSON");
    assert_eq!(
        v["__metadata__"],
        json!({"format": "ftgs-if", "quoted": r#"{"a":[1,2],"b":"x\"y"}"#})
    );
    assert_eq!(
        v["first"],
        json!({"dtype": "F32", "shape": [2], "data_offsets": [0, 8]})
    );
    assert_eq!(
        v["second"],
        json!({"dtype": "F64", "shape": [1, 1], "data_offsets": [8, 16]})
    );
    assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9]);
}

#[test]
fn builder_without_metadata_writes_no_metadata_entry() {
    let bytes = SafetensorsBuilder::new()
        .tensor("t", "F32", &[1], vec![0; 4])
        .build();
    let v: Value = serde_json::from_str(split(&bytes).1).unwrap();
    assert!(v.get("__metadata__").is_none(), "{v}");
}

#[test]
fn builder_writes_what_it_is_given_even_when_invalid() {
    let bytes = SafetensorsBuilder::new()
        .tensor("odd", "X99", &[5, 7], vec![1, 2, 3])
        .build();
    let (_, header, data) = split(&bytes);
    let v: Value = serde_json::from_str(header).unwrap();
    assert_eq!(
        v["odd"],
        json!({"dtype": "X99", "shape": [5, 7], "data_offsets": [0, 3]})
    );
    assert_eq!(data, [1, 2, 3]);
}

#[test]
fn pad_header_adds_spaces_and_keeps_the_header() {
    let plain = two_tensors().build();
    let padded = two_tensors().pad_header(13).build();
    let (_, plain_header, plain_data) = split(&plain);
    let (len, header, data) = split(&padded);
    assert!(header.ends_with(&" ".repeat(13)), "{header:?}");
    assert_eq!((8 + len) % 8, 0);
    assert_eq!(header.trim_end(), plain_header.trim_end());
    assert_eq!(data, plain_data);
}

#[test]
fn misalign_data_moves_the_data_off_the_8_byte_boundary() {
    let plain = two_tensors().build();
    for k in 1..8 {
        let bytes = two_tensors().misalign_data(k).build();
        let (len, header, data) = split(&bytes);
        assert_eq!((8 + len) % 8, k, "misalign_data({k})");
        assert_eq!(header.trim_end(), split(&plain).1.trim_end());
        assert_eq!(data, split(&plain).2);
    }
}

#[test]
fn a_header_above_100_mb_can_be_built() {
    let bytes = two_tensors().pad_header(100 * 1024 * 1024 + 1).build();
    let (len, _, _) = split(&bytes);
    assert!(len > 100 * 1024 * 1024);
}

/// The `safetensors` crate (§2.3, the reader crate of §1.2's container) accepts what the
/// builder writes, padded and misaligned alike, and finds the same tensors.
#[test]
fn the_safetensors_crate_reads_the_builder_output() {
    for bytes in [
        two_tensors().build(),
        two_tensors().pad_header(13).build(),
        two_tensors().misalign_data(3).build(),
    ] {
        let (_, metadata) =
            safetensors::SafeTensors::read_metadata(&bytes).expect("a valid header");
        let meta = metadata.metadata().as_ref().expect("metadata present");
        assert_eq!(meta["format"], "ftgs-if");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("a valid container");
        let first = st.tensor("first").unwrap();
        assert_eq!(first.dtype(), safetensors::Dtype::F32);
        assert_eq!(first.shape(), [2]);
        assert_eq!(first.data(), [1, 2, 3, 4, 5, 6, 7, 8]);
        let second = st.tensor("second").unwrap();
        assert_eq!(second.dtype(), safetensors::Dtype::F64);
        assert_eq!(second.shape(), [1, 1]);
    }
}
