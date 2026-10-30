// SPDX-License-Identifier: Apache-2.0

//! U-4 (D-40 model identity, `cpu` lane): the SHA-256 of a fixture against `sha256sum`'s
//! (the fixtures' `SHA256SUMS`, which `sha256sum` wrote); two byte-identical copies have
//! equal identities; one byte changed gives a different identity.

use std::fs;
use std::path::PathBuf;

use ftgsxr_core::{hex, open_model, read_exchange, sha256};
use ftgsxr_testkit::{fixtures_dir, scratch_dir};

/// `sha256sum`'s digest of a fixture, from `SHA256SUMS`.
fn sha256sum(name: &str) -> String {
    let sums = fs::read_to_string(fixtures_dir().join("SHA256SUMS")).unwrap();
    sums.lines()
        .find(|line| line[66..] == *name)
        .map(|line| line[..64].to_owned())
        .unwrap_or_else(|| panic!("{name} in SHA256SUMS"))
}

/// A fresh scratch directory for this test.
/// Its folder is `ftgsxr_testkit::scratch_dir`'s, one per test binary and process (#94, Q-62).
fn scratch(name: &str) -> PathBuf {
    scratch_dir(env!("CARGO_TARGET_TMPDIR"), env!("CARGO_CRATE_NAME"), name)
}

#[test]
fn u04_sha256_of_fixtures_against_sha256sum() {
    for name in [
        "probes/P1.safetensors",
        "probes/R2_overlap_deg3.safetensors",
        "probes/R2_overlap_deg3.tau1.3.ply",
        "checkpoint/tiny.safetensors",
    ] {
        let path = fixtures_dir().join(name);
        let bytes = fs::read(&path).unwrap();
        assert_eq!(hex(&sha256(&bytes)), sha256sum(name), "{name}");
        assert_eq!(
            hex(&open_model(&path).unwrap().identity),
            sha256sum(name),
            "{name}"
        );
    }
}

#[test]
fn u04_checkpoint_identity_is_its_params_digest() {
    let dir = fixtures_dir().join("checkpoint/tiny");
    assert_eq!(
        hex(&open_model(&dir).unwrap().identity),
        sha256sum("checkpoint/tiny/params.safetensors")
    );
}

#[test]
fn u04_hex_is_64_lowercase_digits() {
    let mut id = [0u8; 32];
    id[0] = 0xab;
    id[31] = 0x0f;
    let h = hex(&id);
    assert_eq!(h.len(), 64);
    assert!(h.starts_with("ab00"));
    assert!(h.ends_with("000f"));
    // The SHA-256 of the empty string (FIPS 180-4's well-known value).
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn u04_identical_copies_are_one_model() {
    let source = fixtures_dir().join("probes/P2.safetensors");
    let dir = scratch("copies");
    let (a, b) = (dir.join("a.safetensors"), dir.join("elsewhere.safetensors"));
    fs::copy(&source, &a).unwrap();
    fs::copy(&source, &b).unwrap();
    let (ma, mb) = (open_model(&a).unwrap(), open_model(&b).unwrap());
    assert_eq!(ma.identity, mb.identity);
    assert_eq!(ma, mb);
}

#[test]
fn u04_one_byte_changed_is_another_model() {
    let bytes = fs::read(fixtures_dir().join("probes/P2.safetensors")).unwrap();
    let original = read_exchange(&bytes).unwrap();
    // The last data byte is the high byte of the last float32 value; flipping its lowest bit
    // gives another finite value. Find a byte that keeps the file valid.
    let mut changed = bytes.clone();
    let mut found = false;
    for i in (bytes.len() - 64..bytes.len()).rev() {
        changed[i] ^= 0x01;
        if read_exchange(&changed).is_ok() {
            found = true;
            break;
        }
        changed[i] ^= 0x01;
    }
    assert!(found, "a one-byte change that keeps the file valid");
    let other = read_exchange(&changed).unwrap();
    assert_ne!(other.identity, original.identity);
}
