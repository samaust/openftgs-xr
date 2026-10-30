// SPDX-License-Identifier: Apache-2.0

//! R-15, `ftgsxr render` end to end through `cmd_render` (§1.7): P1's request gives a
//! `ftgs-renders` file with openftgs's keys (`rgb/<id>`, `alpha/<id>` for every item) and
//! metadata (`format`, `version`, `producer` = `openftgs-xr <version> (<adapter>)`) and
//! the request's SHA-256 (`gpu` lane); a request in another time unit (E-REQ-UNIT) and a
//! request degree above the model's (E-REQ-DEGREE) exit 2 and write no file (`cpu` lane,
//! the checks come before the GPU, Q-T118.2).

use std::fs;
use std::path::{Path, PathBuf};

use ftgsxr_app::{HeadlessError, QualityTier, cmd_render, exit_code};
use ftgsxr_core::{read_renders, read_request};
use ftgsxr_testkit::{fixtures_dir, scratch_dir};

const VERSION: &str = "0.1.0-test";

fn p1(extension: &str) -> PathBuf {
    fixtures_dir().join(format!("probes/P1.{extension}"))
}

fn scratch(name: &str) -> PathBuf {
    scratch_dir(env!("CARGO_TARGET_TMPDIR"), env!("CARGO_CRATE_NAME"), name)
}

/// The value of the metadata key `key` in a safetensors file's header, a JSON string
/// without escapes (the producer and digest strings `cmd_render` writes).
fn metadata(bytes: &[u8], key: &str) -> String {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
    let needle = format!("\"{key}\":\"");
    let start = header
        .find(&needle)
        .unwrap_or_else(|| panic!("no {key} in {header}"))
        + needle.len();
    let end = start + header[start..].find('"').unwrap();
    header[start..end].to_owned()
}

/// P1's request with `field`'s JSON value replaced by `value`, written into `dir`.
fn altered_request(dir: &Path, field: &str, value: &str) -> PathBuf {
    let text = fs::read_to_string(p1("request.json")).unwrap();
    let needle = format!("\"{field}\": ");
    let start = text.find(&needle).expect("the field") + needle.len();
    let end = start + text[start..].find([',', '}']).unwrap();
    let altered = format!("{}{value}{}", &text[..start], &text[end..]);
    let path = dir.join("request.json");
    fs::write(&path, altered).unwrap();
    path
}

/// `cmd_render` on `request` refuses with `Input`, the code `code` and exit 2, and leaves
/// no file at `OUT`.
fn refused(request: &Path, dir: &Path, code: &str) -> HeadlessError {
    let out = dir.join("out.safetensors");
    let e = cmd_render(
        &p1("safetensors"),
        request,
        &out,
        QualityTier::conformance(),
        VERSION,
    )
    .expect_err("refused");
    assert!(matches!(e, HeadlessError::Input(_)), "{e:?}");
    assert_eq!(e.code(), code, "{e}");
    assert_eq!(exit_code(&e), 2);
    assert!(!out.exists(), "no file is written");
    e
}

#[test]
fn r15_a_request_in_another_time_unit_exits_2() {
    let dir = scratch("unit");
    let request = altered_request(&dir, "time_unit", "\"ms\"");
    let e = refused(&request, &dir, "E-REQ-UNIT");
    assert!(
        e.user_text().contains("Request time unit ms, model s"),
        "{e}"
    );
}

#[test]
fn r15_a_request_degree_above_the_models_exits_2() {
    // P1 is degree 0.
    let dir = scratch("degree");
    let request = altered_request(&dir, "sh_degree", "1");
    let e = refused(&request, &dir, "E-REQ-DEGREE");
    assert!(
        e.user_text().contains("Request degree 1, model degree 0"),
        "{e}"
    );
}

#[test]
#[ignore = "gpu"]
fn gpu_r15_p1_request_writes_the_renders_file() {
    let dir = scratch("p1");
    let out = dir.join("P1.renders.safetensors");
    cmd_render(
        &p1("safetensors"),
        &p1("request.json"),
        &out,
        QualityTier::conformance(),
        VERSION,
    )
    .expect("P1 renders");
    let request = read_request(&fs::read(p1("request.json")).unwrap()).unwrap();
    let bytes = fs::read(&out).unwrap();
    // `read_renders` checks the format, the major version, the pairs of keys, their dtype
    // and shape and the request's SHA-256.
    let renders = read_renders(&bytes, Some(request.sha256)).expect("a ftgs-renders file");
    let mut ids: Vec<&str> = request.items.iter().map(|i| i.id.as_str()).collect();
    ids.sort_unstable();
    let got: Vec<&str> = renders.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(got, ids);
    for (id, image) in &renders {
        let item = request.items.iter().find(|i| i.id == *id).unwrap();
        assert_eq!(
            (image.width, image.height),
            (item.camera.width, item.camera.height)
        );
    }
    assert_eq!(metadata(&bytes, "format"), "ftgs-renders");
    assert_eq!(metadata(&bytes, "version"), "1");
    let hex: String = request.sha256.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(metadata(&bytes, "request_sha256"), hex);
    let producer = metadata(&bytes, "producer");
    let adapter = producer
        .strip_prefix(&format!("openftgs-xr {VERSION} ("))
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or_else(|| panic!("producer {producer:?}"));
    assert!(!adapter.is_empty(), "producer {producer:?}");
}
