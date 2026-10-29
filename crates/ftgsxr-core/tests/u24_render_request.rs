// SPDX-License-Identifier: Apache-2.0

//! U-24, the `ftgs-render-request` reader (§1.7, openftgs §2.2, `cpu` lane): the committed
//! probe requests parse; a non-orthonormal rotation, a skewed K, a duplicate id, an empty
//! item list and a bad version are refused; black background and null degree by default.
//! The refusals' codes are Q-25's.

use std::path::PathBuf;

use ftgsxr_core::{CoreError, read_request};
use ftgsxr_testkit::fixtures_dir;
use serde_json::{Value, json};

/// Every committed probe request, sorted by name.
fn probe_requests() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(fixtures_dir().join("probes"))
        .expect("the probes directory")
        .map(|e| e.expect("a directory entry").path())
        .filter(|p| p.to_string_lossy().ends_with(".request.json"))
        .collect();
    out.sort();
    out
}

/// The `request_sha256` openftgs recorded in the renders it wrote for `request`, read from
/// the safetensors header by hand (the 8-byte length, then the JSON header).
fn recorded_sha256(request: &std::path::Path) -> String {
    let renders = request
        .to_string_lossy()
        .replace(".request.json", ".renders.safetensors");
    let bytes = std::fs::read(renders).expect("the probe's renders");
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + n]).expect("the header");
    header["__metadata__"]["request_sha256"]
        .as_str()
        .expect("request_sha256")
        .to_owned()
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// P1's request as JSON, to mutate.
fn p1() -> Value {
    let bytes = std::fs::read(fixtures_dir().join("probes/P1.request.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn read(value: &Value) -> Result<ftgsxr_core::RenderRequest, CoreError> {
    read_request(&serde_json::to_vec(value).unwrap())
}

#[test]
fn u24_committed_probe_requests_parse() {
    let paths = probe_requests();
    assert!(paths.len() >= 18, "the probes of the fixtures README");
    for path in paths {
        let bytes = std::fs::read(&path).unwrap();
        let request =
            read_request(&bytes).unwrap_or_else(|e| panic!("{} refused: {e}", path.display()));
        let raw: Value = serde_json::from_slice(&bytes).unwrap();
        let items = raw["items"].as_array().unwrap();
        assert_eq!(request.items.len(), items.len());
        assert_eq!(request.time_unit, raw["time_unit"].as_str().unwrap());
        for (item, json) in request.items.iter().zip(items) {
            assert_eq!(item.id, json["id"].as_str().unwrap());
            assert_eq!(item.timestamp, json["timestamp"].as_f64().unwrap());
            let cam = &json["camera"];
            assert_eq!(u64::from(item.camera.width), cam["width"].as_u64().unwrap());
            assert_eq!(
                u64::from(item.camera.height),
                cam["height"].as_u64().unwrap()
            );
            for r in 0..3 {
                for c in 0..3 {
                    assert_eq!(item.camera.k.col(c)[r], cam["K"][r][c].as_f64().unwrap());
                }
            }
            for r in 0..4 {
                for c in 0..4 {
                    assert_eq!(
                        item.camera.world_to_camera.col(c)[r],
                        cam["world_to_camera"][r][c].as_f64().unwrap()
                    );
                }
            }
        }
        // The digest is of the file's bytes, as openftgs recorded it in its renders.
        assert_eq!(
            hex(&request.sha256),
            recorded_sha256(&path),
            "{}",
            path.display()
        );
    }
}

#[test]
fn u24_the_given_background_and_degree_are_read() {
    let mut raw = p1();
    raw["background"] = json!([1.0, 0.5, 0.25]);
    raw["sh_degree"] = json!(2);
    let request = read(&raw).unwrap();
    assert_eq!(request.background, [1.0, 0.5, 0.25]);
    assert_eq!(request.sh_degree, Some(2));
}

#[test]
fn u24_black_background_and_null_degree_by_default() {
    let mut raw = p1();
    let map = raw.as_object_mut().unwrap();
    map.remove("background");
    map.remove("sh_degree");
    let request = read(&raw).unwrap();
    assert_eq!(request.background, [0.0, 0.0, 0.0]);
    assert_eq!(request.sh_degree, None);
}

#[test]
fn u24_an_explicit_null_degree_is_the_models_degree() {
    let mut raw = p1();
    raw["sh_degree"] = Value::Null;
    assert_eq!(read(&raw).unwrap().sh_degree, None);
}

#[test]
fn u24_unknown_keys_are_ignored() {
    let mut raw = p1();
    raw["extra"] = json!({"anything": [1, 2]});
    raw["items"][0]["camera"]["model"] = json!("pinhole");
    assert!(read(&raw).is_ok());
}

#[test]
fn u24_a_non_orthonormal_rotation_is_refused() {
    let mut raw = p1();
    // Scaling one row by 1.001 moves RᵀR's diagonal by about 2·10⁻³.
    raw["items"][3]["camera"]["world_to_camera"][0][0] = json!(1.001);
    assert!(matches!(read(&raw), Err(CoreError::Value(_))));
}

#[test]
fn u24_a_reflection_is_refused() {
    let mut raw = p1();
    // Orthonormal but of determinant −1.
    raw["items"][0]["camera"]["world_to_camera"][0][0] = json!(-1.0);
    assert!(matches!(read(&raw), Err(CoreError::Value(_))));
}

#[test]
fn u24_a_rotation_within_the_tolerance_is_accepted() {
    let mut raw = p1();
    // 1 + 4·10⁻⁶ squared is 1 + 8·10⁻⁶ ≤ 1 + 10⁻⁵.
    raw["items"][0]["camera"]["world_to_camera"][0][0] = json!(1.000004);
    assert!(read(&raw).is_ok());
}

#[test]
fn u24_a_skewed_k_is_refused() {
    for (r, c) in [(0, 1), (1, 0), (2, 0), (2, 1)] {
        let mut raw = p1();
        raw["items"][1]["camera"]["K"][r][c] = json!(0.5);
        assert!(
            matches!(read(&raw), Err(CoreError::Value(_))),
            "K[{r}][{c}] = 0.5"
        );
    }
    let mut raw = p1();
    raw["items"][1]["camera"]["K"][2][2] = json!(2.0);
    assert!(matches!(read(&raw), Err(CoreError::Value(_))));
}

#[test]
fn u24_a_non_positive_focal_length_is_refused() {
    let mut raw = p1();
    raw["items"][0]["camera"]["K"][1][1] = json!(0.0);
    assert!(matches!(read(&raw), Err(CoreError::Value(_))));
}

#[test]
fn u24_a_world_to_camera_with_another_last_row_is_refused() {
    let mut raw = p1();
    raw["items"][0]["camera"]["world_to_camera"][3][2] = json!(0.1);
    assert!(matches!(read(&raw), Err(CoreError::Value(_))));
}

#[test]
fn u24_a_duplicate_id_is_refused() {
    let mut raw = p1();
    let first = raw["items"][0]["id"].clone();
    raw["items"][5]["id"] = first;
    assert!(matches!(read(&raw), Err(CoreError::Value(_))));
}

#[test]
fn u24_an_empty_id_is_refused() {
    let mut raw = p1();
    raw["items"][0]["id"] = json!("");
    assert!(matches!(read(&raw), Err(CoreError::Value(_))));
}

#[test]
fn u24_an_empty_item_list_is_refused() {
    let mut raw = p1();
    raw["items"] = json!([]);
    assert!(matches!(read(&raw), Err(CoreError::Value(_))));
}

#[test]
fn u24_a_bad_version_is_refused() {
    for version in [
        json!(2),
        json!("2"),
        json!("2.0"),
        json!("0"),
        json!("x"),
        json!(1.0),
    ] {
        let mut raw = p1();
        raw["version"] = version.clone();
        assert!(
            matches!(read(&raw), Err(CoreError::Version(_))),
            "version {version}"
        );
    }
    let mut raw = p1();
    raw.as_object_mut().unwrap().remove("version");
    assert!(matches!(read(&raw), Err(CoreError::Keys(_))));
}

#[test]
fn u24_a_minor_version_is_accepted() {
    for version in [json!(1), json!("1"), json!("1.3")] {
        let mut raw = p1();
        raw["version"] = version;
        assert!(read(&raw).is_ok());
    }
}

#[test]
fn u24_another_format_is_refused() {
    let mut raw = p1();
    raw["format"] = json!("ftgs-scene");
    assert!(matches!(read(&raw), Err(CoreError::Format(_))));
}

#[test]
fn u24_a_background_outside_the_unit_range_is_refused() {
    let mut raw = p1();
    raw["background"] = json!([0.0, 1.5, 0.0]);
    assert!(matches!(read(&raw), Err(CoreError::Value(_))));
}

#[test]
fn u24_a_degree_outside_0_to_3_is_refused() {
    let mut raw = p1();
    raw["sh_degree"] = json!(4);
    assert!(matches!(read(&raw), Err(CoreError::Sh(_))));
}

#[test]
fn u24_text_that_is_not_json_is_refused() {
    assert!(matches!(
        read_request(b"{\"format\": "),
        Err(CoreError::Json(_))
    ));
}

#[test]
fn u24_the_digest_is_of_the_bytes_given() {
    let bytes = std::fs::read(fixtures_dir().join("probes/P1.request.json")).unwrap();
    let mut spaced = bytes.clone();
    spaced.push(b'\n');
    let a = read_request(&bytes).unwrap();
    let b = read_request(&spaced).unwrap();
    assert_eq!(a.items, b.items);
    assert_ne!(a.sha256, b.sha256);
}
