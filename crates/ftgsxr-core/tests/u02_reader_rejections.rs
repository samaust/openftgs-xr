// SPDX-License-Identifier: Apache-2.0

//! U-2, the reader's part (§1.2 reader rules, §1.15 codes, `cpu` lane): one synthesized
//! `ftgs-if` file per rule, each rejected by `read_exchange` with its E-FILE code; version
//! `1.3` accepted. The files are written by the testkit's builder from a valid random model,
//! with one entry changed.

use ftgsxr_core::{Model, read_exchange};
use ftgsxr_testkit::{SafetensorsBuilder, exchange_bytes, random_model};

/// One tensor entry: name, dtype, shape, bytes.
type Tensor = (String, String, Vec<usize>, Vec<u8>);

/// A valid model's metadata and tensors, as `exchange_bytes` writes them.
struct Parts {
    meta: Vec<(String, String)>,
    tensors: Vec<Tensor>,
}

fn f32_bytes(xs: impl IntoIterator<Item = f32>) -> Vec<u8> {
    xs.into_iter().flat_map(f32::to_le_bytes).collect()
}

fn f64_bytes(xs: impl IntoIterator<Item = f64>) -> Vec<u8> {
    xs.into_iter().flat_map(f64::to_le_bytes).collect()
}

fn parts(m: &Model) -> Parts {
    let n = m.len();
    let k = (usize::from(m.sh_degree) + 1).pow(2);
    let t = |name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>| {
        (name.to_owned(), dtype.to_owned(), shape.to_vec(), bytes)
    };
    let meta = [
        ("format", "ftgs-if".to_owned()),
        ("version", "1".to_owned()),
        ("sh_degree", m.sh_degree.to_string()),
        ("time_unit", m.time.time_unit.clone()),
        ("tau0", format!("{:?}", m.time.tau0)),
        ("tau1", format!("{:?}", m.time.tau1)),
        ("producer", m.producer.clone()),
        ("render_conventions", m.conventions.to_json()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    let tensors = vec![
        t(
            "position",
            "F32",
            &[n, 3],
            f32_bytes(m.position.iter().flatten().copied()),
        ),
        t(
            "time_center",
            "F64",
            &[n],
            f64_bytes(m.time_center.iter().copied()),
        ),
        t(
            "duration",
            "F64",
            &[n],
            f64_bytes(m.duration.iter().copied()),
        ),
        t(
            "velocity",
            "F32",
            &[n, 3],
            f32_bytes(m.velocity.iter().flatten().copied()),
        ),
        t(
            "scale",
            "F32",
            &[n, 3],
            f32_bytes(m.scale.iter().flatten().copied()),
        ),
        t(
            "rotation",
            "F32",
            &[n, 4],
            f32_bytes(m.rotation.iter().flatten().copied()),
        ),
        t("opacity", "F32", &[n], f32_bytes(m.opacity.iter().copied())),
        t("sh", "F32", &[n, k, 3], f32_bytes(m.sh.iter().copied())),
    ];
    Parts { meta, tensors }
}

impl Parts {
    fn set_meta(mut self, key: &str, value: &str) -> Self {
        match self.meta.iter_mut().find(|(k, _)| k == key) {
            Some(entry) => entry.1 = value.to_owned(),
            None => self.meta.push((key.to_owned(), value.to_owned())),
        }
        self
    }

    fn drop_meta(mut self, key: &str) -> Self {
        self.meta.retain(|(k, _)| k != key);
        self
    }

    fn drop_tensor(mut self, key: &str) -> Self {
        self.tensors.retain(|t| t.0 != key);
        self
    }

    fn set_tensor(mut self, key: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>) -> Self {
        let t = self.tensors.iter_mut().find(|t| t.0 == key).unwrap();
        *t = (key.to_owned(), dtype.to_owned(), shape.to_vec(), bytes);
        self
    }

    fn build(self) -> Vec<u8> {
        let mut b = SafetensorsBuilder::new();
        for (k, v) in &self.meta {
            b = b.metadata(k, v);
        }
        for (name, dtype, shape, bytes) in self.tensors {
            b = b.tensor(&name, &dtype, &shape, bytes);
        }
        b.build()
    }
}

const N: usize = 8;

fn base() -> Model {
    random_model(20261029, N, 1)
}

/// The E-FILE code `read_exchange` refuses the file with.
fn code(bytes: &[u8]) -> &'static str {
    match read_exchange(bytes) {
        Ok(_) => panic!("the file was accepted"),
        Err(e) => e.code(),
    }
}

/// The code for the base model changed by `edit`, written by `exchange_bytes`.
fn code_of_model(edit: impl FnOnce(&mut Model)) -> &'static str {
    let mut m = base();
    edit(&mut m);
    code(&exchange_bytes(&m))
}

#[test]
fn u02_the_base_file_loads() {
    let m = base();
    assert_eq!(read_exchange(&parts(&m).build()).unwrap(), {
        let mut expected = m.clone();
        expected.identity = ftgsxr_core::sha256(&parts(&m).build());
        expected
    });
}

#[test]
fn u02_another_format() {
    let bytes = parts(&base()).set_meta("format", "ftgs-checkpoint").build();
    assert_eq!(code(&bytes), "E-FILE-FORMAT");
}

#[test]
fn u02_major_version_2() {
    let bytes = parts(&base()).set_meta("version", "2").build();
    assert_eq!(code(&bytes), "E-FILE-VERSION");
}

#[test]
fn u02_version_1_3_loads() {
    let m = base();
    let bytes = parts(&m).set_meta("version", "1.3").build();
    let mut read = read_exchange(&bytes).unwrap();
    read.identity = m.identity;
    assert_eq!(read, m);
}

#[test]
fn u02_missing_keys() {
    for key in [
        "format",
        "version",
        "sh_degree",
        "tau0",
        "tau1",
        "time_unit",
    ] {
        let bytes = parts(&base()).drop_meta(key).build();
        assert_eq!(code(&bytes), "E-FILE-KEYS", "metadata {key}");
    }
    for key in [
        "position",
        "time_center",
        "duration",
        "velocity",
        "scale",
        "rotation",
        "opacity",
        "sh",
    ] {
        let bytes = parts(&base()).drop_tensor(key).build();
        assert_eq!(code(&bytes), "E-FILE-KEYS", "tensor {key}");
    }
}

#[test]
fn u02_wrong_dtype() {
    let m = base();
    let bytes = parts(&m)
        .set_tensor(
            "time_center",
            "F32",
            &[N],
            f32_bytes(m.time_center.iter().map(|&t| t as f32)),
        )
        .build();
    assert_eq!(code(&bytes), "E-FILE-SHAPE");
    let bytes = parts(&m)
        .set_tensor(
            "opacity",
            "F64",
            &[N],
            f64_bytes(m.opacity.iter().map(|&o| f64::from(o))),
        )
        .build();
    assert_eq!(code(&bytes), "E-FILE-SHAPE");
}

#[test]
fn u02_wrong_shape() {
    let m = base();
    let scale = f32_bytes(m.scale.iter().flatten().copied());
    let bytes = parts(&m)
        .set_tensor("scale", "F32", &[N * 3], scale)
        .build();
    assert_eq!(code(&bytes), "E-FILE-SHAPE");
    // Another N.
    let bytes = parts(&m)
        .set_tensor(
            "opacity",
            "F32",
            &[N - 1],
            f32_bytes(m.opacity[1..].iter().copied()),
        )
        .build();
    assert_eq!(code(&bytes), "E-FILE-SHAPE");
    // N = 0.
    let bytes = parts(&m)
        .set_tensor("position", "F32", &[0, 3], vec![])
        .build();
    assert_eq!(code(&bytes), "E-FILE-SHAPE");
}

#[test]
fn u02_sh_degree_rules() {
    assert_eq!(
        code(&parts(&base()).set_meta("sh_degree", "4").build()),
        "E-FILE-SH"
    );
    // A degree that disagrees with the `sh` shape.
    assert_eq!(
        code(&parts(&base()).set_meta("sh_degree", "2").build()),
        "E-FILE-SH"
    );
}

#[test]
fn u02_metadata_numbers_that_do_not_parse() {
    for (key, value) in [("tau0", "zero"), ("tau1", ""), ("sh_degree", "1.5")] {
        let bytes = parts(&base()).set_meta(key, value).build();
        assert_eq!(code(&bytes), "E-FILE-VALUE", "{key} = {value:?}");
    }
}

#[test]
fn u02_nan() {
    assert_eq!(
        code_of_model(|m| m.position[3][1] = f32::NAN),
        "E-FILE-VALUE"
    );
    assert_eq!(code_of_model(|m| m.sh[5] = f32::NAN), "E-FILE-VALUE");
    assert_eq!(
        code_of_model(|m| m.time_center[0] = f64::INFINITY),
        "E-FILE-VALUE"
    );
}

#[test]
fn u02_duration_not_positive() {
    assert_eq!(code_of_model(|m| m.duration[2] = 0.0), "E-FILE-VALUE");
    assert_eq!(code_of_model(|m| m.duration[2] = -1.0), "E-FILE-VALUE");
}

#[test]
fn u02_scale_not_positive() {
    assert_eq!(code_of_model(|m| m.scale[1][2] = 0.0), "E-FILE-VALUE");
    assert_eq!(code_of_model(|m| m.scale[1][0] = -0.1), "E-FILE-VALUE");
}

#[test]
fn u02_rotation_of_norm_1e_9() {
    assert_eq!(
        code_of_model(|m| m.rotation[4] = [1e-9, 0.0, 0.0, 0.0]),
        "E-FILE-VALUE"
    );
}

#[test]
fn u02_opacity_0_and_1() {
    assert_eq!(code_of_model(|m| m.opacity[0] = 0.0), "E-FILE-VALUE");
    assert_eq!(code_of_model(|m| m.opacity[0] = 1.0), "E-FILE-VALUE");
}

#[test]
fn u02_tau1_below_tau0() {
    let bytes = parts(&base())
        .set_meta("tau0", "2.0")
        .set_meta("tau1", "1.0")
        .build();
    assert_eq!(code(&bytes), "E-FILE-VALUE");
}

#[test]
fn u02_single_timestamp_with_a_velocity() {
    let edit = |m: &mut Model| {
        m.time.tau0 = 0.5;
        m.time.tau1 = 0.5;
        m.time_center.iter_mut().for_each(|t| *t = 0.5);
        m.velocity.iter_mut().for_each(|v| *v = [0.0; 3]);
    };
    // Static as it should be: loads.
    let mut m = base();
    edit(&mut m);
    assert!(read_exchange(&exchange_bytes(&m)).is_ok());
    assert_eq!(
        code_of_model(|m| {
            edit(m);
            m.velocity[6] = [0.0, 0.01, 0.0];
        }),
        "E-FILE-SINGLE"
    );
}

#[test]
fn u02_render_conventions_not_an_object() {
    for block in ["[0.999, 0.3]", "\"classic\"", "{not json"] {
        let bytes = parts(&base()).set_meta("render_conventions", block).build();
        assert_eq!(code(&bytes), "E-FILE-VALUE", "{block}");
    }
}

#[test]
fn u02_background_1_5() {
    let bytes = parts(&base())
        .set_meta("render_conventions", "{\"background\":[1.5,0,0]}")
        .build();
    assert_eq!(code(&bytes), "E-FILE-VALUE");
}

#[test]
fn u02_rules_reported_in_order() {
    // A wrong format beside a NaN: the format first.
    let mut m = base();
    m.position[0][0] = f32::NAN;
    let bytes = parts(&m).set_meta("format", "other").build();
    assert_eq!(code(&bytes), "E-FILE-FORMAT");
    // A bad conventions block beside a bad opacity: the arrays' rule first.
    let mut m = base();
    m.opacity[0] = 1.0;
    let bytes = parts(&m).set_meta("render_conventions", "[]").build();
    assert_eq!(code(&bytes), "E-FILE-VALUE");
    let mut m = base();
    m.time.tau0 = 0.5;
    m.time.tau1 = 0.5;
    m.time_center.iter_mut().for_each(|t| *t = 0.5);
    let bytes = parts(&m).set_meta("render_conventions", "[]").build();
    assert_eq!(code(&bytes), "E-FILE-SINGLE");
}

#[test]
fn u02_not_a_safetensors_file() {
    assert_eq!(code(b""), "E-FILE-FORMAT");
    assert_eq!(code(b"ply\nformat ascii 1.0\n"), "E-FILE-FORMAT");
    let mut bytes = parts(&base()).build();
    bytes.pop();
    assert_eq!(code(&bytes), "E-FILE-FORMAT");
}
