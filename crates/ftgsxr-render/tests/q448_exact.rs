// SPDX-License-Identifier: Apache-2.0

//! Q-448's exactness check (#802): an exact change to the player's passes (§2.4 rule 2)
//! changes no pixel. The splat layers of R-1's, R-2's and R-9's scenes from the headless
//! renderer, per view and in shared-center mode in both of its record layouts, against
//! those recorded at the commit before the change: s = 0 and s\_α = 0 by E15's rule. The
//! headless renderer reads the layer back without the resolve (§1.4 stage 7), so this
//! covers stages 2–5; a change to the resolve or to the adapter's passes is checked where
//! it runs (Q-448).
//!
//! The recorded file keeps a fixed subset, so that it stays a few megabytes (Q-448): R-1's
//! and R-2's items 0, 11 and 22, rendered with the item's camera as both eyes in
//! shared-center mode (the first eye's layer kept, as Q-445's check does for R-12); R-9's
//! first two eye pairs; and R2\_random's three items at 64 × 48 (its intrinsics scaled by
//! 0.2), each camera turned 9° about y and 6° about x about its own center, with a second
//! eye 64 mm to its right. P1–P10 hold isotropic primitives and most of their cameras are
//! axis-aligned, so R2\_random's anisotropic rows under a turned view are what a wrong
//! regrouping of E6's Σ′ (Q-449) would change; its eye pairs put rows in one eye only.
//! The same turned pairs at `pixel_center_offset` 0, at `alpha_skip` 10⁻⁴ and at
//! `screen_extent_max` 2 carry the tightened quad's shifted rectangle and its fallback to
//! μ′ ± r (Q-450, #820).

use std::path::PathBuf;

use ftgsxr_core::{
    Camera, DMat3, DMat4, DVec3, Pose, RenderConventions, RenderImage, RenderRequest, read_request,
};
use ftgsxr_render::{HeadlessRenderer, QualityTier, SortMode};
use ftgsxr_testkit::{fixtures_dir, layer_messages, metrics, probe};

/// Q-448: the recorded layers (`exact_layers`), each with its id, its size, its first
/// pixel as the fill, then each pixel that differs from the fill bitwise, by index.
fn recorded_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/q448_exact.bin")
}

/// Q-448: set to record `exact_layers` into `recorded_path()` instead of comparing, in a
/// commit of its own at the commit before an exact change (Q-445's rule).
const RECORD_VAR: &str = "FTGSXR_RECORD_Q448_EXACT";

/// R-1's probes, which R-9's requests also render.
const R1_PROBES: [&str; 10] = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"];

/// R-2's probes rendered at their own size; R2_random goes through `turned_eyes`.
const R2_PROBES: [&str; 4] = [
    "R2_overlap_deg3",
    "R2_white_background",
    "R2_single_timestamp",
    "R2_epoch_time",
];

/// The items of a single-camera request kept in the file.
const ITEMS: [usize; 3] = [0, 11, 22];

/// R-9's eye pairs kept in the file.
const PAIRS: usize = 2;

/// Probe `p`'s R-9 request, `probes/R9_<p>.request.json` (Q-415).
fn r9_request(p: &str) -> RenderRequest {
    let path = fixtures_dir().join(format!("probes/R9_{p}.request.json"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    read_request(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// R2_random's item camera at 64 × 48, turned 9° about y and 6° about x about its own
/// center, and a second eye 64 mm to its right (along its own x axis).
pub(crate) fn turned_eyes(camera: &Camera) -> [Camera; 2] {
    let scale = 0.2;
    let k = DMat3::from_cols(
        camera.k.x_axis * scale,
        camera.k.y_axis * scale,
        DVec3::new(camera.k.z_axis.x * scale, camera.k.z_axis.y * scale, 1.0),
    );
    let turn = DMat4::from_mat3(
        DMat3::from_rotation_y(9f64.to_radians()) * DMat3::from_rotation_x(6f64.to_radians()),
    );
    let left = turn * camera.world_to_camera;
    let right = DMat4::from_translation(DVec3::new(-0.064, 0.0, 0.0)) * left;
    [left, right].map(|world_to_camera| Camera {
        width: 64,
        height: 48,
        k,
        world_to_camera,
    })
}

/// R2_random's turned pairs with the model's conventions as given, then at
/// `pixel_center_offset` 0, which shifts every kept ellipse by half a pixel, at
/// `alpha_skip` 10⁻⁴, where E6's cap binds from σ_eff ≈ 0.026, and at `screen_extent_max`
/// 2, where the cap binds for every row and the pixels it clips reach α = σ_eff·e⁻² (at
/// the default cap of 3.33 they lie at α ≤ σ_eff·e^(−5.5), within E15's 1/255): the
/// tightened quad's shifted rectangle and its fallback to μ′ ± r (Q-450, #820), recorded
/// before that change (Q-445's rule).
const TURNED_VARIANTS: [(&str, fn(&mut RenderConventions)); 4] = [
    ("", |_| {}),
    ("-offset0", |c| c.pixel_center_offset = 0.0),
    ("-skip1e-4", |c| c.alpha_skip = 1e-4),
    ("-cap2", |c| c.screen_extent_max = 2.0),
];

/// The conformance tier, with the shared-center records' Σ^S in half precision or not
/// (§1.12 `half_records`; the per-view render ignores the flag, Q-109).
fn tier(half_records: bool) -> QualityTier {
    QualityTier {
        half_records,
        ..QualityTier::conformance()
    }
}

/// The shared-center record layout's name in an id.
fn layout(half_records: bool) -> &'static str {
    if half_records {
        "shared-half"
    } else {
        "shared-full"
    }
}

/// The layers of the kept scenes, by id: each probe's kept items per view and, with the
/// item's camera as both eyes, in shared-center mode in both layouts; each R-9 pair's two
/// eyes, and each of R2_random's turned pairs, per view and in shared-center mode in both
/// layouts.
fn exact_layers() -> Vec<(String, RenderImage)> {
    let mut out = Vec::new();
    for half_records in [false, true] {
        let shared = layout(half_records);
        for p in R1_PROBES.iter().chain(&R2_PROBES) {
            let (model, request) = probe(p);
            let mut renderer = HeadlessRenderer::new(tier(half_records)).expect("an adapter");
            let id = renderer
                .load(&model, request.sh_degree)
                .expect("the model loads");
            let instances = [(id, Pose::IDENTITY)];
            for &i in ITEMS.iter().filter(|&&i| i < request.items.len()) {
                let item = &request.items[i];
                let bg = request.background;
                if !half_records {
                    let layer = renderer
                        .render(&instances, &item.camera, item.timestamp, bg)
                        .unwrap_or_else(|e| panic!("{p} {}: {e}", item.id));
                    out.push((format!("per-view/{p}/{}", item.id), layer));
                }
                let eyes: [Camera; 2] = [item.camera, item.camera];
                let [left, _] = renderer
                    .render_stereo(
                        &instances,
                        &eyes,
                        item.timestamp,
                        bg,
                        SortMode::SharedCenter,
                    )
                    .unwrap_or_else(|e| panic!("{p} {}: {e}", item.id));
                out.push((format!("{shared}/{p}/{}", item.id), left));
            }
        }
        for p in R1_PROBES {
            let (model, _) = probe(p);
            let request = r9_request(p);
            let mut renderer = HeadlessRenderer::new(tier(half_records)).expect("an adapter");
            let id = renderer
                .load(&model, request.sh_degree)
                .expect("the model loads");
            let instances = [(id, Pose::IDENTITY)];
            for pair in request.items.chunks(2).take(PAIRS) {
                let (l, r) = (&pair[0], &pair[1]);
                let eyes = [l.camera, r.camera];
                let mut modes = vec![(shared, SortMode::SharedCenter)];
                if !half_records {
                    modes.push(("per-view", SortMode::PerView));
                }
                for (mode, sort) in modes {
                    let layers = renderer
                        .render_stereo(&instances, &eyes, l.timestamp, request.background, sort)
                        .unwrap_or_else(|e| panic!("R9_{p} {}: {e}", l.id));
                    for (item, layer) in [l, r].into_iter().zip(layers) {
                        out.push((format!("{mode}/R9_{p}/{}", item.id), layer));
                    }
                }
            }
        }
        for (variant, conventions) in TURNED_VARIANTS {
            let (mut model, request) = probe("R2_random");
            conventions(&mut model.conventions);
            let mut renderer = HeadlessRenderer::new(tier(half_records)).expect("an adapter");
            let id = renderer
                .load(&model, request.sh_degree)
                .expect("the model loads");
            let instances = [(id, Pose::IDENTITY)];
            for item in &request.items {
                let eyes = turned_eyes(&item.camera);
                let mut modes = vec![(shared, SortMode::SharedCenter)];
                if !half_records {
                    modes.push(("per-view", SortMode::PerView));
                }
                for (mode, sort) in modes {
                    let layers = renderer
                        .render_stereo(&instances, &eyes, item.timestamp, request.background, sort)
                        .unwrap_or_else(|e| panic!("R2_random{variant} {}: {e}", item.id));
                    for (eye, layer) in ["L", "R"].into_iter().zip(layers) {
                        out.push((
                            format!("{mode}/R2_random-turned{variant}/{}{eye}", item.id),
                            layer,
                        ));
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// A pixel's four f32, the RGB then the alpha.
fn pixel(image: &RenderImage, i: usize) -> [f32; 4] {
    [
        image.rgb[3 * i],
        image.rgb[3 * i + 1],
        image.rgb[3 * i + 2],
        image.alpha[i],
    ]
}

/// The encoding of `recorded_path()`.
fn encode(layers: &[(String, RenderImage)]) -> Vec<u8> {
    let mut out = Vec::new();
    let put = |w: u32, out: &mut Vec<u8>| out.extend_from_slice(&w.to_le_bytes());
    put(layers.len() as u32, &mut out);
    for (id, image) in layers {
        put(id.len() as u32, &mut out);
        out.extend_from_slice(id.as_bytes());
        put(image.width, &mut out);
        put(image.height, &mut out);
        let n = (image.width * image.height) as usize;
        let fill = pixel(image, 0);
        for v in fill {
            put(v.to_bits(), &mut out);
        }
        let set: Vec<usize> = (0..n)
            .filter(|&i| {
                pixel(image, i)
                    .iter()
                    .zip(fill)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            })
            .collect();
        put(set.len() as u32, &mut out);
        for i in set {
            put(i as u32, &mut out);
            for v in pixel(image, i) {
                put(v.to_bits(), &mut out);
            }
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Vec<(String, RenderImage)> {
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| {
        let slice = &bytes[*at..*at + n];
        *at += n;
        slice
    };
    let word = |at: &mut usize| u32::from_le_bytes(take(at, 4).try_into().expect("four bytes"));
    let f = |at: &mut usize| f32::from_bits(word(at));
    let n = word(&mut at);
    let mut out = Vec::new();
    for _ in 0..n {
        let len = word(&mut at) as usize;
        let id = String::from_utf8(take(&mut at, len).to_vec()).expect("a UTF-8 id");
        let (width, height) = (word(&mut at), word(&mut at));
        let pixels = (width * height) as usize;
        let fill = [f(&mut at), f(&mut at), f(&mut at), f(&mut at)];
        let mut rgb: Vec<f32> = (0..pixels)
            .flat_map(|_| [fill[0], fill[1], fill[2]])
            .collect();
        let mut alpha = vec![fill[3]; pixels];
        for _ in 0..word(&mut at) {
            let i = word(&mut at) as usize;
            for c in 0..3 {
                rgb[3 * i + c] = f(&mut at);
            }
            alpha[i] = f(&mut at);
        }
        out.push((
            id,
            RenderImage {
                width,
                height,
                rgb,
                alpha,
            },
        ));
    }
    assert_eq!(at, bytes.len(), "the recorded file ends after its layers");
    out
}

/// Q-448 (#802): an exact change to stages 2–5 changes no pixel of the kept scenes of
/// R-1, R-2 and R-9, per view and shared-center in both record layouts, against the layers
/// recorded at the commit before the change: s = 0 and s_α = 0 by E15's rule. With
/// `FTGSXR_RECORD_Q448_EXACT` set it records them instead.
#[test]
#[ignore = "gpu"]
fn gpu_q448_exact_change_changes_no_pixel_of_the_recorded_layers() {
    let _layer = layer_messages(); // R-16, Q-105
    let layers = exact_layers();
    if std::env::var_os(RECORD_VAR).is_some_and(|v| !v.is_empty()) {
        std::fs::write(recorded_path(), encode(&layers)).expect("the recorded layers written");
        return;
    }
    let bytes = std::fs::read(recorded_path()).expect("the recorded layers (Q-448)");
    let recorded = decode(&bytes);
    let ids = |l: &[(String, RenderImage)]| l.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&layers), ids(&recorded), "the same scenes");
    let drawn = layers
        .iter()
        .filter(|(_, l)| l.alpha.iter().any(|&a| a > 0.0))
        .count();
    assert!(drawn * 2 > layers.len(), "most layers draw something");
    let failures: Vec<String> = layers
        .iter()
        .zip(&recorded)
        .filter_map(|((id, now), (_, then))| {
            let m = metrics(then, now);
            (m.share != 0.0 || m.alpha != 0.0)
                .then(|| format!("{id}: s = {}, s_α = {}", m.share, m.alpha))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} layers differ from the recorded ones:\n{}",
        failures.len(),
        layers.len(),
        failures.join("\n")
    );
}
