// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr bench NAME [--out BENCH.json] [--model MODEL]` (§1.7, §3.5): the desktop
//! benchmarks B-1, B-3, B-5 and B-6, their JSON file and the comparison with the stored
//! baseline (Q-67, Q-102, Q-103).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bevy::color::LinearRgba;
use bevy::render::render_resource::{
    CommandEncoderDescriptor, Extent3d, LoadOp, Operations, PollType, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureViewDescriptor, WgpuTextureView,
};
use ftgsxr_core::{
    Camera, DMat3, DVec3, Fov, Mat3, Model, Pose, Quat, TemporalIndex, UVec2, Vec3,
    encode_exchange, hex, pack_for_gpu, write_exchange,
};
use ftgsxr_render::{
    Capacity, ColorTransform, FrameBuffers, FrameStats, GpuModel, ModelDraw, Overlays, QualityTier,
    RenderError, RenderMode, SortPose, SplatDevice, SplatPasses, ViewParams,
};

use super::json::Json;
use super::synthetic::{Persistence, synthetic_model};
use super::{HeadlessError, core_input, identity_file, input_error, load, write_atomic};

/// What `cmd_bench` measured (Q-67): the file's text and the budgets that regressed
/// against the stored baseline, one line each; the binary exits 3 when there is any.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchReport {
    /// The benchmark file's JSON text (Q-103), as written at `out`.
    pub json: String,
    /// One line per budgeted figure more than 10 % worse than the baseline's (§3.5).
    pub regressions: Vec<String>,
}

/// How a budget's value meets its limit (Q-103).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rule {
    /// value ≥ limit; higher is better.
    AtLeast,
    /// value ≤ limit; lower is better.
    AtMost,
    /// |value / limit − 1| ≤ 1 %; lower is better.
    Within1Percent,
}

impl Rule {
    fn name(self) -> &'static str {
        match self {
            Rule::AtLeast => "at_least",
            Rule::AtMost => "at_most",
            Rule::Within1Percent => "within_1_percent",
        }
    }

    fn passes(self, value: f64, limit: f64) -> bool {
        match self {
            Rule::AtLeast => value >= limit,
            Rule::AtMost => value <= limit,
            Rule::Within1Percent => (value / limit - 1.0).abs() <= 0.01,
        }
    }

    /// Whether `value` is more than 10 % worse than `baseline` (§3.5). A
    /// `within_1_percent` budget (B-6) never is: it is deterministic, and its own 1 %
    /// budget is the check (Q-103).
    fn regressed(self, value: f64, baseline: f64) -> bool {
        match self {
            Rule::AtLeast => value < 0.9 * baseline,
            Rule::AtMost => value > 1.1 * baseline,
            Rule::Within1Percent => false,
        }
    }
}

/// One measured figure: `{setup, measure, value, unit}`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Figure {
    pub(crate) setup: Json,
    pub(crate) measure: &'static str,
    pub(crate) value: f64,
    pub(crate) unit: &'static str,
}

/// One budgeted figure: `{id, setup, measure, rule, limit, value, pass}`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Budget {
    pub(crate) id: &'static str,
    pub(crate) setup: Json,
    pub(crate) measure: &'static str,
    pub(crate) rule: Rule,
    pub(crate) limit: f64,
    pub(crate) value: f64,
}

impl Budget {
    fn pass(&self) -> bool {
        self.rule.passes(self.value, self.limit)
    }

    fn to_json(&self) -> Json {
        Json::obj([
            ("id", Json::str(self.id)),
            ("setup", self.setup.clone()),
            ("measure", Json::str(self.measure)),
            ("rule", Json::str(self.rule.name())),
            ("limit", Json::Num(self.limit)),
            ("value", Json::Num(self.value)),
            ("pass", Json::Bool(self.pass())),
        ])
    }
}

/// The budgets of `budgets` more than 10 % worse than the baseline file's budget with the
/// same id, setup and measure, one line each (§3.5, Q-103). A budget the baseline
/// lacks is not compared, nor is a `within_1_percent` budget (B-6), whose own budget is
/// the check.
pub(crate) fn compare(budgets: &[Budget], baseline: &Json) -> Vec<String> {
    let old = baseline
        .get("budgets")
        .and_then(Json::as_array)
        .unwrap_or(&[]);
    budgets
        .iter()
        .filter_map(|b| {
            let base = old
                .iter()
                .find(|o| {
                    o.get("id").and_then(Json::as_str) == Some(b.id)
                        && o.get("measure").and_then(Json::as_str) == Some(b.measure)
                        && o.get("setup").is_some_and(|s| s.same(&b.setup))
                })?
                .get("value")?
                .as_f64()?;
            b.rule.regressed(b.value, base).then(|| {
                format!(
                    "{} {} {}: {} against the baseline's {} ({:+.1} %)",
                    b.id,
                    b.measure,
                    b.setup.compact(),
                    b.value,
                    base,
                    100.0 * (b.value / base - 1.0)
                )
            })
        })
        .collect()
}

/// `compare` against a baseline of the run's `profile` and `machine.gpu` only (Q-103):
/// a baseline with another is not compared, and the error is the note saying which
/// field differs.
// Q-103
pub(crate) fn compare_against(
    budgets: &[Budget],
    profile: &str,
    gpu: &str,
    baseline: &Json,
) -> Result<Vec<String>, String> {
    let fields = [
        ("profile", profile, baseline.get("profile")),
        (
            "machine.gpu",
            gpu,
            baseline.get("machine").and_then(|m| m.get("gpu")),
        ),
    ];
    let differ: Vec<String> = fields
        .iter()
        .filter_map(|&(field, ours, theirs)| {
            let theirs = theirs.and_then(Json::as_str);
            (theirs != Some(ours))
                .then(|| format!("{field} {ours} against {}", theirs.unwrap_or("none")))
        })
        .collect();
    if differ.is_empty() {
        Ok(compare(budgets, baseline))
    } else {
        Err(format!(
            "baseline not compared: {} (Q-103)",
            differ.join(", ")
        ))
    }
}

/// `ftgsxr bench NAME [--out BENCH.json] [--model MODEL]` (§1.7, §3.5, Q-67): runs the
/// desktop benchmark `name` (B-1, B-3, B-5 or B-6) on the high-performance adapter, with
/// the setups of Q-102 or, with `model`, on that one model (any §1.2 input); compares
/// its budgets with the stored baseline `<NAME>.json` when the baseline has the run's
/// profile and GPU, else notes why not, in the file and on standard error (Q-103); and
/// writes the file at `out` when given. A NAME the desktop does not run is `Input`; an
/// invalid model is `Input` with its E-FILE code; no adapter, or no timestamp queries for
/// B-3, is `Environment`; a malformed baseline is `Input` (E-JSON).
// Q-67, Q-102, Q-103
pub fn cmd_bench(
    name: &str,
    out: Option<&Path>,
    model: Option<&Path>,
) -> Result<BenchReport, HeadlessError> {
    let run: fn(&Gpu, Option<&Given>) -> Result<Measured, HeadlessError> = match name {
        "B-1" => bench_b1,
        "B-3" => bench_b3,
        "B-5" => bench_b5,
        "B-6" => bench_b6,
        _ => {
            return Err(HeadlessError::Input(format!(
                "{name}: not a desktop benchmark (B-1, B-3, B-5, B-6)"
            )));
        }
    };
    let given = model
        .map(|path| {
            load(path).map(|model| Given {
                path: path.to_path_buf(),
                model,
            })
        })
        .transpose()?;
    let gpu = Gpu::new()?;
    let measured = run(&gpu, given.as_ref());
    let messages = gpu.dev.take_validation_messages();
    let measured = measured?;
    if let Some(first) = messages.first() {
        return Err(HeadlessError::Environment(format!(
            "E-GPU-SHADER: {} GPU validation messages, the first: {first}",
            messages.len()
        )));
    }

    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let mut notes: Vec<Json> = measured.notes.iter().map(|n| Json::str(*n)).collect();
    let baseline_path = baseline_dir().join(format!("{name}.json"));
    let (baseline, regressions) = if baseline_path.is_file() {
        let text =
            fs::read_to_string(&baseline_path).map_err(|e| input_error(&baseline_path, e))?;
        let parsed = Json::parse(&text).map_err(|e| {
            HeadlessError::Input(format!("E-JSON: {}: {e}", baseline_path.display()))
        })?;
        let path = baseline_path.canonicalize().unwrap_or(baseline_path);
        // A baseline not compared against is not `baseline` (Q-103: "the path compared
        // against, or null").
        match compare_against(&measured.budgets, profile, gpu.dev.adapter_name(), &parsed) {
            Ok(regressions) => (Json::str(path.display().to_string()), regressions),
            Err(note) => {
                eprintln!("ftgsxr: {name}: {note}");
                notes.push(Json::str(note));
                (Json::Null, Vec::new())
            }
        }
    } else {
        (Json::Null, Vec::new())
    };

    let file = Json::obj([
        ("format", Json::str("ftgsxr-bench")),
        ("version", Json::Num(1.0)),
        ("bench", Json::str(name)),
        ("machine", machine(gpu.dev.adapter_name())),
        (
            "versions",
            Json::obj([
                ("ftgsxr", Json::str(env!("CARGO_PKG_VERSION"))),
                ("bevy", Json::str(BEVY_VERSION)),
                ("wgpu", Json::str(WGPU_VERSION)),
            ]),
        ),
        ("profile", Json::str(profile)),
        (
            "results",
            Json::Arr(
                measured
                    .figures
                    .iter()
                    .map(|f| {
                        Json::obj([
                            ("setup", f.setup.clone()),
                            ("measure", Json::str(f.measure)),
                            ("value", Json::Num(f.value)),
                            ("unit", Json::str(f.unit)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "budgets",
            Json::Arr(measured.budgets.iter().map(Budget::to_json).collect()),
        ),
        ("baseline", baseline),
        ("notes", Json::Arr(notes)),
    ]);
    let mut json = file.pretty();
    json.push('\n');
    if let Some(out) = out {
        write_atomic(out, json.as_bytes()).map_err(|e| input_error(out, e))?;
    }
    Ok(BenchReport { json, regressions })
}

/// The pinned Bevy (§2.3, the workspace's `[workspace.dependencies]`), which no runtime
/// API reports; `versions_match_the_lock_file` keeps it true.
const BEVY_VERSION: &str = "0.19.1";
/// The wgpu in the build (Bevy's pin, Q-73), as `BEVY_VERSION`.
const WGPU_VERSION: &str = "29.0.4";

/// The variable that overrides the baselines' directory (Q-103).
const BASELINES_VAR: &str = "FTGSXR_BENCH_BASELINES";

/// The directory of the stored baselines (Q-103): `$FTGSXR_BENCH_BASELINES` when set,
/// else the workspace's `benchmarks/baselines` as the build saw it.
// Q-103
fn baseline_dir() -> PathBuf {
    match std::env::var_os(BASELINES_VAR) {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/baselines"),
    }
}

/// The machine fields (§3.5, Q-103).
fn machine(gpu: &str) -> Json {
    let first_line = |path: &str| {
        fs::read_to_string(path)
            .ok()
            .and_then(|t| t.lines().next().map(|l| l.trim().to_owned()))
    };
    let driver = first_line("/proc/driver/nvidia/version").unwrap_or_else(|| "unknown".into());
    let os_name = fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|t| {
            t.lines()
                .find_map(|l| l.strip_prefix("PRETTY_NAME="))
                .map(|v| v.trim_matches('"').to_owned())
        })
        .unwrap_or_else(|| std::env::consts::OS.to_owned());
    let os = match first_line("/proc/sys/kernel/osrelease") {
        Some(kernel) => format!("{os_name} (kernel {kernel})"),
        None => os_name,
    };
    let cpu = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_owned())
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    Json::obj([
        ("gpu", Json::str(gpu)),
        ("driver", Json::str(driver)),
        ("os", Json::str(os)),
        ("cpu", Json::str(cpu)),
        ("threads", Json::Num(threads as f64)),
    ])
}

/// The model `--model` names, read and validated before the GPU opens.
struct Given {
    path: PathBuf,
    model: Model,
}

impl Given {
    /// The `model` value of a setup: the model's identity.
    fn label(&self) -> Json {
        Json::str(hex(&self.model.identity))
    }
}

/// A benchmark's figures, budgets and notes.
#[derive(Default)]
struct Measured {
    figures: Vec<Figure>,
    budgets: Vec<Budget>,
    notes: Vec<&'static str>,
}

impl Measured {
    fn figure(&mut self, setup: &Json, measure: &'static str, value: f64, unit: &'static str) {
        self.figures.push(Figure {
            setup: setup.clone(),
            measure,
            value,
            unit,
        });
    }
}

/// The view's size (§3.5 B-1: 1920 × 1080).
const WIDTH: u32 = 1920;
/// As `WIDTH`.
const HEIGHT: u32 = 1080;
/// The desktop's default vertical field of view (§1.7).
const FOV_DEG: f32 = 60.0;
/// The orbit's cameras (Q-102).
const ORBIT: usize = 72;
/// Frames rendered before any is measured (Q-102).
const WARM_UP: usize = 8;
/// The layer's format (§1.4 stage 5, D-35).
const LAYER_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// The view target's format (Q-91).
const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

/// The device and the desktop tier's passes.
struct Gpu {
    dev: SplatDevice,
    tier: QualityTier,
    passes: SplatPasses,
}

impl Gpu {
    fn new() -> Result<Gpu, HeadlessError> {
        let dev = SplatDevice::headless().map_err(|e| gpu_error(None, e))?;
        let tier = QualityTier::desktop();
        let passes =
            SplatPasses::new(&dev, LAYER_FORMAT, None, &tier).map_err(|e| gpu_error(None, e))?;
        Ok(Gpu { dev, tier, passes })
    }

    /// `model` packed with `layout`'s options and uploaded (§1.2 load options), with its
    /// loaded degree; `file` is the file it was read from, if any, which a core error names.
    fn upload(
        &self,
        model: &Model,
        tier: &QualityTier,
        file: Option<&Path>,
    ) -> Result<(GpuModel, u8), HeadlessError> {
        let (index, order) = TemporalIndex::build(model);
        let arrays = pack_for_gpu(model, &order, &tier.layout);
        let degree = arrays.sh_degree;
        let gpu_model = GpuModel::upload(&self.dev, &arrays, index, &model.conventions)
            .map_err(|e| gpu_error(file, e))?;
        Ok((gpu_model, degree))
    }

    /// Waits until the queue's work is done.
    fn wait(&self) -> Result<(), HeadlessError> {
        self.dev
            .device()
            .poll(PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| HeadlessError::Environment(format!("E-GPU-LOST: {e}")))
    }

    /// The capacity C of the desktop tier's frame buffers for `n` rows.
    fn capacity(&self, n: u32) -> u32 {
        match self.tier.capacity {
            Capacity::Fixed(c) => c,
            Capacity::SumOfModels => n,
        }
    }
}

/// A render-layer error as a `HeadlessError` (Q-90's mapping): a core error as `Input`,
/// with the path of the `file` the model was read from if there is one (Q-75), a
/// non-finite render as `Numeric`, the GPU's conditions as `Environment` with their §1.15
/// code.
// Q-75, Q-90
fn gpu_error(file: Option<&Path>, e: RenderError) -> HeadlessError {
    match e {
        RenderError::Core(core) => match file {
            Some(path) => core_input(path, &core),
            None => HeadlessError::Input(format!("{}: {}", core.code(), core.user_text())),
        },
        RenderError::NonFinite => HeadlessError::Numeric(String::new()),
        other => HeadlessError::Environment(format!("{}: {}", other.code(), other.user_text())),
    }
}

/// The layer and the view target of one 1920 × 1080 view, kept alive with their views.
struct Targets {
    _textures: Vec<Box<dyn std::any::Any>>,
    layer: WgpuTextureView,
    target: WgpuTextureView,
}

impl Targets {
    fn new(gpu: &Gpu) -> Targets {
        let make = |format, usage| {
            let texture = gpu.dev.device().create_texture(&TextureDescriptor {
                label: Some("ftgsxr bench"),
                size: Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            (Box::new(texture) as Box<dyn std::any::Any>, view)
        };
        let (layer_texture, layer) = make(
            LAYER_FORMAT,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        );
        let (target_texture, target) = make(TARGET_FORMAT, TextureUsages::RENDER_ATTACHMENT);
        Targets {
            _textures: vec![layer_texture, target_texture],
            layer,
            target,
        }
    }
}

/// One desktop frame (Q-102): stages 1–4 (`encode_prepare`), the splat draw into the
/// layer cleared to (0, 0, 0, 0) with its timestamps (Q-114), the resolve onto the target
/// over black and the statistics, submitted and waited for. Returns the frame's wall time in milliseconds,
/// from encoding to completion, and the statistics `FrameStats::poll` has ready.
fn frame(
    gpu: &Gpu,
    frame: &FrameBuffers,
    draws: &[ModelDraw],
    camera: &Camera,
    targets: &Targets,
) -> Result<(f64, Option<FrameStats>), HeadlessError> {
    let device = gpu.dev.device();
    let view = ViewParams {
        camera: *camera,
        target_size: UVec2::new(WIDTH, HEIGHT),
    };
    let start = Instant::now();
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("ftgsxr bench frame"),
    });
    gpu.passes
        .encode_prepare(&mut enc, frame, draws, &sort_pose(camera), &[view]);
    {
        let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
            label: Some("ftgsxr bench splats"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &targets.layer,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(LinearRgba::NONE.into()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            // Q-114
            timestamp_writes: frame.draw_timestamp_writes(),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        gpu.passes.encode_draw(&mut pass, frame, &view, None);
    }
    {
        let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
            label: Some("ftgsxr bench resolve"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &targets.target,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(LinearRgba::BLACK.into()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        gpu.passes
            .encode_resolve(&mut pass, &targets.layer, Some([0.0; 3]));
    }
    gpu.passes.encode_stats(&mut enc, frame);
    gpu.dev.queue().submit([enc.finish()]);
    gpu.wait()?;
    let ms = start.elapsed().as_secs_f64() * 1e3;
    Ok((ms, FrameStats::poll(frame, &gpu.dev)))
}

/// The camera's center and forward axis R_wᵀ ẑ (E4's eye; per-view mode).
fn sort_pose(camera: &Camera) -> SortPose {
    let r = DMat3::from_mat4(camera.world_to_camera);
    SortPose {
        origin: camera.center().as_vec3(),
        forward: (r.transpose() * DVec3::Z).as_vec3(),
    }
}

/// Q-102's orbit about `model`: `ORBIT` cameras at 5° steps on a horizontal circle
/// about the box center of its positions, of radius twice the box's half-diagonal, a
/// quarter of the radius above it, looking at the center; camera k at
/// τ₀ + k/(ORBIT − 1)·(τ₁ − τ₀).
// Q-102
fn orbit(model: &Model) -> Vec<(Camera, f64)> {
    let (lo, hi) = model.position.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), p| {
            let p = Vec3::from_array(*p);
            (lo.min(p), hi.max(p))
        },
    );
    let center = (lo + hi) / 2.0;
    let radius = ((hi - lo).length()).max(1e-3);
    let (tau0, tau1) = (model.time.tau0, model.time.tau1);
    (0..ORBIT)
        .map(|k| {
            let angle = (k as f32 * 5.0).to_radians();
            let eye = center + Vec3::new(radius * angle.cos(), 0.25 * radius, radius * angle.sin());
            let camera = Camera::from_stage_pose(
                eye,
                look_at(center - eye),
                Fov::Vertical {
                    radians: FOV_DEG.to_radians(),
                },
                WIDTH,
                HEIGHT,
            );
            let f = k as f64 / (ORBIT - 1) as f64;
            let tau = tau0 * (1.0 - f) + tau1 * f;
            (camera, tau)
        })
        .collect()
}

/// The stage-space rotation of a camera (looking along −z with y up) that looks along
/// `forward` with y up.
fn look_at(forward: Vec3) -> Quat {
    let back = -forward.normalize();
    let right = Vec3::Y.cross(back).normalize();
    let up = back.cross(right);
    Quat::from_mat3(&Mat3::from_cols(right, up, back))
}

/// The median of `values` (sorted in place); 0 when empty.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// One instance at the identity pose drawn as splats at `tau`.
fn draw(model: &GpuModel, sh_degree: u8, tau: f64) -> ModelDraw<'_> {
    ModelDraw {
        model,
        pose: Pose::IDENTITY,
        tau,
        mode: RenderMode::Splats,
        transform: ColorTransform::IDENTITY,
        sh_degree,
        overlays: Overlays::default(),
    }
}

/// B-1's and B-3's synthetic model (seed 1) of `n` rows at `degree`, every row
/// persistent, so every row is active at every τ (E3), with its setup label (Q-102).
// Q-102
fn persistent_synthetic(n: usize, degree: u8) -> (Model, Json) {
    (
        synthetic_model(1, n, degree, Persistence::All),
        Json::str("synthetic, every row persistent"),
    )
}

/// B-1's synthetic sizes (§3.5).
const B1_N: [usize; 5] = [250_000, 500_000, 1_000_000, 2_000_000, 3_000_000];

/// B-1 (§3.5, Q-102): desktop frames per second along the orbit; P-2.
fn bench_b1(gpu: &Gpu, given: Option<&Given>) -> Result<Measured, HeadlessError> {
    let mut m = Measured {
        notes: vec![FRAME_NOTE, GPU_TIMES_NOTE, SYNTHETIC_LOAD_NOTE],
        ..Measured::default()
    };
    let cases: Vec<(usize, u8)> = match given {
        Some(g) => vec![(g.model.len(), g.model.sh_degree)],
        None => B1_N
            .iter()
            .flat_map(|&n| (0..=3).map(move |l| (n, l)))
            .collect(),
    };
    let targets = Targets::new(gpu);
    for (n, degree) in cases {
        let synthetic;
        let (model, label) = match given {
            Some(g) => (&g.model, g.label()),
            None => {
                let setup_label;
                (synthetic, setup_label) = persistent_synthetic(n, degree);
                (&synthetic, setup_label)
            }
        };
        let (gpu_model, loaded) = gpu.upload(model, &gpu.tier, given.map(|g| g.path.as_path()))?;
        let frames = FrameBuffers::new(&gpu.dev, &gpu.tier, gpu.capacity(gpu_model.n()));
        let cameras = orbit(model);
        for (camera, tau) in cameras.iter().cycle().take(WARM_UP) {
            frame(
                gpu,
                &frames,
                &[draw(&gpu_model, loaded, *tau)],
                camera,
                &targets,
            )?;
        }
        let (mut per_camera, mut visible) = (Vec::new(), Vec::new());
        let (mut cull, mut sort, mut draw_ms, mut total) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (camera, tau) in &cameras {
            let mut times = Vec::new();
            for _ in 0..3 {
                let (ms, stats) = frame(
                    gpu,
                    &frames,
                    &[draw(&gpu_model, loaded, *tau)],
                    camera,
                    &targets,
                )?;
                times.push(ms);
                if let Some(s) = stats {
                    let t = s.gpu_times_ms;
                    cull.push(f64::from(t.cull));
                    sort.push(f64::from(t.sort));
                    draw_ms.push(f64::from(t.draw));
                    total.push(f64::from(t.total));
                    visible.push(f64::from(s.visible_splats));
                }
            }
            per_camera.push(median(&mut times));
        }
        let slowest = per_camera.iter().copied().fold(0.0, f64::max);
        let mean = per_camera.iter().sum::<f64>() / per_camera.len() as f64;
        let setup = Json::obj([
            ("model", label),
            ("n", Json::Num(n as f64)),
            ("sh_degree", Json::Num(f64::from(degree))),
            ("width", Json::Num(f64::from(WIDTH))),
            ("height", Json::Num(f64::from(HEIGHT))),
            ("tier", Json::str(gpu.tier.name.clone())),
        ]);
        let fps_min = 1e3 / slowest;
        m.figure(&setup, "fps_min", fps_min, "fps");
        m.figure(&setup, "fps_mean", 1e3 / mean, "fps");
        m.figure(&setup, "frame_ms_max", slowest, "ms");
        m.figure(&setup, "gpu_cull_ms", median(&mut cull), "ms");
        m.figure(&setup, "gpu_sort_ms", median(&mut sort), "ms");
        m.figure(&setup, "gpu_draw_ms", median(&mut draw_ms), "ms");
        m.figure(&setup, "gpu_total_ms", median(&mut total), "ms");
        m.figure(&setup, "visible_splats", median(&mut visible), "splats");
        if given.is_some() || (n == 1_000_000 && degree == 2) {
            m.budgets.push(Budget {
                id: "P-2",
                setup,
                measure: "fps_min",
                rule: Rule::AtLeast,
                limit: 90.0,
                value: fps_min,
            });
        }
    }
    Ok(m)
}

/// The note on how a frame is timed.
const FRAME_NOTE: &str = "a frame's time is the wall time from encoding to the GPU's completion, each frame submitted and waited for (Q-102)";
/// The note on the GPU times (§1.13, Q-81, Q-114).
const GPU_TIMES_NOTE: &str = "gpu_cull_ms, gpu_sort_ms, gpu_draw_ms and gpu_total_ms are medians over the measured frames of the GPU's timestamps: stages 1-3, stage 4, the splat draw, and the cull's begin to the draw's end; reported figures, not budgets (Q-81, Q-114)";
/// The note on the synthetic sweep's load against P-2's model (§1.12 P-2, Q-102, Q-108),
/// carried by a `--model` run's file too.
const SYNTHETIC_LOAD_NOTE: &str = "without --model, B-1's synthetic models (every row persistent, Q-102) fill far more pixels per splat than a trained model of the same N, so the draw is bound by fill rate and P-2 is checked against a heavier load than its trained model until M-8 adds kitchen (Q-108)";

/// B-3's capacities (§3.5).
const B3_C: [usize; 6] = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000];
/// B-3's measured frames (Q-102).
const B3_FRAMES: usize = 32;

/// B-3 (§3.5, Q-102): the sort's GPU time for C keys; P-4.
fn bench_b3(gpu: &Gpu, given: Option<&Given>) -> Result<Measured, HeadlessError> {
    let mut m = Measured::default();
    let cases: Vec<usize> = match given {
        Some(g) => vec![g.model.len()],
        None => B3_C.to_vec(),
    };
    let targets = Targets::new(gpu);
    for c in cases {
        let synthetic;
        let (model, label) = match given {
            Some(g) => (&g.model, g.label()),
            None => {
                let setup_label;
                (synthetic, setup_label) = persistent_synthetic(c, 0);
                (&synthetic, setup_label)
            }
        };
        let (gpu_model, loaded) = gpu.upload(model, &gpu.tier, given.map(|g| g.path.as_path()))?;
        // C = N, the rows uploaded (at most u32::MAX, `GpuModel` counts them in u32).
        let capacity = gpu_model.n();
        let frames = FrameBuffers::new(&gpu.dev, &gpu.tier, capacity);
        let camera = orbit(model)[0].0;
        let tau = (model.time.tau0 + model.time.tau1) / 2.0;
        let mut stats = Vec::new();
        for i in 0..WARM_UP + B3_FRAMES + 2 {
            let (_, s) = frame(
                gpu,
                &frames,
                &[draw(&gpu_model, loaded, tau)],
                &camera,
                &targets,
            )?;
            if i >= WARM_UP {
                stats.extend(s);
            }
        }
        let stats = &stats[stats.len().saturating_sub(B3_FRAMES)..];
        if stats.is_empty() || stats.iter().all(|s| s.gpu_times_ms.sort == 0.0) {
            return Err(HeadlessError::Environment(
                "E-GPU-ADAPTER: the adapter has no timestamp queries, so B-3 cannot time the sort"
                    .to_owned(),
            ));
        }
        let mut sort: Vec<f64> = stats
            .iter()
            .map(|s| f64::from(s.gpu_times_ms.sort))
            .collect();
        let mut keys: Vec<f64> = stats.iter().map(|s| f64::from(s.visible_splats)).collect();
        let setup = Json::obj([
            ("model", label),
            ("capacity", Json::Num(c as f64)),
            ("tier", Json::str(gpu.tier.name.clone())),
        ]);
        let sort_ms = median(&mut sort);
        m.figure(&setup, "sort_ms", sort_ms, "ms");
        m.figure(&setup, "keys", median(&mut keys), "keys");
        if given.is_none() && c == 1_000_000 {
            m.budgets.push(Budget {
                id: "P-4",
                setup,
                measure: "sort_ms",
                rule: Rule::AtMost,
                limit: 0.4,
                value: sort_ms,
            });
        }
    }
    Ok(m)
}

/// B-5's file sizes in MB of 10⁶ bytes (§3.5).
const B5_MB: [u64; 3] = [10, 100, 500];
/// B-5's synthetic degree (Q-102).
const B5_DEGREE: u8 = 2;
/// The loads timed per file, the median kept (Q-102).
const B5_REPEATS: usize = 3;

/// B-5 (§3.5, Q-102): load time from local storage; P-3.
fn bench_b5(gpu: &Gpu, given: Option<&Given>) -> Result<Measured, HeadlessError> {
    let mut m = Measured::default();
    let mut timed = |path: &Path, label: Json, file_bytes: u64| -> Result<(), HeadlessError> {
        let mut times = Vec::new();
        for _ in 0..B5_REPEATS {
            let start = Instant::now();
            let model = load(path)?;
            let uploaded = gpu.upload(&model, &gpu.tier, Some(path))?;
            gpu.wait()?;
            times.push(start.elapsed().as_secs_f64());
            drop(uploaded);
        }
        let load_s = median(&mut times);
        let per_100mb = load_s * 1e8 / file_bytes as f64;
        let setup = Json::obj([
            ("model", label),
            ("file_bytes", Json::Num(file_bytes as f64)),
            ("tier", Json::str(gpu.tier.name.clone())),
        ]);
        m.figure(&setup, "load_s", load_s, "s");
        m.figure(&setup, "s_per_100mb", per_100mb, "s");
        m.budgets.push(Budget {
            id: "P-3",
            setup,
            measure: "s_per_100mb",
            rule: Rule::AtMost,
            limit: 1.0,
            value: per_100mb,
        });
        Ok(())
    };
    if let Some(g) = given {
        let file = identity_file(&g.path);
        let bytes = fs::metadata(&file)
            .map_err(|e| input_error(&file, e))?
            .len();
        timed(&g.path, g.label(), bytes)?;
        return Ok(Measured {
            notes: vec![CACHE_NOTE],
            ..m
        });
    }
    let dir = std::env::temp_dir().join(format!("ftgsxr-bench-{}", std::process::id()));
    fs::create_dir_all(&dir).map_err(|e| input_error(&dir, e))?;
    let result = (|| {
        let sample = |n| encode_exchange(&synthetic_model(1, n, B5_DEGREE, Persistence::Few)).len();
        let (a, b) = (sample(1000) as u64, sample(2000) as u64);
        let per_row = (b - a) / 1000;
        let header = a - 1000 * per_row;
        for mb in B5_MB {
            let n = ((mb * 1_000_000).saturating_sub(header) / per_row).max(1) as usize;
            let path = dir.join(format!("b5-{mb}mb.ftgs-if"));
            write_exchange(&path, &synthetic_model(1, n, B5_DEGREE, Persistence::Few))
                .map_err(|e| core_input(&path, &e))?;
            let bytes = fs::metadata(&path)
                .map_err(|e| input_error(&path, e))?
                .len();
            let timed_file = timed(&path, Json::str("synthetic"), bytes);
            let _ = fs::remove_file(&path);
            timed_file?;
        }
        Ok(())
    })();
    let _ = fs::remove_dir_all(&dir);
    result?;
    Ok(Measured {
        notes: vec![CACHE_NOTE],
        ..m
    })
}

/// The note on B-5's file reads.
const CACHE_NOTE: &str = "B-5 reads files the page cache may hold (a synthetic file was just written; a cold read needs the cache dropped), so its load times are a lower bound (Q-102)";

/// §1.12's static GPU bytes per primitive for L = 0 … 3: float32 storage, 64 + 12 (L+1)².
const FLOAT32_BYTES: [f64; 4] = [76.0, 112.0, 172.0, 256.0];
/// As `FLOAT32_BYTES` for half-precision storage, 42 + 6 (L+1)².
const HALF_BYTES: [f64; 4] = [48.0, 66.0, 96.0, 138.0];
/// B-6's synthetic N (Q-102).
const B6_N: usize = 100_000;

/// B-6 (§3.5, Q-102): static bytes per primitive in every tier and L against §1.12's
/// table, within 1 %.
fn bench_b6(gpu: &Gpu, given: Option<&Given>) -> Result<Measured, HeadlessError> {
    let mut m = Measured::default();
    let tiers = [
        QualityTier::conformance(),
        QualityTier::desktop(),
        QualityTier::quest_high(),
        QualityTier::quest_balanced(),
        QualityTier::quest_low(),
    ];
    let degrees: Vec<u8> = match given {
        Some(g) => vec![g.model.sh_degree],
        None => (0..=3).collect(),
    };
    for degree in degrees {
        let synthetic;
        let (model, label) = match given {
            Some(g) => (&g.model, g.label()),
            None => {
                synthetic = synthetic_model(1, B6_N, degree, Persistence::Few);
                (&synthetic, Json::str("synthetic"))
            }
        };
        for tier in &tiers {
            let (gpu_model, loaded) = gpu.upload(model, tier, given.map(|g| g.path.as_path()))?;
            let value = gpu_model.memory_bytes() as f64 / f64::from(gpu_model.n());
            let half = tier.layout.half;
            let any_half = half.velocity || half.scale || half.rotation || half.opacity || half.sh;
            let table = if any_half {
                &HALF_BYTES
            } else {
                &FLOAT32_BYTES
            };
            let setup = Json::obj([
                ("model", label.clone()),
                ("n", Json::Num(model.len() as f64)),
                ("sh_degree", Json::Num(f64::from(degree))),
                ("loaded_degree", Json::Num(f64::from(loaded))),
                ("tier", Json::str(tier.name.clone())),
            ]);
            m.figure(&setup, "bytes_per_primitive", value, "bytes");
            m.budgets.push(Budget {
                id: "B-6",
                setup,
                measure: "bytes_per_primitive",
                rule: Rule::Within1Percent,
                limit: table[usize::from(loaded)],
                value,
            });
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ftgsxr_core::CoreError;

    fn budget(rule: Rule, value: f64) -> Budget {
        Budget {
            id: "P-2",
            setup: Json::obj([("n", Json::Num(1e6)), ("sh_degree", Json::Num(2.0))]),
            measure: "fps_min",
            rule,
            limit: 90.0,
            value,
        }
    }

    fn baseline_of(b: &Budget, value: f64) -> Json {
        let mut old = b.clone();
        old.value = value;
        Json::obj([("budgets", Json::Arr(vec![old.to_json()]))])
    }

    #[test]
    fn compare_lists_a_figure_against_a_baseline_20_percent_better() {
        // Higher is better: a baseline 20 % better is 1.2 × the value.
        let b = budget(Rule::AtLeast, 100.0);
        assert_eq!(
            compare(std::slice::from_ref(&b), &baseline_of(&b, 120.0)).len(),
            1
        );
        // Lower is better: a baseline 20 % better is 0.8 × the value.
        let b = budget(Rule::AtMost, 100.0);
        assert_eq!(
            compare(std::slice::from_ref(&b), &baseline_of(&b, 80.0)).len(),
            1
        );
    }

    #[test]
    fn compare_keeps_quiet_within_10_percent_and_when_better() {
        let b = budget(Rule::AtLeast, 100.0);
        for base in [100.0, 110.0, 50.0] {
            assert!(
                compare(std::slice::from_ref(&b), &baseline_of(&b, base)).is_empty(),
                "{base}"
            );
        }
        let b = budget(Rule::AtMost, 100.0);
        for base in [100.0, 91.0, 200.0] {
            assert!(
                compare(std::slice::from_ref(&b), &baseline_of(&b, base)).is_empty(),
                "{base}"
            );
        }
    }

    #[test]
    fn compare_matches_id_setup_and_measure() {
        let b = budget(Rule::AtLeast, 100.0);
        let mut other = b.clone();
        other.setup = Json::obj([("n", Json::Num(2e6)), ("sh_degree", Json::Num(2.0))]);
        assert!(compare(std::slice::from_ref(&b), &baseline_of(&other, 1000.0)).is_empty());
        let mut other = b.clone();
        other.measure = "fps_mean";
        assert!(compare(std::slice::from_ref(&b), &baseline_of(&other, 1000.0)).is_empty());
        let mut other = b.clone();
        other.id = "P-4";
        assert!(compare(std::slice::from_ref(&b), &baseline_of(&other, 1000.0)).is_empty());
        assert!(compare(std::slice::from_ref(&b), &Json::obj([])).is_empty());
        // The setup's members in another order are the same setup.
        let mut reordered = b.clone();
        reordered.setup = Json::obj([("sh_degree", Json::Num(2.0)), ("n", Json::Num(1e6))]);
        assert_eq!(compare(&[b], &baseline_of(&reordered, 1000.0)).len(), 1);
    }

    /// `baseline_of`'s file with `profile` and `machine.gpu` added.
    fn baseline_from(b: &Budget, value: f64, profile: &str, gpu: &str) -> Json {
        let mut old = b.clone();
        old.value = value;
        Json::obj([
            ("machine", Json::obj([("gpu", Json::str(gpu))])),
            ("profile", Json::str(profile)),
            ("budgets", Json::Arr(vec![old.to_json()])),
        ])
    }

    #[test]
    fn compare_against_skips_a_baseline_of_another_profile() {
        // Q-103 (#180): a baseline 20 % better from a release build lists nothing for a
        // debug run, and the note says which field differs.
        let b = budget(Rule::AtLeast, 100.0);
        let base = baseline_from(&b, 120.0, "release", "GPU A");
        assert_eq!(
            compare_against(std::slice::from_ref(&b), "debug", "GPU A", &base),
            Err("baseline not compared: profile debug against release (Q-103)".to_owned())
        );
        assert_eq!(
            compare_against(std::slice::from_ref(&b), "release", "GPU A", &base).map(|r| r.len()),
            Ok(1)
        );
    }

    #[test]
    fn compare_against_skips_a_baseline_of_another_gpu() {
        // Q-103 (#180): the same with another `machine.gpu`.
        let b = budget(Rule::AtLeast, 100.0);
        let base = baseline_from(&b, 120.0, "release", "GPU A");
        assert_eq!(
            compare_against(std::slice::from_ref(&b), "release", "GPU B", &base),
            Err("baseline not compared: machine.gpu GPU B against GPU A (Q-103)".to_owned())
        );
    }

    #[test]
    fn compare_skips_within_1_percent_budgets() {
        // Q-103 (#180): B-6's own 1 % budget is its check; a baseline 20 % different,
        // either way, lists nothing.
        let b = budget(Rule::Within1Percent, 100.0);
        for base in [80.0, 120.0] {
            assert!(
                compare(std::slice::from_ref(&b), &baseline_of(&b, base)).is_empty(),
                "{base}"
            );
        }
    }

    #[test]
    fn rule_passes_its_limit() {
        assert!(Rule::AtLeast.passes(90.0, 90.0) && !Rule::AtLeast.passes(89.9, 90.0));
        assert!(Rule::AtMost.passes(0.4, 0.4) && !Rule::AtMost.passes(0.41, 0.4));
        assert!(Rule::Within1Percent.passes(76.7, 76.0));
        assert!(!Rule::Within1Percent.passes(76.8, 76.0));
        assert!(Rule::Within1Percent.passes(75.3, 76.0));
    }

    /// The version of the package `name` in the workspace's `Cargo.lock`.
    fn locked(name: &str) -> String {
        let lock =
            fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock"))
                .expect("the workspace's Cargo.lock");
        let entry = format!("name = \"{name}\"\nversion = \"");
        let at = lock.find(&entry).expect("the package is locked") + entry.len();
        lock[at..].split('"').next().expect("a version").to_owned()
    }

    #[test]
    fn versions_match_the_lock_file() {
        assert_eq!(BEVY_VERSION, locked("bevy"));
        assert_eq!(WGPU_VERSION, locked("wgpu"));
    }

    #[test]
    fn orbit_looks_at_the_box_center_over_the_span() {
        let model = synthetic_model(3, 50, 0, Persistence::Few);
        let cameras = orbit(&model);
        assert_eq!(cameras.len(), ORBIT);
        assert_eq!(cameras[0].1, model.time.tau0);
        assert_eq!(cameras[ORBIT - 1].1, model.time.tau1);
        let (lo, hi) = model.position.iter().fold(
            ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
            |(mut lo, mut hi), p| {
                for a in 0..3 {
                    lo[a] = lo[a].min(f64::from(p[a]));
                    hi[a] = hi[a].max(f64::from(p[a]));
                }
                (lo, hi)
            },
        );
        let center = DVec3::from_array([0, 1, 2].map(|a| (lo[a] + hi[a]) / 2.0));
        for (camera, _) in &cameras {
            // The center lies on the optical axis, in front: (0, 0, z > 0) in the camera.
            let c = camera.world_to_camera.transform_point3(center);
            assert!(
                c.z > 0.0 && c.x.abs() < 1e-4 * c.z && c.y.abs() < 1e-4 * c.z,
                "{c}"
            );
            // y up in stage space is −y in the camera's OpenCV axes (§1.14).
            let up = DMat3::from_mat4(camera.world_to_camera) * DVec3::Y;
            assert!(up.y < 0.0, "{up}");
        }
    }

    #[test]
    fn persistent_synthetic_has_every_row_persistent() {
        for (n, degree) in [(1, 0), (300, 2), (1000, 3)] {
            let (model, label) = persistent_synthetic(n, degree);
            assert_eq!((model.len(), model.sh_degree), (n, degree));
            assert_eq!(label, Json::str("synthetic, every row persistent"));
            // E3: every row is persistent, so every τ of the span visits all N and no
            // transient row.
            let (index, _) = TemporalIndex::build(&model);
            for tau in [model.time.tau0, 0.5, model.time.tau1] {
                let range = index.range(tau);
                assert_eq!((range.persistent.len(), range.transient.len()), (n, 0));
            }
        }
    }

    #[test]
    fn gpu_error_maps_render_errors_as_q75_and_q90() {
        // Q-75: a core error on a file names the file; without one, no path.
        // §1.15's E-FILE-KEYS text, "Missing key K".
        let core = || RenderError::Core(CoreError::Keys("velocity".to_owned()));
        assert_eq!(
            gpu_error(Some(Path::new("m.ftgs-if")), core()),
            HeadlessError::Input("E-FILE-KEYS: m.ftgs-if: Missing key velocity".to_owned())
        );
        assert_eq!(
            gpu_error(None, core()),
            HeadlessError::Input("E-FILE-KEYS: Missing key velocity".to_owned())
        );
        // Q-90: a non-finite render is Numeric, a GPU condition Environment with its
        // §1.15 code, a file or not.
        let file = Some(Path::new("m.ftgs-if"));
        assert_eq!(
            gpu_error(file, RenderError::NonFinite),
            HeadlessError::Numeric(String::new())
        );
        let budget = || RenderError::Budget {
            needed: 3_000_000,
            budget: 2_000_000,
        };
        let text = "E-MEM-BUDGET: Needs 3 MB, budget 2 MB; reduce to budget or choose a tier";
        assert_eq!(
            gpu_error(file, budget()),
            HeadlessError::Environment(text.to_owned())
        );
        assert_eq!(
            gpu_error(None, budget()),
            HeadlessError::Environment(text.to_owned())
        );
    }

    #[test]
    fn median_takes_the_middle_value() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
        assert_eq!(median(&mut []), 0.0);
    }
}
