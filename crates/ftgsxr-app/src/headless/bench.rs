// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr bench NAME [--out BENCH.json] [--model MODEL]` (§1.7, §3.5): the desktop
//! benchmarks B-1, B-3, B-5, B-6 and B-9 (#289, Q-155), their JSON file and the comparison with the stored
//! baseline (Q-67, Q-102, Q-103).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bevy::color::LinearRgba;
use bevy::log::warn;
use bevy::render::render_resource::{
    CommandEncoderDescriptor, Extent3d, LoadOp, Operations, PollType, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureViewDescriptor, WgpuTextureView,
};
use ftgsxr_core::{
    Camera, DMat3, DVec3, Fov, Model, Pose, TemporalIndex, UVec2, Vec3, encode_exchange, hex,
    pack_for_gpu, write_exchange,
};
use ftgsxr_render::{
    BudgetSource, Capacity, ColorTransform, FrameBuffers, FrameStats, GpuModel, MemoryBudget,
    ModelDraw, Overlays, QualityTier, RenderError, RenderMode, SortMode, SortPose, SplatDevice,
    SplatPasses, ViewParams,
};

mod b9;

use super::{HeadlessError, core_input, identity_file, input_error, load, with_n, write_atomic};
use crate::benchmark::json::Json;
use crate::benchmark::synthetic::{Persistence, synthetic_model};
use crate::benchmark::{Budget, Measured, Rule, look_at, median, report_json};

/// What `cmd_bench` measured (Q-67): the file's text and the budgets that regressed
/// against the stored baseline, one line each; the binary exits 3 when there is any.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchReport {
    /// The benchmark file's JSON text (Q-103), as written at `out`.
    pub json: String,
    /// One line per budgeted figure more than 10 % worse than the baseline's (§3.5).
    pub regressions: Vec<String>,
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
/// desktop benchmark `name` (B-1, B-3, B-5, B-6 or B-9) on the high-performance adapter, with
/// the setups of Q-102 or, with `model`, on that one model (any §1.2 input); compares
/// its budgets with the stored baseline `<NAME>.json` when the baseline has the run's
/// profile and GPU, else notes why not in the file (Q-103); and writes the file at `out`
/// when given. A NAME the desktop does not run is `Input`; a malformed baseline is `Input`
/// (E-JSON), and an `out` whose directory does not exist an `Input` error, both before the
/// GPU opens; an invalid model is `Input` with its E-FILE code; no adapter, or no timestamp
/// queries for B-3, is `Environment`. GPU validation messages are a note, not an error
/// (Q-116).
// Q-67, Q-102, Q-103
pub fn cmd_bench(
    name: &str,
    out: Option<&Path>,
    model: Option<&Path>,
) -> Result<BenchReport, HeadlessError> {
    cmd_bench_on(name, out, model, None)
}

/// `cmd_bench` on a machine that may be a headset (Q-406): with `headset`, the headset's
/// build fingerprint, the run is the headset's, before any XR session: B-3, B-5 or B-6
/// (any other NAME is `Input`), in `quest-balanced` with its 1.5 GiB budget, B-3 against
/// P-4's Quest limit (Q-413), the file's `machine.headset` holding `headset` as given and
/// the stored baseline `<NAME>.quest.json`; B-5's files are written beside `out` when it is
/// given (Q-417). A size the device refuses, over its memory budget or a binding
/// limit (Q-377), is a result with a null `value` and the reason in `skipped`, and no
/// budget (Q-413). Without `headset` it is `cmd_bench`.
// Q-406, Q-413, Q-417
pub fn cmd_bench_on(
    name: &str,
    out: Option<&Path>,
    model: Option<&Path>,
    headset: Option<&str>,
) -> Result<BenchReport, HeadlessError> {
    let on = headset.map_or(On::Desktop, |h| On::Headset(h.to_owned()));
    cmd_bench_in(name, out, model, &on, &baseline_dir())
}

/// The machine a run is on (Q-406).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum On {
    /// The desktop (Q-67, Q-102).
    Desktop,
    /// The headset, with its build fingerprint (Q-406).
    Headset(String),
}

/// A benchmark's runner.
type Runner = fn(&Gpu, Option<&Given>) -> Result<Measured, HeadlessError>;

impl On {
    /// The runner of `name` on this machine, `None` when it does not run it here.
    fn runner(&self, name: &str) -> Option<Runner> {
        let run: Runner = match (self, name) {
            (_, "B-3") => bench_b3,
            (_, "B-5") => bench_b5,
            (_, "B-6") => bench_b6,
            (On::Desktop, "B-1") => bench_b1,
            // Q-155
            (On::Desktop, "B-9") => b9::bench_b9,
            _ => return None,
        };
        Some(run)
    }

    /// The names `runner` takes, for the refusal's text.
    fn names(&self) -> &'static str {
        match self {
            On::Desktop => "not a desktop benchmark (B-1, B-3, B-5, B-6, B-9)",
            On::Headset(_) => "not a headset benchmark run before the session (B-3, B-5, B-6)",
        }
    }

    /// The tier of the run: `desktop` (Q-102), `quest-balanced` on the headset (Q-413).
    // Q-413
    fn tier(&self) -> QualityTier {
        match self {
            On::Desktop => QualityTier::desktop(),
            On::Headset(_) => QualityTier::quest_balanced(),
        }
    }

    /// P-4's limit on this machine (§1.12): 0.4 ms on the RTX 4090, 2.5 ms on the Quest.
    fn p4_limit_ms(&self) -> f64 {
        match self {
            On::Desktop => 0.4,
            On::Headset(_) => 2.5,
        }
    }

    /// The stored baseline's file name (Q-103, Q-406).
    // Q-406
    fn baseline_file(&self, name: &str) -> String {
        match self {
            On::Desktop => format!("{name}.json"),
            On::Headset(_) => format!("{name}.quest.json"),
        }
    }

    /// The headset's build fingerprint, if any.
    fn headset(&self) -> Option<&str> {
        match self {
            On::Desktop => None,
            On::Headset(h) => Some(h),
        }
    }
}

/// `cmd_bench_on` with the baselines' directory given.
fn cmd_bench_in(
    name: &str,
    out: Option<&Path>,
    model: Option<&Path>,
    on: &On,
    baselines: &Path,
) -> Result<BenchReport, HeadlessError> {
    // Q-T731.4: the session benchmarks run on the headset, in the XR session.
    let session =
        matches!(name, "B-2" | "B-7" | "B-8") || (name == "B-9" && on.headset().is_some());
    if session {
        let test = format!("dv_{}_", name.to_lowercase().replace('-', ""));
        return Err(HeadlessError::Input(format!(
            "{name}: a headset benchmark run in the session: the dv test {test}… in \
             crates/ftgsxr-xr/tests/dv_session_bench.rs (Q-T731.4)"
        )));
    }
    let Some(run) = on.runner(name) else {
        return Err(HeadlessError::Input(format!("{name}: {}", on.names())));
    };
    // #194: the baseline and the output directory are checked before any frame, so a bad
    // one doesn't throw a run away (Q-103's errors, unchanged).
    let baseline = read_baseline(&baselines.join(on.baseline_file(name)))?;
    if let Some(out) = out {
        check_out_dir(out)?;
    }
    // Q-417: on the headset the system's temporary directory is not the app's to
    // write, so B-5's files go beside `out`.
    let scratch = match (on, out.and_then(Path::parent)) {
        (On::Headset(_), Some(dir)) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => std::env::temp_dir(),
    }
    .join(format!("ftgsxr-bench-{}", std::process::id()));
    let given = model
        .map(|path| {
            load(path).map(|model| Given {
                path: path.to_path_buf(),
                model,
            })
        })
        .transpose()?;
    let gpu = Gpu::new(on.clone(), scratch)?;
    let measured = run(&gpu, given.as_ref());
    let messages = gpu.dev.take_validation_messages();
    let measured = measured?;

    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let mut notes: Vec<String> = measured.notes.iter().map(|n| (*n).to_owned()).collect();
    // Q-116
    for m in &messages {
        warn!("ftgsxr bench {name}: GPU validation: {m}");
    }
    notes.extend(validation_note(&messages));
    let (baseline, regressions) = match baseline {
        // A baseline not compared against is not `baseline` (Q-103: "the path compared
        // against, or null").
        Some((path, parsed)) => {
            match compare_against(&measured.budgets, profile, gpu.dev.adapter_name(), &parsed) {
                Ok(regressions) => (Json::str(path.display().to_string()), regressions),
                Err(note) => {
                    // #193, Q-110: the library logs; the binary prints the note from the
                    // file's `notes` (Q-103).
                    warn!("ftgsxr bench {name}: {note}");
                    notes.push(note);
                    (Json::Null, Vec::new())
                }
            }
        }
        None => (Json::Null, Vec::new()),
    };
    let json = report_json(
        name,
        gpu.dev.adapter_name(),
        profile,
        &measured,
        baseline,
        notes,
        on.headset(),
    );
    if let Some(out) = out {
        write_atomic(out, json.as_bytes()).map_err(|e| input_error(out, e))?;
    }
    Ok(BenchReport { json, regressions })
}

/// The stored baseline at `path`, canonicalized, and its parsed text; `None` when there is
/// no file, E-JSON when it is malformed (Q-103).
// Q-103
fn read_baseline(path: &Path) -> Result<Option<(PathBuf, Json)>, HeadlessError> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|e| input_error(path, e))?;
    let parsed = Json::parse(&text)
        .map_err(|e| HeadlessError::Input(format!("E-JSON: {}: {e}", path.display())))?;
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    Ok(Some((canonical, parsed)))
}

/// An `Input` error naming `out` when the directory it would be written in is not one, or
/// can't be written: the temporary file `write_atomic` would use is created and removed.
fn check_out_dir(out: &Path) -> Result<(), HeadlessError> {
    let dir = match out.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    match fs::metadata(dir) {
        Ok(m) if m.is_dir() => {}
        Ok(_) => {
            return Err(input_error(
                out,
                std::io::Error::new(std::io::ErrorKind::NotADirectory, "not a directory"),
            ));
        }
        Err(e) => return Err(input_error(out, e)),
    }
    let name = out.file_name().map_or_else(
        || "ftgsxr-bench".into(),
        |n| n.to_string_lossy().into_owned(),
    );
    let probe = dir.join(format!(".{name}.tmp-{}", std::process::id()));
    fs::File::create(&probe).map_err(|e| input_error(out, e))?;
    let _ = fs::remove_file(&probe);
    Ok(())
}

/// The note for the GPU validation messages collected during a run, if any (Q-116): the
/// count and every message, which the binary prints on standard error.
// Q-116
pub(crate) fn validation_note(messages: &[String]) -> Option<String> {
    (!messages.is_empty()).then(|| {
        format!(
            "GPU validation reported {} messages (Q-116): {}",
            messages.len(),
            messages.join("; ")
        )
    })
}

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

/// The device and the run's tier's passes (Q-102, Q-413), the machine and B-5's scratch
/// directory.
struct Gpu {
    dev: SplatDevice,
    tier: QualityTier,
    passes: SplatPasses,
    on: On,
    scratch: PathBuf,
}

impl Gpu {
    fn new(on: On, scratch: PathBuf) -> Result<Gpu, HeadlessError> {
        let mut dev = SplatDevice::headless().map_err(|e| gpu_error(None, e))?;
        let tier = on.tier();
        // Q-86, Q-413: the Quest tier's fixed budget replaces the device's desktop rule,
        // as `HeadlessRenderer` does.
        if let On::Headset(_) = on {
            dev.set_memory_budget(MemoryBudget {
                bytes: tier.memory_budget_bytes,
                source: BudgetSource::Tier,
            });
        }
        let passes =
            SplatPasses::new(&dev, LAYER_FORMAT, None, &tier).map_err(|e| gpu_error(None, e))?;
        Ok(Gpu {
            dev,
            tier,
            passes,
            on,
            scratch,
        })
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
        // Q-117: a refusal over the budget gives N.
        let gpu_model = GpuModel::upload(&self.dev, &arrays, index, &model.conventions)
            .map_err(|e| with_n(gpu_error(file, e), model.len()))?;
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
        trim: ftgsxr_core::Trim::default(),
    }
}

/// B-1's and B-3's synthetic model (seed 1) of `n` rows at `degree`, every row
/// persistent, so every row is active at every τ (E3), with its setup label (Q-102).
// Q-102
fn persistent_synthetic(n: usize, degree: u8) -> (Model, Json) {
    (
        synthetic_model(1, n, degree, Persistence::All),
        Json::str(SYNTHETIC_PERSISTENT),
    )
}

/// The `model` label of `persistent_synthetic`'s models.
const SYNTHETIC_PERSISTENT: &str = "synthetic, every row persistent";

/// B-1's setup (Q-102).
fn b1_setup(label: Json, n: usize, degree: u8, tier: &QualityTier) -> Json {
    Json::obj([
        ("model", label),
        ("n", Json::Num(n as f64)),
        ("sh_degree", Json::Num(f64::from(degree))),
        ("width", Json::Num(f64::from(WIDTH))),
        ("height", Json::Num(f64::from(HEIGHT))),
        ("tier", Json::str(tier.name.clone())),
    ])
}

/// B-3's setup (Q-102), naming the tier (Q-413).
fn b3_setup(label: Json, capacity: usize, tier: &QualityTier) -> Json {
    Json::obj([
        ("model", label),
        ("capacity", Json::Num(capacity as f64)),
        ("tier", Json::str(tier.name.clone())),
    ])
}

/// B-5's setup (Q-102), naming the tier (Q-413).
fn b5_setup(label: Json, file_bytes: u64, tier: &QualityTier) -> Json {
    Json::obj([
        ("model", label),
        ("file_bytes", Json::Num(file_bytes as f64)),
        ("tier", Json::str(tier.name.clone())),
    ])
}

/// §1.12 Memory's record bytes per C: 48 per view, 64 in shared-center mode, 48 with
/// `quest-low`'s half-precision Σ^S.
fn record_bytes(tier: &QualityTier) -> u64 {
    match (tier.sort_mode, tier.half_records) {
        (SortMode::PerView, _) | (SortMode::SharedCenter, true) => 48,
        (SortMode::SharedCenter, false) => 64,
    }
}

/// Why the device refuses `tier`'s frame buffers of capacity `capacity`, or `None` when
/// they fit (Q-413, Q-377): the records buffer, the largest, over the device's storage
/// binding limit or its `max_buffer_size`. `FrameBuffers::new` refuses nothing itself.
// Q-413
fn records_refusal(
    capacity: u32,
    tier: &QualityTier,
    binding_limit: u64,
    max_buffer: u64,
) -> Option<String> {
    let bytes = u64::from(capacity.max(1)) * record_bytes(tier);
    let (limit, name) = if max_buffer < binding_limit {
        (max_buffer, "max_buffer_size")
    } else {
        (binding_limit, "max_storage_buffer_binding_size")
    };
    (bytes > limit).then(|| {
        format!(
            "the frame buffers' records of {bytes} bytes for C = {capacity} exceed the \
             device's {name} of {limit} bytes (Q-377, Q-413)"
        )
    })
}

/// Whether `e` is the headset's refusal of a size (E-MEM-BUDGET, Q-377's binding limits),
/// which the size records as skipped (Q-413); on the desktop it stays the run's error
/// (Q-117).
fn refused(on: &On, e: &HeadlessError) -> bool {
    matches!(on, On::Headset(_)) && e.code() == "E-MEM-BUDGET"
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
        let setup = b1_setup(label, n, degree, &gpu.tier);
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
// Q-102
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
        let setup = b3_setup(label, c, &gpu.tier);
        // Q-413: a size the device refuses is a result without a value.
        let limits = gpu.dev.device().limits();
        let refusal = matches!(gpu.on, On::Headset(_)).then(|| {
            records_refusal(
                u32::try_from(model.len()).unwrap_or(u32::MAX),
                &gpu.tier,
                limits.max_storage_buffer_binding_size,
                limits.max_buffer_size,
            )
        });
        if let Some(Some(reason)) = refusal {
            m.skip(&setup, "sort_ms", "ms", reason);
            continue;
        }
        let (gpu_model, loaded) =
            match gpu.upload(model, &gpu.tier, given.map(|g| g.path.as_path())) {
                Ok(uploaded) => uploaded,
                Err(e) if refused(&gpu.on, &e) => {
                    m.skip(&setup, "sort_ms", "ms", e.to_string());
                    continue;
                }
                Err(e) => return Err(e),
            };
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
        let sort_ms = median(&mut sort);
        m.figure(&setup, "sort_ms", sort_ms, "ms");
        m.figure(&setup, "keys", median(&mut keys), "keys");
        if given.is_none() && c == 1_000_000 {
            m.budgets.push(Budget {
                id: "P-4",
                setup,
                measure: "sort_ms",
                rule: Rule::AtMost,
                // §1.12 P-4, Q-413: the machine's limit.
                limit: gpu.on.p4_limit_ms(),
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
// Q-102
fn bench_b5(gpu: &Gpu, given: Option<&Given>) -> Result<Measured, HeadlessError> {
    let mut m = Measured::default();
    let mut timed = |path: &Path, label: Json, file_bytes: u64| -> Result<(), HeadlessError> {
        let setup = b5_setup(label, file_bytes, &gpu.tier);
        let mut times = Vec::new();
        for _ in 0..B5_REPEATS {
            let start = Instant::now();
            let model = load(path)?;
            let uploaded = match gpu.upload(&model, &gpu.tier, Some(path)) {
                Ok(uploaded) => uploaded,
                // Q-413: a size the device refuses is a result without a value.
                Err(e) if refused(&gpu.on, &e) => {
                    m.skip(&setup, "load_s", "s", e.to_string());
                    m.skip(&setup, "s_per_100mb", "s", e.to_string());
                    return Ok(());
                }
                Err(e) => return Err(e),
            };
            gpu.wait()?;
            times.push(start.elapsed().as_secs_f64());
            drop(uploaded);
        }
        let load_s = median(&mut times);
        let per_100mb = load_s * 1e8 / file_bytes as f64;
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
    // Q-417
    let dir = gpu.scratch.clone();
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
// Q-102
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
    use crate::benchmark::{BEVY_VERSION, WGPU_VERSION};
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
    fn validation_note_reports_the_messages_without_failing() {
        // Q-116: no message, no note; collected messages are one note naming their
        // count and each message, and the note names no §1.15 error code.
        assert_eq!(validation_note(&[]), None);
        let messages = ["VUID-a: first".to_owned(), "VUID-b: second".to_owned()];
        let note = validation_note(&messages).expect("a note");
        assert!(note.contains("2 messages"), "{note}");
        assert!(
            note.contains("VUID-a: first") && note.contains("VUID-b: second"),
            "{note}"
        );
        assert!(
            note.contains("Q-116") && !note.contains("E-GPU-SHADER"),
            "{note}"
        );
    }

    /// A fresh directory under the system's temporary directory.
    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ftgsxr-bench-test-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("a scratch directory");
        dir
    }

    #[test]
    fn cmd_bench_in_refuses_a_malformed_baseline_before_rendering() {
        // #194, Q-103: a malformed baseline is E-JSON (exit 2). The model named does not
        // exist, so reaching the model's load (or the GPU) would give another error: the
        // E-JSON proves the baseline is read first.
        let dir = scratch("malformed");
        fs::write(dir.join("B-6.json"), "{ not json").expect("the baseline");
        let err = cmd_bench_in(
            "B-6",
            None,
            Some(&dir.join("missing.ftgs-if")),
            &On::Desktop,
            &dir,
        )
        .expect_err("a malformed baseline");
        match &err {
            HeadlessError::Input(text) => assert!(text.starts_with("E-JSON: "), "{text}"),
            other => panic!("expected E-JSON, got {other:?}"),
        }
        assert_eq!(super::super::exit_code(&err), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cmd_bench_in_refuses_an_out_in_a_missing_directory_before_rendering() {
        // #194: `--out` in a directory that doesn't exist fails before the model's load
        // (a missing model, which would fail differently) and so before any frame.
        let dir = scratch("out");
        let out = dir.join("no-such-dir").join("b6.json");
        let err = cmd_bench_in(
            "B-6",
            Some(&out),
            Some(&dir.join("missing.ftgs-if")),
            &On::Desktop,
            &dir,
        )
        .expect_err("an unwritable --out");
        match &err {
            HeadlessError::Input(text) => assert!(text.contains("no-such-dir"), "{text}"),
            other => panic!("expected an Input error, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cmd_bench_in_refuses_an_out_in_a_read_only_directory_before_rendering() {
        // #194: `--out` in a directory that exists but can't be written fails before the
        // model's load (a missing model, which would fail differently), as E-IO (exit 2).
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("read-only");
        let locked = dir.join("locked");
        fs::create_dir(&locked).expect("the locked directory");
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o555)).expect("read-only");
        // Root writes a read-only directory anyway, so there is nothing to refuse.
        let writable = fs::write(locked.join("probe"), b"").is_ok();
        if !writable {
            let out = locked.join("b6.json");
            let err = cmd_bench_in(
                "B-6",
                Some(&out),
                Some(&dir.join("missing.ftgs-if")),
                &On::Desktop,
                &dir,
            )
            .expect_err("an unwritable --out");
            match &err {
                HeadlessError::Input(text) => assert!(text.contains("locked"), "{text}"),
                other => panic!("expected an Input error, got {other:?}"),
            }
            assert_eq!(super::super::exit_code(&err), 2);
            assert_eq!(fs::read_dir(&locked).unwrap().count(), 0, "no file left");
        }
        let _ = fs::set_permissions(&locked, fs::Permissions::from_mode(0o755));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn report_json_puts_a_budget_not_met_on_a_pass_false_line() {
        // #198, Q-103: the `ftgsxr` binary (crates/ftgsxr-desktop/src/main.rs) finds a
        // budget not met as a line starting `"pass": false`; the writer must keep that form.
        let measured = Measured {
            budgets: vec![budget(Rule::AtLeast, 50.0)],
            ..Measured::default()
        };
        let json = report_json(
            "B-1",
            "GPU A",
            "debug",
            &measured,
            Json::Null,
            Vec::new(),
            None,
        );
        assert!(
            json.lines()
                .any(|l| l.trim_start().starts_with("\"pass\": false")),
            "{json}"
        );
        let met = Measured {
            budgets: vec![budget(Rule::AtLeast, 100.0)],
            ..Measured::default()
        };
        let json = report_json("B-1", "GPU A", "debug", &met, Json::Null, Vec::new(), None);
        assert!(!json.contains("\"pass\": false"), "{json}");
    }

    #[test]
    fn report_json_names_the_headset_only_when_given() {
        // Q-406, #724: `cmd_bench`'s file has no `machine.headset`; `cmd_bench_on`'s holds
        // the string it was given, as given, and stays at `version` 1 (Q-67).
        let measured = Measured::default();
        let desktop = Json::parse(&report_json(
            "B-3",
            "GPU A",
            "release",
            &measured,
            Json::Null,
            Vec::new(),
            None,
        ))
        .expect("well formed");
        let machine = desktop.get("machine").expect("machine");
        assert!(machine.get("headset").is_none(), "{}", desktop.compact());
        for field in ["gpu", "driver", "os", "cpu", "threads"] {
            assert!(machine.get(field).is_some(), "{field}");
        }
        let print = "oculus/eureka/eureka:14/UP1A.231005.007.A1/123:user/release-keys";
        let quest = Json::parse(&report_json(
            "B-3",
            "GPU A",
            "debug",
            &measured,
            Json::Null,
            Vec::new(),
            Some(print),
        ))
        .expect("well formed");
        assert_eq!(
            quest
                .get("machine")
                .and_then(|m| m.get("headset"))
                .and_then(Json::as_str),
            Some(print)
        );
        assert_eq!(quest.get("version").and_then(Json::as_f64), Some(1.0));
    }

    #[test]
    fn report_json_writes_a_skipped_size_as_a_null_value_with_its_reason() {
        // Q-413, Q-417: a size the device refuses is one result with a null value and
        // the reason, and no budget.
        let mut measured = Measured::default();
        let setup = Json::obj([("capacity", Json::Num(4e6))]);
        measured.skip(&setup, "sort_ms", "ms", "E-MEM-BUDGET: refused".to_owned());
        let file = Json::parse(&report_json(
            "B-3",
            "GPU A",
            "debug",
            &measured,
            Json::Null,
            Vec::new(),
            Some("print"),
        ))
        .expect("well formed");
        let results = file
            .get("results")
            .and_then(Json::as_array)
            .expect("results");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("value"), Some(&Json::Null));
        assert_eq!(
            results[0].get("skipped").and_then(Json::as_str),
            Some("E-MEM-BUDGET: refused")
        );
        assert_eq!(
            results[0].get("measure").and_then(Json::as_str),
            Some("sort_ms")
        );
        assert!(
            file.get("budgets")
                .and_then(Json::as_array)
                .expect("budgets")
                .is_empty()
        );
    }

    #[test]
    fn records_refusal_names_the_binding_limit_the_records_exceed() {
        // Q-413, Q-377: §1.12's 64-byte shared-center records of `quest-balanced` at
        // C = 4 × 10⁶ are 256,000,000 bytes, over the Quest's 134,217,728-byte binding;
        // at 2 × 10⁶ they are 128,000,000 bytes and fit. The desktop's 48-byte records at
        // 4 × 10⁶ (192,000,000 bytes) fit a 4 GiB binding.
        let quest = |c| {
            records_refusal(
                c,
                &QualityTier::quest_balanced(),
                134_217_728,
                1_073_741_824,
            )
        };
        let reason = quest(4_000_000).expect("refused");
        assert!(
            reason.contains("256000000") && reason.contains("134217728"),
            "{reason}"
        );
        assert_eq!(quest(2_000_000), None);
        assert_eq!(
            records_refusal(4_000_000, &QualityTier::desktop(), 4 << 30, 4 << 30),
            None
        );
        // Per-view 48-byte records under a small binding; `quest-low`'s half records too.
        assert!(
            records_refusal(3_000_000, &QualityTier::desktop(), 134_217_728, 1 << 30).is_some()
        );
        assert_eq!(
            records_refusal(2_700_000, &QualityTier::quest_low(), 134_217_728, 1 << 30),
            None
        );
        // `max_buffer_size` below the binding limit is the limit hit.
        let reason = records_refusal(
            2_000_000,
            &QualityTier::quest_balanced(),
            1 << 30,
            100_000_000,
        )
        .expect("over max_buffer_size");
        assert!(reason.contains("max_buffer_size"), "{reason}");
    }

    #[test]
    fn the_headset_runs_b3_b5_and_b6_in_quest_balanced_against_the_quest_baseline() {
        // Q-406, Q-413: the headset's names, tier, P-4 limit and baseline file.
        let quest = On::Headset("print".to_owned());
        assert_eq!(quest.tier().name, "quest-balanced");
        assert_eq!(On::Desktop.tier().name, "desktop");
        assert_eq!(quest.p4_limit_ms(), 2.5);
        assert_eq!(On::Desktop.p4_limit_ms(), 0.4);
        assert_eq!(quest.baseline_file("B-3"), "B-3.quest.json");
        assert_eq!(On::Desktop.baseline_file("B-3"), "B-3.json");
        for name in ["B-3", "B-5", "B-6"] {
            assert!(quest.runner(name).is_some(), "{name}");
        }
        for name in ["B-1", "B-2", "B-4", "B-7", "B-8", "B-9", "B-10"] {
            assert!(quest.runner(name).is_none(), "{name}");
        }
        for name in ["B-1", "B-3", "B-5", "B-6", "B-9"] {
            assert!(On::Desktop.runner(name).is_some(), "{name}");
        }
    }

    /// The stored desktop baseline `<name>.json` (Q-103).
    fn stored(name: &str) -> Json {
        let path = baseline_dir().join(format!("{name}.json"));
        Json::parse(&fs::read_to_string(&path).expect("a stored baseline")).expect("well formed")
    }

    /// `budget` with its value 20 % worse than `base` under its rule.
    fn worse(mut budget: Budget, base: f64) -> Budget {
        budget.value = match budget.rule {
            Rule::AtLeast => 0.8 * base,
            _ => 1.2 * base,
        };
        budget
    }

    #[test]
    fn the_stored_desktop_baselines_still_compare() {
        // Q-103, #724: the desktop's setups, built as the benches build them, still match
        // the stored baselines' budgets, so a value 20 % worse than each is listed: the
        // headset's additions leave the desktop's comparison working, not vacuous.
        let desktop = QualityTier::desktop();
        let cases = [
            (
                "B-1",
                vec![b1_setup(
                    Json::str(SYNTHETIC_PERSISTENT),
                    1_000_000,
                    2,
                    &desktop,
                )],
            ),
            (
                "B-3",
                vec![b3_setup(
                    Json::str(SYNTHETIC_PERSISTENT),
                    1_000_000,
                    &desktop,
                )],
            ),
        ];
        for (name, setups) in cases {
            let base = stored(name);
            let old = base
                .get("budgets")
                .and_then(Json::as_array)
                .expect("budgets");
            assert_eq!(old.len(), setups.len(), "{name}");
            for (setup, o) in setups.into_iter().zip(old) {
                let value = o.get("value").and_then(Json::as_f64).expect("a value");
                let (id, measure, rule, limit) = match name {
                    "B-1" => ("P-2", "fps_min", Rule::AtLeast, 90.0),
                    _ => ("P-4", "sort_ms", Rule::AtMost, 0.4),
                };
                let b = worse(
                    Budget {
                        id,
                        setup,
                        measure,
                        rule,
                        limit,
                        value: 0.0,
                    },
                    value,
                );
                let profile = base.get("profile").and_then(Json::as_str).expect("profile");
                let gpu = base
                    .get("machine")
                    .and_then(|m| m.get("gpu"))
                    .and_then(Json::as_str)
                    .expect("gpu");
                assert_eq!(
                    compare_against(&[b], profile, gpu, &base).map(|r| r.len()),
                    Ok(1),
                    "{name}"
                );
            }
        }
        // B-5: the file sizes are the run's own, taken from the baseline.
        let base = stored("B-5");
        let old = base
            .get("budgets")
            .and_then(Json::as_array)
            .expect("budgets");
        assert_eq!(old.len(), B5_MB.len());
        for o in old {
            let bytes = o
                .get("setup")
                .and_then(|s| s.get("file_bytes"))
                .and_then(Json::as_f64)
                .expect("file_bytes") as u64;
            let value = o.get("value").and_then(Json::as_f64).expect("a value");
            let b = worse(
                Budget {
                    id: "P-3",
                    setup: b5_setup(Json::str("synthetic"), bytes, &desktop),
                    measure: "s_per_100mb",
                    rule: Rule::AtMost,
                    limit: 1.0,
                    value: 0.0,
                },
                value,
            );
            assert_eq!(compare(&[b], &base).len(), 1, "{bytes}");
        }
    }

    #[test]
    fn median_takes_the_middle_value() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
        assert_eq!(median(&mut []), 0.0);
    }

    /// Q-117 (§1.2 load option 4, D-13): a model over the device's budget is refused at
    /// `bench`'s upload with E-MEM-BUDGET, §1.15's text followed by N.
    #[test]
    #[ignore = "gpu"]
    fn gpu_upload_over_the_budget_gives_n_and_the_bytes() {
        let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
        let mut gpu = Gpu::new(On::Desktop, std::env::temp_dir()).expect("a GPU");
        // A budget of one byte: ten rows' static bytes, under 1 MB, cannot fit.
        gpu.dev.set_memory_budget(ftgsxr_render::MemoryBudget {
            bytes: 1,
            source: ftgsxr_render::BudgetSource::Tier,
        });
        let model = ftgsxr_testkit::random_model(5, 10, 0);
        let e = gpu
            .upload(&model, &gpu.tier, Some(Path::new("m.ftgs-if")))
            .err()
            .expect("refused");
        // §1.15's text, the bytes in MB (needed rounded up, the budget down), then N.
        assert_eq!(
            e,
            HeadlessError::Environment(
                "E-MEM-BUDGET: Needs 1 MB, budget 0 MB; reduce to budget or choose a tier (N = 10)"
                    .to_owned()
            )
        );
        assert_eq!(super::super::exit_code(&e), 2);
        assert!(gpu.dev.take_validation_messages().is_empty());
    }
}
