// SPDX-License-Identifier: Apache-2.0

//! The `dv` lane's host-side driver (§2.7, §3.4, D-28, Q-122): the prebuilt debug APK
//! installed with `adb install -r`, the app launched with the test's name in the intent
//! extra `ftgsxr-test`, its verdict read from logcat under the tag `ftgsxr`, and files
//! pushed into and pulled from `dv/` under the app-external files directory.
//!
//! Prerequisites: `adb` on the `PATH` (or under `$ANDROID_HOME/platform-tools`), one
//! headset listed by `adb devices` (or `ANDROID_SERIAL` naming it), awake; the debug
//! APK built by `tools/build-quest.sh debug` (cargo profile `dv`, Q-140), its path in
//! `FTGSXR_QUEST_APK` (no test
//! builds it). `aapt2` from `$ANDROID_HOME/build-tools/34.0.0` reads the APK's
//! features. Run with `--test-threads 1`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

/// The application id (D-36).
pub const PACKAGE: &str = "io.github.samaust.ftgsxr";
/// The launchable activity (§2.5).
pub const ACTIVITY: &str = "io.github.samaust.ftgsxr/.MainActivity";
/// The environment variable holding the debug APK's path.
pub const APK_VAR: &str = "FTGSXR_QUEST_APK";
/// The logcat tag (§1.13).
pub const TAG: &str = "ftgsxr";
/// The device directory of the pushed and pulled files (Q-122).
pub const DV_DIR: &str = "/sdcard/Android/data/io.github.samaust.ftgsxr/files/dv";

fn android_home() -> PathBuf {
    std::env::var_os("ANDROID_HOME").map_or_else(
        || PathBuf::from(std::env::var_os("HOME").expect("HOME is set")).join("Android/Sdk"),
        PathBuf::from,
    )
}

fn tool(name: &str, sub: &str) -> PathBuf {
    let p = android_home().join(sub).join(name);
    if p.exists() { p } else { PathBuf::from(name) }
}

fn run(mut cmd: Command) -> Output {
    let out = cmd.output().unwrap_or_else(|e| panic!("{cmd:?} runs: {e}"));
    assert!(
        out.status.success(),
        "{cmd:?} failed: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// The `adb` program the driver runs.
pub fn adb_path() -> PathBuf {
    tool("adb", "platform-tools")
}

/// `adb` with `args`; panics when it fails.
pub fn adb(args: &[&str]) -> String {
    let mut cmd = Command::new(tool("adb", "platform-tools"));
    cmd.args(args);
    String::from_utf8_lossy(&run(cmd).stdout).into_owned()
}

/// The debug APK named by `FTGSXR_QUEST_APK`.
pub fn apk() -> PathBuf {
    let p = PathBuf::from(std::env::var_os(APK_VAR).unwrap_or_else(|| {
        panic!("{APK_VAR} names the debug APK (tools/build-quest.sh debug prints it)")
    }));
    assert!(p.is_file(), "{APK_VAR}: no file at {}", p.display());
    p
}

/// Waits for a listed headset and wakes it (the power key's wake-up, then the
/// proximity sensor reported as worn).
pub fn device() {
    let listed = adb(&["devices"]);
    assert!(
        listed.lines().skip(1).any(|l| l.ends_with("\tdevice")),
        "a headset is listed by adb: {listed}"
    );
    adb(&["shell", "input", "keyevent", "KEYCODE_WAKEUP"]);
    adb(&[
        "shell",
        "am",
        "broadcast",
        "-a",
        "com.oculus.vrpowermanager.prox_close",
    ]);
}

/// Installs the APK (`adb install -r`).
pub fn install(apk: &Path) {
    let out = adb(&["install", "-r", &apk.to_string_lossy()]);
    assert!(out.contains("Success"), "adb install: {out}");
}

/// Clears logcat, force-stops the app and launches it with `test` in the extra; the
/// instant of the launch.
pub fn launch(test: &str) -> Instant {
    adb(&["logcat", "-c"]);
    let start = Instant::now();
    adb(&[
        "shell",
        "am",
        "start",
        "-S",
        "-n",
        ACTIVITY,
        "--es",
        "ftgsxr-test",
        test,
    ]);
    start
}

/// The device's system property `key` (`adb shell getprop`), trimmed.
pub fn getprop(key: &str) -> String {
    adb(&["shell", "getprop", key]).trim().to_owned()
}

/// Clears logcat, force-stops the app and launches it with the benchmark `name` in the
/// extra `ftgsxr-bench` and `headset` in `ftgsxr-bench-headset` (Q-406, Q-418); the
/// instant of the launch. The device's shell reads the command line, so `headset` is
/// quoted.
pub fn launch_bench(name: &str, headset: &str) -> Instant {
    assert!(
        !headset.contains('\''),
        "a fingerprint without quotes: {headset}"
    );
    adb(&["logcat", "-c"]);
    let start = Instant::now();
    adb(&[
        "shell",
        "am",
        "start",
        "-S",
        "-n",
        ACTIVITY,
        "--es",
        "ftgsxr-bench",
        name,
        "--es",
        "ftgsxr-bench-headset",
        &format!("'{headset}'"),
    ]);
    start
}

/// `launch_bench` with the session benchmark's model, a file in `dv/`, in the extra
/// `ftgsxr-bench-model` (Q-T731.4).
pub fn launch_session_bench(name: &str, headset: &str, model: &str) -> Instant {
    assert!(
        !headset.contains('\'') && !model.contains('\''),
        "a fingerprint and a model name without quotes"
    );
    adb(&["logcat", "-c"]);
    let start = Instant::now();
    adb(&[
        "shell",
        "am",
        "start",
        "-S",
        "-n",
        ACTIVITY,
        "--es",
        "ftgsxr-bench",
        name,
        "--es",
        "ftgsxr-bench-headset",
        &format!("'{headset}'"),
        "--es",
        "ftgsxr-bench-model",
        &format!("'{model}'"),
    ]);
    start
}

/// B-1's every-row-persistent kind (Q-102, Q-420): `random_model(seed, n, degree)` with
/// each transient row made persistent as E3's persistent rows are drawn, opacity in
/// \[0.6, 0.99\] and duration 2–4 T_τ, each mapped from the row's own uniform opacity
/// draw in \[0.002, 0.99\], so the model stays the same for the same arguments.
pub fn every_row_persistent(seed: u64, n: usize, degree: u8) -> ftgsxr_core::Model {
    let mut m = ftgsxr_testkit::random_model(seed, n, degree);
    let span = m.time.tau1 - m.time.tau0;
    for i in 0..n {
        if m.duration[i] >= 2.0 * span {
            continue;
        }
        let u = ((f64::from(m.opacity[i]) - 0.002) / (0.99 - 0.002)).clamp(0.0, 1.0);
        m.opacity[i] = (0.6 + 0.39 * u) as f32;
        m.duration[i] = (2.0 + 2.0 * u) * span;
    }
    m
}

/// The runner's line for the benchmark `name` (Q-406): `(done, detail)`, or `None` when
/// no line arrives within `timeout` or the app's process ends without one (killed for
/// its memory, say: not a size the device refused, Q-413).
pub fn bench_line(name: &str, start: Instant, timeout: Duration) -> Option<(bool, String)> {
    let prefix = format!("ftgsxr-bench: {name} ");
    let find = || {
        log().iter().find_map(|line| {
            let rest = line.split_once(&prefix)?.1;
            let (word, detail) = rest.split_once(' ').unwrap_or((rest, ""));
            Some((word == "DONE", detail.to_owned()))
        })
    };
    while start.elapsed() < timeout {
        if let Some(found) = find() {
            return Some(found);
        }
        if start.elapsed() > Duration::from_secs(15) && !running() {
            return find();
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    None
}

/// Whether the app's process is running.
pub fn running() -> bool {
    !adb_ok(&["shell", "pidof", PACKAGE]).trim().is_empty()
}

/// `adb` with `args`, its standard output whatever its exit status.
fn adb_ok(args: &[&str]) -> String {
    Command::new(adb_path())
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

/// Stops the app.
pub fn stop() {
    adb(&["shell", "am", "force-stop", PACKAGE]);
}

/// The app's log so far (the lines under `ftgsxr`).
pub fn log() -> Vec<String> {
    adb(&["logcat", "-d", "-s", &format!("{TAG}:*")])
        .lines()
        .map(str::to_owned)
        .collect()
}

/// The verdict of `test` (§3.4): `(pass, detail)` and its arrival since `start`, or `None`
/// when no verdict line arrives within `timeout`.
pub fn verdict(test: &str, start: Instant, timeout: Duration) -> Option<(bool, String, Duration)> {
    let prefix = format!("ftgsxr-test: {test} ");
    while start.elapsed() < timeout {
        for line in log() {
            if let Some(rest) = line.split_once(&prefix).map(|(_, r)| r) {
                let (word, detail) = rest.split_once(' ').unwrap_or((rest, ""));
                return Some((word == "PASS", detail.to_owned(), start.elapsed()));
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    None
}

/// Pushes `local` into `dv/` on the device as `name`.
pub fn push(local: &Path, name: &str) {
    adb(&["shell", "mkdir", "-p", DV_DIR]);
    adb(&[
        "push",
        &local.to_string_lossy(),
        &format!("{DV_DIR}/{name}"),
    ]);
}

/// Pulls `dv/<name>` from the device to `local`.
pub fn pull(name: &str, local: &Path) {
    adb(&[
        "pull",
        &format!("{DV_DIR}/{name}"),
        &local.to_string_lossy(),
    ]);
}

/// The installed package's dump (`dumpsys package`).
pub fn package_dump() -> String {
    adb(&["shell", "dumpsys", "package", PACKAGE])
}

/// The APK's badging (`aapt2 dump badging`), which lists its features; the installed
/// package's dump does not (Q-130).
pub fn badging(apk: &Path) -> String {
    let mut cmd = Command::new(tool("aapt2", "build-tools/34.0.0"));
    cmd.args(["dump", "badging"]).arg(apk);
    String::from_utf8_lossy(&run(cmd).stdout).into_owned()
}

/// Removes `dv/` on the device, so no file of an earlier run stands for this one.
pub fn clear() {
    adb(&["shell", "rm", "-rf", DV_DIR]);
}

/// The layer the debug APK bundles (Q-123).
pub const LAYER: &str = "VK_LAYER_KHRONOS_validation";

/// Android's GPU debug-layer settings for the app (§1.13 GPU validation, Q-123), reset
/// when dropped, also when the test fails.
pub struct DebugLayers;

impl DebugLayers {
    /// Sets the three settings before a launch (Q-123).
    pub fn enable() -> DebugLayers {
        adb(&[
            "shell",
            "settings",
            "put",
            "global",
            "enable_gpu_debug_layers",
            "1",
        ]);
        adb(&[
            "shell",
            "settings",
            "put",
            "global",
            "gpu_debug_app",
            PACKAGE,
        ]);
        adb(&[
            "shell",
            "settings",
            "put",
            "global",
            "gpu_debug_layers",
            LAYER,
        ]);
        DebugLayers
    }
}

impl Drop for DebugLayers {
    fn drop(&mut self) {
        for key in [
            "enable_gpu_debug_layers",
            "gpu_debug_app",
            "gpu_debug_layers",
        ] {
            adb(&["shell", "settings", "delete", "global", key]);
        }
    }
}

/// The lines of `tag` in logcat so far.
pub fn tagged(tag: &str) -> Vec<String> {
    adb(&["logcat", "-d", "-s", &format!("{tag}:*")])
        .lines()
        .filter(|l| !l.starts_with("---------"))
        .map(str::to_owned)
        .collect()
}

/// The layer's checks after a run with [`DebugLayers`] (§1.13, Q-123): Android announced
/// the layer for the app, and no line arrived under the tag `VALIDATION`. Read before
/// the settings are reset.
pub struct Validation {
    /// Whether `GraphicsEnvironment` named the layer in its debug layer list.
    pub announced: bool,
    /// The lines under `VALIDATION`.
    pub messages: Vec<String>,
}

impl Validation {
    /// Reads the layer's announcement and messages from logcat.
    pub fn read() -> Validation {
        let announced = tagged("GraphicsEnvironment")
            .iter()
            .any(|l| l.contains("Vulkan debug layer list") && l.contains(LAYER));
        Validation {
            announced,
            messages: tagged("VALIDATION"),
        }
    }

    /// Fails unless the layer was loaded and said nothing.
    pub fn assert_clean(&self) {
        assert!(self.announced, "Android loaded {LAYER} for the app");
        assert!(
            self.messages.is_empty(),
            "no validation message ({} lines), first: {:?}",
            self.messages.len(),
            self.messages.iter().take(6).collect::<Vec<_>>()
        );
    }
}

/// `pm VERB` (`grant` or `revoke`) of each of `permissions` for the app; a failure (a
/// permission the system grants at install, say) is only printed, after `test`'s name.
pub fn pm(test: &str, verb: &str, permissions: &[&str]) {
    for p in permissions {
        let out = Command::new(adb_path())
            .args(["shell", "pm", verb, PACKAGE, p])
            .output();
        match out {
            Ok(o) if o.status.success() => {}
            Ok(o) => println!(
                "{test}: pm {verb} {p}: {}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            ),
            Err(e) => println!("{test}: pm {verb} {p}: {e}"),
        }
    }
}

/// Whether a logcat line containing `line` appears within `timeout`, polled every 200 ms.
pub fn wait_for(line: &str, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if log().iter().any(|l| l.contains(line)) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}
