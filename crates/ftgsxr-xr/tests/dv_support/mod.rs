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
