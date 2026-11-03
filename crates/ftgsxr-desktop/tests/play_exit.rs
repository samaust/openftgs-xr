// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr play`'s exit in its first frames (#376, §1.7 Command line, §2.4 Threads, Q-229,
//! `gpu` lane). With bevy's `multi_threaded`, render pipelines compile on the async compute
//! pool; a process that exited while one was compiling left the worker inside the NVIDIA
//! Vulkan driver, whose exit-time teardown crashed it (SIGSEGV or SIGABRT). `AppPlugin`
//! now holds the `AppExit` until no pipeline compiles, at most Q-229's cap.
//!
//! Each run sends one SIGINT, which bevy's Ctrl-C handler turns into an `AppExit` with
//! code 130, early after the window appears, when pipelines are still compiling, and
//! expects an exit by that code, not a death by signal. One signal only: a second Ctrl-C
//! makes bevy call `process::exit` at once, which is not this path. A compile first queued
//! in the render frame that runs after the release could still race; pipelines queue
//! when something first draws, which a quiet scene doesn't do in that frame.
//!
//! Prerequisites: a desktop session (`DISPLAY` or `WAYLAND_DISPLAY`) and a Vulkan GPU.

use std::io::{BufRead, BufReader};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use ftgsxr_testkit::{fixtures_dir, scratch_dir};

/// The line bevy_winit logs once the window exists.
const WINDOW_LINE: &str = "Creating new window";

/// How long a run may take to show its window, and to exit after the signal.
const LIMIT: Duration = Duration::from_secs(30);

fn scratch(name: &str) -> PathBuf {
    scratch_dir(env!("CARGO_TARGET_TMPDIR"), env!("CARGO_CRATE_NAME"), name)
}

/// `ftgsxr play model` with its home and XDG directories under `home`, on the session's
/// display, standard error piped.
fn spawn_play(home: &Path, model: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_ftgsxr"))
        .arg("play")
        .arg(model)
        .args(["--log-level", "info"])
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_CACHE_HOME", home.join("cache"))
        .env("XDG_DATA_HOME", home.join("data"))
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the binary runs")
}

/// Waits for `child` to exit, at most `LIMIT`; kills it and returns `None` past that.
fn wait_bounded(child: &mut Child) -> Option<ExitStatus> {
    let start = Instant::now();
    while start.elapsed() < LIMIT {
        if let Some(s) = child.try_wait().expect("the child can be waited on") {
            return Some(s);
        }
        thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    let _ = child.wait();
    None
}

/// One run: SIGINT `delay` after the window line; the exit status and standard error.
fn run_once(home: &Path, model: &Path, delay: Duration) -> (Option<ExitStatus>, String) {
    let mut child = spawn_play(home, model);
    let stderr = child.stderr.take().expect("standard error is piped");
    let (tx, rx) = mpsc::channel::<String>();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut log = String::new();
    let start = Instant::now();
    let mut window = false;
    while start.elapsed() < LIMIT {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(line) => {
                window |= line.contains(WINDOW_LINE);
                log.push_str(&line);
                log.push('\n');
                if window {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    if window {
        thread::sleep(delay);
        let sent = Command::new("kill")
            .args(["-INT", &child.id().to_string()])
            .status()
            .expect("kill runs");
        assert!(sent.success(), "kill -INT reached the player");
    }
    let status = wait_bounded(&mut child);
    reader.join().expect("the reader thread ends");
    log.extend(rx.try_iter().map(|l| l + "\n"));
    assert!(
        window,
        "the player showed no window within {LIMIT:?} (status {status:?}):\n{log}"
    );
    (status, log)
}

/// #376 (§1.7 Command line, Q-229): `play` asked to exit in its first frames, while its
/// render pipelines compile, exits by its own code (130 for Ctrl-C), never by a signal.
#[test]
#[ignore = "gpu"]
fn gpu_play_interrupted_in_its_first_frames_exits_by_its_code() {
    assert!(
        std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some(),
        "the gpu lane's play tests need a desktop session (DISPLAY or WAYLAND_DISPLAY)"
    );
    let model = fixtures_dir().join("probes/P1.safetensors");
    let delays_ms = [0, 0, 0, 50, 100, 200];
    let mut failures = Vec::new();
    for (i, ms) in delays_ms.iter().enumerate() {
        let home = scratch(&format!("interrupted-{i}"));
        let (status, log) = run_once(&home, &model, Duration::from_millis(*ms));
        let ok = status.is_some_and(|s| s.signal().is_none() && s.code() == Some(130));
        if !ok {
            let tail: Vec<&str> = log.lines().rev().take(8).collect();
            failures.push(format!(
                "run {i} (SIGINT {ms} ms after the window): {status:?}; last lines:\n{}",
                tail.into_iter().rev().collect::<Vec<_>>().join("\n")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} runs did not exit by code 130:\n{}",
        failures.len(),
        delays_ms.len(),
        failures.join("\n\n")
    );
}
