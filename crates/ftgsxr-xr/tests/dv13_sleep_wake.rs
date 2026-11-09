// SPDX-License-Identifier: Apache-2.0

//! Dv-13 (§3.4, `dv` lane, #229): the headset sleeps and wakes; the session goes idle and
//! returns to running (§1.8, D-18). The runner's `dv13` logs `dv13: running` once the
//! session runs and prints PASS once the session has left Running and come back. The
//! driver sleeps the headset with the proximity sensor reported off and Android's sleep
//! key, and wakes it with the wake key and the sensor reported on (Meta's scriptable-testing
//! page, Android's `KeyEvent`, source log 142 and 143). The sensor broadcasts make
//! the run hold whether or not someone wears the headset. Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::time::{Duration, Instant};

use dv_support::{adb, apk, device, install, launch, log, stop, verdict, wait_for};

/// How long the headset is left asleep.
const ASLEEP: Duration = Duration::from_secs(10);
/// How long the system is left to settle after a wake before the launch.
const SETTLE: Duration = Duration::from_secs(5);

fn sleep_headset() {
    adb(&[
        "shell",
        "am",
        "broadcast",
        "-a",
        "com.oculus.vrpowermanager.prox_far",
    ]);
    adb(&["shell", "input", "keyevent", "KEYCODE_SLEEP"]);
}

/// Wakes the headset, also when the test fails.
struct Awake;
impl Drop for Awake {
    fn drop(&mut self) {
        device();
    }
}

/// Wakes the headset and waits until Android reports it awake, then lets the system
/// settle: a launch right after a wake can come before the shell gives the app focus.
fn wake_and_settle() {
    device();
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10)
        && !adb(&["shell", "dumpsys", "power"]).contains("mWakefulness=Awake")
    {
        std::thread::sleep(Duration::from_millis(200));
    }
    std::thread::sleep(SETTLE);
}

#[test]
#[ignore = "dv"]
fn dv_dv13_sleep_and_wake_the_session_goes_idle_and_runs_again() {
    let apk = apk();
    wake_and_settle();
    install(&apk);
    let _awake = Awake;
    let start = launch("dv13");
    assert!(
        wait_for("dv13: running", Duration::from_secs(40)),
        "the session runs before the sleep: {:?}",
        log()
    );
    sleep_headset();
    std::thread::sleep(ASLEEP);
    let power = adb(&["shell", "dumpsys", "power"]);
    assert!(
        power.contains("mWakefulness=Asleep"),
        "the headset sleeps (dumpsys power)"
    );
    device();
    let result = verdict("dv13", start, Duration::from_secs(120));
    let lines = log();
    stop();
    let (pass, detail, _) = result.unwrap_or_else(|| panic!("a dv13 verdict: {lines:?}"));
    assert!(pass, "dv13 FAIL {detail}");
    let states: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("XR session state"))
        .collect();
    assert!(
        states.iter().any(|l| l.contains("Idle")),
        "the session went idle: {states:?}"
    );
    let errors: Vec<&String> = lines.iter().filter(|l| l.contains("E-XR-")).collect();
    assert!(errors.is_empty(), "no XR error: {errors:?}");
}
