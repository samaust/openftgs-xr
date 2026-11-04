// SPDX-License-Identifier: Apache-2.0

//! Dv-1 (§3.4, `dv` lane): the app installs, launches and reaches the running session
//! within 10 s; the manifest's permissions and features of §2.5 are in the package dump
//! (Q-130: the permissions from the installed package's dump, the features from the
//! APK's badging). Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::time::Duration;

use dv_support::{apk, badging, device, install, launch, package_dump, stop, verdict};

/// §2.5's permissions.
const PERMISSIONS: [&str; 6] = [
    "com.oculus.permission.HAND_TRACKING",
    "com.oculus.permission.USE_SCENE",
    "com.oculus.permission.USE_ANCHOR_API",
    "horizonos.permission.HEADSET_CAMERA",
    "android.permission.CAMERA",
    "android.permission.INTERNET",
];

/// §2.5's features as `aapt2 dump badging` prints them: required ones as `uses-feature`
/// (head tracking with its version 1), the not-required ones as
/// `uses-feature-not-required`.
const FEATURES: [&str; 4] = [
    "uses-feature: name='android.hardware.vr.headtracking' version='1'",
    "uses-feature: name='com.oculus.feature.PASSTHROUGH'",
    "uses-feature-not-required: name='oculus.software.handtracking'",
    "uses-feature-not-required: name='android.hardware.camera2.any'",
];

#[test]
#[ignore = "dv"]
fn dv_dv1_installs_launches_and_runs_the_session_within_10_s() {
    let apk = apk();
    device();
    install(&apk);
    let start = launch("dv1");
    let result = verdict("dv1", start, Duration::from_secs(40));
    stop();
    let (pass, detail, after) = result.expect("a dv1 verdict line in logcat");
    assert!(pass, "dv1 FAIL {detail}");
    assert!(
        after <= Duration::from_secs(10),
        "the running session within 10 s of the launch: {after:?} ({detail})"
    );

    let dump = package_dump();
    let requested = dump
        .split_once("requested permissions:")
        .map(|(_, r)| r)
        .expect("the dump lists the requested permissions");
    for p in PERMISSIONS {
        assert!(
            requested.lines().any(|l| l.trim() == p),
            "permission {p} in the package dump"
        );
    }
    let badging = badging(&apk);
    for f in FEATURES {
        assert!(
            badging.lines().any(|l| l.trim() == f),
            "feature `{f}` in the badging:\n{badging}"
        );
    }
    let meta = [
        "sdkVersion:'32'",
        "targetSdkVersion:'34'",
        "native-code: 'arm64-v8a'",
    ];
    for m in meta {
        assert!(
            badging.lines().any(|l| l.trim() == m),
            "`{m}` in the badging"
        );
    }
}
