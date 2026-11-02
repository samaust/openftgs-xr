# The Quest app in `ftgsxr-quest`

How the Meta Quest 3 build is packaged and run in M-3: the build script and its cargo profiles, the Gradle project and the manifest, the OpenXR loader, the Vulkan validation layer of the debug build, signing, `ftgsxr_quest::main`, the device test runner and its logcat protocol, and the `dv` lane's host-side driver with its adb paths. The spec items are §2.5 Android packaging, §2.1 `ftgsxr_quest::main`, §1.13 (logging, GPU validation), §1.15, §3.4, D-18, D-19 and D-36; this doc explains them and never overrides them.

## The build

`tools/build-quest.sh [debug|release]` builds `libftgsxr_quest.so` for arm64-v8a with `cargo ndk -t arm64-v8a -o android/app/src/main/jniLibs build … -p ftgsxr-quest` (NDK 25.1), then runs Gradle (`assembleDebug` or `assembleRelease`) in `android/` and prints the APK's path.

- **release** builds the library with `--release`, as §2.5 writes.
- **debug** builds it with the cargo profile `dv` (Q-140): it inherits `dev` with opt-level 3 and debug assertions on, so wgpu's validation stays while the app holds Dv-2's 72 Hz, which the unoptimised dev profile cannot (39 to 56 Hz). The debug APK is the `dv` lane's.

On the host the crate builds empty: every module is Android's (§2.7's host checks).

## The Gradle project and the manifest

`android/app/build.gradle.kts` sets the application id `io.github.samaust.ftgsxr` (D-36), `minSdk` 32, `targetSdk` 34, arm64-v8a only, and packages the library with legacy packaging so GameActivity finds it by name. The activity is the Kotlin class `MainActivity : GameActivity` (D-18), which loads `ftgsxr_quest` and does nothing else; bevy\_mod\_openxr did not force the `native-activity` backend, so §2.5's NativeActivity fallback was not needed. GameActivity is pinned at 2.0.2, the version android-activity 0.6.0's glue matches, with the AppCompat and Core libraries its POM leaves to the app. The shim's `Shim.kt` joins with the milestones that need its methods; M-3's JNI calls (`jni_calls`) read the launch intent's string extra and the cache directory (Q-15), which android-activity does not wrap.

`AndroidManifest.xml` declares every entry of §2.5's list; Dv-1 checks them on the installed app, the permissions from `dumpsys package` and the features, SDK levels and ABI from `aapt2 dump badging` of the APK, since Horizon OS's package dump lists no features (Q-130).

## The loader

The Khronos OpenXR loader comes from Maven, `org.khronos.openxr:openxr_loader_for_android`, pinned in `build.gradle.kts` (D-19, at least 1.0.34 as Meta requires). From 1.1.46 the AAR ships `jni/arm64-v8a`, so Gradle packages `libopenxr_loader.so` without a prefab consumer.

## The validation layer

The debug APK bundles the Khronos validation layer `libVkLayer_khronos_validation.so` for arm64-v8a, from a Vulkan-ValidationLayers Android release (Apache-2.0), which the build script downloads once, checks against its SHA-256 and places under `android/app/src/debug/jniLibs/` (Q-123); the release APK has none. The XR crate creates the Vulkan instance through the runtime with no layers and no debug messenger, so the layer is enabled from outside: the `dv` driver sets Android's GPU debug-layer settings for the app before Dv-2's launch and resets them afterwards, also when the test fails. The layer's messages reach logcat under the tag `VALIDATION`.

The validation runs on the instance the vendored bevy\_mod\_openxr creates (Q-139, D-30, `ftgsxr-xr/session.md`): its patches remove the messages a Vulkan 1.1 instance gave and keep wgpu's indirect-call validation off on Android, so the indirect draws stay.

## Signing

Debug builds sign with Gradle's default debug keystore, outside the repository; release signing is the owner's, so `assembleRelease` gives an unsigned APK here. No keystore is committed.

## `ftgsxr_quest::main`

The `#[bevy_main]` entry point, in order:

1. reads the launch intent's extra `ftgsxr-test` (the device test, if any);
2. makes the Quest's `AppPaths` from the internal and external files directories and the cache directory (Q-15) and loads `settings.toml`;
3. logs to logcat under the tag `ftgsxr` at the settings' level, raised to info for a device test so its verdict line shows (§1.13); tracing's events reach android\_logger through tracing's `log` feature, with Bevy's `LogPlugin` disabled;
4. runs Dv-3 or Dv-4 at once if named, before any XR session, on the headless renderer's own device, and exits (Q-122, Q-146);
5. builds the app: Bevy's default plugins with no window (§2.5: the XR session drives the frames), winit's runner set to poll so xrWaitFrame alone paces it, `XrPlugin` through `add_xr` with `SortMode::PerView` (Q-120, Q-121), and `SplatPlugin` at the settings' tier, `quest-balanced` when the name is unknown;
6. exits 2 when the OpenXR instance could not be created, and later when the session cannot be created (E-XR-RUNTIME, §1.15, logged once by the XR crate);
7. spawns every exchange file of the models directory as a model instance at the identity pose, the provisional harness M-4's `AppPlugin` replaces (Q-129);
8. adds the named device test and runs.

The crate names only `ftgsxr-app` and `ftgsxr-xr`, with `bevy` and `bevy_mod_xr` directly (§2.1); what it needs from `ftgsxr-core`, `ftgsxr-io` and `ftgsxr-render` comes through `ftgsxr_app`'s and `ftgsxr_xr`'s re-exports (Q-129, Q-133, Q-137), and Dv-3 runs `ftgsxr_app::sort_check` (Q-146).

## The device test runner and the logcat protocol

The launch intent's string extra `ftgsxr-test` names one §3.4 test; the runner prints its verdict as one logcat line under the tag `ftgsxr`:

```
ftgsxr-test: <name> PASS|FAIL <detail>
```

An unknown name prints FAIL at once. What each test does on the device and what the host judges:

- **dv1**: PASS once the session is running. The host judges the 10 s from its launch, and the manifest (Q-130).
- **dv2**: loads the probe the host pushed as `dv/probe.safetensors`, keeps it 1 m ahead of the left eye once both eyes are splat cameras, and draws it with both for 10 s; PASS when the frames, paced by xrWaitFrame, number between 95 % and 105 % of 72 Hz over that time, so a session at another of the runtime's rates fails, and the first eye's statistics counted the probe (Q-138, Q-125, Q-137). The host fails it on any `VALIDATION` line, checks that Android loaded the layer for the app, fails on any adapter warning about a splat camera, and checks the pixels: 3 s after both eyes draw, a screenshot of the compositor's view must show, in each eye's half, at least 200 pixels of the central fifth that differ from that region's median by more than 40 (Q-141), since the eye images cannot be read back on the device.
- **dv3** and **dv4**: the headless tests above; the host pushes R-4's models (dv3, checked by `ftgsxr_app::sort_check`) or the probes and requests (dv4), and for dv4 pulls the written `ftgs-renders` files and applies E15's metrics on the host, since the APK does not link the testkit (Q-122, Q-146).
- **dv12**: selects 72 Hz then 90 Hz, steps through every foveation level and sets the render scale to 0.8, waiting for the session to run again; PASS once the sequence completes, the host judging the rest from `display`'s log lines (`ftgsxr-xr/display.md`).
- **dv13**: logs `dv13: running` once the session runs, the host's cue to put the headset to sleep, and PASS once the session has left Running and returned to it, with the states it saw (`ftgsxr-xr/session.md`).

Files the host pushes and the device writes live in `dv/` under the app's external files directory.

## The host-side driver and adb paths

The `dv` lane's tests are `crates/ftgsxr-xr/tests/dv*.rs`, sharing `dv_support` (§2.7, §3.4, D-28). They need `adb` on the `PATH` or under `$ANDROID_HOME/platform-tools`, one awake headset (`ANDROID_SERIAL` when several are listed), the prebuilt debug APK's path in `FTGSXR_QUEST_APK` (no test builds it), `aapt2` from `$ANDROID_HOME/build-tools/34.0.0`, and `--test-threads 1`. The driver installs with `adb install -r`, clears logcat, force-stops the app and launches it with the test's name in the extra, reads `adb logcat -d -s ftgsxr:*` for the verdict, pushes and pulls `dv/` files, and takes Dv-2's screenshot with `adb exec-out screencap -p`, decoded with `image` (a dev-dependency of `ftgsxr-xr`).
