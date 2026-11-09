# The Quest app in `ftgsxr-quest`

How the Meta Quest 3 build is packaged and run: the build script and its cargo profiles, the Gradle project and the manifest, the OpenXR loader, the Vulkan validation layer of the debug build, signing, `ftgsxr_quest::main`, the device test runner and its logcat protocol, and the `dv` lane's host-side driver with its adb paths (M-3); the shim, `QuestPlatform`, `main` on `AppPlugin`, and Dv-8's and Dv-10's runners and drivers (M-5, #454, #458, #463); passthrough on in `main`, and Dv-6's, Dv-7's and Dv-9's runners and drivers (M-6, #568, #570, #571); the shim's camera methods, `QuestPlatform`'s camera frames and head pose, and Dv-11's runner and driver (M-7, #667, #668). The spec items are §2.5 Android packaging and The shim, §2.1 `ftgsxr_quest::main` and `QuestPlatform`, §1.5 Sources, §1.13 (logging, GPU validation), §1.15, §3.4, D-15, D-18, D-19, D-36 and D-49; this doc explains them and never overrides them.

## The build

`tools/build-quest.sh [debug|release]` builds `libftgsxr_quest.so` for arm64-v8a with `cargo ndk -t arm64-v8a -P 32 -o android/app/src/main/jniLibs build … -p ftgsxr-quest` (NDK 25.1), then runs Gradle (`assembleDebug` or `assembleRelease`) in `android/` and prints the APK's path.

- **release** builds the library with `--release`, as §2.5 writes.
- **debug** builds it with the cargo profile `dv` (Q-140): it inherits `dev` with opt-level 3 and debug assertions on, so wgpu's validation stays while the app holds Dv-2's 72 Hz, which the unoptimised dev profile cannot (39 to 56 Hz). The debug APK is the `dv` lane's.

`-P 32` builds against NDK platform 32, the `minSdk` (D-18), whose `libaaudio.so` kira's cpal backend links, so the Quest has sound (#454, `ftgsxr-app/audio.md`); CI's Quest check passes the same.

On the host the crate builds empty but for the `tree` module, built for the host tests too (§2.7's host checks).

## The Gradle project and the manifest

`android/app/build.gradle.kts` sets the application id `io.github.samaust.ftgsxr` (D-36), `minSdk` 32, `targetSdk` 34, arm64-v8a only, and packages the library with legacy packaging so GameActivity finds it by name. The activity is the Kotlin class `MainActivity : GameActivity` (D-18), which loads `ftgsxr_quest` and does nothing else; bevy\_mod\_openxr did not force the `native-activity` backend, so §2.5's NativeActivity fallback was not needed. GameActivity is pinned at 2.0.2, the version android-activity 0.6.0's glue matches, with the AppCompat and Core libraries its POM leaves to the app. M-3's JNI calls (`jni_calls`) read the launch intent's string extra and the cache directory (Q-15), which android-activity does not wrap; the shim (below) holds the rest.

`AndroidManifest.xml` declares every entry of §2.5's list; Dv-1 checks them on the installed app, the permissions from `dumpsys package` and the features, SDK levels and ABI from `aapt2 dump badging` of the APK, since Horizon OS's package dump lists no features (Q-130).

## The shim

`Shim.kt` is a Kotlin `object` holding §2.5's static methods and nothing else of the JNI surface; its `internal` methods `attach`, `detach`, `onPickResult` and `onPermissionResult` are `MainActivity`'s hooks, not static and not reachable from JNI (D-15, D-49, Q-319): `pickDocuments` and `pollPick`, `listTree`, `openDocument`, `requestPermission` and `pollPermission`, `startCamera`, `pollCameraThumb` and `stopCamera` (below). Kotlin never calls into Rust: `MainActivity` attaches itself to the shim in `onCreate`, and its `onActivityResult` and `onRequestPermissionsResult` fill the shim's maps by request id, which Rust polls. A pick takes persistable read permission on every URI it returns. `listTree` lists a tree's (or a directory document's) children through `DocumentsContract`, one `name\tsize\tmime\turi` row each, the child's URI built on the tree's so a directory's own listing works. `requestPermission` answers granted at once when `checkSelfPermission` already grants, else shows Android's dialog (Q-276). The shim keeps Android's request codes small and positive (the documentation asks for at least 0 and states no upper bound) and maps them to Rust's request ids, so the ids may take any value.

**The camera** (Q-386). The camera code is private members of `Shim.kt`, behind §2.5's three methods. `startCamera` opens the left passthrough camera (Meta's camera source 0, position 0) through Camera2 with a 1280 × 960 `YUV_420_888` `ImageReader` and a 60 Hz repeating request on a handler thread of its own, and returns 0 when opening, 1 when no passthrough camera is listed, 2 when a camera permission is missing, 3 when the open failed. `acquireLatestImage` keeps the newest frame and a frame arriving during a reduction is dropped. Each frame is reduced to the camera thumbnail by the 20 × 20 block means of the Y plane and the 10 × 10 block means of each chroma plane, converted by BT.709's full-range matrix and clamped. The frame's data space is logged once (API 33 on) as the evidence for the full range; on the Quest 3 it is full range (Dv-11). `pollCameraThumb` returns the array once, null until the next: the 9216 RGB floats, the timestamp as whole seconds and fraction, then, when the characteristics give them, 17 raw floats: the lens pose's translation and rotation as Camera2 gives them, the first four intrinsics (fx, fy, cx, cy), the pre-correction active array's rectangle and the stream's size. The shim reads them and does no lens arithmetic (D-15, #687); it logs them once per start. An error after a 0 closes the camera, and `MainActivity.onPause` closes it through an `internal` hook (Q-319); nothing on the Kotlin side reopens it, Rust's start does (`ftgsxr-xr/light.md`). No Gradle dependency is added.

`shim.rs` is the JNI adapter: it loads the shim's class once through the activity's class loader, since a natively attached thread's `FindClass` sees only the system classes, and turns a Java exception into an `Err` with its text, cleared.

## `QuestPlatform`

The `Platform` and `DocumentProvider` ports of §2.1 over the shim; cloning shares its state, so it is its own document provider. Request ids are a counter, and a pick or a permission is polled by the caller (the library and `tracking`, once per frame on the main thread).

- **Picks.** A picked file is a `ModelRef::Document`, and so is a picked tree, which the library adds as a `documents` source; a tree's URI is logged as `picked tree <uri>`, which Dv-10 needs (Q-273).
- **Permissions.** Each `Permission` stands for the Android permissions of §2.5's manifest (light estimation's camera for both camera permissions); one denied is a denial.
- **Documents.** `list_tree` turns `listTree`'s rows into entries, sorted by name, dot-names skipped, each one's location its row's URI (`tree.rs`, Q-297): a directory row is listed once more and is a `Checkpoint` when it holds `manifest.json` and `params.safetensors`, else a `Dir`; a file is `classify_name`'s kind, `Other` files kept, since `DocumentSource`'s checkpoint copy looks for the manifest among them. `open` turns `openDocument`'s detached descriptor into a `File`. A picked checkpoint is copied from the tree into one download-cache entry before it loads, as Q-61 directs; the copy is `ftgsxr-io`'s `DocumentSource` (Q-295).
- **Camera frames.** `QuestPlatform` is its own `CameraFrames` (Q-386, Q-392). `start` maps `startCamera`'s codes to texts beginning with §1.15's codes: 1 to `E-XR-UNSUPPORTED: no passthrough camera`, 2 to `E-XR-PERMISSION: the camera permissions are not granted`, 3 and a JNI failure to the shim's own text with no code. `latest` polls once per call, parses the array (9218 or 9235 floats, another length a warning; `camera.rs`, built on the host with its tests), converting the raw lens into the pose in OpenXR's axes and the four field-of-view angles, taken against the pre-correction array's centred crop to the stream's aspect (Q-386); characteristics that give no camera (a focal length not above zero, an empty array or stream, a zero rotation, a value not finite) are no lens. It keeps the newest thumbnail until a newer one, and the first thumbnail after a start logs the converted lens beside the shim's raw line; `stop` drops it, so a restart's thumbnails are new frames.
- **The head pose** (Q-387). A system in `Last` writes `XrHands.head`, already in the splats' stage space, into the platform; `latest` composes it with the thumbnail's lens pose into `Camera::from_stage_pose` with `Fov::Angles` at 64 × 48, so `CameraThumb.camera` is in stage space, `None` when the shim gave no pose. The head pose is the current frame's, not the capture's. `Camera` and `Fov` come from `ftgsxr_app`'s re-exports, for the band table's reason below (Q-393).

`Entry`, `EntryKind`, `SourceSpec` and `classify_name` come from `ftgsxr_app`'s re-exports, since the band table keeps `ftgsxr-quest` from naming `ftgsxr-io` (Q-296).

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
5. builds the app: Bevy's default plugins with no window (§2.5: the XR session drives the frames), winit's runner set to poll so xrWaitFrame alone paces it, and `XrPlugin` through `add_xr` with `SortMode::PerView`, `hands` on and `passthrough` on, so the passthrough layer, the environment depth and the anchors run where the runtime offers them (Q-120, Q-121, `ftgsxr-xr/passthrough.md`);
6. exits 2 at once when the OpenXR instance could not be created (E-XR-RUNTIME, §1.15, logged once by the XR crate), before the app's first frame, so no warm-up runs yet (Q-325). When the session cannot be created later (a session-created message with no primary reference space), the app ends instead through an `AppExit` with code 2, so Q-236's hold and the settings' save at exit see it, and the runner drops the app, which joins the adapter's warm-up threads (Q-381), before the process exits (#660);
7. inserts the `Headset` marker, so panel mode, spatial audio and D-11's automatic fit apply (Q-266), then adds `AppPlugin` over `QuestPlatform` at the settings' tier, `quest-balanced` when the name is unknown, with no initial models and no scene: the library lists the models directory and the user's sources (§1.5, D-49). The `SettingsRes` loaded above is left as it is (Q-163). Q-129's provisional harness, which spawned every model file at the identity pose, is retired;
8. without the app-external files directory, warns in the log and a toast that the models and screenshots stay in internal storage (Q-15);
9. adds the named device test and runs; once `App::run` returns, with the app dropped, an error exit ends the process with its code (E-XR-RUNTIME's 2), and a success returns from `main` (#660).

The crate names only `ftgsxr-app` and `ftgsxr-xr`, with `bevy` and `bevy_mod_xr` directly (§2.1); what it needs from `ftgsxr-core`, `ftgsxr-io` and `ftgsxr-render` comes through `ftgsxr_app`'s and `ftgsxr_xr`'s re-exports (Q-129, Q-133, Q-137, Q-296), and Dv-3 runs `ftgsxr_app::sort_check` (Q-146).

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
- **dv8**: spawns an entity named `ftgsxr-test: dv8`, which starts the clauses inside `ftgsxr_xr::tracking`, since they need its crate-private trackers, locates and haptic output; the XR crate prints the verdict line (Q-308, `ftgsxr-xr/tracking.md`). As Q-272 reads the row without a person: the joints and the aim state over a window, a synthetic pinch writing `Select`, and the grab pulse on the active controller, with the rate seen under each frequency hint reported (Q-309).
- **dv6**, **dv7** and **dv9**: each spawns an entity named `ftgsxr-test: dv6` (`dv7`, `dv9`), which starts the clauses inside `ftgsxr_xr`'s `passthrough`, `depth` or `anchors`, since they need those modules' crate-private records, provider and store; the XR crate prints the verdict line (Q-308, Q-346). Dv-6 switches to passthrough and back and checks the layer list, the layer's and the feature's state and the blend mode (`ftgsxr-xr/passthrough.md`); Dv-7 turns occlusion on in passthrough mode and checks two depth frames every frame, then a clean stop (`ftgsxr-xr/depth.md`); Dv-9 creates and persists an anchor on its first launch and locates and erases it on its second (`ftgsxr-xr/anchors.md`). All three passed on the Quest 3 (#568, #570, #571). On Dv-9's second launch the runner also spawns an entity named `ftgsxr-anchor: <detail>`, carrying the first launch's detail from the intent extra `ftgsxr-anchor`.
- **dv10**: lists the library's first source, the models directory, its first clause holding when the probe the host pushed is listed; given a granted tree's URI in the extra `ftgsxr-tree`, lists the tree through `listTree`, reads its first model's header through `openDocument`, and adds the tree as a `documents` source and loads its first model through the library, a checkpoint through Q-61's cache copy. Without the extra the tree clause is reported as not run, not as FAIL (Q-273).
- **dv11**: spawns an entity named `ftgsxr-test: dv11`, which starts the clauses in `ftgsxr_xr::light` (Q-308, Q-390; `ftgsxr-xr/light.md`). Dv-11 passed clause 1 on the Quest 3 at 56.6 Hz.
- **dv13**: logs `dv13: running` once the session runs, the host's cue to put the headset to sleep, and PASS once the session has left Running and returned to it, with the states it saw (`ftgsxr-xr/session.md`).

Files the host pushes and the device writes live in `dv/` under the app's external files directory.

## The host-side driver and adb paths

The `dv` lane's tests are `crates/ftgsxr-xr/tests/dv*.rs`, sharing `dv_support` (§2.7, §3.4, D-28). They need `adb` on the `PATH` or under `$ANDROID_HOME/platform-tools`, one awake headset (`ANDROID_SERIAL` when several are listed), the prebuilt debug APK's path in `FTGSXR_QUEST_APK` (no test builds it), `aapt2` from `$ANDROID_HOME/build-tools/34.0.0`, and `--test-threads 1`. The driver installs with `adb install -r`, clears logcat, force-stops the app and launches it with the test's name in the extra, reads `adb logcat -d -s ftgsxr:*` for the verdict, pushes and pulls `dv/` files, and takes Dv-2's screenshot with `adb exec-out screencap -p`, decoded with `image` (a dev-dependency of `ftgsxr-xr`).

- **Dv-8's driver** grants `com.oculus.permission.HAND_TRACKING` with `pm grant` before the launch, so no dialog waits for a person; a failed grant (a permission the system grants at install) is only printed.
- **Dv-6's driver** launches `dv6`, requires the verdict's detail to show both cycles, the layer and the feature paused in VR mode and the style applied, and fails on any error-level line of the app's own log over the switches, a failed end of frame or passthrough call (PRD #565's spike check 3); the system's error lines naming passthrough or the frame's end are printed for the record only, since the system's camera pipeline logs some without any fault of the app.
- **Dv-7's driver** grants `com.oculus.permission.USE_SCENE` with `pm grant` before the launch, as Dv-8's grants hand tracking, launches `dv7`, and requires the provider's stop and the views' textures kept across frames (spike check 2) in the detail. It sets the GPU debug-layer settings for the run, as Dv-2's driver does, and requires the validation layer loaded and silent, so a format, usage or layout mismatch of the conversion pass fails it (spike check 1, Q-123, Q-356, Q-360).
- **Dv-11's driver** grants `android.permission.CAMERA` and `horizonos.permission.HEADSET_CAMERA` with `pm grant` before the launch, launches `dv11`, prints the cue to cover the cameras when the device logs `dv11: cover`, requires clause 1 and reports clause 2, run or not run (Q-273, Q-390), prints the shim's logged data space and range (Q-386), and revokes the permissions afterwards.
- **Dv-9's driver** grants `com.oculus.permission.USE_ANCHOR_API` the same way, then launches `dv9` twice, restarting the app: it takes the first verdict's `uuid=… pose=…` part and passes it to the second launch in the extra `ftgsxr-anchor`, and requires the second verdict to report the anchor found by its UUID, within Q-346's distance and erased (spike checks 4 and 5; Q-346).
- **Dv-10's driver** pushes P1 as `models/dv10-probe.safetensors`, removed afterwards, also when the test fails. The tree is one the owner granted once through the app's picker; its logged URI is passed from `FTGSXR_DV_TREE` in the extra `ftgsxr-tree`. Without the variable the tree clause is not run, which the test prints, and the ticket stays unverified (`docs/agents/issue-tracker.md`, "Unverified tickets"; Q-273).
