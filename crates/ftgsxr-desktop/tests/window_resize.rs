// SPDX-License-Identifier: Apache-2.0

//! A window resized by the window system (#409, §1.7 Window, §1.13 GPU validation, R-16,
//! D-53, `gpu` lane). When the X server resizes `play`'s window, its surface goes out of
//! date before the new size reaches Bevy's render world, and Bevy 0.19.1 reconfigured the
//! surface at a size the window had left: `vkCreateSwapchainKHR` got an extent outside the
//! surface's current one (`VUID-VkSwapchainCreateInfoKHR-pNext-07781`), once per resize. The
//! vendored bevy_render now reconfigures it at the surface's current extent (D-53).
//!
//! wgpu-hal's messenger callback drops that message by its id, so Q-105's in-process guard
//! never sees it; only the layer's own log does. Each test therefore runs itself again as a
//! child process whose layer writes its log through a settings file, and reads that log
//! once the child exits (Q-113). The child runs a Bevy app whose camera draws to its window
//! and resizes the window through winit, not through Bevy's `Window`, so that Bevy learns
//! each size from the window system, as when the user or the window manager resizes it.
//! Once a resize has settled, Bevy's window, the render world's window and the swapchain
//! texture all have the size asked for, so no frame stays at an old size. The child also
//! holds the in-process guard (R-16, Q-105), and any wgpu validation error ends its app with
//! an error.
//!
//! Resizes seconds apart, growing and shrinking, must leave no message. In a burst 20 to
//! 150 ms apart, a resize can land between Bevy's query of the surface and wgpu-core's own
//! inside wgpu's `configure`; that race's `07781` is accepted (D-53, owner answer on #409)
//! and counted, and any other message fails the test.
//!
//! Prerequisites: an X11 display (`DISPLAY`; XWayland on a Wayland session) and a Vulkan
//! GPU. The child runs on X11 only, where the surface reports a fixed current extent.

use std::fs::{self, File};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use bevy::app::AppExit;
use bevy::ecs::system::NonSendMarker;
use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::render::error_handler::{RenderErrorHandler, RenderErrorPolicy};
use bevy::render::view::{ExtractedWindows, prepare_windows};
use bevy::render::{Render, RenderApp};
use bevy::window::{PresentMode, PrimaryWindow, WindowResolution};
use bevy::winit::{WINIT_WINDOWS, WinitPlugin};
use ftgsxr_testkit::{layer_messages, scratch_dir};

use crate::common;
use common::{layer_log, layer_settings};

/// Set in the child's environment: run the resized app.
const CHILD: &str = "FTGSXR_WINDOW_RESIZE_CHILD";

/// How long the child may run.
const LIMIT: Duration = Duration::from_secs(120);

/// The window's first physical size.
const START: (u32, u32) = (640, 480);

/// A resize: how long after the previous one (or after the first presented frame) it is
/// asked for, and the physical size asked for.
type Resize = (Duration, (u32, u32));

/// Grow, shrink, grow and shrink, seconds apart.
const SECONDS_APART: &[Resize] = &[
    (Duration::from_millis(1500), (800, 600)),
    (Duration::from_millis(2000), (560, 420)),
    (Duration::from_millis(2000), (720, 540)),
    (Duration::from_millis(2000), (600, 450)),
];

/// A burst 20 to 150 ms apart, growing and shrinking, that ends on a shrink.
const BURST: &[Resize] = &[
    (Duration::from_millis(1500), (680, 510)),
    (Duration::from_millis(20), (620, 470)),
    (Duration::from_millis(150), (700, 520)),
    (Duration::from_millis(50), (580, 440)),
    (Duration::from_millis(80), (660, 500)),
    (Duration::from_millis(20), (540, 400)),
];

/// A wait of at least this long before a resize means the previous one has settled.
const SETTLED: Duration = Duration::from_millis(1000);

/// How long after the last resize the child checks it and exits.
const LAST_SETTLE: Duration = Duration::from_millis(2000);

/// The header of the layer's message for a swapchain extent outside the surface's.
const EXTENT_OUTSIDE: &str = "VUID-VkSwapchainCreateInfoKHR-pNext-07781(";

/// What the render world saw of the primary window in one frame, after `prepare_windows`.
#[derive(Clone, Copy, Debug)]
struct Frame {
    /// The window's physical size as extracted.
    window: (u32, u32),
    /// The swapchain texture's size, when the frame acquired one.
    texture: Option<(u32, u32)>,
}

/// What both worlds share.
#[derive(Default)]
struct Seen {
    frames: Vec<Frame>,
    failures: Vec<String>,
}

#[derive(Resource, Clone, Default)]
struct Shared(Arc<Mutex<Seen>>);

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Seen> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The main world's progress through its resizes.
#[derive(Resource)]
struct Driver {
    resizes: &'static [Resize],
    /// When the previous step happened: the first presented frame, then each resize.
    last: Option<Instant>,
    /// The next resize.
    next: usize,
    /// The size last asked for.
    target: Option<(u32, u32)>,
}

/// Render world: keeps the primary window's extracted size and its swapchain texture's.
fn record(windows: Res<ExtractedWindows>, shared: Res<Shared>) {
    let Some(window) = windows.primary.and_then(|e| windows.windows.get(&e)) else {
        return;
    };
    let texture = window.swap_chain_texture.as_ref().map(|t| {
        let size = t.texture.size();
        (size.width, size.height)
    });
    shared.lock().frames.push(Frame {
        window: (window.physical_width, window.physical_height),
        texture,
    });
}

/// Whether, once a resize to `target` has settled, Bevy's window, the render world's
/// window and the swapchain texture all have that size; a failure is kept otherwise.
fn check_settled(target: (u32, u32), window: &Window, shared: &Shared) {
    let mut seen = shared.lock();
    let last = seen.frames.last().copied();
    let bevy = (window.physical_width(), window.physical_height());
    let ok =
        bevy == target && last.is_some_and(|f| f.window == target && f.texture == Some(target));
    if !ok {
        seen.failures.push(format!(
            "after the resize to {target:?} settled: Bevy's window {bevy:?}, the render \
             world's last frame {last:?}"
        ));
    }
}

/// Main world, on the event loop's thread (winit's windows live there): walks the
/// resizes, asking winit for each size, checks each settled resize, then exits.
fn drive(
    _main_thread: NonSendMarker,
    mut driver: ResMut<Driver>,
    windows: Query<(Entity, &Window), With<PrimaryWindow>>,
    shared: Res<Shared>,
    mut exit: MessageWriter<AppExit>,
) {
    let Ok((entity, window)) = windows.single() else {
        return;
    };
    let now = Instant::now();
    let Some(last) = driver.last else {
        // Starts once the render world has a swapchain texture for the window.
        if shared.lock().frames.iter().any(|f| f.texture.is_some()) {
            driver.last = Some(now);
        }
        return;
    };
    let waited = now - last;
    let Some(&(delay, size)) = driver.resizes.get(driver.next) else {
        if waited >= LAST_SETTLE {
            if let Some(target) = driver.target {
                check_settled(target, window, &shared);
            }
            exit.write(AppExit::Success);
        }
        return;
    };
    if waited < delay {
        return;
    }
    if delay >= SETTLED
        && let Some(target) = driver.target
    {
        check_settled(target, window, &shared);
    }
    WINIT_WINDOWS.with_borrow(|winit_windows| {
        let winit_window = winit_windows
            .get_window(entity)
            .expect("bevy_winit holds the primary window");
        let _ = winit_window.request_inner_size(winit::dpi::PhysicalSize::new(size.0, size.1));
    });
    driver.target = Some(size);
    driver.next += 1;
    driver.last = Some(now);
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// The child: the Bevy app resized through winit by `resizes` (see the module's
/// documentation).
fn resized_app(resizes: &'static [Resize]) {
    let layer = layer_messages(); // R-16, Q-105: what wgpu-hal's messenger forwards
    let shared = Shared::default();
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .build()
            // The guard holds the process's logger.
            .disable::<LogPlugin>()
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "ftgsxr window_resize".into(),
                    present_mode: PresentMode::AutoVsync,
                    resolution: WindowResolution::new(START.0, START.1),
                    ..default()
                }),
                ..default()
            })
            // The test harness runs the test off the main thread.
            .set(WinitPlugin {
                run_on_any_thread: true,
            }),
    )
    // As bevy_render 0.19.1's default `RenderErrorHandler` (src/error_handler.rs, MIT OR
    // Apache-2.0) does, but printing the error, which its log line would lose to the guard's
    // logger.
    .insert_resource(RenderErrorHandler(|error, main_world, _| {
        eprintln!("wgpu error: {:?}: {}", error.ty, error.description);
        main_world.write_message(AppExit::error());
        RenderErrorPolicy::StopRendering
    }))
    .insert_resource(shared.clone())
    .insert_resource(Driver {
        resizes,
        last: None,
        next: 0,
        target: None,
    })
    .add_systems(Startup, camera)
    .add_systems(Update, drive);
    app.sub_app_mut(RenderApp)
        .insert_resource(shared.clone())
        .add_systems(Render, record.after(prepare_windows));

    let exit = app.run();

    let seen = shared.lock();
    let mismatched = seen
        .frames
        .iter()
        .filter(|f| f.texture.is_some_and(|t| t != f.window))
        .count();
    eprintln!(
        "{} render frames, {mismatched} with a swapchain texture of another size than the \
         window's",
        seen.frames.len()
    );
    assert_eq!(exit, AppExit::Success, "the resized app's exit");
    assert!(
        seen.failures.is_empty(),
        "{} resize(s) not settled:\n{}",
        seen.failures.len(),
        seen.failures.join("\n")
    );
    drop(seen);
    drop(layer);
}

/// One child run of the test `test` of this file, named by its full path in the crate's one
/// test binary (#413): its exit status, its output, and the layer's log as `layer_log`
/// reads it (whether the layer reported itself active, and its errors and warnings).
fn run_child(test: &str) -> (Option<ExitStatus>, String, (bool, Vec<String>)) {
    assert!(
        std::env::var_os("DISPLAY").is_some(),
        "the resize tests need an X11 display (DISPLAY; XWayland on a Wayland session)"
    );
    let home = scratch_dir(env!("CARGO_TARGET_TMPDIR"), crate::test_file_stem(), test);
    let log = home.join("validation.log");
    let settings = layer_settings(&home, &log);
    let output = home.join("child-output.txt");
    let file = File::create(&output).expect("the child's output file");
    let path = format!("{}::{test}", crate::test_file_stem());
    let mut child = Command::new(std::env::current_exe().expect("the test binary's path"))
        .args([
            path.as_str(),
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(CHILD, "1")
        .env("VK_LAYER_SETTINGS_PATH", &settings)
        // X11, where the surface reports its current extent.
        .env_remove("WAYLAND_DISPLAY")
        .stdout(Stdio::from(file.try_clone().expect("the output file")))
        .stderr(Stdio::from(file))
        .spawn()
        .expect("the test binary runs again as the child");
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("the child can be waited on") {
            break Some(status);
        }
        if start.elapsed() > LIMIT {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(50));
    };
    let text = fs::read_to_string(&output).unwrap_or_default();
    (status, text, layer_log(&log))
}

/// The failures of a child run, given the layer's messages that count: the child's exit
/// and whether it ran its test, and, R-16, §1.13 (Q-105, Q-113), the layer's messages and,
/// in a debug build, its status message.
fn failures(status: Option<ExitStatus>, text: &str, active: bool, messages: &[String]) -> String {
    let mut failures = Vec::new();
    if !status.is_some_and(|s| s.success()) {
        failures.push(format!("the child failed ({status:?}):\n{text}"));
    } else if !text.contains("test result: ok. 1 passed") {
        failures.push(format!("the child ran no test:\n{text}"));
    }
    if cfg!(debug_assertions) && !active {
        failures.push("the Khronos validation layer did not report itself active".into());
    }
    if !messages.is_empty() {
        failures.push(format!(
            "{} validation layer message(s): {messages:#?}",
            messages.len()
        ));
    }
    failures.join("\n\n")
}

/// The child's summary line, for the test's output.
fn summary(text: &str) -> &str {
    text.lines()
        .rev()
        .find(|l| l.contains("render frames"))
        .unwrap_or("")
}

/// #409 (§1.7 Window, §1.13, R-16, D-53): a window that the window system resizes seconds
/// apart, growing and shrinking, leaves no validation message in the layer's own log, and
/// each resize settles at the size asked for.
#[test]
#[ignore = "gpu"]
fn gpu_window_resized_a_second_or_more_apart_leaves_no_validation_message() {
    if std::env::var_os(CHILD).is_some() {
        return resized_app(SECONDS_APART);
    }
    let (status, text, (active, messages)) =
        run_child("gpu_window_resized_a_second_or_more_apart_leaves_no_validation_message");
    let failed = failures(status, &text, active, &messages);
    assert!(failed.is_empty(), "{failed}");
    eprintln!("{}", summary(&text));
}

/// #409 (§1.7 Window, §1.13, R-16, D-53): a window that the window system resizes in a
/// burst 20 to 150 ms apart settles at the last size asked for, and the layer's own log
/// holds no message but the resize race's `07781`, which D-53 accepts.
#[test]
#[ignore = "gpu"]
fn gpu_window_resized_in_a_burst_settles_with_no_message_but_the_resize_race() {
    if std::env::var_os(CHILD).is_some() {
        return resized_app(BURST);
    }
    let (status, text, (active, messages)) =
        run_child("gpu_window_resized_in_a_burst_settles_with_no_message_but_the_resize_race");
    // D-53: a resize that lands between Bevy's query of the surface and wgpu-core's own
    // inside `configure` is accepted (owner answer, #409); counted, so that one per resize,
    // the stale size D-53 fixes, shows in the output.
    let (race, others): (Vec<String>, Vec<String>) = messages
        .into_iter()
        .partition(|m| m.starts_with(EXTENT_OUTSIDE));
    let failed = failures(status, &text, active, &others);
    assert!(failed.is_empty(), "{failed}");
    eprintln!(
        "{}; {} resize race message(s) over {} resizes (accepted, D-53)",
        summary(&text),
        race.len(),
        BURST.len()
    );
}
