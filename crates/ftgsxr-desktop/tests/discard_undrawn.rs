// SPDX-License-Identifier: Apache-2.0

//! A window's surface texture dropped unpresented before the surface is reconfigured
//! (#408, §1.13 GPU validation, R-16, D-52, `gpu` lane). Bevy drops a surface texture it
//! holds unpresented when the window is resized or changes present mode, and wgpu then
//! discards it; Bevy reconfigures the surface next. That sequence leaves no validation
//! message on wgpu 29.0.4, with `Surface::discard` unchanged by D-52: the reconfigure
//! builds the swapchain's semaphores anew. This test guards it.
//!
//! The test makes the window with winit and drives the surface through wgpu directly, as
//! `present_undrawn` does: in each of several rounds, a surface texture acquired and
//! dropped, the surface reconfigured, then frames presented without and with a draw, each
//! acquire semaphore of the swapchain reused more than once. The layer guard (R-16,
//! Q-105) fails it on any validation message.
//!
//! Prerequisites: a desktop session (`DISPLAY` or `WAYLAND_DISPLAY`) and a Vulkan GPU.

use std::sync::Arc;
use std::time::Duration;

use ftgsxr_testkit::layer_messages;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::platform::wayland::EventLoopBuilderExtWayland;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

/// The window, once winit has made it.
#[derive(Default)]
struct Opened(Option<Arc<Window>>);

impl ApplicationHandler for Opened {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.0.is_none() {
            let attributes = Window::default_attributes()
                .with_title("ftgsxr discard_undrawn")
                .with_inner_size(winit::dpi::PhysicalSize::new(320, 240));
            self.0 = Some(Arc::new(
                event_loop.create_window(attributes).expect("a window"),
            ));
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

/// The surface's next texture.
fn acquire(surface: &wgpu::Surface) -> wgpu::SurfaceTexture {
    match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        _ => panic!("the surface gives a texture"),
    }
}

/// One frame: acquire the surface texture, clear it in a render pass when `draw`, present.
fn frame(surface: &wgpu::Surface, device: &wgpu::Device, queue: &wgpu::Queue, draw: bool) {
    let texture = acquire(surface);
    if draw {
        let view = texture.texture.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("discard_undrawn clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        }));
        queue.submit([encoder.finish()]);
    }
    texture.present();
}

/// A surface texture dropped with nothing drawn and no present, then the surface
/// reconfigured as Bevy does, leaves no validation message when the swapchain's images
/// are acquired and presented again (§1.13, R-16; D-52).
#[test]
#[ignore = "gpu"]
fn gpu_texture_dropped_unpresented_before_a_reconfigure_leaves_no_validation_message() {
    let layer = layer_messages(); // R-16, Q-105
    assert!(
        std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some(),
        "the gpu lane's window tests need a desktop session (DISPLAY or WAYLAND_DISPLAY)"
    );

    // The test harness runs tests off the main thread.
    let mut builder = EventLoop::builder();
    EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
    EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
    let mut event_loop = builder.build().expect("an event loop");
    let mut opened = Opened::default();
    for _ in 0..100 {
        let _ = event_loop.pump_app_events(Some(Duration::from_millis(10)), &mut opened);
        if opened.0.is_some() {
            break;
        }
    }
    let window = opened.0.clone().expect("winit made the window");

    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = wgpu::Backends::VULKAN;
    // §1.13: the layers and wgpu's validation in debug builds.
    desc.flags = wgpu::InstanceFlags::from_build_config().with_env();
    let instance = wgpu::Instance::new(desc);
    let surface = instance
        .create_surface(window.clone())
        .expect("a surface for the window");
    let adapter = bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
    }))
    .expect("a Vulkan adapter that can present to the window");
    let (device, queue) =
        bevy::tasks::block_on(adapter.request_device(&Default::default())).expect("a device");
    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width.max(1), size.height.max(1))
        .expect("the adapter supports the surface");
    config.present_mode = wgpu::PresentMode::AutoVsync;
    surface.configure(&device, &config);

    // Each round: a texture dropped unused, the surface reconfigured, then each acquire
    // semaphore of the swapchain reused more than once: undrawn frames, drawn ones,
    // undrawn again.
    for _ in 0..3 {
        let _ = event_loop.pump_app_events(Some(Duration::ZERO), &mut opened);
        drop(acquire(&surface));
        surface.configure(&device, &config);
        let plan = [false; 8].into_iter().chain([true; 8]).chain([false; 8]);
        for draw in plan {
            let _ = event_loop.pump_app_events(Some(Duration::ZERO), &mut opened);
            frame(&surface, &device, &queue, draw);
        }
    }
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device finishes its work");

    let messages = layer.take();
    assert!(
        messages.is_empty(),
        "validation messages from a texture dropped before a reconfigure: {messages:#?}"
    );
    drop(surface);
    drop(window);
}
