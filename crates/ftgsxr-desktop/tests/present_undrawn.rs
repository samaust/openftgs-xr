// SPDX-License-Identifier: Apache-2.0

//! A window presented in frames that nothing drew to (#401, §1.13 GPU validation, R-16,
//! Q-243, `gpu` lane). In `ftgsxr play`'s first frames, while pipelines compile, Bevy
//! acquires the window's surface texture and presents it with no command having used it.
//! wgpu 29.0.4 then presented the image in the `UNDEFINED` layout
//! (`VUID-VkPresentInfoKHR-pImageIndices-01430`) and left its acquire semaphore signaled
//! for the next acquire (`VUID-vkAcquireNextImageKHR-semaphore-01286`). The vendored
//! wgpu-core's present now readies such a texture first (Q-243).
//!
//! The test makes the window with winit and presents through wgpu directly, so it doesn't
//! depend on when Bevy's pipelines happen to compile: frames presented without a draw,
//! then drawn frames, then undrawn again, each image of the swapchain more than once. The
//! layer guard (R-16, Q-105) fails it on any validation message.
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
                .with_title("ftgsxr present_undrawn")
                .with_inner_size(winit::dpi::PhysicalSize::new(320, 240));
            self.0 = Some(Arc::new(
                event_loop.create_window(attributes).expect("a window"),
            ));
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

/// One frame: acquire the surface texture, clear it in a render pass when `draw`, present.
fn frame(surface: &wgpu::Surface, device: &wgpu::Device, queue: &wgpu::Queue, draw: bool) {
    let texture = match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        _ => panic!("the surface gives a texture"),
    };
    if draw {
        let view = texture.texture.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("present_undrawn clear"),
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

/// Frames presented with nothing drawn, between drawn ones, leave no validation message
/// (§1.13, R-16; Q-243).
#[test]
#[ignore = "gpu"]
fn gpu_frames_presented_without_a_draw_leave_no_validation_message() {
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

    // Each image of the swapchain more than once: undrawn, drawn, then undrawn again.
    let plan = [false; 8].into_iter().chain([true; 4]).chain([false; 8]);
    for draw in plan {
        let _ = event_loop.pump_app_events(Some(Duration::ZERO), &mut opened);
        frame(&surface, &device, &queue, draw);
    }
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device finishes its work");

    let messages = layer.take();
    assert!(
        messages.is_empty(),
        "validation messages from undrawn presents: {messages:#?}"
    );
    drop(surface);
    drop(window);
}
