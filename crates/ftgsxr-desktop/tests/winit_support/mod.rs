// SPDX-License-Identifier: Apache-2.0

//! The process's one winit event loop, for the tests that make a window with winit and
//! drive its surface through wgpu directly (D-52, `gpu` lane). winit builds one event loop
//! per process, and the crate's tests share one test binary (#413), whose harness runs each
//! test on a thread of its own, so the loop lives on a thread of its own too: it makes the
//! windows the tests ask for and pumps its events when they ask, one request at a time.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::platform::wayland::EventLoopBuilderExtWayland;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

/// A request to the event loop's thread, answered on its channel.
enum Request {
    /// A window of this title and physical size.
    Window(String, (u32, u32), Sender<Arc<Window>>),
    /// The loop's events pumped once, waiting at most this long for one.
    Pump(Duration, Sender<()>),
}

/// The window asked for, made in the next callback that has the active event loop.
#[derive(Default)]
struct Pending(Option<(String, (u32, u32), Sender<Arc<Window>>)>);

impl Pending {
    fn open(&mut self, event_loop: &ActiveEventLoop) {
        if let Some((title, (width, height), reply)) = self.0.take() {
            let attributes = Window::default_attributes()
                .with_title(title)
                .with_inner_size(winit::dpi::PhysicalSize::new(width, height));
            let window = event_loop.create_window(attributes).expect("a window");
            let _ = reply.send(Arc::new(window));
        }
    }
}

impl ApplicationHandler for Pending {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.open(event_loop);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.open(event_loop);
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

/// The event loop's thread: builds the loop off the main thread and serves the requests.
fn serve(requests: Receiver<Request>) {
    let mut builder = EventLoop::builder();
    EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
    EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
    let mut event_loop = builder.build().expect("an event loop");
    let mut pending = Pending::default();
    for request in requests {
        match request {
            Request::Window(title, size, reply) => {
                pending.0 = Some((title, size, reply));
                for _ in 0..100 {
                    let _ =
                        event_loop.pump_app_events(Some(Duration::from_millis(10)), &mut pending);
                    if pending.0.is_none() {
                        break;
                    }
                }
                // Unanswered, the reply's channel drops and the test fails on it.
                pending.0 = None;
            }
            Request::Pump(timeout, reply) => {
                let _ = event_loop.pump_app_events(Some(timeout), &mut pending);
                let _ = reply.send(());
            }
        }
    }
}

/// The channel to the event loop's thread, which starts on the first request.
fn requests() -> &'static Sender<Request> {
    static REQUESTS: OnceLock<Sender<Request>> = OnceLock::new();
    REQUESTS.get_or_init(|| {
        let (send, receive) = channel();
        thread::Builder::new()
            .name("winit event loop".into())
            .spawn(move || serve(receive))
            .expect("the event loop's thread");
        send
    })
}

/// A window of this title and physical size, made by the process's event loop.
pub(crate) fn window(title: &str, size: (u32, u32)) -> Arc<Window> {
    let (reply, answer) = channel();
    requests()
        .send(Request::Window(title.to_owned(), size, reply))
        .expect("the event loop's thread runs");
    answer.recv().expect("winit made the window")
}

/// The event loop's events pumped once, waiting at most `timeout` for one.
pub(crate) fn pump(timeout: Duration) {
    let (reply, answer) = channel();
    requests()
        .send(Request::Pump(timeout, reply))
        .expect("the event loop's thread runs");
    answer.recv().expect("the event loop's thread runs");
}
