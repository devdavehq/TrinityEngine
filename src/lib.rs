//! TrinityEngine — game/engine crate.
//!
//! The binary (`Triengine`) is the editor+runtime. Pure-engine modules live
//! here too so small tools (e.g. `src/bin/pack.rs`) can reuse them without
//! pulling in the editor.

/// Virtual file system (disk, memory, or packed .pak archives).
pub mod vfs;

/// Runtime hot reload for Rust game code (feature `hotreload`).
#[cfg(feature = "hotreload")]
pub mod hotreload;

// pub struct EngineApp {
//     window: Option<Window>,
// }

// impl EngineApp {
//     pub fn new() -> Self {
//         Self { window: None }
//     }
// }

// impl ApplicationHandler for EngineApp {
//     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
//         // Create window when the app is resumed (ready to draw)
//         let window_attrs = Window::default_attributes()
//             .with_title("My Engine")
//             .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        
//         self.window = Some(event_loop.create_window(window_attrs).unwrap());
//     }

//     fn window_event(
//         &mut self,
//         event_loop: &ActiveEventLoop,
//         _window_id: WindowId,
//         event: WindowEvent,
//     ) {
//         match event {
//             WindowEvent::CloseRequested => {
//                 event_loop.exit();
//             }
//             WindowEvent::RedrawRequested => {
//                 // Request another redraw for continuous rendering
//                 if let Some(window) = &self.window {
//                     window.request_redraw();
//                 }
//             }
//             _ => {}
//         }
//     }

//     fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
//         // Called when the event loop is about to block waiting for events
//         // Good place for idle-time processing
//     }
// }

// pub fn run() {
//     let event_loop = EventLoop::new().unwrap();
//     event_loop.set_control_flow(ControlFlow::Poll);
    
//     let mut app = EngineApp::new();
//     event_loop.run_app(&mut app).unwrap();
// }