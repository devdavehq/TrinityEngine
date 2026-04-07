// src/main.rs
// Engine entry point. Wires all systems together.
// Uses winit 0.30's ApplicationHandler trait (no old EventLoop::run() closure).
//
// ── winit 0.30 pattern ───────────────────────────────────────────────────────
// • Create EventLoop, call run_app() with a struct that impls ApplicationHandler.
// • resumed()      → called when the OS says the app is ready (create window here).
// • window_event() → called for keyboard, resize, close, redraw requests.
// • about_to_wait()→ idle — good place to request the next frame.
//
// ── wgpu 29 changes ──────────────────────────────────────────────────────────
// • Renderer::new() takes Arc<Window> instead of &Window.
// • Renderer has no lifetime parameter.

mod assets;
mod camera;
mod components;
mod input;
mod physics;
mod renderer;
mod scene;
mod scripting;
mod systems;

use std::sync::Arc;
use std::collections::HashMap;

use assets::{AssetStore, Mesh};
use camera::Camera3D;
use components::RigidBody;
use input::InputState;
use physics::physics_system;
use renderer::Renderer;
use scene::SceneManager;
use scripting::ScriptEngine;
use systems::scripting_system;

use hecs::World;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

// ── GameApp ───────────────────────────────────────────────────────────────────
// Owns all engine state. Created before the event loop starts.
// Fields are Option<> for anything that needs a window to initialise.
struct GameApp {
    // GPU renderer — None until resumed() fires and we have a window.
    renderer:   Option<Renderer>,
    // winit window wrapped in Arc so wgpu Surface can hold a reference.
    window:     Option<Arc<Window>>,

    input:      InputState,
    world:      World,
    meshes:     AssetStore<Mesh>,
    mesh_cache: HashMap<String, assets::Handle<Mesh>>,
    camera:     Camera3D,
    scripts:    ScriptEngine,
    scene_mgr:  SceneManager,

    // Hot-reload receivers — Option because they're set up after the watcher starts.
    script_watcher: Option<std::sync::mpsc::Receiver<String>>,
    scene_watcher:  Option<std::sync::mpsc::Receiver<String>>,

    last_frame: std::time::Instant,
}

impl GameApp {
    fn new() -> Self {
        // Camera starts behind and above the scene, looking at origin.
        let mut camera = Camera3D::new(1280.0 / 720.0);
        camera.position = glam::Vec3::new(0.0, 4.0, 8.0);
        camera.target   = glam::Vec3::ZERO;

        Self {
            renderer:       None,
            window:         None,
            input:          InputState::new(),
            world:          World::new(),
            meshes:         AssetStore::new(),
            mesh_cache:     HashMap::new(),
            camera,
            scripts:        ScriptEngine::new(),
            scene_mgr:      SceneManager::new("scenes/main.scene"),
            script_watcher: None,
            scene_watcher:  None,
            last_frame:     std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for GameApp {
    // resumed() fires when the OS tells us we can draw.
    // On desktop this fires once right away. On Android/iOS it may fire multiple times.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create the OS window.
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("My Engine — PBR")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Could not create window"),
        );

        // Update camera aspect ratio now that we know the window size.
        let phys = window.inner_size();
        self.camera.aspect = phys.width as f32 / phys.height as f32;

        // Build the GPU renderer (blocking — we are on the main thread).
        let renderer = pollster::block_on(Renderer::new(Arc::clone(&window)));

        // Check if GPU is low-end and print a note.
        println!("[Engine] GPU: {:?}", renderer.adapter_info.name);

        // Register Lua API and load startup scripts.
        self.scripts.register_api().expect("Lua API registration failed");
        self.scripts.load_script("scripts/player.lua").ok();
        self.scripts.load_script("scripts/enemy.lua").ok();

        // Build the initial scene from scenes/main.scene.
        self.scene_mgr
            .build(&mut self.world, &mut self.meshes, &mut self.mesh_cache)
            .expect("Failed to load main.scene");

        // Start file watchers for hot reload.
        self.script_watcher = Some(self.scripts.start_watching("scripts"));

        // Separate watcher thread for scene files.
        let (scene_tx, scene_rx) = std::sync::mpsc::channel::<String>();
        {
            let tx = scene_tx.clone();
            std::thread::spawn(move || {
                use notify::{Watcher, RecursiveMode, recommended_watcher};
                use std::path::Path;
                let (ntx, nrx) = std::sync::mpsc::channel();
                let mut watcher = recommended_watcher(move |res| { let _ = ntx.send(res); })
                    .expect("Scene watcher failed");
                watcher.watch(Path::new("scenes"), RecursiveMode::Recursive).ok();
                loop {
                    match nrx.recv() {
                        Ok(Ok(event)) => {
                            for path in event.paths {
                                let s = path.to_string_lossy().to_string();
                                if s.ends_with(".scene") {
                                    if tx.send(s).is_err() { return; }
                                }
                            }
                        }
                        _ => break,
                    }
                }
            });
        }
        self.scene_watcher = Some(scene_rx);

        self.window   = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event:      WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event: ke, .. } => {
                self.input.handle_key(ke.physical_key, ke.state == ElementState::Pressed);
            }

            WindowEvent::Resized(new_size) => {
                self.camera.aspect = new_size.width as f32 / new_size.height as f32;
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size);
                }
            }

            // RedrawRequested fires when we call window.request_redraw().
            // This is where one full game-loop iteration happens.
            WindowEvent::RedrawRequested => {
                // ── Delta time ─────────────────────────────────────────────
                let now = std::time::Instant::now();
                let dt  = now.duration_since(self.last_frame).as_secs_f32().min(0.05);
                self.last_frame = now;

                // ── Hot reload: scripts ────────────────────────────────────
                if let Some(rx) = &self.script_watcher {
                    while let Ok(path) = rx.try_recv() {
                        match self.scripts.reload_script(&path) {
                            Ok(_)  => println!("[Hot] Script reloaded: {}", path),
                            Err(e) => eprintln!("[Hot] Script error {}: {}", path, e),
                        }
                    }
                }

                // ── Hot reload: scenes ─────────────────────────────────────
                if let Some(rx) = &self.scene_watcher {
                    while let Ok(path) = rx.try_recv() {
                        println!("[Hot] Scene changed: {}", path);
                        match self.scene_mgr.build(
                            &mut self.world,
                            &mut self.meshes,
                            &mut self.mesh_cache,
                        ) {
                            Ok(_)  => println!("[Hot] Scene rebuilt"),
                            Err(e) => eprintln!("[Hot] Scene error: {}", e),
                        }
                    }
                }

                // ── Reset physics ground flag ──────────────────────────────
                // Must happen before physics_system so entities that walk off
                // edges fall correctly on the next frame.
                for (_, body) in self.world.query_mut::<&mut RigidBody>().iter() {
                    body.on_ground = false;
                }

                // ── Systems ────────────────────────────────────────────────
                scripting_system(&mut self.world, &self.scripts, &self.input, dt);
                self.scripts.drain_destroys(&mut self.world);
                let _collisions = physics_system(&mut self.world, dt);

                // ── Render ─────────────────────────────────────────────────
                if let Some(renderer) = &self.renderer {
                    renderer.draw_world(&self.world, &self.meshes, &self.camera);
                }

                // Request the next frame immediately (Poll mode equivalent).
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Nothing needed here — we drive redraws from RedrawRequested above.
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────
fn main() {
    // env_logger MUST be initialised before any wgpu calls so GPU errors are visible.
    env_logger::init();

    let event_loop = EventLoop::new().expect("Could not create event loop");
    // Poll = keep calling window_event even when idle (continuous game loop).
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = GameApp::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}