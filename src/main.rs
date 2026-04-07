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
mod animation;
mod camera;
mod components;
mod editor;
mod editor_ui;
mod input;
mod jobs;
mod materials;
mod navigation;
mod physics;
mod profiler;
mod renderer;
mod scene;
mod settings;
mod scripting;
mod systems;
mod terrain;

use std::sync::Arc;
use std::collections::HashMap;

use assets::{AssetStore, Mesh, MeshStreamingQueue};
use animation::{animation_system, AnimState, Animator};
use camera::Camera3D;
use components::{PlayerStart, RigidBody, Script};
use editor::EditorShell;
use editor_ui::{EditorUi, UiFrameArgs};
use input::InputState;
use jobs::JobSystem;
use materials::MaterialLibrary;
use navigation::NavGrid;
use physics::physics_system;
use profiler::FrameProfiler;
use renderer::Renderer;
use scene::SceneManager;
use settings::EngineSettings;
use scripting::ScriptEngine;
use systems::scripting_system;
use terrain::{remove_nearby_foliage, spawn_foliage_ring, TerrainGrid};

use hecs::World;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const CONTENT_SCRIPTS_DIR: &str = "Content/Scripts";
const CONTENT_MESHES_DIR: &str = "Content/Meshes";
const CONTENT_TEXTURES_DIR: &str = "Content/Textures";
const APP_ICON_PATH: &str = "assets/trinity_icon.png";

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
    settings:   EngineSettings,
    jobs:       JobSystem,
    profiler:   FrameProfiler,
    mesh_streaming: MeshStreamingQueue,
    editor_shell: EditorShell,
    editor_ui: Option<EditorUi>,
    materials: MaterialLibrary,
    selected_renderable: Option<hecs::Entity>,
    terrain: TerrainGrid,
    terrain_cursor_x: usize,
    terrain_cursor_z: usize,

    // Hot-reload receivers — Option because they're set up after the watcher starts.
    script_watcher: Option<std::sync::mpsc::Receiver<String>>,
    scene_watcher:  Option<std::sync::mpsc::Receiver<String>>,
    asset_watcher:  Option<std::sync::mpsc::Receiver<String>>,

    last_frame: std::time::Instant,
    start_time: std::time::Instant,
    frame_index: u64,
    sim_paused: bool,
    sim_step_once: bool,
    error_log: Vec<String>,
    nav_grid: NavGrid,
    nav_rebuild_requested: bool,
    frame_interval: std::time::Duration,
    next_frame_deadline: std::time::Instant,
    script_hot_reload_enabled: bool,
    preferred_script_editor: String,
    asset_hot_reload_enabled: bool,
    game_preview_mode: bool,
    prev_game_preview_mode: bool,
    mouse_look_active: bool,
    mouse_look_latched: bool,
    last_cursor_pos: Option<winit::dpi::PhysicalPosition<f64>>,
    camera_yaw: f32,
    camera_pitch: f32,
}

impl GameApp {
    fn new() -> Self {
        // Camera starts behind and above the scene, looking at origin.
        let mut camera = Camera3D::new(1280.0 / 720.0);
        camera.position = glam::Vec3::new(0.0, 4.0, 8.0);
        camera.target   = glam::Vec3::ZERO;
        let mut dir = (camera.target - camera.position).normalize_or_zero();
        if dir.length_squared() < 1e-6 {
            dir = glam::Vec3::new(0.0, -0.2, -1.0).normalize();
        }
        let camera_yaw = dir.z.atan2(dir.x);
        let camera_pitch = dir.y.asin();

        let settings = EngineSettings::load("engine_settings.toml");
        let jobs = JobSystem::new(
            settings.runtime.multithreading_enabled,
            settings.runtime.worker_threads,
        );
        let profiler = FrameProfiler::new(
            settings.runtime.profiler_enabled,
            settings.runtime.profiler_log_interval_frames,
        );
        let mesh_streaming = MeshStreamingQueue::new(settings.runtime.asset_streaming_enabled);
        let script_hot_reload_enabled = settings.runtime.script_hot_reload_enabled;
        let preferred_script_editor = settings.runtime.preferred_script_editor.clone();
        let asset_hot_reload_enabled = settings.runtime.asset_hot_reload_enabled;
        let frame_interval = std::time::Duration::from_micros(
            (1_000_000u64 / settings.runtime.max_fps.max(15) as u64).max(1),
        );

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
            settings,
            jobs,
            profiler,
            mesh_streaming,
            editor_shell: EditorShell::new(),
            editor_ui: None,
            materials: MaterialLibrary::new_defaults(),
            selected_renderable: None,
            terrain: TerrainGrid::new(64, 64, 1.0),
            terrain_cursor_x: 32,
            terrain_cursor_z: 32,
            script_watcher: None,
            scene_watcher:  None,
            asset_watcher:  None,
            last_frame:     std::time::Instant::now(),
            start_time:     std::time::Instant::now(),
            frame_index:    0,
            sim_paused: false,
            sim_step_once: false,
            error_log: Vec::new(),
            nav_grid: NavGrid::from_terrain(&TerrainGrid::new(64, 64, 1.0), 0.8),
            nav_rebuild_requested: true,
            frame_interval,
            next_frame_deadline: std::time::Instant::now(),
            script_hot_reload_enabled,
            preferred_script_editor,
            asset_hot_reload_enabled,
            game_preview_mode: false,
            prev_game_preview_mode: false,
            mouse_look_active: false,
            mouse_look_latched: false,
            last_cursor_pos: None,
            camera_yaw,
            camera_pitch,
        }
    }

    fn update_camera_target_from_angles(&mut self) {
        let cp = self.camera_pitch.cos();
        let forward = glam::Vec3::new(
            self.camera_yaw.cos() * cp,
            self.camera_pitch.sin(),
            self.camera_yaw.sin() * cp,
        )
        .normalize_or_zero();
        self.camera.target = self.camera.position + forward;
    }

    fn push_error(&mut self, msg: String) {
        self.error_log.push(msg);
        if self.error_log.len() > 120 {
            self.error_log.remove(0);
        }
    }

    fn cycle_selected_renderable(&mut self, forward: bool) {
        let entities: Vec<hecs::Entity> = self
            .world
            .query::<(hecs::Entity, &components::Renderable)>()
            .iter()
            .map(|(e, _)| e)
            .collect();
        if entities.is_empty() {
            self.selected_renderable = None;
            println!("[Materials] No renderable entities found.");
            return;
        }

        let next_idx = match self
            .selected_renderable
            .and_then(|sel| entities.iter().position(|e| *e == sel))
        {
            Some(i) => {
                if forward {
                    (i + 1) % entities.len()
                } else if i == 0 {
                    entities.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };

        self.selected_renderable = Some(entities[next_idx]);
        println!("[Materials] Selected entity: {:?}", entities[next_idx]);
    }

    fn apply_material_instance_to_selected(&mut self, name: &str) {
        let Some(entity) = self.selected_renderable else {
            println!("[Materials] No selected entity. Use N/M first.");
            return;
        };
        if let Ok(mut rend) = self.world.get::<&mut components::Renderable>(entity) {
            match self.materials.apply_instance(name, &mut rend) {
 git               Ok(_) => println!("[Materials] Applied '{}' to {:?}", name, entity),
                Err(e) => eprintln!("[Materials] {}", e),
            }
        } else {
            eprintln!("[Materials] Selected entity no longer has a Renderable.");
        }
    }

    fn snap_camera_to_selected(&mut self) {
        let Some(entity) = self.selected_renderable else {
            println!("[Camera] No selected entity to snap to.");
            return;
        };
        if let Ok(pos) = self.world.get::<&components::Position>(entity) {
            // Keep a small offset so camera doesn't sit inside the mesh.
            self.camera.position = glam::Vec3::new(pos.x + 2.0, pos.y + 1.5, pos.z + 3.0);
            self.camera.target = glam::Vec3::new(pos.x, pos.y, pos.z);
            let mut dir = (self.camera.target - self.camera.position).normalize_or_zero();
            if dir.length_squared() < 1e-6 {
                dir = glam::Vec3::new(0.0, -0.2, -1.0).normalize();
            }
            self.camera_yaw = dir.z.atan2(dir.x);
            self.camera_pitch = dir.y.asin();
            println!("[Camera] Snapped to selected entity: {:?}", entity);
        } else {
            println!("[Camera] Selected entity has no Position component.");
        }
    }

    fn ensure_content_layout(&mut self) {
        let _ = std::fs::create_dir_all(CONTENT_SCRIPTS_DIR);
        let _ = std::fs::create_dir_all(CONTENT_MESHES_DIR);
        let _ = std::fs::create_dir_all(CONTENT_TEXTURES_DIR);
        let player = format!("{}/player.lua", CONTENT_SCRIPTS_DIR);
        let enemy = format!("{}/enemy.lua", CONTENT_SCRIPTS_DIR);
        if !std::path::Path::new(&player).exists() {
            let _ = std::fs::write(
                &player,
                "function update(entity, dt)\n    -- player script\nend\n",
            );
        }
        if !std::path::Path::new(&enemy).exists() {
            let _ = std::fs::write(
                &enemy,
                "function update(entity, dt)\n    -- enemy script\nend\n",
            );
        }
    }

    fn apply_player_start_on_preview_begin(&mut self) {
        let start = self
            .world
            .query::<(&PlayerStart, &components::Position)>()
            .iter()
            .next()
            .map(|(_, p)| [p.x, p.y, p.z]);
        let Some(start) = start else { return; };
        let player_entities: Vec<hecs::Entity> = self
            .world
            .query::<(hecs::Entity, &Script)>()
            .iter()
            .filter(|(_, s)| s.path.to_ascii_lowercase().contains("player"))
            .map(|(e, _)| e)
            .collect();
        for e in player_entities {
            if let Ok(mut p) = self.world.get::<&mut components::Position>(e) {
                p.x = start[0];
                p.y = start[1];
                p.z = start[2];
            }
        }
    }

    fn start_asset_watcher(&mut self) {
        let (asset_tx, asset_rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            use notify::{recommended_watcher, RecursiveMode, Watcher};
            use std::path::Path;
            let (ntx, nrx) = std::sync::mpsc::channel();
            let mut watcher = recommended_watcher(move |res| { let _ = ntx.send(res); })
                .expect("Asset watcher failed");
            watcher.watch(Path::new("Content"), RecursiveMode::Recursive).ok();
            loop {
                match nrx.recv() {
                    Ok(Ok(event)) => {
                        for p in event.paths {
                            let s = p.to_string_lossy().to_string();
                            if s.ends_with(".obj")
                                || s.ends_with(".png")
                                || s.ends_with(".jpg")
                                || s.ends_with(".jpeg")
                            {
                                if asset_tx.send(s).is_err() {
                                    return;
                                }
                            }
                        }
                    }
                    _ => break,
                }
            }
        });
        self.asset_watcher = Some(asset_rx);
    }
}

fn load_window_icon(path: &str) -> Option<winit::window::Icon> {
    let bytes = std::fs::read(path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
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
                        .with_title("TrinityEngine")
                        .with_window_icon(load_window_icon(APP_ICON_PATH))
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Could not create window"),
        );

        // Update camera aspect ratio now that we know the window size.
        let phys = window.inner_size();
        self.camera.aspect = phys.width as f32 / phys.height as f32;

        // Build the GPU renderer (blocking — we are on the main thread).
        let renderer = pollster::block_on(Renderer::new(Arc::clone(&window)));
        let mut renderer = renderer;
        renderer.features.shadows_enabled = self.settings.render.shadows_enabled;
        renderer.features.pcf_enabled = self.settings.render.pcf_enabled;
        renderer.features.pcss_enabled = self.settings.render.pcss_enabled;
        renderer.features.ibl_enabled = self.settings.render.ibl_enabled;
        renderer.features.probes_enabled = self.settings.render.probes_enabled;
        renderer.features.volumetric_enabled = self.settings.render.volumetric_enabled;
        renderer.features.shadow_resolution = self.settings.render.shadow_resolution;
        renderer.features.pcf_samples = self.settings.render.pcf_samples;
        renderer.features.culling_enabled = self.settings.render.culling_enabled;
        renderer.features.culling_distance = self.settings.render.culling_distance;
        renderer.features.frustum_culling_enabled = self.settings.render.frustum_culling_enabled;
        renderer.features.bloom_enabled = self.settings.render.bloom_enabled;
        renderer.features.bloom_strength = self.settings.render.bloom_strength;
        renderer.features.ssao_enabled = self.settings.render.ssao_enabled;
        renderer.features.ssao_strength = self.settings.render.ssao_strength;
        renderer.features.volumetric_fog_enabled = self.settings.render.volumetric_fog_enabled;
        renderer.features.fog_density = self.settings.render.fog_density;
        renderer.features.voxel_gi_enabled = self.settings.render.voxel_gi_enabled;
        renderer.features.voxel_gi_strength = self.settings.render.voxel_gi_strength;
        renderer.features.sun_azimuth_deg = self.settings.render.sun_azimuth_deg;
        renderer.features.sun_elevation_deg = self.settings.render.sun_elevation_deg;
        renderer.features.sun_intensity = self.settings.render.sun_intensity;

        // Check if GPU is low-end and print a note.
        println!("[Engine] GPU: {:?}", renderer.adapter_info.name);
        println!("[Engine] Render settings loaded from engine_settings.toml");
        if self.jobs.enabled() {
            println!(
                "[Engine] Job system enabled (worker_threads={})",
                self.settings.runtime.worker_threads
            );
        }
        if self.mesh_streaming.enabled() {
            println!("[Engine] Threaded mesh streaming queue enabled");
            if let Ok(descs) = scene::parse_scene(&self.scene_mgr.scene_path) {
                for desc in descs {
                    let dx = desc.position[0] - self.camera.position.x;
                    let dy = desc.position[1] - self.camera.position.y;
                    let dz = desc.position[2] - self.camera.position.z;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);
                    let priority = 1.0 / dist;
                    self.mesh_streaming
                        .request_mesh_with_priority(&desc.mesh, priority);
                }
            }
        }

        self.input.configure_gamepad(
            self.settings.input.gamepad_enabled,
            self.settings.input.left_stick_deadzone,
        );

        self.ensure_content_layout();

        // Register Lua API and load startup scripts.
        self.scripts.register_api().expect("Lua API registration failed");
        EditorShell::print_help();
        MaterialLibrary::print_help();
        self.scripts.load_script(&format!("{}/player.lua", CONTENT_SCRIPTS_DIR)).ok();
        self.scripts.load_script(&format!("{}/enemy.lua", CONTENT_SCRIPTS_DIR)).ok();

        // Build the initial scene from scenes/main.scene.
        self.scene_mgr
            .build(&mut self.world, &mut self.meshes, &mut self.mesh_cache)
            .expect("Failed to load main.scene");
        self.nav_grid.rebuild(&self.terrain);

        // Add a starter animator to the first renderable entity as proof-of-system.
        let first_renderable = {
            self.world
                .query::<(hecs::Entity, &components::Renderable)>()
                .iter()
                .next()
                .map(|(e, _)| e)
        };
        if let Some(e) = first_renderable {
            let _ = self.world.insert(
                e,
                (Animator {
                    state: AnimState::Idle,
                    ..Animator::humanoid_default()
                },),
            );
            println!("[Animation] Animator attached to {:?}", e);
            println!("[Animation] Press J/K/L to set Idle/Walk/Run on selected entity.");
        }

        // Start file watchers for hot reload.
        if self.script_hot_reload_enabled {
            self.script_watcher = Some(self.scripts.start_watching(CONTENT_SCRIPTS_DIR));
        }
        if self.asset_hot_reload_enabled {
            self.start_asset_watcher();
        }

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

        self.window = Some(window);
        self.renderer = Some(renderer);
        if let (Some(window_ref), Some(renderer_ref)) = (self.window.as_ref(), self.renderer.as_ref()) {
            self.editor_ui = Some(EditorUi::new(window_ref, renderer_ref));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event:      WindowEvent,
    ) {
        if let (Some(window), Some(ui)) = (self.window.as_ref(), self.editor_ui.as_mut()) {
            let _ = ui.on_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event: ke, .. } => {
                self.input.handle_key(ke.physical_key, ke.state == ElementState::Pressed);
                if ke.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = ke.physical_key {
                        match code {
                            KeyCode::F1 => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.bloom_enabled = !r.features.bloom_enabled;
                                    println!(
                                        "[Toggle] Bloom: {}",
                                        if r.features.bloom_enabled { "ON" } else { "OFF" }
                                    );
                                    println!("[Info] {}", editor::describe_toggle("bloom"));
                                }
                            }
                            KeyCode::F2 => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.ssao_enabled = !r.features.ssao_enabled;
                                    println!(
                                        "[Toggle] SSAO: {}",
                                        if r.features.ssao_enabled { "ON" } else { "OFF" }
                                    );
                                    println!("[Info] {}", editor::describe_toggle("ssao"));
                                }
                            }
                            KeyCode::F3 => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.volumetric_fog_enabled = !r.features.volumetric_fog_enabled;
                                    println!(
                                        "[Toggle] Volumetric Fog: {}",
                                        if r.features.volumetric_fog_enabled { "ON" } else { "OFF" }
                                    );
                                    println!("[Info] {}", editor::describe_toggle("fog"));
                                }
                            }
                            KeyCode::F4 => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.voxel_gi_enabled = !r.features.voxel_gi_enabled;
                                    println!(
                                        "[Toggle] Voxel GI Prototype: {}",
                                        if r.features.voxel_gi_enabled { "ON" } else { "OFF" }
                                    );
                                    println!("[Info] {}", editor::describe_toggle("voxel"));
                                }
                            }
                            KeyCode::F5 => {
                                self.settings.render.preset = editor::cycle_preset(self.settings.render.preset);
                                println!("[Preset] Switched to {:?}", self.settings.render.preset);
                                println!("[Preset] In full visual editor this becomes a one-click dropdown.");
                            }
                            KeyCode::F10 => {
                                self.editor_shell.visible = !self.editor_shell.visible;
                                println!(
                                    "[Editor] Shell {}",
                                    if self.editor_shell.visible { "OPEN" } else { "CLOSED" }
                                );
                            }
                            KeyCode::F11 => {
                                self.editor_shell.show_advanced = !self.editor_shell.show_advanced;
                                println!(
                                    "[Editor] Advanced panel {}",
                                    if self.editor_shell.show_advanced { "ON" } else { "OFF" }
                                );
                            }
                            KeyCode::BracketLeft => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.bloom_strength = (r.features.bloom_strength - 0.02).max(0.0);
                                    println!("[Inspector] Bloom strength -> {:.2}", r.features.bloom_strength);
                                }
                            }
                            KeyCode::BracketRight => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.bloom_strength = (r.features.bloom_strength + 0.02).min(2.0);
                                    println!("[Inspector] Bloom strength -> {:.2}", r.features.bloom_strength);
                                }
                            }
                            KeyCode::KeyH => editor::print_hierarchy(&self.world),
                            KeyCode::KeyB => editor::print_asset_browser(),
                            KeyCode::KeyN => self.cycle_selected_renderable(false),
                            KeyCode::KeyM => self.cycle_selected_renderable(true),
                            KeyCode::Digit1 => self.apply_material_instance_to_selected("matte_black"),
                            KeyCode::Digit2 => self.apply_material_instance_to_selected("silver_brushed"),
                            KeyCode::Digit3 => self.apply_material_instance_to_selected("foliage_leaf"),
                            KeyCode::KeyJ => {
                                if let Some(entity) = self.selected_renderable {
                                    if let Ok(mut a) = self.world.get::<&mut Animator>(entity) {
                                        a.state = AnimState::Idle;
                                        println!("[Animation] {:?} -> Idle", entity);
                                    }
                                }
                            }
                            KeyCode::KeyK => {
                                if let Some(entity) = self.selected_renderable {
                                    if let Ok(mut a) = self.world.get::<&mut Animator>(entity) {
                                        a.state = AnimState::Walk;
                                        println!("[Animation] {:?} -> Walk", entity);
                                    }
                                }
                            }
                            KeyCode::KeyL => {
                                if let Some(entity) = self.selected_renderable {
                                    if let Ok(mut a) = self.world.get::<&mut Animator>(entity) {
                                        a.state = AnimState::Run;
                                        println!("[Animation] {:?} -> Run", entity);
                                    }
                                }
                            }
                            KeyCode::KeyF => {
                                editor::add_foliage_patch(
                                    &mut self.world,
                                    &mut self.meshes,
                                    &mut self.mesh_cache,
                                );
                            }
                            KeyCode::KeyT => {
                                self.terrain.raise_brush(self.terrain_cursor_x, self.terrain_cursor_z, 4, 0.15);
                                println!("[Terrain] Raised terrain brush at ({}, {})", self.terrain_cursor_x, self.terrain_cursor_z);
                            }
                            KeyCode::KeyG => {
                                self.terrain.lower_brush(self.terrain_cursor_x, self.terrain_cursor_z, 4, 0.15);
                                println!("[Terrain] Lowered terrain brush at ({}, {})", self.terrain_cursor_x, self.terrain_cursor_z);
                            }
                            KeyCode::KeyY => {
                                if let Some(handle) = self.mesh_cache.get("meshes/cube.obj").copied() {
                                    spawn_foliage_ring(
                                        &mut self.world,
                                        handle,
                                        self.terrain_cursor_x as f32 * self.terrain.cell_size - 32.0,
                                        self.terrain_cursor_z as f32 * self.terrain.cell_size - 32.0,
                                        4.0,
                                        24,
                                        true,
                                    );
                                    println!("[Terrain/Foliage] Added foliage ring with tree physics.");
                                } else {
                                    println!("[Terrain/Foliage] Load scene first so cube mesh exists.");
                                }
                            }
                            KeyCode::KeyU => {
                                let removed = remove_nearby_foliage(
                                    &mut self.world,
                                    self.terrain_cursor_x as f32 * self.terrain.cell_size - 32.0,
                                    self.terrain_cursor_z as f32 * self.terrain.cell_size - 32.0,
                                    4.5,
                                );
                                println!("[Terrain/Foliage] Removed {} nearby foliage entities.", removed);
                            }
                            KeyCode::KeyP => self.snap_camera_to_selected(),
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    self.mouse_look_active = state == ElementState::Pressed;
                    self.last_cursor_pos = None;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 0.7,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                };
                let forward = (self.camera.target - self.camera.position).normalize_or_zero();
                self.camera.position += forward * amount;
                self.update_camera_target_from_angles();
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_look_active || self.mouse_look_latched {
                    if let Some(prev) = self.last_cursor_pos {
                        let dx = (position.x - prev.x) as f32;
                        let dy = (position.y - prev.y) as f32;
                        let sensitivity = 0.0035;
                        self.camera_yaw += dx * sensitivity;
                        self.camera_pitch = (self.camera_pitch - dy * sensitivity).clamp(-1.5, 1.5);
                        self.update_camera_target_from_angles();
                    }
                    self.last_cursor_pos = Some(position);
                } else {
                    self.last_cursor_pos = None;
                }
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
                self.frame_index = self.frame_index.wrapping_add(1);
                let frame_start = std::time::Instant::now();
                // ── Delta time ─────────────────────────────────────────────
                let now = std::time::Instant::now();
                let dt  = now.duration_since(self.last_frame).as_secs_f32().min(0.05);
                self.last_frame = now;
                self.input.update_gamepads();

                let asset_start = std::time::Instant::now();
                for (path, result) in self.mesh_streaming.poll_loaded() {
                    match result {
                        Ok(mesh) => {
                            if let Some(handle) = self.mesh_cache.get(&path) {
                                self.meshes.replace(handle, mesh);
                            } else {
                                let handle = self.meshes.add(mesh);
                                self.mesh_cache.insert(path.clone(), handle);
                            }
                            println!("[Assets] Mesh ready: {}", path);
                        }
                        Err(e) => self.push_error(format!("[Assets] Mesh load failed {}: {}", path, e)),
                    }
                }
                self.mesh_streaming.pump_requests();
                let asset_time = asset_start.elapsed();

                // ── Hot reload: scripts ────────────────────────────────────
                if self.script_hot_reload_enabled && self.script_watcher.is_none() {
                    self.script_watcher = Some(self.scripts.start_watching(CONTENT_SCRIPTS_DIR));
                } else if !self.script_hot_reload_enabled {
                    self.script_watcher = None;
                }
                if let Some(rx) = &self.script_watcher {
                    let mut pending_errors: Vec<String> = Vec::new();
                    while let Ok(path) = rx.try_recv() {
                        match self.scripts.reload_script(&path) {
                            Ok(_)  => println!("[Hot] Script reloaded: {}", path),
                            Err(e) => pending_errors.push(format!("[Hot] Script error {}: {}", path, e)),
                        }
                    }
                    for e in pending_errors {
                        self.push_error(e);
                    }
                }
                if !self.asset_hot_reload_enabled {
                    self.asset_watcher = None;
                } else if self.asset_watcher.is_none() {
                    self.start_asset_watcher();
                }
                if let Some(rx) = &self.asset_watcher {
                    let mut pending_errors: Vec<String> = Vec::new();
                    while let Ok(path) = rx.try_recv() {
                        let norm = path.replace('\\', "/");
                        if norm.ends_with(".obj") {
                            if let Some(handle) = self.mesh_cache.get(&norm).copied() {
                                match Mesh::load(&norm) {
                                    Ok(mesh) => {
                                        self.meshes.replace(&handle, mesh);
                                        println!("[Hot] Mesh reloaded: {}", norm);
                                    }
                                    Err(e) => pending_errors.push(format!("[Hot] Mesh reload failed {}: {}", norm, e)),
                                }
                            }
                        } else {
                            println!("[Hot] Texture changed: {}", norm);
                        }
                    }
                    for e in pending_errors {
                        self.push_error(e);
                    }
                }

                // ── Hot reload: scenes ─────────────────────────────────────
                if let Some(rx) = &self.scene_watcher {
                    let mut pending_errors: Vec<String> = Vec::new();
                    let mut pending_toasts: Vec<String> = Vec::new();
                    while let Ok(path) = rx.try_recv() {
                        println!("[Hot] Scene changed: {}", path);
                        match self.scene_mgr.build(
                            &mut self.world,
                            &mut self.meshes,
                            &mut self.mesh_cache,
                        ) {
                            Ok(_)  => {
                                println!("[Hot] Scene rebuilt");
                                pending_toasts.push(format!("Scene hot reloaded: {}", path));
                            }
                            Err(e) => pending_errors.push(format!("[Hot] Scene error: {}", e)),
                        }
                    }
                    for e in pending_errors {
                        self.push_error(e);
                    }
                    if let Some(ui) = self.editor_ui.as_mut() {
                        let now = self.start_time.elapsed().as_secs_f32();
                        for msg in pending_toasts {
                            ui.push_toast(msg, now);
                        }
                    }
                }

                if self.nav_rebuild_requested {
                    self.nav_grid.rebuild(&self.terrain);
                    self.nav_rebuild_requested = false;
                }

                // ── Reset physics ground flag ──────────────────────────────
                // Must happen before physics_system so entities that walk off
                // edges fall correctly on the next frame.
                for body in self.world.query_mut::<&mut RigidBody>() {
                    body.on_ground = false;
                }

                // ── Systems ────────────────────────────────────────────────
                // Editor Scene = authoring mode (no gameplay simulation).
                // Game Preview = runs scripts/physics/animation, like Unreal PIE.
                let run_sim = self.game_preview_mode && (!self.sim_paused || self.sim_step_once);
                if self.game_preview_mode && !self.prev_game_preview_mode {
                    self.apply_player_start_on_preview_begin();
                }
                self.prev_game_preview_mode = self.game_preview_mode;
                let mut script_time = std::time::Duration::ZERO;
                let mut physics_time = std::time::Duration::ZERO;
                if run_sim {
                    let script_start = std::time::Instant::now();
                    scripting_system(&mut self.world, &mut self.scripts, &self.input, dt);
                    self.scripts.drain_destroys(&mut self.world);
                    script_time = script_start.elapsed();

                    let physics_start = std::time::Instant::now();
                    let sim_time = self.start_time.elapsed().as_secs_f32();
                    let divisor = self.settings.runtime.foliage_wind_update_divisor.max(1) as u64;
                    let wind_this_frame = (self.frame_index % divisor) == 0;
                    let _collisions = physics_system(
                        &mut self.world,
                        dt,
                        sim_time,
                        &self.jobs,
                        self.settings.runtime.foliage_wind_enabled,
                        wind_this_frame,
                    );
                    physics_time = physics_start.elapsed();

                    animation_system(&mut self.world, dt, &self.jobs);
                    if self.sim_step_once {
                        self.sim_step_once = false;
                        self.sim_paused = true;
                    }
                }

                // Seamless free-fly camera: RMB look + WASD move, Shift sprint, Space/Ctrl up/down.
                {
                    let forward = (self.camera.target - self.camera.position).normalize_or_zero();
                    let right = forward.cross(glam::Vec3::Y).normalize_or_zero();
                    let up = glam::Vec3::Y;
                    let mut move_dir = glam::Vec3::ZERO;
                    if self.input.is_held(KeyCode::KeyW) { move_dir += forward; }
                    if self.input.is_held(KeyCode::KeyS) { move_dir -= forward; }
                    if self.input.is_held(KeyCode::KeyD) { move_dir += right; }
                    if self.input.is_held(KeyCode::KeyA) { move_dir -= right; }
                    if self.input.is_held(KeyCode::Space) { move_dir += up; }
                    if self.input.is_held(KeyCode::ControlLeft) || self.input.is_held(KeyCode::ControlRight) {
                        move_dir -= up;
                    }
                    if move_dir.length_squared() > 0.0 {
                        let speed = if self.input.is_held(KeyCode::ShiftLeft) || self.input.is_held(KeyCode::ShiftRight) {
                            14.0
                        } else {
                            6.0
                        };
                        self.camera.position += move_dir.normalize() * speed * dt.max(0.0);
                        self.update_camera_target_from_angles();
                    }
                }

                // ── Render ─────────────────────────────────────────────────
                let render_start = std::time::Instant::now();
                let mut draw_stats = renderer::DrawStats::default();
                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    if let Some(ui) = self.editor_ui.as_mut() {
                        let mut frame_args = UiFrameArgs {
                            world: &mut self.world,
                            settings: &mut self.settings,
                            renderer,
                            camera: &self.camera,
                            profiler: &self.profiler,
                            mesh_cache: &mut self.mesh_cache,
                            meshes: &mut self.meshes,
                            materials: &mut self.materials,
                            selected_renderable: &mut self.selected_renderable,
                            terrain: &mut self.terrain,
                            terrain_cursor_x: self.terrain_cursor_x,
                            terrain_cursor_z: self.terrain_cursor_z,
                            app_time_seconds: self.start_time.elapsed().as_secs_f32(),
                            sim_paused: &mut self.sim_paused,
                            sim_step_once: &mut self.sim_step_once,
                            game_preview_mode: &mut self.game_preview_mode,
                            mouse_look_latched: &mut self.mouse_look_latched,
                            error_log: &mut self.error_log,
                            nav_grid: &mut self.nav_grid,
                            nav_rebuild_requested: &mut self.nav_rebuild_requested,
                            scripts: &mut self.scripts,
                            scripts_dir: CONTENT_SCRIPTS_DIR,
                            script_hot_reload_enabled: &mut self.script_hot_reload_enabled,
                            preferred_script_editor: &mut self.preferred_script_editor,
                            asset_hot_reload_enabled: &mut self.asset_hot_reload_enabled,
                        };
                        ui.begin_and_build(window, &mut frame_args);
                    }

                    let mut draw_ui = |device: &wgpu::Device,
                                       queue: &wgpu::Queue,
                                       encoder: &mut wgpu::CommandEncoder,
                                       view: &wgpu::TextureView| {
                        if let Some(ui) = self.editor_ui.as_mut() {
                            ui.paint_on(device, queue, encoder, view);
                        }
                    };
                    draw_stats = renderer.draw_world(
                        &self.world,
                        &self.meshes,
                        &self.camera,
                        &self.jobs,
                        Some(&mut draw_ui),
                    );
                }
                let render_time = render_start.elapsed();

                self.profiler.record(
                    frame_start.elapsed(),
                    script_time,
                    physics_time,
                    render_time,
                    asset_time,
                    draw_stats,
                    self.jobs.enabled(),
                );
                if let Some(window) = &self.window {
                    if let Some(title) = self.profiler.overlay_text() {
                        window.set_title(title);
                    }
                }
                self.editor_shell.render_snapshot(
                    &self.world,
                    &self.settings,
                    self.renderer.as_ref(),
                    &self.profiler,
                );

            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame_deadline));
        if let Some(window) = &self.window {
            let now = std::time::Instant::now();
            if now >= self.next_frame_deadline {
                window.request_redraw();
                self.next_frame_deadline = now + self.frame_interval;
            }
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────
fn main() {
    // env_logger MUST be initialised before any wgpu calls so GPU errors are visible.
    env_logger::init();

    let event_loop = EventLoop::new().expect("Could not create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = GameApp::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}