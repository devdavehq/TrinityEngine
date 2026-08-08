#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]
#![allow(dead_code)]

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

mod core;
mod render;
mod environment;

#[cfg(feature = "audio")]
mod audio;

mod assets;
mod animation;
mod camera;
mod components;
#[cfg(feature = "editor")]
mod editor;
#[cfg(feature = "editor")]
mod editor_assets;
#[cfg(feature = "editor")]
mod editor_persist;
#[cfg(feature = "editor")]
mod editor_ui;
mod project_registry;
mod input;
mod jobs;
mod materials;
mod navigation;
mod navmesh;
mod boids;
mod cinematics;
mod ai;
mod physics;
mod particles;
mod profiler;
mod renderer;
mod scene;
mod levels;
mod ui;
mod settings;
#[cfg(feature = "scripting")]
mod scripting;
#[cfg(feature = "scripting")]
mod scripting_api;
mod systems;
mod terrain;
mod vfs;
mod resources;
mod engine_subsystems;
mod demo_plugin;
mod save_plugin;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use assets::Mesh;
use animation::{animation_system, AnimState, Animator};
use animation::blending::animation_blending_system;
use animation::anim_graph::anim_graph_system;
use components::{PlayerStart, RigidBody, Script, Position, Rotation, PointLight};
#[cfg(feature = "editor")]
use editor::EditorShell;
#[cfg(feature = "editor")]
use editor::backend::{EditorBackend, HeadlessEditor};
#[cfg(feature = "editor")]
use editor_ui::{EditorUi, UiFrameArgs};
use engine_subsystems::{EnvironmentState, LevelState, AssetState, CameraInputState};
use jobs::JobSystem;
use materials::MaterialLibrary;
use navigation::NavGrid;
use ai::AiRegistry;
use ai::components::ai_system;
use physics::{physics_system, character_controller_system, ragdoll_system, water_trigger_system};
use profiler::FrameProfiler;
use renderer::Renderer;
use scene::{SceneManager, SubSceneManager, SceneTransition};
use settings::EngineSettings;
#[cfg(feature = "scripting")]
use scripting::ScriptEngine;
use systems::scripting_system;
use terrain::{remove_nearby_foliage, spawn_foliage_ring, TerrainWorld};

// New core systems
use core::{EventBus, BeginFrameEvent, EndFrameEvent};
use core::events::{TimeOfDayChangedEvent, WeatherChangedEvent, ThunderEvent};
use render::{InstancingManager, ShaderManager};
#[cfg(feature = "audio")]
use audio::AudioSystem;

// Environment system
use environment::splash::SplashManager;
// Flood system function (state now in LevelState).
use environment::flood::flood_system;

use hecs::World;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const CONTENT_SCRIPTS_DIR: &str = "Content/Scripts";
const CONTENT_PLUGINS_DIR: &str = "Content/Scripts/plugins";
const CONTENT_MESHES_DIR: &str = "Content/Meshes";
const CONTENT_TEXTURES_DIR: &str = "Content/Textures";
const CONTENT_MATERIALS_DIR: &str = "Content/Materials";
const CONTENT_PREFABS_DIR: &str = "Content/Prefabs";
const APP_ICON_PATH: &str = "assets/trinity_icon.png";
pub const TRINITY_ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppStage {
    BootSplash,
    ProjectHub,
    EditorLoading,
    EditorReady,
}

// ── PlaySnapshot ─────────────────────────────────────────────────────────────
// Captures the full editor state before entering Game Preview so we can
// restore it when the user exits play mode.  This ensures the scene always
// returns to exactly where the user left it in the editor.
struct PlaySnapshot {
    positions:  std::collections::HashMap<hecs::Entity, Position>,
    rotations:  std::collections::HashMap<hecs::Entity, Rotation>,
    rigid_bodies: std::collections::HashMap<hecs::Entity, RigidBody>,
    point_lights: std::collections::HashMap<hecs::Entity, PointLight>,
    camera_pos: glam::Vec3,
    camera_target: glam::Vec3,
    camera_yaw: f32,
    camera_pitch: f32,
}

// ── GameApp ───────────────────────────────────────────────────────────────────
// Owns all engine state. Created before the event loop starts.
// Fields are Option<> for anything that needs a window to initialise.
struct GameApp {
    // GPU renderer — None until resumed() fires and we have a window.
    renderer:   Option<Renderer>,
    // winit window wrapped in Arc so wgpu Surface can hold a reference.
    window:     Option<Arc<Window>>,

    world:      World,
    // ── Asset subsystem ──────────────────────────────────────────────────
    assets: AssetState,
    // ── Camera & input subsystem ─────────────────────────────────────────
    input_state: CameraInputState,
    #[cfg(feature = "scripting")]
    scripts:    ScriptEngine,
    /// Demo plugin: shows the engine's formal ScriptPlugin extension pattern.
    /// Registered onto Lua globals and ticked each frame so Lua can call its
    /// `demo.*` functions and read per-frame state.
    demo_plugin: demo_plugin::DemoPlugin,
    scene_mgr:  SceneManager,
    /// Sub-scene manager: loads scenes inside the current scene at world offsets.
    sub_scene_mgr: SubSceneManager,
    /// Scene transition controller: fade-to-black effect for scene switches.
    transition: SceneTransition,
    settings:   EngineSettings,
    jobs:       JobSystem,
    profiler:   FrameProfiler,
    // (mesh_streaming now in assets: AssetState)
    #[cfg(feature = "editor")]
    editor_shell: EditorShell,
    #[cfg(feature = "editor")]
    editor_ui: Option<EditorUi>,
    #[cfg(feature = "editor")]
    editor_backend: Box<dyn EditorBackend>,
    // (materials now in assets: AssetState)
    selected_renderable: Option<hecs::Entity>,
    terrain_world: TerrainWorld,
    terrain_cursor_x: usize,
    terrain_cursor_z: usize,

    // ── New core systems ──────────────────────────────────────────────────
    // Event bus: the nervous system. All inter-system communication goes here.
    events: EventBus,
    // GPU instancing: batches identical meshes into one draw call.
    instancing: InstancingManager,
    // Shader management: compilation, caching, hot-reload.
    shader_mgr: ShaderManager,

    // ── Environment subsystem ────────────────────────────────────────────
    env: EnvironmentState,

    // ── Splash visual system ─────────────────────────────────────────────
    splash_manager: SplashManager,

    // ── Level streaming subsystem ────────────────────────────────────────
    levels: LevelState,

    // ── Audio system ────────────────────────────────────────────────────
    #[cfg(feature = "audio")]
    audio: Option<AudioSystem>,

    // ── Particle system ────────────────────────────────────────────────
    particles: particles::ParticleSystem,
    particle_indices: [usize; 4],

    // ── Boids / flocking system ─────────────────────────────────────────
    boids: boids::BoidRegistry,

    // ── Jolt Physics backend ────────────────────────────────────────────
    #[cfg(feature = "jolt")]
    jolt: Option<physics::jolt_bridge::JoltBridge>,

    // Hot-reload receivers — Option because they're set up after the watcher starts.
    script_watcher: Option<std::sync::mpsc::Receiver<String>>,
    scene_watcher:  Option<std::sync::mpsc::Receiver<String>>,
    asset_watcher:  Option<std::sync::mpsc::Receiver<String>>,

    last_frame: std::time::Instant,
    start_time: std::time::Instant,
    frame_index: u64,
    sim_paused: bool,
    sim_step_once: bool,
    script_skip_frames_remaining: u32,
    error_log: Vec<String>,
    nav_grid: NavGrid,
    navmesh: navmesh::NavMesh,
    ai_registry: AiRegistry,
    nav_rebuild_requested: bool,
    frame_interval: std::time::Duration,
    next_frame_deadline: std::time::Instant,
    script_hot_reload_enabled: bool,
    preferred_script_editor: String,
    asset_hot_reload_enabled: bool,
    game_preview_mode: bool,
    prev_game_preview_mode: bool,
    /// Snapshot of entity state before entering Game Preview — restored on exit.
    play_snapshot: Option<PlaySnapshot>,
    app_stage: AppStage,
    project_stage_started_at: std::time::Instant,
    request_return_to_hub: bool,
    available_scene_paths: Vec<String>,
    scene_list_dirty: bool,
    requested_scene_switch: Option<String>,
    stop_asset_watch: Arc<AtomicBool>,
    stop_scene_watch: Arc<AtomicBool>,
}

impl GameApp {
    fn new() -> Self {
        let settings = EngineSettings::load("engine_settings.toml");
        let jobs = JobSystem::new(
            settings.runtime.multithreading_enabled,
            settings.runtime.worker_threads,
        );
        let profiler = FrameProfiler::new(
            settings.runtime.profiler_enabled,
            settings.runtime.profiler_log_interval_frames,
        );
        let script_hot_reload_enabled = settings.runtime.script_hot_reload_enabled;
        let preferred_script_editor = settings.runtime.preferred_script_editor.clone();
        let asset_hot_reload_enabled = settings.runtime.asset_hot_reload_enabled;
        let frame_interval = std::time::Duration::from_micros(
            (1_000_000u64 / settings.runtime.max_fps.max(15) as u64).max(1),
        );

        Self {
            renderer:       None,
            window:         None,
            world:          World::new(),
            assets:         AssetState::with_streaming(settings.runtime.asset_streaming_enabled),
            input_state:    CameraInputState::new(),
            #[cfg(feature = "scripting")]
            scripts:        ScriptEngine::new(),
            demo_plugin:    demo_plugin::DemoPlugin::new(),
            scene_mgr:      SceneManager::new(&resolve_primary_scene_path(&settings.runtime.startup_scene_path)),
            sub_scene_mgr:  SubSceneManager::new(),
            transition:     SceneTransition::new(),
            settings,
            jobs,
            profiler,
            #[cfg(feature = "editor")]
            editor_shell: EditorShell::new(),
            #[cfg(feature = "editor")]
            editor_ui: None,
            #[cfg(feature = "editor")]
            editor_backend: Box::new(HeadlessEditor::new()),
            selected_renderable: None,
            terrain_world: TerrainWorld::new(64, 64, 16, 1.0),
            terrain_cursor_x: 32,
            terrain_cursor_z: 32,
            // New core systems
            events:     EventBus::new(),
            instancing: InstancingManager::new(),
            shader_mgr: ShaderManager::new(),
            // Environment subsystem
            env: EnvironmentState::new(),
            splash_manager: SplashManager::new(),
            // Level subsystem
            levels: LevelState::new(),
            // Audio
            #[cfg(feature = "audio")]
            audio: AudioSystem::new(),
            // Particles
            particles: particles::ParticleSystem::new(),
            particle_indices: [0, 1, 2, 3],
            // Boids
            boids: boids::BoidRegistry::new(),
            // Jolt Physics
            #[cfg(feature = "jolt")]
            jolt: None,
            script_watcher: None,
            scene_watcher:  None,
            asset_watcher:  None,
            last_frame:     std::time::Instant::now(),
            start_time:     std::time::Instant::now(),
            frame_index:    0,
            sim_paused: false,
            sim_step_once: false,
            script_skip_frames_remaining: 0,
            error_log: Vec::new(),
            nav_grid: NavGrid { width: 64, depth: 64, walkable: vec![true; 64 * 64], max_slope: 0.8, contour_edges: Vec::new(), region_count: 0 },
            navmesh: navmesh::NavMesh::empty(),
            ai_registry: AiRegistry::new(),
            nav_rebuild_requested: true,
            frame_interval,
            next_frame_deadline: std::time::Instant::now(),
            script_hot_reload_enabled,
            preferred_script_editor,
            asset_hot_reload_enabled,
            game_preview_mode: false,
            prev_game_preview_mode: false,
            play_snapshot: None,
            app_stage: AppStage::BootSplash,
            project_stage_started_at: std::time::Instant::now(),
            request_return_to_hub: false,
            available_scene_paths: Vec::new(),
            scene_list_dirty: true,
            requested_scene_switch: None,
            stop_asset_watch: Arc::new(AtomicBool::new(false)),
            stop_scene_watch: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Attempt to initialize the Jolt Physics backend.
    /// Called once at startup or when the user explicitly requests Jolt.
    /// Requires CMake to build the Jolt native library — falls back gracefully.
    #[cfg(feature = "jolt")]
    fn init_jolt(&mut self) {
        if self.jolt.is_some() {
            tracing::info!("[Jolt] Already initialized.");
            return;
        }
        tracing::info!("[Jolt] Attempting to initialize Jolt Physics backend...");
        // rolt v0.3 wraps Jolt internally. If CMake or the build is missing,
        // JoltBridge::new() still creates the struct (stub), but we log a warning.
        let bridge = physics::jolt_bridge::JoltBridge::new();
        if bridge.initialized {
            tracing::info!("[Jolt] Physics backend ready (gravity: {:?}).", bridge.gravity);
        } else {
            tracing::warn!("[Jolt] Backend created but not fully initialized — using stub.");
        }
        self.jolt = Some(bridge);
    }

    fn refresh_available_scenes(&mut self) {
        self.available_scene_paths.clear();
        if let Ok(rd) = std::fs::read_dir("scenes") {
            for p in rd.flatten().map(|e| e.path()) {
                if p.extension().and_then(|x| x.to_str()) == Some("scene") {
                    self.available_scene_paths
                        .push(p.to_string_lossy().to_string());
                }
            }
        }
        self.available_scene_paths.sort();
        if self.available_scene_paths.is_empty() {
            self.available_scene_paths.push("scenes/main.scene".to_string());
        }
        self.scene_list_dirty = false;
    }

    fn refresh_available_scenes_if_needed(&mut self) {
        if self.scene_list_dirty {
            self.refresh_available_scenes();
        }
    }

    fn mark_editor_content_dirty(&mut self) {
        if let Some(ui) = self.editor_ui.as_mut() {
            ui.mark_content_dirty();
        }
    }

    fn stop_project_watchers(&mut self) {
        self.script_watcher = None;
        self.asset_watcher = None;
        self.scene_watcher = None;
        self.stop_asset_watch.store(true, Ordering::SeqCst);
        self.stop_scene_watch.store(true, Ordering::SeqCst);
    }

    fn update_camera_target_from_angles(&mut self) {
        self.input_state.update_camera_target_from_angles();
    }

    fn push_error(&mut self, msg: String) {
        self.error_log.push(msg);
        if self.error_log.len() > 500 {
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
            tracing::info!("[Materials] No renderable entities found.");
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
        tracing::info!("[Materials] Selected entity: {:?}", entities[next_idx]);
    }

    fn apply_material_instance_to_selected(&mut self, name: &str) {
        let Some(entity) = self.selected_renderable else {
            tracing::info!("[Materials] No selected entity. Use N/M first.");
            return;
        };
        if let Ok(mut rend) = self.world.get::<&mut components::Renderable>(entity) {
            match self.assets.materials.apply_instance(name, &mut rend) {
                Ok(_) => tracing::info!("[Materials] Applied '{}' to {:?}", name, entity),
                Err(e) => tracing::error!("[Materials] {}", e),
            }
        } else {
            tracing::error!("[Materials] Selected entity no longer has a Renderable.");
        }
    }

    fn snap_camera_to_selected(&mut self) {
        let Some(entity) = self.selected_renderable else {
            tracing::info!("[Camera] No selected entity to snap to.");
            return;
        };
        if let Ok(pos) = self.world.get::<&components::Position>(entity) {
            // Keep a small offset so camera doesn't sit inside the mesh.
            self.input_state.camera.position = glam::Vec3::new(pos.x + 2.0, pos.y + 1.5, pos.z + 3.0);
            self.input_state.camera.target = glam::Vec3::new(pos.x, pos.y, pos.z);
            let mut dir = (self.input_state.camera.target - self.input_state.camera.position).normalize_or_zero();
            if dir.length_squared() < 1e-6 {
                dir = glam::Vec3::new(0.0, -0.2, -1.0).normalize();
            }
            self.input_state.camera_yaw = dir.z.atan2(dir.x);
            self.input_state.camera_pitch = dir.y.asin();
            tracing::info!("[Camera] Snapped to selected entity: {:?}", entity);
        } else {
            tracing::info!("[Camera] Selected entity has no Position component.");
        }
    }

    fn focus_selected_frame(&mut self) {
        let Some(entity) = self.selected_renderable else {
            return;
        };
        let Ok(pos) = self.world.get::<&components::Position>(entity) else {
            return;
        };
        let mut radius = 1.5f32;
        if let Ok(r) = self.world.get::<&components::Renderable>(entity) {
            radius = r.scale[0]
                .abs()
                .max(r.scale[1].abs())
                .max(r.scale[2].abs())
                .max(0.6)
                * 2.6;
        }
        let dir = (self.input_state.camera.position - glam::Vec3::new(pos.x, pos.y, pos.z)).normalize_or_zero();
        let fallback = glam::Vec3::new(0.45, 0.25, 0.85).normalize();
        let d = if dir.length_squared() < 1e-6 { fallback } else { dir };
        self.input_state.camera.target = glam::Vec3::new(pos.x, pos.y, pos.z);
        self.input_state.camera.position = self.input_state.camera.target + d * radius;
        self.input_state.orbit_distance = radius;
        let mut view_dir = (self.input_state.camera.target - self.input_state.camera.position).normalize_or_zero();
        if view_dir.length_squared() < 1e-6 {
            view_dir = glam::Vec3::new(0.0, -0.2, -1.0).normalize();
        }
        self.input_state.camera_yaw = view_dir.z.atan2(view_dir.x);
        self.input_state.camera_pitch = view_dir.y.asin();
    }

    fn spawn_scene_watcher(&mut self) {
        let prev = Arc::clone(&self.stop_scene_watch);
        prev.store(true, Ordering::SeqCst);
        self.scene_watcher = None;
        self.stop_scene_watch = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&self.stop_scene_watch);
        let (scene_tx, scene_rx) = std::sync::mpsc::channel::<String>();
        {
            let tx = scene_tx.clone();
            std::thread::spawn(move || {
                use notify::{recommended_watcher, RecursiveMode, Watcher};
                use std::path::Path;
                let (ntx, nrx) = std::sync::mpsc::channel();
                let mut watcher = match recommended_watcher(move |res| {
                    let _ = ntx.send(res);
                }) {
                    Ok(w) => w,
                    Err(e) => {
                        tracing::error!("[Scene] Scene watcher failed: {}", e);
                        return;
                    }
                };
                watcher.watch(Path::new("scenes"), RecursiveMode::Recursive).ok();
                loop {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    match nrx.recv_timeout(Duration::from_millis(400)) {
                        Ok(Ok(event)) => {
                            for path in event.paths {
                                let s = path.to_string_lossy().to_string();
                                if s.ends_with(".scene") {
                                    if tx.send(s).is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                        Ok(Err(_)) => {}
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            });
        }
        self.scene_watcher = Some(scene_rx);
    }

    fn switch_to_project(&mut self, project_dir: PathBuf) {
        if std::env::set_current_dir(&project_dir).is_err() {
            self.error_log
                .push(format!("[Hub] Could not open project path {:?}", project_dir));
            return;
        }
        self.scene_mgr.scene_path = resolve_primary_scene_path(&self.settings.runtime.startup_scene_path);
        self.scene_list_dirty = true;
        self.refresh_available_scenes_if_needed();
        self.ensure_content_layout();
        self.mark_editor_content_dirty();
        // Load data-driven materials from Content/Materials/
        self.assets.materials.load_from_directory(CONTENT_MATERIALS_DIR);
        // Load prefabs from Content/Prefabs/
        self.assets.prefab_registry.load_from_directory("Content/Prefabs");
        self.assets.mesh_cache.clear();
        match self.scene_mgr.build(&mut self.world, &mut self.assets.meshes, &mut self.assets.mesh_cache, Some(&self.assets.prefab_registry)) {
            Ok(()) => {}
            Err(e) => {
                self.error_log.push(format!("[Hub] Scene load failed: {}", e));
            }
        }
self.nav_grid.rebuild_from_heights(&self.terrain_world);
                                        self.navmesh = navmesh::NavMesh::from_terrain(&self.nav_grid, &self.terrain_world);
        self.selected_renderable = None;

        self.scripts = ScriptEngine::new();
        // Enforce the scripting sandbox: 128 MB heap cap and a ~25 ms
        // execution budget per Lua call.  os/io/loadfile/dofile/load/require
        // are stripped by default in register_api(); these caps stop runaway
        // scripts from exhausting memory or stalling a frame.
        self.scripts.set_sandbox(crate::scripting::SandboxConfig {
            max_memory_bytes: 128 << 20,
            max_execution_time_ms: 25,
            ..crate::scripting::SandboxConfig::default()
        });
        self.scripts.register_plugin(Box::new(demo_plugin::DemoPlugin::with_runtime(
            self.demo_plugin.runtime(),
        )));
        self.scripts.register_plugin(Box::new(save_plugin::WorldStatePlugin::new(
            self.levels.world_state.clone(),
        )));
        if self.scripts.register_api().is_ok() {
            self.scripts
                .load_script(&format!("{}/player.lua", CONTENT_SCRIPTS_DIR))
                .ok();
            self.scripts
                .load_script(&format!("{}/enemy.lua", CONTENT_SCRIPTS_DIR))
                .ok();
            // Lua-native plugins: hot-reloadable, no Rust recompilation.
            // Any .lua file returning { name, start, update, on_event } under
            // Content/Scripts/plugins is loaded and ticked automatically.
            let _ = std::fs::create_dir_all(CONTENT_PLUGINS_DIR);
            let _ = self.scripts.load_plugins(CONTENT_PLUGINS_DIR);
        }

        self.stop_project_watchers();
        if self.script_hot_reload_enabled {
            self.script_watcher = Some(self.scripts.start_watching(CONTENT_SCRIPTS_DIR));
        }
        if self.asset_hot_reload_enabled {
            self.start_asset_watcher();
        }
        self.spawn_scene_watcher();

        if self.assets.mesh_streaming.enabled() {
            if let Ok(descs) = scene::parse_scene(&self.scene_mgr.scene_path) {
                for desc in descs {
                    let dx = desc.position[0] - self.input_state.camera.position.x;
                    let dy = desc.position[1] - self.input_state.camera.position.y;
                    let dz = desc.position[2] - self.input_state.camera.position.z;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);
                    let priority = 1.0 / dist;
                    self.assets.mesh_streaming
                        .request_mesh_with_priority(&desc.mesh, priority);
                }
            }
        }

        let first_renderable = self
            .world
            .query::<(hecs::Entity, &components::Renderable)>()
            .iter()
            .next()
            .map(|(e, _)| e);
        if let Some(e) = first_renderable {
            let _ = self.world.insert(
                e,
                (Animator {
                    state: AnimState::Idle,
                    ..Animator::humanoid_default()
                },),
            );
        }

        tracing::info!(
            "[Hub] Opened project {:?} (scene {})",
            project_dir, self.scene_mgr.scene_path
        );
    }

    fn ensure_content_layout(&mut self) {
        let _ = std::fs::create_dir_all(CONTENT_SCRIPTS_DIR);
        let _ = std::fs::create_dir_all(CONTENT_MESHES_DIR);
        let _ = std::fs::create_dir_all(CONTENT_TEXTURES_DIR);
        let _ = std::fs::create_dir_all(CONTENT_MATERIALS_DIR);
        let _ = std::fs::create_dir_all(CONTENT_PREFABS_DIR);
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

    /// Snapshot the entire editor state before entering Game Preview.
    /// Stores per-entity Position, Rotation, RigidBody, PointLight and camera state.
    fn capture_play_snapshot(&mut self) {
        let mut snap = PlaySnapshot {
            positions:    std::collections::HashMap::new(),
            rotations:    std::collections::HashMap::new(),
            rigid_bodies: std::collections::HashMap::new(),
            point_lights: std::collections::HashMap::new(),
            camera_pos:   self.input_state.camera.position,
            camera_target: self.input_state.camera.target,
            camera_yaw:   self.input_state.camera_yaw,
            camera_pitch: self.input_state.camera_pitch,
        };
        for (entity, pos) in self.world.query_mut::<(hecs::Entity, &Position)>() {
            snap.positions.insert(entity, *pos);
        }
        for (entity, rot) in self.world.query_mut::<(hecs::Entity, &Rotation)>() {
            snap.rotations.insert(entity, *rot);
        }
        for (entity, rb) in self.world.query_mut::<(hecs::Entity, &RigidBody)>() {
            snap.rigid_bodies.insert(entity, *rb);
        }
        for (entity, pl) in self.world.query_mut::<(hecs::Entity, &PointLight)>() {
            snap.point_lights.insert(entity, *pl);
        }
        tracing::info!(
            "[Play] Snapshot captured: {} positions, {} rotations, {} rigid bodies, {} lights",
            snap.positions.len(), snap.rotations.len(),
            snap.rigid_bodies.len(), snap.point_lights.len(),
        );
        self.play_snapshot = Some(snap);
    }

    /// Restore the editor state that was saved before Game Preview began.
    /// Resets all entity positions, rotations, velocities to their pre-simulation values.
    fn restore_play_snapshot(&mut self) {
        let Some(snap) = self.play_snapshot.take() else { return };
        let mut restored = 0u32;
        for (entity, pos) in self.world.query_mut::<(hecs::Entity, &mut Position)>() {
            if let Some(&original) = snap.positions.get(&entity) {
                *pos = original;
                restored += 1;
            }
        }
        for (entity, rot) in self.world.query_mut::<(hecs::Entity, &mut Rotation)>() {
            if let Some(&original) = snap.rotations.get(&entity) {
                *rot = original;
            }
        }
        for (entity, rb) in self.world.query_mut::<(hecs::Entity, &mut RigidBody)>() {
            if let Some(&original) = snap.rigid_bodies.get(&entity) {
                *rb = original;
            }
        }
        for (entity, pl) in self.world.query_mut::<(hecs::Entity, &mut PointLight)>() {
            if let Some(&original) = snap.point_lights.get(&entity) {
                *pl = original;
            }
        }
        // Restore camera to editor position.
        self.input_state.camera.position = snap.camera_pos;
        self.input_state.camera.target   = snap.camera_target;
        self.input_state.camera_yaw      = snap.camera_yaw;
        self.input_state.camera_pitch    = snap.camera_pitch;
        self.update_camera_target_from_angles();
        tracing::info!("[Play] Snapshot restored: {} entities.", restored);
    }

    fn start_asset_watcher(&mut self) {
        let prev = Arc::clone(&self.stop_asset_watch);
        prev.store(true, Ordering::SeqCst);
        self.asset_watcher = None;
        self.stop_asset_watch = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&self.stop_asset_watch);
        let (asset_tx, asset_rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            use notify::{recommended_watcher, RecursiveMode, Watcher};
            use std::path::Path;
            let (ntx, nrx) = std::sync::mpsc::channel();
            let mut watcher = match recommended_watcher(move |res| {
                let _ = ntx.send(res);
            }) {
                Ok(w) => w,
                Err(e) => {
                    tracing::error!("[Assets] Asset watcher failed: {}", e);
                    return;
                }
            };
            watcher.watch(Path::new("Content"), RecursiveMode::Recursive).ok();
            loop {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                match nrx.recv_timeout(Duration::from_millis(400)) {
                    Ok(Ok(event)) => {
                        for p in event.paths {
                            let s = p.to_string_lossy().to_string();
                            if s.ends_with(".obj")
                                || s.ends_with(".gltf")
                                || s.ends_with(".glb")
                                || s.ends_with(".png")
                                || s.ends_with(".jpg")
                                || s.ends_with(".jpeg")
                                || s.ends_with(".mat")
                                || s.ends_with(".material")
                                || s.ends_with(".prefab")
                            {
                                if asset_tx.send(s).is_err() {
                                    return;
                                }
                            }
                        }
                    }
                    Ok(Err(_)) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
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

fn resolve_primary_scene_path(preferred: &str) -> String {
    if !preferred.trim().is_empty() {
        let pref = std::path::Path::new(preferred.trim());
        if pref.exists() {
            return preferred.trim().to_string();
        }
    }
    let main = std::path::Path::new("scenes/main.scene");
    if main.exists() {
        return "scenes/main.scene".to_string();
    }
    if let Ok(rd) = std::fs::read_dir("scenes") {
        let mut paths: Vec<std::path::PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("scene"))
            .collect();
        paths.sort();
        if let Some(p) = paths.first() {
            return p.to_string_lossy().to_string();
        }
    }
    "scenes/main.scene".to_string()
}

impl ApplicationHandler for GameApp {
    // resumed() fires when the OS tells us we can draw.
    // On desktop this fires once right away. On Android/iOS it may fire multiple times.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create the OS window.
        let wp = editor_persist::load_window_prefs();
        let mut win_attrs = Window::default_attributes()
            .with_title("TrinityEngine")
            .with_window_icon(load_window_icon(APP_ICON_PATH));
        win_attrs = if let Some(ref p) = wp {
            win_attrs.with_inner_size(winit::dpi::PhysicalSize::new(
                p.width.max(960),
                p.height.max(540),
            ))
        } else {
            win_attrs.with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))
        };
        if let Some(p) = wp {
            if let (Some(x), Some(y)) = (p.pos_x, p.pos_y) {
                win_attrs =
                    win_attrs.with_position(winit::dpi::PhysicalPosition::new(x, y));
            }
        }
        let window = Arc::new(event_loop.create_window(win_attrs).expect("Could not create window"));

        // Update camera aspect ratio now that we know the window size.
        let phys = window.inner_size();
        self.input_state.camera.aspect = phys.width as f32 / phys.height as f32;

        // Build the GPU renderer (blocking — we are on the main thread).
        let renderer = pollster::block_on(Renderer::new(Arc::clone(&window)));
        let mut renderer = renderer;
        self.settings.render.apply_to_features(&mut renderer.features);
        if !self.settings.render.sky_hdr_path.trim().is_empty() {
            if let Err(e) = renderer.apply_sky_environment(&self.settings.render.sky_hdr_path) {
                self.error_log.push(format!("[Sky] Startup apply failed: {}", e));
            }
        }
        if !self.settings.runtime.gpu_scalability_tier.eq_ignore_ascii_case("auto") {
            renderer.features = renderer::RenderFeatures::from_tier_name(
                &self.settings.runtime.gpu_scalability_tier,
            );
            self.settings
                .sync_render_from_renderer_features(&renderer.features);
        }

        // Check if GPU is low-end and print a note.
        tracing::info!("[Engine] GPU: {:?}", renderer.adapter_info.name);
        tracing::info!("[Engine] Render settings loaded from engine_settings.toml");
        if self.jobs.enabled() {
            tracing::info!(
                "[Engine] Job system enabled (worker_threads={})",
                self.settings.runtime.worker_threads
            );
        }
        if self.assets.mesh_streaming.enabled() {
            tracing::info!("[Engine] Threaded mesh streaming queue enabled");
        }

        self.input_state.input.configure_gamepad(
            self.settings.input.gamepad_enabled,
            self.settings.input.left_stick_deadzone,
        );

        EditorShell::print_help();
        MaterialLibrary::print_help();

        self.window = Some(window);
        self.renderer = Some(renderer);
        if let (Some(window_ref), Some(renderer_ref)) = (self.window.as_ref(), self.renderer.as_ref()) {
            self.editor_ui = Some(EditorUi::new(window_ref, renderer_ref));
        }
        if let Some(ui) = self.editor_ui.as_mut() {
            ui.mark_icons_dirty();
        }
        self.app_stage = AppStage::ProjectHub;
        self.project_stage_started_at = std::time::Instant::now();
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
            WindowEvent::CloseRequested => {
                if let Some(w) = &self.window {
                    let sz = w.inner_size();
                    let pos = w.outer_position().ok();
                    editor_persist::save_window_prefs(&editor_persist::EditorWindowPrefs {
                        width: sz.width,
                        height: sz.height,
                        pos_x: pos.as_ref().map(|p| p.x),
                        pos_y: pos.as_ref().map(|p| p.y),
                    });
                }
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event: ke, .. } => {
                self.input_state.input.handle_key(ke.physical_key, ke.state == ElementState::Pressed);
                if ke.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = ke.physical_key {
                        match code {
                            KeyCode::F1 => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.bloom_enabled = !r.features.bloom_enabled;
                                    tracing::info!(
                                        "[Toggle] Bloom: {}",
                                        if r.features.bloom_enabled { "ON" } else { "OFF" }
                                    );
                                    tracing::info!("[Info] {}", editor::describe_toggle("bloom"));
                                }
                            }
                            KeyCode::F2 => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.ssao_enabled = !r.features.ssao_enabled;
                                    tracing::info!(
                                        "[Toggle] SSAO: {}",
                                        if r.features.ssao_enabled { "ON" } else { "OFF" }
                                    );
                                    tracing::info!("[Info] {}", editor::describe_toggle("ssao"));
                                }
                            }
                            KeyCode::F3 => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.volumetric_fog_enabled = !r.features.volumetric_fog_enabled;
                                    tracing::info!(
                                        "[Toggle] Volumetric Fog: {}",
                                        if r.features.volumetric_fog_enabled { "ON" } else { "OFF" }
                                    );
                                    tracing::info!("[Info] {}", editor::describe_toggle("fog"));
                                }
                            }
                            KeyCode::F4 => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.voxel_gi_enabled = !r.features.voxel_gi_enabled;
                                    tracing::info!(
                                        "[Toggle] Voxel GI Prototype: {}",
                                        if r.features.voxel_gi_enabled { "ON" } else { "OFF" }
                                    );
                                    tracing::info!("[Info] {}", editor::describe_toggle("voxel"));
                                }
                            }
                            KeyCode::F5 => {
                                self.settings.render.preset = editor::cycle_preset(self.settings.render.preset);
                                tracing::info!("[Preset] Switched to {:?}", self.settings.render.preset);
                                tracing::info!("[Preset] In full visual editor this becomes a one-click dropdown.");
                            }
                            KeyCode::F10 => {
                                self.editor_shell.visible = !self.editor_shell.visible;
                                tracing::info!(
                                    "[Editor] Shell {}",
                                    if self.editor_shell.visible { "OPEN" } else { "CLOSED" }
                                );
                            }
                            KeyCode::F11 => {
                                self.editor_shell.show_advanced = !self.editor_shell.show_advanced;
                                tracing::info!(
                                    "[Editor] Advanced panel {}",
                                    if self.editor_shell.show_advanced { "ON" } else { "OFF" }
                                );
                            }
                            KeyCode::BracketLeft => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.bloom_strength = (r.features.bloom_strength - 0.02).max(0.0);
                                    tracing::info!("[Inspector] Bloom strength -> {:.2}", r.features.bloom_strength);
                                }
                            }
                            KeyCode::BracketRight => {
                                if let Some(r) = &mut self.renderer {
                                    r.features.bloom_strength = (r.features.bloom_strength + 0.02).min(2.0);
                                    tracing::info!("[Inspector] Bloom strength -> {:.2}", r.features.bloom_strength);
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
                                        tracing::info!("[Animation] {:?} -> Idle", entity);
                                    }
                                }
                            }
                            KeyCode::KeyK => {
                                if let Some(entity) = self.selected_renderable {
                                    if let Ok(mut a) = self.world.get::<&mut Animator>(entity) {
                                        a.state = AnimState::Walk;
                                        tracing::info!("[Animation] {:?} -> Walk", entity);
                                    }
                                }
                            }
                            KeyCode::KeyL => {
                                if let Some(entity) = self.selected_renderable {
                                    if let Ok(mut a) = self.world.get::<&mut Animator>(entity) {
                                        a.state = AnimState::Run;
                                        tracing::info!("[Animation] {:?} -> Run", entity);
                                    }
                                }
                            }
                            KeyCode::KeyF => {
                                editor::add_foliage_patch(
                                    &mut self.world,
                                    &mut self.assets.meshes,
                                    &mut self.assets.mesh_cache,
                                );
                            }
                            KeyCode::KeyT => {
                                let cs = self.terrain_world.cell_size;
                                let w = self.terrain_world.grid.total_width;
                                let d = self.terrain_world.grid.total_depth;
                                let wx = self.terrain_cursor_x as f32 * cs - (w as f32 * cs * 0.5);
                                let wz = self.terrain_cursor_z as f32 * cs - (d as f32 * cs * 0.5);
                                self.terrain_world.raise(wx, wz, 4.0 * cs, 0.15);
                                tracing::info!("[Terrain] Raised terrain brush at ({}, {})", self.terrain_cursor_x, self.terrain_cursor_z);
                            }
                            KeyCode::KeyG => {
                                let cs = self.terrain_world.cell_size;
                                let w = self.terrain_world.grid.total_width;
                                let d = self.terrain_world.grid.total_depth;
                                let wx = self.terrain_cursor_x as f32 * cs - (w as f32 * cs * 0.5);
                                let wz = self.terrain_cursor_z as f32 * cs - (d as f32 * cs * 0.5);
                                self.terrain_world.lower(wx, wz, 4.0 * cs, 0.15);
                                tracing::info!("[Terrain] Lowered terrain brush at ({}, {})", self.terrain_cursor_x, self.terrain_cursor_z);
                            }
                            KeyCode::KeyY => {
                                if let Some(handle) = self.assets.mesh_cache.get("meshes/cube.obj").copied() {
                                    let cs = self.terrain_world.cell_size;
                                    let w = self.terrain_world.grid.total_width;
                                    let d = self.terrain_world.grid.total_depth;
                                    spawn_foliage_ring(
                                        &mut self.world,
                                        handle,
                                        self.terrain_cursor_x as f32 * cs - (w as f32 * cs * 0.5),
                                        self.terrain_cursor_z as f32 * cs - (d as f32 * cs * 0.5),
                                        4.0,
                                        24,
                                        true,
                                    );
                                    tracing::info!("[Terrain/Foliage] Added foliage ring with tree physics.");
                                } else {
                                    tracing::info!("[Terrain/Foliage] Load scene first so cube mesh exists.");
                                }
                            }
                            KeyCode::KeyU => {
                                let cs = self.terrain_world.cell_size;
                                let w = self.terrain_world.grid.total_width;
                                let d = self.terrain_world.grid.total_depth;
                                let removed = remove_nearby_foliage(
                                    &mut self.world,
                                    self.terrain_cursor_x as f32 * cs - (w as f32 * cs * 0.5),
                                    self.terrain_cursor_z as f32 * cs - (d as f32 * cs * 0.5),
                                    4.5,
                                );
                                tracing::info!("[Terrain/Foliage] Removed {} nearby foliage entities.", removed);
                            }
                            KeyCode::KeyP => self.snap_camera_to_selected(),
                            KeyCode::Home => self.focus_selected_frame(),
                            KeyCode::KeyO => {
                                self.input_state.orbit_mode = !self.input_state.orbit_mode;
                                tracing::info!(
                                    "[Camera] Orbit mode: {}",
                                    if self.input_state.orbit_mode { "ON" } else { "OFF" }
                                );
                            }
                            KeyCode::Minus => {
                                self.input_state.nav_speed_scalar = (self.input_state.nav_speed_scalar * 0.9).max(0.6);
                                tracing::info!("[Camera] Move speed: {:.2}", self.input_state.nav_speed_scalar);
                            }
                            KeyCode::Equal => {
                                self.input_state.nav_speed_scalar = (self.input_state.nav_speed_scalar * 1.12).min(80.0);
                                tracing::info!("[Camera] Move speed: {:.2}", self.input_state.nav_speed_scalar);
                            }
                            // ── Scene navigation: go back to previous scene ──
                            // Backspace triggers a transition back to the
                            // previously loaded scene (from the recent list).
                            KeyCode::Backspace => {
                                if !self.transition.is_active() {
                                    if let Some(prev_path) = self.scene_mgr.previous_scene().cloned() {
                                        let path_str = prev_path.to_string_lossy().to_string();
                                        self.transition.start_transition(&path_str);
                                        tracing::info!("[Scene] Backspace → transitioning to previous scene: {}", path_str);
                                    } else {
                                        tracing::info!("[Scene] No previous scene to go back to");
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    self.input_state.mouse_look_active = state == ElementState::Pressed;
                    self.input_state.last_cursor_pos = None;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 0.7,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                };
                if self.input_state.orbit_mode && self.selected_renderable.is_some() {
                    self.input_state.orbit_distance = (self.input_state.orbit_distance - amount * 0.75).clamp(0.8, 120.0);
                    let dir = (self.input_state.camera.position - self.input_state.camera.target).normalize_or_zero();
                    if dir.length_squared() > 1e-6 {
                        self.input_state.camera.position = self.input_state.camera.target + dir * self.input_state.orbit_distance;
                    }
                } else {
                    let forward = (self.input_state.camera.target - self.input_state.camera.position).normalize_or_zero();
                    self.input_state.camera.position += forward * amount;
                    self.update_camera_target_from_angles();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.input_state.mouse_look_active || self.input_state.mouse_look_latched {
                    if let Some(prev) = self.input_state.last_cursor_pos {
                        let dx = (position.x - prev.x) as f32;
                        let dy = (position.y - prev.y) as f32;
                        let mag = ((dx * dx + dy * dy).sqrt() / 24.0).clamp(0.0, 1.8);
                        let sensitivity = self.input_state.look_sensitivity * (1.0 + 0.45 * mag);
                        self.input_state.camera_yaw += dx * sensitivity;
                        self.input_state.camera_pitch =
                            (self.input_state.camera_pitch - dy * sensitivity).clamp(-1.5, 1.5);
                        if self.input_state.orbit_mode && self.selected_renderable.is_some() {
                            let dir = glam::Vec3::new(
                                self.input_state.camera_yaw.cos() * self.input_state.camera_pitch.cos(),
                                self.input_state.camera_pitch.sin(),
                                self.input_state.camera_yaw.sin() * self.input_state.camera_pitch.cos(),
                            )
                            .normalize_or_zero();
                            self.input_state.camera.position = self.input_state.camera.target - dir * self.input_state.orbit_distance;
                        } else {
                            self.update_camera_target_from_angles();
                        }
                    }
                    self.input_state.last_cursor_pos = Some(position);
                } else {
                    self.input_state.last_cursor_pos = None;
                }
            }

            WindowEvent::Resized(new_size) => {
                self.input_state.camera.aspect = new_size.width as f32 / new_size.height as f32;
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
                self.input_state.input.update_gamepads();

                // ── Begin frame event ──────────────────────────────────────
                self.events.emit(BeginFrameEvent {
                    frame_index: self.frame_index,
                    delta_time: dt,
                });

                // ── Environment update ─────────────────────────────────────
                // Advance time of day, update sky/weather/clouds from new state.
                // This runs every frame (even in editor mode) so the sky preview
                // stays live. The time speed can be set to 0 to pause.
                let prev_hour = self.env.time_of_day.hour;
                self.env.time_of_day.advance(dt);
                self.env.sky.update_from_time(&self.env.time_of_day);
                self.env.clouds.update(&self.env.weather, &self.env.time_of_day, dt);
                self.env.lightning.update(&self.env.weather, dt);

                // Play thunder sound + emit event when lightning strikes.
                if self.env.lightning.thunder_just_fired {
                    #[cfg(feature = "audio")]
                    if let Some(audio) = &mut self.audio {
                        let vol = 0.4 + 0.6 * self.env.weather.intensity;
                        audio.play_thunder(vol);
                    }
                    self.events.emit(ThunderEvent {
                        intensity: self.env.lightning.flash_intensity,
                        delay: self.env.lightning.thunder_delay,
                    });
                }

                // ── Weather Zone evaluation ─────────────────────────────────
                // Check if any WeatherZone entity is near the camera and
                // override the global weather accordingly.
                {
                    use crate::environment::weather_zone::evaluate_weather_at;
                    let cam = [self.input_state.camera.position.x, self.input_state.camera.position.y, self.input_state.camera.position.z];
                    let zone_weather = evaluate_weather_at(&self.world, &self.env.weather, cam);
                    self.env.weather.condition = zone_weather.condition;
                    self.env.weather.intensity = zone_weather.intensity;
                }

                // ── Wind Zone evaluation ────────────────────────────────────
                // Check if any WindZone entity is near the camera and override
                // the global wind accordingly.
                {
                    use crate::environment::wind_zone::evaluate_wind_at;
                    let cam = [self.input_state.camera.position.x, self.input_state.camera.position.y, self.input_state.camera.position.z];
                    let global_dir = [self.env.weather.wind_direction.x, 0.0, self.env.weather.wind_direction.y];
                    let (dir, str) = evaluate_wind_at(
                        &self.world,
                        global_dir,
                        self.env.weather.wind_strength,
                        cam,
                    );
                    self.env.weather.wind_direction = glam::Vec2::new(dir[0], dir[2]);
                    self.env.weather.wind_strength = str;
                }

                // Emit environment events through the event bus.
                // TimeOfDayChangedEvent fires when the hour changes by >= 0.1 (about 2.4 minutes of game time).
                if (self.env.time_of_day.hour - prev_hour).abs() > 0.1 {
                    self.events.emit(TimeOfDayChangedEvent {
                        time: self.env.time_of_day.hour,
                    });
                }

                // ── Shader hot-reload check ────────────────────────────────
                // Check if any .wgsl files were modified on disk and recompile
                // them. Returns true if a shader was reloaded.
                if let Some(renderer) = &self.renderer {
                    self.shader_mgr.check_hot_reload(&renderer.device);
                }

                // ── Audio system update ────────────────────────────────────
                // Clean up finished sounds each frame; sync 3D listener.
                if let Some(audio) = &mut self.audio {
                    let cp = self.input_state.camera.position;
                    let forward = (self.input_state.camera.target - self.input_state.camera.position).normalize_or_zero();
                    audio.set_listener_position([cp.x, cp.y, cp.z]);
                    audio.set_listener_forward([forward.x, forward.y, forward.z]);
                    audio.update();
                }

                let asset_start = std::time::Instant::now();
                for (path, result) in self.assets.mesh_streaming.poll_loaded() {
                    match result {
                        Ok(mesh) => {
                            if let Some(handle) = self.assets.mesh_cache.get(&path) {
                                self.assets.meshes.replace(handle, mesh);
                            } else {
                                let handle = self.assets.meshes.add(mesh);
                                self.assets.mesh_cache.insert(path.clone(), handle);
                            }
                            tracing::info!("[Assets] Mesh ready: {}", path);
                        }
                        Err(e) => self.push_error(format!("[Assets] Mesh load failed {}: {}", path, e)),
                    }
                }
                self.assets.mesh_streaming.pump_requests();
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
                        // Plugins live under Content/Scripts/plugins — hot-reload
                        // them through the plugin host so a changed file re-runs
                        // start() cleanly (no stale event handlers or timers).
                        let norm_path = path.replace('\\', "/");
                        let is_plugin = norm_path.contains("/plugins/");
                        match self.scripts.reload_plugin(&path) {
                            Ok(true) => tracing::info!("[Hot] Plugin reloaded: {}", path),
                            _ if is_plugin => {
                                // A plugin file changed but wasn't loaded before.
                                let _ = self.scripts.load_plugin(&path);
                            }
                            _ => match self.scripts.reload_script(&path) {
                                Ok(_) => tracing::info!("[Hot] Script reloaded: {}", path),
                                Err(e) => pending_errors.push(format!("[Hot] Script error {}: {}", path, e)),
                            },
                        }
                    }
                    for e in pending_errors {
                        self.push_error(e);
                    }
                }
                if !self.asset_hot_reload_enabled {
                    self.stop_asset_watch.store(true, Ordering::SeqCst);
                    self.asset_watcher = None;
                } else if self.asset_watcher.is_none() {
                    self.start_asset_watcher();
                }
                if let Some(rx) = &self.asset_watcher {
                    let mut pending_errors: Vec<String> = Vec::new();
                    let mut changed_assets = HashSet::new();
                    while let Ok(path) = rx.try_recv() {
                        changed_assets.insert(path.replace('\\', "/"));
                    }
                    for norm in changed_assets {
                        self.mark_editor_content_dirty();
                        if norm.ends_with(".obj") || norm.ends_with(".gltf") || norm.ends_with(".glb") {
                            if let Some(handle) = self.assets.mesh_cache.get(&norm).copied() {
                                match Mesh::load(&norm) {
                                    Ok(mesh) => {
                                        self.assets.meshes.replace(&handle, mesh);
                                        tracing::info!("[Hot] Mesh reloaded: {}", norm);
                                    }
                                    Err(e) => pending_errors.push(format!("[Hot] Mesh reload failed {}: {}", norm, e)),
                                }
                            }
                        } else if norm.ends_with(".png")
                            || norm.ends_with(".jpg")
                            || norm.ends_with(".jpeg")
                        {
                            if let Some(renderer) = self.renderer.as_ref() {
                                renderer.invalidate_texture_path(&norm);
                            }
                            tracing::info!("[Hot] Texture reloaded: {}", norm);
                        } else if norm.ends_with(".mat") || norm.ends_with(".material") {
                            match self.assets.materials.reload_material_file(std::path::Path::new(&norm)) {
                                Ok(()) => tracing::info!("[Hot] Material reloaded: {}", norm),
                                Err(e) => pending_errors.push(format!("[Hot] Material reload failed {}: {}", norm, e)),
                            }
                        } else if norm.ends_with(".prefab") {
                            match self.assets.prefab_registry.reload_file(std::path::Path::new(&norm)) {
                                Ok(()) => tracing::info!("[Hot] Prefab reloaded: {}", norm),
                                Err(e) => pending_errors.push(format!("[Hot] Prefab reload failed {}: {}", norm, e)),
                            }
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
                    let mut content_dirty = false;
                    while let Ok(path) = rx.try_recv() {
                        self.scene_list_dirty = true;
                        tracing::info!("[Hot] Scene changed: {}", path);
                        match self.scene_mgr.build(
                            &mut self.world,
                            &mut self.assets.meshes,
                            &mut self.assets.mesh_cache,
                            Some(&self.assets.prefab_registry),
                        ) {
                            Ok(_)  => {
                                tracing::info!("[Hot] Scene rebuilt");
                                content_dirty = true;
                                pending_toasts.push(format!("Scene hot reloaded: {}", path));
                            }
                            Err(e) => pending_errors.push(format!("[Hot] Scene error: {}", e)),
                        }
                    }
                    if content_dirty {
                        self.mark_editor_content_dirty();
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
                    self.nav_grid.rebuild_from_heights(&self.terrain_world);
                    self.navmesh = navmesh::NavMesh::from_terrain(&self.nav_grid, &self.terrain_world);
                    self.nav_rebuild_requested = false;
                }

                // ── Reset physics ground flag ──────────────────────────────
                // Must happen before physics_system so entities that walk off
                // edges fall correctly on the next frame.
                for body in self.world.query_mut::<&mut RigidBody>() {
                    body.on_ground = false;
                }

                // ── Systems ────────────────────────────────────────────────
                // TODO: Migrate these system calls to EngineSystems.scheduler:
                //   - scripting_system
                //   - character_controller_system
                //   - physics_system
                //   - water_trigger_system
                //   - ragdoll_system
                //   - animation_system
                //   - animation_blending_system
                // Editor Scene = authoring mode (no gameplay simulation).
                // Game Preview = runs scripts/physics/animation, like Unreal PIE.
                let run_sim = self.game_preview_mode && (!self.sim_paused || self.sim_step_once);

                // ── Snapshot capture on Play ─────────────────────────────
                // Save all entity state before the simulation starts so we
                // can restore it when the user presses Stop.
                if self.game_preview_mode && !self.prev_game_preview_mode {
                    self.capture_play_snapshot();
                    self.apply_player_start_on_preview_begin();
                }

                // ── Snapshot restore on Stop ─────────────────────────────
                // When exiting Game Preview, restore entities to their
                // pre-simulation positions/rotations/velocities.
                if !self.game_preview_mode && self.prev_game_preview_mode {
                    self.restore_play_snapshot();
                }

                self.prev_game_preview_mode = self.game_preview_mode;
                let mut script_time = std::time::Duration::ZERO;
                let mut physics_time = std::time::Duration::ZERO;
                if run_sim {
                    let script_start = std::time::Instant::now();
                    if self.script_skip_frames_remaining > 0 {
                        self.script_skip_frames_remaining -= 1;
                    } else {
                        let (screen_w, screen_h) = self.window.as_ref()
                            .map(|w| { let s = w.inner_size(); (s.width as f32, s.height as f32) })
                            .unwrap_or((1280.0, 720.0));
                        let camera_fov = self.input_state.camera.fov_degrees;
                        scripting_system(
                            &mut self.world,
                            &mut self.scripts,
                            &self.input_state.input,
                            self.input_state.camera.position.to_array(),
                            self.input_state.camera.target.to_array(),
                            dt,
                            self.audio.as_mut(),
                            &self.nav_grid,
                            &self.navmesh,
                            &mut self.ai_registry,
                            &mut self.terrain_world,
                            &mut self.assets.meshes,
                            &mut self.env.weather,
                            &mut self.particles,
                            &mut self.levels,
                            &mut self.boids,
                            screen_w,
                            screen_h,
                            camera_fov,
                        );
                        self.scripts.drain_destroys(&mut self.world);
                        self.demo_plugin.tick();
                        let _ = self.scripts.tick_timers(dt);
                        let _ = self.scripts.tick_plugins(dt);
                        self.scripts.tick_cinematics(dt);
                        if let Some((pos, target)) = self.scripts.consume_camera_request() {
                            self.input_state.camera.position = glam::Vec3::from_array(pos);
                            self.input_state.camera.target = glam::Vec3::from_array(target);
                            let mut dir =
                                (self.input_state.camera.target - self.input_state.camera.position).normalize_or_zero();
                            if dir.length_squared() < 1e-6 {
                                dir = glam::Vec3::new(0.0, -0.2, -1.0).normalize();
                            }
                            self.input_state.camera_yaw = dir.z.atan2(dir.x);
                            self.input_state.camera_pitch = dir.y.asin();
                        }
                        let skip_n = self.scripts.consume_frame_skip_request();
                        if skip_n > 0 {
                            self.script_skip_frames_remaining = skip_n;
                        }
                    }
                    script_time = script_start.elapsed();

                    let physics_start = std::time::Instant::now();
                    let sim_time = self.start_time.elapsed().as_secs_f32();
                    let divisor = self.settings.runtime.foliage_wind_update_divisor.max(1) as u64;
                    let _wind_this_frame = (self.frame_index % divisor) == 0;
                    // Bullet/high-speed safety: small fixed substeps reduce tunneling.
                    let speed_peak = self
                        .world
                        .query::<&RigidBody>()
                        .iter()
                        .map(|b| b.velocity_x.abs().max(b.velocity_y.abs()).max(b._velocity_z.abs()))
                        .fold(0.0f32, f32::max);
                    let max_sub = self.settings.runtime.physics_max_substeps.clamp(1, 12);
                    let substeps = if self.settings.runtime.physics_ccd_enabled {
                        (((speed_peak * dt) / 0.45).ceil() as u32).clamp(1, max_sub)
                    } else {
                        1
                    };
                    let sub_dt = dt / substeps as f32;
                    // Character controller: applies input-driven movement, ground detection, jump.
                    character_controller_system(&mut self.world, sub_dt);
                    let mut collision_events = Vec::new();
                    for s in 0..substeps {
                        let collisions = physics_system(
                            &mut self.world,
                            sub_dt,
                            sim_time + sub_dt * s as f32,
                            &self.jobs,
                            &self.settings.runtime,
                        );
                        collision_events.extend(collisions);
                    }
                    if let Err(err) = self.scripts.dispatch_collision_events(&collision_events) {
                        tracing::error!("[Scripting] Collision callback error: {}", err);
                    }
                    // ── Emit collision events through EventBus ──────────────
                    // This enables loose coupling: audio, VFX, editor, etc.
                    // can listen for collisions without the physics system
                    // knowing they exist.
                    for cp in &collision_events {
                        use crate::components::CollisionPhase;
                        match cp.phase {
                            CollisionPhase::Started => {
                                self.events.emit(core::CollisionStartedEvent {
                                    entity_a_bits: cp.entity_a.to_bits().get(),
                                    entity_b_bits: cp.entity_b.to_bits().get(),
                                    normal_x: cp.normal[0],
                                    normal_y: cp.normal[1],
                                    normal_z: cp.normal[2],
                                    penetration: cp.penetration,
                                });
                            }
                            CollisionPhase::Ended => {
                                self.events.emit(core::CollisionEndedEvent {
                                    entity_a_bits: cp.entity_a.to_bits().get(),
                                    entity_b_bits: cp.entity_b.to_bits().get(),
                                });
                            }
                            CollisionPhase::Ongoing => {}
                        }
                    }
                    physics_time = physics_start.elapsed();
                    // Water trigger: detect dynamic entities entering water surfaces.
                    let splash_events = water_trigger_system(&mut self.world);
                    for splash in &splash_events {
                        self.events.emit(core::WaterSplashEvent {
                            entity_bits: splash.entity_bits,
                            water_entity_bits: splash.water_entity_bits,
                            impact_velocity: splash.impact_velocity,
                            splash_intensity: splash.splash_intensity,
                        });
                        // Feed splash visual manager with resolved position.
                        let water_pos = self.world.query_mut::<(hecs::Entity, &Position)>()
                            .into_iter()
                            .find(|(e, _)| e.to_bits().get() == splash.water_entity_bits)
                            .map(|(_, p)| [p.x, p.y, p.z])
                            .unwrap_or([0.0, 0.0, 0.0]);
                        let sim_t = self.start_time.elapsed().as_secs_f32();
                        self.splash_manager.on_splash(
                            water_pos,
                            splash.impact_velocity,
                            splash.splash_intensity,
                            sim_t,
                        );
                    }
                    // Ragdoll: post-physics bone constraint solving.
                    ragdoll_system(&mut self.world, dt);

                    ai_system(&mut self.world, &mut self.ai_registry, &self.nav_grid, Some(&self.navmesh), dt, sim_time);

                    boids::boids_system(&mut self.world, &mut self.boids, dt);

                    animation_system(&mut self.world, dt, &self.jobs);
                    // Skeletal animation blending: reads BT "ai_state" from blackboard,
                    // triggers crossfade transitions, evaluates blended joint matrices.
                    animation_blending_system(&mut self.world, dt);
                    // Node-based animation graph: evaluates state machines per-layer,
                    // selects states based on parameters (speed, is_attacking, etc.),
                    // feeds transitions into SkeletalAnimator for crossfade.
                    anim_graph_system(&mut self.world, dt);
                    // Flood system: advances water level toward target, logs newly
                    // submerged entities. Runs after animation so submerged VFX
                    // can react to the updated water_level on the same frame.
                    flood_system(&mut self.levels.flood, &mut self.world, dt);
                    if self.sim_step_once {
                        self.sim_step_once = false;
                        self.sim_paused = true;
                    }
                }

                // Seamless free-fly camera: RMB look + WASD move, Shift sprint, Space/Ctrl up/down.
                {
                    let forward = (self.input_state.camera.target - self.input_state.camera.position).normalize_or_zero();
                    let right = forward.cross(glam::Vec3::Y).normalize_or_zero();
                    let up = glam::Vec3::Y;
                    let mut move_dir = glam::Vec3::ZERO;
                    if self.input_state.input.is_held(KeyCode::KeyW) { move_dir += forward; }
                    if self.input_state.input.is_held(KeyCode::KeyS) { move_dir -= forward; }
                    if self.input_state.input.is_held(KeyCode::KeyD) { move_dir += right; }
                    if self.input_state.input.is_held(KeyCode::KeyA) { move_dir -= right; }
                    if self.input_state.input.is_held(KeyCode::Space) { move_dir += up; }
                    if self.input_state.input.is_held(KeyCode::ControlLeft) || self.input_state.input.is_held(KeyCode::ControlRight) {
                        move_dir -= up;
                    }
                    let mut speed = self.input_state.nav_speed_scalar;
                    if self.input_state.input.is_held(KeyCode::ShiftLeft) || self.input_state.input.is_held(KeyCode::ShiftRight) {
                        speed *= 2.2;
                    }
                    if self.input_state.input.is_held(KeyCode::AltLeft) || self.input_state.input.is_held(KeyCode::AltRight) {
                        speed *= 0.35;
                    }
                    let desired_velocity = if move_dir.length_squared() > 0.0 {
                        move_dir.normalize() * speed
                    } else {
                        glam::Vec3::ZERO
                    };
                    let accel_blend = (1.0 - (-dt * 10.0).exp()).clamp(0.0, 1.0);
                    self.input_state.camera_move_velocity =
                        self.input_state.camera_move_velocity.lerp(desired_velocity, accel_blend);
                    if self.input_state.camera_move_velocity.length_squared() < 1e-5 {
                        self.input_state.camera_move_velocity = glam::Vec3::ZERO;
                    }
                    let delta = self.input_state.camera_move_velocity * dt.max(0.0);
                    if delta.length_squared() > 0.0 {
                        if self.input_state.orbit_mode && self.selected_renderable.is_some() {
                            self.input_state.camera.target += delta;
                            self.input_state.camera.position += delta;
                        } else {
                            self.input_state.camera.position += delta;
                            self.update_camera_target_from_angles();
                        }
                    }
                }

                // ── Loading screen update ────────────────────────────────────
                // Advance fade transitions each frame, even outside sim,
                // so the loading screen fades out smoothly after streaming.
                self.levels.loading_screen.update(dt);

                // ── Level streaming check ──────────────────────────────────
                // Periodically check player distance to registered levels and
                // queue load/unload operations. This is the core of the
                // streaming level system — levels load when the player is
                // nearby and unload when far away.
                {
                    let player_pos = self.input_state.camera.position.to_array();
                    if let Some(streaming_result) = levels::check_streaming(
                        &self.levels.level_manager,
                        player_pos,
                        dt,
                        &mut self.levels.streaming_config,
                    ) {
                        // Unload levels first (free memory/entity slots).
                        for level_id in &streaming_result.levels_to_unload {
                            if self.levels.level_manager.unload_level(*level_id) {
                                tracing::info!("[Streaming] Unloaded level {}", level_id);
                            }
                        }
                        // Show loading screen before loading new levels so the
                        // player sees visual feedback during the transition.
                        if !streaming_result.levels_to_load.is_empty() {
                            self.levels.loading_screen.show("Loading level...");
                        }
                        // Load levels (mark as loaded; entities are spawned
                        // by the scene system when needed).
                        for level_id in &streaming_result.levels_to_load {
                            if self.levels.level_manager.load_level(*level_id) {
                                tracing::info!("[Streaming] Loaded level {}", level_id);
                            }
                        }
                        // Hide loading screen once all levels have been loaded.
                        // The fade-out is animated by loading_screen.update(dt)
                        // which runs every frame above.
                        if !streaming_result.levels_to_load.is_empty() {
                            self.levels.loading_screen.hide();
                        }
                    }
                }

                // ── Render ─────────────────────────────────────────────────
                let render_start = std::time::Instant::now();
                let mut draw_stats = renderer::DrawStats::default();
                // Sync environment state (sun position, fog) into renderer.
                if let Some(renderer) = &mut self.renderer {
                    renderer.apply_environment(&self.env.time_of_day, &self.env.weather, &self.env.sky, &self.env.clouds, &self.env.lightning);
                }
                // ── Particle system update ──────────────────────────────────
                // Sync weather → emitters (enable rain/snow/mist, adjust intensity).
                // Then advance particle physics. Particles are passed to draw_world as GpuParticle slice.
                self.particles.apply_weather(
                    self.particle_indices,
                    self.env.weather.condition,
                    self.env.weather.intensity,
                    glam::Vec3::new(self.env.weather.wind_direction.x, 0.0, self.env.weather.wind_direction.y),
                    self.env.weather.wind_strength,
                );
                // ── Fire source sync ──────────────────────────────────────────
                // Query all entities with FireSource + Position and create/update
                // fire/smoke/ember particle emitters attached to them.
                {
                    let mut seen = std::collections::HashSet::new();
                    for (e, pos, fs) in self.world.query::<(hecs::Entity, &components::Position, &components::FireSource)>().iter() {
                        let bits = u64::from(e.to_bits());
                        seen.insert(bits);
                        self.particles.add_fire_source(bits, glam::Vec3::new(pos.x, pos.y, pos.z), fs.intensity);
                    }
                    // Remove fire sources for entities that no longer have FireSource.
                    let to_remove: Vec<u64> = self.particles.fire_source_keys()
                        .copied()
                        .filter(|k| !seen.contains(k))
                        .collect();
                    for k in to_remove {
                        self.particles.remove_fire_source(k);
                    }
                }
                self.particles.update(dt, glam::Vec3::new(
                    self.input_state.camera.position.x, self.input_state.camera.position.y, self.input_state.camera.position.z,
                ), self.start_time.elapsed().as_secs_f32());
                // ── Dynamic light emission for fire and lava surfaces ─────
                // Every frame, query all FireSurface and LavaSurface entities and
                // ensure they have a PointLight whose color / intensity matches
                // their emissive_light fields.  Flicker is applied by sampling a
                // cheap sin-based function driven by elapsed time and each entity's
                // unique ID so nearby fires don't pulse in lock-step.
                {
                    let sim_t = self.start_time.elapsed().as_secs_f32();
                    // Collect (entity, is_fire) pairs so we can query the world
                    // for each surface type and spawn / update a PointLight.
                    // We build a Vec first to avoid borrowing issues with the world.
                    struct EmissiveEntry {
                        entity:  hecs::Entity,
                        color:   [f32; 3],
                        base_strength: f32,
                        radius:  f32,
                        flicker: f32,
                    }
                    let mut entries: Vec<EmissiveEntry> = Vec::new();
                    // Query FireSurface entities — flicker is derived from the
                    // surface's flicker_strength field.
                    for (e, fs) in self.world.query::<(hecs::Entity, &components::FireSurface)>().iter() {
                        entries.push(EmissiveEntry {
                            entity: e,
                            color:  fs.emissive_light_color,
                            base_strength: fs.emissive_light_strength,
                            radius: fs.emissive_light_radius,
                            flicker: fs.flicker_strength,
                        });
                    }
                    // Query LavaSurface entities — lava flickers less than fire,
                    // using a small fixed flicker factor of 0.08 for subtle
                    // variation in the glow.
                    for (e, ls) in self.world.query::<(hecs::Entity, &components::LavaSurface)>().iter() {
                        entries.push(EmissiveEntry {
                            entity: e,
                            color:  ls.emissive_light_color,
                            base_strength: ls.emissive_light_strength,
                            radius: ls.emissive_light_radius,
                            flicker: 0.08,
                        });
                    }
                    // Now spawn or update PointLight for each entry.
                    for entry in &entries {
                        // Compute a unique per-entity flicker offset so that
                        // multiple fires/lava pools don't pulse in sync.
                        let id_f = entry.entity.to_bits().get() as f32;
                        // Two-phase sin for a more organic, less periodic feel.
                        let flicker_a = (sim_t * 4.7 + id_f * 1.3).sin();
                        let flicker_b = (sim_t * 7.3 + id_f * 2.1).sin();
                        let flicker_mod = 1.0 + entry.flicker * (0.6 * flicker_a + 0.4 * flicker_b);
                        let final_intensity = entry.base_strength * flicker_mod;
                        // Check if entity already has a PointLight — if so,
                        // update it in-place to avoid re-creating the component.
                        if let Ok(mut pl) = self.world.get::<&mut components::PointLight>(entry.entity) {
                            pl.color     = entry.color;
                            pl.intensity = final_intensity;
                            pl.range     = entry.radius;
                            pl.light_type = 1.0; // ensure point light type
                        } else {
                            // Entity has no PointLight yet — insert one.  The
                            // renderer's multi-light pass will pick it up.
                            let _ = self.world.insert(entry.entity, (
                                components::PointLight {
                                    color:          entry.color,
                                    intensity:      final_intensity,
                                    range:          entry.radius,
                                    light_type:     1.0, // point
                                    spot_angle:     45.0,
                                    shadow_casting: false,
                                },
                            ));
                        }
                    }
                }
                let gpu_particles = self.particles.gpu_instances();
                if self.app_stage == AppStage::ProjectHub {
                    let hub_open = match (self.window.as_ref(), self.editor_ui.as_mut()) {
                        (Some(w), Some(ui)) => {
                            let elapsed =
                                self.project_stage_started_at.elapsed().as_secs_f32();
                            ui.begin_project_hub(w, elapsed)
                        }
                        _ => None,
                    };
                    if let Some(proj) = hub_open {
                        let proj = std::fs::canonicalize(&proj).unwrap_or(proj);
                        self.switch_to_project(proj);
                        self.app_stage = AppStage::EditorLoading;
                        self.project_stage_started_at =
                            std::time::Instant::now();
                    }
                }
                self.refresh_available_scenes_if_needed();
                if self.app_stage == AppStage::BootSplash
                    && self.project_stage_started_at.elapsed().as_millis() > 1100
                {
                    self.app_stage = AppStage::ProjectHub;
                    self.project_stage_started_at = std::time::Instant::now();
                }
                let mut content_dirty_after_render = false;
                let mut return_to_hub_after_render = false;
                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    if self.app_stage == AppStage::BootSplash {
                        if let Some(ui) = self.editor_ui.as_mut() {
                            ui.begin_editor_loading(
                                window,
                                self.project_stage_started_at.elapsed().as_secs_f32(),
                            );
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
                            &self.assets.meshes,
                            &self.input_state.camera,
                            &self.jobs,
                            &mut self.instancing,
                            Some(&gpu_particles),
                            Some(&mut draw_ui),
                        );
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
                        return;
                    }
                    if self.app_stage == AppStage::ProjectHub {
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
                            &self.assets.meshes,
                            &self.input_state.camera,
                            &self.jobs,
                            &mut self.instancing,
                            Some(&gpu_particles),
                            Some(&mut draw_ui),
                        );
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
                        return;
                    }
                    if self.app_stage == AppStage::EditorLoading
                        && self.project_stage_started_at.elapsed().as_millis() > 900
                    {
                        self.app_stage = AppStage::EditorReady;
                    }
                    if let Some(ui) = self.editor_ui.as_mut() {
                        // Snapshot weather condition before editor can modify it.
                        let prev_condition = self.env.weather.condition;
                        let prev_intensity = self.env.weather.intensity;
                        let mut frame_args = UiFrameArgs {
                            world: &mut self.world,
                            settings: &mut self.settings,
                            renderer,
                            camera: &self.input_state.camera,
                            profiler: &self.profiler,
                            mesh_cache: &mut self.assets.mesh_cache,
                            meshes: &mut self.assets.meshes,
                            materials: &mut self.assets.materials,
                            selected_renderable: &mut self.selected_renderable,
                            terrain_world: &mut self.terrain_world,
                            terrain_cursor_x: self.terrain_cursor_x,
                            terrain_cursor_z: self.terrain_cursor_z,
                            app_time_seconds: self.start_time.elapsed().as_secs_f32(),
                            sim_paused: &mut self.sim_paused,
                            sim_step_once: &mut self.sim_step_once,
                            game_preview_mode: &mut self.game_preview_mode,
                            mouse_look_latched: &mut self.input_state.mouse_look_latched,
                            error_log: &mut self.error_log,
                            nav_grid: &mut self.nav_grid,
                            nav_rebuild_requested: &mut self.nav_rebuild_requested,
                            scripts: &mut self.scripts,
                            scripts_dir: CONTENT_SCRIPTS_DIR,
                            script_hot_reload_enabled: &mut self.script_hot_reload_enabled,
                            preferred_script_editor: &mut self.preferred_script_editor,
                            asset_hot_reload_enabled: &mut self.asset_hot_reload_enabled,
                            return_to_hub: &mut self.request_return_to_hub,
                            scene_path: &mut self.scene_mgr.scene_path,
                            available_scene_paths: &self.available_scene_paths,
                            requested_scene_switch: &mut self.requested_scene_switch,
                            camera_nav_speed: self.input_state.nav_speed_scalar,
                            time_of_day: &mut self.env.time_of_day,
                            weather: &mut self.env.weather,
                            audio: &mut self.audio,
                        };
                        if self.app_stage == AppStage::EditorLoading {
                            ui.begin_editor_loading(window, self.project_stage_started_at.elapsed().as_secs_f32());
                        } else {
                            ui.begin_and_build(window, &mut frame_args);
                        }
                        // Emit WeatherChangedEvent if the editor changed weather.
                        if self.env.weather.condition != prev_condition
                            || (self.env.weather.intensity - prev_intensity).abs() > 0.01
                        {
                            self.events.emit(WeatherChangedEvent {
                                weather_type: format!("{:?}", self.env.weather.condition),
                                intensity: self.env.weather.intensity,
                            });
                        }
                    }
                    if let Some(scene_path) = self.requested_scene_switch.take() {
                        // Instead of loading immediately, start a fade-to-black transition.
                        // The actual load happens when the screen is fully black (in transition.update).
                        if !self.transition.is_active() {
                            self.transition.start_transition(&scene_path);
                        } else {
                            // A transition is already in progress — ignore the request.
                            self.error_log.push(format!("[Scene] Transition already in progress, ignoring switch to {}", scene_path));
                        }
                    }
                    // ── Scene transition update ─────────────────────────────
                    // Each frame, advance the fade-to-black effect. When the
                    // screen is fully black, transition.update() returns the
                    // pending scene path — that's when we actually load it.
                    if let Some(pending_path) = self.transition.update(dt) {
                        // Screen is fully black — perform the scene swap now.
                        self.scene_mgr.scene_path = pending_path.clone();
                        self.assets.mesh_cache.clear();
                        match self.scene_mgr.build(
                            &mut self.world,
                            &mut self.assets.meshes,
                            &mut self.assets.mesh_cache,
                            Some(&self.assets.prefab_registry),
                        ) {
                            Ok(()) => {
        self.nav_grid.rebuild_from_heights(&self.terrain_world);
                                        self.navmesh = navmesh::NavMesh::from_terrain(&self.nav_grid, &self.terrain_world);
                                self.selected_renderable = None;
                                content_dirty_after_render = true;
                                self.error_log.push(format!(
                                    "[Scene] Loaded '{}' via transition",
                                    pending_path
                                ));
                            }
                            Err(e) => {
                                self.error_log.push(format!(
                                    "[Scene] Transition load failed: {}",
                                    e
                                ));
                            }
                        }
                    }
                    if self.request_return_to_hub {
                        self.request_return_to_hub = false;
                        return_to_hub_after_render = true;
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
                        &self.assets.meshes,
                        &self.input_state.camera,
                        &self.jobs,
                        &mut self.instancing,
                        Some(&gpu_particles),
                        Some(&mut draw_ui),
                    );
                }
                if content_dirty_after_render {
                    self.mark_editor_content_dirty();
                }
                if return_to_hub_after_render {
                    self.stop_project_watchers();
                    self.app_stage = AppStage::ProjectHub;
                    self.project_stage_started_at = std::time::Instant::now();
                    self.sim_paused = true;
                    self.error_log.push(format!(
                        "[Editor] Returned to Project Hub (engine v {}).",
                        TRINITY_ENGINE_VERSION
                    ));
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

                // ── Flush event bus ────────────────────────────────────────
                // Process all events emitted this frame. This is the single
                // point where events are dispatched. Systems should NOT call
                // flush() themselves.
                self.events.flush();
                self.events.emit(EndFrameEvent {
                    frame_index: self.frame_index,
                });
                // Process end-frame events immediately.
                self.events.flush();
                self.events.reset_frame_stats();
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
    // tracing MUST be initialised before any wgpu calls so GPU errors are visible.
    // RUST_LOG env var controls verbosity, e.g. RUST_LOG=info or RUST_LOG=Triengine=debug
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let event_loop = EventLoop::new().expect("Could not create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = GameApp::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}
