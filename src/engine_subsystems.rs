// src/engine_subsystems.rs
// Extracted subsystem groups from the monolithic GameApp.
// Each struct bundles related fields, reducing GameApp's field count
// from ~78 to ~30 by grouping logically coupled state.

use std::collections::HashMap;

use crate::camera::Camera3D;
use crate::assets::{AssetStore, Mesh, MeshStreamingQueue, Handle};
use crate::materials::MaterialLibrary;
use crate::scene::PrefabRegistry;
use crate::levels::{LevelManager, StreamingConfig, WorldStateManager, LoadingScreen};
use crate::environment::flood::FloodSystem;
use crate::environment::time_of_day::TimeOfDay;
use crate::environment::sky::SkyParams;
use crate::environment::weather::WeatherState;
use crate::environment::clouds::CloudParams;
use crate::environment::lightning::LightningState;

use glam::Vec3;
use winit::dpi::PhysicalPosition;

// ── EnvironmentState ─────────────────────────────────────────────────────────
// Bundles all atmospheric/environmental state: time of day, sky, weather,
// clouds, and lightning. Updated each frame by the environment system.

pub struct EnvironmentState {
    pub time_of_day: TimeOfDay,
    pub sky: SkyParams,
    pub weather: WeatherState,
    pub clouds: CloudParams,
    pub lightning: LightningState,
}

impl EnvironmentState {
    pub fn new() -> Self {
        Self {
            time_of_day: TimeOfDay::new(),
            sky: SkyParams::default(),
            weather: WeatherState::default(),
            clouds: CloudParams::default(),
            lightning: LightningState::new(),
        }
    }

    /// Advance time, update sky/weather/clouds/lightning for one frame.
    pub fn update(&mut self, dt: f32) {
        self.time_of_day.advance(dt);
        self.sky.update_from_time(&self.time_of_day);
        self.clouds.update(&self.weather, &self.time_of_day, dt);
        self.lightning.update(&self.weather, dt);
    }
}

impl Default for EnvironmentState {
    fn default() -> Self {
        Self::new()
    }
}

// ── LevelState ───────────────────────────────────────────────────────────────
// Bundles level management, streaming, world persistence, loading screen,
// and flood system — all related to level/streaming lifecycle.

pub struct LevelState {
    pub level_manager: LevelManager,
    pub streaming_config: StreamingConfig,
    pub world_state: WorldStateManager,
    pub loading_screen: LoadingScreen,
    pub flood: FloodSystem,
}

impl LevelState {
    pub fn new() -> Self {
        Self {
            level_manager: LevelManager::new(),
            streaming_config: StreamingConfig::new(),
            world_state: WorldStateManager::new(),
            loading_screen: LoadingScreen::new(),
            flood: FloodSystem::new(),
        }
    }
}

impl Default for LevelState {
    fn default() -> Self {
        Self::new()
    }
}

// ── AssetState ───────────────────────────────────────────────────────────────
// Bundles mesh storage, mesh cache, material library, prefab registry,
// and mesh streaming queue — all related to asset management.

pub struct AssetState {
    pub meshes: AssetStore<Mesh>,
    pub mesh_cache: HashMap<String, Handle<Mesh>>,
    pub materials: MaterialLibrary,
    pub prefab_registry: PrefabRegistry,
    pub mesh_streaming: MeshStreamingQueue,
}

impl AssetState {
    pub fn new() -> Self {
        Self {
            meshes: AssetStore::new(),
            mesh_cache: HashMap::new(),
            materials: MaterialLibrary::new_defaults(),
            prefab_registry: PrefabRegistry::new(),
            mesh_streaming: MeshStreamingQueue::new(false),
        }
    }

    pub fn with_streaming(enabled: bool) -> Self {
        Self {
            meshes: AssetStore::new(),
            mesh_cache: HashMap::new(),
            materials: MaterialLibrary::new_defaults(),
            prefab_registry: PrefabRegistry::new(),
            mesh_streaming: MeshStreamingQueue::new(enabled),
        }
    }
}

impl Default for AssetState {
    fn default() -> Self {
        Self::new()
    }
}

// ── CameraInputState ─────────────────────────────────────────────────────────
// Bundles camera, input state, and all camera-control parameters:
// mouse look, orbit mode, movement velocity, sensitivity, etc.
// Named CameraInputState to avoid conflict with crate::input::InputState.

pub struct CameraInputState {
    pub input: crate::input::InputState,
    pub camera: Camera3D,
    pub mouse_look_active: bool,
    pub mouse_look_latched: bool,
    pub last_cursor_pos: Option<PhysicalPosition<f64>>,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_move_velocity: Vec3,
    pub nav_speed_scalar: f32,
    pub look_sensitivity: f32,
    pub orbit_mode: bool,
    pub orbit_distance: f32,
}

impl CameraInputState {
    pub fn new() -> Self {
        let mut camera = Camera3D::new(1280.0 / 720.0);
        camera.position = Vec3::new(0.0, 4.0, 8.0);
        camera.target = Vec3::ZERO;
        let mut dir = (camera.target - camera.position).normalize_or_zero();
        if dir.length_squared() < 1e-6 {
            dir = Vec3::new(0.0, -0.2, -1.0).normalize();
        }
        let camera_yaw = dir.z.atan2(dir.x);
        let camera_pitch = dir.y.asin();

        Self {
            input: crate::input::InputState::new(),
            camera,
            mouse_look_active: false,
            mouse_look_latched: false,
            last_cursor_pos: None,
            camera_yaw,
            camera_pitch,
            camera_move_velocity: Vec3::ZERO,
            nav_speed_scalar: 6.0,
            look_sensitivity: 0.0035,
            orbit_mode: false,
            orbit_distance: 8.0,
        }
    }

    /// Recompute camera.target from yaw/pitch angles + position.
    pub fn update_camera_target_from_angles(&mut self) {
        let cp = self.camera_pitch.cos();
        let forward = Vec3::new(
            self.camera_yaw.cos() * cp,
            self.camera_pitch.sin(),
            self.camera_yaw.sin() * cp,
        )
        .normalize_or_zero();
        self.camera.target = self.camera.position + forward;
    }
}

impl Default for CameraInputState {
    fn default() -> Self {
        Self::new()
    }
}

// ── SimControl ───────────────────────────────────────────────────────────────
// Bundles simulation playback state: pause, step, preview mode, and snapshot.
// These fields are tightly coupled — they all control whether/when the sim advances.

pub struct SimControl {
    pub sim_paused: bool,
    pub sim_step_once: bool,
    pub script_skip_frames_remaining: u32,
    pub game_preview_mode: bool,
    pub prev_game_preview_mode: bool,
}

impl SimControl {
    pub fn new() -> Self {
        Self {
            sim_paused: false,
            sim_step_once: false,
            script_skip_frames_remaining: 0,
            game_preview_mode: false,
            prev_game_preview_mode: false,
        }
    }
}

impl Default for SimControl {
    fn default() -> Self {
        Self::new()
    }
}

// ── FrameClock ───────────────────────────────────────────────────────────────
// Bundles frame timing: last frame time, start time, frame index, interval, deadline.
// All fields are read/written together during the frame loop.

pub struct FrameClock {
    pub last_frame: std::time::Instant,
    pub start_time: std::time::Instant,
    pub frame_index: u64,
    pub frame_interval: std::time::Duration,
    pub next_frame_deadline: std::time::Instant,
}

impl FrameClock {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            last_frame: now,
            start_time: now,
            frame_index: 0,
            frame_interval: std::time::Duration::from_micros(16_666), // ~60fps
            next_frame_deadline: now,
        }
    }

    pub fn elapsed_secs(&self) -> f32 {
        self.start_time.elapsed().as_secs_f32()
    }

    pub fn frame_dt(&self) -> f32 {
        self.last_frame.elapsed().as_secs_f32()
    }

    pub fn advance(&mut self) {
        self.frame_index += 1;
        self.last_frame = std::time::Instant::now();
    }
}

impl Default for FrameClock {
    fn default() -> Self {
        Self::new()
    }
}

// ── ProjectState ─────────────────────────────────────────────────────────────
// Bundles project/scene lifecycle: scene list, scene switching, hub.
// These fields control which scene is loaded and what UI stage is active.
// AppStage is defined in main.rs and used here by value (u8 tag) to avoid
// a circular dependency.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProjectStage {
    BootSplash,
    ProjectHub,
    EditorLoading,
    EditorReady,
}

pub struct ProjectState {
    pub project_stage: ProjectStage,
    pub project_stage_started_at: std::time::Instant,
    pub request_return_to_hub: bool,
    pub available_scene_paths: Vec<String>,
    pub scene_list_dirty: bool,
    pub requested_scene_switch: Option<String>,
}

impl ProjectState {
    pub fn new() -> Self {
        Self {
            project_stage: ProjectStage::BootSplash,
            project_stage_started_at: std::time::Instant::now(),
            request_return_to_hub: false,
            available_scene_paths: Vec::new(),
            scene_list_dirty: true,
            requested_scene_switch: None,
        }
    }
}

impl Default for ProjectState {
    fn default() -> Self {
        Self::new()
    }
}

// ── HotReloadWatchers ────────────────────────────────────────────────────────
// Bundles hot-reload file watchers for scripts, scenes, and assets.
// These receivers + stop flags are always managed together.

pub struct HotReloadWatchers {
    pub script_watcher: Option<std::sync::mpsc::Receiver<String>>,
    pub scene_watcher: Option<std::sync::mpsc::Receiver<String>>,
    pub asset_watcher: Option<std::sync::mpsc::Receiver<String>>,
    pub script_hot_reload_enabled: bool,
    pub preferred_script_editor: String,
    pub asset_hot_reload_enabled: bool,
    pub stop_asset_watch: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub stop_scene_watch: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl HotReloadWatchers {
    pub fn new(
        script_hot_reload_enabled: bool,
        preferred_script_editor: String,
        asset_hot_reload_enabled: bool,
    ) -> Self {
        use std::sync::atomic::AtomicBool;
        Self {
            script_watcher: None,
            scene_watcher: None,
            asset_watcher: None,
            script_hot_reload_enabled,
            preferred_script_editor,
            asset_hot_reload_enabled,
            stop_asset_watch: std::sync::Arc::new(AtomicBool::new(false)),
            stop_scene_watch: std::sync::Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn stop_all(&mut self) {
        self.script_watcher = None;
        self.asset_watcher = None;
        self.scene_watcher = None;
        use std::sync::atomic::Ordering;
        self.stop_asset_watch.store(true, Ordering::SeqCst);
        self.stop_scene_watch.store(true, Ordering::SeqCst);
    }
}

// ── NavAiState ───────────────────────────────────────────────────────────────
// Bundles navigation grid, AI registry, and nav rebuild flag.
// These fields are always used together for pathfinding and AI behavior.

pub struct NavAiState {
    pub nav_grid: crate::navigation::NavGrid,
    pub ai_registry: crate::ai::AiRegistry,
    pub nav_rebuild_requested: bool,
}

impl NavAiState {
    pub fn new() -> Self {
        Self {
            nav_grid: crate::navigation::NavGrid {
                width: 64,
                depth: 64,
                walkable: vec![true; 64 * 64],
                max_slope: 0.8,
                contour_edges: Vec::new(),
                region_count: 0,
            },
            ai_registry: crate::ai::AiRegistry::new(),
            nav_rebuild_requested: true,
        }
    }
}

impl Default for NavAiState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_state_new() {
        let env = EnvironmentState::new();
        assert!(env.lightning.flash_intensity >= 0.0);
        assert!(!env.weather.condition.has_precipitation());
    }

    #[test]
    fn environment_state_update_advances_time() {
        let mut env = EnvironmentState::new();
        let initial_hour = env.time_of_day.hour;
        env.update(1.0);
        assert!(env.time_of_day.hour > initial_hour);
    }

    #[test]
    fn level_state_new() {
        let lvl = LevelState::new();
        assert_eq!(lvl.loading_screen.fade_alpha, 0.0);
        assert!(!lvl.loading_screen.visible);
        assert!(!lvl.flood.active);
        assert_eq!(lvl.flood.water_level, 0.0);
    }

    #[test]
    fn asset_state_new() {
        let assets = AssetState::new();
        assert!(!assets.mesh_streaming.enabled());
    }

    #[test]
    fn asset_state_with_streaming() {
        let assets = AssetState::with_streaming(true);
        assert!(assets.mesh_streaming.enabled());
    }

    #[test]
    fn camera_input_state_new() {
        let cis = CameraInputState::new();
        assert!(!cis.mouse_look_active);
        assert!(!cis.orbit_mode);
        assert!((cis.orbit_distance - 8.0).abs() < 0.01);
        assert!((cis.nav_speed_scalar - 6.0).abs() < 0.01);
        assert!((cis.look_sensitivity - 0.0035).abs() < 1e-6);
    }

    #[test]
    fn camera_input_state_update_target() {
        let mut cis = CameraInputState::new();
        cis.camera_yaw = 0.5;
        cis.camera_pitch = 0.3;
        cis.update_camera_target_from_angles();
        let forward = (cis.camera.target - cis.camera.position).normalize_or_zero();
        assert!(forward.length() > 0.9);
    }

    #[test]
    fn sim_control_new() {
        let sc = SimControl::new();
        assert!(!sc.sim_paused);
        assert!(!sc.sim_step_once);
        assert_eq!(sc.script_skip_frames_remaining, 0);
        assert!(!sc.game_preview_mode);
    }

    #[test]
    fn frame_clock_new() {
        let fc = FrameClock::new();
        assert_eq!(fc.frame_index, 0);
        assert!(fc.elapsed_secs() >= 0.0);
        assert!(fc.frame_dt() >= 0.0);
    }

    #[test]
    fn frame_clock_advance() {
        let mut fc = FrameClock::new();
        let prev = fc.frame_index;
        fc.advance();
        assert_eq!(fc.frame_index, prev + 1);
    }

    #[test]
    fn project_state_new() {
        let ps = ProjectState::new();
        assert!(ps.available_scene_paths.is_empty());
        assert!(ps.scene_list_dirty);
        assert!(ps.requested_scene_switch.is_none());
        assert!(!ps.request_return_to_hub);
    }

    #[test]
    fn hot_reload_watchers_stop_all() {
        use std::sync::mpsc;
        let mut hr = HotReloadWatchers::new(true, "code".into(), true);
        let (_tx, rx) = mpsc::channel::<String>();
        let (_tx2, rx2) = mpsc::channel::<String>();
        hr.script_watcher = Some(rx);
        hr.asset_watcher = Some(rx2);
        hr.stop_all();
        assert!(hr.script_watcher.is_none());
        assert!(hr.asset_watcher.is_none());
        assert!(hr.scene_watcher.is_none());
    }

    #[test]
    fn nav_ai_state_new() {
        let nai = NavAiState::new();
        assert_eq!(nai.nav_grid.width, 64);
        assert!(nai.nav_rebuild_requested);
    }
}
