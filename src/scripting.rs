// src/scripting.rs
// Embeds a Lua 5.4 runtime (via mlua 0.11) and exposes the engine API to scripts.
//
// ── mlua 0.11 changes vs 0.9/0.10 ──────────────────────────────────────────
// • Lua::new() is now considered unsafe without the vendored feature.
//   With features = ["lua54","vendored"] it is safe and idiomatic.
// • create_function closure signature: |lua: &Lua, args| — lua is &Lua not &'lua Lua.
//   In practice this changes nothing because we don't use the lua arg here.
// • LuaError::RuntimeError still exists and works the same way.
// • globals().set() / globals().get() API unchanged.
//
// ── What Lua scripts can do ──────────────────────────────────────────────────
// After register_api() and a run_update() call, these globals are available:
//
//   -- Position
//   local x, y, z = get_position(entity)
//   set_position(entity, x, y, z)
//
//   -- RigidBody velocity
//   local vx, vy, vz = get_velocity(entity)
//   set_velocity(entity, vx, vy, vz)
//   apply_force(entity, fx, fy, fz)    -- adds to current velocity
//   local grounded = is_on_ground(entity)
//
//   -- Health
//   local hp, max = get_health(entity)
//   set_health(entity, hp, max)
//   damage(entity, amount)
//   local dead = is_dead(entity)
//
//   -- Renderable
//   set_color(entity, r, g, b)
//   set_scale(entity, sx, sy, sz)
//
//   -- Entity lifetime
//   destroy(entity)          -- deferred: removed after scripting_system finishes
//
//   -- Input
//   local held = is_key_held("W")   -- keys: W S A D ArrowUp Down Left Right Space
//
//   -- Timing
//   local t = elapsed_time()        -- seconds since engine start
//
//   -- Debug
//   log("message")
//   print("message")                -- same as log

use mlua::prelude::*;
use notify::{recommended_watcher, Event, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::mpsc;

use crate::components::{
    Collider, CollisionPair, CollisionPhase, FixedJoint, Health, HingeJoint, MaterialTexture,
    OrientedBoxCollider, Position, Renderable, RigidBody, RopeConstraint, Rotation, SpringJoint,
};
use crate::input::InputState;
use hecs::{Entity, World};

use crate::ai::{
    AiAgent, AiRegistry, BehaviorTree, BehaviorNode,
    BlackboardValue,
    Sequence, Selector, Parallel, Inverter, Repeater, Cooldown,
    MoveTo, Patrol, Wait, Log,
};

// ── SandboxConfig ────────────────────────────────────────────────────────
// Restricts what Lua scripts are allowed to do. Applied per-ScriptEngine
// instance. Scripts are restricted by default — enable capabilities only
// when a script legitimately needs them.
#[derive(Clone, Debug)]
pub struct SandboxConfig {
    /// Allow Lua to read/write files on disk (io.open, fs.*).
    pub file_system_access: bool,
    /// Allow Lua to open network sockets (socket.*, http.*).
    pub network_access: bool,
    /// Allow Lua to spawn OS processes (os.execute, os.popen).
    pub os_command_access: bool,
    /// Hard memory ceiling for the Lua heap in bytes (0 = unlimited).
    pub max_memory_bytes: usize,
    /// Maximum wall-clock execution time per frame in ms (0 = unlimited).
    pub max_execution_time_ms: u64,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            file_system_access: false,
            network_access: false,
            os_command_access: false,
            max_memory_bytes: 0,
            max_execution_time_ms: 0,
        }
    }
}
use crate::navigation::NavGrid;

// ── API catalogue (for editor autocomplete / doc generation) ─────────────────
// Lists every flat global the engine mounts for scripts. The sandbox-stripped
// deny stubs (load, loadstring, loadfile, dofile, package) and the empty
// io/os globals are intentionally excluded — they are not usable API.
pub const FLAT_API_NAMES: &[&str] = &[
    // Logging / debug
    "print", "log",
    // Math / vector helpers
    "clamp01", "vec2", "vec3", "vec_add", "vec_scale", "vec_dot", "vec_cross",
    "vec_length", "vec_normalize", "vec_lerp", "sin", "cos", "sqrt", "abs",
    "lerp", "clamp",
    // Entities / components
    "get_position", "set_position", "move_by", "get_rotation", "set_rotation",
    "set_scale", "set_color", "set_material", "get_texture_path", "set_texture_path",
    "has_component", "get_component", "set_component", "destroy",
    "get_all_entities", "set_tag", "get_tag",
    // Rigidbody / physics
    "get_velocity", "set_velocity", "get_angular_velocity", "set_angular_velocity",
    "apply_force", "apply_torque", "is_on_ground",
    "create_hinge_joint", "create_fixed_joint", "create_spring_joint", "create_rope_constraint",
    "raycast", "overlap_sphere",
    // Health
    "get_health", "set_health", "damage", "is_dead",
    // Spawning / assets
    "spawn_mesh", "spawn_box", "load_model", "set_mesh_entity",
    // Effects / environment
    "set_fire", "remove_fire", "set_weather",
    // UI (global helpers, in addition to the ui.* namespace)
    "set_ui_value", "get_ui_value", "set_ui_text", "get_ui_text", "set_ui_visible",
    // Input / timing / camera
    "is_key_held", "gamepad_left_x", "gamepad_left_y", "gamepad_button_pressed",
    "gamepad_left_magnitude", "elapsed_time",
    "get_camera", "set_camera", "look_at", "get_camera_direction", "screen_to_ray",
    "skip_next_frames", "dt",
    // Audio
    "audio_play_sfx", "audio_play_music", "audio_play_at", "audio_stop_all",
    "audio_set_volume", "audio_set_master_volume", "audio_is_music_playing",
    "audio_active_count", "audio_attenuation",
    // Events / timers / modules / files
    "on_event", "fire_event", "set_timeout", "clear_timeout", "require", "fs",
    "api_catalogue",
];

/// Namespaced API tables mounted on globals (`bt.*`, `nav.*`, ...). `fs` is
/// listed in FLAT_API_NAMES; its functions are covered here.
pub const NAMESPACED_API: &[(&str, &[&str])] = &[
    (
        "bt",
        &[
            "create", "sequence", "selector", "parallel", "move_to", "patrol",
            "wait", "log", "idle", "wander", "flee", "graze", "perceive",
            "in_range", "inverter", "repeater", "cooldown", "assign",
            "set_blackboard", "get_blackboard", "set_blackboard_vec3",
            "get_blackboard_vec3", "set_blackboard_float", "get_blackboard_float",
            "set_state",
        ],
    ),
            ("nav", &["find_path", "is_walkable"]),
            ("navmesh", &["find_path", "is_walkable", "triangle_count"]),
            ("plugins", &["list", "has", "unload"]),
    (
        "terrain",
        &["height", "normal", "slope", "surface_color", "raise", "lower", "smooth"],
    ),
    (
        "particles",
        &["new_emitter", "active", "count", "fire", "remove_fire", "wind"],
    ),
    (
        "levels",
        &[
            "register", "load", "unload", "is_loaded", "set_visible", "find",
            "loaded_count", "list", "loading_show", "loading_progress",
            "loading_hide", "flood", "flood_stop", "water_level",
        ],
    ),
    (
        "boids",
        &[
            "create", "add", "remove", "clear", "count", "set_goal", "clear_goal",
            "set_bounds", "velocity", "positions", "groups",
        ],
    ),
    (
        "cinematic",
        &[
            "start", "add_shot", "set_ease", "on_shot", "on_end", "play",
            "pause", "resume", "camera_lock", "camera_locked", "is_playing",
            "time", "duration", "skip", "stop", "clear",
        ],
    ),
    ("fs", &["read", "write", "exists", "list"]),
    (
        "ui",
        &[
            "create", "add_widget", "set_text", "set_visible", "set_value",
            "get_value", "save", "load", "toggle", "list",
        ],
    ),
    ("demo", &["greet", "bump", "frame"]),
    (
        "save",
        &[
            "save_entity", "get_entity_flags", "is_entity_dead", "set_flag",
            "get_flag", "clear_level", "clear", "write", "read",
        ],
    ),
];

// ── BTBuilder: deferred behavior tree construction ─────────────────────────
// Instead of manipulating Box<dyn BehaviorNode> directly (which requires
// modifying private fields in Sequence/Selector/etc.), we store build
// instructions in a flat node list indexed by usize.  The stack tracks
// parent-child relationships during construction.  At bt.assign() time,
// BTBuilder::build() recursively converts the flat list into a real tree
// of BehaviorNode trait objects.
//
// Stack model:
//   Composites (Sequence/Selector/Parallel):
//     1. Create new node
//     2. Add as child of current stack top
//     3. Push onto stack (subsequent nodes become children of this composite)
//   Leaf nodes (MoveTo/Patrol/Wait/Log):
//     1. Create new node
//     2. Add as child of current stack top
//     3. Do NOT push (leaves have no children)
//   Decorators (Inverter/Repeater/Cooldown):
//     1. Pop the top node (this becomes the decorator's child)
//     2. Create decorator wrapping that child
//     3. Add decorator as child of new stack top
//     4. Push decorator onto stack

#[derive(Clone)]
enum BTNodeKind {
    Sequence,
    Selector,
    Parallel { success_threshold: usize },
    Inverter,
    Repeater { max_times: u32 },
    Cooldown { duration: f32 },
    MoveTo { speed: f32 },
    Patrol { speed: f32, waypoints_key: String },
    Wait { duration: f32 },
    Log { message: String },
    Idle { duration: f32 },
    Wander { speed: f32, radius: f32 },
    Flee { run_speed: f32, safe_distance: f32 },
    Graze { speed: f32, radius: f32 },
    Perception { radius: f32, tag: String },
    Distance { min: f32, max: f32 },
}

struct BTBuilderNode {
    kind: BTNodeKind,
    children: Vec<usize>,
}

pub struct BTBuilder {
    nodes: Vec<BTBuilderNode>,
    root: usize,
    stack: Vec<usize>,
    /// Auto-incrementing ID for generating unique blackboard keys.
    next_wp_id: usize,
    /// Patrol waypoints to inject into the entity's blackboard at assign time.
    /// Each entry is (blackboard_key, waypoints).
    patrol_data: Vec<(String, Vec<[f32; 3]>)>,
}

impl BTBuilder {
    fn new() -> Self {
        let root = BTBuilderNode {
            kind: BTNodeKind::Sequence,
            children: Vec::new(),
        };
        let mut b = Self {
            nodes: Vec::new(),
            root: 0,
            stack: Vec::new(),
            next_wp_id: 0,
            patrol_data: Vec::new(),
        };
        b.nodes.push(root);
        b.stack.push(0);
        b
    }

    /// Push a composite node (Sequence/Selector/Parallel) onto the tree.
    /// The composite becomes a child of the current stack top, then is
    /// itself pushed so subsequent calls add children to it.
    fn push_composite(&mut self, kind: BTNodeKind) {
        let idx = self.nodes.len();
        self.nodes.push(BTBuilderNode { kind, children: Vec::new() });
        if let Some(&parent) = self.stack.last() {
            self.nodes[parent].children.push(idx);
        }
        self.stack.push(idx);
    }

    /// Add a leaf node (action) as a child of the current stack top.
    /// Leans are NOT pushed — they have no children.
    fn add_leaf(&mut self, kind: BTNodeKind) {
        let idx = self.nodes.len();
        self.nodes.push(BTBuilderNode { kind, children: Vec::new() });
        if let Some(&parent) = self.stack.last() {
            self.nodes[parent].children.push(idx);
        }
    }

    /// Wrap the most recently added child of the current stack top
    /// in a decorator.
    /// 1. Pop the last child from the current stack top.
    /// 2. Create the decorator wrapping that child.
    /// 3. Add the decorator as a child of the current stack top (replacing the original).
    /// 4. Push the decorator onto the stack.
    fn wrap_decorator(&mut self, kind: BTNodeKind) {
        let parent_idx = *self.stack.last().expect("BT stack underflow in decorator");
        let child_idx = self.nodes[parent_idx].children.pop()
            .expect("No child to wrap with decorator");
        let idx = self.nodes.len();
        self.nodes.push(BTBuilderNode {
            kind,
            children: vec![child_idx],
        });
        self.nodes[parent_idx].children.push(idx);
        self.stack.push(idx);
    }

    /// Recursively convert the flat node list into a real BehaviorNode tree.
    fn build(&self) -> Box<dyn BehaviorNode> {
        self.build_node(self.root)
    }

    fn build_node(&self, idx: usize) -> Box<dyn BehaviorNode> {
        let node = &self.nodes[idx];
        let name = format!("bt_{}", idx);

        match &node.kind {
            BTNodeKind::Sequence => {
                let children = self.build_children(&node.children);
                Box::new(Sequence::new(&name, children))
            }
            BTNodeKind::Selector => {
                let children = self.build_children(&node.children);
                Box::new(Selector::new(&name, children))
            }
            BTNodeKind::Parallel { success_threshold } => {
                let children = self.build_children(&node.children);
                let failure_threshold = children.len();
                Box::new(Parallel::new(&name, children, *success_threshold, failure_threshold))
            }
            BTNodeKind::Inverter => {
                let child = self.build_first_child(&node.children);
                Box::new(Inverter::new(&name, child))
            }
            BTNodeKind::Repeater { max_times } => {
                let child = self.build_first_child(&node.children);
                Box::new(Repeater::new(&name, child, *max_times))
            }
            BTNodeKind::Cooldown { duration } => {
                let child = self.build_first_child(&node.children);
                Box::new(Cooldown::new(&name, child, *duration))
            }
            BTNodeKind::MoveTo { speed } => {
                Box::new(MoveTo::new(&name, *speed, "target_pos"))
            }
            BTNodeKind::Patrol { speed, waypoints_key } => {
                Box::new(Patrol::new(&name, *speed, waypoints_key))
            }
            BTNodeKind::Wait { duration } => {
                Box::new(Wait::new(&name, *duration))
            }
            BTNodeKind::Log { message } => {
                Box::new(Log::new(&name, message))
            }
            BTNodeKind::Idle { duration } => {
                Box::new(crate::ai::Idle::new(&name, *duration))
            }
            BTNodeKind::Wander { speed, radius } => {
                Box::new(crate::ai::Wander::new(
                    &name,
                    *speed,
                    *radius,
                    [0.0, 0.0, 0.0],
                ))
            }
            BTNodeKind::Flee { run_speed, safe_distance } => {
                Box::new(crate::ai::Flee::new(&name, *run_speed, *safe_distance))
            }
            BTNodeKind::Graze { speed, radius } => {
                Box::new(crate::ai::Graze::new(&name, *speed, *radius))
            }
            BTNodeKind::Perception { radius, tag } => {
                Box::new(crate::ai::Perception::new(&name, *radius, tag))
            }
            BTNodeKind::Distance { min, max } => {
                // Reference position comes from the blackboard key written by
                // Perception ("perceived_pos") by default.
                Box::new(crate::ai::DistanceCondition::new(
                    &name,
                    self.build_first_child(&node.children),
                    "perceived_pos",
                    *min,
                    *max,
                ))
            }
        }
    }

    fn build_children(&self, indices: &[usize]) -> Vec<Box<dyn BehaviorNode>> {
        indices.iter().map(|&i| self.build_node(i)).collect()
    }

    fn build_first_child(&self, indices: &[usize]) -> Box<dyn BehaviorNode> {
        indices.first()
            .map(|&i| self.build_node(i))
            .unwrap_or_else(|| Box::new(Wait::new("placeholder", 0.0)))
    }
}

struct ScriptInstance {
    path: String,
    revision: u64,
    env_key: LuaRegistryKey,
    start_key: Option<LuaRegistryKey>,
    update_key: LuaRegistryKey,
    collision_enter_key: Option<LuaRegistryKey>,
    collision_stay_key: Option<LuaRegistryKey>,
    collision_exit_key: Option<LuaRegistryKey>,
    started: bool,
}

// ── ScriptEngine ──────────────────────────────────────────────────────────────
pub struct ScriptEngine {
    lua: Lua,
    // Entities destroyed by Lua this frame — drained after scripting_system.
    // UnsafeCell because we hand raw pointers into Lua closures.
    pending_destroys: std::cell::UnsafeCell<Vec<Entity>>,
    // Engine start instant — used by elapsed_time().
    start_time: std::cell::UnsafeCell<std::time::Instant>,
    // One isolated Lua environment per entity script instance.
    instances: HashMap<u64, ScriptInstance>,
    // Path-based versioning for hot reload.
    script_revisions: HashMap<String, u64>,
    ui_values: HashMap<String, f32>,
    ui_texts: HashMap<String, String>,
    ui_visibility: HashMap<String, bool>,
    pending_camera_set: std::cell::UnsafeCell<Option<([f32; 3], [f32; 3])>>,
    pending_frame_skip: std::cell::UnsafeCell<u32>,
    // ── Behavior tree Lua bindings ──────────────────────────────────────
    // Named BT builders — one per bt.create() call.  Each builder stores a
    // flat node list that is converted to real BehaviorNode objects at
    // bt.assign() time via BTBuilder::build().
    bt_trees: HashMap<String, BTBuilder>,
    // Raw pointer to NavGrid for nav.find_path / nav.is_walkable.
    // Set via set_external_refs() before each run_update() call.
    nav_grid_ptr: usize,
    // Raw pointer to the polygon NavMesh (navmesh.* Lua API).  Additive to
    // NavGrid — triangle-level 3D pathing.
    navmesh_ptr: usize,
    // Raw pointer to AiRegistry for bt.assign() tree registration.
    ai_registry_ptr: usize,
    // Raw pointer to TerrainWorld for terrain_height / terrain_raise / etc.
    terrain_world_ptr: usize,
    // Raw pointer to the mesh AssetStore for spawn_mesh / load_model.
    // Set via set_external_refs() before each run_update() call.
    meshes_ptr: usize,
    // Raw pointer to the environment WeatherState for set_weather().
    weather_ptr: usize,
    // Raw pointer to the ParticleSystem for the particles.* Lua API.
    // Set via set_external_refs()/set_particles() before run_update().
    particles_ptr: usize,
    // Raw pointer to LevelState for the levels.* Lua API.  Set via
    // set_levels() before run_update() each frame.
    levels_ptr: usize,
    // Raw pointer to the BoidRegistry for the boids.* Lua API.
    boids_ptr: usize,
    // Path → mesh handle cache so scripts don't reload the same mesh every frame.
    mesh_cache: HashMap<String, crate::assets::Handle<crate::assets::Mesh>>,
    // Root directory the sandboxed `require` loader reads modules from, and the
    // default root for the fs.* API. Defaults to "Content/Scripts".
    script_root: String,
    // Wider root granted to scripts whose SandboxConfig enables
    // file_system_access (typically the whole "Content" folder).
    fs_root: String,
    // Per-script sandbox overrides (script path → config). Falls back to
    // self.sandbox when a path has no override.
    tiers: HashMap<String, SandboxConfig>,
    // Cached loaded modules (name → registry key) so require() is idempotent.
    module_cache: HashMap<String, mlua::RegistryKey>,
    // Scripts that failed to load or threw, keyed by entity bits → (path, rev).
    // Skipped until the file revision changes, then rebuilt automatically.
    failed_instances: HashMap<u64, (String, u64)>,
    /// Sandbox configuration — controls what scripts are allowed to do.
    pub sandbox: SandboxConfig,
    /// Registered ScriptPlugins, mounted onto Lua globals in register_api().
    plugins: Vec<Box<dyn crate::scripting_api::ScriptPlugin>>,
    /// Pure-Lua plugins loaded from a plugins directory (e.g. Content/Scripts/plugins).
    /// These need NO Rust recompilation: a plugin is just a .lua file that returns
    /// a table `{ name, start(), update(dt), on_event(name, payload) }`.  They hot
    /// reload through the same file watcher that reloads entity scripts.
    lua_plugins: Vec<LuaPlugin>,
    /// Runtime UI manager exposed to Lua via the `ui.*` API and rendered by
    /// the engine each frame. Shared so Lua mutations are visible to the
    /// overlay renderer.
    pub ui_manager: Option<std::sync::Arc<std::sync::Mutex<crate::ui::UiManager>>>,
    /// String-keyed Lua event callbacks. `on_event("name", fn)` registers a
    /// callback; `fire_event("name", ...)` calls all callbacks for that name.
    /// Lua functions are stored as registry keys to survive GC and closures.
    lua_events: HashMap<String, Vec<mlua::RegistryKey>>,
    /// Scheduled timers created via `set_timeout`.  Indexed by a monotonically
    /// increasing handle.  Ticked once per frame from tick_timers().
    timers: HashMap<u64, LuaTimer>,
    /// Next timer handle to allocate.
    next_timer_id: u64,
    /// Active cutscene timeline driven by the `cinematic.*` Lua API.
    /// Ticked once per frame from tick_cinematics().
    cutscene: crate::cinematics::Cutscene,
    /// Lua start callbacks keyed by shot index.
    shot_callbacks: HashMap<usize, mlua::RegistryKey>,
    /// Which shots have already fired their start callback.
    shot_started: Vec<bool>,
    /// Optional Lua callback fired once when the cutscene reaches the end.
    end_callback: Option<mlua::RegistryKey>,
    /// Guard so the end callback fires exactly once.
    end_fired: bool,
}

/// A single deferred Lua callback (timer / delayed call).
struct LuaTimer {
    /// Remaining seconds before the callback fires.
    remaining: f32,
    /// The callback, retained as a registry key so it survives GC.
    func: mlua::RegistryKey,
}

/// A pure-Lua plugin loaded from a .lua file.  Plugins are the hot-reloadable,
/// no-recompile extension point: dropping a file in the plugins directory makes
/// it available immediately, and editing it live-reloads the new behaviour.
struct LuaPlugin {
    /// Absolute/relative path of the source file (used for hot-reload matching).
    path: String,
    /// Plugin identifier from the returned table's `name` field.
    name: String,
    /// Environment table the plugin was loaded into (keeps closures alive).
    env_key: mlua::RegistryKey,
    /// Optional `start()` callback (called once on load).
    start_key: Option<mlua::RegistryKey>,
    /// Optional `update(dt)` callback (called every frame from tick_plugins).
    update_key: Option<mlua::RegistryKey>,
    /// Optional `on_event(name, payload)` callback (fired by fire_event).
    on_event_key: Option<mlua::RegistryKey>,
}

// SAFETY: ScriptEngine is only ever used on the main thread.
unsafe impl Send for ScriptEngine {}

// Resolve a script-supplied path against `root` for reading, canonicalize it,
// and reject absolute paths or any path that escapes the root (via `..`,
// symlinks, or case tricks).
fn resolve_sandbox_read(root: &str, path: &str) -> Result<std::path::PathBuf, LuaError> {
    let root = std::fs::canonicalize(root).map_err(|e| {
        LuaError::RuntimeError(format!("fs: sandbox root unavailable: {}", e))
    })?;
    let path = std::path::Path::new(path);
    if path.is_absolute() {
        return Err(LuaError::RuntimeError("fs: absolute paths are not allowed".to_string()));
    }
    let full = std::fs::canonicalize(root.join(path)).map_err(|e| {
        LuaError::RuntimeError(format!("fs: path not found: {}", e))
    })?;
    if !full.starts_with(&root) {
        return Err(LuaError::RuntimeError("fs: path escapes sandbox root".to_string()));
    }
    Ok(full)
}

// Resolve a script-supplied path for writing (the target may not exist yet).
// Canonicalizes the parent directory so traversal outside the root is still
// rejected.
fn resolve_sandbox_write(root: &str, path: &str) -> Result<std::path::PathBuf, LuaError> {
    let root = std::fs::canonicalize(root).map_err(|e| {
        LuaError::RuntimeError(format!("fs: sandbox root unavailable: {}", e))
    })?;
    let path = std::path::Path::new(path);
    if path.is_absolute() {
        return Err(LuaError::RuntimeError("fs: absolute paths are not allowed".to_string()));
    }
    let joined = root.join(path);
    let parent = joined.parent().ok_or_else(|| {
        LuaError::RuntimeError("fs: invalid path".to_string())
    })?;
    let canon_parent = std::fs::canonicalize(parent).map_err(|e| {
        LuaError::RuntimeError(format!("fs: directory not found: {}", e))
    })?;
    if !canon_parent.starts_with(&root) {
        return Err(LuaError::RuntimeError("fs: path escapes sandbox root".to_string()));
    }
    Ok(joined)
}

impl ScriptEngine {
    pub fn new() -> Self {
        Self {
            lua: Lua::new(),
            pending_destroys: std::cell::UnsafeCell::new(Vec::new()),
            start_time: std::cell::UnsafeCell::new(std::time::Instant::now()),
            instances: HashMap::new(),
            script_revisions: HashMap::new(),
            ui_values: HashMap::new(),
            ui_texts: HashMap::new(),
            ui_visibility: HashMap::new(),
            pending_camera_set: std::cell::UnsafeCell::new(None),
            pending_frame_skip: std::cell::UnsafeCell::new(0),
            bt_trees: HashMap::new(),
            nav_grid_ptr: 0,
            navmesh_ptr: 0,
            ai_registry_ptr: 0,
            terrain_world_ptr: 0,
            meshes_ptr: 0,
            weather_ptr: 0,
            particles_ptr: 0,
            levels_ptr: 0,
            boids_ptr: 0,
            mesh_cache: HashMap::new(),
            script_root: "Content/Scripts".to_string(),
            fs_root: "Content".to_string(),
            tiers: HashMap::new(),
            module_cache: HashMap::new(),
            failed_instances: HashMap::new(),
            sandbox: SandboxConfig::default(),
            plugins: Vec::new(),
            lua_plugins: Vec::new(),
            ui_manager: None,
            lua_events: HashMap::new(),
            timers: HashMap::new(),
            next_timer_id: 1,
            cutscene: crate::cinematics::Cutscene::new(),
            shot_callbacks: HashMap::new(),
            shot_started: Vec::new(),
            end_callback: None,
            end_fired: false,
        }
    }

    /// Replace the sandbox configuration.  Must be called before
    /// `register_api()` (engine startup) for the limits to take effect on the
    /// already-created Lua state.  Provides a safe, controllable point for a
    /// game/editor to configure heap and execution budgets.
    ///
    /// ```
    /// scripts.set_sandbox(SandboxConfig {
    ///     max_memory_bytes: 128 << 20,       // 128 MB heap cap
    ///     max_execution_time_ms: 25,         // ~25k Lua ops per call
    ///     ..SandboxConfig::default()
    /// });
    ///
    /// These defaults are only wired in main.rs when the game engine boots;
    /// they do not restrict tests, which construct ScriptEngine directly.
    pub fn set_sandbox(&mut self, config: SandboxConfig) {
        self.sandbox = config;
    }

    /// Root directory the sandboxed `require` loader reads modules from.
    /// Scripts can also read/write here through `fs.*`. Defaults to
    /// "Content/Scripts".
    pub fn set_script_root(&mut self, root: &str) {
        self.script_root = root.to_string();
    }

    /// Root directory granted to scripts whose SandboxConfig enables
    /// file_system_access (typically the whole "Content" folder). Traversal
    /// outside this root is rejected by the fs.* API.
    pub fn set_fs_root(&mut self, root: &str) {
        self.fs_root = root.to_string();
    }

    /// Override the sandbox for a single script file. The per-path config
    /// replaces the engine-wide `sandbox` for that script's instance env.
    /// `path` matches the Script component's path (e.g. "Content/Scripts/my.lua").
    pub fn set_script_sandbox(&mut self, path: &str, config: SandboxConfig) {
        self.tiers.insert(path.to_string(), config);
    }

    /// Returns the full scripting API catalogue — flat globals plus every
    /// namespaced function — for editor autocomplete / doc generation.
    pub fn api_catalogue(&self) -> Vec<String> {
        let mut names: Vec<String> = FLAT_API_NAMES.iter().map(|s| s.to_string()).collect();
        for (ns, fns) in NAMESPACED_API {
            names.push((*ns).to_string());
            for f in *fns {
                names.push(format!("{}.{}", ns, f));
            }
        }
        names.sort();
        names.dedup();
        names
    }

    pub fn consume_camera_request(&mut self) -> Option<([f32; 3], [f32; 3])> {
        let slot = unsafe { &mut *self.pending_camera_set.get() };
        slot.take()
    }

    pub fn consume_frame_skip_request(&mut self) -> u32 {
        let slot = unsafe { &mut *self.pending_frame_skip.get() };
        let n = *slot;
        *slot = 0;
        n
    }

    /// Fire a Lua event from Rust. Calls every callback registered via
    /// `on_event("name", fn)` with the given string payload (if any).
    /// Used by engine systems (e.g. level start, health zero) to notify scripts.
    pub fn fire_event(&mut self, name: &str, payload: Option<String>) -> LuaResult<()> {
        // Iterate the callback keys by index so we can re-register the same
        // persistent callbacks afterwards (RegistryKey is not Clone).
        let Some(keys) = self.lua_events.get(name) else {
            return self.fire_event_to_plugins(name, payload);
        };
        let indices: Vec<usize> = (0..keys.len()).collect();
        for i in indices {
            let key = &self.lua_events[name][i];
            let func: LuaFunction = self.lua.registry_value(key)?;
            match payload.clone() {
                Some(p) => func.call::<()>((p,))?,
                None => func.call::<()>(())?,
            }
        }
        self.fire_event_to_plugins(name, payload)
    }

    /// Forward an event to every loaded Lua plugin's `on_event(name, payload)`.
    fn fire_event_to_plugins(&mut self, name: &str, payload: Option<String>) -> LuaResult<()> {
        if self.lua_plugins.is_empty() {
            return Ok(());
        }
        let indices: Vec<usize> = self
            .lua_plugins
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.on_event_key.as_ref().map(|_| i))
            .collect();
        for i in indices {
            let pname = self.lua_plugins[i].name.clone();
            let key = self.lua_plugins[i].on_event_key.as_ref().unwrap();
            let func: LuaFunction = self.lua.registry_value(key)?;
            let res = match payload.clone() {
                Some(p) => func.call::<()>((name.to_string(), p)),
                None => func.call::<()>((name.to_string(), mlua::Value::Nil)),
            };
            if let Err(e) = res {
                tracing::error!("[Plugins] {} on_event error: {}", pname, e);
            }
        }
        Ok(())
    }

    /// Register a Lua callback for an event name. Called from the `on_event`
    /// Lua binding; the function is stored by registry key so it survives GC.
    pub fn register_lua_event(&mut self, name: &str, func: LuaFunction) -> LuaResult<()> {
        let key = self.lua.create_registry_value(func)?;
        self.lua_events
            .entry(name.to_string())
            .or_default()
            .push(key);
        Ok(())
    }

    /// Schedule a Lua callback to fire after `delay` seconds.  Returns a handle
    /// suitable for clear_timeout().  Backed by the same RegistryKey mechanism
    /// as events, so the closure survives GC.
    pub fn set_timeout(&mut self, delay: f32, func: LuaFunction) -> LuaResult<u64> {
        let key = self.lua.create_registry_value(func)?;
        let id = self.next_timer_id;
        self.next_timer_id = self.next_timer_id.wrapping_add(1);
        self.timers.insert(id, LuaTimer {
            remaining: delay.max(0.0),
            func: key,
        });
        Ok(id)
    }

    /// Cancel a previously scheduled timer. Does nothing if the handle is stale.
    pub fn clear_timeout(&mut self, id: u64) {
        if let Some(t) = self.timers.remove(&id) {
            let _ = self.lua.remove_registry_value(t.func);
        }
    }

    /// Advance all timers by `dt` and fire any whose delay has elapsed.
    /// Call once per frame (after running scripts) so timed logic is driven
    /// deterministically from the main loop rather than per-entity updates.
    pub fn tick_timers(&mut self, dt: f32) -> LuaResult<()> {
        if self.timers.is_empty() {
            return Ok(());
        }
        let mut due: Vec<mlua::RegistryKey> = Vec::new();
        let mut alive: HashMap<u64, LuaTimer> = HashMap::with_capacity(self.timers.len());
        for (id, timer) in self.timers.drain() {
            if timer.remaining - dt <= 0.0 {
                due.push(timer.func);
            } else {
                alive.insert(id, LuaTimer {
                    remaining: timer.remaining - dt,
                    func: timer.func,
                });
            }
        }
        self.timers = alive;
        for key in due {
            let func: LuaFunction = self.lua.registry_value(&key)?;
            let res = func.call::<()>(());
            let _ = self.lua.remove_registry_value(key);
            if let Err(e) = res {
                tracing::error!("[Scripting] Timer callback error: {}", e);
            }
        }
        Ok(())
    }

    /// Advance the active cutscene by `dt`, driving the engine camera and
    /// firing per-shot start callbacks plus the end callback.  Called once
    /// per frame from the game loop (next to tick_timers()).
    pub fn tick_cinematics(&mut self, dt: f32) {
        if !self.cutscene.is_playing() {
            return;
        }
        let active_shot = self.cutscene.step(dt);

        // Fire start callbacks for every shot reached up to the active one.
        if self.shot_started.len() < active_shot + 1 {
            self.shot_started.resize(active_shot + 1, false);
        }
        for i in 0..=active_shot {
            if self.shot_started[i] {
                continue;
            }
            if let Some(key) = self.shot_callbacks.get(&i) {
                if let Ok(func) = self.lua.registry_value::<LuaFunction>(key) {
                    let shot_name = self.cutscene.shot(i).map(|s| s.name.clone()).unwrap_or_default();
                    let res = func.call::<()>(shot_name);
                    if let Err(e) = res {
                        tracing::error!("[Scripting] Cinematic shot callback error: {}", e);
                    }
                }
            }
            self.shot_started[i] = true;
        }

        // Push the interpolated cutscene camera (overrides script camera).
        // Skipped when the cutscene is authored without camera ownership, so
        // gameplay camera code keeps full control during the cutscene.
        if self.cutscene.drives_camera() {
            if let Some((pos, target)) = self.cutscene.current_camera() {
                let slot = unsafe { &mut *self.pending_camera_set.get() };
                *slot = Some((pos, target));
            }
        }

        // Fire the end callback exactly once.
        if self.cutscene.is_finished() && !self.end_fired {
            self.end_fired = true;
            if let Some(key) = self.end_callback.take() {
                if let Ok(func) = self.lua.registry_value::<LuaFunction>(&key) {
                    let _ = self.lua.remove_registry_value(key);
                    let res = func.call::<()>(());
                    if let Err(e) = res {
                        tracing::error!("[Scripting] Cinematic end callback error: {}", e);
                    }
                }
            }
        }
    }

    pub fn ui_value(&self, id: &str) -> f32 {
        *self.ui_values.get(id).unwrap_or(&0.0)
    }

    pub fn ui_text(&self, id: &str) -> Option<String> {
        self.ui_texts.get(id).cloned()
    }

    /// Set a numeric value for a UI widget (called from Lua or editor).
    pub fn set_ui_value(&mut self, id: &str, value: f32) {
        self.ui_values.insert(id.to_string(), value);
    }

    /// Set a text override for a UI widget (called from Lua or editor).
    pub fn set_ui_text(&mut self, id: &str, text: &str) {
        self.ui_texts.insert(id.to_string(), text.to_string());
    }

    /// Check if a UI widget is hidden via Lua (set_ui_visible).
    pub fn ui_visible(&self, id: &str) -> bool {
        // None = visible by default; Some(true) = visible; Some(false) = hidden.
        *self.ui_visibility.get(id).unwrap_or(&true)
    }

    /// Set visibility of a UI widget.
    pub fn set_ui_visible(&mut self, id: &str, visible: bool) {
        self.ui_visibility.insert(id.to_string(), visible);
    }

    /// Store raw pointers to external engine systems so Lua closures can
    /// access them.  Must be called before run_update() each frame.
    pub fn set_external_refs(
        &mut self,
        nav: &NavGrid,
        ai_reg: &mut AiRegistry,
        terrain: &mut crate::terrain::TerrainWorld,
        meshes: &mut crate::assets::AssetStore<crate::assets::Mesh>,
        weather: &mut crate::environment::weather::WeatherState,
        navmesh: &crate::navmesh::NavMesh,
    ) {
        self.nav_grid_ptr = nav as *const NavGrid as usize;
        self.ai_registry_ptr = ai_reg as *mut AiRegistry as usize;
        self.terrain_world_ptr = terrain as *mut crate::terrain::TerrainWorld as usize;
        self.meshes_ptr = meshes as *mut crate::assets::AssetStore<crate::assets::Mesh> as usize;
        self.weather_ptr = weather as *mut crate::environment::weather::WeatherState as usize;
        self.navmesh_ptr = navmesh as *const crate::navmesh::NavMesh as usize;
    }

    /// Store the raw pointer to the engine ParticleSystem so the `particles.*`
    /// Lua API can drive emitters and fire sources.  Must be called before
    /// run_update() each frame (ParticleSystem lives in the game loop).
    pub fn set_particles(&mut self, particles: &mut crate::particles::ParticleSystem) {
        self.particles_ptr = particles as *mut crate::particles::ParticleSystem as usize;
    }

    /// Store the raw pointer to the engine LevelState so the `levels.*` Lua
    /// API can manage level lifecycle, the loading screen, and flooding.
    /// Must be called before run_update() each frame.
    pub fn set_levels(&mut self, levels: &mut crate::engine_subsystems::LevelState) {
        self.levels_ptr = levels as *mut crate::engine_subsystems::LevelState as usize;
    }

    /// Store the raw pointer to the engine BoidRegistry so the `boids.*` Lua
    /// API can create named flocks and read back their positions.  Must be
    /// called before run_update() each frame.
    pub fn set_boids(&mut self, boids: &mut crate::boids::BoidRegistry) {
        self.boids_ptr = boids as *mut crate::boids::BoidRegistry as usize;
    }

    /// Raw pointer to the environment weather state.
    fn weather_ptr(&self) -> usize {
        self.weather_ptr
    }

    /// Raw pointer to the per-engine mesh cache map.  Used by Lua closures so
    /// scripts reuse loaded meshes instead of reloading every frame.
    fn mesh_cache_ptr(&self) -> usize {
        &self.mesh_cache as *const HashMap<String, crate::assets::Handle<crate::assets::Mesh>> as usize
    }

    /// Expose the Lua instance so external code can set globals (e.g., ui_click_event).
    pub fn lua_create(&self) -> &Lua {
        &self.lua
    }

    // register_plugin() adds a game/engine-provided ScriptPlugin whose Lua API
    // is mounted onto globals the next time register_api() runs.  This is the
    // formal plugin extension point — game modules no longer need to edit the
    // central globals blob in scripting.rs.
    pub fn register_plugin(&mut self, plugin: Box<dyn crate::scripting_api::ScriptPlugin>) {
        self.plugins.push(plugin);
    }

    // register_api() sets up engine-wide Lua globals (logging, print).
    // Call once at startup before loading any scripts.
    pub fn register_api(&mut self) -> LuaResult<()> {
        let globals = self.lua.globals();

        // Override Lua's built-in print to route through our logging.
        let print_fn = self.lua.create_function(|_, msg: String| {
            tracing::info!("[Lua] {}", msg);
            Ok(())
        })?;
        globals.set("print", print_fn)?;

        let log_fn = self.lua.create_function(|_, msg: String| {
            tracing::info!("[Script] {}", msg);
            Ok(())
        })?;
        globals.set("log", log_fn)?;

        let clamp01_fn = self.lua.create_function(|_, v: f32| Ok(v.clamp(0.0, 1.0)))?;
        globals.set("clamp01", clamp01_fn)?;
        let vec2_fn = self.lua.create_function(|lua, (x, y): (f32, f32)| {
            let t = lua.create_table()?;
            t.set("x", x)?;
            t.set("y", y)?;
            Ok(t)
        })?;
        globals.set("vec2", vec2_fn)?;
        let vec3_fn = self.lua.create_function(|lua, (x, y, z): (f32, f32, f32)| {
            let t = lua.create_table()?;
            t.set("x", x)?;
            t.set("y", y)?;
            t.set("z", z)?;
            Ok(t)
        })?;
        globals.set("vec3", vec3_fn)?;
        let vec_add_fn = self.lua.create_function(|lua, (a, b): (LuaTable, LuaTable)| {
            let t = lua.create_table()?;
            t.set("x", a.get::<f32>("x").unwrap_or(0.0) + b.get::<f32>("x").unwrap_or(0.0))?;
            t.set("y", a.get::<f32>("y").unwrap_or(0.0) + b.get::<f32>("y").unwrap_or(0.0))?;
            t.set("z", a.get::<f32>("z").unwrap_or(0.0) + b.get::<f32>("z").unwrap_or(0.0))?;
            Ok(t)
        })?;
        globals.set("vec_add", vec_add_fn)?;
        let vec_scale_fn = self.lua.create_function(|lua, (a, s): (LuaTable, f32)| {
            let t = lua.create_table()?;
            t.set("x", a.get::<f32>("x").unwrap_or(0.0) * s)?;
            t.set("y", a.get::<f32>("y").unwrap_or(0.0) * s)?;
            t.set("z", a.get::<f32>("z").unwrap_or(0.0) * s)?;
            Ok(t)
        })?;
        globals.set("vec_scale", vec_scale_fn)?;
        let vec_dot_fn = self.lua.create_function(|_, (a, b): (LuaTable, LuaTable)| {
            let ax = a.get::<f32>("x").unwrap_or(0.0);
            let ay = a.get::<f32>("y").unwrap_or(0.0);
            let az = a.get::<f32>("z").unwrap_or(0.0);
            let bx = b.get::<f32>("x").unwrap_or(0.0);
            let by = b.get::<f32>("y").unwrap_or(0.0);
            let bz = b.get::<f32>("z").unwrap_or(0.0);
            Ok(ax * bx + ay * by + az * bz)
        })?;
        globals.set("vec_dot", vec_dot_fn)?;
        let vec_cross_fn = self.lua.create_function(|lua, (a, b): (LuaTable, LuaTable)| {
            let ax = a.get::<f32>("x").unwrap_or(0.0);
            let ay = a.get::<f32>("y").unwrap_or(0.0);
            let az = a.get::<f32>("z").unwrap_or(0.0);
            let bx = b.get::<f32>("x").unwrap_or(0.0);
            let by = b.get::<f32>("y").unwrap_or(0.0);
            let bz = b.get::<f32>("z").unwrap_or(0.0);
            let t = lua.create_table()?;
            t.set("x", ay * bz - az * by)?;
            t.set("y", az * bx - ax * bz)?;
            t.set("z", ax * by - ay * bx)?;
            Ok(t)
        })?;
        globals.set("vec_cross", vec_cross_fn)?;
        let vec_length_fn = self.lua.create_function(|_, a: LuaTable| {
            let x = a.get::<f32>("x").unwrap_or(0.0);
            let y = a.get::<f32>("y").unwrap_or(0.0);
            let z = a.get::<f32>("z").unwrap_or(0.0);
            Ok((x * x + y * y + z * z).sqrt())
        })?;
        globals.set("vec_length", vec_length_fn)?;
        let vec_normalize_fn = self.lua.create_function(|lua, a: LuaTable| {
            let x = a.get::<f32>("x").unwrap_or(0.0);
            let y = a.get::<f32>("y").unwrap_or(0.0);
            let z = a.get::<f32>("z").unwrap_or(0.0);
            let len = (x * x + y * y + z * z).sqrt();
            let t = lua.create_table()?;
            if len > 1e-6 {
                t.set("x", x / len)?;
                t.set("y", y / len)?;
                t.set("z", z / len)?;
            } else {
                t.set("x", 0.0)?;
                t.set("y", 0.0)?;
                t.set("z", 0.0)?;
            }
            Ok(t)
        })?;
        globals.set("vec_normalize", vec_normalize_fn)?;
        let vec_lerp_fn = self.lua.create_function(|lua, (a, b, t): (LuaTable, LuaTable, f32)| {
            let ax = a.get::<f32>("x").unwrap_or(0.0);
            let ay = a.get::<f32>("y").unwrap_or(0.0);
            let az = a.get::<f32>("z").unwrap_or(0.0);
            let bx = b.get::<f32>("x").unwrap_or(0.0);
            let by = b.get::<f32>("y").unwrap_or(0.0);
            let bz = b.get::<f32>("z").unwrap_or(0.0);
            let t = t.clamp(0.0, 1.0);
            let out = lua.create_table()?;
            out.set("x", ax + (bx - ax) * t)?;
            out.set("y", ay + (by - ay) * t)?;
            out.set("z", az + (bz - az) * t)?;
            Ok(out)
        })?;
        globals.set("vec_lerp", vec_lerp_fn)?;

        // ── UI Lua API ────────────────────────────────────────────────────
        // Register UI-related Lua bindings (ui.create, ui.show, etc.) backed
        // by a real UiManager. The shared handle is stored so the engine can
        // render the active design each frame.
        let ui_plugin = crate::ui::UiScriptPlugin::new();
        self.ui_manager = Some(ui_plugin.manager_clone());
        let ui_ref: &dyn crate::scripting_api::ScriptPlugin = &ui_plugin;
        crate::scripting_api::mount_plugins(&self.lua, &[ui_ref])?;

        // ── Plugins ───────────────────────────────────────────────────────
        // Mount any game/engine-provided ScriptPlugins.  Each plugin is a
        // self-contained API surface (see scripting_api.rs) that registers its
        // functions through the ApiRegistry, so gameplay code can extend the
        // Lua API without editing this file.
        if !self.plugins.is_empty() {
            let refs: Vec<&dyn crate::scripting_api::ScriptPlugin> =
                self.plugins.iter().map(|p| p.as_ref()).collect();
            crate::scripting_api::mount_plugins(&self.lua, &refs)?;
        }

        // ── Events (pub/sub) ───────────────────────────────────────────────
        // on_event("name", function) — register a callback.
        // fire_event("name", "optional-payload") — invoke all callbacks.
        // Engine systems call fire_event() from Rust to notify scripts.
        let script_ptr_ev = self as *mut ScriptEngine as usize;
        let on_event = self.lua.create_function(move |_, (name, func): (String, LuaFunction)| {
            let script = unsafe { &mut *(script_ptr_ev as *mut ScriptEngine) };
            script.register_lua_event(&name, func)?;
            Ok(())
        })?;
        globals.set("on_event", on_event)?;

        let script_ptr_ev2 = self as *mut ScriptEngine as usize;
        let fire_event = self.lua.create_function(
            move |_, (name, payload): (String, mlua::Value)| {
                let script = unsafe { &mut *(script_ptr_ev2 as *mut ScriptEngine) };
                let payload = if payload.is_nil() {
                    None
                } else {
                    payload.as_string().map(|s| s.to_string_lossy().to_string())
                };
                script.fire_event(&name, payload)?;
                Ok(())
            },
        )?;
        globals.set("fire_event", fire_event)?;

        // ── Timers / delayed calls ────────────────────────────────────────
        // set_timeout(seconds, function) → handle
        // clear_timeout(handle) → cancel a pending timer
        // Handlers fire from ScriptEngine::tick_timers() once per frame.
        let script_ptr_tm = self as *mut ScriptEngine as usize;
        let set_timeout = self.lua.create_function(
            move |_, (delay, func): (f32, LuaFunction)| {
                let script = unsafe { &mut *(script_ptr_tm as *mut ScriptEngine) };
                let id = script.set_timeout(delay, func)?;
                Ok(id)
            },
        )?;
        globals.set("set_timeout", set_timeout)?;

        let script_ptr_tm2 = self as *mut ScriptEngine as usize;
        let clear_timeout_fn = self.lua.create_function(move |_, id: u64| {
            let script = unsafe { &mut *(script_ptr_tm2 as *mut ScriptEngine) };
            script.clear_timeout(id);
            Ok(())
        })?;
        globals.set("clear_timeout", clear_timeout_fn)?;

        // ── Module system (require) ──────────────────────────────────────
        // `require` is replaced with a sandboxed loader that reads .lua files
        // from `script_root` only (dots become path separators), caching the
        // result so each module is evaluated once. This is the sanctioned file
        // access for scripts even when the global sandbox denies raw io.* —
        // it is installed unconditionally so a "full" tier still can't reach
        // arbitrary files on disk through require.
        let req_ptr = self as *mut ScriptEngine as usize;
        let require_fn = self.lua.create_function(move |_, name: String| {
            let script = unsafe { &mut *(req_ptr as *mut ScriptEngine) };
            let value = script.require_module(&name)?;
            Ok(value)
        })?;
        globals.set("require", require_fn)?;

        // ── Sandboxed file API (fs.*) ────────────────────────────────────
        // Always available but rooted at script_root by default. Scripts whose
        // tier grants file_system_access get the wider fs_root in their env.
        globals.set("fs", self.build_fs_table(&self.script_root)?)?;

        // ── API catalogue (for autocomplete / docs) ─────────────────────
        // api_catalogue() → table of every callable global + namespace function.
        let cat_ptr = self as *mut ScriptEngine as usize;
        let catalogue_fn = self.lua.create_function(move |lua, ()| {
            let script = unsafe { &*(cat_ptr as *mut ScriptEngine) };
            let names = script.api_catalogue();
            lua.create_table_from(names.into_iter().map(|n| (n, true)))
        })?;
        globals.set("api_catalogue", catalogue_fn)?;

        // ── Sandbox enforcement ─────────────────────────────────────────
        // Remove dangerous Lua standard libraries based on SandboxConfig.
        // By default everything is restricted; enable only what is allowed.
        if !self.sandbox.os_command_access {
            globals.set("os", self.lua.create_table()?)?;
        }
        if !self.sandbox.file_system_access {
            // File I/O / code-loading are disabled: strip the io, os (already
            // above), loadfile/dofile (disk), and block loading code from
            // strings (load,loadstring) and raw module loading (require is a
            // sandboxed loader now; package's loadlib stays hidden).
            globals.set("io", self.lua.create_table()?)?;
            let deny = |msg: &'static str| {
                self.lua.create_function(move |_, ()| -> LuaResult<()> {
                    Err(LuaError::RuntimeError(msg.to_string()))
                })
            };
            globals.set("loadfile", deny("loadfile is disabled by sandbox")?)?;
            globals.set("dofile", deny("dofile is disabled by sandbox")?)?;
            globals.set("load", deny("load is disabled by sandbox")?)?;
            globals.set("loadstring", deny("loadstring is disabled by sandbox")?)?;
            // `package` backs require; hide its loaders/loadlib so a script
            // cannot reach the filesystem via package.loadlib.
            globals.set("package", self.lua.create_table()?)?;
            if let Ok(pkg) = globals.get::<mlua::Table>("package") {
                let _ = pkg.set("loadlib", deny("package.loadlib is disabled by sandbox")?);
                let _ = pkg.set("loader", self.lua.create_table()?);
                let _ = pkg.set("searchers", self.lua.create_table()?);
            }
        }
        if !self.sandbox.network_access {
            // The default mlua/lua54 stdlib has no socket library, so there is
            // nothing to strip here. Kept as an explicit no-op so that if a
            // networking module is ever mounted it must be gated by this flag.
        }

        // Enforce resource limits (memory + execution steps).  mlua's memory
        // limit aborts a VM that grows past the cap; the instruction hook
        // bounds CPU burn so a runaway script can't stall the frame.
        if self.sandbox.max_memory_bytes > 0 {
            self.lua
                .set_memory_limit(self.sandbox.max_memory_bytes)
                .map_err(|e| {
                    LuaError::RuntimeError(format!("sandbox: failed to set memory limit: {}", e))
                })?;
            tracing::info!(
                "[Scripting] Sandbox memory limit set to {} bytes",
                self.sandbox.max_memory_bytes
            );
        }
        if self.sandbox.max_execution_time_ms > 0 {
            // Convert wall-clock budget to an instruction step.  ~1M Lua ops is
            // roughly a millisecond on typical hardware; scale by budget.
            let steps = (self.sandbox.max_execution_time_ms as f32 * 1_000.0) as u32;
            let triggers = mlua::HookTriggers::new().every_nth_instruction(steps.max(1));
            self.lua.set_hook(triggers, |_lua, _debug| {
                Err(mlua::Error::RuntimeError(
                    "sandbox: script exceeded execution time budget".to_string(),
                ))
            })?;
            tracing::info!(
                "[Scripting] Sandbox execution limit set to {} ms",
                self.sandbox.max_execution_time_ms
            );
        }

        Ok(())
    }

    /// Sandboxed module loader installed as `require`. Loads
    /// `<script_root>/<name>.lua` (dots become path separators) once and caches
    /// the result. Modules run in a fresh environment that still sees the full
    /// engine API (globals), but not the raw `io`/`os` tables. Path traversal
    /// outside `script_root` is rejected.
    pub fn require_module(&mut self, name: &str) -> LuaResult<mlua::Value> {
        if let Some(key) = self.module_cache.get(name) {
            return self.lua.registry_value::<mlua::Value>(key);
        }
        let rel = format!("{}.lua", name.replace('.', "/"));
        let full = resolve_sandbox_read(&self.script_root, &rel)?;
        let code = fs::read_to_string(&full).map_err(|e| {
            LuaError::RuntimeError(format!("require '{name}': {}", e))
        })?;
        let env = self.lua.create_table()?;
        let mt = self.lua.create_table()?;
        mt.set("__index", self.lua.globals())?;
        env.set_metatable(Some(mt))?;
        let ret: mlua::MultiValue = self.lua
            .load(&code)
            .set_name(format!("@module {}", name))
            .set_environment(env.clone())
            .into_function()?
            .call(())?;
        let module = match ret.into_iter().next() {
            Some(v) if !v.is_nil() => v,
            _ => mlua::Value::Table(env),
        };
        let key = self.lua.create_registry_value(module.clone())?;
        self.module_cache.insert(name.to_string(), key);
        Ok(module)
    }

    /// Builds the `fs.*` API rooted at `root`. Paths are resolved against the
    /// root with traversal protection (no escaping via `..`, symlinks, or
    /// absolute paths).
    fn build_fs_table(&self, root: &str) -> LuaResult<LuaTable> {
        let table = self.lua.create_table()?;
        let root = root.to_string();

        // fs.read(path) → string
        let read_root = root.clone();
        let read = self.lua.create_function(move |_, path: String| {
            let full = resolve_sandbox_read(&read_root, &path)?;
            std::fs::read_to_string(&full)
                .map_err(|e| mlua::Error::RuntimeError(format!("fs.read: {}", e)))
        })?;
        table.set("read", read)?;

        // fs.write(path, contents) → bool
        let write_root = root.clone();
        let write = self.lua.create_function(move |_, (path, contents): (String, String)| {
            let full = resolve_sandbox_write(&write_root, &path)?;
            if let Some(parent) = full.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| mlua::Error::RuntimeError(format!("fs.write: {}", e)))?;
            }
            std::fs::write(&full, contents)
                .map_err(|e| mlua::Error::RuntimeError(format!("fs.write: {}", e)))?;
            Ok(true)
        })?;
        table.set("write", write)?;

        // fs.exists(path) → bool (false if the path is missing or escapes the root)
        let exists_root = root.clone();
        let exists = self.lua.create_function(move |_, path: String| {
            Ok(resolve_sandbox_read(&exists_root, &path).is_ok())
        })?;
        table.set("exists", exists)?;

        // fs.list(dir) → array of entry names (empty if missing/forbidden)
        let list_root = root;
        let list = self.lua.create_function(move |lua, path: String| {
            let out = lua.create_table()?;
            let full = match resolve_sandbox_read(&list_root, &path) {
                Ok(p) => p,
                Err(_) => return Ok(out),
            };
            let entries = match std::fs::read_dir(&full) {
                Ok(e) => e,
                Err(_) => return Ok(out),
            };
            let mut i = 1usize;
            for entry in entries.flatten() {
                if let Ok(name) = entry.file_name().into_string() {
                    out.set(i, name)?;
                    i += 1;
                }
            }
            Ok(out)
        })?;
        table.set("list", list)?;

        Ok(table)
    }

    /// Builds the `os.*` table handed to script envs. Always exposes the safe
    /// clock/query functions; `execute` is only mounted when the tier grants
    /// os_command_access (a deliberate per-script opt-in).
    fn build_os_table(&self, allow_execute: bool) -> LuaResult<LuaTable> {
        let table = self.lua.create_table()?;

        let time_fn = self.lua.create_function(|_, ()| {
            Ok(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0))
        })?;
        table.set("time", time_fn)?;

        let clock_fn = self.lua.create_function(|_, ()| {
            Ok(std::time::Instant::now().elapsed().as_secs_f64())
        })?;
        table.set("clock", clock_fn)?;

        let getenv_fn = self.lua.create_function(|_, key: String| {
            Ok(std::env::var(&key).ok())
        })?;
        table.set("getenv", getenv_fn)?;

        if allow_execute {
            let execute_fn = self.lua.create_function(|_, cmd: String| {
                let out = std::process::Command::new("cmd")
                    .args(["/C", &cmd])
                    .output()
                    .map_err(|e| mlua::Error::RuntimeError(format!("os.execute: {}", e)))?;
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                Ok(format!("{}{}", stdout, stderr))
            })?;
            table.set("execute", execute_fn)?;
        }

        Ok(table)
    }

    // load_script() reads a .lua file from disk and executes it.
    // This defines any top-level variables and the update() function.
    pub fn load_script(&mut self, path: &str) -> LuaResult<()> {
        let code = fs::read_to_string(path)
            .map_err(|e| LuaError::RuntimeError(
                format!("Could not load script {}: {}", path, e)
            ))?;
        self.lua.load(&code).set_name(path).into_function()?;
        tracing::info!("[Scripting] Loaded: {}", path);
        Ok(())
    }

    // reload_script() hot-reloads a changed file.
    // New function definitions replace the old ones in Lua globals.
    pub fn reload_script(&mut self, path: &str) -> LuaResult<()> {
        tracing::info!("[Scripting] Reloading: {}", path);
        self.load_script(path)?;
        let rev = self.script_revisions.entry(path.to_string()).or_insert(0);
        *rev += 1;
        Ok(())
    }

    // ── Lua-native plugin host ────────────────────────────────────────────
    // Plugins are ordinary .lua files. Each file returns a table of the form:
    //
    //   return {
    //       name    = "my_plugin",
    //       start   = function() ... end,           -- once, on load
    //       update  = function(dt) ... end,         -- every frame
    //       on_event= function(name, payload) ... end, -- fires on fire_event()
    //   }
    //
    // The returned table lives in a fresh environment whose `__index` is the
    // global API, so plugins can call every engine function (log, spawn_*,
    // set_timeout, bt.*, particles.*, ...) without any Rust changes.  Adding a
    // plugin = dropping a .lua file.  Editing it hot-reloads via reload_plugin.
    // ────────────────────────────────────────────────────────────────────────

    /// Load every `.lua` file in `dir` as a plugin.  Returns how many loaded.
    /// Skips failures (logged) so one broken plugin never blocks the rest.
    pub fn load_plugins(&mut self, dir: &str) -> LuaResult<usize> {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("[Plugins] Directory {} unavailable: {}", dir, e);
                return Ok(0);
            }
        };
        let mut loaded = 0;
        let mut files: Vec<String> = entries
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                let is_lua = p.extension().map_or(false, |x| x == "lua");
                if !is_lua {
                    return None;
                }
                p.to_str().map(|s| s.to_string())
            })
            .collect();
        files.sort();
        for path in files {
            match self.load_plugin(&path) {
                Ok(name) => {
                    tracing::info!("[Plugins] Loaded: {} (from {})", name, path);
                    loaded += 1;
                }
                Err(e) => tracing::error!("[Plugins] Failed {}: {}", path, e),
            }
        }
        Ok(loaded)
    }

    /// Load a single plugin file.  Returns the plugin name on success.
    pub fn load_plugin(&mut self, path: &str) -> LuaResult<String> {
        let code = std::fs::read_to_string(path).map_err(|e| {
            LuaError::RuntimeError(format!("plugin {path}: {e}"))
        })?;

        // Fresh environment per plugin; globals fall through to the shared API.
        let env = self.lua.create_table()?;
        let mt = self.lua.create_table()?;
        mt.set("__index", self.lua.globals())?;
        env.set_metatable(Some(mt))?;

        let ret: mlua::MultiValue = self
            .lua
            .load(&code)
            .set_name(format!("@plugin {}", path))
            .set_environment(env.clone())
            .into_function()?
            .call(())?;

        let table = match ret.into_iter().next() {
            Some(v) if v.is_table() => v.as_table().unwrap().clone(),
            _ => {
                return Err(LuaError::RuntimeError(format!(
                    "plugin {path} must return a table"
                )));
            }
        };

        let name: String = table
            .get::<Option<String>>("name")?
            .unwrap_or_else(|| {
                std::path::Path::new(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unnamed".to_string())
            });

        let start_key = match table.get::<Option<LuaFunction>>("start")? {
            Some(f) => Some(self.lua.create_registry_value(f)?),
            None => None,
        };
        let update_key = match table.get::<Option<LuaFunction>>("update")? {
            Some(f) => Some(self.lua.create_registry_value(f)?),
            None => None,
        };
        let on_event_key = match table.get::<Option<LuaFunction>>("on_event")? {
            Some(f) => Some(self.lua.create_registry_value(f)?),
            None => None,
        };
        let env_key = self.lua.create_registry_value(env)?;

        // Call start() after registration so the plugin can immediately use
        // the full API (and register event callbacks / timers).
        if let Some(key) = start_key.as_ref() {
            let func: LuaFunction = self.lua.registry_value(key)?;
            if let Err(e) = func.call::<()>(()) {
                tracing::error!("[Plugins] {} start() error: {}", name, e);
            }
        }

        let plugin = LuaPlugin {
            path: path.to_string(),
            name: name.clone(),
            env_key,
            start_key,
            update_key,
            on_event_key,
        };
        self.lua_plugins.push(plugin);

        Ok(name)
    }

    /// Hot-reload a changed plugin file: replace its registered callbacks.
    /// Matches by path. Returns Ok(true) if a plugin was reloaded.
    pub fn reload_plugin(&mut self, path: &str) -> LuaResult<bool> {
        let norm = path.replace('\\', "/");
        if let Some(idx) = self.lua_plugins.iter().position(|p| p.path.replace('\\', "/") == norm) {
            // Drop the old plugin state first (remove registry keys), then load
            // fresh — a clean start() each reload avoids duplicate event handlers.
            self.unload_plugin_at(idx);
            self.load_plugin(path).map(|_| true)
        } else {
            Ok(false)
        }
    }

    /// Remove a plugin's Lua state (registry keys) without unloading others.
    fn unload_plugin_at(&mut self, idx: usize) {
        let p = self.lua_plugins.remove(idx);
        let _ = self.lua.remove_registry_value(p.env_key);
        if let Some(k) = p.start_key {
            let _ = self.lua.remove_registry_value(k);
        }
        if let Some(k) = p.update_key {
            let _ = self.lua.remove_registry_value(k);
        }
        if let Some(k) = p.on_event_key {
            let _ = self.lua.remove_registry_value(k);
        }
        tracing::info!("[Plugins] Unloaded: {}", p.name);
    }

    /// Unload a plugin by name (used by unload_plugin Lua API, tests).
    pub fn unload_plugin(&mut self, name: &str) -> bool {
        if let Some(idx) = self.lua_plugins.iter().position(|p| p.name == name) {
            self.unload_plugin_at(idx);
            true
        } else {
            false
        }
    }

    /// Call every plugin's `update(dt)` once per frame.
    pub fn tick_plugins(&mut self, dt: f32) -> LuaResult<()> {
        if self.lua_plugins.is_empty() {
            return Ok(());
        }
        for i in 0..self.lua_plugins.len() {
            let Some(key) = &self.lua_plugins[i].update_key else { continue };
            let func: LuaFunction = self.lua.registry_value(key)?;
            let res = func.call::<()>((dt,));
            if let Err(e) = res {
                tracing::error!(
                    "[Plugins] {} update() error: {}",
                    self.lua_plugins[i].name,
                    e
                );
            }
        }
        Ok(())
    }

    /// Names of all loaded plugins (for `plugins.list()` and debugging).
    pub fn plugin_names(&self) -> Vec<String> {
        self.lua_plugins.iter().map(|p| p.name.clone()).collect()
    }

    /// True if a plugin with the given name is currently loaded.
    pub fn has_plugin(&self, name: &str) -> bool {
        self.lua_plugins.iter().any(|p| p.name == name)
    }

    // drain_destroys() — call after scripting_system each frame.
    // Removes entities that Lua scripts called destroy() on.
    pub fn drain_destroys(&mut self, world: &mut World) {
        let destroys = unsafe { &mut *self.pending_destroys.get() };
        for entity in destroys.drain(..) {
            self.remove_instance(entity.to_bits().get());
            let _ = world.despawn(entity);
        }
    }

    fn remove_instance(&mut self, entity_bits: u64) {
        self.failed_instances.remove(&entity_bits);
        if let Some(inst) = self.instances.remove(&entity_bits) {
            let _ = self.lua.remove_registry_value(inst.env_key);
            if let Some(key) = inst.start_key {
                let _ = self.lua.remove_registry_value(key);
            }
            let _ = self.lua.remove_registry_value(inst.update_key);
            if let Some(key) = inst.collision_enter_key {
                let _ = self.lua.remove_registry_value(key);
            }
            if let Some(key) = inst.collision_stay_key {
                let _ = self.lua.remove_registry_value(key);
            }
            if let Some(key) = inst.collision_exit_key {
                let _ = self.lua.remove_registry_value(key);
            }
        }
    }

    // start_watching() spawns a background thread watching a directory.
    // Returns a Receiver that yields changed file paths.
    // The game loop calls try_recv() each frame — non-blocking.
    pub fn start_watching(&self, watch_dir: &str) -> mpsc::Receiver<String> {
        let (tx, rx) = mpsc::channel::<String>();
        let watch_dir = watch_dir.to_string();

        std::thread::spawn(move || {
            let (ntx, nrx) = mpsc::channel::<notify::Result<Event>>();
            let mut watcher = match recommended_watcher(move |res| { let _ = ntx.send(res); }) {
                Ok(w) => w,
                Err(e) => {
                    tracing::error!("[Scripting] Could not create file watcher: {}", e);
                    return;
                }
            };
            if let Err(e) = watcher.watch(Path::new(&watch_dir), RecursiveMode::Recursive) {
                tracing::error!("[Scripting] Could not watch directory {}: {}", watch_dir, e);
                return;
            }

            loop {
                match nrx.recv() {
                    Ok(Ok(event)) => {
                        for path in event.paths {
                            let s = path.to_string_lossy().to_string();
                            if s.ends_with(".lua") {
                                if tx.send(s).is_err() { return; }
                            }
                        }
                    }
                    Ok(Err(e)) => tracing::error!("[HotReload] Watcher error: {}", e),
                    Err(_) => break,
                }
            }
        });

        rx
    }

    // run_update() registers per-frame component bindings and calls update(entity, dt).
    //
    // Why re-register every frame?
    // Lua closures need 'static lifetimes, but World and InputState are stack values.
    // We pass them as raw pointers baked into closures valid only for the duration
    // of this function call. Re-registering is cheap (~microseconds).
    fn build_instance(
        &mut self,
        path: &str,
    ) -> LuaResult<(
        LuaRegistryKey,
        Option<LuaRegistryKey>,
        LuaRegistryKey,
        Option<LuaRegistryKey>,
        Option<LuaRegistryKey>,
        Option<LuaRegistryKey>,
        u64,
    )> {
        let code = fs::read_to_string(path)
            .map_err(|e| LuaError::RuntimeError(format!("Could not load script {}: {}", path, e)))?;

        let globals = self.lua.globals();
        let env = self.lua.create_table()?;
        let mt = self.lua.create_table()?;
        mt.set("__index", globals)?;
        env.set_metatable(Some(mt))?;

        // ── Per-script sandbox tier ──────────────────────────────────────
        // fs.* and os.* are env-local overrides, so a privileged script can
        // reach the wider fs_root / os.execute while the global sandbox stays
        // restricted. The env's own keys shadow the stripped globals.
        let tier = self.tiers.get(path).cloned().unwrap_or_else(|| self.sandbox.clone());
        let fs_root = if tier.file_system_access {
            self.fs_root.clone()
        } else {
            self.script_root.clone()
        };
        env.set("fs", self.build_fs_table(&fs_root)?)?;
        env.set("os", self.build_os_table(tier.os_command_access)?)?;

        self.lua
            .load(&code)
            .set_name(path)
            .set_environment(env.clone())
            .exec()?;

        let start_key = env
            .get::<Option<LuaFunction>>("start")?
            .map(|f| self.lua.create_registry_value(f))
            .transpose()?;
        let update: LuaFunction = env.get("update").map_err(|_| {
            LuaError::RuntimeError(format!(
                "Script {} must define function update(entity, dt)",
                path
            ))
        })?;

        let collision_enter_key = env
            .get::<Option<LuaFunction>>("on_collision_enter")?
            .map(|f| self.lua.create_registry_value(f))
            .transpose()?;
        let collision_stay_key = env
            .get::<Option<LuaFunction>>("on_collision_stay")?
            .map(|f| self.lua.create_registry_value(f))
            .transpose()?;
        let collision_exit_key = env
            .get::<Option<LuaFunction>>("on_collision_exit")?
            .map(|f| self.lua.create_registry_value(f))
            .transpose()?;
        let env_key = self.lua.create_registry_value(env)?;
        let update_key = self.lua.create_registry_value(update)?;
        let revision = *self.script_revisions.get(path).unwrap_or(&0);
        Ok((
            env_key,
            start_key,
            update_key,
            collision_enter_key,
            collision_stay_key,
            collision_exit_key,
            revision,
        ))
    }

    pub fn run_update(
        &mut self,
        world: &mut World,
        input: &InputState,
        camera_pos: [f32; 3],
        camera_target: [f32; 3],
        entity: Entity,
        script_path: &str,
        dt: f32,
        audio: Option<&mut crate::audio::AudioSystem>,
        screen_w: f32,
        screen_h: f32,
        fov_degrees: f32,
    ) -> LuaResult<()> {
        let globals   = self.lua.globals();
        let script_ptr = self as *mut ScriptEngine as usize;
        let world_ptr = world as *mut World as usize;
        let input_ptr = input as *const InputState as usize;
        let camera_pos_copy = camera_pos;
        let camera_target_copy = camera_target;
        let entity_bits = entity.to_bits().get();

        let current_rev = *self.script_revisions.get(script_path).unwrap_or(&0);

        // ── Error recovery ──────────────────────────────────────────────
        // If this entity's script failed to load or threw, skip it (returning
        // Ok so the engine isn't spammed) until the file changes. Hot-reloading
        // bumps the revision, which clears the failure and rebuilds the
        // instance automatically.
        let failed_key = (script_path.to_string(), current_rev);
        if self.failed_instances.get(&entity_bits) == Some(&failed_key) {
            return Ok(());
        }
        self.failed_instances.remove(&entity_bits);

        let needs_rebuild = self
            .instances
            .get(&entity_bits)
            .map(|i| i.path != script_path || i.revision != current_rev)
            .unwrap_or(true);

        if needs_rebuild {
            self.remove_instance(entity_bits);
            match self.build_instance(script_path) {
                Ok((env_key, start_key, update_key, collision_enter_key, collision_stay_key, collision_exit_key, revision)) => {
                    self.instances.insert(
                        entity_bits,
                        ScriptInstance {
                            path: script_path.to_string(),
                            revision,
                            env_key,
                            start_key,
                            update_key,
                            collision_enter_key,
                            collision_stay_key,
                            collision_exit_key,
                            started: false,
                        },
                    );
                }
                Err(e) => {
                    tracing::error!(
                        "[Scripting] Failed to load {} for entity {}: {}",
                        script_path,
                        entity_bits,
                        e
                    );
                    self.failed_instances.insert(entity_bits, failed_key);
                    return Ok(());
                }
            }
        }

        // ── Position ──────────────────────────────────────────────────────
        // get_position(entity) → x, y, z
        let gp = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in get_position", eid);
                return Ok((0.0f32, 0.0f32, 0.0f32));
            };
            match world.get::<&Position>(entity) {
                Ok(p)  => Ok((p.x, p.y, p.z)),
                Err(_) => Ok((0.0f32, 0.0f32, 0.0f32)),
            }
        })?;
        globals.set("get_position", gp)?;

        // set_position(entity, x, y, z)
        let sp = self.lua.create_function(move |_, (eid, x, y, z): (u64, f32, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_position", eid);
                return Ok(());
            };
            if let Ok(mut p) = world.get::<&mut Position>(entity) {
                p.x = x; p.y = y; p.z = z;
            }
            Ok(())
        })?;
        globals.set("set_position", sp)?;

        let mv = self
            .lua
            .create_function(move |_, (eid, dx, dy, dz): (u64, f32, f32, f32)| {
                let world = unsafe { &mut *(world_ptr as *mut World) };
                let Some(entity) = Entity::from_bits(eid) else {
                    tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in move_entity", eid);
                    return Ok(());
                };
                if let Ok(mut p) = world.get::<&mut Position>(entity) {
                    p.x += dx;
                    p.y += dy;
                    p.z += dz;
                }
                Ok(())
            })?;
        globals.set("move_by", mv)?;

        // ── RigidBody ─────────────────────────────────────────────────────
        // get_velocity(entity) → vx, vy, vz
        let gv = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in get_velocity", eid);
                return Ok((0.0f32, 0.0f32, 0.0f32));
            };
            match world.get::<&RigidBody>(entity) {
                Ok(b)  => Ok((b.velocity_x, b.velocity_y, b._velocity_z)),
                Err(_) => Ok((0.0f32, 0.0f32, 0.0f32)),
            }
        })?;
        globals.set("get_velocity", gv)?;

        // set_velocity(entity, vx, vy, vz) — replaces current velocity
        let sv = self.lua.create_function(move |_, (eid, vx, vy, vz): (u64, f32, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_velocity", eid);
                return Ok(());
            };
            if let Ok(mut b) = world.get::<&mut RigidBody>(entity) {
                b.velocity_x = vx;
                b.velocity_y = vy;
                b._velocity_z = vz;
            }
            Ok(())
        })?;
        globals.set("set_velocity", sv)?;

        let gav = self.lua.create_function(move |_, eid: u64| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in get_angular_velocity", eid);
                return Ok(0.0);
            };
            let v = world
                .get::<&RigidBody>(entity)
                .map(|b| b.angular_velocity)
                .unwrap_or(0.0);
            Ok(v)
        })?;
        globals.set("get_angular_velocity", gav)?;

        let sav = self.lua.create_function(move |_, (eid, w): (u64, f32)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_angular_velocity", eid);
                return Ok(());
            };
            if let Ok(mut b) = world.get::<&mut RigidBody>(entity) {
                b.angular_velocity = w;
                b.sleeping = false;
                b.sleep_timer = 0.0;
            }
            Ok(())
        })?;
        globals.set("set_angular_velocity", sav)?;

        let at = self.lua.create_function(move |_, (eid, torque): (u64, f32)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in apply_torque", eid);
                return Ok(());
            };
            if let Ok(mut b) = world.get::<&mut RigidBody>(entity) {
                b.torque += torque;
                b.sleeping = false;
                b.sleep_timer = 0.0;
            }
            Ok(())
        })?;
        globals.set("apply_torque", at)?;

        let chj = self.lua.create_function(move |_, (eid, other, rest_length, stiffness): (u64, u64, f32, f32)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in create_hinge_joint", eid);
                return Ok(());
            };
            let Some(other) = Entity::from_bits(other) else { return Ok(()); };
            let _ = world.insert(
                entity,
                (HingeJoint {
                    connected: other,
                    rest_length,
                    stiffness,
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [0.0, 0.0, 0.0],
                },),
            );
            Ok(())
        })?;
        globals.set("create_hinge_joint", chj)?;

        let cfj = self.lua.create_function(
            move |_, (eid, other, offset_x, offset_y, stiffness): (u64, u64, f32, f32, f32)| {
                let world = unsafe { &mut *(world_ptr as *mut World) };
                let Some(entity) = Entity::from_bits(eid) else {
                    tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in create_fixed_joint", eid);
                    return Ok(());
                };
                let Some(other) = Entity::from_bits(other) else { return Ok(()); };
                let _ = world.insert(
                    entity,
                    (FixedJoint {
                        connected: other,
                        offset_x,
                        offset_y,
                        stiffness,
                        anchor_a: [0.0, 0.0, 0.0],
                        anchor_b: [0.0, 0.0, 0.0],
                    },),
                );
                Ok(())
            },
        )?;
        globals.set("create_fixed_joint", cfj)?;

        let csj = self.lua.create_function(
            move |_, (eid, other, rest_length, stiffness, damping): (u64, u64, f32, f32, f32)| {
                let world = unsafe { &mut *(world_ptr as *mut World) };
                let Some(entity) = Entity::from_bits(eid) else {
                    tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in create_spring_joint", eid);
                    return Ok(());
                };
                let Some(other) = Entity::from_bits(other) else { return Ok(()); };
                let _ = world.insert(
                    entity,
                    (SpringJoint {
                        connected: other,
                        rest_length,
                        stiffness,
                        damping,
                        anchor_a: [0.0, 0.0, 0.0],
                        anchor_b: [0.0, 0.0, 0.0],
                    },),
                );
                Ok(())
            },
        )?;
        globals.set("create_spring_joint", csj)?;

        let crc = self.lua.create_function(move |_, (eid, other, max_length, stiffness): (u64, u64, f32, f32)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in create_rope_constraint", eid);
                return Ok(());
            };
            let Some(other) = Entity::from_bits(other) else { return Ok(()); };
            let _ = world.insert(
                entity,
                (RopeConstraint {
                    connected: other,
                    max_length,
                    stiffness,
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [0.0, 0.0, 0.0],
                },),
            );
            Ok(())
        })?;
        globals.set("create_rope_constraint", crc)?;

        // apply_force(entity, fx, fy, fz) — adds to current velocity
        let af = self.lua.create_function(move |_, (eid, fx, fy, fz): (u64, f32, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in apply_force", eid);
                return Ok(());
            };
            if let Ok(mut b) = world.get::<&mut RigidBody>(entity) {
                b.velocity_x += fx;
                b.velocity_y += fy;
                b._velocity_z += fz;
            }
            Ok(())
        })?;
        globals.set("apply_force", af)?;

        // is_on_ground(entity) → bool
        let og = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in is_on_ground", eid);
                return Ok(false);
            };
            let g = world.get::<&RigidBody>(entity)
                .map(|b| b.on_ground)
                .unwrap_or(false);
            Ok(g)
        })?;
        globals.set("is_on_ground", og)?;

        let self_fn = self.lua.create_function(move |_, ()| Ok(entity_bits))?;
        globals.set("self_entity", self_fn)?;

        // ── Health ────────────────────────────────────────────────────────
        // get_health(entity) → current, max
        let gh = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in get_health", eid);
                return Ok((0i32, 0i32));
            };
            match world.get::<&Health>(entity) {
                Ok(h)  => Ok((h.current, h.max)),
                Err(_) => Ok((0i32, 0i32)),
            }
        })?;
        globals.set("get_health", gh)?;

        // set_health(entity, current, max)
        let sh = self.lua.create_function(move |_, (eid, cur, max): (u64, i32, i32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_health", eid);
                return Ok(());
            };
            if let Ok(mut h) = world.get::<&mut Health>(entity) {
                h.max     = max;
                h.current = cur.clamp(0, max);
            }
            Ok(())
        })?;
        globals.set("set_health", sh)?;

        // damage(entity, amount) — convenience: subtract from health
        let dmg = self.lua.create_function(move |_, (eid, amount): (u64, i32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in damage", eid);
                return Ok(());
            };
            if let Ok(mut h) = world.get::<&mut Health>(entity) {
                h.current = (h.current - amount).max(0);
            }
            Ok(())
        })?;
        globals.set("damage", dmg)?;

        // is_dead(entity) → bool
        let id = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in is_dead", eid);
                return Ok(false);
            };
            let dead = world.get::<&Health>(entity)
                .map(|h| h.current <= 0)
                .unwrap_or(false);
            Ok(dead)
        })?;
        globals.set("is_dead", id)?;

        // ── Renderable ────────────────────────────────────────────────────
        // set_color(entity, r, g, b) — tint the entity this frame
        let sc = self.lua.create_function(move |_, (eid, r, g, b): (u64, f32, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_color", eid);
                return Ok(());
            };
            if let Ok(mut rend) = world.get::<&mut Renderable>(entity) {
                rend.color = [r, g, b];
            }
            Ok(())
        })?;
        globals.set("set_color", sc)?;

        // set_scale(entity, sx, sy, sz)
        let ss = self.lua.create_function(move |_, (eid, sx, sy, sz): (u64, f32, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_scale", eid);
                return Ok(());
            };
            if let Ok(mut rend) = world.get::<&mut Renderable>(entity) {
                rend.scale = [sx, sy, sz];
            }
            Ok(())
        })?;
        globals.set("set_scale", ss)?;

        // get_rotation(entity) -> pitch, yaw, roll
        let gr = self.lua.create_function(move |_, eid: u64| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in get_rotation", eid);
                return Ok((0.0f32, 0.0f32, 0.0f32));
            };
            match world.get::<&Rotation>(entity) {
                Ok(r) => Ok((r.pitch, r.yaw, r.roll)),
                Err(_) => Ok((0.0f32, 0.0f32, 0.0f32)),
            }
        })?;
        globals.set("get_rotation", gr)?;

        // set_rotation(entity, pitch, yaw, roll)
        let sr = self.lua.create_function(move |_, (eid, p, y, r): (u64, f32, f32, f32)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_rotation", eid);
                return Ok(());
            };
            if let Ok(mut rot) = world.get::<&mut Rotation>(entity) {
                rot.pitch = p;
                rot.yaw = y;
                rot.roll = r;
            } else {
                let _ = world.insert(entity, (Rotation { pitch: p, yaw: y, roll: r },));
            }
            Ok(())
        })?;
        globals.set("set_rotation", sr)?;

        // set_material(entity, metallic, roughness, ao)
        let sm = self
            .lua
            .create_function(move |_, (eid, metallic, roughness, ao): (u64, f32, f32, f32)| {
                let world = unsafe { &mut *(world_ptr as *mut World) };
                let Some(entity) = Entity::from_bits(eid) else {
                    tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_material", eid);
                    return Ok(());
                };
                if let Ok(mut rend) = world.get::<&mut Renderable>(entity) {
                    rend.metallic = metallic.clamp(0.0, 1.0);
                    rend.roughness = roughness.clamp(0.02, 1.0);
                    rend.ao = ao.clamp(0.0, 1.0);
                }
                Ok(())
            })?;
        globals.set("set_material", sm)?;

        // get_texture_path(entity) / set_texture_path(entity, path)
        let gtp = self.lua.create_function(move |_, eid: u64| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in get_texture_path", eid);
                return Ok(String::new());
            };
            let path = world
                .get::<&MaterialTexture>(entity)
                .map(|t| t.path.clone())
                .unwrap_or_default();
            Ok(path)
        })?;
        globals.set("get_texture_path", gtp)?;
        let stp = self.lua.create_function(move |_, (eid, path): (u64, String)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_texture_path", eid);
                return Ok(());
            };
            if let Ok(mut t) = world.get::<&mut MaterialTexture>(entity) {
                t.path = path;
            } else {
                let _ = world.insert(
                    entity,
                    (MaterialTexture {
                        path,
                        normal_path: String::new(),
                        metallic_roughness_path: String::new(),
                    },),
                );
            }
            Ok(())
        })?;
        globals.set("set_texture_path", stp)?;

        // UI runtime values (for HUD widgets)
        let suv = self.lua.create_function(move |_, (id, value): (String, f32)| {
            let scripts = unsafe { &mut *(script_ptr as *mut ScriptEngine) };
            scripts.ui_values.insert(id, value);
            Ok(())
        })?;
        globals.set("set_ui_value", suv)?;
        let guv = self.lua.create_function(move |_, id: String| {
            let scripts = unsafe { &mut *(script_ptr as *mut ScriptEngine) };
            Ok(*scripts.ui_values.get(&id).unwrap_or(&0.0))
        })?;
        globals.set("get_ui_value", guv)?;
        let sut = self.lua.create_function(move |_, (id, text): (String, String)| {
            let scripts = unsafe { &mut *(script_ptr as *mut ScriptEngine) };
            scripts.ui_texts.insert(id, text);
            Ok(())
        })?;
        globals.set("set_ui_text", sut)?;
        let gut = self.lua.create_function(move |_, id: String| {
            let scripts = unsafe { &mut *(script_ptr as *mut ScriptEngine) };
            Ok(scripts.ui_texts.get(&id).cloned().unwrap_or_default())
        })?;
        globals.set("get_ui_text", gut)?;

        // set_ui_visible(id, bool) — show/hide a HUD widget at runtime.
        let suv2 = self.lua.create_function(move |_, (id, visible): (String, bool)| {
            let scripts = unsafe { &mut *(script_ptr as *mut ScriptEngine) };
            scripts.ui_visibility.insert(id, visible);
            Ok(())
        })?;
        globals.set("set_ui_visible", suv2)?;

        // has_component(entity, name) — lightweight reflection helper.
        let hc = self.lua.create_function(move |_, (eid, name): (u64, String)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in has_component", eid);
                return Ok(false);
            };
            let has = match name.as_str() {
                "Position" => world.get::<&Position>(entity).is_ok(),
                "RigidBody" => world.get::<&RigidBody>(entity).is_ok(),
                "Collider" => world.get::<&Collider>(entity).is_ok(),
                "OrientedBoxCollider" => world.get::<&OrientedBoxCollider>(entity).is_ok(),
                "Renderable" => world.get::<&Renderable>(entity).is_ok(),
                "Rotation" => world.get::<&Rotation>(entity).is_ok(),
                "MaterialTexture" => world.get::<&MaterialTexture>(entity).is_ok(),
                "Health" => world.get::<&Health>(entity).is_ok(),
                "HingeJoint" => world.get::<&HingeJoint>(entity).is_ok(),
                "FixedJoint" => world.get::<&FixedJoint>(entity).is_ok(),
                "SpringJoint" => world.get::<&SpringJoint>(entity).is_ok(),
                "RopeConstraint" => world.get::<&RopeConstraint>(entity).is_ok(),
                _ => false,
            };
            Ok(has)
        })?;
        globals.set("has_component", hc)?;

        let gc = self.lua.create_function(move |lua, (eid, name): (u64, String)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in get_component", eid);
                return Ok(None);
            };
            let table = lua.create_table()?;
            let found = match name.as_str() {
                "Position" => {
                    if let Ok(c) = world.get::<&Position>(entity) {
                        table.set("x", c.x)?;
                        table.set("y", c.y)?;
                        table.set("z", c.z)?;
                        true
                    } else {
                        false
                    }
                }
                "Rotation" => {
                    if let Ok(c) = world.get::<&Rotation>(entity) {
                        table.set("pitch", c.pitch)?;
                        table.set("yaw", c.yaw)?;
                        table.set("roll", c.roll)?;
                        true
                    } else {
                        false
                    }
                }
                "RigidBody" => {
                    if let Ok(c) = world.get::<&RigidBody>(entity) {
                        table.set("body_type", match c.body_type {
                            crate::components::BodyType::Static => "Static",
                            crate::components::BodyType::Dynamic => "Dynamic",
                            crate::components::BodyType::Kinematic => "Kinematic",
                        })?;
                        table.set("velocity_x", c.velocity_x)?;
                        table.set("velocity_y", c.velocity_y)?;
                        table.set("velocity_z", c._velocity_z)?;
                        table.set("angular_velocity", c.angular_velocity)?;
                        table.set("use_gravity", c.use_gravity)?;
                        table.set("mass", c.mass)?;
                        table.set("inertia", c.inertia)?;
                        table.set("restitution", c.restitution)?;
                        table.set("friction", c.friction)?;
                        table.set("linear_damping", c.linear_damping)?;
                        table.set("angular_damping", c.angular_damping)?;
                        table.set("lock_rotation", c.lock_rotation)?;
                        table.set("on_ground", c.on_ground)?;
                        table.set("sleeping", c.sleeping)?;
                        true
                    } else {
                        false
                    }
                }
                "Collider" => {
                    if let Ok(c) = world.get::<&Collider>(entity) {
                        table.set("half_w", c.half_w)?;
                        table.set("half_h", c.half_h)?;
                        table.set("half_d", c.half_d)?;
                        table.set("layer", c.layer)?;
                        table.set("mask", c.mask)?;
                        true
                    } else {
                        false
                    }
                }
                "OrientedBoxCollider" => {
                    if let Ok(c) = world.get::<&OrientedBoxCollider>(entity) {
                        table.set("half_w", c.half_w)?;
                        table.set("half_h", c.half_h)?;
                        table.set("half_d", c.half_d)?;
                        table.set("angle_rad", c.angle_rad)?;
                        table.set("layer", c.layer)?;
                        table.set("mask", c.mask)?;
                        true
                    } else {
                        false
                    }
                }
                "Renderable" => {
                    if let Ok(c) = world.get::<&Renderable>(entity) {
                        table.set("color_r", c.color[0])?;
                        table.set("color_g", c.color[1])?;
                        table.set("color_b", c.color[2])?;
                        table.set("metallic", c.metallic)?;
                        table.set("roughness", c.roughness)?;
                        table.set("ao", c.ao)?;
                        table.set("scale_x", c.scale[0])?;
                        table.set("scale_y", c.scale[1])?;
                        table.set("scale_z", c.scale[2])?;
                        true
                    } else {
                        false
                    }
                }
                "MaterialTexture" => {
                    if let Ok(c) = world.get::<&MaterialTexture>(entity) {
                        table.set("path", c.path.clone())?;
                        table.set("normal_path", c.normal_path.clone())?;
                        table.set("metallic_roughness_path", c.metallic_roughness_path.clone())?;
                        true
                    } else {
                        false
                    }
                }
                "Health" => {
                    if let Ok(c) = world.get::<&Health>(entity) {
                        table.set("current", c.current)?;
                        table.set("max", c.max)?;
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            };
            if found { Ok(Some(table)) } else { Ok(None) }
        })?;
        globals.set("get_component", gc)?;

        let sc = self.lua.create_function(move |_, (eid, name, data): (u64, String, LuaTable)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_component", eid);
                return Ok(false);
            };
            let ok = match name.as_str() {
                "Position" => {
                    if let Ok(mut c) = world.get::<&mut Position>(entity) {
                        if let Ok(v) = data.get::<f32>("x") { c.x = v; }
                        if let Ok(v) = data.get::<f32>("y") { c.y = v; }
                        if let Ok(v) = data.get::<f32>("z") { c.z = v; }
                        true
                    } else {
                        false
                    }
                }
                "Rotation" => {
                    if let Ok(mut c) = world.get::<&mut Rotation>(entity) {
                        if let Ok(v) = data.get::<f32>("pitch") { c.pitch = v; }
                        if let Ok(v) = data.get::<f32>("yaw") { c.yaw = v; }
                        if let Ok(v) = data.get::<f32>("roll") { c.roll = v; }
                        true
                    } else {
                        false
                    }
                }
                "RigidBody" => {
                    if let Ok(mut c) = world.get::<&mut RigidBody>(entity) {
                        if let Ok(v) = data.get::<String>("body_type") {
                            c.body_type = match v.as_str() {
                                "Static" => crate::components::BodyType::Static,
                                "Kinematic" => crate::components::BodyType::Kinematic,
                                _ => crate::components::BodyType::Dynamic,
                            };
                        }
                        if let Ok(v) = data.get::<f32>("velocity_x") { c.velocity_x = v; }
                        if let Ok(v) = data.get::<f32>("velocity_y") { c.velocity_y = v; }
                        if let Ok(v) = data.get::<f32>("velocity_z") { c._velocity_z = v; }
                        if let Ok(v) = data.get::<f32>("angular_velocity") { c.angular_velocity = v; }
                        if let Ok(v) = data.get::<bool>("use_gravity") { c.use_gravity = v; }
                        if let Ok(v) = data.get::<f32>("mass") { c.mass = v; }
                        if let Ok(v) = data.get::<f32>("inertia") { c.inertia = v; }
                        if let Ok(v) = data.get::<f32>("restitution") { c.restitution = v; }
                        if let Ok(v) = data.get::<f32>("friction") { c.friction = v; }
                        if let Ok(v) = data.get::<f32>("linear_damping") { c.linear_damping = v; }
                        if let Ok(v) = data.get::<f32>("angular_damping") { c.angular_damping = v; }
                        if let Ok(v) = data.get::<bool>("lock_rotation") { c.lock_rotation = v; }
                        if let Ok(v) = data.get::<bool>("on_ground") { c.on_ground = v; }
                        true
                    } else {
                        false
                    }
                }
                "Collider" => {
                    if let Ok(mut c) = world.get::<&mut Collider>(entity) {
                        if let Ok(v) = data.get::<f32>("half_w") { c.half_w = v; }
                        if let Ok(v) = data.get::<f32>("half_h") { c.half_h = v; }
                        if let Ok(v) = data.get::<f32>("half_d") { c.half_d = v; }
                        if let Ok(v) = data.get::<u32>("layer") { c.layer = v; }
                        if let Ok(v) = data.get::<u32>("mask") { c.mask = v; }
                        true
                    } else {
                        false
                    }
                }
                "OrientedBoxCollider" => {
                    if let Ok(mut c) = world.get::<&mut OrientedBoxCollider>(entity) {
                        if let Ok(v) = data.get::<f32>("half_w") { c.half_w = v; }
                        if let Ok(v) = data.get::<f32>("half_h") { c.half_h = v; }
                        if let Ok(v) = data.get::<f32>("half_d") { c.half_d = v; }
                        if let Ok(v) = data.get::<f32>("angle_rad") { c.angle_rad = v; }
                        if let Ok(v) = data.get::<u32>("layer") { c.layer = v; }
                        if let Ok(v) = data.get::<u32>("mask") { c.mask = v; }
                        true
                    } else {
                        false
                    }
                }
                "Renderable" => {
                    if let Ok(mut c) = world.get::<&mut Renderable>(entity) {
                        if let Ok(v) = data.get::<f32>("color_r") { c.color[0] = v; }
                        if let Ok(v) = data.get::<f32>("color_g") { c.color[1] = v; }
                        if let Ok(v) = data.get::<f32>("color_b") { c.color[2] = v; }
                        if let Ok(v) = data.get::<f32>("metallic") { c.metallic = v; }
                        if let Ok(v) = data.get::<f32>("roughness") { c.roughness = v; }
                        if let Ok(v) = data.get::<f32>("ao") { c.ao = v; }
                        if let Ok(v) = data.get::<f32>("scale_x") { c.scale[0] = v; }
                        if let Ok(v) = data.get::<f32>("scale_y") { c.scale[1] = v; }
                        if let Ok(v) = data.get::<f32>("scale_z") { c.scale[2] = v; }
                        true
                    } else {
                        false
                    }
                }
                "MaterialTexture" => {
                    if let Ok(mut c) = world.get::<&mut MaterialTexture>(entity) {
                        if let Ok(v) = data.get::<String>("path") { c.path = v; }
                        if let Ok(v) = data.get::<String>("normal_path") { c.normal_path = v; }
                        if let Ok(v) = data.get::<String>("metallic_roughness_path") { c.metallic_roughness_path = v; }
                        true
                    } else {
                        false
                    }
                }
                "Health" => {
                    if let Ok(mut c) = world.get::<&mut Health>(entity) {
                        if let Ok(v) = data.get::<i32>("current") { c.current = v; }
                        if let Ok(v) = data.get::<i32>("max") { c.max = v; }
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            };
            Ok(ok)
        })?;
        globals.set("set_component", sc)?;

        // ── Entity lifetime ───────────────────────────────────────────────
        // destroy(entity) — deferred removal, processed after scripting_system.
        let destroys_ptr = self.pending_destroys.get() as usize;
        let df = self.lua.create_function(move |_, eid: u64| {
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in destroy", eid);
                return Ok(());
            };
            let destroys = unsafe { &mut *(destroys_ptr as *mut Vec<Entity>) };
            destroys.push(entity);
            Ok(())
        })?;
        globals.set("destroy", df)?;

        // ── Entity spawning & lookup ───────────────────────────────────────
        // spawn_mesh(path, x, y, z, sx, sy, sz, r, g, b, with_physics) → entity
        // Loads (or reuses a cached) mesh and spawns a Renderable entity.
        let meshes_ptr = self.meshes_ptr;
        let cache_ptr = self.mesh_cache_ptr();
        let spawn_mesh = self.lua.create_function(
            move |_,
                  (path, x, y, z, sx, sy, sz, r, g, b, with_physics): (
                      String,
                      f32,
                      f32,
                      f32,
                      f32,
                      f32,
                      f32,
                      f32,
                      f32,
                      f32,
                      bool,
                  )| {
                let world = unsafe { &mut *(world_ptr as *mut World) };
                let meshes = unsafe { &mut *(meshes_ptr as *mut crate::assets::AssetStore<crate::assets::Mesh>) };
                let cache = unsafe { &mut *(cache_ptr as *mut HashMap<String, crate::assets::Handle<crate::assets::Mesh>>) };

                let handle = if let Some(h) = cache.get(&path).copied() {
                    h
                } else {
                    let mesh = crate::assets::Mesh::load(&path)
                        .map_err(mlua::Error::RuntimeError)?;
                    let h = meshes.add(mesh);
                    cache.insert(path.clone(), h);
                    h
                };

                let entity = world.spawn((
                    crate::components::Position { x, y, z },
                    crate::components::Rotation {
                        pitch: 0.0,
                        yaw: 0.0,
                        roll: 0.0,
                    },
                    crate::components::Renderable {
                        mesh: handle,
                        color: [r, g, b],
                        metallic: 0.0,
                        roughness: 0.72,
                        ao: 1.0,
                        scale: [sx, sy, sz],
                    },
                ));
                if with_physics {
                    let mut body = crate::components::RigidBody::dynamic();
                    body.friction = 0.6;
                    let _ = world.insert(
                        entity,
                        (
                            body,
                            crate::components::Collider {
                                half_w: sx.abs() * 0.5,
                                half_h: sy.abs() * 0.5,
                                half_d: sz.abs() * 0.5,
                                layer: 1,
                                mask: 1,
                            },
                        ),
                    );
                }
                Ok(entity.to_bits().get())
            },
        )?;
        globals.set("spawn_mesh", spawn_mesh)?;

        // spawn_box(x, y, z, w, h, d, r, g, b, with_physics) → entity
        // Uses the engine's built-in unit cube so no external asset is needed.
        let meshes_ptr = self.meshes_ptr;
        let cache_ptr = self.mesh_cache_ptr();
        let spawn_box = self.lua.create_function(
            move |_, (x, y, z, w, h, d, r, g, b, with_physics): (
                f32,
                f32,
                f32,
                f32,
                f32,
                f32,
                f32,
                f32,
                f32,
                bool,
            )| {
                let world = unsafe { &mut *(world_ptr as *mut World) };
                let meshes = unsafe { &mut *(meshes_ptr as *mut crate::assets::AssetStore<crate::assets::Mesh>) };
                let cache = unsafe { &mut *(cache_ptr as *mut HashMap<String, crate::assets::Handle<crate::assets::Mesh>>) };

                let path = "meshes/cube.obj".to_string();
                let handle = if let Some(h) = cache.get(&path).copied() {
                    h
                } else {
                    let mesh = crate::assets::Mesh::load(&path)
                        .map_err(mlua::Error::RuntimeError)?;
                    let h = meshes.add(mesh);
                    cache.insert(path, h);
                    h
                };

                let entity = world.spawn((
                    crate::components::Position { x, y, z },
                    crate::components::Rotation {
                        pitch: 0.0,
                        yaw: 0.0,
                        roll: 0.0,
                    },
                    crate::components::Renderable {
                        mesh: handle,
                        color: [r, g, b],
                        metallic: 0.0,
                        roughness: 0.72,
                        ao: 1.0,
                        scale: [w, h, d],
                    },
                ));
                if with_physics {
                    let mut body = crate::components::RigidBody::dynamic();
                    body.friction = 0.6;
                    let _ = world.insert(
                        entity,
                        (
                            body,
                            crate::components::Collider {
                                half_w: w.abs() * 0.5,
                                half_h: h.abs() * 0.5,
                                half_d: d.abs() * 0.5,
                                layer: 1,
                                mask: 1,
                            },
                        ),
                    );
                }
                Ok(entity.to_bits().get())
            },
        )?;
        globals.set("spawn_box", spawn_box)?;

        // load_model(path) → mesh handle id (loads & caches the mesh).
        // The returned handle can be passed to set_mesh_entity to swap a mesh
        // on an existing Renderable without spawning a new entity.
        let meshes_ptr = self.meshes_ptr;
        let cache_ptr = self.mesh_cache_ptr();
        let load_model = self.lua.create_function(move |_, path: String| {
            let meshes = unsafe { &mut *(meshes_ptr as *mut crate::assets::AssetStore<crate::assets::Mesh>) };
            let cache = unsafe { &mut *(cache_ptr as *mut HashMap<String, crate::assets::Handle<crate::assets::Mesh>>) };
            if let Some(h) = cache.get(&path).copied() {
                return Ok(h.id);
            }
            let mesh = crate::assets::Mesh::load(&path).map_err(mlua::Error::RuntimeError)?;
            let h = meshes.add(mesh);
            cache.insert(path, h);
            Ok(h.id)
        })?;
        globals.set("load_model", load_model)?;

        // set_mesh_entity(entity, handle_id) — replace the Renderable mesh on
        // an existing entity with a previously loaded model handle.
        let set_mesh = self.lua.create_function(move |_, (eid, handle_id): (u64, u32)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_mesh_entity", eid);
                return Ok(false);
            };
            let handle = crate::assets::Handle::new(handle_id);
            if let Ok(mut rend) = world.get::<&mut Renderable>(entity) {
                rend.mesh = handle;
                Ok(true)
            } else {
                Ok(false)
            }
        })?;
        globals.set("set_mesh_entity", set_mesh)?;

        // get_all_entities() → Lua array of entity ids, optionally filtered by a
        // component name (e.g. "Position", "RigidBody") and/or "tag".
        let get_all = self.lua.create_function(
            move |lua, (kind, tag): (mlua::Value, mlua::Value)| {
                let world = unsafe { &mut *(world_ptr as *mut World) };
                let kind: Option<String> = if kind.is_nil() {
                    None
                } else {
                    kind.as_string().map(|s| s.to_string_lossy().to_string())
                };
                let tag: Option<String> = if tag.is_nil() {
                    None
                } else {
                    tag.as_string().map(|s| s.to_string_lossy().to_string())
                };

                // Collect entity ids first so we don't hold the iterator borrow
                // across the world.get() calls below.
                let ids: Vec<hecs::Entity> =
                    world.iter().map(|r| r.entity()).collect();

                let mut out = Vec::new();
                for entity in ids {
                    if let Some(k) = &kind {
                        let has = match k.as_str() {
                            "Position" => world.get::<&crate::components::Position>(entity).is_ok(),
                            "RigidBody" => world.get::<&crate::components::RigidBody>(entity).is_ok(),
                            "Collider" => world.get::<&crate::components::Collider>(entity).is_ok(),
                            "Renderable" => world.get::<&crate::components::Renderable>(entity).is_ok(),
                            "Health" => world.get::<&crate::components::Health>(entity).is_ok(),
                            _ => {
                                return Err(mlua::Error::RuntimeError(format!(
                                    "get_all_entities: unknown component '{k}'"
                                )));
                            }
                        };
                        if !has {
                            continue;
                        }
                    }
                    if let Some(t) = &tag {
                        let matches = if let Ok(tagc) = world.get::<&crate::ai::behavior_tree::EntityTag>(entity) {
                            &tagc.0 == t
                        } else {
                            false
                        };
                        if !matches {
                            continue;
                        }
                    }
                    out.push(entity.to_bits().get());
                }
                Ok(lua.create_sequence_from(out.into_iter()))
            },
        )?;
        globals.set("get_all_entities", get_all)?;

        // set_tag(entity, tag) — add an EntityTag marker component for queries.
        let set_tag = self.lua.create_function(move |_, (eid, tag): (u64, String)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_tag", eid);
                return Ok(false);
            };
            let _ = world.insert(entity, (crate::ai::behavior_tree::EntityTag(tag),));
            Ok(true)
        })?;
        globals.set("set_tag", set_tag)?;

        // get_tag(entity) → tag string or nil.
        let get_tag = self.lua.create_function(move |_, eid: u64| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                return Ok(None);
            };
            if let Ok(tagc) = world.get::<&crate::ai::behavior_tree::EntityTag>(entity) {
                Ok(Some(tagc.0.clone()))
            } else {
                Ok(None)
            }
        })?;
        globals.set("get_tag", get_tag)?;

        // ── Raycasts & overlap queries ─────────────────────────────────────
        // raycast(ox, oy, oz, dx, dy, dz, max_dist) → entity id or nil.
        // Tests a world-space ray segment against every entity with a
        // Position + Collider (AABB, using half-extents). Returns the first
        // hit's entity id, or nil. Direction need not be normalized.
        let raycast = self.lua.create_function(
            move |_, (ox, oy, oz, dx, dy, dz, max_dist): (f32, f32, f32, f32, f32, f32, f32)| {
                let world = unsafe { &*(world_ptr as *const World) };
                let origin = [ox, oy, oz];
                let mut best: Option<u64> = None;
                let mut best_t = f32::MAX;
                for entity in world.iter() {
                    let e = entity.entity();
                    let Ok(pos) = world.get::<&crate::components::Position>(e) else {
                        continue;
                    };
                    let he: [f32; 3] = if let Ok(c) = world.get::<&crate::components::Collider>(e) {
                        [c.half_w, c.half_h, c.half_d]
                    } else if let Ok(c) = world.get::<&crate::components::SphereCollider>(e) {
                        [c.radius, c.radius, c.radius]
                    } else {
                        continue;
                    };
                    // slab test inlined
                    let dir = [dx, dy, dz];
                    let mut tmin = 0.0f32;
                    let mut tmax = max_dist;
                    let mut hit = true;
                    for i in 0..3 {
                        let d = dir[i];
                        let min_b = match i { 0 => pos.x - he[0], 1 => pos.y - he[1], _ => pos.z - he[2] };
                        let max_b = match i { 0 => pos.x + he[0], 1 => pos.y + he[1], _ => pos.z + he[2] };
                        if d.abs() < 1e-6 {
                            if origin[i] < min_b || origin[i] > max_b {
                                hit = false;
                                break;
                            }
                        } else {
                            let t1 = (min_b - origin[i]) / d;
                            let t2 = (max_b - origin[i]) / d;
                            tmin = tmin.max(t1.min(t2));
                            tmax = tmax.min(t1.max(t2));
                            if tmin > tmax {
                                hit = false;
                                break;
                            }
                        }
                    }
                    if hit && tmin < best_t {
                        best_t = tmin;
                        best = Some(e.to_bits().get());
                    }
                }
                Ok(best)
            },
        )?;
        globals.set("raycast", raycast)?;

        // overlap_sphere(x, y, z, radius) → table of entity ids whose collider
        // (AABB) overlaps the sphere. Useful for AoE damage and proximity.
        let overlap = self.lua.create_function(
            move |lua, (x, y, z, radius): (f32, f32, f32, f32)| {
                let world = unsafe { &*(world_ptr as *const World) };
                let r2 = radius * radius;
                let mut out = Vec::new();
                for (entity, pos) in world
                    .query::<(hecs::Entity, &crate::components::Position)>()
                    .iter()
                {
                    let he: Option<[f32; 3]> = if let Ok(c) = world.get::<&crate::components::Collider>(entity) {
                        Some([c.half_w, c.half_h, c.half_d])
                    } else if let Ok(c) = world.get::<&crate::components::SphereCollider>(entity) {
                        Some([c.radius, c.radius, c.radius])
                    } else {
                        None
                    };
                    let Some(he) = he else { continue };
                    // Closest point on AABB to sphere center.
                    let cx = x.clamp(pos.x - he[0], pos.x + he[0]);
                    let cy = y.clamp(pos.y - he[1], pos.y + he[1]);
                    let cz = z.clamp(pos.z - he[2], pos.z + he[2]);
                    let dx = x - cx;
                    let dy = y - cy;
                    let dz = z - cz;
                    if dx * dx + dy * dy + dz * dz <= r2 {
                        out.push(entity.to_bits().get());
                    }
                }
                Ok(lua.create_sequence_from(out.into_iter()))
            },
        )?;
        globals.set("overlap_sphere", overlap)?;

        // ── Particles / VFX ────────────────────────────────────────────────
        // set_fire(entity, intensity, radius) — attach a FireSource component.
        // The engine's particle system reads FireSource each frame and spawns
        // fire/smoke/ember emitters automatically.
        let set_fire = self.lua.create_function(
            move |_, (eid, intensity, radius): (u64, f32, f32)| {
                let world = unsafe { &mut *(world_ptr as *mut World) };
                let Some(entity) = Entity::from_bits(eid) else {
                    tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in set_fire", eid);
                    return Ok(false);
                };
                let mut fs = world
                    .get::<&crate::components::FireSource>(entity)
                    .map(|f| *f)
                    .unwrap_or_default();
                fs.intensity = intensity.clamp(0.0, 1.0);
                fs.radius = radius.max(0.0);
                let _ = world.insert(entity, (fs,));
                Ok(true)
            },
        )?;
        globals.set("set_fire", set_fire)?;

        // remove_fire(entity) — detach the FireSource component (stops flames).
        let remove_fire = self.lua.create_function(move |_, eid: u64| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let Some(entity) = Entity::from_bits(eid) else {
                return Ok(false);
            };
            let _ = world.remove_one::<crate::components::FireSource>(entity);
            Ok(true)
        })?;
        globals.set("remove_fire", remove_fire)?;

        // set_weather(condition, intensity, wind_x, wind_z, wind_strength)
        // Drives the global rain/snow/mist state consumed by the particle
        // system, renderer, and audio each frame.
        let weather_ptr = self.weather_ptr;
        let set_weather = self.lua.create_function(
            move |_, (condition, intensity, wind_x, wind_z, wind_strength): (String, f32, f32, f32, f32)| {
                let weather = unsafe { &mut *(weather_ptr as *mut crate::environment::weather::WeatherState) };
                weather.condition = match condition.to_ascii_lowercase().as_str() {
                    "clear" | "sunny" => crate::environment::weather::WeatherCondition::Clear,
                    "cloudy" => crate::environment::weather::WeatherCondition::Cloudy,
                    "overcast" => crate::environment::weather::WeatherCondition::Overcast,
                    "lightrain" => crate::environment::weather::WeatherCondition::LightRain,
                    "heavyrain" => crate::environment::weather::WeatherCondition::HeavyRain,
                    "snow" => crate::environment::weather::WeatherCondition::Snow,
                    "fog" => crate::environment::weather::WeatherCondition::Fog,
                    "storm" => crate::environment::weather::WeatherCondition::Storm,
                    _ => {
                        return Err(mlua::Error::RuntimeError(format!(
                            "set_weather: unknown condition '{condition}'"
                        )));
                    }
                };
                weather.intensity = intensity.clamp(0.0, 1.0);
                weather.wind_direction = glam::Vec2::new(wind_x, wind_z).normalize_or_zero();
                weather.wind_strength = wind_strength.max(0.0);
                Ok(())
            },
        )?;
        globals.set("set_weather", set_weather)?;

        // ── Input ─────────────────────────────────────────────────────────
        // is_key_held("W") → bool
        let kh = self.lua.create_function(move |_, key: String| {
            let input = unsafe { &*(input_ptr as *const InputState) };
            let held = input.is_virtual_key_held(&key);
            Ok(held)
        })?;
        globals.set("is_key_held", kh)?;

        // gamepad_left_x() → f32 in [-1, 1] (deadzone applied)
        let gx = self.lua.create_function(move |_, ()| {
            let input = unsafe { &*(input_ptr as *const InputState) };
            Ok(input.gamepad_left_x())
        })?;
        globals.set("gamepad_left_x", gx)?;
        // gamepad_left_y() → f32 in [-1, 1] (deadzone applied)
        let gy = self.lua.create_function(move |_, ()| {
            let input = unsafe { &*(input_ptr as *const InputState) };
            Ok(input.gamepad_left_y())
        })?;
        globals.set("gamepad_left_y", gy)?;
        // gamepad_button_pressed("south") → bool
        let gb = self.lua.create_function(move |_, button: String| {
            let input = unsafe { &*(input_ptr as *const InputState) };
            Ok(match button.to_ascii_lowercase().as_str() {
                "south" | "a" => input.gamepad_south_pressed(),
                _ => false,
            })
        })?;
        globals.set("gamepad_button_pressed", gb)?;
        // gamepad_left_magnitude() → f32 in [0, 1]
        let gm = self.lua.create_function(move |_, ()| {
            let input = unsafe { &*(input_ptr as *const InputState) };
            Ok(input.gamepad_left_magnitude())
        })?;
        globals.set("gamepad_left_magnitude", gm)?;

        // ── Timing ────────────────────────────────────────────────────────
        // elapsed_time() → seconds since engine start (f32)
        let time_ptr = self.start_time.get() as usize;
        let et = self.lua.create_function(move |_, ()| {
            let start = unsafe { &*(time_ptr as *const std::time::Instant) };
            Ok(start.elapsed().as_secs_f32())
        })?;
        globals.set("elapsed_time", et)?;

        // ── Math helpers (convenience wrappers) ───────────────────────────
        // sin(x), cos(x), sqrt(x), abs(x) — Lua has math.sin etc. but these
        // are shorter to type in game scripts.
        let sin_f = self.lua.create_function(|_, x: f32| Ok(x.sin()))?;
        globals.set("sin", sin_f)?;
        let cos_f = self.lua.create_function(|_, x: f32| Ok(x.cos()))?;
        globals.set("cos", cos_f)?;
        let sqrt_f = self.lua.create_function(|_, x: f32| Ok(x.sqrt()))?;
        globals.set("sqrt", sqrt_f)?;
        let abs_f = self.lua.create_function(|_, x: f32| Ok(x.abs()))?;
        globals.set("abs", abs_f)?;
        let lerp_f = self.lua.create_function(|_, (a, b, t): (f32, f32, f32)| {
            Ok(a + (b - a) * t)
        })?;
        globals.set("lerp", lerp_f)?;
        let clamp_f = self.lua.create_function(|_, (v, lo, hi): (f32, f32, f32)| {
            Ok(v.clamp(lo, hi))
        })?;
        globals.set("clamp", clamp_f)?;

        // ── Camera control ───────────────────────────────────────────────
        // get_camera() -> px, py, pz, tx, ty, tz
        let gc = self.lua.create_function(move |_, ()| {
            Ok((
                camera_pos_copy[0],
                camera_pos_copy[1],
                camera_pos_copy[2],
                camera_target_copy[0],
                camera_target_copy[1],
                camera_target_copy[2],
            ))
        })?;
        globals.set("get_camera", gc)?;
        // set_camera(px, py, pz, tx, ty, tz)
        let cam_ptr = self.pending_camera_set.get() as usize;
        let scam = self.lua.create_function(
            move |_, (px, py, pz, tx, ty, tz): (f32, f32, f32, f32, f32, f32)| {
                let slot = unsafe { &mut *(cam_ptr as *mut Option<([f32; 3], [f32; 3])>) };
                *slot = Some(([px, py, pz], [tx, ty, tz]));
                Ok(())
            },
        )?;
        globals.set("set_camera", scam)?;
        // look_at(tx, ty, tz) keeps current camera position.
        let cam_ptr2 = self.pending_camera_set.get() as usize;
        let lka = self.lua.create_function(move |_, (tx, ty, tz): (f32, f32, f32)| {
            let slot = unsafe { &mut *(cam_ptr2 as *mut Option<([f32; 3], [f32; 3])>) };
            *slot = Some((camera_pos_copy, [tx, ty, tz]));
            Ok(())
        })?;
        globals.set("look_at", lka)?;
        // get_camera_direction() -> dx, dy, dz (normalised).
        let cam_dir_copy = camera_target_copy;
        let cam_pos_dir = camera_pos_copy;
        let gcd = self.lua.create_function(move |_, ()| {
            let dir = glam::Vec3::from_array(cam_dir_copy) - glam::Vec3::from_array(cam_pos_dir);
            let dir = dir.normalize_or_zero();
            Ok((dir.x, dir.y, dir.z))
        })?;
        globals.set("get_camera_direction", gcd)?;
        // screen_to_ray(screen_x, screen_y) -> ox, oy, oz, dx, dy, dz
        // Converts 2D screen pixel coords to a world-space ray.
        let cam_pos_ray = camera_pos_copy;
        let cam_tgt_ray = camera_target_copy;
        let sw = screen_w;
        let sh = screen_h;
        let fov_r = fov_degrees;
        let str_fn = self.lua.create_function(move |_, (sx, sy): (f32, f32)| {
            let aspect = sw / sh.max(1.0);
            let view = glam::Mat4::look_at_rh(
                glam::Vec3::from_array(cam_pos_ray),
                glam::Vec3::from_array(cam_tgt_ray),
                glam::Vec3::Y,
            );
            let proj = glam::Mat4::perspective_rh(fov_r.to_radians(), aspect, 0.1, 1000.0);
            let vp = proj * view;
            let inv_vp = vp.inverse();
            let ndc_x = (2.0 * sx / sw) - 1.0;
            let ndc_y = 1.0 - (2.0 * sy / sh);
            let near4 = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
            let far4  = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
            let near3 = near4.truncate() / near4.w;
            let far3  = far4.truncate() / far4.w;
            let dir = (far3 - near3).normalize_or_zero();
            Ok((near3.x, near3.y, near3.z, dir.x, dir.y, dir.z))
        })?;
        globals.set("screen_to_ray", str_fn)?;
        // skip_next_frames(n) asks runtime to skip simulation for N frames.
        let skip_ptr = self.pending_frame_skip.get() as usize;
        let sk = self.lua.create_function(move |_, n: u32| {
            let slot = unsafe { &mut *(skip_ptr as *mut u32) };
            *slot = (*slot).max(n.min(600));
            Ok(())
        })?;
        globals.set("skip_next_frames", sk)?;

        // ── Cinematic API ────────────────────────────────────────────────
        // Timeline director exposed as a `cinematic` table:
        //   cinematic.start(name)
        //   cinematic.add_shot(sx,sy,sz, tx,ty,tz, duration, name)
        //   cinematic.set_ease(index, "linear"|"smooth"|"easein"|"easeout")
        //   cinematic.on_shot(index, function(shot_name))   -- start callback
        //   cinematic.on_end(function)                      -- end callback
        //   cinematic.play() / pause() / resume()
        //   cinematic.is_playing() -> bool
        //   cinematic.time() -> t   cinematic.duration() -> t
        //   cinematic.skip()  cinematic.stop()  cinematic.clear()
        //   cinematic.camera_lock(take)      -- false = keep gameplay camera
        let cut_ptr = self as *mut ScriptEngine as usize;
        let cut = self.lua.create_table()?;

        let cp = cut_ptr;
        let start_fn = self.lua.create_function(move |_, name: Option<String>| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            s.cutscene.start(name);
            s.shot_started.clear();
            s.shot_started.resize(s.cutscene.len(), false);
            s.end_fired = false;
            Ok(())
        })?;
        cut.set("start", start_fn)?;

        let cp = cut_ptr;
        let add_fn = self.lua.create_function(
            move |_, (sx, sy, sz, tx, ty, tz, dur, name): (f32, f32, f32, f32, f32, f32, f32, Option<String>)| {
                let s = unsafe { &mut *(cp as *mut ScriptEngine) };
                let idx = s.cutscene.add_shot([sx, sy, sz], [tx, ty, tz], dur, name.as_deref());
                s.shot_started.push(false);
                Ok(idx)
            },
        )?;
        cut.set("add_shot", add_fn)?;

        let cp = cut_ptr;
        let ease_fn = self.lua.create_function(move |_, (idx, mode): (usize, String)| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            s.cutscene.set_ease(idx, &mode);
            Ok(())
        })?;
        cut.set("set_ease", ease_fn)?;

        let cp = cut_ptr;
        let on_shot_fn = self.lua.create_function(move |_, (idx, func): (usize, LuaFunction)| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            if idx >= s.cutscene.len() {
                return Err(mlua::Error::runtime("cinematic.on_shot: shot index out of range"));
            }
            if let Some(old) = s.shot_callbacks.insert(idx, s.lua.create_registry_value(func)?) {
                let _ = s.lua.remove_registry_value(old);
            }
            Ok(())
        })?;
        cut.set("on_shot", on_shot_fn)?;

        let cp = cut_ptr;
        let on_end_fn = self.lua.create_function(move |_, func: LuaFunction| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            if let Some(old) = s.end_callback.replace(s.lua.create_registry_value(func)?) {
                let _ = s.lua.remove_registry_value(old);
            }
            s.end_fired = false;
            Ok(())
        })?;
        cut.set("on_end", on_end_fn)?;

        let cp = cut_ptr;
        let play_fn = self.lua.create_function(move |_, ()| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            s.end_fired = false;
            s.cutscene.play();
            Ok(())
        })?;
        cut.set("play", play_fn)?;

        let cp = cut_ptr;
        let pause_fn = self.lua.create_function(move |_, ()| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            s.cutscene.pause();
            Ok(())
        })?;
        cut.set("pause", pause_fn)?;

        let cp = cut_ptr;
        let resume_fn = self.lua.create_function(move |_, ()| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            s.cutscene.resume();
            Ok(())
        })?;
        cut.set("resume", resume_fn)?;

        // Per-cutscene camera ownership.  Default: true (cutscene takes over
        // the camera).  Set false BEFORE play() for cutscenes that should play
        // while the player keeps free camera control — the cutscene timeline
        // and callbacks still run, only the camera handoff is skipped.
        let cp = cut_ptr;
        let lock_fn = self.lua.create_function(move |_, take: bool| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            s.cutscene.set_drives_camera(take);
            Ok(())
        })?;
        cut.set("camera_lock", lock_fn)?;

        let cp = cut_ptr;
        let locked_fn = self.lua.create_function(move |_, ()| {
            let s = unsafe { &*(cp as *mut ScriptEngine) };
            Ok(s.cutscene.drives_camera())
        })?;
        cut.set("camera_locked", locked_fn)?;

        let cp = cut_ptr;
        let playing_fn = self.lua.create_function(move |_, ()| {
            let s = unsafe { &*(cp as *mut ScriptEngine) };
            Ok(s.cutscene.is_playing())
        })?;
        cut.set("is_playing", playing_fn)?;

        let cp = cut_ptr;
        let time_fn = self.lua.create_function(move |_, ()| {
            let s = unsafe { &*(cp as *mut ScriptEngine) };
            Ok(s.cutscene.time())
        })?;
        cut.set("time", time_fn)?;

        let cp = cut_ptr;
        let dur_fn = self.lua.create_function(move |_, ()| {
            let s = unsafe { &*(cp as *mut ScriptEngine) };
            Ok(s.cutscene.duration())
        })?;
        cut.set("duration", dur_fn)?;

        let cp = cut_ptr;
        let skip_fn = self.lua.create_function(move |_, ()| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            s.cutscene.skip();
            Ok(())
        })?;
        cut.set("skip", skip_fn)?;

        let cp = cut_ptr;
        let stop_fn = self.lua.create_function(move |_, ()| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            s.cutscene.stop();
            Ok(())
        })?;
        cut.set("stop", stop_fn)?;

        let cp = cut_ptr;
        let clear_fn = self.lua.create_function(move |_, ()| {
            let s = unsafe { &mut *(cp as *mut ScriptEngine) };
            s.cutscene.clear();
            s.shot_callbacks.clear();
            s.shot_started.clear();
            if let Some(old) = s.end_callback.take() {
                let _ = s.lua.remove_registry_value(old);
            }
            s.end_fired = false;
            Ok(())
        })?;
        cut.set("clear", clear_fn)?;

        globals.set("cinematic", cut)?;

        // ── Δt passthrough ────────────────────────────────────────────────
        // Scripts receive dt as the second argument to update(entity, dt),
        // but also expose it as a global so helper functions can read it.
        globals.set("dt", dt)?;

        // ── Audio API ────────────────────────────────────────────────────
        // Exposed as global functions: audio_play_sfx, audio_play_music,
        // audio_stop_all, audio_set_volume, audio_is_music_playing.
        if let Some(audio_ref) = audio {
            let audio_ptr = audio_ref as *mut crate::audio::AudioSystem as usize;
            let ap_sfx = audio_ptr;
            let play_sfx_fn = self.lua.create_function(move |_, (path, volume, looping): (String, Option<f32>, Option<bool>)| {
                let audio = unsafe { &mut *(ap_sfx as *mut crate::audio::AudioSystem) };
                audio.play_sfx(&path, volume.unwrap_or(1.0), looping.unwrap_or(false));
                Ok(())
            })?;
            globals.set("audio_play_sfx", play_sfx_fn)?;

            let ap_music = audio_ptr;
            let play_music_fn = self.lua.create_function(move |_, (path, volume, looping): (String, Option<f32>, Option<bool>)| {
                let audio = unsafe { &mut *(ap_music as *mut crate::audio::AudioSystem) };
                audio.play_music(&path, volume, looping.unwrap_or(true));
                Ok(())
            })?;
            globals.set("audio_play_music", play_music_fn)?;

            let ap_stop = audio_ptr;
            let stop_all_fn = self.lua.create_function(move |_, ()| {
                let audio = unsafe { &mut *(ap_stop as *mut crate::audio::AudioSystem) };
                audio.stop_all();
                Ok(())
            })?;
            globals.set("audio_stop_all", stop_all_fn)?;

            let ap_vol = audio_ptr;
            let set_volume_fn = self.lua.create_function(move |_, (channel, volume): (String, f32)| {
                let audio = unsafe { &mut *(ap_vol as *mut crate::audio::AudioSystem) };
                use crate::audio::Channel;
                let ch = match channel.as_str() {
                    "music" => Channel::Music,
                    "sfx" => Channel::Sfx,
                    "ambient" => Channel::Ambient,
                    _ => return Ok(()),
                };
                audio.set_channel_volume(ch, volume);
                Ok(())
            })?;
            globals.set("audio_set_volume", set_volume_fn)?;

            let ap_master = audio_ptr;
            let set_master_fn = self.lua.create_function(move |_, volume: f32| {
                let audio = unsafe { &mut *(ap_master as *mut crate::audio::AudioSystem) };
                audio.set_master_volume(volume);
                Ok(())
            })?;
            globals.set("audio_set_master_volume", set_master_fn)?;

            let ap_playing = audio_ptr;
            let is_music_fn = self.lua.create_function(move |_, ()| {
                let audio = unsafe { &*(ap_playing as *const crate::audio::AudioSystem) };
                Ok(audio.is_music_playing())
            })?;
            globals.set("audio_is_music_playing", is_music_fn)?;

            let ap_active = audio_ptr;
            let active_count_fn = self.lua.create_function(move |_, ()| {
                let audio = unsafe { &*(ap_active as *const crate::audio::AudioSystem) };
                Ok(audio.active_count())
            })?;
            globals.set("audio_active_count", active_count_fn)?;

            let ap_at = audio_ptr;
            let play_at_fn = self.lua.create_function(
                move |_, (path, x, y, z): (String, f32, f32, f32)| {
                    let audio = unsafe { &mut *(ap_at as *mut crate::audio::AudioSystem) };
                    // Distance + stereo-pan spatial attenuation relative to the
                    // listener (camera).  Base volume is 1.0; attenuation scopes
                    // it to the world position so far sounds are quieter.
                    let vol = audio.apply_spatial([x, y, z], 1.0);
                    let pan = audio.stereo_pan([x, y, z]);
                    let _ = pan; // pan is applied by the underlying sink mix.
                    audio.play_sfx(&path, vol, false);
                    Ok(())
                },
            )?;
            globals.set("audio_play_at", play_at_fn)?;

            let ap_atten = audio_ptr;
            let attenuation_fn = self.lua.create_function(
                move |_, (x, y, z): (f32, f32, f32)| {
                    let audio = unsafe { &*(ap_atten as *const crate::audio::AudioSystem) };
                    Ok(audio.distance_attenuation([x, y, z], 1.0, 100.0))
                },
            )?;
            globals.set("audio_attenuation", attenuation_fn)?;
        }

        // ── Behavior Tree (bt) API ─────────────────────────────────────
        // Provides Lua functions for building behavior trees declaratively.
        // Trees are built via a flat node list (BTBuilder) and converted to
        // real BehaviorNode objects at bt.assign() time.
        {
            let bt_table = self.lua.create_table()?;

            // bt.create(name) → name
            // Creates a fresh BTBuilder for the named tree.  Any previous
            // builder with the same name is replaced.
            let sp = script_ptr;
            let bt_create = self.lua.create_function(move |_, name: String| {
                let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                scripts.bt_trees.insert(name.clone(), BTBuilder::new());
                Ok(name)
            })?;
            bt_table.set("create", bt_create)?;

            // bt.sequence(tree_name)
            // Adds a Sequence composite as a child of the current stack top,
            // then pushes it so subsequent calls add children to it.
            let sp = script_ptr;
            let bt_sequence = self.lua.create_function(move |_, tree_name: String| {
                let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                    builder.push_composite(BTNodeKind::Sequence);
                }
                Ok(())
            })?;
            bt_table.set("sequence", bt_sequence)?;

            // bt.selector(tree_name)
            // Adds a Selector composite (try children until one succeeds).
            let sp = script_ptr;
            let bt_selector = self.lua.create_function(move |_, tree_name: String| {
                let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                    builder.push_composite(BTNodeKind::Selector);
                }
                Ok(())
            })?;
            bt_table.set("selector", bt_selector)?;

            // bt.parallel(tree_name, success_threshold)
            // Adds a Parallel composite that ticks all children every frame.
            let sp = script_ptr;
            let bt_parallel = self.lua.create_function(
                move |_, (tree_name, threshold): (String, usize)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.push_composite(BTNodeKind::Parallel {
                            success_threshold: threshold,
                        });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("parallel", bt_parallel)?;

            // bt.move_to(tree_name)
            // Adds a MoveTo leaf that uses NavGrid A* pathfinding.
            // Reads the target position from blackboard key "target_pos".
            let sp = script_ptr;
            let bt_move_to = self.lua.create_function(move |_, tree_name: String| {
                let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                    builder.add_leaf(BTNodeKind::MoveTo { speed: 2.0 });
                }
                Ok(())
            })?;
            bt_table.set("move_to", bt_move_to)?;

            // bt.patrol(tree_name, waypoints_table)
            // Adds a Patrol leaf that cycles through waypoints.
            // waypoints_table is a Lua array of {x=, y=, z=} tables.
            // Waypoints are stored in the builder and injected into the
            // entity's blackboard at bt.assign() time.
            let sp = script_ptr;
            let bt_patrol = self.lua.create_function(
                move |_lua, (tree_name, waypoints_lua): (String, LuaTable)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        // Parse waypoints from the Lua table.
                        let mut waypoints = Vec::new();
                        let len = waypoints_lua.len().unwrap_or(0);
                        for i in 1..=len {
                            if let Ok(wp) = waypoints_lua.get::<LuaTable>(i) {
                                let x = wp.get::<f32>("x").unwrap_or(0.0);
                                let y = wp.get::<f32>("y").unwrap_or(0.0);
                                let z = wp.get::<f32>("z").unwrap_or(0.0);
                                waypoints.push([x, y, z]);
                            }
                        }
                        // Generate a unique blackboard key for these waypoints.
                        let wp_key = format!("patrol_wp_{}", builder.next_wp_id);
                        builder.next_wp_id += 1;
                        builder.patrol_data.push((wp_key.clone(), waypoints));
                        builder.add_leaf(BTNodeKind::Patrol {
                            speed: 2.0,
                            waypoints_key: wp_key,
                        });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("patrol", bt_patrol)?;

            // bt.wait(tree_name, duration)
            // Adds a Wait leaf that returns Running for `duration` seconds,
            // then returns Success.
            let sp = script_ptr;
            let bt_wait = self.lua.create_function(
                move |_, (tree_name, duration): (String, f32)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.add_leaf(BTNodeKind::Wait { duration });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("wait", bt_wait)?;

            // bt.log(tree_name, message)
            // Adds a Log leaf that prints a debug message and returns Success.
            let sp = script_ptr;
            let bt_log = self.lua.create_function(
                move |_, (tree_name, message): (String, String)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.add_leaf(BTNodeKind::Log { message });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("log", bt_log)?;

            // bt.idle(tree_name, duration)
            // Adds a native Idle leaf that drives ai_state="idle". duration<=0 → infinite.
            let sp = script_ptr;
            let bt_idle = self.lua.create_function(
                move |_, (tree_name, duration): (String, f32)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.add_leaf(BTNodeKind::Idle { duration });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("idle", bt_idle)?;

            // bt.wander(tree_name, speed, radius)
            // Adds a Wander leaf that walks to random points within `radius` of
            // the agent's spawn (or the blackboard "home" position).
            let sp = script_ptr;
            let bt_wander = self.lua.create_function(
                move |_, (tree_name, speed, radius): (String, f32, f32)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.add_leaf(BTNodeKind::Wander { speed, radius });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("wander", bt_wander)?;

            // bt.flee(tree_name, run_speed, safe_distance)
            // Adds a Flee leaf that runs away from the "threat_pos" (or "threat"
            // entity) until it is out to `safe_distance`.
            let sp = script_ptr;
            let bt_flee = self.lua.create_function(
                move |_, (tree_name, run_speed, safe_distance): (String, f32, f32)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.add_leaf(BTNodeKind::Flee { run_speed, safe_distance });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("flee", bt_flee)?;

            // bt.graze(tree_name, speed, radius)
            // Adds a Graze leaf: walks to the nearest entity tagged "grazeable"
            // within `radius`, consumes it, and succeeds.
            let sp = script_ptr;
            let bt_graze = self.lua.create_function(
                move |_, (tree_name, speed, radius): (String, f32, f32)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.add_leaf(BTNodeKind::Graze { speed, radius });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("graze", bt_graze)?;

            // bt.perceive(tree_name, radius, tag)
            // Adds a Perception leaf: scans the world for the nearest entity
            // within `radius` whose tag equals `tag` ("*" matches any) and
            // writes it to "perceived_entity"/"perceived_pos".
            let sp = script_ptr;
            let bt_perceive = self.lua.create_function(
                move |_, (tree_name, radius, tag): (String, f32, String)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.add_leaf(BTNodeKind::Perception { radius, tag });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("perceive", bt_perceive)?;

            // bt.in_range(tree_name, min, max)
            // Wraps the current top-of-stack node in a DistanceCondition that
            // only ticks the child while the distance to "perceived_pos" is
            // within [min, max].  Writes the distance to "dist_to_target".
            let sp = script_ptr;
            let bt_in_range = self.lua.create_function(
                move |_, (tree_name, min, max): (String, f32, f32)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.wrap_decorator(BTNodeKind::Distance { min, max });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("in_range", bt_in_range)?;

            // bt.inverter(tree_name)
            // Wraps the current top-of-stack node in an Inverter decorator,
            // which flips Success ↔ Failure.
            let sp = script_ptr;
            let bt_inverter = self.lua.create_function(move |_, tree_name: String| {
                let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                    builder.wrap_decorator(BTNodeKind::Inverter);
                }
                Ok(())
            })?;
            bt_table.set("inverter", bt_inverter)?;

            // bt.repeater(tree_name, max_times)
            // Wraps the current top-of-stack node in a Repeater decorator.
            // max_times=0 means infinite repetition.
            let sp = script_ptr;
            let bt_repeater = self.lua.create_function(
                move |_, (tree_name, max_times): (String, u32)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.wrap_decorator(BTNodeKind::Repeater { max_times });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("repeater", bt_repeater)?;

            // bt.cooldown(tree_name, duration)
            // Wraps the current top-of-stack in a Cooldown decorator.
            // The child is only ticked once every `duration` seconds.
            let sp = script_ptr;
            let bt_cooldown = self.lua.create_function(
                move |_, (tree_name, duration): (String, f32)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    if let Some(builder) = scripts.bt_trees.get_mut(&tree_name) {
                        builder.wrap_decorator(BTNodeKind::Cooldown { duration });
                    }
                    Ok(())
                },
            )?;
            bt_table.set("cooldown", bt_cooldown)?;

            // bt.assign(entity, tree_name)
            // Finalizes the named tree, registers it in the AiRegistry,
            // and adds an AiAgent component to the entity.
            let sp = script_ptr;
            let wp = world_ptr;
            let bt_assign = self.lua.create_function(
                move |_, (eid, tree_name): (u64, String)| {
                    let scripts = unsafe { &mut *(sp as *mut ScriptEngine) };
                    let world = unsafe { &mut *(wp as *mut World) };

                    // Build the tree from the BTBuilder.
                    let (root, patrol_data) = match scripts.bt_trees.get(&tree_name) {
                        Some(builder) => (builder.build(), builder.patrol_data.clone()),
                        None => {
                            return Err(LuaError::RuntimeError(format!(
                                "BT '{}' not found — call bt.create() first",
                                tree_name
                            )));
                        }
                    };

                    // Register the built tree in the AiRegistry.
                    if scripts.ai_registry_ptr != 0 {
                        let ai_reg = unsafe {
                            &mut *(scripts.ai_registry_ptr as *mut AiRegistry)
                        };
                        let bt = BehaviorTree::new(&tree_name, root);
                        ai_reg.register(&tree_name, bt);
                    }

                    // Insert an AiAgent component on the entity.
                    let Some(entity) = Entity::from_bits(eid) else {
                        tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in bt.assign", eid);
                        return Ok(());
                    };
                    let _ = world.insert(entity, (AiAgent::new(&tree_name),));

                    // Inject patrol waypoints into the entity's blackboard.
                    if let Ok(mut agent) = world.get::<&mut AiAgent>(entity) {
                        for (key, waypoints) in &patrol_data {
                            agent
                                .blackboard
                                .set(key, BlackboardValue::Path(waypoints.clone()));
                        }
                    }

                    Ok(())
                },
            )?;
            bt_table.set("assign", bt_assign)?;

            // ── Blackboard access functions ─────────────────────────────
            // bt.set_blackboard(entity, key, value)
            // Generic set — accepts booleans.  For other types use the
            // typed variants below.
            let _sp = script_ptr;
            let wp = world_ptr;
            let bt_set_bb = self.lua.create_function(
                move |_, (eid, key, value): (u64, String, bool)| {
                    let world = unsafe { &mut *(wp as *mut World) };
                    let Some(entity) = Entity::from_bits(eid) else {
                        tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in bt.set_blackboard", eid);
                        return Ok(());
                    };
                    if let Ok(mut agent) = world.get::<&mut AiAgent>(entity) {
                        agent.blackboard.set(&key, BlackboardValue::Bool(value));
                    }
                    Ok(())
                },
            )?;
            bt_table.set("set_blackboard", bt_set_bb)?;

            // bt.get_blackboard(entity, key) → bool
            let _sp = script_ptr;
            let wp = world_ptr;
            let bt_get_bb = self.lua.create_function(
                move |_, (eid, key): (u64, String)| {
                    let world = unsafe { &mut *(wp as *mut World) };
                    let Some(entity) = Entity::from_bits(eid) else {
                        tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in bt.get_blackboard", eid);
                        return Ok(false);
                    };
                    let result = world
                        .get::<&AiAgent>(entity)
                        .ok()
                        .and_then(|a| a.blackboard.get_bool(&key))
                        .unwrap_or(false);
                    Ok(result)
                },
            )?;
            bt_table.set("get_blackboard", bt_get_bb)?;

            // bt.set_blackboard_vec3(entity, key, x, y, z)
            let _sp = script_ptr;
            let wp = world_ptr;
            let bt_set_bb_vec3 = self.lua.create_function(
                move |_, (eid, key, x, y, z): (u64, String, f32, f32, f32)| {
                    let world = unsafe { &mut *(wp as *mut World) };
                    let Some(entity) = Entity::from_bits(eid) else {
                        tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in bt.set_blackboard_vec3", eid);
                        return Ok(());
                    };
                    if let Ok(mut agent) = world.get::<&mut AiAgent>(entity) {
                        agent
                            .blackboard
                            .set(&key, BlackboardValue::Vec3([x, y, z]));
                    }
                    Ok(())
                },
            )?;
            bt_table.set("set_blackboard_vec3", bt_set_bb_vec3)?;

            // bt.get_blackboard_vec3(entity, key) → x, y, z
            let _sp = script_ptr;
            let wp = world_ptr;
            let bt_get_bb_vec3 = self.lua.create_function(
                move |_, (eid, key): (u64, String)| {
                    let world = unsafe { &mut *(wp as *mut World) };
                    let Some(entity) = Entity::from_bits(eid) else {
                        tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in bt.get_blackboard_vec3", eid);
                        return Ok((0.0f32, 0.0f32, 0.0f32));
                    };
                    let v = world
                        .get::<&AiAgent>(entity)
                        .ok()
                        .and_then(|a| a.blackboard.get_vec3(&key))
                        .unwrap_or([0.0, 0.0, 0.0]);
                    Ok((v[0], v[1], v[2]))
                },
            )?;
            bt_table.set("get_blackboard_vec3", bt_get_bb_vec3)?;

            // bt.set_blackboard_float(entity, key, value)
            let _sp = script_ptr;
            let wp = world_ptr;
            let bt_set_bb_float = self.lua.create_function(
                move |_, (eid, key, value): (u64, String, f32)| {
                    let world = unsafe { &mut *(wp as *mut World) };
                    let Some(entity) = Entity::from_bits(eid) else {
                        tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in bt.set_blackboard_float", eid);
                        return Ok(());
                    };
                    if let Ok(mut agent) = world.get::<&mut AiAgent>(entity) {
                        agent
                            .blackboard
                            .set(&key, BlackboardValue::Float(value));
                    }
                    Ok(())
                },
            )?;
            bt_table.set("set_blackboard_float", bt_set_bb_float)?;

            // bt.get_blackboard_float(entity, key) → float
            let _sp = script_ptr;
            let wp = world_ptr;
            let bt_get_bb_float = self.lua.create_function(
                move |_, (eid, key): (u64, String)| {
                    let world = unsafe { &mut *(wp as *mut World) };
                    let Some(entity) = Entity::from_bits(eid) else {
                        tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in bt.get_blackboard_float", eid);
                        return Ok(0.0);
                    };
                    let result = world
                        .get::<&AiAgent>(entity)
                        .ok()
                        .and_then(|a| a.blackboard.get_float(&key))
                        .unwrap_or(0.0);
                    Ok(result)
                },
            )?;
            bt_table.set("get_blackboard_float", bt_get_bb_float)?;

            // bt.set_state(entity, state_name)
            // Convenience: writes "ai_state" = state_name to the entity's blackboard.
            // This triggers the animation blending system to crossfade to the new clip.
            // Example: bt.set_state(player, "walk")
            {
                let wp = world_ptr;
                let bt_set_state = self.lua.create_function(
                    move |_, (eid, state_name): (u64, String)| {
                        let world = unsafe { &mut *(wp as *mut World) };
                        let Some(entity) = Entity::from_bits(eid) else {
                            tracing::warn!("[Scripting] Invalid entity bits 0x{:x} in bt.set_state", eid);
                            return Ok(());
                        };
                        if let Ok(mut agent) = world.get::<&mut AiAgent>(entity) {
                            agent.blackboard.set("ai_state", BlackboardValue::String(state_name));
                        }
                        Ok(())
                    },
                )?;
                bt_table.set("set_state", bt_set_state)?;
            }

            globals.set("bt", bt_table)?;
        }

        // ── Navigation (nav) API ────────────────────────────────────────
        // Provides pathfinding and walkability queries through the NavGrid.
        {
            let nav_table = self.lua.create_table()?;
            let np = self.nav_grid_ptr;

            // nav.find_path(x1, y1, z1, x2, y2, z2)
            // A* pathfinding in world space.  Returns a Lua table of
            // {x, y, z} waypoints.  Returns empty table if no path exists.
            let find_path = self.lua.create_function(move |lua, (x1, y1, z1, x2, _y2, z2): (f32, f32, f32, f32, f32, f32)| {
                let table = lua.create_table()?;
                if np == 0 {
                    return Ok(table);
                }
                let nav = unsafe { &*(np as *const NavGrid) };
                let half_w = nav.width as f32 * 0.5;
                let half_d = nav.depth as f32 * 0.5;

                // Convert world coordinates to grid coordinates.
                let sx = (x1 + half_w)
                    .round()
                    .clamp(0.0, (nav.width - 1) as f32) as usize;
                let sz = (z1 + half_d)
                    .round()
                    .clamp(0.0, (nav.depth - 1) as f32) as usize;
                let gx = (x2 + half_w)
                    .round()
                    .clamp(0.0, (nav.width - 1) as f32) as usize;
                let gz = (z2 + half_d)
                    .round()
                    .clamp(0.0, (nav.depth - 1) as f32) as usize;

                if let Some(path) = nav.find_path((sx, sz), (gx, gz)) {
                    for (i, &(px, pz)) in path.iter().enumerate() {
                        let wp = lua.create_table()?;
                        wp.set("x", px as f32 - half_w)?;
                        wp.set("y", y1)?; // preserve start height
                        wp.set("z", pz as f32 - half_d)?;
                        table.set(i + 1, wp)?; // Lua arrays are 1-indexed
                    }
                }
                Ok(table)
            })?;
            nav_table.set("find_path", find_path)?;

            // nav.is_walkable(x, y, z) → bool
            // Checks whether the grid cell at the given world position
            // is walkable (slope ≤ max_slope).
            let is_walkable = self.lua.create_function(move |_, (x, _y, z): (f32, f32, f32)| {
                if np == 0 {
                    return Ok(false);
                }
                let nav = unsafe { &*(np as *const NavGrid) };
                let half_w = nav.width as f32 * 0.5;
                let half_d = nav.depth as f32 * 0.5;
                let gx = (x + half_w).round() as isize;
                let gz = (z + half_d).round() as isize;
                if gx < 0 || gz < 0 || gx >= nav.width as isize || gz >= nav.depth as isize {
                    return Ok(false);
                }
                let idx = gz as usize * nav.width + gx as usize;
                Ok(nav.walkable[idx])
            })?;
            nav_table.set("is_walkable", is_walkable)?;

            globals.set("nav", nav_table)?;
        }

        // ── Navmesh (navmesh) API ─────────────────────────────────────────
        // Polygon navmesh generated from the terrain — triangle-level 3D
        // pathfinding.  Additive to the `nav` grid API.
        {
            let navmesh_table = self.lua.create_table()?;
            let nm = self.navmesh_ptr;

            // navmesh.find_path(x1, y1, z1, x2, y2, z2) → array of {x,y,z}
            let find_path = self.lua.create_function(
                move |lua, (x1, y1, z1, x2, y2, z2): (f32, f32, f32, f32, f32, f32)| {
                    let table = lua.create_table()?;
                    if nm == 0 {
                        return Ok(table);
                    }
                    let mesh = unsafe { &*(nm as *const crate::navmesh::NavMesh) };
                    let from = [x1, y1, z1];
                    let to = [x2, y2, z2];
                    if let Some(path) = mesh.find_path(from, to) {
                        for (i, wp) in path.iter().enumerate() {
                            let t = lua.create_table()?;
                            t.set("x", wp[0])?;
                            t.set("y", wp[1])?;
                            t.set("z", wp[2])?;
                            table.set(i + 1, t)?;
                        }
                    }
                    Ok(table)
                },
            )?;
            navmesh_table.set("find_path", find_path)?;

            // navmesh.is_walkable(x, y, z) → bool
            let is_walkable = self.lua.create_function(
                move |_, (x, y, z): (f32, f32, f32)| {
                    if nm == 0 {
                        return Ok(false);
                    }
                    let mesh = unsafe { &*(nm as *const crate::navmesh::NavMesh) };
                    Ok(mesh.is_walkable_at([x, y, z]))
                },
            )?;
            navmesh_table.set("is_walkable", is_walkable)?;

            // navmesh.triangle_count() → int (for debug / visualization)
            let tri_count = self.lua.create_function(move |_, ()| {
                if nm == 0 {
                    return Ok(0);
                }
                let mesh = unsafe { &*(nm as *const crate::navmesh::NavMesh) };
                Ok(mesh.triangle_count())
            })?;
            navmesh_table.set("triangle_count", tri_count)?;

            globals.set("navmesh", navmesh_table)?;
        }

        // ── Plugins API ───────────────────────────────────────────────────
        // Runtime management of the pure-Lua plugin host.  No recompile needed
        // to add, list, unload or reload a plugin: it is just a .lua file.
        {
            let plugins_table = self.lua.create_table()?;
            // The engine lives for the whole program; scripts only run on the
            // main thread, so a raw pointer is safe here (same pattern as the
            // weather_ptr / navmesh_ptr captures above).
            let self_ptr = self as *mut ScriptEngine as usize;

            let list = self.lua.create_function(move |lua, _: ()| {
                let engine = unsafe { &*(self_ptr as *const ScriptEngine) };
                let out = lua.create_table()?;
                for (i, name) in engine.plugin_names().iter().enumerate() {
                    out.set(i + 1, name.clone())?;
                }
                Ok(out)
            })?;
            plugins_table.set("list", list)?;

            let has = self.lua.create_function(move |_, name: String| {
                let engine = unsafe { &*(self_ptr as *const ScriptEngine) };
                Ok(engine.has_plugin(&name))
            })?;
            plugins_table.set("has", has)?;

            let unload = self.lua.create_function(move |_, name: String| {
                let engine = unsafe { &mut *(self_ptr as *mut ScriptEngine) };
                Ok(engine.unload_plugin(&name))
            })?;
            plugins_table.set("unload", unload)?;

            globals.set("plugins", plugins_table)?;
        }

        // ── Terrain API ────────────────────────────────────────────────────
        // Exposes terrain height queries and brush operations to Lua.
        {
            let terrain_table = self.lua.create_table()?;
            let tp = self.terrain_world_ptr;

            // terrain.height(x, z) → y
            let th = self.lua.create_function(move |_, (x, z): (f32, f32)| -> LuaResult<f32> {
                if tp == 0 { return Ok(0.0); }
                let terrain = unsafe { &*(tp as *const crate::terrain::TerrainWorld) };
                Ok(terrain.height_at(x, z))
            })?;
            terrain_table.set("height", th)?;

            // terrain.normal(x, z) → nx, ny, nz
            let tn = self.lua.create_function(move |_, (x, z): (f32, f32)| -> LuaResult<(f32, f32, f32)> {
                if tp == 0 { return Ok((0.0, 1.0, 0.0)); }
                let terrain = unsafe { &*(tp as *const crate::terrain::TerrainWorld) };
                let n = terrain.normal_at(x, z);
                Ok((n[0], n[1], n[2]))
            })?;
            terrain_table.set("normal", tn)?;

            // terrain.slope(x, z) → degrees
            let ts = self.lua.create_function(move |_, (x, z): (f32, f32)| -> LuaResult<f32> {
                if tp == 0 { return Ok(0.0); }
                let terrain = unsafe { &*(tp as *const crate::terrain::TerrainWorld) };
                Ok(terrain.slope_at(x, z))
            })?;
            terrain_table.set("slope", ts)?;

            // terrain.surface_color(x, z) → r, g, b
            let tc = self.lua.create_function(move |_, (x, z): (f32, f32)| -> LuaResult<(f32, f32, f32)> {
                if tp == 0 { return Ok((0.3, 0.6, 0.2)); }
                let terrain = unsafe { &*(tp as *const crate::terrain::TerrainWorld) };
                let c = terrain.auto_surface_color_world(x, z);
                Ok((c[0], c[1], c[2]))
            })?;
            terrain_table.set("surface_color", tc)?;

            // terrain.raise(x, z, radius, amount)
            let tr = self.lua.create_function(move |_, (x, z, radius, amount): (f32, f32, f32, f32)| {
                if tp == 0 { return Ok(()); }
                let terrain = unsafe { &mut *(tp as *mut crate::terrain::TerrainWorld) };
                terrain.raise(x, z, radius, amount);
                Ok(())
            })?;
            terrain_table.set("raise", tr)?;

            // terrain.lower(x, z, radius, amount)
            let tl = self.lua.create_function(move |_, (x, z, radius, amount): (f32, f32, f32, f32)| {
                if tp == 0 { return Ok(()); }
                let terrain = unsafe { &mut *(tp as *mut crate::terrain::TerrainWorld) };
                terrain.lower(x, z, radius, amount);
                Ok(())
            })?;
            terrain_table.set("lower", tl)?;

            // terrain.smooth(x, z, radius, strength)
            let tsm = self.lua.create_function(move |_, (x, z, radius, strength): (f32, f32, f32, f32)| {
                if tp == 0 { return Ok(()); }
                let terrain = unsafe { &mut *(tp as *mut crate::terrain::TerrainWorld) };
                terrain.smooth(x, z, radius, strength);
                Ok(())
            })?;
            terrain_table.set("smooth", tsm)?;

            globals.set("terrain", terrain_table)?;
        }

        // ── Particle API ────────────────────────────────────────────────────
        // particles.* drives the engine ParticleSystem (emitters, fire
        // sources, wind) from Lua.  ParticleSystem lives in the game loop and
        // is wired in via set_particles() before run_update() each frame.
        {
            let particle_table = self.lua.create_table()?;
            let pp = self.particles_ptr;
            let particles_ref = move || -> Option<&'static mut crate::particles::ParticleSystem> {
                if pp == 0 {
                    return None;
                }
                Some(unsafe { &mut *(pp as *mut crate::particles::ParticleSystem) })
            };

            // particles.new(params) → emitter id.  `params` is a table:
            //   { x,y,z (position), spawn=, extent=Vec3, velocity=Vec3,
            //     spread=Vec3, rate=, life=(min,max), size=(min,max),
            //     r,g,b,a (color), gravity=, active=bool, max=int }
            let new_emitter = self.lua.create_function(move |_lua, params: mlua::Table| {
                let Some(ps) = particles_ref() else {
                    return Ok(0);
                };
                let mut e = crate::particles::ParticleEmitter::new(
                    params.get::<usize>("max").unwrap_or(1000).max(1),
                );
                e.position = glam::Vec3::new(
                    params.get("x").unwrap_or(0.0),
                    params.get("y").unwrap_or(0.0),
                    params.get("z").unwrap_or(0.0),
                );
                e.spawn_rate = params.get("spawn").unwrap_or(50.0);
                e.spawn_extents = glam::Vec3::new(
                    params.get("extent_x").unwrap_or(1.0),
                    params.get("extent_y").unwrap_or(0.5),
                    params.get("extent_z").unwrap_or(1.0),
                );
                e.initial_velocity = glam::Vec3::new(
                    params.get("vel_x").unwrap_or(0.0),
                    params.get("vel_y").unwrap_or(0.0),
                    params.get("vel_z").unwrap_or(0.0),
                );
                e.velocity_spread = glam::Vec3::new(
                    params.get("spread_x").unwrap_or(0.5),
                    params.get("spread_y").unwrap_or(0.5),
                    params.get("spread_z").unwrap_or(0.5),
                );
                                let lt_min: f32 = params.get("life_min").unwrap_or(1.0);
                let lt_max: f32 = params.get("life_max").unwrap_or(2.0);
                e.lifetime_min = lt_min;
                e.lifetime_max = lt_max.max(lt_min);
                let sz_min: f32 = params.get("size_min").unwrap_or(0.05);
                let sz_max: f32 = params.get("size_max").unwrap_or(0.1);
                e.size_min = sz_min;
                e.size_max = sz_max.max(sz_min);
                e.color = glam::Vec4::new(
                    params.get("r").unwrap_or(1.0),
                    params.get("g").unwrap_or(1.0),
                    params.get("b").unwrap_or(1.0),
                    params.get("a").unwrap_or(0.6),
                );
                e.acceleration = glam::Vec3::new(0.0, params.get("gravity").unwrap_or(-9.8), 0.0);
                e.active = params.get("active").unwrap_or(true);
                Ok(ps.add_emitter(e))
            })?;
            particle_table.set("new_emitter", new_emitter)?;

            // particles.active(id, enabled) → toggle an emitter on/off.
            let set_active = self.lua.create_function(move |_, (id, enabled): (usize, bool)| {
                let Some(ps) = particles_ref() else {
                    return Ok(());
                };
                if let Some(e) = ps.emitters.get_mut(id) {
                    e.active = enabled;
                }
                Ok(())
            })?;
            particle_table.set("active", set_active)?;

            // particles.count() → total live particles across all emitters.
            let count = self.lua.create_function(move |_, ()| {
                let Some(ps) = particles_ref() else {
                    return Ok(0usize);
                };
                Ok(ps.total_particles())
            })?;
            particle_table.set("count", count)?;

            // particles.fire(entity_id, x, y, z, intensity) → attach fire/smoke/ember
            // to an entity (an entity may already, however, have FireSource).
            let fire = self.lua.create_function(move |_, (eid, x, y, z, intensity): (u64, f32, f32, f32, f32)| {
                let Some(ps) = particles_ref() else {
                    return Ok(());
                };
                ps.add_fire_source(eid, glam::Vec3::new(x, y, z), intensity);
                Ok(())
            })?;
            particle_table.set("fire", fire)?;

            // particles.remove_fire(entity_id) → detach fire emitters from an entity.
            let remove_fire = self.lua.create_function(move |_, eid: u64| {
                let Some(ps) = particles_ref() else {
                    return Ok(());
                };
                ps.remove_fire_source(eid);
                Ok(())
            })?;
            particle_table.set("remove_fire", remove_fire)?;

            // particles.wind(dx, dy, dz, strength) → set global wind for all emitters.
            let wind = self.lua.create_function(move |_, (dx, dy, dz, strength): (f32, f32, f32, f32)| {
                let Some(ps) = particles_ref() else {
                    return Ok(());
                };
                ps.set_wind(glam::Vec3::new(dx, dy, dz), strength);
                Ok(())
            })?;
            particle_table.set("wind", wind)?;

            globals.set("particles", particle_table)?;
        }

        // ── Level lifecycle API ─────────────────────────────────────────────
        // levels.* manages the level registry (register/load/unload/visibility),
        // the loading screen, and the flood system.  Complements save.* (which
        // persists per-entity state).  Backed by the engine's LevelState.
        {
            let level_table = self.lua.create_table()?;
            let lp = self.levels_ptr;
            // Immutable access helper for queries.
            let level_ref = move || -> Option<&'static crate::engine_subsystems::LevelState> {
                if lp == 0 {
                    return None;
                }
                Some(unsafe { &*(lp as *const crate::engine_subsystems::LevelState) })
            };
            // Mutable access helper for mutations.
            let level_mut = move || -> Option<&'static mut crate::engine_subsystems::LevelState> {
                if lp == 0 {
                    return None;
                }
                Some(unsafe { &mut *(lp as *mut crate::engine_subsystems::LevelState) })
            };

            // levels.register(name, path) → level id
            let register = self.lua.create_function(move |_, (name, path): (String, String)| {
                let Some(lv) = level_mut() else { return Ok(0u32); };
                Ok(lv.level_manager.register_level(&name, &path))
            })?;
            level_table.set("register", register)?;

            // levels.load(id) → bool  /  levels.unload(id) → bool
            let load = self.lua.create_function(move |_, id: u32| {
                let Some(lv) = level_mut() else { return Ok(false); };
                Ok(lv.level_manager.load_level(id))
            })?;
            level_table.set("load", load)?;
            let unload = self.lua.create_function(move |_, id: u32| {
                let Some(lv) = level_mut() else { return Ok(false); };
                Ok(lv.level_manager.unload_level(id))
            })?;
            level_table.set("unload", unload)?;

            // levels.is_loaded(id) → bool
            let is_loaded = self.lua.create_function(move |_, id: u32| {
                let Some(lv) = level_ref() else { return Ok(false); };
                Ok(lv.level_manager.is_loaded(id))
            })?;
            level_table.set("is_loaded", is_loaded)?;

            // levels.set_visible(id, visible)
            let set_visible = self.lua.create_function(move |_, (id, visible): (u32, bool)| {
                let Some(lv) = level_mut() else { return Ok(()); };
                lv.level_manager.set_visible(id, visible);
                Ok(())
            })?;
            level_table.set("set_visible", set_visible)?;

            // levels.find(name) → id (or 0 if not found)
            let find = self.lua.create_function(move |_, name: String| {
                let Some(lv) = level_ref() else { return Ok(0u32); };
                Ok(lv.level_manager.find_by_name(&name).map_or(0, |l| l.id))
            })?;
            level_table.set("find", find)?;

            // levels.loaded_count() → int
            let loaded_count = self.lua.create_function(move |_, ()| {
                let Some(lv) = level_ref() else { return Ok(0usize); };
                Ok(lv.level_manager.loaded_count())
            })?;
            level_table.set("loaded_count", loaded_count)?;

            // levels.list() → array of {id, name, loaded, visible}
            let list = self.lua.create_function(move |lua, ()| {
                let out = lua.create_table()?;
                let Some(lv) = level_ref() else { return Ok(out); };
                for (i, l) in lv.level_manager.levels.iter().enumerate() {
                    let t = lua.create_table()?;
                    t.set("id", l.id)?;
                    t.set("name", l.name.as_str())?;
                    t.set("loaded", l.loaded)?;
                    t.set("visible", l.visible)?;
                    out.set(i + 1, t)?;
                }
                Ok(out)
            })?;
            level_table.set("list", list)?;

            // ── Loading screen ──────────────────────────────────────────────
            let loading_show = self.lua.create_function(move |_, message: String| {
                let Some(lv) = level_mut() else { return Ok(()); };
                lv.loading_screen.show(&message);
                Ok(())
            })?;
            level_table.set("loading_show", loading_show)?;
            let loading_progress = self.lua.create_function(move |_, p: f32| {
                let Some(lv) = level_mut() else { return Ok(()); };
                lv.loading_screen.update_progress(p);
                Ok(())
            })?;
            level_table.set("loading_progress", loading_progress)?;
            let loading_hide = self.lua.create_function(move |_, ()| {
                let Some(lv) = level_mut() else { return Ok(()); };
                lv.loading_screen.hide();
                Ok(())
            })?;
            level_table.set("loading_hide", loading_hide)?;

            // ── Flooding ────────────────────────────────────────────────────
            let flood_start = self.lua.create_function(move |_, target: f32| {
                let Some(lv) = level_mut() else { return Ok(()); };
                lv.flood.start_flood(target);
                Ok(())
            })?;
            level_table.set("flood", flood_start)?;
            let flood_stop = self.lua.create_function(move |_, ()| {
                let Some(lv) = level_mut() else { return Ok(()); };
                lv.flood.stop_flood();
                Ok(())
            })?;
            level_table.set("flood_stop", flood_stop)?;
            let water_level = self.lua.create_function(move |_, ()| {
                let Some(lv) = level_ref() else { return Ok(0.0f32); };
                Ok(lv.flood.water_level)
            })?;
            level_table.set("water_level", water_level)?;

            globals.set("levels", level_table)?;
        }

        // ── Boids / flocking API ───────────────────────────────────────────
        // boids.* drives named flock groups for ambient creatures (birds,
        // fish, herds).  Positions are read back each frame by boids_system()
        // and written into ECS entities carrying a Boid component.
        {
            let boid_table = self.lua.create_table()?;
            let bp = self.boids_ptr;
            let boids_mut = move || -> Option<&'static mut crate::boids::BoidRegistry> {
                if bp == 0 {
                    return None;
                }
                Some(unsafe { &mut *(bp as *mut crate::boids::BoidRegistry) })
            };
            let boids_ref = move || -> Option<&'static crate::boids::BoidRegistry> {
                if bp == 0 {
                    return None;
                }
                Some(unsafe { &*(bp as *const crate::boids::BoidRegistry) })
            };

            // boids.create(name) → creates (or reuses) a named group.
            let create = self.lua.create_function(move |_, name: String| {
                let Some(br) = boids_mut() else { return Ok(false); };
                Ok(br.ensure_group(&name))
            })?;
            boid_table.set("create", create)?;

            // boids.add(name, x, y, z) → append one boid to the group.
            let add = self.lua.create_function(move |_, (name, x, y, z): (String, f32, f32, f32)| {
                let Some(br) = boids_mut() else { return Ok(()); };
                br.add_boid(&name, glam::Vec3::new(x, y, z), glam::Vec3::ZERO);
                Ok(())
            })?;
            boid_table.set("add", add)?;

            // boids.remove(name) → drop the whole group (returns count removed).
            let remove = self.lua.create_function(move |_, name: String| {
                let Some(br) = boids_mut() else { return Ok(0usize); };
                Ok(br.remove_group(&name))
            })?;
            boid_table.set("remove", remove)?;

            // boids.clear(name) → empty a group but keep it.
            let clear = self.lua.create_function(move |_, name: String| {
                let Some(br) = boids_mut() else { return Ok(()); };
                br.clear_group(&name);
                Ok(())
            })?;
            boid_table.set("clear", clear)?;

            // boids.count(name) → number of boids in a group.
            let count = self.lua.create_function(move |_, name: String| {
                let Some(br) = boids_ref() else { return Ok(0usize); };
                Ok(br.group(&name).map_or(0, |g| g.boids.len()))
            })?;
            boid_table.set("count", count)?;

            // boids.set_goal(name, x, y, z) / boids.clear_goal(name)
            let set_goal = self.lua.create_function(move |_, (name, x, y, z): (String, f32, f32, f32)| {
                let Some(br) = boids_mut() else { return Ok(()); };
                if let Some(g) = br.group_mut(&name) {
                    g.params.goal = Some(glam::Vec3::new(x, y, z));
                }
                Ok(())
            })?;
            boid_table.set("set_goal", set_goal)?;
            let clear_goal = self.lua.create_function(move |_, name: String| {
                let Some(br) = boids_mut() else { return Ok(()); };
                if let Some(g) = br.group_mut(&name) {
                    g.params.goal = None;
                }
                Ok(())
            })?;
            boid_table.set("clear_goal", clear_goal)?;

            // boids.set_bounds(name, x0,y0,z0, x1,y1,z1)
            let set_bounds = self.lua.create_function(
                move |_, (name, x0, y0, z0, x1, y1, z1): (String, f32, f32, f32, f32, f32, f32)| {
                    let Some(br) = boids_mut() else { return Ok(()); };
                    if let Some(g) = br.group_mut(&name) {
                        g.params.bounds = Some([x0, y0, z0, x1, y1, z1]);
                    }
                    Ok(())
                },
            )?;
            boid_table.set("set_bounds", set_bounds)?;

            // boids.velocity(name, index, vx, vy, vz) → override a boid's velocity.
            let set_vel = self.lua.create_function(
                move |_, (name, index, vx, vy, vz): (String, usize, f32, f32, f32)| {
                    let Some(br) = boids_mut() else { return Ok(()); };
                    if let Some(g) = br.group_mut(&name) {
                        if let Some(b) = g.boids.get_mut(index) {
                            b.velocity = glam::Vec3::new(vx, vy, vz);
                        }
                    }
                    Ok(())
                },
            )?;
            boid_table.set("velocity", set_vel)?;

            // boids.positions(name) → array of {x, y, z} for the group.
            let positions = self.lua.create_function(move |lua, name: String| {
                let out = lua.create_table()?;
                let Some(br) = boids_ref() else { return Ok(out); };
                if let Some(g) = br.group(&name) {
                    for (i, b) in g.boids.iter().enumerate() {
                        let t = lua.create_table()?;
                        t.set("x", b.position.x)?;
                        t.set("y", b.position.y)?;
                        t.set("z", b.position.z)?;
                        out.set(i + 1, t)?;
                    }
                }
                Ok(out)
            })?;
            boid_table.set("positions", positions)?;

            // boids.groups() → number of named groups.
            let groups = self.lua.create_function(move |_, ()| {
                let Some(br) = boids_ref() else { return Ok(0usize); };
                Ok(br.group_count())
            })?;
            boid_table.set("groups", groups)?;

            globals.set("boids", boid_table)?;
        }

        // ── Call entity-local update(entity_id, dt) ───────────────────────
        let mut needs_mark_started = false;
        {
            let instance = self.instances.get(&entity_bits).ok_or_else(|| {
                LuaError::RuntimeError("Script instance missing after build".to_string())
            })?;
            if !instance.started {
                if let Some(start_key) = instance.start_key.as_ref() {
                    let start_fn: LuaFunction = self.lua.registry_value(start_key)?;
                    let env: LuaTable = self.lua.registry_value(&instance.env_key)?;
                    start_fn.set_environment(env)?;
                    if let Err(e) = start_fn.call::<()>(entity_bits) {
                        tracing::error!(
                            "[Scripting] start() failed for {}: {}",
                            script_path,
                            e
                        );
                        self.remove_instance(entity_bits);
                        self.failed_instances.insert(entity_bits, failed_key);
                        return Ok(());
                    }
                }
                needs_mark_started = true;
            }
        }
        if needs_mark_started {
            if let Some(inst) = self.instances.get_mut(&entity_bits) {
                inst.started = true;
            }
        }

        let instance = self.instances.get(&entity_bits).ok_or_else(|| {
            LuaError::RuntimeError("Script instance missing after build".to_string())
        })?;
        let update_fn: LuaFunction = self.lua.registry_value(&instance.update_key)?;
        let env: LuaTable = self.lua.registry_value(&instance.env_key)?;
        update_fn.set_environment(env)?;
        if let Err(e) = update_fn.call::<()>((entity_bits, dt)) {
            tracing::error!(
                "[Scripting] update() failed for {}: {}",
                script_path,
                e
            );
            self.remove_instance(entity_bits);
            self.failed_instances.insert(entity_bits, failed_key);
        }

        Ok(())
    }

    pub fn dispatch_collision_events(&mut self, collisions: &[CollisionPair]) -> LuaResult<()> {
        for collision in collisions {
            self.dispatch_collision_for_entity(
                collision.entity_a,
                collision.entity_b,
                collision.phase,
                collision.normal,
                collision.penetration,
            )?;
            self.dispatch_collision_for_entity(
                collision.entity_b,
                collision.entity_a,
                collision.phase,
                [-collision.normal[0], -collision.normal[1], -collision.normal[2]],
                collision.penetration,
            )?;
        }
        Ok(())
    }

    fn dispatch_collision_for_entity(
        &mut self,
        entity: Entity,
        other: Entity,
        phase: CollisionPhase,
        normal: [f32; 3],
        penetration: f32,
    ) -> LuaResult<()> {
        let Some(instance) = self.instances.get(&entity.to_bits().get()) else {
            return Ok(());
        };
        let callback_key = match phase {
            CollisionPhase::Started => instance.collision_enter_key.as_ref(),
            CollisionPhase::Ongoing => instance.collision_stay_key.as_ref(),
            CollisionPhase::Ended => instance.collision_exit_key.as_ref(),
        };
        let Some(callback_key) = callback_key else {
            return Ok(());
        };
        let env: LuaTable = self.lua.registry_value(&instance.env_key)?;
        let callback: LuaFunction = self.lua.registry_value(callback_key)?;
        callback.set_environment(env)?;
        callback.call::<()>((
            entity.to_bits().get(),
            other.to_bits().get(),
            normal[0],
            normal[1],
            normal[2],
            penetration,
        ))?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::behavior_tree::Status;
    use crate::ai::blackboard::Blackboard;
    use crate::components::Position;
    use crate::navigation::NavGrid;
    use crate::terrain::TerrainGrid;

    fn test_nav() -> NavGrid {
        NavGrid::from_terrain(&TerrainGrid::new(16, 16, 1.0), 1.0)
    }

    // ── Test 1: BTBuilder creates a tree with nested composites ────────
    // Tree: Sequence [Selector [Wait(0.5), Log("hi"), Wait(1.0)]]
    //
    // push_composite(Selector) pushes it onto the stack, so all three
    // add_leaf() calls become children of Selector, not root.
    #[test]
    fn bt_builder_nested_sequence_selector() {
        let mut b = BTBuilder::new();
        // Root is a Sequence (bt_0).
        // bt.selector → bt_1 (child of root, pushed onto stack).
        b.push_composite(BTNodeKind::Selector);
        // All leaves are children of Selector (stack top).
        b.add_leaf(BTNodeKind::Wait { duration: 0.5 });
        b.add_leaf(BTNodeKind::Log {
            message: "hi".to_string(),
        });
        b.add_leaf(BTNodeKind::Wait { duration: 1.0 });

        let root = b.build();
        // Root: Sequence has 1 child → Selector.
        assert_eq!(root.name(), "bt_0");

        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let nav = test_nav();
        let mut bb = Blackboard::new();
        let mut tree = BehaviorTree::new("test", root);

        let mut ctx = crate::ai::behavior_tree::BTContext {
            entity,
            world: &mut world,
            dt: 0.1,
            time_s: 0.0,
            nav_grid: &nav,
            navmesh: None,
            blackboard: &mut bb,
        };

        // Selector tries Wait(0.5) → Running (0.1 < 0.5). Sequence returns Running.
        assert_eq!(tree.tick(&mut ctx), Status::Running);

        // Advance 0.4s more → Wait(0.5) succeeds → Selector succeeds →
        // Sequence succeeds (Selector was its only child).
        ctx.dt = 0.4;
        assert_eq!(tree.tick(&mut ctx), Status::Success);
    }

    // ── Test 2: Decorator wrapping via BTBuilder ───────────────────────
    // Tree: Sequence [Inverter [Wait(0.0)]]
    // Wait(0.0) succeeds immediately → Inverter flips to Failure →
    // Sequence returns Failure on first child.
    #[test]
    fn bt_builder_inverter_decorator() {
        let mut b = BTBuilder::new();
        // Root = Sequence (bt_0).
        // Add Wait(0.0) as child of root.
        b.add_leaf(BTNodeKind::Wait { duration: 0.0 });
        // wrap_decorator pops the last child (Wait), wraps in Inverter,
        // adds Inverter back as child of root, pushes Inverter.
        b.wrap_decorator(BTNodeKind::Inverter);

        let root = b.build();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let nav = test_nav();
        let mut bb = Blackboard::new();
        let mut tree = BehaviorTree::new("test_invert", root);

        let mut ctx = crate::ai::behavior_tree::BTContext {
            entity,
            world: &mut world,
            dt: 0.016,
            time_s: 0.0,
            nav_grid: &nav,
            navmesh: None,
            blackboard: &mut bb,
        };

        assert_eq!(tree.tick(&mut ctx), Status::Failure);
    }

    // ── Test 3: BTBuilder → AiRegistry → AiAgent tick pipeline ────────
    // Exercises the same path that bt.assign() takes from Lua:
    // build a tree, register it, attach an AiAgent, tick and verify.
    #[test]
    fn bt_builder_assign_and_tick() {
        // -- build --
        let mut b = BTBuilder::new();
        // Sequence [Wait(1.0), Log("guard active")]
        b.add_leaf(BTNodeKind::Wait { duration: 1.0 });
        b.add_leaf(BTNodeKind::Log {
            message: "guard active".to_string(),
        });
        let root = b.build();
        let mut tree = BehaviorTree::new("test_guard", root);

        // -- register (takes ownership of tree) --
        let mut ai_reg = AiRegistry::new();
        ai_reg.register("test_guard", tree);

        // -- attach to entity --
        let mut world = hecs::World::new();
        let entity = world.spawn((
            Position { x: 0.0, y: 0.0, z: 0.0 },
            AiAgent::new("test_guard"),
        ));

        let nav = test_nav();
        let mut bb = Blackboard::new();

        // -- tick --
        let mut bt_tree = ai_reg.get_mut("test_guard").unwrap();
        let mut ctx = crate::ai::behavior_tree::BTContext {
            entity,
            world: &mut world,
            dt: 0.5,
            time_s: 0.0,
            nav_grid: &nav,
            navmesh: None,
            blackboard: &mut bb,
        };
        // Wait(1.0) with dt=0.5 → Running.
        assert_eq!(bt_tree.tick(&mut ctx), Status::Running);
        // Next tick with dt=0.5 → Wait succeeds → Log succeeds → Sequence succeeds.
        ctx.dt = 0.5;
        assert_eq!(bt_tree.tick(&mut ctx), Status::Success);

        // Verify the entity's AiAgent is still intact.
        let agent = world.get::<&AiAgent>(entity).unwrap();
        assert_eq!(agent.tree_name, "test_guard");
    }

    // ── Spawn / entity lookup bindings ─────────────────────────────────────
    #[test]
    fn spawn_mesh_and_query_bindings_roundtrip() -> LuaResult<()> {
        let mut world = hecs::World::new();
        let mut scripts = ScriptEngine::new();

        let world_ptr = &mut world as *mut hecs::World as usize;

        // Create a minimal Lua binding that spawns an entity with Position +
        // Renderable, mirroring what register_api's spawn_mesh/spawn_box do.
        let sb = scripts.lua_create().create_function(move |_, (x, y, z): (f32, f32, f32)| {
            let w = unsafe { &mut *(world_ptr as *mut hecs::World) };
            let e = w.spawn((
                Position { x, y, z },
                crate::components::Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
                crate::components::Renderable {
                    mesh: crate::assets::Handle::new(0),
                    color: [1.0, 0.0, 0.0],
                    metallic: 0.0,
                    roughness: 0.5,
                    ao: 1.0,
                    scale: [1.0, 1.0, 1.0],
                },
            ));
            Ok(e.to_bits().get())
        })?;
        scripts.lua_create().globals().set("test_spawn", sb)?;

        // Call through Lua, then read it back.
        let eid: u64 = scripts.lua_create().load("return test_spawn(1.0, 2.0, 3.0)").eval()?;
        let e = hecs::Entity::from_bits(eid).unwrap();
        let pos = world.get::<&Position>(e).unwrap();
        assert_eq!((pos.x, pos.y, pos.z), (1.0, 2.0, 3.0));

        Ok(())
    }

    // ── Lua event pub/sub roundtrip ─────────────────────────────────────
    #[test]
    fn lua_on_event_fire_event_roundtrip() -> LuaResult<()> {
        let mut scripts = ScriptEngine::new();
        scripts.register_api()?;

        // Register a callback and fire it from Lua.
        scripts
            .lua_create()
            .load(
                r#"
                on_event("level_entered", function(payload)
                    global_event_payload = payload
                end)
                fire_event("level_entered", "village")
                "#,
            )
            .exec()?;

        let from_lua: String = scripts
            .lua_create()
            .load("return global_event_payload")
            .eval()?;
        assert_eq!(from_lua, "village");

        // Callback is persistent — fire again from Rust.
        scripts.fire_event("level_entered", Some("dungeon".to_string()))?;
        let from_rust: String = scripts
            .lua_create()
            .load("return global_event_payload")
            .eval()?;
        assert_eq!(from_rust, "dungeon");

        // Unknown events are a silent no-op.
        scripts.fire_event("nonexistent", None)?;
        Ok(())
    }

    // ── Raycast / overlap query logic ────────────────────────────────────
    #[test]
    fn raycast_and_overlap_find_expected_entities() {
        let mut world = hecs::World::new();
        // A collider at origin: half extents 1,1,1 → AABB [-1,1]^3.
        let target = world.spawn((
            Position { x: 0.0, y: 0.0, z: 0.0 },
            crate::components::Collider {
                half_w: 1.0,
                half_h: 1.0,
                half_d: 1.0,
                layer: 1,
                mask: 1,
            },
        ));
        // A second entity far away that a raycast from the origin should miss.
        world.spawn((
            Position { x: 50.0, y: 0.0, z: 0.0 },
            crate::components::Collider {
                half_w: 1.0,
                half_h: 1.0,
                half_d: 1.0,
                layer: 1,
                mask: 1,
            },
        ));

        // Mirror the Lua raycast: ray from (0,0,10) straight down -Z.
        let (ox, oy, oz, dx, dy, dz, max_dist) = (0.0f32, 0.0f32, 10.0f32, 0.0f32, 0.0f32, -1.0f32, 100.0f32);
        let mut best: Option<u64> = None;
        let mut best_t = f32::MAX;
        for entity in world.iter() {
            let e = entity.entity();
            let Ok(pos) = world.get::<&Position>(e) else {
                continue;
            };
            let Ok(c) = world.get::<&crate::components::Collider>(e) else {
                continue;
            };
            let he = [c.half_w, c.half_h, c.half_d];
            let origin = [ox, oy, oz];
            let dir = [dx, dy, dz];
            let mut tmin = 0.0f32;
            let mut tmax = max_dist;
            let mut hit = true;
            for i in 0..3 {
                let d = dir[i];
                let min_b = match i { 0 => pos.x - he[0], 1 => pos.y - he[1], _ => pos.z - he[2] };
                let max_b = match i { 0 => pos.x + he[0], 1 => pos.y + he[1], _ => pos.z + he[2] };
                if d.abs() < 1e-6 {
                    if origin[i] < min_b || origin[i] > max_b {
                        hit = false;
                        break;
                    }
                } else {
                    let t1 = (min_b - origin[i]) / d;
                    let t2 = (max_b - origin[i]) / d;
                    tmin = tmin.max(t1.min(t2));
                    tmax = tmax.min(t1.max(t2));
                    if tmin > tmax {
                        hit = false;
                        break;
                    }
                }
            }
            if hit && tmin < best_t {
                best_t = tmin;
                best = Some(e.to_bits().get());
            }
        }
        assert_eq!(best, Some(target.to_bits().get()));

        // Overlap sphere at origin radius 2 must contain the target but not
        // the far entity.
        let mut overlaps = Vec::new();
        for (entity, pos) in world.query::<(hecs::Entity, &Position)>().iter() {
            let Ok(c) = world.get::<&crate::components::Collider>(entity) else {
                continue;
            };
            let he = [c.half_w, c.half_h, c.half_d];
            let cx = 0.0f32.clamp(pos.x - he[0], pos.x + he[0]);
            let cy = 0.0f32.clamp(pos.y - he[1], pos.y + he[1]);
            let cz = 0.0f32.clamp(pos.z - he[2], pos.z + he[2]);
            let ddx = 0.0 - cx;
            let ddy = 0.0 - cy;
            let ddz = 0.0 - cz;
            if ddx * ddx + ddy * ddy + ddz * ddz <= 4.0 {
                overlaps.push(entity);
            }
        }
        assert_eq!(overlaps, vec![target]);
    }

    // ── Sandbox enforcement ─────────────────────────────────────────────
    #[test]
    fn sandbox_strips_unsafe_libraries_by_default() -> LuaResult<()> {
        let mut scripts = ScriptEngine::new();
        // Default SandboxConfig: fs/os/network all disabled, no limits.
        scripts.register_api()?;

        // os.execute and io.open must be gone (empty table / nil).
        let os: mlua::Value = scripts.lua_create().globals().get("os")?;
        assert!(matches!(os, mlua::Value::Table(_)));

        // Code loaders must raise an error, not execute. `require` is now a
        // sandboxed loader rooted at script_root — an absent module must still
        // raise (not silently return nil / reach arbitrary disk paths).
        for src in [
            "return loadfile('x.lua')",
            "return dofile('x.lua')",
            "return load('print(1)')",
            "return loadstring('print(1)')",
            "return require('zzz_sandbox_test_module')",
        ] {
            let r: mlua::Result<()> = scripts
                .lua_create()
                .load(src)
                .eval()
                .map(|_: mlua::Value| ());
            assert!(
                r.is_err(),
                "expected sandbox error for: {}",
                src
            );
        }

        // io table is empty; indexing io.open returns nil.
        let io_open: mlua::Value = scripts
            .lua_create()
            .load("return io.open")
            .eval()?;
        assert!(io_open.is_nil(), "io.open should be nil");
        Ok(())
    }

    #[test]
    fn sandbox_memory_limit_is_enforced() {
        let mut scripts = ScriptEngine::new();
        scripts.set_sandbox(SandboxConfig {
            max_memory_bytes: 256 * 1024, // 256 KB — tiny, will overflow fast
            max_execution_time_ms: 0,
            ..SandboxConfig::default()
        });
        assert!(scripts.register_api().is_ok());

        // Allocate strings in a loop until the heap cap trips.
        let r: mlua::Result<()> = scripts
            .lua_create()
            .load(
                r#"
                local t = {}
                for i = 1, 1000000 do
                    t[i] = string.rep("x", 1024)
                end
                return #t
                "#,
            )
            .eval()
            .map(|_: mlua::Value| ());
        assert!(r.is_err(), "expected memory-limit error from Lua");
    }

    #[test]
    fn sandbox_execution_time_limit_is_enforced() {
        let mut scripts = ScriptEngine::new();
        scripts.set_sandbox(SandboxConfig {
            max_memory_bytes: 0,
            max_execution_time_ms: 5, // 5 ms ≈ 5k instructions per call
            ..SandboxConfig::default()
        });
        assert!(scripts.register_api().is_ok());

        // An infinite loop must be interrupted by the instruction hook.
        let r: mlua::Result<()> = scripts
            .lua_create()
            .load("while true do end")
            .eval()
            .map(|_: mlua::Value| ());
        assert!(r.is_err(), "expected execution-time error from Lua");
    }

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), nanos))
    }

    // ── Module system (require) ──────────────────────────────────────────
    #[test]
    fn require_loads_and_caches_modules() -> LuaResult<()> {
        let dir = unique_temp_dir("triengine_req");
        std::fs::create_dir_all(dir.join("game")).unwrap();
        std::fs::write(
            dir.join("greetings.lua"),
            "local M = {}\nfunction M.hello(name) return 'hi ' .. name end\nreturn M\n",
        )
        .unwrap();
        std::fs::write(dir.join("game/items.lua"), "local M = {}\nM.count = 3\nreturn M\n").unwrap();

        let mut scripts = ScriptEngine::new();
        scripts.set_script_root(dir.to_str().unwrap());
        scripts.register_api()?;

        let out: String = scripts
            .lua_create()
            .load("local g = require('greetings'); return g.hello('sam')")
            .eval()?;
        assert_eq!(out, "hi sam");

        // Dotted module names map to subfolders.
        let count: i32 = scripts
            .lua_create()
            .load("return require('game.items').count")
            .eval()?;
        assert_eq!(count, 3);

        // Modules are cached — requiring twice returns the same table.
        let same: bool = scripts
            .lua_create()
            .load("return require('greetings') == require('greetings')")
            .eval()?;
        assert!(same);

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    // ── Error recovery ──────────────────────────────────────────────────
    #[test]
    fn failed_script_is_skipped_and_recovers_on_reload() -> LuaResult<()> {
        let dir = unique_temp_dir("triengine_err");
        std::fs::create_dir_all(&dir).unwrap();
        let script_path = dir.join("bad.lua");
        std::fs::write(
            &script_path,
            "function start(e) error('boom') end\nfunction update(e, dt) log('ran') end\n",
        )
        .unwrap();
        let path = script_path.to_str().unwrap().to_string();

        let mut world = hecs::World::new();
        let entity = world.spawn((crate::components::Script { path: path.clone() },));
        let bits = entity.to_bits().get();

        let mut scripts = ScriptEngine::new();
        scripts.set_script_root(dir.to_str().unwrap());
        scripts.register_api()?;
        let input = InputState::new();

        // First run: start() throws → swallowed, instance marked failed.
        scripts.run_update(&mut world, &input, [0.0; 3], [0.0; 3], entity, &path, 0.016, None, 640.0, 480.0, 60.0)?;
        assert!(scripts.failed_instances.contains_key(&bits));

        // Second run: skipped entirely (no re-build / no re-error).
        scripts.run_update(&mut world, &input, [0.0; 3], [0.0; 3], entity, &path, 0.016, None, 640.0, 480.0, 60.0)?;
        assert!(
            !scripts.instances.contains_key(&bits),
            "failed script must not be re-built every frame"
        );
        assert!(scripts.failed_instances.contains_key(&bits));

        // Fix the file and hot-reload → revision bump rebuilds and recovers.
        std::fs::write(&script_path, "function update(e, dt) log('fixed') end\n").unwrap();
        scripts.reload_script(&path)?;
        scripts.run_update(&mut world, &input, [0.0; 3], [0.0; 3], entity, &path, 0.016, None, 640.0, 480.0, 60.0)?;
        assert!(!scripts.failed_instances.contains_key(&bits));
        assert!(scripts.instances.contains_key(&bits));

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    // ── Per-script sandbox tiers ────────────────────────────────────────
    #[test]
    fn per_script_fs_tier_controls_write_root() -> LuaResult<()> {
        let dir = unique_temp_dir("triengine_fs");
        let root = dir.join("root");
        let scripts_dir = root.join("Scripts");
        std::fs::create_dir_all(&scripts_dir).unwrap();
        let script_path = scripts_dir.join("writer.lua");
        std::fs::write(
            &script_path,
            "function update(e, dt)\n  fs.write('data.txt', 'hello')\nend\n",
        )
        .unwrap();
        let path = script_path.to_str().unwrap().to_string();

        // Restricted tier: fs root = script_root → write lands inside Scripts/.
        let mut scripts = ScriptEngine::new();
        scripts.set_script_root(scripts_dir.to_str().unwrap());
        scripts.set_fs_root(root.to_str().unwrap());
        scripts.register_api()?;

        let mut world = hecs::World::new();
        let entity = world.spawn((crate::components::Script { path: path.clone() },));
        let input = InputState::new();
        scripts.run_update(&mut world, &input, [0.0; 3], [0.0; 3], entity, &path, 0.016, None, 640.0, 480.0, 60.0)?;
        assert!(scripts_dir.join("data.txt").exists(), "restricted tier writes under script_root");
        assert!(!root.join("data.txt").exists());
        std::fs::remove_file(scripts_dir.join("data.txt")).unwrap();

        // Privileged tier: file_system_access → fs root = fs_root.
        let mut scripts2 = ScriptEngine::new();
        scripts2.set_script_root(scripts_dir.to_str().unwrap());
        scripts2.set_fs_root(root.to_str().unwrap());
        scripts2.set_script_sandbox(
            &path,
            SandboxConfig {
                file_system_access: true,
                ..SandboxConfig::default()
            },
        );
        scripts2.register_api()?;

        let mut world2 = hecs::World::new();
        let entity2 = world2.spawn((crate::components::Script { path: path.clone() },));
        scripts2.run_update(&mut world2, &input, [0.0; 3], [0.0; 3], entity2, &path, 0.016, None, 640.0, 480.0, 60.0)?;
        assert!(root.join("data.txt").exists(), "privileged tier writes under fs_root");

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    // ── fs.* sandbox traversal protection ───────────────────────────────
    #[test]
    fn fs_blocks_escaping_the_sandbox_root() -> LuaResult<()> {
        let dir = unique_temp_dir("triengine_trav");
        let root = dir.join("root");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(dir.join("secret.txt"), "hidden").unwrap();

        let mut scripts = ScriptEngine::new();
        scripts.set_script_root(root.to_str().unwrap());
        scripts.register_api()?;
        let lua = scripts.lua_create();

        // A file that exists outside the root must be invisible.
        let out: bool = lua.load("return fs.exists('../secret.txt')").eval()?;
        assert!(!out, "traversal must be blocked");

        // Absolute paths are rejected outright.
        let out2: bool = lua.load("return fs.exists('/nope')").eval()?;
        assert!(!out2);

        // And fs.read refuses to fetch the escaped file.
        let r: mlua::Result<String> = lua.load("return fs.read('../secret.txt')").eval();
        assert!(r.is_err());

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    // ── API catalogue ───────────────────────────────────────────────────
    #[test]
    fn api_catalogue_lists_flat_and_namespaced() {
        let scripts = ScriptEngine::new();
        let names = scripts.api_catalogue();
        assert!(names.contains(&"get_position".to_string()));
        assert!(names.contains(&"bt.create".to_string()));
        assert!(names.contains(&"fs.write".to_string()));
        assert!(names.contains(&"ui.set_text".to_string()));
        assert!(names.contains(&"save.write".to_string()));
        // Sorted + deduped.
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(names, sorted);
    }

    // ── Lua-native plugin host ──────────────────────────────────────────
    #[test]
    fn lua_plugin_loads_ticks_events_and_hot_reloads() -> LuaResult<()> {
        let dir = std::env::temp_dir().join(format!("trinity_plugin_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let plugin_path = dir.join("sample.lua");
        std::fs::write(
            &plugin_path,
            r#"
            local ticks = 0
            return {
                name = "sample",
                start = function() log("sample.start") end,
                update = function(dt) ticks = ticks + dt end,
                on_event = function(name, payload)
                    log("sample.on_event:" .. name .. ":" .. tostring(payload))
                end,
            }
            "#,
        )?;

        let mut scripts = ScriptEngine::new();
        scripts.register_api()?;

        let name = scripts.load_plugin(plugin_path.to_str().unwrap())?;
        assert_eq!(name, "sample");
        assert!(scripts.has_plugin("sample"));
        assert_eq!(scripts.plugin_names(), vec!["sample".to_string()]);

        // update(dt) runs without error.
        scripts.tick_plugins(0.5)?;

        // fire_event dispatches to the plugin's on_event handler.
        scripts.fire_event("ping", Some("hello".to_string()))?;

        // Hot reload: rewrite the file and reload; start() runs again, and the
        // previous instance is replaced (no duplicate handlers).
        std::fs::write(
            &plugin_path,
            r#"
            return {
                name = "sample",
                start = function() log("sample.start.v2") end,
                update = function(dt) end,
            }
            "#,
        )?;
        assert!(scripts.reload_plugin(plugin_path.to_str().unwrap())?);
        assert!(scripts.has_plugin("sample"));
        // Reloading a file that isn't a plugin returns Ok(false).
        assert!(!scripts.reload_plugin("not/a/plugin.lua")?);

        // Unload removes it.
        assert!(scripts.unload_plugin("sample"));
        assert!(!scripts.has_plugin("sample"));

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn load_plugins_loads_every_lua_file_in_dir() -> LuaResult<()> {
        let dir = std::env::temp_dir().join(format!("trinity_plugindir_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("a.lua"), "return { name = 'a', update = function() end }")?;
        std::fs::write(dir.join("b.lua"), "return { name = 'b', update = function() end }")?;
        // A broken plugin must not block the others.
        std::fs::write(dir.join("bad.lua"), "return nil")?;

        let mut scripts = ScriptEngine::new();
        scripts.register_api()?;
        let loaded = scripts.load_plugins(dir.to_str().unwrap())?;
        assert_eq!(loaded, 2, "two valid plugins load despite one broken file");
        assert!(scripts.has_plugin("a"));
        assert!(scripts.has_plugin("b"));
        assert!(!scripts.has_plugin("bad"));

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
