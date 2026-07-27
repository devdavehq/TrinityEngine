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
    // Raw pointer to AiRegistry for bt.assign() tree registration.
    ai_registry_ptr: usize,
    // Raw pointer to TerrainWorld for terrain_height / terrain_raise / etc.
    terrain_world_ptr: usize,
    /// Sandbox configuration — controls what scripts are allowed to do.
    pub sandbox: SandboxConfig,
}

// SAFETY: ScriptEngine is only ever used on the main thread.
unsafe impl Send for ScriptEngine {}

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
            ai_registry_ptr: 0,
            terrain_world_ptr: 0,
            sandbox: SandboxConfig::default(),
        }
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
    ) {
        self.nav_grid_ptr = nav as *const NavGrid as usize;
        self.ai_registry_ptr = ai_reg as *mut AiRegistry as usize;
        self.terrain_world_ptr = terrain as *mut crate::terrain::TerrainWorld as usize;
    }

    /// Expose the Lua instance so external code can set globals (e.g., ui_click_event).
    pub fn lua_create(&self) -> &Lua {
        &self.lua
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

        // ── Sandbox enforcement ─────────────────────────────────────────
        // Remove dangerous Lua standard libraries based on SandboxConfig.
        // By default everything is restricted; enable only what is allowed.
        if !self.sandbox.os_command_access {
            globals.set("os", self.lua.create_table()?)?;
        }
        if !self.sandbox.file_system_access {
            globals.set("io", self.lua.create_table()?)?;
            globals.set("loadfile", self.lua.create_function(|_, ()| -> LuaResult<()> {
                Err(LuaError::RuntimeError(
                    "loadfile is disabled by sandbox".to_string()
                ))
            })?)?;
            globals.set("dofile", self.lua.create_function(|_, ()| -> LuaResult<()> {
                Err(LuaError::RuntimeError(
                    "dofile is disabled by sandbox".to_string()
                ))
            })?)?;
        }

        Ok(())
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
        let needs_rebuild = self
            .instances
            .get(&entity_bits)
            .map(|i| i.path != script_path || i.revision != current_rev)
            .unwrap_or(true);

        if needs_rebuild {
            self.remove_instance(entity_bits);
            let (env_key, start_key, update_key, collision_enter_key, collision_stay_key, collision_exit_key, revision) =
                self.build_instance(script_path)?;
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

        // ── Input ─────────────────────────────────────────────────────────
        // is_key_held("W") → bool
        let kh = self.lua.create_function(move |_, key: String| {
            let input = unsafe { &*(input_ptr as *const InputState) };
            let held = input.is_virtual_key_held(&key);
            Ok(held)
        })?;
        globals.set("is_key_held", kh)?;

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
            let near3 = near4.xyz() / near4.w;
            let far3  = far4.xyz() / far4.w;
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
            let sp = script_ptr;
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
            let sp = script_ptr;
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
            let sp = script_ptr;
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
            let sp = script_ptr;
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
            let sp = script_ptr;
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
            let sp = script_ptr;
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
            let find_path = self.lua.create_function(move |lua, (x1, y1, z1, x2, y2, z2): (f32, f32, f32, f32, f32, f32)| {
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
                    start_fn.call::<()>(entity_bits)?;
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
        update_fn.call::<()>((entity_bits, dt))?;

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
}
