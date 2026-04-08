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
    pending_camera_set: std::cell::UnsafeCell<Option<([f32; 3], [f32; 3])>>,
    pending_frame_skip: std::cell::UnsafeCell<u32>,
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
            pending_camera_set: std::cell::UnsafeCell::new(None),
            pending_frame_skip: std::cell::UnsafeCell::new(0),
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

    // register_api() sets up engine-wide Lua globals (logging, print).
    // Call once at startup before loading any scripts.
    pub fn register_api(&mut self) -> LuaResult<()> {
        let globals = self.lua.globals();

        // Override Lua's built-in print to route through our logging.
        let print_fn = self.lua.create_function(|_, msg: String| {
            println!("[Lua] {}", msg);
            Ok(())
        })?;
        globals.set("print", print_fn)?;

        let log_fn = self.lua.create_function(|_, msg: String| {
            println!("[Script] {}", msg);
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
        println!("[Scripting] Loaded: {}", path);
        Ok(())
    }

    // reload_script() hot-reloads a changed file.
    // New function definitions replace the old ones in Lua globals.
    pub fn reload_script(&mut self, path: &str) -> LuaResult<()> {
        println!("[Scripting] Reloading: {}", path);
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
            let mut watcher = recommended_watcher(move |res| { let _ = ntx.send(res); })
                .expect("Could not create file watcher");
            watcher.watch(Path::new(&watch_dir), RecursiveMode::Recursive)
                .expect("Could not watch directory");

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
                    Ok(Err(e)) => eprintln!("[HotReload] Watcher error: {}", e),
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
            let entity = Entity::from_bits(eid).unwrap();
            match world.get::<&Position>(entity) {
                Ok(p)  => Ok((p.x, p.y, p.z)),
                Err(_) => Ok((0.0f32, 0.0f32, 0.0f32)),
            }
        })?;
        globals.set("get_position", gp)?;

        // set_position(entity, x, y, z)
        let sp = self.lua.create_function(move |_, (eid, x, y, z): (u64, f32, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = Entity::from_bits(eid).unwrap();
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
                let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
            match world.get::<&RigidBody>(entity) {
                Ok(b)  => Ok((b.velocity_x, b.velocity_y, b._velocity_z)),
                Err(_) => Ok((0.0f32, 0.0f32, 0.0f32)),
            }
        })?;
        globals.set("get_velocity", gv)?;

        // set_velocity(entity, vx, vy, vz) — replaces current velocity
        let sv = self.lua.create_function(move |_, (eid, vx, vy, vz): (u64, f32, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
            let v = world
                .get::<&RigidBody>(entity)
                .map(|b| b.angular_velocity)
                .unwrap_or(0.0);
            Ok(v)
        })?;
        globals.set("get_angular_velocity", gav)?;

        let sav = self.lua.create_function(move |_, (eid, w): (u64, f32)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
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
                let entity = Entity::from_bits(eid).unwrap();
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
                let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
            match world.get::<&Health>(entity) {
                Ok(h)  => Ok((h.current, h.max)),
                Err(_) => Ok((0i32, 0i32)),
            }
        })?;
        globals.set("get_health", gh)?;

        // set_health(entity, current, max)
        let sh = self.lua.create_function(move |_, (eid, cur, max): (u64, i32, i32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
            if let Ok(mut h) = world.get::<&mut Health>(entity) {
                h.current = (h.current - amount).max(0);
            }
            Ok(())
        })?;
        globals.set("damage", dmg)?;

        // is_dead(entity) → bool
        let id = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
            if let Ok(mut rend) = world.get::<&mut Renderable>(entity) {
                rend.color = [r, g, b];
            }
            Ok(())
        })?;
        globals.set("set_color", sc)?;

        // set_scale(entity, sx, sy, sz)
        let ss = self.lua.create_function(move |_, (eid, sx, sy, sz): (u64, f32, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = Entity::from_bits(eid).unwrap();
            if let Ok(mut rend) = world.get::<&mut Renderable>(entity) {
                rend.scale = [sx, sy, sz];
            }
            Ok(())
        })?;
        globals.set("set_scale", ss)?;

        // get_rotation(entity) -> pitch, yaw, roll
        let gr = self.lua.create_function(move |_, eid: u64| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let entity = Entity::from_bits(eid).unwrap();
            match world.get::<&Rotation>(entity) {
                Ok(r) => Ok((r.pitch, r.yaw, r.roll)),
                Err(_) => Ok((0.0f32, 0.0f32, 0.0f32)),
            }
        })?;
        globals.set("get_rotation", gr)?;

        // set_rotation(entity, pitch, yaw, roll)
        let sr = self.lua.create_function(move |_, (eid, p, y, r): (u64, f32, f32, f32)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let entity = Entity::from_bits(eid).unwrap();
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
                let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
            let path = world
                .get::<&MaterialTexture>(entity)
                .map(|t| t.path.clone())
                .unwrap_or_default();
            Ok(path)
        })?;
        globals.set("get_texture_path", gtp)?;
        let stp = self.lua.create_function(move |_, (eid, path): (u64, String)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let entity = Entity::from_bits(eid).unwrap();
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

        // has_component(entity, name) — lightweight reflection helper.
        let hc = self.lua.create_function(move |_, (eid, name): (u64, String)| {
            let world = unsafe { &mut *(world_ptr as *mut World) };
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
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
            let entity = Entity::from_bits(eid).unwrap();
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
