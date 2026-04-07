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
//   local vx, vy = get_velocity(entity)
//   set_velocity(entity, vx, vy)
//   apply_force(entity, fx, fy)        -- adds to current velocity
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
use notify::{Event, RecursiveMode, Watcher, recommended_watcher};
use std::path::Path;
use std::sync::mpsc;
use std::fs;

use hecs::{Entity, World};
use crate::components::{Health, Position, Renderable, RigidBody};
use crate::input::InputState;

// ── ScriptEngine ──────────────────────────────────────────────────────────────
pub struct ScriptEngine {
    lua:              Lua,
    // Entities destroyed by Lua this frame — drained after scripting_system.
    // UnsafeCell because we hand raw pointers into Lua closures.
    pending_destroys: std::cell::UnsafeCell<Vec<Entity>>,
    // Engine start instant — used by elapsed_time().
    start_time:       std::cell::UnsafeCell<std::time::Instant>,
}

// SAFETY: ScriptEngine is only ever used on the main thread.
unsafe impl Send for ScriptEngine {}

impl ScriptEngine {
    pub fn new() -> Self {
        Self {
            lua:              Lua::new(),
            pending_destroys: std::cell::UnsafeCell::new(Vec::new()),
            start_time:       std::cell::UnsafeCell::new(std::time::Instant::now()),
        }
    }

    // register_api() sets up engine-wide Lua globals (logging, print).
    // Call once at startup before loading any scripts.
    pub fn register_api(&self) -> LuaResult<()> {
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

        Ok(())
    }

    // load_script() reads a .lua file from disk and executes it.
    // This defines any top-level variables and the update() function.
    pub fn load_script(&self, path: &str) -> LuaResult<()> {
        let code = fs::read_to_string(path)
            .map_err(|e| LuaError::RuntimeError(
                format!("Could not load script {}: {}", path, e)
            ))?;
        self.lua.load(&code).set_name(path).exec()?;
        println!("[Scripting] Loaded: {}", path);
        Ok(())
    }

    // reload_script() hot-reloads a changed file.
    // New function definitions replace the old ones in Lua globals.
    pub fn reload_script(&self, path: &str) -> LuaResult<()> {
        println!("[Scripting] Reloading: {}", path);
        self.load_script(path)
    }

    // drain_destroys() — call after scripting_system each frame.
    // Removes entities that Lua scripts called destroy() on.
    pub fn drain_destroys(&self, world: &mut World) {
        let destroys = unsafe { &mut *self.pending_destroys.get() };
        for entity in destroys.drain(..) {
            let _ = world.despawn(entity);
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
    pub fn run_update(
        &self,
        world:  &mut World,
        input:  &InputState,
        entity: Entity,
        dt:     f32,
    ) -> LuaResult<()> {
        let globals   = self.lua.globals();
        let world_ptr = world as *mut World as usize;
        let input_ptr = input as *const InputState as usize;

        // ── Position ──────────────────────────────────────────────────────
        // get_position(entity) → x, y, z
        let gp = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
            match world.get::<&Position>(entity) {
                Ok(p)  => Ok((p.x, p.y, p.z)),
                Err(_) => Ok((0.0f32, 0.0f32, 0.0f32)),
            }
        })?;
        globals.set("get_position", gp)?;

        // set_position(entity, x, y, z)
        let sp = self.lua.create_function(move |_, (eid, x, y, z): (u64, f32, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
            if let Ok(mut p) = world.get::<&mut Position>(entity) {
                p.x = x; p.y = y; p.z = z;
            }
            Ok(())
        })?;
        globals.set("set_position", sp)?;

        // ── RigidBody ─────────────────────────────────────────────────────
        // get_velocity(entity) → vx, vy
        let gv = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
            match world.get::<&RigidBody>(entity) {
                Ok(b)  => Ok((b.velocity_x, b.velocity_y)),
                Err(_) => Ok((0.0f32, 0.0f32)),
            }
        })?;
        globals.set("get_velocity", gv)?;

        // set_velocity(entity, vx, vy) — replaces current velocity
        let sv = self.lua.create_function(move |_, (eid, vx, vy): (u64, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
            if let Ok(mut b) = world.get::<&mut RigidBody>(entity) {
                b.velocity_x = vx;
                b.velocity_y = vy;
            }
            Ok(())
        })?;
        globals.set("set_velocity", sv)?;

        // apply_force(entity, fx, fy) — adds to current velocity
        let af = self.lua.create_function(move |_, (eid, fx, fy): (u64, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
            if let Ok(mut b) = world.get::<&mut RigidBody>(entity) {
                b.velocity_x += fx;
                b.velocity_y += fy;
            }
            Ok(())
        })?;
        globals.set("apply_force", af)?;

        // is_on_ground(entity) → bool
        let og = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
            let g = world.get::<&RigidBody>(entity)
                .map(|b| b.on_ground)
                .unwrap_or(false);
            Ok(g)
        })?;
        globals.set("is_on_ground", og)?;

        // ── Health ────────────────────────────────────────────────────────
        // get_health(entity) → current, max
        let gh = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
            match world.get::<&Health>(entity) {
                Ok(h)  => Ok((h.current, h.max)),
                Err(_) => Ok((0i32, 0i32)),
            }
        })?;
        globals.set("get_health", gh)?;

        // set_health(entity, current, max)
        let sh = self.lua.create_function(move |_, (eid, cur, max): (u64, i32, i32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
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
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
            if let Ok(mut h) = world.get::<&mut Health>(entity) {
                h.current = (h.current - amount).max(0);
            }
            Ok(())
        })?;
        globals.set("damage", dmg)?;

        // is_dead(entity) → bool
        let id = self.lua.create_function(move |_, eid: u64| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
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
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
            if let Ok(mut rend) = world.get::<&mut Renderable>(entity) {
                rend.color = [r, g, b];
            }
            Ok(())
        })?;
        globals.set("set_color", sc)?;

        // set_scale(entity, sx, sy, sz)
        let ss = self.lua.create_function(move |_, (eid, sx, sy, sz): (u64, f32, f32, f32)| {
            let world  = unsafe { &mut *(world_ptr as *mut World) };
            let entity = unsafe { Entity::from_bits(eid).unwrap() };
            if let Ok(mut rend) = world.get::<&mut Renderable>(entity) {
                rend.scale = [sx, sy, sz];
            }
            Ok(())
        })?;
        globals.set("set_scale", ss)?;

        // ── Entity lifetime ───────────────────────────────────────────────
        // destroy(entity) — deferred removal, processed after scripting_system.
        let destroys_ptr = self.pending_destroys.get() as usize;
        let df = self.lua.create_function(move |_, eid: u64| {
            let entity   = unsafe { Entity::from_bits(eid).unwrap() };
            let destroys = unsafe { &mut *(destroys_ptr as *mut Vec<Entity>) };
            destroys.push(entity);
            Ok(())
        })?;
        globals.set("destroy", df)?;

        // ── Input ─────────────────────────────────────────────────────────
        // is_key_held("W") → bool
        let kh = self.lua.create_function(move |_, key: String| {
            let input = unsafe { &*(input_ptr as *const InputState) };
            use winit::keyboard::KeyCode;
            let held = match key.as_str() {
                "W"          | "ArrowUp"    => input.is_held(KeyCode::KeyW)    || input.is_held(KeyCode::ArrowUp),
                "S"          | "ArrowDown"  => input.is_held(KeyCode::KeyS)    || input.is_held(KeyCode::ArrowDown),
                "A"          | "ArrowLeft"  => input.is_held(KeyCode::KeyA)    || input.is_held(KeyCode::ArrowLeft),
                "D"          | "ArrowRight" => input.is_held(KeyCode::KeyD)    || input.is_held(KeyCode::ArrowRight),
                "Space"                     => input.is_held(KeyCode::Space),
                "Shift"                     => input.is_held(KeyCode::ShiftLeft) || input.is_held(KeyCode::ShiftRight),
                "Ctrl"                      => input.is_held(KeyCode::ControlLeft) || input.is_held(KeyCode::ControlRight),
                "E"                         => input.is_held(KeyCode::KeyE),
                "Q"                         => input.is_held(KeyCode::KeyQ),
                "R"                         => input.is_held(KeyCode::KeyR),
                "F"                         => input.is_held(KeyCode::KeyF),
                _                           => false,
            };
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

        // ── Δt passthrough ────────────────────────────────────────────────
        // Scripts receive dt as the second argument to update(entity, dt),
        // but also expose it as a global so helper functions can read it.
        globals.set("dt", dt)?;

        // ── Call update(entity_id, dt) in the script ──────────────────────
        let update_fn: Option<LuaFunction> = globals.get("update")?;
        if let Some(f) = update_fn {
            // entity.to_bits() gives u64 — Lua stores this as a number.
            // Scripts should treat it as opaque and pass it back unchanged.
            f.call::<()>((entity.to_bits(), dt))?;
        }

        Ok(())
    }
}