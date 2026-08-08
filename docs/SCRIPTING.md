# Scripting (Lua)

Triengine embeds a Lua 5.4 runtime (via `mlua`) for gameplay logic. Every
entity that carries a `Script` component runs its own isolated Lua environment,
so scripts can't accidentally share state with each other.

The scripting system is feature-gated behind `scripting` (default on) and lives
in `src/scripting.rs`, `src/scripting_api.rs`, and the plugins in
`src/ui.rs`, `src/demo_plugin.rs`, and `src/save_plugin.rs`.

## 1) Writing and attaching a script

1. Put a `.lua` file in `Content/Scripts` (see the shipped `player.lua`,
   `enemy.lua`, `guard_ai.lua`, `demo_plugin.lua`).
2. Select a mesh in the Hierarchy.
3. In the Inspector -> `Script` section click `Add script`, type the path
   (e.g. `scripts/my_script.lua` — the path is matched against
   `Content/Scripts/...`), then `Apply path`.
4. Press `Play`. Every frame the engine calls your script (see lifecycle).

Scripts live inside the `Content/Scripts` folder because the file watcher
hot-reloads them: save the file and the running engine picks up the change.

## 2) Lifecycle callbacks

A script is a plain Lua file that defines functions on its own environment.
The engine looks up these names:

| Function | When it runs | Signature |
| -------- | ------------ | --------- |
| `start`  | Once, the first frame the script runs | `start(entity)` |
| `update` | Every frame (**required**) | `update(entity, dt)` |
| `on_collision_enter` | A collision with another entity begins | `(entity, other, nx, ny, nz, penetration)` |
| `on_collision_stay` | A collision continues | `(entity, other, nx, ny, nz, penetration)` |
| `on_collision_exit` | A collision ends | `(entity, other, nx, ny, nz, penetration)` |

`entity` is the entity's numeric ID (`u64`), which is what every entity API
takes as its first argument. `dt` is the frame delta time.

```lua
local move_speed = 6.0

function update(entity, dt)
    local vx, vy, vz = get_velocity(entity)
    local input_x = 0.0
    if is_key_held("A") then input_x = input_x - 1.0 end
    if is_key_held("D") then input_x = input_x + 1.0 end
    set_velocity(entity, input_x * move_speed, vy, vz)
end
```

## 3) API surface

The full catalogue is available at runtime:

```lua
-- Dump every callable global + namespace function
local api = api_catalogue()
```

### Flat globals (a selection)

- **Position / transform**: `get_position`, `set_position`, `move_by`,
  `get_rotation`, `set_rotation`, `set_scale`, `set_color`
- **Physics**: `get_velocity`, `set_velocity`, `apply_force`, `apply_torque`,
  `is_on_ground`, `create_hinge_joint`, `create_fixed_joint`,
  `create_spring_joint`, `create_rope_constraint`, `raycast`, `overlap_sphere`
- **Health**: `get_health`, `set_health`, `damage`, `is_dead`
- **Spawning**: `spawn_mesh`, `spawn_box`, `load_model`, `set_mesh_entity`,
  `destroy`
- **Entities**: `get_all_entities`, `set_tag`, `get_tag`, `has_component`,
  `get_component`, `set_component`
- **Input**: `is_key_held`, `gamepad_left_x`, `gamepad_left_y`,
  `gamepad_button_pressed`, `gamepad_left_magnitude`
- **Camera**: `get_camera`, `set_camera`, `look_at`, `get_camera_direction`,
  `screen_to_ray`
- **Audio**: `audio_play_sfx`, `audio_play_music`, `audio_play_at`,
  `audio_set_volume`, `audio_set_master_volume`, `audio_stop_all`
- **UI**: `set_ui_value`, `get_ui_value`, `set_ui_text`, `set_ui_visible`
- **Effects**: `set_fire`, `remove_fire`, `set_weather`
- **Math**: `vec2`, `vec3`, `vec_add`, `vec_scale`, `vec_dot`, `vec_cross`,
  `vec_normalize`, `vec_lerp`, `lerp`, `clamp`, `clamp01`, `sin`, `cos`, `sqrt`
- **Logging**: `log`, `print`

### Namespaced APIs

| Namespace | Purpose |
| --------- | ------- |
| `bt.*` | Behavior trees (build + assign + blackboard access) |
| `nav.*` | NavGrid A* pathfinding and walkability |
| `navmesh.*` | Polygon navmesh pathfinding |
| `terrain.*` | Terrain height/normal/slope queries + brush ops |
| `particles.*` | Particle emitters, fire sources, wind |
| `levels.*` | Level lifecycle, loading screen, flooding |
| `boids.*` | Named flock groups |
| `cinematic.*` | Cutscene timeline director |
| `ui.*` | Runtime UI designs (create designs, add widgets, set text/values) |
| `fs.*` | Sandboxed file read/write (see Sandbox below) |
| `demo.*` | Example plugin API (`demo.greet`, `demo.bump`, `demo.frame`) |
| `save.*` | World-state persistence (`save.save_entity`, `save.write`, ...) |

### Events, timers, modules

```lua
-- Event pub/sub (engine systems can fire_event() from Rust too)
on_event("level_entered", function(payload) log("entered " .. payload) end)
fire_event("level_entered", "village")

-- Delayed calls
local id = set_timeout(2.0, function() log("2s later") end)
clear_timeout(id)

-- Modules: split code into reusable files under Content/Scripts
-- require("player_movement")      → Content/Scripts/player_movement.lua
-- require("game.items")           → Content/Scripts/game/items.lua
local items = require("game.items")
```

## 4) Sandbox and security

By default scripts are restricted:

- `os`, `io`, `load`, `loadstring`, `loadfile`, `dofile`, `package` are
  stripped/denied.
- A 128 MB heap cap and ~25 ms per-call execution budget are enforced by the
  engine (`src/main.rs` -> `ScriptEngine::set_sandbox`).
- **`require` is a sandboxed loader**: it can only read `.lua` modules from
  `Content/Scripts` (configured via `set_script_root`). Absolute paths and
  `..` traversal are rejected.
- **`fs.*` is the sanctioned file API.** All scripts can read/write inside the
  script root; it never escapes that root.

### Per-script sandbox tiers

The engine-wide sandbox is the baseline. A specific script can be granted more
capabilities without relaxing everything else:

```rust
// Rust, before register_api():
self.scripts.set_script_sandbox(
    "Content/Scripts/tools/export.lua",
    crate::scripting::SandboxConfig {
        file_system_access: true,   // fs.* root widens from script_root to fs_root
        ..crate::scripting::SandboxConfig::default()
    },
);
```

- `file_system_access: true` widens that script's `fs.*` root to the `fs_root`
  (default `Content/`) instead of `Content/Scripts`.
- `os_command_access: true` adds `os.execute(...)` to that script's `os`
  table (off by default; a deliberate opt-in).
- Memory/execution limits are still the engine-wide values.

## 5) Error recovery

- A Lua error is logged and **never crashes the engine**.
- The failing script is **disabled for that entity** until its file changes —
  it is not re-attempted every frame (no error spam).
- Hot-reloading the file bumps its revision, which clears the failure and
  rebuilds the script automatically.

## 6) Extending the API with a plugin

The formal extension point is the `ScriptPlugin` trait
(`src/scripting_api.rs`) plus the `ApiRegistry`. A plugin is a self-contained
API surface mounted onto the Lua globals — no edits to `scripting.rs`.

```rust
// src/my_plugin.rs
#[cfg(feature = "scripting")]
mod inner {
    use mlua::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use crate::scripting_api::{ApiRegistry, ScriptPlugin};

    pub struct MyPlugin { pub counter: Arc<AtomicU32> }

    impl MyPlugin {
        pub fn new() -> Self {
            Self { counter: Arc::new(AtomicU32::new(0)) }
        }
    }

    impl ScriptPlugin for MyPlugin {
        fn name(&self) -> &'static str { "my" }

        fn register(&self, registry: &mut ApiRegistry) -> LuaResult<()> {
            // Namespaced: my.bump() → number
            let c = Arc::clone(&self.counter);
            registry.register_namespaced("my", "bump", move |_, ()| {
                Ok(c.fetch_add(1, Ordering::Relaxed) + 1)
            })?;
            // Flat global: greet(name) → string
            registry.register_function("greet", |_, name: String| {
                Ok(format!("Hello, {name}!"))
            })?;
            Ok(())
        }
    }
}
#[cfg(feature = "scripting")]
pub use inner::MyPlugin;
#[cfg(not(feature = "scripting"))]
pub struct MyPlugin;
```

Register it once at startup, before `register_api()`:

```rust
self.scripts.register_plugin(Box::new(my_plugin::MyPlugin::new()));
```

Then scripts can call `my.bump()` and `greet("engine")`. For the full worked
example see `src/demo_plugin.rs` (and its Lua counterpart
`Content/Scripts/demo_plugin.lua`).

### Plugin rules

- `register()` pushes functions through `register_namespaced(ns, name, f)` or
  `register_function(name, f)`. Namespaces become Lua tables (`my.bump`).
- Closures must be `Fn + Send + Sync + 'static`; capture shared state with
  `Arc` (a `Mutex`/`Atomic*` inside is fine) — you cannot capture `&self`.
- `name()` identifies the plugin for debugging/teardown.
- Gate the plugin body behind `#[cfg(feature = "scripting")]` and provide a
  placeholder struct when scripting is disabled so `main.rs` can allocate it
  unconditionally (mirror `demo_plugin.rs`).
