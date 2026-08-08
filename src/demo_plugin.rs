// src/demo_plugin.rs
// ──────────────────────────────────────────────────────────────────────────────
// A playable demonstration of the engine's formal plugin pattern.
//
// This file shows EVERYTHING you need to extend the Lua scripting API from a
// game module WITHOUT editing scripting.rs.
//
// The plugin pays attention to the scripting feature gate (mlua is optional),
// so it can only exist when scripting is enabled.  It documents:
//   1. the ScriptPlugin trait (name + register)
//   2. a namespaced API surface (e.g. `demo.greet(...)`)
//   3. per-frame data shared with the engine through a Mutex handle
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "scripting")]
mod inner {
    use mlua::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::scripting_api::{ApiRegistry, ScriptPlugin};

    /// Per-agent data a game might track.  Here it's a tiny counter to show
    /// that plugins can own mutable state that survives across Lua calls.
    #[derive(Default)]
    pub struct DemoRuntime {
        /// Total `demo.bump("goblin")` calls — lets a game count spawns.
        pub bumps: Mutex<std::collections::HashMap<String, u32>>,
        /// A tick counter the engine advances each frame.
        pub frame: AtomicU32,
    }

    impl DemoRuntime {
        fn bump(&self, key: &str) -> u32 {
            let mut map = self.bumps.lock().unwrap();
            let v = *map.get(key).unwrap_or(&0) + 1;
            map.insert(key.to_string(), v);
            v
        }
    }

    /// The plugin itself.  `new()` creates its shared state; `register()` pushes
    /// the Lua functions.  `name()` identifies the plugin for teardown/debug.
    pub struct DemoPlugin {
        pub runtime: Arc<DemoRuntime>,
    }

    impl DemoPlugin {
        pub fn new() -> Self {
            Self {
                runtime: Arc::new(DemoRuntime::default()),
            }
        }

        /// Build a plugin that shares the caller's runtime. Lets the engine
        /// register one instance with the script engine while ticking a second
        /// handle to the same state, so `demo.frame()` reflects real frames.
        pub fn with_runtime(runtime: Arc<DemoRuntime>) -> Self {
            Self { runtime }
        }

        pub fn runtime(&self) -> Arc<DemoRuntime> {
            Arc::clone(&self.runtime)
        }

        /// Called by the engine each frame to drive per-frame state.
        pub fn tick(&self) {
            self.runtime.frame.fetch_add(1, Ordering::Relaxed);
        }

        /// Accessor for the editor/tests to read the current frame counter.
        pub fn frame(&self) -> u32 {
            self.runtime.frame.load(Ordering::Relaxed)
        }
    }

    impl Default for DemoPlugin {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ScriptPlugin for DemoPlugin {
        fn name(&self) -> &'static str {
            "demo"
        }

        fn register(&self, registry: &mut ApiRegistry) -> LuaResult<()> {
            // demo.greet(name) → "Hello, name!" — pure function.
            registry.register_namespaced("demo", "greet", move |_, name: String| {
                Ok(format!("Hello, {name}!"))
            })?;

            // demo.bump(key) → count — demonstrates shared mutable state.
            let rt2 = Arc::clone(&self.runtime);
            registry.register_namespaced("demo", "bump", move |_, key: String| {
                Ok(rt2.bump(&key))
            })?;

            // demo.frame() → current engine-side frame number.
            let rt3 = Arc::clone(&self.runtime);
            registry.register_namespaced("demo", "frame", move |_, ()| {
                Ok(rt3.frame.load(Ordering::Relaxed))
            })?;

            Ok(())
        }
    }
}

// Re-export so main.rs can allocate and tick the plugin uniformly, regardless
// of the scripting feature.  With scripting disabled the public type is a
// placeholder; scripting.rs registration simply won't see it.
#[cfg(feature = "scripting")]
pub use inner::DemoPlugin;

#[cfg(feature = "scripting")]
pub type DemoRuntime = inner::DemoRuntime;

// With scripting disabled the plugin is a no-op but the struct still exists so
// main.rs can allocate it unconditionally.
#[cfg(not(feature = "scripting"))]
pub struct DemoPlugin;

#[cfg(not(feature = "scripting"))]
pub const DEMO_PLUGIN: DemoPlugin = DemoPlugin;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[cfg(feature = "scripting")]
    #[test]
    fn demo_plugin_registers_and_acts() -> mlua::prelude::LuaResult<()> {
        use mlua::prelude::{Lua, LuaFunction};
        use crate::scripting_api::{ApiRegistry, ScriptPlugin};
        use super::inner::DemoPlugin;

        let demo = DemoPlugin::new();
        let lua = Lua::new();
        let mut reg = ApiRegistry::new(&lua);
        ScriptPlugin::register(&demo, &mut reg)?;
        reg.apply()?;

        let globals = lua.globals();
        let demo_tbl: mlua::Table = globals.get("demo")?;
        let greet: LuaFunction = demo_tbl.get("greet")?;
        assert_eq!(greet.call::<String>("goblin")?, "Hello, goblin!");
        Ok(())
    }
}