// src/save_plugin.rs
// ──────────────────────────────────────────────────────────────────────────────
// Bombs a save/load scripting API onto the `save.*` namespace.  This is a
// practical example of the plugin pattern bridging an engine subsystem
// (the levels::WorldStateManager) into Lua without touching scripting.rs.
//
// A game uses this to persist per-entity state across a session:
//   save.set_flag("level1", "door", "opened", "true")
//   save.get_flag("level1", "door", "opened")
//   save.save_entity("level1", "goblin_01", 10, 0, 5, 50, true)
//   save.is_entity_dead("level1", "goblin_01")
//   save.write("saves/current.json")
//   save.read("saves/current.json")
//   save.clear()
//
// The plugin stores a shared handle to the real WorldStateManager that the
// engine (main.rs) holds, so serialization is shared with C++-side systems.
// ──────────────────────────────────────────────────────────────────────────────

use std::sync::{Arc, Mutex};

#[cfg(feature = "scripting")]
mod inner {
    use super::*;
    use mlua::prelude::*;
    use crate::scripting_api::{ApiRegistry, ScriptPlugin};

    pub struct WorldStatePlugin {
        state: Arc<Mutex<crate::levels::WorldStateManager>>,
    }

    impl WorldStatePlugin {
        pub fn new(state: Arc<Mutex<crate::levels::WorldStateManager>>) -> Self {
            Self { state }
        }
    }

    impl ScriptPlugin for WorldStatePlugin {
        fn name(&self) -> &'static str {
            "world_state"
        }

        fn register(&self, registry: &mut ApiRegistry) -> LuaResult<()> {
            // save.save_entity(level, name, x, y, z, health, alive)
            let st = Arc::clone(&self.state);
            registry.register_namespaced(
                "save",
                "save_entity",
                move |_, (level, name, x, y, z, hp, alive): (String, String, f32, f32, f32, i32, bool)| {
                    st.lock().unwrap().save_entity(&level, crate::levels::EntityState {
                        entity_name: name,
                        position: [x, y, z],
                        health: Some(hp),
                        is_alive: alive,
                        custom_flags: Default::default(),
                    });
                    Ok(())
                },
            )?;

            // save.get_entity_flags(level, name) → table of custom flags (may be empty).
            let st = Arc::clone(&self.state);
            registry.register_namespaced(
                "save",
                "get_entity_flags",
                move |lua, (level, name): (String, String)| {
                    let mgr = st.lock().unwrap();
                    let t = lua.create_table()?;
                    if let Some(state) = mgr.get_entity(&level, &name) {
                        for (k, v) in &state.custom_flags {
                            t.set(k.as_str(), v.as_str())?;
                        }
                    }
                    Ok(t)
                },
            )?;

            // is_entity_dead(level, name) → bool
            let st = Arc::clone(&self.state);
            registry.register_namespaced(
                "save",
                "is_entity_dead",
                move |_, (level, name): (String, String)| {
                    Ok(st.lock().unwrap().is_entity_dead(&level, &name))
                },
            )?;

            // set_flag(level, name, key, value)
            let st = Arc::clone(&self.state);
            registry.register_namespaced(
                "save",
                "set_flag",
                move |_, (level, name, key, value): (String, String, String, String)| {
                    st.lock().unwrap().set_flag(&level, &name, &key, &value);
                    Ok(())
                },
            )?;

            // get_flag(level, name, key) → string or nil
            let st = Arc::clone(&self.state);
            registry.register_namespaced(
                "save",
                "get_flag",
                move |_, (level, name, key): (String, String, String)| {
                    let v = st.lock().unwrap().get_flag(&level, &name, &key).map(str::to_string);
                    Ok(v)
                },
            )?;

            // clear_level(level) / clear() — reset persisted state.
            let st = Arc::clone(&self.state);
            registry.register_namespaced("save", "clear_level", move |_, level: String| {
                st.lock().unwrap().clear_level(&level);
                Ok(())
            })?;
            let st = Arc::clone(&self.state);
            registry.register_namespaced("save", "clear", move |_, ()| {
                st.lock().unwrap().clear_all();
                Ok(())
            })?;

            // write(path) → serialize whole world to JSON. read(path) → load.
            let st = Arc::clone(&self.state);
            registry.register_namespaced("save", "write", move |_, path: String| {
                st.lock().unwrap().save_to_file(&path).map_err(mlua::Error::RuntimeError)?;
                Ok(path)
            })?;
            let st = Arc::clone(&self.state);
            registry.register_namespaced("save", "read", move |_lua, path: String| {
                let loaded = crate::levels::WorldStateManager::load_from_file(&path)
                    .map_err(mlua::Error::RuntimeError)?;
                *st.lock().unwrap() = loaded;
                // Return a count of saved entities so Lua can confirm success.
                let n = st.lock().unwrap().total_entities();
                Ok(n)
            })?;

            Ok(())
        }
    }
}

// Public re-export similar to demo_plugin so main.rs can allocate unconditionally.
#[cfg(feature = "scripting")]
pub use inner::WorldStatePlugin;

// Non-scripting build: placeholder.
#[cfg(not(feature = "scripting"))]
pub struct WorldStatePlugin;