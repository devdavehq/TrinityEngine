// src/scripting_api.rs
// ──────────────────────────────────────────────────────────────────────────────
// Formal Lua API registry + plugin pattern.
//
// PROBLEM IT SOLVES:
//   Historically the Lua API surface was scattered across hand-written
//   globals.set() calls in three places (register_api(), run_update(),
//   ui::register_ui_lua_api()). There was no single source of truth, no way to
//   list what a script can call, and adding a global meant editing a central
//   blob of code.
//
// DESIGN:
//   ApiRegistry — an explicit catalogue of named Lua functions.  Every API
//                 entry is registered once under a unique name and then
//                 mounted onto the Lua globals (either flat, or namespaced
//                 under a table like `ui.`).
//   ScriptPlugin — the extension point.  A plugin is a self-contained unit
//                  that mounts a named API surface (a table) onto the
//                  registry.  Engine subsystems ship as plugins; a game can
//                  add its own plugins without touching scripting.rs.
//
// WHY A FORMAL REGISTRY:
//   1. One source of truth — the registry is inspectable, so doc-gen,
//      editor autocomplete and sandbox allow-lists can enumerate the API.
//   2. Decoupled growth — a new feature subsystem registers its own plugin
//      instead of editing a shared globals blob.
//   3. Security tiering — a registry entry can carry a sandbox tier so the
//      enforcement layer can strip bindings a project hasn't granted.
//   4. Deterministic teardown — unloading a plugin means removing exactly
//      the entries it registered.
//
// USAGE:
//   let mut registry = ApiRegistry::new(lua);
//   registry.mount_plugin(&UiPlugin);
//   registry.mount_plugin(&MyGamePlugin);
//   registry.apply()?;   // installs everything onto Lua globals
// ──────────────────────────────────────────────────────────────────────────────

use mlua::prelude::*;

/// A single registered Lua function (or value) plus the table it lives under.
/// `namespace` of "" means a flat global; otherwise the entry is nested under
/// `namespace` (creating the table if needed).
#[derive(Clone)]
pub struct ApiEntry {
    pub namespace: &'static str,
    pub name: &'static str,
    pub function: LuaFunction,
}

/// One self-contained Lua API surface.
///
/// Implement this for each subsystem (engine or game-side) that wants to
/// expose functions to scripts.  `name()` is used for debugging/teardown;
/// `register()` pushes that subsystem's functions into the registry.
pub trait ScriptPlugin: Send + Sync {
    /// Plugin identifier, e.g. "ui", "events", "gameplay".
    fn name(&self) -> &'static str;

    /// Push all of this plugin's Lua functions into `registry`.
    fn register(&self, registry: &mut ApiRegistry) -> LuaResult<()>;
}

/// The catalogue of Lua API entries and the thing that mounts them onto the
/// Lua state.  Created once at ScriptEngine startup.
pub struct ApiRegistry<'a> {
    lua: &'a Lua,
    entries: Vec<ApiEntry>,
}

impl<'a> ApiRegistry<'a> {
    pub fn new(lua: &'a Lua) -> Self {
        Self {
            lua,
            entries: Vec::new(),
        }
    }

    /// Register a flat global function, e.g. `log("hi")`.
    pub fn register_function<A, R, F>(
        &mut self,
        name: &'static str,
        f: F,
    ) -> LuaResult<()>
    where
        A: mlua::FromLuaMulti + 'static,
        R: mlua::IntoLuaMulti + 'static,
        F: Fn(&Lua, A) -> LuaResult<R> + Send + Sync + 'static,
    {
        let function = self.lua.create_function(f)?;
        self.entries.push(ApiEntry {
            namespace: "",
            name,
            function,
        });
        Ok(())
    }

    /// Register a namespaced function, e.g. `ui.set_text(...)`.
    /// The `namespace` table is created lazily on apply().
    pub fn register_namespaced<A, R, F>(
        &mut self,
        namespace: &'static str,
        name: &'static str,
        f: F,
    ) -> LuaResult<()>
    where
        A: mlua::FromLuaMulti + 'static,
        R: mlua::IntoLuaMulti + 'static,
        F: Fn(&Lua, A) -> LuaResult<R> + Send + Sync + 'static,
    {
        let function = self.lua.create_function(f)?;
        self.entries.push(ApiEntry {
            namespace,
            name,
            function,
        });
        Ok(())
    }

    /// Register one plugin's whole API surface (all entries it pushes into the
    /// registry) onto this registry.
    pub fn mount_plugin(&mut self, plugin: &dyn ScriptPlugin) -> LuaResult<()> {
        plugin.register(self)
    }

    /// Mount all registered entries onto the Lua globals.
    /// Call once after every plugin has been registered.
    pub fn apply(&self) -> LuaResult<()> {
        let globals = self.lua.globals();
        for entry in &self.entries {
            if entry.namespace.is_empty() {
                globals.set(entry.name, entry.function.clone())?;
            } else {
                let ns: LuaTable = match globals.get::<Option<LuaTable>>(entry.namespace)? {
                    Some(ns) => ns,
                    None => self.lua.create_table()?,
                };
                ns.set(entry.name, entry.function.clone())?;
                globals.set(entry.namespace, ns)?;
            }
        }
        Ok(())
    }

    /// The number of registered entries — useful for diagnostics/tests.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Iterate the registered entry names (for doc-gen / autocomplete).
    pub fn entry_names(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(|e| {
                if e.namespace.is_empty() {
                    e.name.to_string()
                } else {
                    format!("{}.{}", e.namespace, e.name)
                }
            })
            .collect()
    }
}

/// Convenience: mount a list of plugins into the registry, then apply.
pub fn mount_plugins(
    lua: &Lua,
    plugins: &[&dyn ScriptPlugin],
) -> LuaResult<()> {
    let mut registry = ApiRegistry::new(lua);
    for plugin in plugins {
        registry.mount_plugin(*plugin)?;
    }
    registry.apply()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_mounts_flat_and_namespaced() -> LuaResult<()> {
        let lua = Lua::new();
        let mut registry = ApiRegistry::new(&lua);

        registry
            .register_function("answer", |_, _: ()| Ok(42i32))?;
        registry
            .register_namespaced("mathx", "double", |_, (v,): (f32,)| Ok(v * 2.0))?;

        registry.apply()?;

        let globals = lua.globals();
        let answer: LuaFunction = globals.get("answer")?;
        assert_eq!(answer.call::<i32>(())?, 42);
        let mathx: LuaTable = globals.get("mathx")?;
        let double: LuaFunction = mathx.get("double")?;
        assert_eq!(double.call::<f32>((3.0,))?, 6.0);
        Ok(())
    }

    struct TestPlugin;

    impl ScriptPlugin for TestPlugin {
        fn name(&self) -> &'static str {
            "test"
        }

        fn register(&self, registry: &mut ApiRegistry) -> LuaResult<()> {
            registry.register_namespaced("test", "hello", |_, _: ()| Ok("world"))?;
            Ok(())
        }
    }

    #[test]
    fn plugin_pattern_registers_surface() -> LuaResult<()> {
        let lua = Lua::new();
        mount_plugins(&lua, &[&TestPlugin])?;
        let globals = lua.globals();
        let t: LuaTable = globals.get("test")?;
        let hello: LuaFunction = t.get("hello")?;
        assert_eq!(hello.call::<String>(())?, "world");
        Ok(())
    }
}
