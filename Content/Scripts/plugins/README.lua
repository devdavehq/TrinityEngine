-- Content/Scripts/plugins/README.lua
--
-- LUA-NATIVE PLUGINS
-- ==================
-- Everything in this directory is a plugin.  To add one, just drop a .lua file
-- here.  NO RUST RECOMPILATION IS NEEDED.  The engine's plugin host
-- (ScriptEngine::load_plugins) scans this folder at startup and hot-reloads any
-- file that changes while the game is running (same file watcher that reloads
-- entity scripts).
--
-- CONTRACT
-- --------
-- A plugin file must return a Lua table.  Every field is optional:
--
--   return {
--       name    = "my_plugin",            -- id (defaults to file name)
--       start   = function() ... end,     -- called once when loaded/reloaded
--       update  = function(dt) ... end,   -- called every frame
--       on_event = function(name, payload) ... end, -- fired by fire_event(...)
--   }
--
-- The plugin runs in its own environment whose __index is the global engine
-- API, so it can call everything scripts can: log(), spawn_mesh(),
-- set_timeout(), fire_event(), on_event(), bt.*, particles.*, navmesh.*,
-- set_weather(), audio_*, cinematic.*, save.*, ui.* and so on.
--
-- WHY PLUGINS (vs editing scripting.rs)
-- -------------------------------------
-- 1. Zero recompile: game/systems code lives in content, not the binary.
-- 2. Hot reload: edit a plugin file and it re-runs start() live.
-- 3. Isolation: each plugin has a fresh environment; a broken plugin fails
--    alone and is logged, never crashing the engine or other plugins.
-- 4. Composition: plugins communicate with each other and with entity scripts
--    through fire_event()/on_event().  E.g. the disaster plugin fires
--    "disaster.storm", and an audio plugin listens and plays a thunder sfx.
--
-- The plugins in this folder are a working default set.  Delete the ones you
-- don't want, or copy one to make your own.
--
-- (This file is itself a valid no-op plugin so the loader treats it the same
-- as any other — documentation you can safely delete.)

return {
    name = "plugins_readme",
}

