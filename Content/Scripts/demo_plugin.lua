-- Content/Scripts/demo_plugin.lua
-- Demonstrates the engine's formal ScriptPlugin extension pattern.
--
-- The `demo.*` namespace is provided by a Rust plugin (src/demo_plugin.rs)
-- mounted onto the Lua globals through ApiRegistry.  A game can ship its own
-- plugins the same way without ever editing scripting.rs.
--
-- Registration side (Rust, once at startup):
--   self.scripts.register_plugin(Box::new(demo_plugin::DemoPlugin::new()));
--
-- Try loading this file from the editor's script panel, or bind an entity
-- to it, and watch the log output.

local bumps = 0

function start(entity)
    log("demo_plugin.start: plugin demo ready for entity " .. entity)
    log("demo_plugin.start: " .. demo.greet("engine"))
end

function update(entity, dt)
    bumps = bumps + 1

    -- demo.bump tracks shared state inside the Rust plugin (counts spawns).
    -- demo.frame returns the engine-side frame counter, advanced by the
    -- engine each frame — proving the plugin keeps live state between calls.
    local total_bumps = demo.bump("player")
    local frame = demo.frame()
    log("demo_plugin.update: bumps=" .. bumps .. " total=" .. total_bumps
        .. " engine_frame=" .. frame)
end
