-- Content/Scripts/plugins/input.lua
-- Default plugin: input remapping + gamepad helpers.
--
-- This is a pure-Lua plugin — no engine recompile needed.  It layers a small
-- convenience API over the engine's flat input globals (is_key_held,
-- gamepad_*) and re-broadcasts raw input as named events so other plugins and
-- entity scripts can react without polling.
--
-- Events fired by this plugin:
--   "input.key_down"    payload = key name (string)
--   "input.gamepad_a"   payload = nil (A button just pressed)
--
-- Exposed helpers (call from any script; the plugin's env falls through to
-- the global API, and these are just functions on the `input` table it owns):
--   input.pressed("KeyW")        → true while held
--   input.axis()                 → (lx, ly) gamepad left stick + movement
--   input.forward_movement()     → -1/0/1 from WASD
--   input.strafe_movement()      → -1/0/1 from AD

local input = {}

local keys = {
    forward  = "KeyW",
    backward = "KeyS",
    left     = "KeyA",
    right    = "KeyD",
    jump     = "Space",
}

local last_a = false

function input.pressed(key)
    return is_key_held(key)
end

function input.axis()
    local lx = gamepad_left_x()
    local ly = gamepad_left_y()
    -- Blend gamepad into WASD so keyboard-only play still works.
    local kx = 0
    if is_key_held(keys.left) then kx = kx - 1 end
    if is_key_held(keys.right) then kx = kx + 1 end
    local ky = 0
    if is_key_held(keys.forward) then ky = ky + 1 end
    if is_key_held(keys.backward) then ky = ky - 1 end
    return kx + lx, ky + ly
end

function input.forward_movement()
    local _, y = input.axis()
    return y
end

function input.strafe_movement()
    local x = input.axis()
    return x
end

local plugin = {
    name = "input",
}

function plugin.start()
    log("input: plugin loaded (WASD + gamepad helpers)")
end

function plugin.update()
    local a = gamepad_button_pressed("a")
    if a and not last_a then
        fire_event("input.gamepad_a")
    end
    last_a = a
end

return plugin
