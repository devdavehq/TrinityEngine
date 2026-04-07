-- enemy.lua
-- This script controls enemy behaviour.
-- It is called every frame by the engine.
-- "dt" = delta time — seconds since last frame (we'll wire this up in Rust).
-- Using dt makes movement frame-rate independent:
--   speed * dt moves the same distance per second regardless of fps.

-- The enemy bounces left and right.
-- We track direction in a Lua global so it persists between frames.
local direction = 1.0
local speed     = 0.8  -- world units per second

function update(entity, dt)
    -- get_position() is a function the Rust engine exposes to Lua.
    -- It returns x, y, z as three separate values.
    local x, y, z = get_position(entity)

    -- Move horizontally.
    x = x + direction * speed * dt

    -- Reverse direction at the edges of the visible world.
    if x >  2.0 then direction = -1.0 end
    if x < -2.0 then direction =  1.0 end

    -- set_position() is another engine function exposed to Lua.
    -- We call it to actually apply the new position.
    set_position(entity, x, y, z)
end