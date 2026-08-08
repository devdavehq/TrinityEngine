-- Content/Scripts/plugins/camera.lua
-- Default plugin: camera director / cutscene helper.
--
-- Pure-Lua plugin that makes scripted camera moves and quick cutscenes
-- trivial: it owns a small timeline of keyframes and eases between them.
--
-- Exposed helpers:
--   camera.move_to(x, y, z, tx, ty, tz, duration, ease)
--       Schedules a smooth camera flight from the current camera to a new
--       position/target.  `ease` = "linear" | "ease_in" | "ease_out" |
--       "ease_in_out" (default).
--   camera.shake(strength, duration)     → quick handheld shake.
--   camera.cut(x, y, z, tx, ty, tz)      → snap instantly.
--
-- Listens for events:
--   "camera.cut"      payload = "x y z tx ty tz"
--   "camera.flight"   payload = "x y z tx ty tz duration ease"

local camera = {}

local plugin = {
    name = "camera",
}

local function lerp(a, b, t) return a + (b - a) * t end

local function ease(t, mode)
    if mode == "linear" then return t end
    if mode == "ease_in" then return t * t end
    if mode == "ease_out" then return 1 - (1 - t) * (1 - t) end
    if mode == "ease_in_out" then
        return t < 0.5 and 2 * t * t or 1 - ((-2 * t + 2) ^ 2) / 2
    end
    return t
end

local flight = nil -- { from, to, t, dur, mode }

function camera.cut(x, y, z, tx, ty, tz)
    set_camera(x, y, z, tx, ty, tz)
    flight = nil
end

function camera.move_to(x, y, z, tx, ty, tz, dur, mode)
    local px, py, pz, fx, fy, fz = get_camera()
    flight = {
        from = { px, py, pz, fx, fy, fz },
        to = { x, y, z, tx, ty, tz },
        t = 0.0,
        dur = dur or 2.0,
        mode = mode or "ease_in_out",
    }
end

function camera.shake(strength, duration)
    local px, py, pz, fx, fy, fz = get_camera()
    flight = {
        from = { px, py, pz, fx, fy, fz },
        to = { px + strength, py - strength * 0.5, pz + strength, fx, fy, fz },
        t = 0.0,
        dur = duration or 0.4,
        mode = "linear",
    }
end

function plugin.start()
    log("camera: plugin loaded")
end

function plugin.update(dt)
    if flight then
        flight.t = flight.t + dt
        local k = flight.t / flight.dur
        local f = flight.from
        local t = flight.to
        if k >= 1.0 then
            set_camera(t[1], t[2], t[3], t[4], t[5], t[6])
            flight = nil
        else
            local e = ease(k, flight.mode)
            set_camera(
                lerp(f[1], t[1], e), lerp(f[2], t[2], e), lerp(f[3], t[3], e),
                lerp(f[4], t[4], e), lerp(f[5], t[5], e), lerp(f[6], t[6], e)
            )
        end
    end
end

function plugin.on_event(name, payload)
    if name == "camera.cut" and payload then
        local x, y, z, tx, ty, tz = payload:match("^(-?[%d%.]+) (-?[%d%.]+) (-?[%d%.]+) (-?[%d%.]+) (-?[%d%.]+) (-?[%d%.]+)$")
        if x then camera.cut(tonumber(x), tonumber(y), tonumber(z), tonumber(tx), tonumber(ty), tonumber(tz)) end
    elseif name == "camera.flight" and payload then
        local x, y, z, tx, ty, tz, dur, mode = payload:match("^(-?[%d%.]+) (-?[%d%.]+) (-?[%d%.]+) (-?[%d%.]+) (-?[%d%.]+) (-?[%d%.]+) (-?[%d%.]+) (%S+)$")
        if x then camera.move_to(tonumber(x), tonumber(y), tonumber(z), tonumber(tx), tonumber(ty), tonumber(tz), tonumber(dur), mode) end
    end
end

return plugin
