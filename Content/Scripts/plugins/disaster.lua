-- Content/Scripts/plugins/disaster.lua
-- Default plugin: disaster / weather director.
--
-- WHY IS THIS A PLUGIN?
--   Disaster and weather are pure gameplay systems: they drive engine state
--   (set_weather, set_camera, fire_event, spawn_mesh) but contain zero engine
--   code.  Shipping them as a Lua plugin means:
--     • You can tune every number live (hot reload) without recompiling.
--     • The plugin can be deleted, renamed or replaced by another weather
--       behaviour without touching the binary.
--     • It composes with the other plugins through events: it fires
--       "disaster.storm" and the audio plugin plays thunder; it fires
--       "camera.shake" style events for hit reactions.
--
-- Trigger an event:
--   fire_event("disaster.storm", "3.0")   -- intensity 0..5 (default 2)
--   fire_event("disaster.clear")
--   fire_event("disaster.earthquake", "0.5")  -- seconds of shake
--
-- Or call directly:
--   disaster.storm(2.5)

local disaster = {}

local plugin = {
    name = "disaster",
}

local active_storm = false
local storm_intensity = 0.0
local storm_fade = 0.0 -- current intensity actually applied to weather
local earthquake_remaining = 0.0
local quake_strength = 0.0

local WEATHER = {
    clear = "clear",
    light = "lightrain",
    storm = "heavyrain",
}

local function clamp01(v) return math.max(0.0, math.min(1.0, v)) end

-- Apply the current storm fade value to engine weather + a wind vector.
local function apply_weather()
    local wind = storm_fade * 0.6
    set_weather(WEATHER.storm, storm_fade, 1.0, 0.3, wind)
end

function disaster.storm(intensity)
    storm_intensity = intensity or 2.0
    active_storm = true
    fire_event("audio.play_sfx", "Content/Audio/thunder.wav")
    log("disaster: storm starting (intensity " .. storm_intensity .. ")")
end

function disaster.clear()
    active_storm = false
    storm_fade = 0.0
    set_weather("clear", 0.0, 0.0, 0.0, 0.0)
    log("disaster: weather cleared")
end

function disaster.earthquake(seconds, strength)
    earthquake_remaining = seconds or 0.5
    quake_strength = strength or 1.0
    log("disaster: earthquake")
end

function plugin.start()
    log("disaster: plugin loaded")
end

function plugin.update(dt)
    -- Ease storm intensity in/out so transitions are smooth.
    if active_storm then
        storm_fade = storm_fade + (clamp01(storm_intensity / 5.0) - storm_fade) * dt * 2.0
    else
        storm_fade = storm_fade * math.max(0.0, 1.0 - dt * 4.0)
        if storm_fade < 0.001 then storm_fade = 0.0 end
    end
    apply_weather()

    -- Earthquake: apply camera shake for the requested duration.
    if earthquake_remaining > 0.0 then
        earthquake_remaining = earthquake_remaining - dt
        local px, py, pz, tx, ty, tz = get_camera()
        local s = quake_strength * (earthquake_remaining / 0.5)
        set_camera(
            px + (math.sin(elapsed_time() * 40.0)) * s * 0.5,
            py + (math.cos(elapsed_time() * 47.0)) * s * 0.3,
            pz + (math.sin(elapsed_time() * 33.0)) * s * 0.5,
            tx, ty, tz
        )
        if earthquake_remaining <= 0.0 then
            earthquake_remaining = 0.0
        end
    end
end

function plugin.on_event(name, payload)
    if name == "disaster.storm" then
        disaster.storm(payload and tonumber(payload) or 2.0)
    elseif name == "disaster.clear" then
        disaster.clear()
    elseif name == "disaster.earthquake" then
        local seconds = payload and tonumber(payload) or 0.5
        disaster.earthquake(seconds, 1.0)
    end
end

return plugin
