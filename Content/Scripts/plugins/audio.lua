-- Content/Scripts/plugins/audio.lua
-- Default plugin: audio director.
--
-- Pure-Lua plugin.  Listens for named events and turns them into audio cues,
-- so gameplay/other plugins just fire an event instead of calling audio APIs.
--
-- Listen for (and handle):
--   "audio.play_sfx"   payload = "path/to/sfx.wav"
--   "audio.play_music" payload = "path/to/music.wav"
--   "audio.set_volume" payload = "channel|volume"  (channel = music/sfx/ambient)
--   "audio.stop_all"   payload = nil
--
-- It also fades the music in on start.

local plugin = {
    name = "audio",
}

local music_started = false

function plugin.start()
    log("audio: plugin loaded")
    -- Kick off a quiet music bed if any music file exists (override below).
    -- audio_play_music("Content/Audio/theme.ogg", 0.6, true)
    music_started = false
end

function plugin.update()
    -- Re-fire any music loop requests that arrived before audio was ready.
    if not music_started and audio_is_music_playing() then
        music_started = true
    end
end

function plugin.on_event(name, payload)
    if name == "audio.play_sfx" and payload then
        audio_play_sfx(payload, 1.0, false)
    elseif name == "audio.play_music" and payload then
        audio_play_music(payload, 0.7, true)
        music_started = true
    elseif name == "audio.set_volume" and payload then
        local channel, volume = payload:match("^(%S+)%|([%d%.]+)$")
        if channel and volume then
            audio_set_volume(channel, tonumber(volume))
        end
    elseif name == "audio.stop_all" then
        audio_stop_all()
    end
end

return plugin
