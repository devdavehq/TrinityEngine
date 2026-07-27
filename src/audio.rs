// src/audio.rs
// Audio system for the engine.
// Uses rodio for cross-platform audio playback.
//
// ── Architecture ─────────────────────────────────────────────────────────────
// The AudioSystem owns:
//   - A rodio::OutputStream (the audio device handle)
// >   - Separate sinks for music and SFX channels
//   - Volume control per channel (master, music, sfx, ambient)
//
// Integration with the rest of the engine:
//   - main.rs creates AudioSystem at startup
//   - EventBus dispatches PlaySoundEvent, PlayMusicEvent, StopAudioEvent
//   - AudioSystem polls the EventBus each frame and plays/stops sounds
//   - Future: 3D positional audio via listener position + entity positions
//
// ── Usage from Lua scripts ───────────────────────────────────────────────────
//   audio_play_sfx("Content/Audio/door_open.wav")
//   audio_play_music("Content/Audio/ambient_forest.ogg", nil, true)
//   audio_stop_all()
//   audio_set_master_volume(0.8)

pub mod listener;
pub mod music;

use std::collections::HashMap;
use rodio::Source;

// ── Volume channels ──────────────────────────────────────────────────────────
/// Master volume control that affects all output.
/// Individual channels multiply against master.
#[derive(Clone, Copy, Debug)]
pub struct VolumeControl {
    pub master: f32,
    pub music: f32,
    pub sfx: f32,
    pub ambient: f32,
}

impl Default for VolumeControl {
    fn default() -> Self {
        Self {
            master: 0.8,
            music: 0.7,
            sfx: 1.0,
            ambient: 0.6,
        }
    }
}

impl VolumeControl {
    /// Effective volume for a channel (master * channel).
    pub fn effective(&self, channel: Channel) -> f32 {
        let channel_vol = match channel {
            Channel::Music => self.music,
            Channel::Sfx => self.sfx,
            Channel::Ambient => self.ambient,
        };
        (self.master * channel_vol).clamp(0.0, 1.0)
    }
}

/// Audio channel type — determines which volume knob controls it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    Music,
    Sfx,
    Ambient,
}

// ── Sound handle ─────────────────────────────────────────────────────────────
/// Handle to a playing sound. Can be used to stop or adjust volume.
#[derive(Clone, Debug)]
pub struct SoundHandle {
    id: u64,
    channel: Channel,
}

// ── Sound entry (internal) ───────────────────────────────────────────────────
struct SoundEntry {
    id: u64,
    channel: Channel,
    sink: rodio::Sink,
}

// ── AudioSystem ──────────────────────────────────────────────────────────────
/// Central audio system. Owns the output stream and manages all playback.
///
/// Thread safety: rodio's OutputStream must be created on the main thread.
/// Sinks are internally thread-safe in rodio.
pub struct AudioSystem {
    /// Rodio output stream — must stay alive for audio to play.
    _stream: rodio::OutputStream,
    /// Output stream handle for creating new sinks.
    stream_handle: rodio::OutputStreamHandle,
    /// Active sounds by ID.
    sounds: HashMap<u64, SoundEntry>,
    /// Music sink (separate for independent volume control).
    music_sink: Option<rodio::Sink>,
    /// Volume settings.
    pub volume: VolumeControl,
    /// Monotonically increasing ID counter.
    next_id: u64,
    /// 3D listener position for spatial audio.
    listener_position: [f32; 3],
    /// 3D listener forward direction (normalized).
    listener_forward: [f32; 3],
    /// 3D listener up vector.
    listener_up: [f32; 3],
    /// Spatial audio enabled flag.
    spatial_enabled: bool,
}

impl AudioSystem {
    /// Create the audio system. Call once at engine startup.
    ///
    /// Returns None if the audio device cannot be opened (headless/CI).
    pub fn new() -> Option<Self> {
        let (stream, stream_handle) = match rodio::OutputStream::try_default() {
            Ok(result) => result,
            Err(e) => {
                tracing::error!("[Audio] Failed to open audio output: {}", e);
                tracing::error!("[Audio] Running without audio.");
                return None;
            }
        };

        tracing::info!("[Audio] Audio system initialized.");

        Some(Self {
            _stream: stream,
            stream_handle,
            sounds: HashMap::new(),
            music_sink: None,
            volume: VolumeControl::default(),
            next_id: 1,
            listener_position: [0.0, 0.0, 0.0],
            listener_forward: [0.0, 0.0, -1.0],
            listener_up: [0.0, 1.0, 0.0],
            spatial_enabled: true,
        })
    }

    /// Play a sound effect. Returns a handle for stopping it.
    ///
    /// - `path`: File path to an audio file (wav, ogg, mp3, flac).
    /// - `volume`: Per-sound volume multiplier (0.0 = silent, 1.0 = full).
    /// - `looping`: Whether to loop the sound.
    pub fn play_sfx(
        &mut self,
        path: &str,
        volume: f32,
        looping: bool,
    ) -> Option<SoundHandle> {
        let sink = rodio::Sink::try_new(&self.stream_handle).ok()?;
        let effective_vol = self.volume.effective(Channel::Sfx) * volume;
        sink.set_volume(effective_vol);

        let source = Self::decode_file(path)?;
        if looping {
            sink.append(source.repeat_infinite());
        } else {
            sink.append(source);
        }

        let id = self.next_id;
        self.next_id += 1;

        self.sounds.insert(id, SoundEntry {
            id,
            channel: Channel::Sfx,
            sink,
        });

        Some(SoundHandle { id, channel: Channel::Sfx })
    }

    /// Play procedural thunder sound (layered rumble with crack and rumble layers).
    /// Uses rodio's built-in signal generators — no audio file needed.
    /// Tries to load a real thunder audio file first; falls back to procedural.
    pub fn play_thunder(&mut self, volume: f32) -> Option<SoundHandle> {
        use rodio::source::SineWave;

        let sink = rodio::Sink::try_new(&self.stream_handle).ok()?;
        let effective_vol = self.volume.effective(Channel::Sfx) * volume;
        sink.set_volume(effective_vol);

        // Try loading a real thunder file first (search common paths)
        let thunder_paths = [
            "assets/audio/thunder.wav",
            "assets/audio/thunder.ogg",
            "Content/Audio/thunder.wav",
            "Content/Audio/thunder.ogg",
        ];
        for path in &thunder_paths {
            if let Some(source) = Self::decode_file(path) {
                sink.append(source);
                let id = self.next_id;
                self.next_id += 1;
                self.sounds.insert(id, SoundEntry {
                    id,
                    channel: Channel::Sfx,
                    sink,
                });
                return Some(SoundHandle { id, channel: Channel::Sfx });
            }
        }

        // Fallback: layered procedural thunder
        // Layer 1: Deep rumble (40 Hz sine, 2s duration)
        let rumble = SineWave::new(40.0)
            .take_duration(std::time::Duration::from_secs_f32(2.0))
            .amplify(0.6);

        // Layer 2: Mid rumble (80 Hz sine, 1.5s, delayed start via take/skip)
        let mid_rumble = SineWave::new(80.0)
            .take_duration(std::time::Duration::from_secs_f32(1.5))
            .amplify(0.3);

        // Layer 3: High crack (200 Hz sine, 0.3s, for the initial crack sound)
        let crack = SineWave::new(200.0)
            .take_duration(std::time::Duration::from_secs_f32(0.3))
            .amplify(0.8);

        // Layer 4: Sub-bass (25 Hz sine, 2.5s for visceral feel)
        let sub = SineWave::new(25.0)
            .take_duration(std::time::Duration::from_secs_f32(2.5))
            .amplify(0.4);

        // Append crack first (immediate), then mix layers
        // rodio Sink::append chains sources sequentially, so we mix via amplitude envelope
        let layered = crack.mix(rumble).mix(mid_rumble).mix(sub);

        sink.append(layered);

        let id = self.next_id;
        self.next_id += 1;
        self.sounds.insert(id, SoundEntry {
            id,
            channel: Channel::Sfx,
            sink,
        });

        Some(SoundHandle { id, channel: Channel::Sfx })
    }

    /// Play background music. Replaces any currently playing music.
    ///
    /// - `path`: File path to an audio file.
    /// - `volume`: Music volume override (None = use volume.music).
    /// - `looping`: Whether to loop (default: true for music).
    pub fn play_music(
        &mut self,
        path: &str,
        volume: Option<f32>,
        looping: bool,
    ) -> Option<SoundHandle> {
        // Stop existing music.
        if let Some(old_sink) = self.music_sink.take() {
            old_sink.stop();
        }

        let sink = rodio::Sink::try_new(&self.stream_handle).ok()?;
        let vol = volume.unwrap_or(self.volume.music);
        sink.set_volume(self.volume.master * vol);

        let source = Self::decode_file(path)?;
        if looping {
            sink.append(source.repeat_infinite());
        } else {
            sink.append(source);
        }

        let id = self.next_id;
        self.next_id += 1;

        self.music_sink = Some(sink);

        Some(SoundHandle { id, channel: Channel::Music })
    }

    /// Stop a specific sound by handle.
    pub fn stop(&mut self, handle: SoundHandle) {
        if let Some(entry) = self.sounds.remove(&handle.id) {
            entry.sink.stop();
        }
        if handle.channel == Channel::Music {
            if let Some(sink) = self.music_sink.take() {
                sink.stop();
            }
        }
    }

    /// Stop all sounds on a specific channel.
    pub fn stop_channel(&mut self, channel: Channel) {
        match channel {
            Channel::Music => {
                if let Some(sink) = self.music_sink.take() {
                    sink.stop();
                }
            }
            _ => {
                self.sounds.retain(|_, entry| {
                    if entry.channel == channel {
                        entry.sink.stop();
                        false
                    } else {
                        true
                    }
                });
            }
        }
    }

    /// Stop all sounds.
    pub fn stop_all(&mut self) {
        // Stop all SFX sounds.
        for (_, entry) in self.sounds.drain() {
            entry.sink.stop();
        }
        // Stop music.
        if let Some(sink) = self.music_sink.take() {
            sink.stop();
        }
    }

    /// Set the master volume. Affects all channels proportionally.
    pub fn set_master_volume(&mut self, vol: f32) {
        self.volume.master = vol.clamp(0.0, 1.0);
    }

    /// Set volume for a specific channel.
    pub fn set_channel_volume(&mut self, channel: Channel, vol: f32) {
        match channel {
            Channel::Music => {
                self.volume.music = vol.clamp(0.0, 1.0);
                if let Some(sink) = &self.music_sink {
                    sink.set_volume(self.volume.effective(Channel::Music));
                }
            }
            Channel::Sfx => {
                self.volume.sfx = vol.clamp(0.0, 1.0);
            }
            Channel::Ambient => {
                self.volume.ambient = vol.clamp(0.0, 1.0);
            }
        }
    }

    /// Returns true if music is currently playing.
    pub fn is_music_playing(&self) -> bool {
        self.music_sink
            .as_ref()
            .map(|s| !s.empty())
            .unwrap_or(false)
    }

    /// Returns the number of active sounds.
    pub fn active_count(&self) -> usize {
        self.sounds.len()
    }

    /// Update the audio system. Call once per frame.
    /// Cleans up finished sounds.
    pub fn update(&mut self) {
        self.sounds.retain(|_, entry| !entry.sink.empty());
    }

    // ── 3D Spatial Audio ─────────────────────────────────────────────────────

    /// Update the listener position (camera position).
    pub fn set_listener_position(&mut self, pos: [f32; 3]) {
        self.listener_position = pos;
    }

    /// Update the listener forward direction (camera look direction).
    pub fn set_listener_forward(&mut self, forward: [f32; 3]) {
        self.listener_forward = forward;
    }

    /// Update the listener up vector.
    pub fn set_listener_up(&mut self, up: [f32; 3]) {
        self.listener_up = up;
    }

    /// Enable/disable spatial audio processing.
    pub fn set_spatial_enabled(&mut self, enabled: bool) {
        self.spatial_enabled = enabled;
    }

    /// Get current listener position.
    pub fn listener_position(&self) -> [f32; 3] {
        self.listener_position
    }

    /// Compute distance attenuation for a source at the given world position.
    /// Returns a volume multiplier (0.0 to 1.0).
    pub fn distance_attenuation(&self, source_pos: [f32; 3], min_distance: f32, max_distance: f32) -> f32 {
        if !self.spatial_enabled { return 1.0; }
        let dx = source_pos[0] - self.listener_position[0];
        let dy = source_pos[1] - self.listener_position[1];
        let dz = source_pos[2] - self.listener_position[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist <= min_distance { return 1.0; }
        if dist >= max_distance { return 0.0; }
        // Quadratic falloff
        let t = (dist - min_distance) / (max_distance - min_distance);
        1.0 - t * t
    }

    /// Compute stereo pan for a source (-1.0 left, 0.0 center, 1.0 right).
    pub fn stereo_pan(&self, source_pos: [f32; 3]) -> f32 {
        if !self.spatial_enabled { return 0.0; }
        let forward = glam::Vec3::from(self.listener_forward).normalize();
        let up = glam::Vec3::from(self.listener_up).normalize();
        let right = forward.cross(up).normalize();
        let to_source = glam::Vec3::new(
            source_pos[0] - self.listener_position[0],
            source_pos[1] - self.listener_position[1],
            source_pos[2] - self.listener_position[2],
        );
        let right_component = right.dot(to_source);
        let dist = to_source.length();
        if dist < 0.001 { return 0.0; }
        (right_component / dist).clamp(-1.0, 1.0)
    }

    /// Apply spatial audio to a sound source at the given world position.
    /// Adjusts volume based on distance and left/right balance.
    pub fn apply_spatial(&self, source_pos: [f32; 3], base_volume: f32) -> f32 {
        let attenuation = self.distance_attenuation(source_pos, 1.0, 100.0);
        base_volume * attenuation
    }

    // ── Internal ────────────────────────────────────────────────────────────

    /// Decode an audio file into a rodio Source.
    fn decode_file(path: &str) -> Option<rodio::Decoder<std::io::BufReader<std::fs::File>>> {
        let file = std::fs::File::open(path)
            .map_err(|e| {
                tracing::error!("[Audio] Cannot open {}: {}", path, e);
                e
            })
            .ok()?;

        rodio::Decoder::new(std::io::BufReader::new(file))
            .map_err(|e| {
                tracing::error!("[Audio] Cannot decode {}: {}", path, e);
                e
            })
            .ok()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_control_effective() {
        let vol = VolumeControl {
            master: 0.5,
            music: 0.8,
            sfx: 1.0,
            ambient: 0.4,
        };
        // Master * music = 0.5 * 0.8 = 0.4
        assert!((vol.effective(Channel::Music) - 0.4).abs() < 0.001);
        // Master * sfx = 0.5 * 1.0 = 0.5
        assert!((vol.effective(Channel::Sfx) - 0.5).abs() < 0.001);
        // Master * ambient = 0.5 * 0.4 = 0.2
        assert!((vol.effective(Channel::Ambient) - 0.2).abs() < 0.001);
    }

    #[test]
    fn volume_control_clamped() {
        let vol = VolumeControl {
            master: 1.5, // Over-max
            music: 1.0,
            sfx: 1.0,
            ambient: 1.0,
        };
        assert!((vol.effective(Channel::Music) - 1.0).abs() < 0.001);
    }

    #[test]
    fn audio_system_creation() {
        // AudioSystem::new() may return None in CI (no audio device).
        // We just test that it doesn't panic.
        let _sys = AudioSystem::new();
    }

    #[test]
    fn default_volumes() {
        let vol = VolumeControl::default();
        assert!((vol.master - 0.8).abs() < 0.001);
        assert!((vol.music - 0.7).abs() < 0.001);
        assert!((vol.sfx - 1.0).abs() < 0.001);
        assert!((vol.ambient - 0.6).abs() < 0.001);
    }

    #[test]
    fn channel_equality() {
        assert_eq!(Channel::Music, Channel::Music);
        assert_ne!(Channel::Music, Channel::Sfx);
    }
}
