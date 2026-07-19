// src/core/events.rs
// ──────────────────────────────────────────────────────────────────────────────
// All engine event types live here.
//
// WHY a dedicated file:
//   When you want to know "what can happen in this engine?", you open ONE file.
//   Every system emits events defined here. Every system subscribes to events
//   defined here. This is the single source of truth for inter-system communication.
//
// RULES FOR ADDING EVENTS:
//   1. Keep structs flat. No nested heap allocations in hot events.
//   2. Derive Clone + Copy where possible (for profiling, hot-path events).
//   3. Name them {Noun}{Verb} or {Noun}{PastVerb}: EntityCreated, SceneLoaded.
//   4. If an event fires every frame, mark it with a comment "// HOT PATH".
//   5. Never put system-specific types (wgpu handles, Lua states) in events.
//      Use IDs, paths, or indices instead.
// ──────────────────────────────────────────────────────────────────────────────

// ══════════════════════════════════════════════════════════════════════════════
// APP LIFECYCLE
// ══════════════════════════════════════════════════════════════════════════════

/// Engine is starting up. Subsystems should initialize here.
#[derive(Debug, Clone)]
pub struct StartupEvent;

/// Engine is shutting down. Subsystems should clean up here.
#[derive(Debug, Clone)]
pub struct ShutdownEvent;

/// A new frame is beginning. Reset per-frame counters here.
#[derive(Debug, Clone, Copy)]
pub struct BeginFrameEvent {
    pub frame_index: u64,
    pub delta_time: f32,
}

/// Frame is done. Swap buffers, present, etc.
#[derive(Debug, Clone, Copy)]
pub struct EndFrameEvent {
    pub frame_index: u64,
}

// ══════════════════════════════════════════════════════════════════════════════
// SCENE
// ══════════════════════════════════════════════════════════════════════════════

/// A scene file has finished loading.
#[derive(Debug, Clone)]
pub struct SceneLoadedEvent {
    pub path: String,
}

/// A scene is about to be unloaded.
#[derive(Debug, Clone)]
pub struct SceneUnloadedEvent;

/// Scene file changed on disk (hot-reload trigger).
#[derive(Debug, Clone)]
pub struct SceneModifiedEvent {
    pub path: String,
}

// ══════════════════════════════════════════════════════════════════════════════
// ENTITY
// ══════════════════════════════════════════════════════════════════════════════

/// Entity was spawned. The u64 is the entity's generational ID bits.
#[derive(Debug, Clone, Copy)]
pub struct EntityCreatedEvent {
    pub entity_bits: u64,
}

/// Entity was destroyed.
#[derive(Debug, Clone, Copy)]
pub struct EntityDestroyedEvent {
    pub entity_bits: u64,
}

/// Entity's transform changed (position/rotation/scale).
/// HOT PATH — only emit when something actually moves.
#[derive(Debug, Clone, Copy)]
pub struct EntityMovedEvent {
    pub entity_bits: u64,
}

// ══════════════════════════════════════════════════════════════════════════════
// PHYSICS
// ══════════════════════════════════════════════════════════════════════════════

/// Two entities just started touching.
#[derive(Debug, Clone, Copy)]
pub struct CollisionStartedEvent {
    pub entity_a_bits: u64,
    pub entity_b_bits: u64,
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
    pub penetration: f32,
}

/// Two entities stopped touching.
#[derive(Debug, Clone, Copy)]
pub struct CollisionEndedEvent {
    pub entity_a_bits: u64,
    pub entity_b_bits: u64,
}

// ══════════════════════════════════════════════════════════════════════════════
// RENDERING
// ══════════════════════════════════════════════════════════════════════════════

/// A render feature was toggled on/off (shadows, bloom, SSAO, etc).
#[derive(Debug, Clone)]
pub struct RenderFeatureToggledEvent {
    pub feature_name: String,
    pub enabled: bool,
}

/// The GPU quality tier changed (low -> balanced -> high -> cinematic).
#[derive(Debug, Clone)]
pub struct QualityTierChangedEvent {
    pub old_tier: String,
    pub new_tier: String,
}

/// A shader was hot-reloaded.
#[derive(Debug, Clone)]
pub struct ShaderReloadedEvent {
    pub shader_path: String,
    pub success: bool,
    pub error_message: Option<String>,
}

// ══════════════════════════════════════════════════════════════════════════════
// ASSETS
// ══════════════════════════════════════════════════════════════════════════════

/// An asset finished loading.
#[derive(Debug, Clone)]
pub struct AssetLoadedEvent {
    pub path: String,
    pub asset_type: String,
}

/// An asset was hot-reloaded from disk.
#[derive(Debug, Clone)]
pub struct AssetHotReloadedEvent {
    pub path: String,
}

/// An asset failed to load.
#[derive(Debug, Clone)]
pub struct AssetErrorEvent {
    pub path: String,
    pub error: String,
}

// ══════════════════════════════════════════════════════════════════════════════
// SCRIPTING
// ══════════════════════════════════════════════════════════════════════════════

/// A Lua script was hot-reloaded.
#[derive(Debug, Clone)]
pub struct ScriptReloadedEvent {
    pub script_path: String,
    pub success: bool,
}

/// A Lua script encountered an error.
#[derive(Debug, Clone)]
pub struct ScriptErrorEvent {
    pub script_path: String,
    pub error: String,
}

// ══════════════════════════════════════════════════════════════════════════════
// INPUT
// ══════════════════════════════════════════════════════════════════════════════

/// A key was pressed.
#[derive(Debug, Clone, Copy)]
pub struct KeyPressedEvent {
    pub key_code: u32,
}

/// A key was released.
#[derive(Debug, Clone, Copy)]
pub struct KeyReleasedEvent {
    pub key_code: u32,
}

/// The mouse moved.
#[derive(Debug, Clone, Copy)]
pub struct MouseMovedEvent {
    pub x: f64,
    pub y: f64,
}

/// A mouse button was pressed.
#[derive(Debug, Clone, Copy)]
pub struct MouseButtonPressedEvent {
    pub button: u32,
}

/// A mouse button was released.
#[derive(Debug, Clone, Copy)]
pub struct MouseButtonReleasedEvent {
    pub button: u32,
}

// ══════════════════════════════════════════════════════════════════════════════
// ENVIRONMENT (for future Environment System)
// ══════════════════════════════════════════════════════════════════════════════

/// Time of day changed (0.0 = midnight, 0.5 = noon, 1.0 = midnight).
#[derive(Debug, Clone, Copy)]
pub struct TimeOfDayChangedEvent {
    pub time: f32,
}

/// Weather state changed.
#[derive(Debug, Clone)]
pub struct WeatherChangedEvent {
    pub weather_type: String,
    pub intensity: f32,
}

/// An entity entered a water surface.
#[derive(Debug, Clone, Copy)]
pub struct WaterSplashEvent {
    pub entity_bits: u64,
    pub water_entity_bits: u64,
    pub impact_velocity: f32,
    pub splash_intensity: f32,
}

// ══════════════════════════════════════════════════════════════════════════════
// AUDIO (for future Audio System)
// ══════════════════════════════════════════════════════════════════════════════

/// Request to play a sound effect.
#[derive(Debug, Clone)]
pub struct PlaySoundEvent {
    pub sound_path: String,
    pub volume: f32,
    pub entity_bits: Option<u64>, // None = non-positional, Some = 3D at entity
}

/// Request to start background music.
#[derive(Debug, Clone)]
pub struct PlayMusicEvent {
    pub music_path: String,
    pub volume: f32,
    pub fade_in_seconds: f32,
}

/// Stop all audio.
#[derive(Debug, Clone)]
pub struct StopAudioEvent;

/// Lightning bolt struck — play thunder sound.
#[derive(Debug, Clone, Copy)]
pub struct ThunderEvent {
    pub intensity: f32,
    pub delay: f32,
}

// ══════════════════════════════════════════════════════════════════════════════
// UI / EDITOR
// ══════════════════════════════════════════════════════════════════════════════

/// Log a message to the editor console.
#[derive(Debug, Clone)]
pub struct ConsoleLogEvent {
    pub level: LogLevel,
    pub message: String,
    pub source: String, // "script", "physics", "editor", etc.
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warning,
    Error,
    Debug,
}

/// User wants to focus the camera on an entity.
#[derive(Debug, Clone, Copy)]
pub struct FocusEntityEvent {
    pub entity_bits: u64,
}

/// Play mode was entered or exited.
#[derive(Debug, Clone, Copy)]
pub struct PlayModeChangedEvent {
    pub playing: bool,
}
