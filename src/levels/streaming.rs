// src/levels/streaming.rs
// ──────────────────────────────────────────────────────────────────────────────
// Distance-based level streaming.
//
// The streaming system periodically checks the player's position against
// each non-persistent level's origin + streaming distances. When the player
// enters the streaming_distance, the level is queued for loading. When they
// leave the unloading_distance, it's queued for unloading.
//
// The gap between streaming_distance and unloading_distance creates a
// hysteresis band — this prevents levels from rapidly toggling on/off
// when the player is near the boundary (a classic streaming problem).
//
// Checks are throttled by check_interval to avoid per-frame overhead.
// The actual load/unload is performed by the caller (main loop) after
// receiving the StreamingResult.
// ──────────────────────────────────────────────────────────────────────────────

use crate::levels::LevelManager;

/// Configuration for the streaming check frequency.
/// Checks aren't run every frame — they're throttled to save CPU.
pub struct StreamingConfig {
    /// Check interval in seconds (e.g., 0.5 = check twice per second).
    pub check_interval: f32,
    /// Internal timer — resets each time a check runs.
    pub timer: f32,
}

impl StreamingConfig {
    pub fn new() -> Self {
        Self {
            check_interval: 0.5,
            timer: 0.0,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a streaming check — lists levels that need loading/unloading.
pub struct StreamingResult {
    /// Level IDs that should be loaded (player entered streaming_distance).
    pub levels_to_load: Vec<u32>,
    /// Level IDs that should be unloaded (player left unloading_distance).
    pub levels_to_unload: Vec<u32>,
}

/// Check which levels should be loaded/unloaded based on player position.
///
/// Returns None if it's not time to check yet (throttled by check_interval).
/// Returns Some(StreamingResult) when levels need to change state.
pub fn check_streaming(
    level_manager: &LevelManager,
    player_position: [f32; 3],
    dt: f32,
    config: &mut StreamingConfig,
) -> Option<StreamingResult> {
    // Throttle: only check periodically, not every frame.
    config.timer += dt;
    if config.timer < config.check_interval {
        return None;
    }
    config.timer = 0.0;

    let mut to_load = Vec::new();
    let mut to_unload = Vec::new();

    for level in &level_manager.levels {
        if level.persistent {
            continue; // Never stream the persistent level.
        }

        // Calculate distance from player to level origin.
        let dx = player_position[0] - level.origin[0];
        let dy = player_position[1] - level.origin[1];
        let dz = player_position[2] - level.origin[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if !level.loaded && dist <= level.streaming_distance {
            // Player is close enough — queue for loading.
            to_load.push(level.id);
        } else if level.loaded && dist > level.unloading_distance {
            // Player moved away — queue for unloading.
            to_unload.push(level.id);
        }
    }

    if to_load.is_empty() && to_unload.is_empty() {
        None
    } else {
        Some(StreamingResult {
            levels_to_load: to_load,
            levels_to_unload: to_unload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_load_trigger() {
        let mut mgr = LevelManager::new();
        // Level at origin (0,0,0) with streaming_distance=50, unloading_distance=100.
        let id = mgr.register_level("Nearby", "nearby.scene");
        mgr.get_mut(id).unwrap().streaming_distance = 50.0;
        mgr.get_mut(id).unwrap().unloading_distance = 100.0;

        let mut config = StreamingConfig::new();
        config.check_interval = 0.0; // Check every frame for test.

        // Player is 30 units away — within streaming distance.
        let result = check_streaming(&mgr, [30.0, 0.0, 0.0], 1.0, &mut config);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.levels_to_load, vec![id]);
        assert!(r.levels_to_unload.is_empty());
    }

    #[test]
    fn test_streaming_unload_trigger() {
        let mut mgr = LevelManager::new();
        let id = mgr.register_level("Distant", "distant.scene");
        mgr.get_mut(id).unwrap().streaming_distance = 50.0;
        mgr.get_mut(id).unwrap().unloading_distance = 100.0;
        mgr.load_level(id);

        let mut config = StreamingConfig::new();
        config.check_interval = 0.0;

        // Player is 120 units away — beyond unloading distance.
        let result = check_streaming(&mgr, [120.0, 0.0, 0.0], 1.0, &mut config);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.levels_to_load.is_empty());
        assert_eq!(r.levels_to_unload, vec![id]);
    }

    #[test]
    fn test_streaming_hysteresis_band() {
        let mut mgr = LevelManager::new();
        let id = mgr.register_level("Hysteresis", "hyst.scene");
        mgr.get_mut(id).unwrap().streaming_distance = 50.0;
        mgr.get_mut(id).unwrap().unloading_distance = 100.0;

        let mut config = StreamingConfig::new();
        config.check_interval = 0.0;

        // Player at 75 units — between streaming (50) and unloading (100).
        // Level is unloaded and player is beyond streaming_distance → no load.
        let result = check_streaming(&mgr, [75.0, 0.0, 0.0], 1.0, &mut config);
        assert!(result.is_none()); // Nothing to do.

        // Now load the level and put player at 75 — still within unloading_distance.
        mgr.load_level(id);
        let result = check_streaming(&mgr, [75.0, 0.0, 0.0], 1.0, &mut config);
        assert!(result.is_none()); // Don't unload either — hysteresis works!
    }

    #[test]
    fn test_streaming_skips_persistent() {
        let mut mgr = LevelManager::new();
        let id = mgr.register_level("Persistent", "main.scene");
        mgr.set_persistent(id);

        let mut config = StreamingConfig::new();
        config.check_interval = 0.0;

        // Even though player is far away, persistent level is never streamed.
        let result = check_streaming(&mgr, [9999.0, 0.0, 0.0], 1.0, &mut config);
        assert!(result.is_none());
    }

    #[test]
    fn test_streaming_throttle() {
        let mut mgr = LevelManager::new();
        let id = mgr.register_level("Test", "test.scene");
        mgr.get_mut(id).unwrap().streaming_distance = 10.0;

        let mut config = StreamingConfig::new();
        config.check_interval = 1.0;

        // Player is far from the level origin (500 units) — well outside
        // both streaming and unloading distances, so no state change needed.
        let far_pos = [500.0, 0.0, 0.0];

        // First call: dt=0.1, timer becomes 0.1 < 1.0 → skipped.
        let result = check_streaming(&mgr, far_pos, 0.1, &mut config);
        assert!(result.is_none());
        assert!((config.timer - 0.1).abs() < 0.001);

        // Second call: dt=0.1, timer becomes 0.2 < 1.0 → skipped.
        let result = check_streaming(&mgr, far_pos, 0.1, &mut config);
        assert!(result.is_none());
        assert!((config.timer - 0.2).abs() < 0.001);

        // Third call: dt=0.1, timer becomes 0.3 < 1.0 → skipped.
        let result = check_streaming(&mgr, far_pos, 0.1, &mut config);
        assert!(result.is_none());
        assert!((config.timer - 0.3).abs() < 0.001);

        // Fourth call: dt=1.0, timer becomes 1.3 >= 1.0 → check runs, timer resets to 0.0.
        let result = check_streaming(&mgr, far_pos, 1.0, &mut config);
        // Player is far, level unloaded → no levels to load or unload.
        assert!(result.is_none());
        // Timer was reset after the check ran.
        assert!(config.timer < 0.001);
    }
}
