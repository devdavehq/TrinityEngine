// src/environment/lightning.rs
// ── Lightning System ────────────────────────────────────────────────────────
//
// WHY IT EXISTS:
//   Storms need lightning. This system generates lightning bolt events that
//   the renderer uses for screen flashes and the audio system uses for thunder.
//
// ARCHITECTURE:
//   LightningState is a pure data struct. It tracks flash timing and intensity.
//   The renderer reads lightning.flash_intensity to overlay a white screen flash.
//   The sky shader reads lightning.flash_intensity for cloud illumination.
//   Thunder is triggered via the event bus after a configurable delay.
//
// DATA FLOW:
//   WeatherState → LightningState → Renderer (screen flash, cloud illumination)
//                                  → Audio (thunder sound after delay)
//
// PERFORMANCE:
//   ~5ns per frame (timer math only).
//
// MEMORY:
//   ~32 bytes. Stack only.

use crate::environment::weather::{WeatherCondition, WeatherState};

/// Lightning system state. Updated each frame, read by renderer and audio.
#[derive(Clone, Debug)]
pub struct LightningState {
    /// Current flash intensity (0 = no flash, 1 = full white overlay).
    /// Decays exponentially after a bolt fires.
    pub flash_intensity: f32,
    /// Time since last bolt (seconds). Used for random interval.
    pub time_since_bolt: f32,
    /// Minimum interval between bolts (seconds).
    pub min_interval: f32,
    /// Maximum interval between bolts (seconds).
    pub max_interval: f32,
    /// Countdown to next bolt (decrements each frame).
    pub next_bolt_timer: f32,
    /// Flash decay rate (higher = faster fade). Controls how long the flash lasts.
    pub flash_decay_rate: f32,
    /// Cloud illumination intensity (read by sky shader).
    /// Separate from flash: clouds light up slightly even between visible flashes.
    pub cloud_illumination: f32,
    /// Whether a thunder event should fire (consumed by audio system).
    pub thunder_pending: bool,
    /// Thunder delay in seconds (distance-based: light arrives before sound).
    pub thunder_delay: f32,
    /// Accumulated thunder delay timer.
    pub thunder_timer: f32,
    /// True for exactly one frame when a bolt fires (cleared at end of update).
    pub thunder_just_fired: bool,
    /// Where the bolt starts (cloud position).
    pub bolt_origin: [f32; 3],
    /// Where the bolt strikes (ground position).
    pub bolt_target: [f32; 3],
    /// Normalized direction from origin to target.
    pub bolt_direction: [f32; 3],
    /// Distance from origin to target.
    pub bolt_distance: f32,
}

impl Default for LightningState {
    fn default() -> Self {
        Self {
            flash_intensity: 0.0,
            time_since_bolt: 0.0,
            min_interval: 3.0,
            max_interval: 15.0,
            next_bolt_timer: 5.0,
            flash_decay_rate: 8.0,
            cloud_illumination: 0.0,
            thunder_pending: false,
            thunder_delay: 2.0,
            thunder_timer: 0.0,
            thunder_just_fired: false,
            bolt_origin: [0.0; 3],
            bolt_target: [0.0; 3],
            bolt_direction: [0.0, -1.0, 0.0],
            bolt_distance: 80.0,
        }
    }
}

impl LightningState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update lightning state. Call once per frame.
    /// `dt` is frame time in seconds.
    pub fn update(&mut self, weather: &WeatherState, dt: f32) {
        // Only active during storms.
        let is_storm = weather.condition == WeatherCondition::Storm;
        let storm_intensity = if is_storm { weather.intensity } else { 0.0 };

        // Decay flash intensity exponentially.
        if self.flash_intensity > 0.001 {
            self.flash_intensity *= (-self.flash_decay_rate * dt).exp();
        } else {
            self.flash_intensity = 0.0;
        }

        // Decay cloud illumination (slower than flash).
        if self.cloud_illumination > 0.001 {
            self.cloud_illumination *= (-3.0 * dt).exp();
        } else {
            self.cloud_illumination = 0.0;
        }

        // Tick thunder delay.
        if self.thunder_pending {
            self.thunder_timer -= dt;
            if self.thunder_timer <= 0.0 {
                self.thunder_pending = false; // consumed
            }
        }

        if storm_intensity < 0.3 {
            // Not stormy enough for lightning.
            self.time_since_bolt = 0.0;
            self.next_bolt_timer = self.max_interval;
            return;
        }

        // Count time and fire bolts.
        self.time_since_bolt += dt;
        self.next_bolt_timer -= dt;

        if self.next_bolt_timer <= 0.0 {
            self.fire_bolt(storm_intensity);
        }

        // Clear one-shot flag at end of frame so readers see it for exactly one frame.
        self.thunder_just_fired = false;
    }

    /// Fire a lightning bolt. Sets flash intensity and schedules thunder.
    fn fire_bolt(&mut self, storm_intensity: f32) {
        // Flash intensity scales with storm intensity.
        self.flash_intensity = 0.6 + 0.4 * storm_intensity;

        // Cloud illumination persists longer than the visible flash.
        self.cloud_illumination = 0.3 + 0.4 * storm_intensity;

        // Schedule thunder after a delay (simulates sound travel time).
        self.thunder_pending = true;
        self.thunder_timer = self.thunder_delay + (1.0 - storm_intensity) * 2.0;
        self.thunder_just_fired = true;

        // Compute bolt spatial info for the renderer.
        // Pseudo-random from timer (good enough for visual variety).
        let t = self.time_since_bolt + self.next_bolt_timer;
        let rand_x = ((t * 127.1).sin() * 43758.5453).fract();
        let rand_z = ((t * 269.5).sin() * 43758.5453).fract();
        let rx = (rand_x - 0.5) * 100.0;
        let rz = (rand_z - 0.5) * 100.0;
        self.bolt_origin = [rx, 80.0, rz];
        self.bolt_target = [rx + (rand_x - 0.5) * 20.0, 0.0, rz + (rand_z - 0.5) * 20.0];
        let dx = self.bolt_target[0] - self.bolt_origin[0];
        let dy = self.bolt_target[1] - self.bolt_origin[1];
        let dz = self.bolt_target[2] - self.bolt_origin[2];
        self.bolt_distance = (dx * dx + dy * dy + dz * dz).sqrt();
        let inv_len = 1.0 / self.bolt_distance.max(0.001);
        self.bolt_direction = [dx * inv_len, dy * inv_len, dz * inv_len];

        // Reset timer with randomized interval (shorter during intense storms).
        self.time_since_bolt = 0.0;
        let range = self.max_interval - self.min_interval;
        let intensity_factor = 1.0 - storm_intensity * 0.6; // intense storms = shorter intervals
        self.next_bolt_timer = self.min_interval + range * intensity_factor * 0.5
            + self.next_bolt_timer * 0.3; // some randomness from leftover timer
    }

    /// GPU uniform data for the sky shader.
    /// Packs into cloud_type.yz of the sky uniform (CloudUniformData).
    pub fn cloud_uniform_contribution(&self) -> [f32; 2] {
        // storm_darken = cloud_illumination (0-1, used to darken cloud bases).
        // lightning = flash_intensity (0-1, used to illuminate clouds during flash).
        [self.cloud_illumination, self.flash_intensity]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::weather::WeatherState;

    #[test]
    fn no_lightning_when_clear() {
        let mut lightning = LightningState::new();
        let weather = WeatherState::clear();
        lightning.update(&weather, 1.0);
        assert_eq!(lightning.flash_intensity, 0.0);
        assert!(!lightning.thunder_pending);
    }

    #[test]
    fn storm_triggers_lightning() {
        let mut lightning = LightningState::new();
        let weather = WeatherState::stormy();
        // Force bolt timer to fire immediately.
        lightning.next_bolt_timer = 0.0;
        lightning.update(&weather, 0.016);
        assert!(lightning.flash_intensity > 0.0);
        assert!(lightning.thunder_pending);
    }

    #[test]
    fn flash_decays() {
        let mut lightning = LightningState::new();
        lightning.flash_intensity = 1.0;
        let weather = WeatherState::stormy();
        lightning.update(&weather, 0.5);
        assert!(lightning.flash_intensity < 0.1);
    }

    #[test]
    fn thunder_timer_counts_down() {
        let mut lightning = LightningState::new();
        lightning.thunder_pending = true;
        lightning.thunder_timer = 1.0;
        let weather = WeatherState::stormy();
        lightning.update(&weather, 0.5);
        assert!(lightning.thunder_timer < 1.0);
        lightning.update(&weather, 0.6);
        assert!(!lightning.thunder_pending); // consumed after timer expires
    }
}
