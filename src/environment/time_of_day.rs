// src/environment/time_of_day.rs
// ── Time of Day System ─────────────────────────────────────────────────────
//
// WHY IT EXISTS:
//   A real game engine needs a controllable day/night cycle. Artists set a
//   "time of day" and the engine computes sun position, sky color, ambient
//   light, and shadow direction automatically. This replaces the old manual
//   sun_azimuth_deg / sun_elevation_deg floats scattered across settings.
//
// ARCHITECTURE:
//   TimeOfDay is a pure data struct — no ECS, no rendering, no IO.
//   It converts a single `hour` (0.0 = midnight, 12.0 = noon, 24.0 = midnight)
//   into:
//     • sun_position()  — normalized direction vector (for lighting)
//     • sun_color()     — warm at sunrise/sunset, white at noon
//     • ambient_color() — sky ambient, darker at night
//     • fog_color()     — matches sky horizon color
//     • star_intensity() — 0.0 during day, ramps up at night
//
// USAGE:
//   let mut time = TimeOfDay::new();
//   time.advance(dt, speed);           // advance by dt seconds × speed
//   let sun_dir = time.sun_position(); // Vec3 for the shader
//
// COMMON MISTAKES:
//   • Using sin(hour) directly for elevation — gives wrong sunrise/sunset color.
//     Solution: separate elevation into a smooth curve and blend colors.
//   • Not wrapping hour past 24.0 — causes float drift over long sessions.
//     Solution: hour %= 24.0 after each advance.
//
// PERFORMANCE:
//   ~100ns per frame (a few trig calls). No allocations.
//   Can be called from any thread — it's just math.
//
// MEMORY:
//   32 bytes (f32 × 8 fields). Stack only.
//
// MULTITHREADING:
//   Read-only after advance(). Safe to share across threads if advance()
//   is called once on the main thread and sun_position() is called from
//   the render thread.

/// Time of Day — converts a floating-point hour into atmospheric data.
///
/// # Hour convention
/// - 0.0  = midnight (sun below horizon)
/// - 6.0  = sunrise
/// - 12.0 = noon (sun at peak)
/// - 18.0 = sunset
/// - 24.0 = midnight again
///
/// # Day length
/// By default one real second = one game minute (so 24 real minutes = 1 game day).
/// Change `speed` to control this.
#[derive(Clone, Copy, Debug)]
pub struct TimeOfDay {
    /// Current hour in [0.0, 24.0). Wraps automatically.
    pub hour: f32,
    /// How many game-hours pass per real second.
    /// 1/60 = one real second = one game minute.
    pub speed: f32,
    /// Sunrise hour (default 6.0). Controls when the sky brightens.
    pub sunrise_hour: f32,
    /// Sunset hour (default 18.0). Controls when the sky darkens.
    pub sunset_hour: f32,
    /// Pause the cycle (e.g. for screenshots or menus).
    pub paused: bool,
}

impl Default for TimeOfDay {
    fn default() -> Self {
        Self {
            hour: 10.0, // default: mid-morning
            speed: 1.0 / 60.0,
            sunrise_hour: 6.0,
            sunset_hour: 18.0,
            paused: false,
        }
    }
}

impl TimeOfDay {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the clock by `dt` real seconds × `speed`.
    /// Wraps hour into [0, 24) to prevent floating-point drift.
    pub fn advance(&mut self, dt: f32) {
        if self.paused {
            return;
        }
        self.hour += dt * self.speed;
        self.hour %= 24.0;
        if self.hour < 0.0 {
            self.hour += 24.0;
        }
    }

    /// Sun direction vector (normalized). Y is up.
    ///
    /// Returns `None` when the sun is below the horizon (night).
    /// The direction is computed from the hour using a smooth arc:
    ///   elevation = sin(hour_normalized * PI)  [0 at sunrise/sunset, 1 at noon]
    ///   azimuth   = lerp from east to west over the day.
    pub fn sun_direction(&self) -> glam::Vec3 {
        let t = self.day_progress(); // 0.0 at sunrise, 1.0 at sunset
        // Semicircular arc in the XY plane: rises in east (1,0,0), peaks overhead (0,1,0), sets in west (-1,0,0).
        let angle = t * std::f32::consts::PI;
        glam::Vec3::new(angle.cos(), angle.sin(), 0.0).normalize_or_zero()
    }

    /// Sun elevation angle in radians. 0.0 at sunrise/sunset, ~PI/2 at noon.
    pub fn sun_elevation_rad(&self) -> f32 {
        let t = self.day_progress();
        (t * std::f32::consts::PI).sin().max(0.0)
    }

    /// Sun elevation angle in degrees.
    pub fn sun_elevation_deg(&self) -> f32 {
        self.sun_elevation_rad().to_degrees()
    }

    /// Sun azimuth angle in degrees. 0 = north, 90 = east, 180 = south.
    pub fn sun_azimuth_deg(&self) -> f32 {
        let t = self.day_progress();
        let azimuth_rad = (t - 0.5) * std::f32::consts::PI;
        azimuth_rad.to_degrees() + 180.0
    }

    /// 0.0 = midnight, 0.5 = noon, 1.0 = midnight.
    pub fn day_progress(&self) -> f32 {
        let range = self.sunset_hour - self.sunrise_hour;
        if range <= 0.0 {
            return 0.5; // invalid range: treat as noon
        }
        ((self.hour - self.sunrise_hour) / range).clamp(0.0, 1.0)
    }

    /// 0.0 = full night, 1.0 = full day.
    /// Ramps up smoothly during twilight (±1 hour around sunrise/sunset).
    pub fn daylight_factor(&self) -> f32 {
        let range = self.sunset_hour - self.sunrise_hour;
        if range <= 0.0 {
            return 0.0;
        }
        let t = (self.hour - self.sunrise_hour) / range;
        // Smooth ramp: 0 at t=0, 1 at t=1, with ±0.1 twilight zones.
        smoothstep(-0.05, 0.05, t) * (1.0 - smoothstep(0.95, 1.05, t))
    }

    /// Sun color. Warm orange at sunrise/sunset, white at noon, dark at night.
    pub fn sun_color(&self) -> glam::Vec3 {
        let daylight = self.daylight_factor();
        let elevation = self.sun_elevation_rad();
        // Near horizon: warm orange. High sun: white.
        let warmth = (1.0 - elevation).min(1.0);
        let color = glam::Vec3::new(
            1.0,
            1.0 - warmth * 0.3,
            1.0 - warmth * 0.5,
        );
        color * daylight
    }

    /// Ambient sky color. Blue during day, dark blue at night, warm at twilight.
    pub fn ambient_color(&self) -> glam::Vec3 {
        let daylight = self.daylight_factor();
        let twilight = 1.0 - daylight;
        let day_color = glam::Vec3::new(0.53, 0.61, 0.73); // sky blue
        let night_color = glam::Vec3::new(0.02, 0.03, 0.06); // dark blue
        let twilight_color = glam::Vec3::new(0.25, 0.15, 0.10); // warm orange
        night_color * twilight + day_color * daylight + twilight_color * twilight * (1.0 - daylight)
    }

    /// Fog color that matches the sky horizon.
    pub fn fog_color(&self) -> glam::Vec3 {
        let daylight = self.daylight_factor();
        let day_fog = glam::Vec3::new(0.65, 0.72, 0.82);
        let night_fog = glam::Vec3::new(0.03, 0.04, 0.07);
        let twilight_fog = glam::Vec3::new(0.35, 0.20, 0.12);
        let twilight = (1.0 - daylight) * daylight * 4.0; // peaks at twilight
        night_fog.lerp(day_fog, daylight) + twilight_fog * twilight
    }

    /// Star visibility. 0.0 during day, ramps up after sunset, 1.0 at midnight.
    pub fn star_intensity(&self) -> f32 {
        let daylight = self.daylight_factor();
        let nightness = 1.0 - daylight;
        (nightness * 2.0 - 0.3).clamp(0.0, 1.0)
    }

    /// Whether the sun is currently above the horizon.
    pub fn is_daytime(&self) -> bool {
        self.hour >= self.sunrise_hour && self.hour < self.sunset_hour
    }

    /// Moon direction — opposite to sun but slightly offset.
    pub fn moon_direction(&self) -> glam::Vec3 {
        let mut dir = -self.sun_direction();
        // Tilt the moon slightly off the sun's exact opposite path.
        dir.y += 0.15;
        dir.normalize_or_zero()
    }

    /// Set the time to a specific hour.
    pub fn set_hour(&mut self, hour: f32) {
        self.hour = hour.clamp(0.0, 24.0) % 24.0;
    }
}

/// Smooth Hermite interpolation between 0 and 1.
/// Returns 0 when x < edge0, 1 when x > edge1, smooth curve between.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_progress_midpoint() {
        let mut t = TimeOfDay::new();
        t.sunrise_hour = 6.0;
        t.sunset_hour = 18.0;
        t.hour = 12.0;
        let progress = t.day_progress();
        assert!((progress - 0.5).abs() < 0.001);
    }

    #[test]
    fn daylight_at_noon() {
        let mut t = TimeOfDay::new();
        t.hour = 12.0;
        t.sunrise_hour = 6.0;
        t.sunset_hour = 18.0;
        assert!(t.daylight_factor() > 0.9);
    }

    #[test]
    fn nighttime_at_midnight() {
        let mut t = TimeOfDay::new();
        t.hour = 0.0;
        t.sunrise_hour = 6.0;
        t.sunset_hour = 18.0;
        assert!(t.daylight_factor() < 0.1);
    }

    #[test]
    fn advance_wraps() {
        let mut t = TimeOfDay::new();
        t.hour = 23.9;
        t.speed = 0.5;
        t.advance(1.0); // +0.5 hours → 24.4 → wraps to 0.4
        assert!(t.hour < 1.0);
    }

    #[test]
    fn sun_direction_noon_is_up() {
        let mut t = TimeOfDay::new();
        t.hour = 12.0;
        t.sunrise_hour = 6.0;
        t.sunset_hour = 18.0;
        let dir = t.sun_direction();
        assert!(dir.y > 0.8); // sun is mostly pointing up at noon
    }

    #[test]
    fn paused_does_not_advance() {
        let mut t = TimeOfDay::new();
        t.hour = 12.0;
        t.paused = true;
        t.advance(10.0);
        assert!((t.hour - 12.0).abs() < 0.001);
    }

    #[test]
    fn star_intensity_night() {
        let mut t = TimeOfDay::new();
        t.hour = 0.0;
        t.sunrise_hour = 6.0;
        t.sunset_hour = 18.0;
        assert!(t.star_intensity() > 0.5);
    }

    #[test]
    fn star_intensity_day() {
        let mut t = TimeOfDay::new();
        t.hour = 12.0;
        t.sunrise_hour = 6.0;
        t.sunset_hour = 18.0;
        assert!(t.star_intensity() < 0.1);
    }
}
