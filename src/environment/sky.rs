// src/environment/sky.rs
// ── Procedural Sky System ──────────────────────────────────────────────────
//
// WHY IT EXISTS:
//   Games need a sky. Loading an HDR cubemap works but wastes memory and
//   doesn't adapt to time-of-day changes. A procedural sky generates a sky
//   gradient from a handful of parameters — zero texture cost, infinitely
//   adjustable, and artist-friendly via the editor.
//
// ARCHITECTURE:
//   SkyParams is a pure data struct (no rendering code). The renderer reads
//   SkyParams to generate the sky (either via a fullscreen quad or a
//   pre-computed cubemap). This separation means:
//     • SkyParams can be serialized (save/load scenes)
//     • SkyParams can be hot-edited in the inspector
//     • The renderer can switch between procedural and HDR sky without
//       touching the environment system
//
// DATA FLOW:
//   TimeOfDay → SkyParams → Renderer (sky shader)
//                         ↘ EnvironmentEvent (for editor sync)
//
// COMMON MISTAKES:
//   • Computing sky colors in the fragment shader every frame — expensive
//     for mobile. Solution: pre-compute a 1D gradient texture at startup
//     and sample it by elevation angle.
//   • Not accounting for Rayleigh scattering at sunset — sky should turn
//     orange/red, not just dark blue. Solution: blend toward horizon color
//     as sun approaches the horizon.
//
// PERFORMANCE:
//   SkyParams update is ~50ns (just setting fields). The actual sky rendering
//   is the renderer's responsibility and can be a fullscreen quad (cheap) or
//   a ray-marched volume (expensive).
//
// MEMORY:
//   ~100 bytes (f32 × ~24 fields). Stack only.
//
// MULTITHREADING:
//   Read-only after update. Safe to share across threads.

use glam::Vec3;

/// Parameters for procedural sky rendering.
///
/// The renderer reads these each frame to produce the sky image.
/// Artists can tweak these via the inspector or from scripts.
#[derive(Clone, Debug)]
pub struct SkyParams {
    // ── Sky gradient ────────────────────────────────────────────────────
    /// Zenith color (straight up). Deep blue during day, dark at night.
    pub zenith_color: Vec3,
    /// Horizon color (at 0° elevation). Lighter blue during day, warm at sunset.
    pub horizon_color: Vec3,
    /// Ground color (below horizon). Dark earth tone.
    pub ground_color: Vec3,

    // ── Sun disc ────────────────────────────────────────────────────────
    /// Sun direction (normalized). Set by TimeOfDay.
    pub sun_direction: Vec3,
    /// Sun disc color. White at noon, warm at sunset.
    pub sun_color: Vec3,
    /// Sun disc intensity. Higher = brighter disc halo.
    pub sun_intensity: f32,
    /// Sun disc angular radius in degrees (~0.25° real sun, ~1-3° for stylized).
    pub sun_disc_radius_deg: f32,
    /// Sun halo/falloff sharpness. Higher = sharper disc edge.
    pub sun_halo_falloff: f32,

    // ── Moon ────────────────────────────────────────────────────────────
    pub moon_direction: Vec3,
    pub moon_color: Vec3,
    pub moon_intensity: f32,
    pub moon_radius_deg: f32,
    /// Master toggle for the moon disc.
    pub moon_enabled: bool,
    /// When true, `update_from_time` re-derives moon position + brightness from
    /// the clock each frame. When false (script/editor took manual control),
    /// the stored `moon_direction` / `moon_intensity` are left untouched.
    pub moon_auto: bool,

    // ── Stars ───────────────────────────────────────────────────────────
    /// Star brightness (0 = invisible, 1 = full brightness).
    pub star_intensity: f32,
    /// Star density (0 = none, 1 = many).
    pub star_density: f32,
    /// Master toggle for the entire star field (Milky Way + individual stars).
    pub stars_enabled: bool,
    /// When true, `update_from_time` re-derives star intensity from the clock
    /// each frame. When false (script/editor took manual control), the stored
    /// `star_intensity` is left untouched.
    pub stars_auto: bool,

    // ── Atmosphere ──────────────────────────────────────────────────────
    /// Rayleigh scattering coefficient. Controls how much blue light scatters.
    pub rayleigh_scatter: Vec3,
    /// Rayleigh scattering density. Higher = more scattering.
    pub rayleigh_density: f32,
    /// Mie scattering coefficient. Controls haze/fog around the sun.
    pub mie_scatter: f32,
    /// Mie scattering density. Higher = more haze.
    pub mie_density: f32,
    /// Mie scattering direction (asymmetry). Range [-1, 1]. 0.76 = forward scatter.
    pub mie_direction: f32,

    // ── Environment map ─────────────────────────────────────────────────
    /// Optional path to an HDR sky map (.hdr). When set, this overrides
    /// procedural sky for IBL (image-based lighting).
    pub hdr_sky_path: String,
}

impl Default for SkyParams {
    fn default() -> Self {
        Self {
            zenith_color: Vec3::new(0.15, 0.30, 0.65),
            horizon_color: Vec3::new(0.55, 0.65, 0.78),
            ground_color: Vec3::new(0.05, 0.06, 0.04),

            sun_direction: Vec3::new(0.5, 0.7, 0.3).normalize(),
            sun_color: Vec3::ONE,
            sun_intensity: 1.0,
            sun_disc_radius_deg: 0.5,
            sun_halo_falloff: 800.0,

            moon_direction: Vec3::new(-0.5, 0.3, -0.3).normalize(),
            moon_color: Vec3::new(0.7, 0.75, 0.85),
            moon_intensity: 0.15,
            moon_radius_deg: 0.3,
            moon_enabled: true,
            moon_auto: true,

            star_intensity: 0.0,
            star_density: 0.8,
            stars_enabled: true,
            stars_auto: true,

            rayleigh_scatter: Vec3::new(5.5e-6, 13.0e-6, 22.4e-6),
            rayleigh_density: 1.0,
            mie_scatter: 21.0e-6,
            mie_density: 1.0,
            mie_direction: 0.76,

            hdr_sky_path: String::new(),
        }
    }
}

impl SkyParams {
    /// Update sky parameters from the current time of day.
    /// Call once per frame after TimeOfDay::advance().
    pub fn update_from_time(&mut self, time: &super::time_of_day::TimeOfDay) {
        let daylight = time.daylight_factor();
        let elevation = time.sun_elevation_rad();
        let twilight = (1.0 - daylight) * daylight * 4.0; // peaks during twilight

        // ── Sky gradient ────────────────────────────────────────────────
        let day_zenith = Vec3::new(0.15, 0.30, 0.65);
        let night_zenith = Vec3::new(0.01, 0.01, 0.03);
        let twilight_zenith = Vec3::new(0.12, 0.10, 0.25);

        let day_horizon = Vec3::new(0.55, 0.65, 0.78);
        let night_horizon = Vec3::new(0.03, 0.04, 0.06);
        let sunset_horizon = Vec3::new(0.85, 0.45, 0.15);

        self.zenith_color = night_zenith.lerp(day_zenith, daylight)
            + twilight_zenith * twilight * 0.3;
        self.horizon_color = night_horizon.lerp(day_horizon, daylight)
            + sunset_horizon * twilight * 0.8;
        self.ground_color = Vec3::new(0.05, 0.06, 0.04) * daylight.max(0.15);

        // ── Sun ─────────────────────────────────────────────────────────
        self.sun_direction = time.sun_direction();
        let warmth = (1.0 - elevation).min(1.0);
        self.sun_color = Vec3::new(1.0, 1.0 - warmth * 0.3, 1.0 - warmth * 0.5);
        self.sun_intensity = daylight;

        // ── Moon ────────────────────────────────────────────────────────
        if self.moon_auto {
            self.moon_direction = time.moon_direction();
            self.moon_intensity = time.star_intensity() * 0.15;
        }
        if !self.moon_enabled {
            self.moon_intensity = 0.0;
        }

        // ── Stars ───────────────────────────────────────────────────────
        if self.stars_auto {
            self.star_intensity = time.star_intensity();
        }
        if !self.stars_enabled {
            self.star_intensity = 0.0;
        }

        // ── Atmosphere ──────────────────────────────────────────────────
        // Increase rayleigh density at sunset for warm scattering.
        self.rayleigh_density = 1.0 + twilight * 0.5;
        self.mie_density = 1.0 + twilight * 1.2;
    }

    /// Whether the procedural sky should be used (no HDR map set).
    pub fn use_procedural_sky(&self) -> bool {
        self.hdr_sky_path.is_empty()
    }

    /// Convert the sky into a packed uniform block for the GPU shader.
    /// Returns (zenith, horizon, ground, sun_dir, sun_color, etc.)
    /// as a flat [f32; N] array ready for a uniform buffer.
    pub fn to_uniform_data(&self) -> SkyUniformData {
        SkyUniformData {
            zenith_color: [self.zenith_color.x, self.zenith_color.y, self.zenith_color.z, 0.0],
            horizon_color: [self.horizon_color.x, self.horizon_color.y, self.horizon_color.z, 0.0],
            ground_color: [self.ground_color.x, self.ground_color.y, self.ground_color.z, 0.0],
            sun_direction: [self.sun_direction.x, self.sun_direction.y, self.sun_direction.z, 0.0],
            sun_color: [self.sun_color.x, self.sun_color.y, self.sun_color.z, self.sun_intensity],
            moon_direction: [self.moon_direction.x, self.moon_direction.y, self.moon_direction.z, self.moon_intensity],
            atmosphere: [
                self.rayleigh_density,
                self.mie_scatter * 1e6, // scale for GPU
                self.mie_density,
                self.mie_direction,
            ],
            stars_params: [
                self.star_intensity,
                self.star_density,
                self.sun_disc_radius_deg,
                self.sun_halo_falloff,
            ],
            sky_visibility: [
                if self.stars_enabled { 1.0 } else { 0.0 },
                if self.moon_enabled { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        }
    }
}

/// GPU-ready uniform data for the sky shader.
/// Layout: 9 × vec4 = 144 bytes. Must be a multiple of 16.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniformData {
    pub zenith_color: [f32; 4],
    pub horizon_color: [f32; 4],
    pub ground_color: [f32; 4],
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub moon_direction: [f32; 4],
    pub atmosphere: [f32; 4],
    pub stars_params: [f32; 4],
    pub sky_visibility: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::time_of_day::TimeOfDay;

    #[test]
    fn sky_updates_from_time() {
        let mut time = TimeOfDay::new();
        time.hour = 12.0;
        time.sunrise_hour = 6.0;
        time.sunset_hour = 18.0;

        let mut sky = SkyParams::default();
        sky.update_from_time(&time);

        // At noon, zenith should be blue (high Y component).
        assert!(sky.zenith_color.y > 0.2);
        // Stars should be invisible.
        assert!(sky.star_intensity < 0.1);
    }

    #[test]
    fn sky_sunset_colors() {
        let mut time = TimeOfDay::new();
        time.hour = 18.0; // sunset
        time.sunrise_hour = 6.0;
        time.sunset_hour = 18.0;

        let mut sky = SkyParams::default();
        sky.update_from_time(&time);

        // Sunset should have warm horizon.
        assert!(sky.horizon_color.x > sky.horizon_color.z);
    }

    #[test]
    fn uniform_data_size() {
        let sky = SkyParams::default();
        let data = sky.to_uniform_data();
        let bytes = std::mem::size_of::<SkyUniformData>();
        assert_eq!(bytes, 144); // 9 vec4s × 16 bytes
    }
}
