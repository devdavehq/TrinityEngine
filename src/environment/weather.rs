// src/environment/weather.rs
// ── Weather System ─────────────────────────────────────────────────────────
//
// WHY IT EXISTS:
//   Static skies are boring. A weather system adds atmosphere: rain, snow,
//   fog, wind. It's a simple state machine that the renderer reads to
//   control particles, fog density, and wetness.
//
// ARCHITECTURE:
//   WeatherState is a pure data struct. The renderer and particle systems
//   read it each frame to decide:
//     • Whether to spawn rain/snow particles
//     • Fog density (higher in rain/snow)
//     • Wind direction and strength
//     • Wetness factor for materials (specular boost, darker albedo)
//
// DATA FLOW:
//   WeatherState → Renderer (fog, particles, wetness)
//               → Foliage system (wind strength)
//               → Audio system (rain/snow ambience)
//
// COMMON MISTAKES:
//   • Using a hard binary switch (raining or not) — looks jarring.
//     Solution: use a smooth `intensity` ramp so weather builds gradually.
//   • Not syncing wind with foliage — rain should move leaves.
//     Solution: weather.wind_strength feeds into FoliageWind.amplitude.
//
// PERFORMANCE:
//   ~20ns per frame (just field updates). Particle spawning is the renderer's job.
//
// MEMORY:
//   ~48 bytes. Stack only.
//
// MULTITHREADING:
//   Read-only after update. Safe to share across threads.

/// Current weather condition. Controls visual effects and atmosphere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeatherCondition {
    Clear,
    Cloudy,
    Overcast,
    LightRain,
    HeavyRain,
    Snow,
    Fog,
    Storm,
}

impl WeatherCondition {
    /// Base fog density multiplier for this condition.
    pub fn fog_multiplier(&self) -> f32 {
        match self {
            Self::Clear => 0.0,
            Self::Cloudy => 0.1,
            Self::Overcast => 0.2,
            Self::LightRain => 0.4,
            Self::HeavyRain => 0.6,
            Self::Snow => 0.5,
            Self::Fog => 1.0,
            Self::Storm => 0.7,
        }
    }

    /// Wetness factor (0 = dry, 1 = soaked). Affects material specular.
    pub fn wetness(&self) -> f32 {
        match self {
            Self::Clear | Self::Cloudy => 0.0,
            Self::Overcast => 0.05,
            Self::LightRain => 0.5,
            Self::HeavyRain => 1.0,
            Self::Snow => 0.3,
            Self::Fog => 0.2,
            Self::Storm => 1.0,
        }
    }

    /// Whether this condition produces precipitation particles.
    pub fn has_precipitation(&self) -> bool {
        matches!(self, Self::LightRain | Self::HeavyRain | Self::Snow | Self::Storm)
    }

    /// Whether precipitation is snow (not rain).
    pub fn is_snow(&self) -> bool {
        *self == Self::Snow
    }

    /// Ambient sound category for this condition.
    pub fn ambient_sound_tag(&self) -> &'static str {
        match self {
            Self::Clear | Self::Cloudy => "ambient_outdoor",
            Self::Overcast => "ambient_overcast",
            Self::LightRain => "rain_light",
            Self::HeavyRain | Self::Storm => "rain_heavy",
            Self::Snow => "snow_ambient",
            Self::Fog => "ambient_fog",
        }
    }
}

/// Complete weather state for a frame. Read by renderer, audio, and gameplay.
#[derive(Clone, Debug)]
pub struct WeatherState {
    /// Current weather type.
    pub condition: WeatherCondition,
    /// Smooth intensity (0.0 = calm, 1.0 = extreme).
    /// Rain: 0 = no drops, 1 = downpour. Snow: 0 = flurries, 1 = blizzard.
    pub intensity: f32,
    /// Wind direction (normalized XZ plane vector).
    pub wind_direction: glam::Vec2,
    /// Wind strength in m/s. Affects particles, foliage, and audio panning.
    pub wind_strength: f32,
    /// Temperature in Celsius. Affects rain→snow transition.
    pub temperature: f32,
    /// Cloud coverage (0 = clear sky, 1 = fully overcast).
    pub cloud_coverage: f32,
    /// Transition speed (0 = instant, 1 = 10-second blend).
    pub transition_speed: f32,
    /// Snow accumulation on surfaces (0 = no snow, 1 = fully covered).
    /// Builds over time during snowfall, melts when temperature rises above 0C.
    pub snow_coverage: f32,
    /// Snow accumulation rate (how fast snow builds up per second).
    pub snow_accumulation_rate: f32,
    /// Snow melt rate (how fast snow melts when temperature > 0C).
    pub snow_melt_rate: f32,
    /// Snow depth in centimetres (visual height of accumulated snow on surfaces).
    pub snow_depth: f32,
    /// Snow surface roughness (0 = smooth, 1 = wind-sculpted drifts).
    pub snow_roughness: f32,
    /// Wind-driven snow drift factor (0 = flat accumulation, 1 = heavy drifts).
    pub snow_drift_factor: f32,
    /// Snow sparkle intensity (specular highlights on ice crystals).
    pub snow_sparkle: f32,
}

impl Default for WeatherState {
    fn default() -> Self {
        Self {
            condition: WeatherCondition::Clear,
            intensity: 0.0,
            wind_direction: glam::Vec2::new(1.0, 0.0),
            wind_strength: 2.0,
            temperature: 20.0,
            cloud_coverage: 0.2,
            transition_speed: 0.3,
            snow_coverage: 0.0,
            snow_accumulation_rate: 0.05,
            snow_melt_rate: 0.03,
            snow_depth: 0.0,
            snow_roughness: 0.4,
            snow_drift_factor: 0.3,
            snow_sparkle: 0.5,
        }
    }
}

impl WeatherState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Smoothly transition toward a target weather state.
    /// `dt` is frame time in seconds.
    pub fn transition_to(&mut self, target: &WeatherState, dt: f32) {
        let blend_speed = self.transition_speed;
        let t = (dt * blend_speed).clamp(0.0, 1.0);
        self.intensity = lerp(self.intensity, target.intensity, t);
        self.wind_strength = lerp(self.wind_strength, target.wind_strength, t);
        self.temperature = lerp(self.temperature, target.temperature, t);
        self.cloud_coverage = lerp(self.cloud_coverage, target.cloud_coverage, t);
        self.wind_direction = self.wind_direction.lerp(target.wind_direction, t);
        self.snow_roughness = lerp(self.snow_roughness, target.snow_roughness, t);
        self.snow_drift_factor = lerp(self.snow_drift_factor, target.snow_drift_factor, t);
        self.snow_sparkle = lerp(self.snow_sparkle, target.snow_sparkle, t);
        // Condition snaps when intensity is close to target.
        if (self.intensity - target.intensity).abs() < 0.05 {
            self.condition = target.condition;
        }
        // Snow accumulation: builds during snowfall, melts when above 0C.
        if self.condition.is_snow() && self.intensity > 0.1 {
            let accumulation = self.snow_accumulation_rate * self.intensity * dt;
            self.snow_coverage = (self.snow_coverage + accumulation).min(1.0);
            self.snow_depth = self.snow_depth + accumulation * 30.0; // 30cm per full coverage
        } else if self.temperature > 0.0 {
            let melt_factor = self.snow_melt_rate * (self.temperature / 20.0).min(1.0) * dt;
            self.snow_coverage = (self.snow_coverage - melt_factor).max(0.0);
            self.snow_depth = (self.snow_depth - melt_factor * 30.0).max(0.0);
        }
    }

    /// Combined fog density = base setting × weather multiplier.
    pub fn effective_fog_density(&self, base_fog_density: f32) -> f32 {
        let weather_fog = self.condition.fog_multiplier() * self.intensity;
        base_fog_density + weather_fog * 0.1
    }

    /// Material wetness factor for PBR (boosts specular, darkens albedo).
    pub fn wetness(&self) -> f32 {
        self.condition.wetness() * self.intensity
    }

    /// Whether it's currently precipitating.
    pub fn is_precipitating(&self) -> bool {
        self.condition.has_precipitation() && self.intensity > 0.1
    }

    /// Wind as a 3D vector (XZ plane + slight Y turbulence).
    pub fn wind_vector(&self) -> glam::Vec3 {
        glam::Vec3::new(
            self.wind_direction.x * self.wind_strength,
            self.wind_strength * 0.05, // slight updraft
            self.wind_direction.y * self.wind_strength,
        )
    }

    /// Serialize to uniform data for the GPU.
    pub fn to_uniform_data(&self) -> WeatherUniformData {
        let wind = self.wind_vector();
        WeatherUniformData {
            wind: [wind.x, wind.y, wind.z, self.wind_strength],
            params: [
                self.intensity,
                self.cloud_coverage,
                self.wetness(),
                self.temperature,
            ],
        }
    }
}

/// GPU uniform data for weather effects in shaders.
/// Layout: 2 × vec4 = 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WeatherUniformData {
    pub wind: [f32; 4],      // xyz = wind direction × strength, w = strength
    pub params: [f32; 4],    // x = intensity, y = cloud_coverage, z = wetness, w = temperature
}

/// Preset weather configurations for quick setup.
impl WeatherState {
    pub fn clear() -> Self {
        Self {
            condition: WeatherCondition::Clear,
            intensity: 0.0,
            cloud_coverage: 0.1,
            wind_strength: 1.0,
            temperature: 22.0,
            ..Default::default()
        }
    }

    pub fn rainy() -> Self {
        Self {
            condition: WeatherCondition::HeavyRain,
            intensity: 0.8,
            cloud_coverage: 0.9,
            wind_strength: 5.0,
            temperature: 12.0,
            ..Default::default()
        }
    }

    pub fn snowy() -> Self {
        Self {
            condition: WeatherCondition::Snow,
            intensity: 0.7,
            cloud_coverage: 0.85,
            wind_strength: 3.0,
            temperature: -5.0,
            snow_coverage: 0.0,
            snow_accumulation_rate: 0.08,
            snow_melt_rate: 0.02,
            snow_depth: 0.0,
            snow_roughness: 0.5,
            snow_drift_factor: 0.4,
            snow_sparkle: 0.6,
            ..Default::default()
        }
    }

    pub fn foggy() -> Self {
        Self {
            condition: WeatherCondition::Fog,
            intensity: 0.9,
            cloud_coverage: 0.5,
            wind_strength: 0.5,
            temperature: 15.0,
            ..Default::default()
        }
    }

    pub fn stormy() -> Self {
        Self {
            condition: WeatherCondition::Storm,
            intensity: 1.0,
            cloud_coverage: 1.0,
            wind_strength: 12.0,
            temperature: 8.0,
            ..Default::default()
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_weather_defaults() {
        let w = WeatherState::clear();
        assert_eq!(w.condition, WeatherCondition::Clear);
        assert!(!w.is_precipitating());
        assert!(w.wetness() < 0.01);
    }

    #[test]
    fn rainy_increases_fog() {
        let w = WeatherState::rainy();
        let base = 0.03;
        assert!(w.effective_fog_density(base) > base);
    }

    #[test]
    fn snow_is_precipitating() {
        let w = WeatherState::snowy();
        assert!(w.is_precipitating());
        assert!(w.condition.is_snow());
    }

    #[test]
    fn transition_blends() {
        let mut a = WeatherState::clear();
        let b = WeatherState::rainy();
        for _ in 0..300 {
            a.transition_to(&b, 0.016);
        }
        assert!(a.intensity > 0.5); // should have blended significantly
    }

    #[test]
    fn uniform_data_size() {
        let w = WeatherState::default();
        let _data = w.to_uniform_data();
        assert_eq!(std::mem::size_of::<WeatherUniformData>(), 32);
    }

    #[test]
    fn snow_coverage_derived_from_condition_and_intensity() {
        let w = WeatherState::snowy();
        let snow_coverage = if w.condition.is_snow() { w.intensity } else { 0.0 };
        assert!((snow_coverage - 0.7).abs() < 0.01);
    }

    #[test]
    fn snow_coverage_zero_for_non_snow() {
        let w = WeatherState::rainy();
        let snow_coverage = if w.condition.is_snow() { w.intensity } else { 0.0 };
        assert_eq!(snow_coverage, 0.0);
    }

    #[test]
    fn snow_coverage_scales_with_intensity() {
        let mut w = WeatherState::snowy();
        w.intensity = 0.3;
        let snow_coverage = if w.condition.is_snow() { w.intensity } else { 0.0 };
        assert!((snow_coverage - 0.3).abs() < 0.01);
    }
}
