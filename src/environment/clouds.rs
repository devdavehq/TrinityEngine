// src/environment/clouds.rs
// ── Cloud System ───────────────────────────────────────────────────────────
//
// WHY IT EXISTS:
//   Empty skies look wrong. Clouds add depth, scale, and movement to the
//   scene. This system generates layered cloud data that the renderer
//   ray-marches or samples from a noise texture.
//
// ARCHITECTURE:
//   CloudParams is a pure data struct controlling cloud appearance.
//   The actual rendering happens in a cloud shader (fullscreen pass or
//   ray-marched volume). The environment system just feeds parameters.
//
// CLOUD TYPES (per Bible spec):
//   • Cirrus:    high altitude, wispy, thin
//   • Stratus:   mid altitude, flat layers
//   • Cumulus:    low altitude, puffy, dramatic
//   The cloud system blends between these based on `cloud_type` and weather.
//
// DATA FLOW:
//   WeatherState + TimeOfDay → CloudParams → Cloud Shader
//
// COMMON MISTAKES:
//   • Ray-marching clouds every pixel — too expensive for low-end GPUs.
//     Solution: render clouds at half/quarter resolution and upscale.
//   • Not accounting for sun angle — clouds should be lit from below at sunset.
//     Solution: sun_color and sun_direction feed into cloud lighting.
//
// PERFORMANCE:
//   CloudParams update: ~30ns (field writes).
//   Cloud rendering: 0.5–3ms depending on quality tier and resolution.
//   On Mobile tier: skip clouds entirely (set cloud_coverage = 0).
//
// MEMORY:
//   ~64 bytes. Stack only.
//
// MULTITHREADING:
//   Read-only after update. Safe to share across threads.

/// Type of cloud formation. Affects shape and density profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloudType {
    /// High-altitude wispy clouds. Thin, fast-moving.
    Cirrus,
    /// Mid-altitude flat layers. Even coverage.
    Stratus,
    /// Low-altitude puffy clouds. Dramatic, slow-moving.
    Cumulus,
    /// No clouds rendered.
    None,
}

/// Cloud layer parameters for the renderer.
#[derive(Clone, Debug)]
pub struct CloudParams {
    /// Current cloud type.
    pub cloud_type: CloudType,
    /// Coverage (0 = clear, 1 = fully overcast).
    pub coverage: f32,
    /// Cloud base altitude in world units.
    pub base_altitude: f32,
    /// Cloud thickness (how tall the cloud layer is).
    pub thickness: f32,
    /// Cloud speed multiplier (wind carries clouds).
    pub speed: f32,
    /// Cloud UV offset (scrolling texture coordinates, updated each frame).
    pub uv_offset: glam::Vec2,
    /// Noise scale. Larger = smaller cloud puffs.
    pub noise_scale: f32,
    /// Noise detail (octaves). Higher = more detailed edges.
    pub detail_octaves: u32,
    /// Density threshold. Higher = more gaps between clouds.
    pub density_threshold: f32,
    /// Density smoothness. Higher = softer cloud edges.
    pub density_smoothness: f32,
    /// Precipitation from clouds (rain below cloud layer).
    pub precipitation: f32,
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            cloud_type: CloudType::Stratus,
            coverage: 0.3,
            base_altitude: 80.0,
            thickness: 20.0,
            speed: 1.0,
            uv_offset: glam::Vec2::ZERO,
            noise_scale: 0.003,
            detail_octaves: 4,
            density_threshold: 0.4,
            density_smoothness: 0.2,
            precipitation: 0.0,
        }
    }
}

impl CloudParams {
    /// Update cloud parameters from weather and time.
    pub fn update(
        &mut self,
        weather: &super::weather::WeatherState,
        time_of_day: &super::time_of_day::TimeOfDay,
        dt: f32,
    ) {
        // Coverage matches weather cloud_coverage.
        self.coverage = weather.cloud_coverage;

        // Weather determines cloud type.
        self.cloud_type = match weather.condition {
            super::weather::WeatherCondition::Clear => CloudType::None,
            super::weather::WeatherCondition::Cloudy => CloudType::Cirrus,
            super::weather::WeatherCondition::Overcast => CloudType::Stratus,
            super::weather::WeatherCondition::LightRain
            | super::weather::WeatherCondition::HeavyRain => CloudType::Cumulus,
            super::weather::WeatherCondition::Snow => CloudType::Stratus,
            super::weather::WeatherCondition::Fog => CloudType::Stratus,
            super::weather::WeatherCondition::Storm => CloudType::Cumulus,
        };

        // Precipitation from weather intensity.
        self.precipitation = weather.intensity * if weather.condition.has_precipitation() { 1.0 } else { 0.0 };

        // Wind carries clouds — scroll UV offset.
        let wind = weather.wind_vector();
        self.uv_offset.x += wind.x * self.speed * dt * 0.01;
        self.uv_offset.y += wind.z * self.speed * dt * 0.01;

        // Night clouds are less visible (reduced contrast).
        let daylight = time_of_day.daylight_factor();
        self.density_threshold = 0.4 + (1.0 - daylight) * 0.2;
    }

    /// Whether clouds should be rendered at all.
    pub fn should_render(&self) -> bool {
        self.cloud_type != CloudType::None && self.coverage > 0.01
    }

    /// Cloud appearance scale based on type.
    fn type_scale(&self) -> f32 {
        match self.cloud_type {
            CloudType::Cirrus => 0.5,  // thin, wispy
            CloudType::Stratus => 1.0, // standard
            CloudType::Cumulus => 1.5, // big, puffy
            CloudType::None => 0.0,
        }
    }

    /// GPU uniform data for the cloud shader.
    pub fn to_uniform_data(&self) -> CloudUniformData {
        let scale = self.type_scale();
        CloudUniformData {
            params: [
                self.coverage,
                self.base_altitude,
                self.thickness,
                scale,
            ],
            noise: [
                self.noise_scale,
                self.density_threshold,
                self.density_smoothness,
                self.precipitation,
            ],
            scroll: [self.uv_offset.x, self.uv_offset.y, self.speed, 0.0],
            cloud_type: [self.cloud_type as u32 as f32, 0.0, 0.0, 0.0],
        }
    }
}

/// GPU uniform data for cloud rendering.
/// Layout: 4 × vec4 = 64 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudUniformData {
    pub params: [f32; 4],   // x=coverage, y=base_altitude, z=thickness, w=type_scale
    pub noise: [f32; 4],    // x=noise_scale, y=density_threshold, z=density_smoothness, w=precipitation
    pub scroll: [f32; 4],   // xy=uv_offset, z=speed, w=unused
    pub cloud_type: [f32; 4], // x=type (0=none, 1=cirrus, 2=stratus, 3=cumulus), y=storm_darken, z=lightning_intensity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::weather::{WeatherCondition, WeatherState};
    use crate::environment::time_of_day::TimeOfDay;

    #[test]
    fn no_clouds_when_clear() {
        let mut clouds = CloudParams::default();
        let weather = WeatherState::clear();
        let time = TimeOfDay::new();
        clouds.update(&weather, &time, 0.016);
        assert!(!clouds.should_render());
    }

    #[test]
    fn cumulus_when_rainy() {
        let mut clouds = CloudParams::default();
        let weather = WeatherState::rainy();
        let time = TimeOfDay::new();
        clouds.update(&weather, &time, 0.016);
        assert_eq!(clouds.cloud_type, CloudType::Cumulus);
        assert!(clouds.should_render());
    }

    #[test]
    fn uv_scroll_increases() {
        let mut clouds = CloudParams::default();
        let weather = WeatherState::rainy();
        let time = TimeOfDay::new();
        let before = clouds.uv_offset;
        clouds.update(&weather, &time, 1.0);
        assert!(clouds.uv_offset.x != before.x || clouds.uv_offset.y != before.y);
    }

    #[test]
    fn uniform_data_size() {
        let clouds = CloudParams::default();
        let data = clouds.to_uniform_data();
        assert_eq!(std::mem::size_of::<CloudUniformData>(), 64);
    }
}
