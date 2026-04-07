use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EngineSettings {
    pub render: RenderSettings,
    pub input: InputSettings,
    pub runtime: RuntimeSettings,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            render: RenderSettings::default(),
            input: InputSettings::default(),
            runtime: RuntimeSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RenderSettings {
    pub preset: RenderPreset,
    pub shadows_enabled: bool,
    pub pcf_enabled: bool,
    pub pcss_enabled: bool,
    pub ibl_enabled: bool,
    pub probes_enabled: bool,
    pub volumetric_enabled: bool,
    pub shadow_resolution: u32,
    pub pcf_samples: u32,
    pub culling_enabled: bool,
    pub culling_distance: f32,
    pub frustum_culling_enabled: bool,
    pub bloom_enabled: bool,
    pub bloom_strength: f32,
    pub ssao_enabled: bool,
    pub ssao_strength: f32,
    pub volumetric_fog_enabled: bool,
    pub fog_density: f32,
    pub voxel_gi_enabled: bool,
    pub voxel_gi_strength: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            preset: RenderPreset::Balanced,
            shadows_enabled: true,
            pcf_enabled: true,
            pcss_enabled: false,
            ibl_enabled: true,
            probes_enabled: false,
            volumetric_enabled: false,
            shadow_resolution: 1024,
            pcf_samples: 4,
            culling_enabled: true,
            culling_distance: 80.0,
            frustum_culling_enabled: true,
            bloom_enabled: false,
            bloom_strength: 0.15,
            ssao_enabled: false,
            ssao_strength: 0.35,
            volumetric_fog_enabled: false,
            fog_density: 0.03,
            voxel_gi_enabled: false,
            voxel_gi_strength: 0.20,
            sun_azimuth_deg: 35.0,
            sun_elevation_deg: 42.0,
            sun_intensity: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub enum RenderPreset {
    Mobile,
    Balanced,
    Cinematic,
    Custom,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct InputSettings {
    pub gamepad_enabled: bool,
    pub left_stick_deadzone: f32,
}

impl Default for InputSettings {
    fn default() -> Self {
        Self {
            gamepad_enabled: true,
            left_stick_deadzone: 0.2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RuntimeSettings {
    pub multithreading_enabled: bool,
    pub worker_threads: usize,
    pub profiler_enabled: bool,
    pub profiler_log_interval_frames: u32,
    pub asset_streaming_enabled: bool,
    pub foliage_wind_enabled: bool,
    pub foliage_wind_update_divisor: u32,
    pub max_fps: u32,
    pub script_hot_reload_enabled: bool,
    pub preferred_script_editor: String,
    pub asset_hot_reload_enabled: bool,
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            multithreading_enabled: false,
            worker_threads: 0,
            profiler_enabled: true,
            profiler_log_interval_frames: 120,
            asset_streaming_enabled: true,
            foliage_wind_enabled: true,
            foliage_wind_update_divisor: 2,
            max_fps: 60,
            script_hot_reload_enabled: true,
            preferred_script_editor: String::new(),
            asset_hot_reload_enabled: true,
        }
    }
}

impl EngineSettings {
    pub fn load(path: &str) -> Self {
        let mut settings = match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str::<Self>(&text).unwrap_or_else(|err| {
                eprintln!("[Settings] Invalid settings file ({}): {}", path, err);
                Self::default()
            }),
            Err(_) => {
                println!("[Settings] No settings file found. Using defaults.");
                Self::default()
            }
        };
        settings.apply_preset();
        settings
    }

    fn apply_preset(&mut self) {
        match self.render.preset {
            RenderPreset::Custom => {}
            RenderPreset::Mobile => {
                self.render.shadows_enabled = true;
                self.render.pcf_enabled = true;
                self.render.pcss_enabled = false;
                self.render.ibl_enabled = true;
                self.render.probes_enabled = false;
                self.render.shadow_resolution = 1024;
                self.render.pcf_samples = 4;
                self.render.culling_enabled = true;
                self.render.culling_distance = 60.0;
                self.render.frustum_culling_enabled = true;
                self.render.bloom_enabled = false;
                self.render.ssao_enabled = false;
                self.render.volumetric_fog_enabled = false;
                self.render.voxel_gi_enabled = false;
                self.render.sun_elevation_deg = 38.0;
                self.render.sun_azimuth_deg = 25.0;
                self.render.sun_intensity = 0.95;
            }
            RenderPreset::Balanced => {
                self.render.shadows_enabled = true;
                self.render.pcf_enabled = true;
                self.render.pcss_enabled = false;
                self.render.ibl_enabled = true;
                self.render.probes_enabled = true;
                self.render.shadow_resolution = 2048;
                self.render.pcf_samples = 9;
                self.render.culling_enabled = true;
                self.render.culling_distance = 100.0;
                self.render.frustum_culling_enabled = true;
                self.render.bloom_enabled = true;
                self.render.bloom_strength = 0.15;
                self.render.ssao_enabled = true;
                self.render.ssao_strength = 0.35;
                self.render.volumetric_fog_enabled = false;
                self.render.voxel_gi_enabled = false;
                self.render.sun_elevation_deg = 46.0;
                self.render.sun_azimuth_deg = 35.0;
                self.render.sun_intensity = 1.0;
            }
            RenderPreset::Cinematic => {
                self.render.shadows_enabled = true;
                self.render.pcf_enabled = true;
                self.render.pcss_enabled = true;
                self.render.ibl_enabled = true;
                self.render.probes_enabled = true;
                self.render.shadow_resolution = 4096;
                self.render.pcf_samples = 16;
                self.render.culling_enabled = true;
                self.render.culling_distance = 180.0;
                self.render.frustum_culling_enabled = true;
                self.render.bloom_enabled = true;
                self.render.bloom_strength = 0.25;
                self.render.ssao_enabled = true;
                self.render.ssao_strength = 0.5;
                self.render.volumetric_fog_enabled = true;
                self.render.fog_density = 0.05;
                self.render.voxel_gi_enabled = true;
                self.render.voxel_gi_strength = 0.3;
                self.render.sun_elevation_deg = 28.0;
                self.render.sun_azimuth_deg = 318.0;
                self.render.sun_intensity = 1.15;
            }
        }
    }
}
