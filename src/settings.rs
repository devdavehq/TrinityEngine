use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    /// Optional path to an HDR sky environment map (.hdr).
    pub sky_hdr_path: String,
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
            sky_hdr_path: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum RenderPreset {
    Mobile,
    Balanced,
    Cinematic,
    Custom,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    /// Continuous-collision style substep mode for fast projectiles.
    pub physics_ccd_enabled: bool,
    /// Maximum physics substeps per frame when CCD mode is enabled.
    pub physics_max_substeps: u32,
    /// Use a spatial-hash broadphase to cut pair counts before SAT testing.
    pub physics_broadphase_enabled: bool,
    /// Spatial hash cell size for broadphase buckets.
    pub physics_broadphase_cell_size: f32,
    /// Iterative solver passes for contact stability.
    pub physics_solver_iterations: u32,
    /// Enable positional correction after contacts are found.
    pub physics_position_correction_enabled: bool,
    /// Enable friction impulses and sliding response.
    pub physics_friction_enabled: bool,
    /// Allow inactive bodies to sleep.
    pub physics_sleeping_enabled: bool,
    /// Emit collision phases (started/stay/ended).
    pub physics_collision_events_enabled: bool,
    /// Smooth foliage motion instead of hard snapping transforms.
    pub physics_smooth_foliage_motion: bool,
    /// Enable angular dynamics and spin response.
    pub physics_angular_dynamics_enabled: bool,
    /// Enable heavier articulated constraint solving.
    pub physics_advanced_constraints_enabled: bool,
    /// Additional iterations used by advanced constraints.
    pub physics_constraint_iterations: u32,
    /// Use local anchor offsets when solving joints.
    pub physics_local_anchor_constraints_enabled: bool,
    /// Enable full 3D OBB contact testing for rotated boxes.
    pub physics_3d_obb_contacts_enabled: bool,
    /// Experimental articulated impulse solver path.
    pub physics_articulated_impulse_solver_enabled: bool,
    /// Experimental manifold caching and warm starting.
    pub physics_manifold_warm_start_enabled: bool,
    /// Experimental full 3-axis angular dynamics path.
    pub physics_full_angular_3d_enabled: bool,
    /// GPU scalability profile: "auto" | "low" | "balanced" | "high" | "experimental".
    pub gpu_scalability_tier: String,
    /// Startup scene file path, e.g. "scenes/main.scene".
    pub startup_scene_path: String,
    /// When true, use the old undocked multi-panel editor instead of egui_dock (deprecated).
    pub legacy_editor_ui: bool,
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
            physics_ccd_enabled: false,
            physics_max_substeps: 2,
            physics_broadphase_enabled: true,
            physics_broadphase_cell_size: 2.5,
            physics_solver_iterations: 2,
            physics_position_correction_enabled: true,
            physics_friction_enabled: true,
            physics_sleeping_enabled: true,
            physics_collision_events_enabled: false,
            physics_smooth_foliage_motion: false,
            physics_angular_dynamics_enabled: true,
            physics_advanced_constraints_enabled: false,
            physics_constraint_iterations: 1,
            physics_local_anchor_constraints_enabled: false,
            physics_3d_obb_contacts_enabled: false,
            physics_articulated_impulse_solver_enabled: false,
            physics_manifold_warm_start_enabled: false,
            physics_full_angular_3d_enabled: false,
            gpu_scalability_tier: "auto".to_string(),
            startup_scene_path: "scenes/main.scene".to_string(),
            legacy_editor_ui: false,
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

    pub fn save(&self, path: &str) -> Result<(), String> {
        let s = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, s).map_err(|e| e.to_string())
    }

    /// Keep `engine_settings.toml` fields aligned with live `RenderFeatures` after editor toggles.
    pub fn sync_render_from_renderer_features(&mut self, f: &crate::renderer::RenderFeatures) {
        self.render.preset = RenderPreset::Custom;
        self.render.shadows_enabled = f.shadows_enabled;
        self.render.pcf_enabled = f.pcf_enabled;
        self.render.pcss_enabled = f.pcss_enabled;
        self.render.ibl_enabled = f.ibl_enabled;
        self.render.probes_enabled = f.probes_enabled;
        self.render.volumetric_enabled = f.volumetric_enabled;
        self.render.shadow_resolution = f.shadow_resolution;
        self.render.pcf_samples = f.pcf_samples;
        self.render.culling_enabled = f.culling_enabled;
        self.render.culling_distance = f.culling_distance;
        self.render.frustum_culling_enabled = f.frustum_culling_enabled;
        self.render.bloom_enabled = f.bloom_enabled;
        self.render.bloom_strength = f.bloom_strength;
        self.render.ssao_enabled = f.ssao_enabled;
        self.render.ssao_strength = f.ssao_strength;
        self.render.volumetric_fog_enabled = f.volumetric_fog_enabled;
        self.render.fog_density = f.fog_density;
        self.render.voxel_gi_enabled = f.voxel_gi_enabled;
        self.render.voxel_gi_strength = f.voxel_gi_strength;
        self.render.sun_azimuth_deg = f.sun_azimuth_deg;
        self.render.sun_elevation_deg = f.sun_elevation_deg;
        self.render.sun_intensity = f.sun_intensity;
    }
}
