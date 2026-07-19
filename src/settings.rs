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
    // ── Tone mapping + colour grading ───────────────────────────────────────
    pub tonemap_enabled: bool,
    pub tonemap_exposure: f32,
    pub tonemap_temperature: f32,
    pub tonemap_saturation: f32,
    pub tonemap_contrast: f32,
    pub tonemap_vibrance: f32,
    pub tonemap_grain: f32,
    // ── Wind ─────────────────────────────────────────────────────────────────
    pub wind_dir_x: f32,
    pub wind_dir_z: f32,
    pub wind_strength: f32,
    // ── Screen-Space Reflections ────────────────────────────────────────────
    pub ssr_enabled: bool,
    pub ssr_max_steps: u32,
    pub ssr_max_distance: f32,
    pub ssr_thickness: f32,
    pub ssr_intensity: f32,
    // ── Water rendering ─────────────────────────────────────────────────────
    pub water_enabled: bool,
    // ── Lava rendering ─────────────────────────────────────────────────────
    pub lava_enabled: bool,
    // ── Fire rendering ─────────────────────────────────────────────────────
    pub fire_enabled: bool,
    // ── Temporal Anti-Aliasing (TAA) ──────────────────────────────────────
    pub taa_enabled: bool,
    pub taa_blend_factor: f32,
    // ── Motion Blur ───────────────────────────────────────────────────────
    pub motion_blur_enabled: bool,
    pub motion_blur_strength: f32,
    // ── Depth of Field ────────────────────────────────────────────────────
    pub dof_enabled: bool,
    pub dof_focus_distance: f32,
    pub dof_strength: f32,
    pub dof_aperture: f32,
    // ── God Rays ─────────────────────────────────────────────────────────
    pub god_rays_enabled: bool,
    pub god_rays_intensity: f32,
    pub god_rays_decay: f32,
    pub god_rays_density: f32,
    pub god_rays_weight: f32,
    pub god_rays_num_samples: u32,
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
            // Tone mapping defaults — neutral.
            tonemap_enabled: true,
            tonemap_exposure: 0.0,
            tonemap_temperature: 0.0,
            tonemap_saturation: 0.0,
            tonemap_contrast: 0.0,
            tonemap_vibrance: 0.0,
            tonemap_grain: 0.0,
            // Wind defaults.
            wind_dir_x: 1.0,
            wind_dir_z: 0.3,
            wind_strength: 0.1,
            // SSR defaults.
            ssr_enabled: false,
            ssr_max_steps: 64,
            ssr_max_distance: 50.0,
            ssr_thickness: 0.05,
            ssr_intensity: 1.0,
            // Water defaults.
            water_enabled: true,
            // Lava defaults.
            lava_enabled: true,
            // Fire defaults.
            fire_enabled: true,
            // TAA defaults — enabled by default for anti-aliasing.
            taa_enabled: true,
            taa_blend_factor: 0.1,
            // Motion blur defaults — off by default.
            motion_blur_enabled: false,
            motion_blur_strength: 0.5,
            // DOF defaults — off by default.
            dof_enabled: false,
            dof_focus_distance: 10.0,
            dof_strength: 4.0,
            dof_aperture: 0.02,
            // God rays — off by default.
            god_rays_enabled: false,
            god_rays_intensity: 0.4,
            god_rays_decay: 0.96,
            god_rays_density: 1.2,
            god_rays_weight: 0.04,
            god_rays_num_samples: 32,
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
        let mut settings = match crate::vfs::read_to_string(path) {
            Ok(text) => toml::from_str::<Self>(&text).unwrap_or_else(|err| {
                tracing::error!("[Settings] Invalid settings file ({}): {}", path, err);
                Self::default()
            }),
            Err(_) => {
                tracing::info!("[Settings] No settings file found. Using defaults.");
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
                // Tone mapping always on — cheap and prevents washed-out HDR.
                self.render.tonemap_enabled = true;
                self.render.tonemap_exposure = 0.0;
                self.render.tonemap_temperature = 0.0;
                self.render.tonemap_saturation = 0.0;
                self.render.tonemap_contrast = 0.0;
                self.render.tonemap_vibrance = 0.0;
                self.render.tonemap_grain = 0.0;
                // No wind on mobile.
                self.render.wind_strength = 0.0;
                // No SSR on mobile.
                self.render.ssr_enabled = false;
                // No fire on mobile — too expensive.
                self.render.fire_enabled = false;
                // No TAA, motion blur, DOF on mobile.
                self.render.taa_enabled = false;
                self.render.motion_blur_enabled = false;
                self.render.dof_enabled = false;
                // No god rays on mobile.
                self.render.god_rays_enabled = false;
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
                // TAA enabled on balanced — good anti-aliasing at moderate cost.
                self.render.taa_enabled = true;
                self.render.taa_blend_factor = 0.1;
                // No motion blur or DOF on balanced.
                self.render.motion_blur_enabled = false;
                self.render.dof_enabled = false;
                // God rays on balanced — moderate sample count.
                self.render.god_rays_enabled = true;
                self.render.god_rays_intensity = 0.3;
                self.render.god_rays_num_samples = 24;
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
                // Cinematic tone mapping — subtle warm film look.
                self.render.tonemap_enabled = true;
                self.render.tonemap_exposure = 0.1;
                self.render.tonemap_temperature = 0.05;
                self.render.tonemap_saturation = 0.1;
                self.render.tonemap_contrast = 0.1;
                self.render.tonemap_vibrance = 0.15;
                self.render.tonemap_grain = 0.02;
                // Full wind on cinematic.
                self.render.wind_dir_x = 1.0;
                self.render.wind_dir_z = 0.3;
                self.render.wind_strength = 0.3;
                // SSR on cinematic.
                self.render.ssr_enabled = true;
                self.render.ssr_max_steps = 64;
                self.render.ssr_max_distance = 50.0;
                self.render.ssr_thickness = 0.05;
                self.render.ssr_intensity = 1.0;
                // TAA on cinematic.
                self.render.taa_enabled = true;
                self.render.taa_blend_factor = 0.1;
                // Motion blur on cinematic.
                self.render.motion_blur_enabled = true;
                self.render.motion_blur_strength = 0.5;
                // DOF on cinematic.
                self.render.dof_enabled = true;
                self.render.dof_focus_distance = 10.0;
                self.render.dof_strength = 4.0;
                self.render.dof_aperture = 0.02;
                // God rays on cinematic.
                self.render.god_rays_enabled = true;
                self.render.god_rays_intensity = 0.5;
                self.render.god_rays_decay = 0.96;
                self.render.god_rays_density = 1.0;
                self.render.god_rays_weight = 0.05;
                self.render.god_rays_num_samples = 48;
            }
        }
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let s = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        crate::vfs::write_string(path, &s).map_err(|e| e.to_string())
    }

    /// Keep `engine_settings.toml` fields aligned with live `RenderFeatures` after editor toggles.
    pub fn sync_render_from_renderer_features(&mut self, f: &crate::renderer::RenderFeatures) {
        self.render = RenderSettings::from_features(f);
    }
}

impl RenderSettings {
    /// Create RenderSettings from a live RenderFeatures snapshot (editor → settings sync).
    pub fn from_features(f: &crate::renderer::RenderFeatures) -> Self {
        Self {
            preset: RenderPreset::Custom,
            shadows_enabled: f.shadows_enabled,
            pcf_enabled: f.pcf_enabled,
            pcss_enabled: f.pcss_enabled,
            ibl_enabled: f.ibl_enabled,
            probes_enabled: f.probes_enabled,
            volumetric_enabled: f.volumetric_enabled,
            shadow_resolution: f.shadow_resolution,
            pcf_samples: f.pcf_samples,
            culling_enabled: f.culling_enabled,
            culling_distance: f.culling_distance,
            frustum_culling_enabled: f.frustum_culling_enabled,
            bloom_enabled: f.bloom_enabled,
            bloom_strength: f.bloom_strength,
            ssao_enabled: f.ssao_enabled,
            ssao_strength: f.ssao_strength,
            volumetric_fog_enabled: f.volumetric_fog_enabled,
            fog_density: f.fog_density,
            voxel_gi_enabled: f.voxel_gi_enabled,
            voxel_gi_strength: f.voxel_gi_strength,
            sun_azimuth_deg: f.sun_azimuth_deg,
            sun_elevation_deg: f.sun_elevation_deg,
            sun_intensity: f.sun_intensity,
            sky_hdr_path: String::new(),
            tonemap_enabled: f.tonemap_enabled,
            tonemap_exposure: f.tonemap_exposure,
            tonemap_temperature: f.tonemap_temperature,
            tonemap_saturation: f.tonemap_saturation,
            tonemap_contrast: f.tonemap_contrast,
            tonemap_vibrance: f.tonemap_vibrance,
            tonemap_grain: f.tonemap_grain,
            wind_dir_x: f.wind_dir[0],
            wind_dir_z: f.wind_dir[2],
            wind_strength: f.wind_strength,
            ssr_enabled: f.ssr_enabled,
            ssr_max_steps: f.ssr_max_steps,
            ssr_max_distance: f.ssr_max_distance,
            ssr_thickness: f.ssr_thickness,
            ssr_intensity: f.ssr_intensity,
            water_enabled: f.water_enabled,
            lava_enabled: f.lava_enabled,
            fire_enabled: f.fire_enabled,
            // TAA.
            taa_enabled: f.taa_enabled,
            taa_blend_factor: f.taa_blend_factor,
            // Motion blur.
            motion_blur_enabled: f.motion_blur_enabled,
            motion_blur_strength: f.motion_blur_strength,
            // DOF.
            dof_enabled: f.dof_enabled,
            dof_focus_distance: f.dof_focus_distance,
            dof_strength: f.dof_strength,
            dof_aperture: f.dof_aperture,
            // God rays.
            god_rays_enabled: f.god_rays_enabled,
            god_rays_intensity: f.god_rays_intensity,
            god_rays_decay: f.god_rays_decay,
            god_rays_density: f.god_rays_density,
            god_rays_weight: f.god_rays_weight,
            god_rays_num_samples: f.god_rays_num_samples,
        }
    }

    /// Apply these settings to a live RenderFeatures (settings → renderer sync).
    pub fn apply_to_features(&self, f: &mut crate::renderer::RenderFeatures) {
        f.shadows_enabled = self.shadows_enabled;
        f.pcf_enabled = self.pcf_enabled;
        f.pcss_enabled = self.pcss_enabled;
        f.ibl_enabled = self.ibl_enabled;
        f.probes_enabled = self.probes_enabled;
        f.volumetric_enabled = self.volumetric_enabled;
        f.shadow_resolution = self.shadow_resolution;
        f.pcf_samples = self.pcf_samples;
        f.culling_enabled = self.culling_enabled;
        f.culling_distance = self.culling_distance;
        f.frustum_culling_enabled = self.frustum_culling_enabled;
        f.bloom_enabled = self.bloom_enabled;
        f.bloom_strength = self.bloom_strength;
        f.ssao_enabled = self.ssao_enabled;
        f.ssao_strength = self.ssao_strength;
        f.volumetric_fog_enabled = self.volumetric_fog_enabled;
        f.fog_density = self.fog_density;
        f.voxel_gi_enabled = self.voxel_gi_enabled;
        f.voxel_gi_strength = self.voxel_gi_strength;
        f.sun_azimuth_deg = self.sun_azimuth_deg;
        f.sun_elevation_deg = self.sun_elevation_deg;
        f.sun_intensity = self.sun_intensity;
        // Tone mapping.
        f.tonemap_enabled = self.tonemap_enabled;
        f.tonemap_exposure = self.tonemap_exposure;
        f.tonemap_temperature = self.tonemap_temperature;
        f.tonemap_saturation = self.tonemap_saturation;
        f.tonemap_contrast = self.tonemap_contrast;
        f.tonemap_vibrance = self.tonemap_vibrance;
        f.tonemap_grain = self.tonemap_grain;
        // Wind.
        f.wind_dir = [self.wind_dir_x, 0.0, self.wind_dir_z];
        f.wind_strength = self.wind_strength;
        // SSR.
        f.ssr_enabled = self.ssr_enabled;
        f.ssr_max_steps = self.ssr_max_steps;
        f.ssr_max_distance = self.ssr_max_distance;
        f.ssr_thickness = self.ssr_thickness;
        f.ssr_intensity = self.ssr_intensity;
        // Water.
        f.water_enabled = self.water_enabled;
        // Lava.
        f.lava_enabled = self.lava_enabled;
        // Fire.
        f.fire_enabled = self.fire_enabled;
        // TAA.
        f.taa_enabled = self.taa_enabled;
        f.taa_blend_factor = self.taa_blend_factor;
        // Motion blur.
        f.motion_blur_enabled = self.motion_blur_enabled;
        f.motion_blur_strength = self.motion_blur_strength;
        // DOF.
        f.dof_enabled = self.dof_enabled;
        f.dof_focus_distance = self.dof_focus_distance;
        f.dof_strength = self.dof_strength;
        f.dof_aperture = self.dof_aperture;
        // God rays.
        f.god_rays_enabled = self.god_rays_enabled;
        f.god_rays_intensity = self.god_rays_intensity;
        f.god_rays_decay = self.god_rays_decay;
        f.god_rays_density = self.god_rays_density;
        f.god_rays_weight = self.god_rays_weight;
        f.god_rays_num_samples = self.god_rays_num_samples;
    }
}
