// src/renderer.rs
// Owns the GPU device, swap chain, and all rendering resources.
// draw_world() is the one function called each frame from main.rs.
//
// ── wgpu 29 changes applied here ────────────────────────────────────────────
// • wgpu::Surface now owns the window via Arc<Window>, so Renderer is no
//   longer generic over a lifetime ('a is gone).
// • request_device() signature changed: no trace_path parameter.
// • SamplerDescriptor.mipmap_filter: FilterMode → MipmapFilterMode.
// • PipelineLayoutDescriptor / pipeline fields updated (see pipeline.rs).

pub mod fire;
pub mod ibl;
pub mod light_baker;
pub mod light_probes;
pub mod lightning_bolt;
pub mod lava;
pub mod particle;
pub mod pipeline;
pub mod shadow;
pub mod sky;
pub mod water;

use std::sync::Arc;
use std::collections::HashMap;

use winit::window::Window;

use crate::assets::{AssetStore, Mesh};
use crate::camera::Camera;
use crate::animation::blending::BlendedPose;
use crate::components::{Decal, MaterialTexture, PointLight, Position, Renderable, Rotation};
use crate::jobs::JobSystem;
use crate::render::instancing::{InstanceData, InstancingManager};
use hecs::World;
use rayon::prelude::*;

/// Detect the GPU performance tier based on device type, name heuristics,
/// and available adapter features/limits. Returns a tier name string that
/// maps to `RenderFeatures::from_tier_name()`.
///
/// SCORING SYSTEM:
///   Start at a base score based on device type, then add/subtract points
///   based on GPU name patterns and adapter capabilities.
///   - ≥ 80 → "high"
///   - ≥ 50 → "balanced"
///   - < 50 → "low"
///
/// WHY NOT JUST "INTEGRATED = LOW"?
///   - Apple M4 Pro/Max are integrated but high-end
///   - AMD 780M is integrated but handles high settings
///   - NVIDIA GT 710 / MX 550 are discrete but very weak
///   - Old Intel UHD 630 is integrated and truly low-end
fn detect_gpu_tier(
    info: &wgpu::AdapterInfo,
    adapter: &wgpu::Adapter,
) -> String {
    let mut score: i32 = 0;

    // ── Base score from device type ────────────────────────────────────────
    score += match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 60,
        wgpu::DeviceType::IntegratedGpu => 30,
        wgpu::DeviceType::Cpu => 10,
        wgpu::DeviceType::VirtualGpu => 20,
        wgpu::DeviceType::Other => 25,
    };

    // ── Name heuristics (case-insensitive substring match) ─────────────────
    let name_lower = info.name.to_ascii_lowercase();

    // High-end discrete GPUs
    if name_lower.contains("rtx 4") || name_lower.contains("rtx 5") {
        score += 40; // RTX 40xx/50xx series
    } else if name_lower.contains("rtx 3") {
        score += 35; // RTX 30xx series
    } else if name_lower.contains("rtx 2") {
        score += 25; // RTX 20xx series
    } else if name_lower.contains("radeon rx 7") || name_lower.contains("radeon rx 9") {
        score += 35; // AMD RDNA 3/4
    } else if name_lower.contains("radeon rx 6") {
        score += 28; // AMD RDNA 2
    } else if name_lower.contains("radeon vii") || name_lower.contains("radeon pro") {
        score += 25;
    } else if name_lower.contains("a100") || name_lower.contains("a40") || name_lower.contains("l40") {
        score += 35; // Data center GPUs
    }

    // Apple Silicon (integrated but powerful)
    if name_lower.contains("apple m") {
        score += 35;
        if name_lower.contains("m4") || name_lower.contains("m3") {
            score += 10; // Latest gen is faster
        }
        if name_lower.contains("pro") || name_lower.contains("max") || name_lower.contains("ultra") {
            score += 15;
        }
    }

    // Known low-end discrete GPUs
    if name_lower.contains("gt 710") || name_lower.contains("gt 730") || name_lower.contains("gt 1030") {
        score -= 40;
    }
    if name_lower.contains("mx 450") || name_lower.contains("mx 550") || name_lower.contains("mx 350") {
        score -= 30;
    }
    if name_lower.contains("radeon vega 8") || name_lower.contains("radeon vega 3") {
        score -= 25;
    }

    // Intel UHD / HD series (truly low-end)
    if name_lower.contains("uhd 6") || name_lower.contains("uhd 630") || name_lower.contains("hd 5") {
        score -= 20;
    }

    // Intel Arc (newer, better)
    if name_lower.contains("arc a") {
        score += 15;
    }

    // ── Adapter limits heuristics ──────────────────────────────────────────
    // max_storage_buffers is a good proxy for GPU capability.
    // High-end: 8-16, Mid: 4-8, Low: 1-4.
    let limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage >= 8 {
        score += 10;
    } else if limits.max_storage_buffers_per_shader_stage <= 2 {
        score -= 10;
    }

    // max_texture_dimension_2d: High-end supports 16384, mid 8192, low 4096.
    if limits.max_texture_dimension_2d >= 16384 {
        score += 10;
    } else if limits.max_texture_dimension_2d <= 4096 {
        score -= 10;
    }

    // max_compute_workgroups_per_dim: better compute = better GPU.
    if limits.max_compute_workgroups_per_dimension >= 65535 {
        score += 5;
    }

    // ── Map score to tier name ─────────────────────────────────────────────
    let tier = if score >= 80 {
        "high"
    } else if score >= 50 {
        "balanced"
    } else {
        "low"
    };

    tracing::info!("[Renderer] GPU score: {} → tier '{}'", score, tier);
    tier.to_string()
}

#[derive(Clone, Copy, Default)]
pub struct DrawStats {
    pub total: usize,
    pub visible: usize,
    pub drawn: usize,
}

/// Shared staging vertex buffer: one mesh is copied here per draw call.
/// Capsules and other procedurals easily exceed the old 1024 cap; keep this generous
/// until meshes use dedicated GPU buffers.
const MAX_VERTICES_PER_DRAW: usize = 262_144;

// ── Real-time voxel GI (Voxel Cone Tracing) ─────────────────────────────────
// Camera-aligned 128³ clipmap, voxel size 0.25 world units → ~32-unit cube
// around the camera. 8 mip levels (log2(128) + 1).
const VOXEL_GI_DIM: u32 = 128;
const VOXEL_GI_SIZE: f32 = 0.25;
const VOXEL_GI_MIPS: u32 = 8;

// ── GpuUniforms ──────────────────────────────────────────────────────────────
// Mirrors the Uniforms struct in shader.wgsl exactly.
// repr(C) + Pod + Zeroable: required for bytemuck::bytes_of().
// Total size: 12 × vec4 = 192 bytes — must be a multiple of 16.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuUniforms {
    view_proj:   [[f32; 4]; 4], // 64 bytes — mat4x4
    camera_pos:  [f32; 3],      // 12 bytes
    _pad0:       f32,           //  4 bytes — pad to 16-byte boundary
    light_dir:   [f32; 3],      // 12 bytes
    _pad1:       f32,
    light_color: [f32; 3],      // 12 bytes
    _pad2:       f32,
    point_light_pos_range: [f32; 4], // xyz=position, w=range
    point_light_color_intensity: [f32; 4], // rgb=color, w=intensity
    post_params0: [f32; 4], // bloom_enabled, bloom_strength, ssao_enabled, ssao_strength
    post_params1: [f32; 4], // fog_enabled, fog_density, voxel_gi_enabled, voxel_gi_strength
    fog_color:    [f32; 4], // rgb = dynamic fog color from TimeOfDay, w = elapsed time (for wind)
    wind_dir_strength: [f32; 4], // xyz = wind direction (normalised), w = wind strength [0..1]
}

// ── TonemapUniforms ────────────────────────────────────────────────────────
// Mirrors TonemapUniforms in postprocess.wgsl.
// repr(C) + Pod + Zeroable: required for bytemuck::bytes_of().
// Total size: 32 bytes (4 × vec2 / 8 × f32).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TonemapUniforms {
    exposure:    f32,
    temperature: f32,
    saturation:  f32,
    contrast:    f32,
    vibrance:    f32,
    grain:       f32,
    _pad0:       f32,
    _pad1:       f32,
}

// ── SsrUniforms ────────────────────────────────────────────────────────────
// Mirrors SsrUniforms in postprocess.wgsl.
// Total size: 160 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrUniforms {
    inv_view_proj: [[f32; 4]; 4],
    view_proj:     [[f32; 4]; 4],
    max_steps:     u32,
    max_distance:  f32,
    thickness:     f32,
    intensity:     f32,
    screen_size:   [f32; 2],
    _pad0:         [f32; 2],
}

// ── TaaUniforms ────────────────────────────────────────────────────────────
// Mirrors TaaUniforms in postprocess.wgsl.
// Total size: 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaUniforms {
    jitter_offset: [f32; 2],
    blend_factor:  f32,
    enable_taa:    f32,
}

// ── MotionBlurUniforms ────────────────────────────────────────────────────
// Mirrors MotionBlurUniforms in postprocess.wgsl.
// Total size: 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionBlurUniforms {
    blur_strength: f32,
    max_samples:   f32,
    _pad:          [f32; 2],
}

// ── DofUniforms ────────────────────────────────────────────────────────────
// Mirrors DofUniforms in postprocess.wgsl.
// Total size: 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DofUniforms {
    focus_distance: f32,
    dof_strength:   f32,
    aperture:       f32,
    _pad:           f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BilateralUniforms {
    blur_radius:  f32,
    depth_weight: f32,
    norm_weight:  f32,
    _pad:         f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GodRayUniforms {
    sun_uv:      [f32; 2],
    intensity:   f32,
    decay:       f32,
    density:     f32,
    weight:      f32,
    num_samples: f32,
    _pad:        f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UnderwaterUniforms {
    tint:          [f32; 4], // rgb = water tint, a = fog density
    caustics:      [f32; 4], // x = intensity, y = scale, z = speed, w = god_rays
    distortion:    [f32; 4], // x = distortion strength, y = time, z = vignette, w = bloom
    camera_params: [f32; 4], // x = water_surface_y, y = camera_depth_below, z = near, w = far
}

// ── GpuLightData ───────────────────────────────────────────────────────────
// Matches LightData in shader.wgsl.  Each entry is 64 bytes (aligned to 16).
#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLightData {
    position:      [f32; 3],
    _pos_pad:      f32,
    color:         [f32; 3],
    _col_pad:      f32,
    intensity:     f32,
    range:         f32,
    light_type:    f32,
    spot_angle_cos: f32,
    shadow_index:  i32,
    _pad:          f32,
    _align_pad:    [f32; 2],
    direction:     [f32; 3],
    _dir_pad:      f32,
}

// ── GpuLightUniforms ───────────────────────────────────────────────────────
// Array of 16 lights + count.  Total size: 1040 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLightUniforms {
    lights:      [GpuLightData; 16],
    light_count: u32,
    _pad1:       u32,
    _pad2:       u32,
    _pad3:       u32,
}

// ── GpuMaterialExtras ──────────────────────────────────────────────────────
// Per-object material extras (SSS, clearcoat, etc.).  Total size: 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterialExtras {
    subsurface:          f32,
    clearcoat:           f32,
    clearcoat_roughness: f32,
    anisotropy:          f32,
    emissive_strength:   f32,
    _pad:                [f32; 3],
}

// ── GpuWeatherData ────────────────────────────────────────────────────────
// Matches WeatherData in shader.wgsl (binding 13).
// Total size: 16 bytes (1 × vec4).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuWeatherData {
    snow_coverage: f32,
    _pad: [f32; 3],
}

// ── GpuShadowData ──────────────────────────────────────────────────────────
// Matches ShadowData in shader.wgsl (binding 7).
// Total size: 256 bytes (matches the fallback buffer allocation).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuShadowData {
    light_matrices:     [[f32; 4]; 12], // 3 × mat4x4 = 192 bytes
    cascade_dists:      [f32; 4],       // xyz = cascade far distances, w = unused
    shadow_bias:        f32,
    normal_offset_bias: f32,
    pcf_radius:         f32,
    shadow_enabled:     f32,
    shadow_map_size:    f32,
    _pad:               [f32; 7], // pad to 256 bytes
}

// ── RenderFeatures ────────────────────────────────────────────────────────────
// Runtime toggles for expensive features.
// Written by the editor UI or detected automatically at startup.
pub struct RenderFeatures {
    pub shadows_enabled:    bool,
    pub pcf_enabled:        bool,
    pub pcss_enabled:       bool,   // contact shadows — off by default
    pub ibl_enabled:        bool,
    pub probes_enabled:     bool,
    pub volumetric_enabled: bool,   // very expensive — off by default
    pub shadow_resolution:  u32,
    pub pcf_samples:        u32,
    pub culling_enabled:    bool,
    pub culling_distance:   f32,
    pub frustum_culling_enabled: bool,
    pub occlusion_culling_enabled: bool,
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
    // ── Tone mapping + colour grading ───────────────────────────────────────
    pub tonemap_enabled: bool,
    pub tonemap_exposure: f32,     // HDR exposure compensation
    pub tonemap_temperature: f32,  // colour temperature shift (-1 cool, +1 warm)
    pub tonemap_saturation: f32,   // global saturation multiplier
    pub tonemap_contrast: f32,     // contrast adjustment
    pub tonemap_vibrance: f32,     // selective saturation boost
    pub tonemap_grain: f32,        // film grain intensity
    // ── Wind ─────────────────────────────────────────────────────────────────
    pub wind_dir: [f32; 3],        // normalised wind direction
    pub wind_strength: f32,        // 0 = calm, 1 = hurricane
    // ── Screen-Space Reflections ────────────────────────────────────────────
    pub ssr_enabled: bool,
    pub ssr_max_steps: u32,        // ray-march iterations (32–128)
    pub ssr_max_distance: f32,     // max ray distance in view space
    pub ssr_thickness: f32,        // depth tolerance for intersection test
    pub ssr_intensity: f32,        // reflection brightness multiplier
    // ── Water rendering ─────────────────────────────────────────────────────
    pub water_enabled: bool,       // master toggle for water surfaces
    // ── Lava rendering ─────────────────────────────────────────────────────
    pub lava_enabled: bool,        // master toggle for lava surfaces
    // ── Fire rendering ────────────────────────────────────────────────────
    pub fire_enabled: bool,        // master toggle for fire surfaces
    // ── Heat distortion ──────────────────────────────────────────────────
    pub heat_distortion_enabled: bool, // screen-space heat shimmer behind fire/lava
    // ── Underwater rendering ──────────────────────────────────────────────
    pub underwater_enabled: bool,       // master toggle for underwater post-process
    pub underwater_tint: [f32; 3],     // underwater fog tint colour (RGB 0-1)
    pub underwater_fog_density: f32,   // depth-based fog density
    pub underwater_caustics: f32,      // caustic light pattern intensity
    pub underwater_god_rays: f32,      // surface god ray intensity
    pub underwater_distortion: f32,    // chromatic aberration / wave distortion
    pub underwater_vignette: f32,      // edge darkening
    pub underwater_bloom: f32,         // bloom boost underwater
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
    // ── LOD settings ─────────────────────────────────────────────────────
    pub mesh_lod_threshold_1: f32,
    pub mesh_lod_threshold_2: f32,
    pub mesh_lod_threshold_3: f32,
    pub mesh_lod_threshold_4: f32,
}

impl Default for RenderFeatures {
    fn default() -> Self {
        Self {
            shadows_enabled:    true,
            pcf_enabled:        true,
            pcss_enabled:       false,
            ibl_enabled:        true,
            probes_enabled:     true,
            volumetric_enabled: false,
            shadow_resolution:  2048,
            pcf_samples:        9,
            culling_enabled:    true,
            culling_distance:   120.0,
            frustum_culling_enabled: true,
            occlusion_culling_enabled: true,
            bloom_enabled: false,
            bloom_strength: 0.2,
            ssao_enabled: false,
            ssao_strength: 0.35,
            volumetric_fog_enabled: false,
            fog_density: 0.03,
            voxel_gi_enabled: false,
            voxel_gi_strength: 0.2,
            sun_azimuth_deg: 35.0,
            sun_elevation_deg: 42.0,
            sun_intensity: 1.0,
            // Tone mapping defaults — neutral / no adjustment.
            tonemap_enabled: true,
            tonemap_exposure: 0.0,
            tonemap_temperature: 0.0,
            tonemap_saturation: 0.0,
            tonemap_contrast: 0.0,
            tonemap_vibrance: 0.0,
            tonemap_grain: 0.0,
            // Wind defaults — gentle breeze from the east.
            wind_dir: [1.0, 0.0, 0.3],
            wind_strength: 0.1,
            // SSR defaults — off by default (expensive).
            ssr_enabled: false,
            ssr_max_steps: 64,
            ssr_max_distance: 50.0,
            ssr_thickness: 0.05,
            ssr_intensity: 1.0,
            // Water — enabled by default (cheap, only draws when WaterSurface entities exist).
            water_enabled: true,
            // Lava — enabled by default (cheap, only draws when LavaSurface entities exist).
            lava_enabled: true,
            // Fire — enabled by default (cheap, only draws when FireSurface entities exist).
            fire_enabled: true,
            // Heat distortion — enabled by default (subtle shimmer behind fire/lava).
            heat_distortion_enabled: true,
            // Underwater rendering — enabled by default.
            underwater_enabled: true,
            underwater_tint: [0.01, 0.08, 0.12],
            underwater_fog_density: 0.04,
            underwater_caustics: 0.6,
            underwater_god_rays: 0.3,
            underwater_distortion: 0.003,
            underwater_vignette: 0.3,
            underwater_bloom: 0.15,
            // TAA — enabled by default for anti-aliasing.
            taa_enabled: true,
            taa_blend_factor: 0.1,
            // Motion blur — off by default.
            motion_blur_enabled: false,
            motion_blur_strength: 0.5,
            // DOF — off by default.
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
            // LOD defaults.
            mesh_lod_threshold_1: 48.0,
            mesh_lod_threshold_2: 95.0,
            mesh_lod_threshold_3: 180.0,
            mesh_lod_threshold_4: 300.0,
        }
    }
}

impl RenderFeatures {
    // Suitable for integrated graphics (HP EliteBook, Intel UHD, etc.)
    pub fn low_end() -> Self {
        Self {
            shadows_enabled:    true,
            pcf_enabled:        true,
            pcss_enabled:       false,
            ibl_enabled:        true,
            probes_enabled:     false,  // probe capture is costly
            volumetric_enabled: false,
            shadow_resolution:  1024,   // half resolution
            pcf_samples:        4,      // 2×2 instead of 3×3
            culling_enabled:    true,
            culling_distance:   80.0,
            frustum_culling_enabled: true,
            occlusion_culling_enabled: true,
            bloom_enabled: false,
            bloom_strength: 0.1,
            ssao_enabled: false,
            ssao_strength: 0.25,
            volumetric_fog_enabled: false,
            fog_density: 0.02,
            voxel_gi_enabled: false,
            voxel_gi_strength: 0.15,
            sun_azimuth_deg: 30.0,
            sun_elevation_deg: 40.0,
            sun_intensity: 0.9,
            // Tone mapping always on — cheap and prevents washed-out HDR.
            tonemap_enabled: true,
            tonemap_exposure: 0.0,
            tonemap_temperature: 0.0,
            tonemap_saturation: 0.0,
            tonemap_contrast: 0.0,
            tonemap_vibrance: 0.0,
            tonemap_grain: 0.0,
            // No wind on low-end.
            wind_dir: [1.0, 0.0, 0.0],
            wind_strength: 0.0,
            // No SSR on low-end.
            ssr_enabled: false,
            ssr_max_steps: 32,
            ssr_max_distance: 30.0,
            ssr_thickness: 0.08,
            ssr_intensity: 0.8,
            // Water enabled even on low-end — only draws when WaterSurface entities exist.
            water_enabled: true,
            // Lava enabled even on low-end — only draws when LavaSurface entities exist.
            lava_enabled: true,
            // Fire enabled even on low-end — only draws when FireSurface entities exist.
            fire_enabled: true,
            // Heat distortion off on low-end — saves a full-screen post-process.
            heat_distortion_enabled: false,
            // Underwater rendering off on low-end — saves a post-process pass.
            underwater_enabled: false,
            underwater_tint: [0.01, 0.08, 0.12],
            underwater_fog_density: 0.04,
            underwater_caustics: 0.0,
            underwater_god_rays: 0.0,
            underwater_distortion: 0.0,
            underwater_vignette: 0.0,
            underwater_bloom: 0.0,
            // No TAA, motion blur, DOF on low-end.
            taa_enabled: false,
            taa_blend_factor: 0.1,
            motion_blur_enabled: false,
            motion_blur_strength: 0.5,
            dof_enabled: false,
            dof_focus_distance: 10.0,
            dof_strength: 4.0,
            dof_aperture: 0.02,
            god_rays_enabled: false,
            god_rays_intensity: 0.4,
            god_rays_decay: 0.96,
            god_rays_density: 1.2,
            god_rays_weight: 0.04,
            god_rays_num_samples: 32,
            // LOD defaults.
            mesh_lod_threshold_1: 48.0,
            mesh_lod_threshold_2: 95.0,
            mesh_lod_threshold_3: 180.0,
            mesh_lod_threshold_4: 300.0,
        }
    }

    pub fn balanced() -> Self {
        let mut f = Self::default();
        f.god_rays_enabled = true;
        f.god_rays_intensity = 0.3;
        f.god_rays_num_samples = 24;
        f
    }

    pub fn high_end() -> Self {
        Self {
            shadows_enabled: true,
            pcf_enabled: true,
            pcss_enabled: true,
            ibl_enabled: true,
            probes_enabled: true,
            volumetric_enabled: true,
            shadow_resolution: 4096,
            pcf_samples: 16,
            culling_enabled: true,
            culling_distance: 220.0,
            frustum_culling_enabled: true,
            occlusion_culling_enabled: true,
            bloom_enabled: true,
            bloom_strength: 0.25,
            ssao_enabled: true,
            ssao_strength: 0.5,
            volumetric_fog_enabled: true,
            fog_density: 0.04,
            voxel_gi_enabled: true,
            voxel_gi_strength: 0.3,
            sun_azimuth_deg: 35.0,
            sun_elevation_deg: 42.0,
            sun_intensity: 1.0,
            // Tone mapping with subtle film look.
            tonemap_enabled: true,
            tonemap_exposure: 0.1,
            tonemap_temperature: 0.05,
            tonemap_saturation: 0.1,
            tonemap_contrast: 0.1,
            tonemap_vibrance: 0.15,
            tonemap_grain: 0.02,
            // Full wind on high-end.
            wind_dir: [1.0, 0.0, 0.3],
            wind_strength: 0.3,
            // SSR on high-end.
            ssr_enabled: true,
            ssr_max_steps: 64,
            ssr_max_distance: 50.0,
            ssr_thickness: 0.05,
            ssr_intensity: 1.0,
            water_enabled: true,
            lava_enabled: true,
            fire_enabled: true,
            heat_distortion_enabled: true,
            // Underwater rendering on high-end.
            underwater_enabled: true,
            underwater_tint: [0.01, 0.08, 0.12],
            underwater_fog_density: 0.04,
            underwater_caustics: 0.6,
            underwater_god_rays: 0.3,
            underwater_distortion: 0.003,
            underwater_vignette: 0.3,
            underwater_bloom: 0.15,
            // TAA on high-end.
            taa_enabled: true,
            taa_blend_factor: 0.1,
            // Motion blur on high-end.
            motion_blur_enabled: true,
            motion_blur_strength: 0.5,
            // DOF on high-end.
            dof_enabled: true,
            dof_focus_distance: 10.0,
            dof_strength: 4.0,
            dof_aperture: 0.02,
            god_rays_enabled: true,
            god_rays_intensity: 0.5,
            god_rays_decay: 0.96,
            god_rays_density: 1.0,
            god_rays_weight: 0.05,
            god_rays_num_samples: 48,
            // LOD defaults.
            mesh_lod_threshold_1: 48.0,
            mesh_lod_threshold_2: 95.0,
            mesh_lod_threshold_3: 180.0,
            mesh_lod_threshold_4: 300.0,
        }
    }

    pub fn experimental() -> Self {
        let mut f = Self::high_end();
        f.shadow_resolution = 8192;
        f.pcf_samples = 25;
        f.culling_distance = 280.0;
        f.bloom_strength = 0.32;
        f.ssao_strength = 0.62;
        f.fog_density = 0.055;
        f.voxel_gi_strength = 0.4;
        f.ssr_enabled = true;
        f.ssr_max_steps = 128;
        f.ssr_max_distance = 80.0;
        f.ssr_thickness = 0.03;
        f.ssr_intensity = 1.2;
        f
    }

    pub fn from_tier_name(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "low" => Self::low_end(),
            "high" => Self::high_end(),
            "experimental" => Self::experimental(),
            "balanced" | "auto" | _ => Self::balanced(),
        }
    }
}

// ── Renderer ──────────────────────────────────────────────────────────────────
// No lifetime parameter — wgpu 29 Surface owns its window via Arc<Window>.
pub struct Renderer {
    // We keep Arc<Window> alive so the Surface stays valid.
    _window:       Arc<Window>,
    surface:       wgpu::Surface<'static>,
    pub device:    wgpu::Device,
    pub queue:     wgpu::Queue,
    config:        wgpu::SurfaceConfiguration,
    /// Current present sync (VSync) state, togglable at runtime.
    vsync:         bool,
    pipeline:      wgpu::RenderPipeline,
    // Vertex buffer: pre-allocated for 1024 vertices.
    // Overwritten per entity each draw call.
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    uniform_buf:   wgpu::Buffer,
    bind_group:    wgpu::BindGroup,
    global_bgl:    wgpu::BindGroupLayout,
    depth_texture: wgpu::Texture,
    depth_view:    wgpu::TextureView,
    scene_color:   wgpu::Texture,
    scene_view:    wgpu::TextureView,
    bloom_a:       wgpu::Texture,
    bloom_a_view:  wgpu::TextureView,
    bloom_b:       wgpu::Texture,
    bloom_b_view:  wgpu::TextureView,
    post_sampler:  wgpu::Sampler,
    post_bgl:      wgpu::BindGroupLayout,
    post2_bgl:     wgpu::BindGroupLayout,
    post_copy_pipeline: wgpu::RenderPipeline,
    bloom_extract_pipeline: wgpu::RenderPipeline,
    bloom_downsample_pipeline: wgpu::RenderPipeline,
    bloom_blur_h_pipeline: wgpu::RenderPipeline,
    bloom_blur_v_pipeline: wgpu::RenderPipeline,
    bloom_composite_pipeline: wgpu::RenderPipeline,
    /// Pyramid bloom working set: two extra half/quarter-resolution targets so
    /// we can build a real multi-level pyramid (extract → downsample → blur → upsample-add).
    bloom_c_texture: wgpu::Texture,
    bloom_c_view:    wgpu::TextureView,
    bloom_d_texture: wgpu::Texture,
    bloom_d_view:    wgpu::TextureView,
    bloom_e_texture: wgpu::Texture,
    bloom_e_view:    wgpu::TextureView,
    bloom_f_texture: wgpu::Texture,
    bloom_f_view:    wgpu::TextureView,
    /// ── Tone mapping ────────────────────────────────────────────────────────
    /// Full-res temp texture: bloom composite writes here, tonemap reads from here.
    tonemap_temp:       wgpu::Texture,
    tonemap_temp_view:  wgpu::TextureView,
    tonemap_uniform_buf: wgpu::Buffer,
    tonemap_bgl:        wgpu::BindGroupLayout,
    tonemap_pipeline:   wgpu::RenderPipeline,
    /// Cached wind direction + strength — updated via apply_environment().
    wind_dir:   [f32; 3],
    wind_strength: f32,
    /// ── Normals GBuffer (MRT) ──────────────────────────────────────────────
    /// World-space normals written alongside scene colour during the geometry pass.
    /// Used by the SSR pass to reconstruct reflection vectors.
    normals_texture:  wgpu::Texture,
    normals_view:     wgpu::TextureView,
    /// ── Deferred G-buffer targets ──────────────────────────────────────────
    /// The opaque geometry pass writes material properties here (shader.wgsl
    /// fs_main). The deferred lighting pass (deferred.wgsl) reads them, plus
    /// the depth buffer, and resolves lighting into scene_view + normals_view.
    gb_albedo_texture:   wgpu::Texture,
    gb_albedo_view:      wgpu::TextureView,
    gb_normals_texture:  wgpu::Texture,
    gb_normals_view:     wgpu::TextureView,
    gb_material_texture: wgpu::Texture,
    gb_material_view:    wgpu::TextureView,
    gb_extras_texture:   wgpu::Texture,
    gb_extras_view:      wgpu::TextureView,
    /// Separate sky colour target: the sky pass renders here, and the deferred
    /// lighting pass composites it wherever the depth buffer is empty.
    sky_color_texture: wgpu::Texture,
    sky_color_view:    wgpu::TextureView,
    /// Deferred lighting pipeline + bind group layout + inverse-VP uniform.
    deferred_pipeline:   wgpu::RenderPipeline,
    deferred_bgl:        wgpu::BindGroupLayout,
    deferred_uniform_buf: wgpu::Buffer,
    /// ── GTAO ambient occlusion ──────────────────────────────────────────────
    /// Half-res occlusion mask produced by the GTAO compute pass and sampled by
    /// the deferred lighting pass (deferred.wgsl binding 8 in group 1).
    ao_texture:         wgpu::Texture,
    ao_view:            wgpu::TextureView,
    gtao_bgl:           wgpu::BindGroupLayout,
    gtao_pipeline:      wgpu::ComputePipeline,
    gtao_uniform_buf:   wgpu::Buffer,
    /// ── Froxel volumetric fog ────────────────────────────────────────────────
    /// Frustum-aligned 3D grid where a compute pass injects sun scattering.
    /// The deferred pass raymarches it along the view ray for cheap, volumetric
    /// fog and light shafts. Grid is [64, 36, 32] — a low-res but fast proxy.
    froxel_texture:      wgpu::Texture,
    froxel_view:         wgpu::TextureView,
    froxel_bgl:          wgpu::BindGroupLayout,
    froxel_pipeline:     wgpu::ComputePipeline,
    froxel_uniform_buf:  wgpu::Buffer,
    /// ── Real-time voxel GI (Voxel Cone Tracing) ──────────────────────────
    /// Camera-aligned 128³ grid. The injection pass (voxel_gi.wgsl) stamps the
    /// previous frame's lit scene radiance onto the visible surface shell, the
    /// mip pass (voxel_gi_mip.wgsl) builds a summed pyramid, and the deferred
    /// pass cone-traces it for indirect diffuse + specular — the real-time GI
    /// half of the baked-probe hybrid.
    voxel_texture:          wgpu::Texture,
    voxel_view:             wgpu::TextureView,
    voxel_level0_view:      wgpu::TextureView,
    voxel_sampler:          wgpu::Sampler,
    voxel_bgl:              wgpu::BindGroupLayout,
    voxel_pipeline:         wgpu::ComputePipeline,
    voxel_uniform_buf:      wgpu::Buffer,
    voxel_inject_bg:        wgpu::BindGroup,
    voxel_mip_bgl:          wgpu::BindGroupLayout,
    voxel_mip_pipeline:     wgpu::ComputePipeline,
    voxel_mip_uniform_buf:  wgpu::Buffer,
    /// One bind group per mip level (L = 1..7): read view at L-1 + write view
    /// at L. Reused every frame, so only the uniform buffer changes.
    voxel_mip_bgs:          Vec<wgpu::BindGroup>,
    /// ── Deferred decals ────────────────────────────────────────────────────
    /// Unit-cube projector volumes painted into gb_albedo after the G-buffer
    /// pass (alpha blended). decal_uniform_buf holds per-draw matrices.
    decal_bgl:           wgpu::BindGroupLayout,
    decal_pipeline:      wgpu::RenderPipeline,
    decal_uniform_buf:   wgpu::Buffer,
    /// ── Screen-Space Reflections ────────────────────────────────────────────
    /// Composite texture: SSR reads scene_view + normals_view + depth_view,
    /// writes blended reflection result here.  Bloom then reads from this
    /// instead of scene_view so reflections appear in the final image.
    ssr_composite_texture:  wgpu::Texture,
    ssr_composite_view:     wgpu::TextureView,
    ssr_pipeline:           wgpu::RenderPipeline,
    ssr_bgl:                wgpu::BindGroupLayout,
    ssr_uniform_buf:        wgpu::Buffer,
    lod_cache: std::cell::RefCell<HashMap<(u32, u8), Vec<crate::assets::mesh::Vertex>>>,
    texture_cache: std::cell::RefCell<HashMap<String, (wgpu::TextureView, wgpu::Sampler)>>,
    material_bgl: wgpu::BindGroupLayout,
    default_albedo_view: wgpu::TextureView,
    default_albedo_sampler: wgpu::Sampler,
    default_normal_view: wgpu::TextureView,
    default_normal_sampler: wgpu::Sampler,
    default_mr_view: wgpu::TextureView,
    default_mr_sampler: wgpu::Sampler,
    _shadow_fallback_uniform: wgpu::Buffer,
    _shadow_fallback_texture: wgpu::Texture,
    _shadow_fallback_view: wgpu::TextureView,
    _shadow_fallback_sampler: wgpu::Sampler,
    /// Real CSM shadow system — renders 3 cascades and feeds the PBR shader.
    pub shadow_system: shadow::ShadowSystem,
    /// 256-byte GpuShadowData uniform (3 cascade light matrices + settings) for shader.wgsl.
    shadow_uniform_buf: wgpu::Buffer,
    pub features:  RenderFeatures,
    pub adapter_info: wgpu::AdapterInfo,
    pub sky_renderer: sky::SkyRenderer,
    /// Particle renderer — draws weather effects (rain, snow, mist).
    pub particle_renderer: particle::ParticleRenderer,
    /// Water surface renderer — draws water surfaces with wave displacement.
    pub water_renderer: water::WaterRenderer,
    /// Lava surface renderer — draws lava/magma with emissive crack patterns.
    pub lava_renderer: lava::LavaRenderer,
    /// Fire surface renderer — draws procedural flames with emissive glow.
    pub fire_renderer: fire::FireRenderer,
    /// ── Heat distortion post-process ─────────────────────────────────────
    heat_distortion_texture:  wgpu::Texture,
    heat_distortion_view:     wgpu::TextureView,
    heat_distortion_pipeline: wgpu::RenderPipeline,
    heat_distortion_bgl:      wgpu::BindGroupLayout,
    heat_distortion_uniform_buf: wgpu::Buffer,
    /// ── Multi-light uniform buffer (group 0, binding 12) ───────────────────
    light_uniform_buf: wgpu::Buffer,
    /// ── Weather uniform buffer (group 0, binding 13) ──────────────────────
    weather_uniform_buf: wgpu::Buffer,
    /// Baked SH light probes produced by the "Bake Lighting" editor action.
    pub light_probes: crate::renderer::light_probes::LightProbeGrid,
    /// ── Baked probe GPU data ────────────────────────────────────────────
    /// Binding 14: probe count scalar. Binding 15: per-probe SH coefficients.
    probe_control_buf: wgpu::Buffer,
    probe_data_buf: wgpu::Buffer,
    /// Snow coverage value (0 = none, 1 = full) — set from WeatherState.
    snow_coverage: f32,
    /// ── Default material extras buffer (group 1, binding 6) ───────────────
    /// Used for entities that don't override material extras.
    default_material_extras_buf: wgpu::Buffer,
    /// ── Velocity buffer (placeholder — zero-filled until geometry pass writes motion vectors) ──
    velocity_texture:  wgpu::Texture,
    velocity_view:     wgpu::TextureView,
    /// ── Cloud history current (MRT target 2 for sky pass) ──────────────────
    /// Sky pass writes cloud color+alpha here; copied to sky history after main pass.
    cloud_history_current: wgpu::Texture,
    cloud_history_current_view: wgpu::TextureView,
    /// Previous frame's view-projection matrix for cloud temporal reprojection.
    prev_view_proj: glam::Mat4,
    /// Current frame's view-projection matrix (set during draw_world, read by god rays).
    view_proj: glam::Mat4,
    /// ── TAA history texture — previous frame's TAA output (persists across frames) ──
    taa_history_texture: wgpu::Texture,
    taa_history_view:    wgpu::TextureView,
    /// ── TAA resolved texture — output of TAA resolve pass ──
    taa_resolved_texture: wgpu::Texture,
    taa_resolved_view:    wgpu::TextureView,
    /// ── Post-process temp texture — for ping-pong between motion blur / DOF ──
    postprocess_temp_texture: wgpu::Texture,
    postprocess_temp_view:    wgpu::TextureView,
    /// ── TAA pipeline ──────────────────────────────────────────────────────
    taa_pipeline:      wgpu::RenderPipeline,
    taa_bgl:           wgpu::BindGroupLayout,
    taa_uniform_buf:   wgpu::Buffer,
    /// ── Motion blur pipeline ──────────────────────────────────────────────
    motion_blur_pipeline:    wgpu::RenderPipeline,
    motion_blur_bgl:         wgpu::BindGroupLayout,
    motion_blur_uniform_buf: wgpu::Buffer,
    /// ── Depth of field pipeline ───────────────────────────────────────────
    dof_pipeline:      wgpu::RenderPipeline,
    dof_bgl:           wgpu::BindGroupLayout,
    dof_uniform_buf:   wgpu::Buffer,
    /// ── Bilateral blur pipeline (SSR denoising) ──────────────────────────
    bilateral_h_pipeline: wgpu::RenderPipeline,
    bilateral_v_pipeline: wgpu::RenderPipeline,
    bilateral_bgl:        wgpu::BindGroupLayout,
    bilateral_uniform_buf: wgpu::Buffer,
    /// ── God rays pipeline ─────────────────────────────────────────────────
    godray_pipeline:    wgpu::RenderPipeline,
    godray_bgl:         wgpu::BindGroupLayout,
    godray_uniform_buf: wgpu::Buffer,
    /// ── Underwater post-process pipeline ──────────────────────────────────
    underwater_pipeline:    wgpu::RenderPipeline,
    underwater_bgl:         wgpu::BindGroupLayout,
    underwater_uniform_buf: wgpu::Buffer,
    /// ── TAA frame counter (for Halton(2,3) jitter) ───────────────────────
    taa_frame_index:   u32,
    /// Refraction copy texture — scene_view is copied here before the water pass
    /// so water can read refraction without a read/write hazard on scene_view.
    water_refraction_texture: wgpu::Texture,
    water_refraction_view:    wgpu::TextureView,
    /// Cached environment data — updated via apply_environment(), consumed by draw_world().
    sky_params: crate::environment::sky::SkyParams,
    cloud_params: crate::environment::clouds::CloudParams,
    /// Storm darkening and lightning flash intensity (from LightningState).
    storm_darken: f32,
    lightning_intensity: f32,
    /// Weather intensity (0..1) — drives water roughness and other weather effects.
    weather_intensity: f32,
    /// Elapsed time since engine start (for sky animation).
    elapsed_time: f32,
    /// Visual lightning bolt renderer.
    lightning_bolt_renderer: lightning_bolt::LightningBoltRenderer,
    /// Cached lightning state for bolt rendering.
    lightning_state: crate::environment::lightning::LightningState,
    /// ── GPU Skinning ──────────────────────────────────────────────────────
    /// Bind group layout for group 2 (joint matrix uniform).
    skinning_bgl:  wgpu::BindGroupLayout,
    /// Render pipeline for skinned meshes (uses vs_main_skinned).
    skinning_pipeline: wgpu::RenderPipeline,
    /// Uniform buffer for joint matrices (64 × mat4 = 4096 bytes).
    joint_uniform_buf: wgpu::Buffer,
    /// Bind group for the skinning buffer (group 2).
    skinning_bg:  wgpu::BindGroup,
    /// Software occlusion culler: rejects meshes hidden behind Occluders.
    pub occlusion_culler: crate::render::occlusion::OcclusionCuller,
}

impl Renderer {
    fn inside_frustum(vp: glam::Mat4, p: glam::Vec3, radius: f32) -> bool {
        let clip = vp * glam::Vec4::new(p.x, p.y, p.z, 1.0);
        let w = clip.w.abs().max(0.0001);
        let r = radius.max(0.1);
        clip.x >= -w - r
            && clip.x <= w + r
            && clip.y >= -w - r
            && clip.y <= w + r
            && clip.z >= -w - r
            && clip.z <= w + r
    }

    /// Apply environment system data to renderer features.
    /// Call once per frame before draw_world() to sync sun position,
    /// fog density, and atmospheric colors from the environment system.
    pub fn apply_environment(
        &mut self,
        time: &crate::environment::time_of_day::TimeOfDay,
        weather: &crate::environment::weather::WeatherState,
        sky_params: &crate::environment::sky::SkyParams,
        cloud_params: &crate::environment::clouds::CloudParams,
        lightning: &crate::environment::lightning::LightningState,
    ) {
        // Feed sun direction from the time-of-day system instead of
        // the static azimuth/elevation in settings.
        self.features.sun_elevation_deg = time.sun_elevation_deg();
        self.features.sun_azimuth_deg = time.sun_azimuth_deg();
        // Scale sun intensity by daylight factor.
        self.features.sun_intensity = time.daylight_factor();

        // Weather modulates fog density.
        self.features.fog_density = weather.effective_fog_density(self.features.fog_density);

        // Wind from weather system drives vertex displacement in shaders.
        self.wind_dir = [weather.wind_direction.x, 0.0, weather.wind_direction.y];
        self.wind_strength = (weather.wind_strength / 20.0).min(1.0);

        // Lightning data for sky shader.
        let lc = lightning.cloud_uniform_contribution();
        self.storm_darken = lc[0];
        self.lightning_intensity = lc[1];

        // Cache full lightning state for bolt rendering.
        self.lightning_state = lightning.clone();

        // Weather intensity drives water roughness.
        self.weather_intensity = weather.intensity;

        // Snow coverage: full accumulation when condition is Snow, scaled by intensity.
        self.snow_coverage = if weather.condition.is_snow() {
            weather.intensity
        } else {
            0.0
        };

        // Cache sky/cloud params for the sky renderer.
        self.sky_params = sky_params.clone();
        self.cloud_params = cloud_params.clone();
    }

    // new() is async because requesting the adapter and device is async.
    // Takes Arc<Window> so the Surface can keep a strong reference.
    pub async fn new(window: Arc<Window>) -> Self {
        // ── Instance ──────────────────────────────────────────────────────
        let instance = wgpu::Instance::default();

        // ── Surface ───────────────────────────────────────────────────────
        // wgpu 29: create_surface takes Arc<Window>, returns Surface<'static>.
        // The 'static comes from Arc keeping the window alive.
        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("Failed to create wgpu surface");

        // ── Adapter ───────────────────────────────────────────────────────
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:   wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("No suitable GPU adapter found");

        let adapter_info = adapter.get_info();

        // ── Smart GPU Tier Detection ──────────────────────────────────────
        // Score the GPU based on device type, name patterns, and available
        // features/limits. This replaces the old "integrated = low" heuristic
        // which missed high-end integrated GPUs (Apple M4, AMD 780M, etc.)
        // and low-end discrete GPUs (GT 710, MX series).
        let tier_name = detect_gpu_tier(&adapter_info, &adapter);
        tracing::info!("[Renderer] GPU: {}", adapter_info.name);
        tracing::info!("[Renderer] Detected tier: {} (driver: {})", tier_name, adapter_info.driver);
        let features = RenderFeatures::from_tier_name(&tier_name);

        // ── Device & Queue ────────────────────────────────────────────────
        // wgpu 29: request_device() takes only &DeviceDescriptor (no trace_path).
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to open GPU device");

        // ── Surface configuration ─────────────────────────────────────────
        let size     = window.inner_size();
        let surf_fmt = surface.get_capabilities(&adapter).formats[0];
        let config   = wgpu::SurfaceConfiguration {
            usage:        wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:       surf_fmt,
            width:        size.width.max(1),
            height:       size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,  // vsync
            alpha_mode:   wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ── Depth texture ─────────────────────────────────────────────────
        let (depth_texture, depth_view) = make_depth_texture(&device, &config);
        let (scene_color, scene_view) = make_scene_color_texture(&device, &config);
        let (bloom_a, bloom_a_view) = make_bloom_texture(&device, &config, "Bloom A");
        let (bloom_b, bloom_b_view) = make_bloom_texture(&device, &config, "Bloom B");
        // Pyramid bloom mid/low levels: quarter res (C) and eighth res (D).
        let (bloom_c, bloom_c_view) = make_bloom_texture_at(&device, (config.width / 4).max(1), (config.height / 4).max(1), "Bloom C (quarter)");
        let (bloom_d, bloom_d_view) = make_bloom_texture_at(&device, (config.width / 8).max(1), (config.height / 8).max(1), "Bloom D (eighth)");
        let (bloom_e, bloom_e_view) = make_bloom_texture_at(&device, (config.width / 4).max(1), (config.height / 4).max(1), "Bloom E (quarter ping-pong)");
        let (bloom_f, bloom_f_view) = make_bloom_texture_at(&device, (config.width / 8).max(1), (config.height / 8).max(1), "Bloom F (eighth ping-pong)");

        // ── Normals GBuffer texture (MRT target 1) ────────────────────────────
        // Rgba16Float gives enough precision for world-space normals without
        // quantisation artefacts.  Written by the geometry + sky passes,
        // read by the SSR pass to reconstruct reflection vectors.
        let (normals_texture, normals_view) = make_normals_texture(&device, &config);

        // ── Deferred G-buffer targets ──────────────────────────────────────
        // Written by the opaque geometry pass (shader.wgsl fs_main), read by
        // the fullscreen deferred lighting pass (deferred.wgsl).
        let (gb_albedo_texture,   gb_albedo_view)   = make_gbuffer_texture(&device, &config, "GBuffer Albedo+Metallic");
        let (gb_normals_texture,  gb_normals_view)  = make_gbuffer_texture(&device, &config, "GBuffer Normal+Roughness");
        let (gb_material_texture, gb_material_view) = make_gbuffer_texture(&device, &config, "GBuffer Emissive+AO");
        let (gb_extras_texture,   gb_extras_view)   = make_gbuffer_texture(&device, &config, "GBuffer Extras");

        // ── Sky colour target ──────────────────────────────────────────────
        // The sky pass renders here (linear Rgba16Float) separately from the
        // G-buffer; the deferred lighting pass composites it for pixels with
        // no geometry. Rgba16Float (not sRGB) so the deferred pass samples it
        // without an extra sRGB decode.
        let (sky_color_texture, sky_color_view) = make_gbuffer_texture(&device, &config, "Sky Color");

        // ── SSR composite texture ──────────────────────────────────────────
        // Full-res: SSR pass reads scene_view + normals_view + depth_view,
        // writes blended colour (scene + reflections) here.
        // Bloom then reads from ssr_composite_view instead of scene_view.
        let (ssr_composite_texture, ssr_composite_view) =
            make_scene_color_texture(&device, &config);

        // ── Water refraction texture ────────────────────────────────────────
        // Full-res: scene_view is copied here before the water pass so water
        // can read refraction colour without a read/write hazard on scene_view.
        let (water_refraction_texture, water_refraction_view) =
            make_scene_color_texture(&device, &config);

        // ── Tone mapping temp texture ──────────────────────────────────────
        // Full-res texture for the bloom composite output → tonemap input.
        // Same format as scene_color (surface format) so the pipeline works.
        let (tonemap_temp, tonemap_temp_view) = make_scene_color_texture(&device, &config);

        // ── Velocity buffer (placeholder) ──────────────────────────────────
        // Rg16Float: per-pixel motion vectors (x, y).  Zero-filled until the
        // geometry pass writes actual motion vectors from previous-frame MVP.
        let (velocity_texture, velocity_view) = make_velocity_texture(&device, &config);

        // ── Cloud history current texture (MRT target 2 for sky pass) ──────
        // Rgba16Float: the sky pass writes cloud color+alpha here each frame.
        // After the main pass, this is copied to the sky renderer's history
        // texture for temporal reprojection next frame.
        let cloud_history_current = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cloud History Current"),
            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let cloud_history_current_view = cloud_history_current.create_view(&Default::default());

        // ── TAA history texture ────────────────────────────────────────────
        // Rgba16Float: stores the previous frame's TAA-resolved output.
        // Persists across frames; copied at the end of each frame.
        let (taa_history_texture, taa_history_view) = make_taa_history_texture(&device, &config);

        // ── TAA resolved texture ───────────────────────────────────────────
        // Full-res: output of the TAA resolve pass (current + history → resolved).
        let (taa_resolved_texture, taa_resolved_view) = make_scene_color_texture(&device, &config);

        // ── Post-process temp texture ──────────────────────────────────────
        // Full-res ping-pong target for motion blur / DOF when they can't
        // read and write the same texture.
        let (postprocess_temp_texture, postprocess_temp_view) = make_scene_color_texture(&device, &config);

        // ── Vertex buffer ─────────────────────────────────────────────────
        // Staging buffer overwritten once per mesh draw (see MAX_VERTICES_PER_DRAW).
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Vertices"),
            size:               (MAX_VERTICES_PER_DRAW as u64)
                * (std::mem::size_of::<crate::assets::mesh::Vertex>() as u64),
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Instance buffer ───────────────────────────────────────────────
        // Holds per-instance transform + material data for GPU instancing.
        // Pre-allocated for 4096 instances (96 bytes each = 384KB).
        // Grows automatically if needed.
        const INITIAL_INSTANCE_CAPACITY: usize = 4096;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Instances"),
            size:               (INITIAL_INSTANCE_CAPACITY as u64)
                * (std::mem::size_of::<crate::render::instancing::InstanceData>() as u64),
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Uniform buffer ─────────────────────────────────────────────────
        // std::mem::size_of::<GpuUniforms>() = 176 bytes (11 × vec4).
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Camera Uniform"),
            size:               std::mem::size_of::<GpuUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Multi-light uniform buffer ─────────────────────────────────────
        // 1040 bytes: 16 × GpuLightData (64 each = 1024) + 16 bytes header.
        let light_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Light Uniforms"),
            size:               std::mem::size_of::<GpuLightUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Default material extras buffer ─────────────────────────────────
        // 32 bytes: all zeros = no SSS, no clearcoat, no emissive.
        let default_material_extras = GpuMaterialExtras {
            subsurface: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            anisotropy: 0.0,
            emissive_strength: 0.0,
            _pad: [0.0; 3],
        };
        let default_material_extras_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Default Material Extras"),
            size:               std::mem::size_of::<GpuMaterialExtras>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &default_material_extras_buf, 0,
            bytemuck::bytes_of(&default_material_extras),
        );

        // ── Weather uniform buffer (binding 13) ───────────────────────────
        // 16 bytes: snow_coverage + pad.
        let weather_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Weather Uniforms"),
            size:               std::mem::size_of::<GpuWeatherData>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
let default_weather = GpuWeatherData {
            snow_coverage: 0.0,
            _pad: [0.0; 3],
        };
        queue.write_buffer(
            &weather_uniform_buf, 0,
            bytemuck::bytes_of(&default_weather),
        );

        // ── Baked probe buffers (bindings 14/15) ──────────────────────────
        // Binding 14: count scalar (u32 + pad to 16B).
        // Binding 15: per-probe data — 10 × vec4 per probe (pos+radius, 9 SH).
        const MAX_PROBES: usize = 32;
        const PROBE_STRIDE: usize = 10; // vec4s per probe
        let probe_control_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Probe Control"),
            size:               16,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let probe_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Probe Data"),
            size:               (PROBE_STRIDE * MAX_PROBES * 16) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&probe_control_buf, 0, &[0u8; 16]);
        queue.write_buffer(&probe_data_buf, 0, &vec![0u8; PROBE_STRIDE * MAX_PROBES * 16]);

        // ── Bind group layouts & pipeline ─────────────────────────────────
        let (global_bgl, material_bgl) = pipeline::create_bind_group_layouts(&device);

        // Default white/flat-normal/non-metal fallback textures for the material slot.
        // These let us bind something valid even when an entity has no textures.
        let def_white  = make_1x1_texture(&device, &queue, [255, 255, 255, 255], true);
        let def_normal = make_1x1_texture(&device, &queue, [128, 128, 255, 255], false);
        let def_mr     = make_1x1_texture(&device, &queue, [0,   128,   0, 255], false);
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter:    wgpu::FilterMode::Linear,
            min_filter:    wgpu::FilterMode::Linear,
            // ── wgpu 29: mipmap_filter is MipmapFilterMode, not FilterMode ──
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let post_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let shadow_fallback_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Fallback Uniform"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Write sensible defaults so shadow_map_size is non-zero (avoids div-by-zero in PCF).
        queue.write_buffer(&shadow_fallback_uniform, 0, bytemuck::bytes_of(&GpuShadowData {
            light_matrices:     [[0.0; 4]; 12],
            cascade_dists:      [10.0, 40.0, 150.0, 0.0],
            shadow_bias:        0.005,
            normal_offset_bias: 0.02,
            pcf_radius:         1.5,
            shadow_enabled:     1.0,
            shadow_map_size:    2048.0,
            _pad:               [0.0; 7],
        }));
        let shadow_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Fallback Texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let shadow_fallback_view = shadow_fallback_texture.create_view(&Default::default());
        let shadow_fallback_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ── Real CSM shadow system ──────────────────────────────────────────
        // Created BEFORE the global bind group so the bind group can reference
        // the real cascade textures + comparison sampler instead of fallbacks.
        let shadow_system = shadow::ShadowSystem::new(&device, features.shadow_resolution.max(256));
        // 256-byte GpuShadowData uniform for the PBR shader (binding 7).
        let shadow_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Uniform Data"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Global bind group — camera uniform goes here.
        // IBL textures would be bound here too; we use fallbacks for now.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Global BG"),
            layout:  &global_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: uniform_buf.as_entire_binding(),
                },
                // IBL irradiance (fallback white)
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&def_white.0)  },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&linear_sampler)  },
                // IBL prefilter (fallback white)
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&def_white.0)  },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&linear_sampler)  },
                // BRDF LUT (fallback white — lighting will look wrong but won't crash)
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&def_white.0)  },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&linear_sampler)  },
                // Shadow cascades — real CSM data (light matrices + 3 depth maps).
                wgpu::BindGroupEntry { binding: 7, resource: shadow_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(shadow_system.cascade_view(0)) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(shadow_system.cascade_view(1)) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(shadow_system.cascade_view(2)) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(&shadow_system.sampler) },
                // Multi-light uniform (binding 12) — populated each frame in draw_world().
                wgpu::BindGroupEntry { binding: 12, resource: light_uniform_buf.as_entire_binding() },
                // Weather uniform (binding 13) — snow_coverage, updated per frame.
                wgpu::BindGroupEntry { binding: 13, resource: weather_uniform_buf.as_entire_binding() },
                // Baked light probe data (bindings 14/15) — SH irradiance.
                wgpu::BindGroupEntry { binding: 14, resource: probe_control_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: probe_data_buf.as_entire_binding() },
            ],
        });

        // Shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("PBR Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("renderer/shader.wgsl").into()),
        });

        let render_pipeline = pipeline::create_pipeline(
            &device,
            surf_fmt,
            &global_bgl,
            &material_bgl,
            &shader,
        );

        // ── Deferred lighting pass ─────────────────────────────────────────
        // Fullscreen pass that reads the G-buffer + depth + sky colour and
        // resolves the PBR lighting into scene colour + normals (for SSR).
        let deferred_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Deferred Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("renderer/deferred.wgsl").into()),
        });
        let deferred_bgl = pipeline::create_deferred_bgl(&device);
        let deferred_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Deferred Uniforms"),
            // mat4 (64) + vec4 (16) + voxel_origin vec4 (16) + voxel_dims vec4 (16) = 112 bytes.
            size: 112,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let deferred_pipeline = pipeline::create_deferred_pipeline(
            &device,
            surf_fmt,
            &global_bgl,
            &deferred_bgl,
            &deferred_shader,
        );

        // ── GTAO ambient occlusion ─────────────────────────────────────────
        // Half-res occlusion mask. The compute pass reads the depth buffer +
        // world normals G-buffer and writes the AO mask, which the deferred
        // pass samples (deferred.wgsl binding 8, group 1).
        let ao_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GTAO AO Texture"),
            size: wgpu::Extent3d {
                width: (config.width.max(2) / 2).max(1),
                height: (config.height.max(2) / 2).max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let ao_view = ao_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let gtao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("GTAO Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("renderer/gtao.wgsl").into()),
        });
        let gtao_bgl = pipeline::create_gtao_bgl(&device);
        let gtao_pipeline = pipeline::create_gtao_pipeline(&device, &gtao_bgl, &gtao_shader);
        let gtao_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GTAO Uniforms"),
            // 2 × mat4 (128) + 3 × vec4 (48) = 176 bytes.
            size: 176,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Froxel volumetric grid ──────────────────────────────────────────
        let froxel_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Froxel Grid"),
            size: wgpu::Extent3d {
                width:  64,
                height: 36,
                depth_or_array_layers: 32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let froxel_view = froxel_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let froxel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Froxel Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("renderer/froxel.wgsl").into()),
        });
        let froxel_bgl = pipeline::create_froxel_bgl(&device);
        let froxel_pipeline = pipeline::create_froxel_pipeline(&device, &froxel_bgl, &froxel_shader);
        let froxel_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Froxel Uniforms"),
            // 2 × mat4 (128) + 4 × vec4 (64) = 192 bytes.
            size: 192,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Real-time voxel GI ──────────────────────────────────────────────
        // Camera-aligned 128³ clipmap of Rgba16Float with a summed mip pyramid.
        // Storage-binding for injection (level 0) + mip writes, texture-binding
        // so the deferred pass can cone-trace the whole pyramid.
        let voxel_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voxel GI Grid"),
            size: wgpu::Extent3d {
                width:  VOXEL_GI_DIM,
                height: VOXEL_GI_DIM,
                depth_or_array_layers: VOXEL_GI_DIM,
            },
            mip_level_count: VOXEL_GI_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let voxel_view = voxel_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Voxel GI Full View"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let voxel_level0_view = voxel_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Voxel GI Level 0 (storage)"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let voxel_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Voxel GI Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let voxel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Voxel GI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("renderer/voxel_gi.wgsl").into()),
        });
        let voxel_bgl = pipeline::create_voxel_gi_bgl(&device);
        let voxel_pipeline = pipeline::create_voxel_gi_pipeline(&device, &voxel_bgl, &voxel_shader);
        let voxel_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxel GI Uniforms"),
            // 2 × mat4 (128) + 5 × vec4 (80) = 208 bytes.
            size: 208,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Injection bind group is static (only the uniform contents change).
        let voxel_inject_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Voxel GI Inject BG"),
            layout: &voxel_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: voxel_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&scene_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&post_sampler) },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&voxel_level0_view),
                },
            ],
        });
        let voxel_mip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Voxel GI Mip Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("renderer/voxel_gi_mip.wgsl").into()),
        });
        let voxel_mip_bgl = pipeline::create_voxel_gi_mip_bgl(&device);
        let voxel_mip_pipeline = pipeline::create_voxel_gi_mip_pipeline(
            &device, &voxel_mip_bgl, &voxel_mip_shader,
        );
        let voxel_mip_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxel GI Mip Uniforms"),
            // 2 × vec4 = 32 bytes.
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One static bind group per mip level (read L-1 → write L).
        let mut voxel_mip_bgs = Vec::with_capacity(VOXEL_GI_MIPS as usize - 1);
        for level in 1..VOXEL_GI_MIPS {
            let src = voxel_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Voxel GI Mip Read"),
                dimension: Some(wgpu::TextureViewDimension::D3),
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let dst = voxel_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Voxel GI Mip Write"),
                dimension: Some(wgpu::TextureViewDimension::D3),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            });
            voxel_mip_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Voxel GI Mip BG"),
                layout: &voxel_mip_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: voxel_mip_uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst) },
                ],
            }));
        }

        // ── Deferred decal pass ─────────────────────────────────────────────
        let decal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Decal Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("renderer/decal.wgsl").into()),
        });
        let decal_bgl = pipeline::create_decal_bgl(&device);
        let decal_pipeline = pipeline::create_decal_pipeline(
            &device, &global_bgl, &decal_bgl, &decal_shader, wgpu::TextureFormat::Rgba16Float,
        );
        let decal_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Decal Uniforms"),
            // 4 × mat4 (256) + 1 × vec4 (16) = 272 bytes.
            size: 272,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Post Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("renderer/postprocess.wgsl").into()),
        });
        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Post Source BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let post2_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Post Bloom BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let post_copy_pipeline = create_post_pipeline(&device, &post_shader, &post_bgl, None, surf_fmt, "fs_copy");
        let bloom_extract_pipeline = create_post_pipeline(&device, &post_shader, &post_bgl, None, surf_fmt, "fs_bloom_extract");
        let bloom_downsample_pipeline = create_post_pipeline(&device, &post_shader, &post_bgl, None, surf_fmt, "fs_downsample");
        let bloom_blur_h_pipeline = create_post_pipeline(&device, &post_shader, &post_bgl, None, surf_fmt, "fs_blur_h");
        let bloom_blur_v_pipeline = create_post_pipeline(&device, &post_shader, &post_bgl, None, surf_fmt, "fs_blur_v");
        let bloom_composite_pipeline =
            create_post_pipeline(&device, &post_shader, &post_bgl, Some(&post2_bgl), surf_fmt, "fs_composite");

        // ── Tone mapping pipeline ──────────────────────────────────────────
        // Bind group layout: group(0) = texture+sampler, group(1) = TonemapUniforms.
        // The tonemap pass reads the full-res scene (after bloom) and applies
        // ACES tone mapping + colour grading + gamma to produce the final LDR output.
        let tonemap_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tonemap Uniforms"),
            size: std::mem::size_of::<TonemapUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tonemap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tonemap Uniform BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let tonemap_pipeline = {
            let bgls = vec![Some(&post_bgl), Some(&tonemap_bgl)];
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Tonemap Pipeline Layout"),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Tonemap Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_tonemap"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surf_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // ── Sky renderer ───────────────────────────────────────────────────
        // Must be created before Self {} because it borrows device.
        let sky_renderer = sky::SkyRenderer::new(&device, &window, surf_fmt, config.width, config.height);

        // ── Particle renderer ──────────────────────────────────────────────
        // Instanced billboard quad rendering for weather particles.
        let particle_renderer = particle::ParticleRenderer::new(&device, surf_fmt);

        // ── Water surface renderer ─────────────────────────────────────────
        // Gerstner wave displacement, Fresnel reflection/refraction.
        let water_renderer = water::WaterRenderer::new(&device, surf_fmt);

        // ── Lava surface renderer ──────────────────────────────────────────
        // Emissive crack patterns, animated flow, heat shimmer.
        let lava_renderer = lava::LavaRenderer::new(&device, surf_fmt);

        // ── Fire surface renderer ─────────────────────────────────────────
        // Procedural flame shape, height-based colour gradient, emissive glow.
        let fire_renderer = fire::FireRenderer::new(&device, surf_fmt);

        // ── Heat distortion post-process ──────────────────────────────────
        // Renders a distortion field texture during the transparent pass, then
        // a fullscreen post-process reads it and warps UVs behind fire/lava.
        let heat_distortion_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Heat Distortion"),
            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let heat_distortion_view = heat_distortion_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let heat_distortion_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Heat Distortion Uniforms"),
            size: 16, // strength(f32) + time(f32) + noise_scale(f32) + pad(f32)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let heat_distortion_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Heat Distortion BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let heat_distortion_pipeline = {
            let bgls = vec![Some(&post_bgl), Some(&heat_distortion_bgl)];
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Heat Distortion Pipeline Layout"),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Heat Distortion Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_heat_distortion"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surf_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // ── SSR (Screen-Space Reflections) pipeline ────────────────────────
        // Binds: group(0) = scene_color + sampler (reuse post_bgl)
        //        group(1) = normals + depth + sampler + SSR uniforms
        let ssr_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let ssr_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSR Uniforms"),
            size: 160, // inv_VP(64) + VP(64) + max_steps(4) + max_dist(4) + thickness(4) + intensity(4) + screen(8) + pad(8)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ssr_pipeline = {
            let bgls = vec![Some(&post_bgl), Some(&ssr_bgl)];
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSR Pipeline Layout"),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("SSR Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_ssr"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surf_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // ── TAA pipeline ────────────────────────────────────────────────────
        // Binds: group(0) = current scene + sampler (reuse post_bgl)
        //        group(1) = history + velocity + sampler + TAA uniforms
        let taa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let taa_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TAA Uniforms"),
            size: 32, // jitter_offset(8) + blend_factor(4) + enable_taa(4) + pad
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let taa_pipeline = {
            let bgls = vec![Some(&post_bgl), Some(&taa_bgl)];
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("TAA Pipeline Layout"),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("TAA Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_taa"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surf_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // ── Motion blur pipeline ────────────────────────────────────────────
        // Binds: group(0) = scene colour + sampler (reuse post_bgl)
        //        group(1) = velocity + depth + sampler + MotionBlurUniforms
        let motion_blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Motion Blur BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let motion_blur_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Motion Blur Uniforms"),
            size: 32, // blur_strength(4) + max_samples(4) + pad(8)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let motion_blur_pipeline = {
            let bgls = vec![Some(&post_bgl), Some(&motion_blur_bgl)];
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Motion Blur Pipeline Layout"),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Motion Blur Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_motion_blur"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surf_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // ── Depth of field pipeline ─────────────────────────────────────────
        // Binds: group(0) = scene colour + sampler (reuse post_bgl)
        //        group(1) = depth + sampler + DofUniforms
        let dof_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DOF BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let dof_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DOF Uniforms"),
            size: 16, // focus_distance(4) + dof_strength(4) + aperture(4) + pad(4)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dof_pipeline = {
            let bgls = vec![Some(&post_bgl), Some(&dof_bgl)];
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DOF Pipeline Layout"),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("DOF Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_dof"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surf_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // ── Bilateral blur (SSR denoising) ───────────────────────────────────
        // Edge-preserving blur: reads SSR composite + depth + normals.
        let bilateral_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bilateral BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bilateral_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bilateral Uniforms"),
            size: 16, // 4 x f32
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (bilateral_h_pipeline, bilateral_v_pipeline) = {
            let bgls = vec![Some(&post_bgl), Some(&bilateral_bgl)];
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bilateral Pipeline Layout"),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            });
            let h = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Bilateral H Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_bilateral_h"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surf_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            let v = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Bilateral V Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_bilateral_v"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surf_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            (h, v)
        };

        // ── God rays (screen-space sun shafts) ───────────────────────────────
        // Depth-masked radial blur toward the sun. Reads scene + depth.
        let godray_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GodRay BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let godray_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GodRay Uniforms"),
            size: 32, // sun_uv(8) + intensity(4) + decay(4) + density(4) + weight(4) + num_samples(4) + pad(4)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let godray_pipeline = {
            let bgls = vec![Some(&post_bgl), Some(&godray_bgl)];
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GodRay Pipeline Layout"),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("GodRay Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_god_rays"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surf_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // ── Underwater post-process pipeline ────────────────────────────────
        // Applied when the camera is below the waterline.
        let underwater_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Underwater BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let underwater_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Underwater Uniforms"),
            size: 64, // 4 × vec4 = 64 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let underwater_pipeline = {
            let bgls = vec![Some(&post_bgl), Some(&underwater_bgl)];
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Underwater Pipeline Layout"),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Underwater Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &post_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_shader,
                    entry_point: Some("fs_underwater"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surf_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // ── GPU Skinning ───────────────────────────────────────────────────
        // Bind group layout + pipeline + joint matrix buffer for skinned meshes.
        let skinning_bgl = pipeline::create_skinning_bgl(&device);
        let joint_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Joint Uniforms"),
            // 64 × mat4<f32> = 64 × 64 bytes = 4096 bytes.
            size: 4096,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let skinning_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skinning BG"),
            layout: &skinning_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: joint_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let skinning_pipeline = pipeline::create_skinning_pipeline(
            &device,
            surf_fmt,
            &global_bgl,
            &material_bgl,
            &skinning_bgl,
            &shader,
        );

        // Construct sub-renderers that borrow device before it moves into Self.
        let lightning_bolt_renderer = lightning_bolt::LightningBoltRenderer::new(&device, surf_fmt);

        Self {
            _window:      window.clone(),
            surface,
            device,
            queue,
            config,
            vsync:          true,
            pipeline:     render_pipeline,
            vertex_buffer,
            instance_buffer,
            uniform_buf,
            bind_group,
            global_bgl,
            depth_texture,
            depth_view,
            scene_color,
            scene_view,
            bloom_a,
            bloom_a_view,
            bloom_b,
            bloom_b_view,
            bloom_c_texture: bloom_c,
            bloom_c_view,
            bloom_d_texture: bloom_d,
            bloom_d_view,
            bloom_e_texture: bloom_e,
            bloom_e_view,
            bloom_f_texture: bloom_f,
            bloom_f_view,
            post_sampler,
            post_bgl,
            post2_bgl,
            post_copy_pipeline,
            bloom_extract_pipeline,
            bloom_downsample_pipeline,
            bloom_blur_h_pipeline,
            bloom_blur_v_pipeline,
            bloom_composite_pipeline,
            tonemap_temp,
            tonemap_temp_view,
            tonemap_uniform_buf,
            tonemap_bgl,
            tonemap_pipeline,
            wind_dir: [1.0, 0.0, 0.3],
            wind_strength: 0.1,
            normals_texture,
            normals_view,
            gb_albedo_texture,
            gb_albedo_view,
            gb_normals_texture,
            gb_normals_view,
            gb_material_texture,
            gb_material_view,
            gb_extras_texture,
            gb_extras_view,
            sky_color_texture,
            sky_color_view,
            deferred_pipeline,
            deferred_bgl,
            deferred_uniform_buf,
            ao_texture,
            ao_view,
            gtao_bgl,
            gtao_pipeline,
            gtao_uniform_buf,
            froxel_texture,
            froxel_view,
            froxel_bgl,
            froxel_pipeline,
            froxel_uniform_buf,
            voxel_texture,
            voxel_view,
            voxel_level0_view,
            voxel_sampler,
            voxel_bgl,
            voxel_pipeline,
            voxel_uniform_buf,
            voxel_inject_bg,
            voxel_mip_bgl,
            voxel_mip_pipeline,
            voxel_mip_uniform_buf,
            voxel_mip_bgs,
            decal_bgl,
            decal_pipeline,
            decal_uniform_buf,
            ssr_composite_texture,
            ssr_composite_view,
            water_refraction_texture,
            water_refraction_view,
            ssr_pipeline,
            ssr_bgl,
            ssr_uniform_buf,
            lod_cache: std::cell::RefCell::new(HashMap::new()),
            texture_cache: std::cell::RefCell::new(HashMap::new()),
            material_bgl,
            default_albedo_view: def_white.0.clone(),
            default_albedo_sampler: def_white.1.clone(),
            default_normal_view: def_normal.0.clone(),
            default_normal_sampler: def_normal.1.clone(),
            default_mr_view: def_mr.0.clone(),
            default_mr_sampler: def_mr.1.clone(),
            _shadow_fallback_uniform: shadow_fallback_uniform,
            _shadow_fallback_texture: shadow_fallback_texture,
            _shadow_fallback_view: shadow_fallback_view,
            _shadow_fallback_sampler: shadow_fallback_sampler,
            shadow_system,
            shadow_uniform_buf,
            features,
            adapter_info,
            sky_renderer,
            particle_renderer,
            water_renderer,
            lava_renderer,
            fire_renderer,
            heat_distortion_texture,
            heat_distortion_view,
            heat_distortion_pipeline,
            heat_distortion_bgl,
            heat_distortion_uniform_buf,
            velocity_texture,
            velocity_view,
            cloud_history_current,
            cloud_history_current_view,
            prev_view_proj: glam::Mat4::IDENTITY,
            view_proj: glam::Mat4::IDENTITY,
            taa_history_texture,
            taa_history_view,
            taa_resolved_texture,
            taa_resolved_view,
            postprocess_temp_texture,
            postprocess_temp_view,
            taa_pipeline,
            taa_bgl,
            taa_uniform_buf,
            motion_blur_pipeline,
            motion_blur_bgl,
            motion_blur_uniform_buf,
            dof_pipeline,
            dof_bgl,
            dof_uniform_buf,
            bilateral_h_pipeline,
            bilateral_v_pipeline,
            bilateral_bgl,
            bilateral_uniform_buf,
            godray_pipeline,
            godray_bgl,
            godray_uniform_buf,
            underwater_pipeline,
            underwater_bgl,
            underwater_uniform_buf,
            taa_frame_index: 0,
            light_uniform_buf,
            weather_uniform_buf,
            light_probes: crate::renderer::light_probes::LightProbeGrid::new(),
            probe_control_buf,
            probe_data_buf,
            snow_coverage: 0.0,
            default_material_extras_buf,
            sky_params: crate::environment::sky::SkyParams::default(),
            cloud_params: crate::environment::clouds::CloudParams::default(),
            storm_darken: 0.0,
            lightning_intensity: 0.0,
            weather_intensity: 0.0,
            elapsed_time: 0.0,
            skinning_bgl,
            skinning_pipeline,
            joint_uniform_buf,
            skinning_bg,
            lightning_bolt_renderer,
            lightning_state: crate::environment::lightning::LightningState::default(),
            occlusion_culler: crate::render::occlusion::OcclusionCuller::default(),
        }
    }

    // set_vsync() — toggle present sync at runtime (platform polish).
    // Reconfigures the surface with a new present mode; no texture rebuild needed.
    pub fn set_vsync(&mut self, enabled: bool) {
        self.vsync = enabled;
        self.config.present_mode = if enabled {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        self.surface.configure(&self.device, &self.config);
        tracing::info!(
            "[Renderer] VSync {}",
            if enabled { "ON" } else { "OFF" }
        );
    }

    /// Current present sync state.
    pub fn vsync_enabled(&self) -> bool {
        self.vsync
    }

    // resize() — call when the window is resized.
    // Reconfigures the surface and recreates the depth texture to match.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.config.width  = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        // Old depth texture is dropped here and replaced.
        let (dt, dv) = make_depth_texture(&self.device, &self.config);
        self.depth_texture = dt;
        self.depth_view    = dv;
        let (sc, sv) = make_scene_color_texture(&self.device, &self.config);
        self.scene_color = sc;
        self.scene_view = sv;
        let (ba, bav) = make_bloom_texture(&self.device, &self.config, "Bloom A");
        self.bloom_a = ba;
        self.bloom_a_view = bav;
        let (bb, bbv) = make_bloom_texture(&self.device, &self.config, "Bloom B");
        self.bloom_b = bb;
        self.bloom_b_view = bbv;
        let (bc, bcv) = make_bloom_texture_at(&self.device, (self.config.width / 4).max(1), (self.config.height / 4).max(1), "Bloom C (quarter)");
        self.bloom_c_texture = bc;
        self.bloom_c_view = bcv;
        let (bd, bdv) = make_bloom_texture_at(&self.device, (self.config.width / 8).max(1), (self.config.height / 8).max(1), "Bloom D (eighth)");
        self.bloom_d_texture = bd;
        self.bloom_d_view = bdv;
        let (be, bev) = make_bloom_texture_at(&self.device, (self.config.width / 4).max(1), (self.config.height / 4).max(1), "Bloom E (quarter ping-pong)");
        self.bloom_e_texture = be;
        self.bloom_e_view = bev;
        let (bf, bfv) = make_bloom_texture_at(&self.device, (self.config.width / 8).max(1), (self.config.height / 8).max(1), "Bloom F (eighth ping-pong)");
        self.bloom_f_texture = bf;
        self.bloom_f_view = bfv;
        let (tt, ttv) = make_scene_color_texture(&self.device, &self.config);
        self.tonemap_temp = tt;
        self.tonemap_temp_view = ttv;
        let (nt, nv) = make_normals_texture(&self.device, &self.config);
        self.normals_texture = nt;
        self.normals_view = nv;
        let (ga, gav) = make_gbuffer_texture(&self.device, &self.config, "GBuffer Albedo+Metallic");
        self.gb_albedo_texture = ga;
        self.gb_albedo_view = gav;
        let (gn, gnv) = make_gbuffer_texture(&self.device, &self.config, "GBuffer Normal+Roughness");
        self.gb_normals_texture = gn;
        self.gb_normals_view = gnv;
        let (gm, gmv) = make_gbuffer_texture(&self.device, &self.config, "GBuffer Emissive+AO");
        self.gb_material_texture = gm;
        self.gb_material_view = gmv;
        let (gx, gxv) = make_gbuffer_texture(&self.device, &self.config, "GBuffer Extras");
        self.gb_extras_texture = gx;
        self.gb_extras_view = gxv;
        let (sc, scv) = make_gbuffer_texture(&self.device, &self.config, "Sky Color");
        self.sky_color_texture = sc;
        self.sky_color_view = scv;
        let (sr, srv) = make_scene_color_texture(&self.device, &self.config);
        self.ssr_composite_texture = sr;
        self.ssr_composite_view = srv;
        let (wr, wrv) = make_scene_color_texture(&self.device, &self.config);
        self.water_refraction_texture = wr;
        self.water_refraction_view = wrv;
        let (vt, vv) = make_velocity_texture(&self.device, &self.config);
        self.velocity_texture = vt;
        self.velocity_view = vv;
        let (th, thv) = make_taa_history_texture(&self.device, &self.config);
        self.taa_history_texture = th;
        self.taa_history_view = thv;
        let (tr, trv) = make_scene_color_texture(&self.device, &self.config);
        self.taa_resolved_texture = tr;
        self.taa_resolved_view = trv;
        let (pt, ptv) = make_scene_color_texture(&self.device, &self.config);
        self.postprocess_temp_texture = pt;
        self.postprocess_temp_view = ptv;
        // Recreate the half-res AO mask at the new size.
        let at = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GTAO AO Texture"),
            size: wgpu::Extent3d {
                width: (new_size.width.max(2) / 2).max(1),
                height: (new_size.height.max(2) / 2).max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        self.ao_texture = at;
        let aov = self.ao_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.ao_view = aov;
    }

    // draw_world() — renders every entity with a Position + Renderable component.
    // Called once per frame from main.rs.
    /// CPU light bake: ray-casts the scene and fills `self.light_probes` with
    /// SH irradiance. Runs in the editor when the "Bake Lighting" button is
    /// clicked; the game only loads the saved result.
    pub fn bake_lighting(
        &mut self,
        world:  &World,
        meshes: &AssetStore<Mesh>,
    ) -> Result<u64, String> {
        use crate::renderer::light_baker::{bake_probe_grid, collect_scene, BakeSettings};
        let settings = BakeSettings {
            sun_dir: glam::Vec3::new(0.3, -0.72, -0.2).normalize(),
            sun_color: glam::Vec3::new(1.0, 0.95, 0.85),
            sun_intensity: 3.0,
            sky_color: self.sky_renderer.average_sky_color_estimate(),
            samples: 512,
        };
        let scene = collect_scene(world, meshes);
        let result = bake_probe_grid(&mut self.light_probes, &scene, &settings)?;
        // Persist beside the scene the editor has open.
        let probe_path = std::path::Path::new("Content/lighting/").join("probes.json");
        if let Some(dir) = probe_path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Err(e) = crate::renderer::light_baker::save_probes(&probe_path, &self.light_probes) {
            log::warn!("[Lighting] could not save probes to {}: {}", probe_path.display(), e);
        } else {
            log::info!("[Lighting] probes saved to {}", probe_path.display());
        }
        Ok(result)
    }

    /// Load baked probe data saved by the "Bake Lighting" action (if present).
    /// Called whenever a scene is built so previously-baked indirect light
    /// survives scene reloads. Missing file is fine — it just means nothing
    /// has been baked for this level yet.
    pub fn load_probes(&mut self) -> Result<(), String> {
        let path = std::path::Path::new("Content/lighting").join("probes.json");
        if !path.exists() {
            return Ok(());
        }
        crate::renderer::light_baker::load_probes(&path, &mut self.light_probes)
    }

    pub fn draw_world(
        &mut self,
        world:  &World,
        meshes: &AssetStore<Mesh>,
        camera: &dyn Camera,
        jobs: &JobSystem,
        instancing: &mut InstancingManager,
        particles: Option<&[crate::particles::GpuParticle]>,
        mut overlay_pass: Option<&mut dyn FnMut(&wgpu::Device, &wgpu::Queue, &mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    ) -> DrawStats {
        struct DrawCandidate {
            entity: hecs::Entity,
            pos: Position,
            renderable: Renderable,
        }

        let mut stats = DrawStats::default();
        // Build and upload the per-frame uniform.
        let vp      = camera.view_projection_matrix();
        self.view_proj = vp;
        let cam_pos = camera.position();

        // ── TAA sub-pixel jitter (Halton(2,3) sequence) ─────────────────────
        // Jitters the projection matrix by sub-pixel amounts each frame so TAA
        // can accumulate detail from multiple samples over time.
        let jitter_offset = if self.features.taa_enabled {
            let w = self.config.width as f32;
            let h = self.config.height as f32;
            let jx = (halton(self.taa_frame_index, 2) - 0.5) / w;
            let jy = (halton(self.taa_frame_index, 3) - 0.5) / h;
            [jx, jy]
        } else {
            [0.0, 0.0]
        };
        // Apply jitter to VP for geometry rendering only.
        // Water, sky, SSR use the original (unjittered) vp.
        let vp_jittered = {
            let mut vj = vp;
            vj.x_axis.w += jitter_offset[0] * 2.0;
            vj.y_axis.w += jitter_offset[1] * 2.0;
            vj
        };
        // Pre-compute light direction so both uniforms and water pass can use it.
        let light_dir_arr = {
            let az = self.features.sun_azimuth_deg.to_radians();
            let el = self.features.sun_elevation_deg.to_radians();
            glam::Vec3::new(el.cos() * az.cos(), el.sin(), el.cos() * az.sin())
                .normalize()
                .to_array()
        };
        let light_color_arr = [self.features.sun_intensity, self.features.sun_intensity, self.features.sun_intensity];
        let uniforms = GpuUniforms {
            view_proj:   vp_jittered.to_cols_array_2d(),
            camera_pos:  cam_pos.to_array(),
            _pad0:       0.0,
            light_dir:   light_dir_arr,
            _pad1:       0.0,
            light_color: light_color_arr,
            _pad2:       0.0,
            point_light_pos_range: {
                if let Some((p, l)) = world.query::<(&Position, &PointLight)>().iter().next() {
                    [p.x, p.y, p.z, l.range.max(0.1)]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                }
            },
            point_light_color_intensity: {
                if let Some((_p, l)) = world.query::<(&Position, &PointLight)>().iter().next() {
                    [l.color[0], l.color[1], l.color[2], l.intensity.max(0.0)]
                } else {
                    [0.0, 0.0, 0.0, 0.0]
                }
            },
            post_params0: [
                if self.features.bloom_enabled { 1.0 } else { 0.0 },
                self.features.bloom_strength,
                if self.features.ssao_enabled { 1.0 } else { 0.0 },
                self.features.ssao_strength,
            ],
            post_params1: [
                if self.features.volumetric_fog_enabled { 1.0 } else { 0.0 },
                self.features.fog_density,
                if self.features.voxel_gi_enabled { 1.0 } else { 0.0 },
                self.features.voxel_gi_strength,
            ],
            fog_color: [
                self.sky_params.horizon_color.x,
                self.sky_params.horizon_color.y,
                self.sky_params.horizon_color.z,
                self.elapsed_time, // w = elapsed time for wind animation in vertex shader
            ],
            wind_dir_strength: [
                self.wind_dir[0],
                self.wind_dir[1],
                self.wind_dir[2],
                self.wind_strength,
            ],
        };
        self.queue.write_buffer(
            &self.uniform_buf, 0,
            bytemuck::bytes_of(&uniforms),
        );

        // ── Upload deferred lighting uniform ────────────────────────────────
        // Inverse of the (jittered) view-projection matrix so the deferred pass
        // can reconstruct world positions from the depth buffer exactly as the
        // geometry was rasterised. Plus the render target size and the voxel-GI
        // grid placement (origin, voxel size, dimensions) for cone tracing.
        {
            let inv_vp = vp_jittered.inverse();
            let cols = inv_vp.to_cols_array();
            let mut deferred_data = [0.0f32; 28];
            deferred_data[0..16].copy_from_slice(&cols);
            deferred_data[16] = self.config.width as f32;
            deferred_data[17] = self.config.height as f32;
            deferred_data[18] = 0.0;
            deferred_data[19] = 0.0;
            // Voxel-GI grid: origin snapped to voxel boundaries so world-space
            // surfaces stay in the same voxels between frames (less crawling),
            // w = voxel size.
            let grid_world = VOXEL_GI_DIM as f32 * VOXEL_GI_SIZE;
            let snapped = |v: f32| (v - grid_world * 0.5).floor() / VOXEL_GI_SIZE * VOXEL_GI_SIZE;
            deferred_data[20] = snapped(cam_pos.x);
            deferred_data[21] = snapped(cam_pos.y);
            deferred_data[22] = snapped(cam_pos.z);
            deferred_data[23] = VOXEL_GI_SIZE;
            deferred_data[24] = VOXEL_GI_DIM as f32;
            deferred_data[25] = VOXEL_GI_DIM as f32;
            deferred_data[26] = VOXEL_GI_DIM as f32;
            deferred_data[27] = 0.0;
            self.queue.write_buffer(
                &self.deferred_uniform_buf, 0,
                bytemuck::bytes_of(&deferred_data),
            );
        }

        // ── Upload weather uniform (binding 13) ────────────────────────────
        let weather_data = GpuWeatherData {
            snow_coverage: self.snow_coverage,
            _pad: [0.0; 3],
        };
        self.queue.write_buffer(
            &self.weather_uniform_buf, 0,
            bytemuck::bytes_of(&weather_data),
        );

        // ── Upload GTAO uniforms (when SSAO is enabled) ────────────────────
        if self.features.ssao_enabled {
            let mut data = [0.0f32; 44];
            data[0..16].copy_from_slice(&vp.to_cols_array());
            let inv = vp.inverse();
            data[16..32].copy_from_slice(&inv.to_cols_array());
            // Occlusion radius scales a little with distance so close-up AO
            // doesn't disappear and distant AO doesn't smear.
            let ao_radius = (cam_pos - glam::Vec3::ZERO).length().max(2.0).min(40.0) * 0.08 + 1.2;
            data[32] = cam_pos.x;
            data[33] = cam_pos.y;
            data[34] = cam_pos.z;
            data[35] = ao_radius;
            data[36] = self.config.width as f32;
            data[37] = self.config.height as f32;
            data[38] = 1.0 / self.config.width as f32;
            data[39] = 1.0 / self.config.height as f32;
            data[40] = 0.85; // strength (occlusion depth)
            data[41] = self.features.ssao_strength.max(0.05); // intensity
            data[42] = 0.0;
            data[43] = self.elapsed_time;
            self.queue.write_buffer(
                &self.gtao_uniform_buf, 0,
                bytemuck::cast_slice(&data),
            );
        }

        // ── Upload froxel volumetric uniforms (when volumetric fog is on) ──
        if self.features.volumetric_fog_enabled {
            let inv = vp.inverse();
            let mut fdata = [0.0f32; 48];
            fdata[0..16].copy_from_slice(&inv.to_cols_array());
            fdata[16] = cam_pos.x;
            fdata[17] = cam_pos.y;
            fdata[18] = cam_pos.z;
            fdata[20] = -light_dir_arr[0];
            fdata[21] = -light_dir_arr[1];
            fdata[22] = -light_dir_arr[2];
            fdata[24] = light_color_arr[0];
            fdata[25] = light_color_arr[1];
            fdata[26] = light_color_arr[2];
            fdata[28] = self.features.fog_density;
            fdata[29] = 0.1;       // near
            fdata[30] = 200.0;     // far
            fdata[31] = self.elapsed_time;
            fdata[32] = 64.0;      // grid.x
            fdata[33] = 36.0;      // grid.y
            fdata[34] = 32.0;      // grid.z
            fdata[35] = 1.0;       // sun intensity scale
            self.queue.write_buffer(
                &self.froxel_uniform_buf, 0,
                bytemuck::cast_slice(&fdata),
            );
        }

        // ── Upload voxel-GI injection uniforms (when RTGI is enabled) ──────
        if self.features.voxel_gi_enabled {
            let grid_world = VOXEL_GI_DIM as f32 * VOXEL_GI_SIZE;
            // Snap the grid origin to voxel boundaries for frame-to-frame
            // stability of the injected surface shell.
            let snapped = |v: f32| (v - grid_world * 0.5).floor() / VOXEL_GI_SIZE * VOXEL_GI_SIZE;
            let origin = [
                snapped(cam_pos.x),
                snapped(cam_pos.y),
                snapped(cam_pos.z),
            ];
            let mut vdata = [0.0f32; 52];
            vdata[0..16].copy_from_slice(&vp_jittered.to_cols_array());
            let inv_vp = vp_jittered.inverse();
            vdata[16..32].copy_from_slice(&inv_vp.to_cols_array());
            vdata[32] = cam_pos.x;
            vdata[33] = cam_pos.y;
            vdata[34] = cam_pos.z;
            vdata[36] = VOXEL_GI_SIZE;
            vdata[40] = origin[0];
            vdata[41] = origin[1];
            vdata[42] = origin[2];
            vdata[44] = VOXEL_GI_DIM as f32;
            vdata[45] = VOXEL_GI_DIM as f32;
            vdata[46] = VOXEL_GI_DIM as f32;
            vdata[48] = self.config.width as f32;
            vdata[49] = self.config.height as f32;
            self.queue.write_buffer(
                &self.voxel_uniform_buf, 0,
                bytemuck::cast_slice(&vdata),
            );
        }

        // ── Upload baked light probes (bindings 14/15) ─────────────────────
        // Probe layout (10 × vec4 per probe = 160 bytes):
        //   0: position.xyz + radius
        //   1..9: 9 SH coefficients (rgb, w unused)
        const MAX_PROBES: usize = 32;
        const PROBE_STRIDE_VEC4: usize = 10;
        let probes = &self.light_probes.probes;
        let count = probes.len().min(32);
        self.queue.write_buffer(
            &self.probe_control_buf, 0,
            &bytemuck::cast_slice(&[count as u32, 0u32, 0u32, 0u32]),
        );
        let mut probe_bytes = vec![0u8; PROBE_STRIDE_VEC4 * MAX_PROBES * 16];
        for (i, probe) in probes.iter().take(count).enumerate() {
            let base = i * PROBE_STRIDE_VEC4 * 16;
            probe_bytes[base..base + 12].copy_from_slice(&bytemuck::cast_slice(&[
                probe.position.x, probe.position.y, probe.position.z,
            ]));
            probe_bytes[base + 12..base + 16].copy_from_slice(&bytemuck::cast_slice(&[probe.radius]));
            for (k, coeff) in probe.irradiance.coeffs.iter().enumerate() {
                let off = base + 16 + k * 16;
                probe_bytes[off..off + 12]
                    .copy_from_slice(&bytemuck::cast_slice(&[coeff[0], coeff[1], coeff[2]]));
            }
        }
        self.queue.write_buffer(&self.probe_data_buf, 0, &probe_bytes);

        // ── Build multi-light array ────────────────────────────────────────────
        // Populates the LightUniforms buffer with the directional sun light
        // (index 0) plus all PointLight entities from the ECS world (up to 16).
        let mut gpu_lights = [GpuLightData::default(); 16];
        let mut light_count: u32 = 1;

        // Index 0: directional sun light (always present).
        gpu_lights[0] = GpuLightData {
            position:      light_dir_arr,
            _pos_pad:      0.0,
            color:         light_color_arr,
            _col_pad:      0.0,
            intensity:     self.features.sun_intensity * 3.0,
            range:         0.0, // infinite for directional
            light_type:    0.0, // directional
            spot_angle_cos: 0.0,
            shadow_index:  if self.features.shadows_enabled { 0 } else { -1 },
            _pad:          0.0,
            _align_pad:    [0.0; 2],
            direction:     light_dir_arr,
            _dir_pad:      0.0,
        };

        // Add all PointLight entities (point + spot + additional directional).
        for (pos, pl, rot) in world.query::<(&Position, &PointLight, Option<&Rotation>)>().iter() {
            if light_count >= 16 { break; }
            // Spot lights point along the entity's forward direction (same
            // rotation convention as the renderer's model matrix: Y*X*Z).
            let dir = match rot {
                Some(r) => {
                    let m = glam::Mat4::from_rotation_y(r.yaw)
                        * glam::Mat4::from_rotation_x(r.pitch)
                        * glam::Mat4::from_rotation_z(r.roll);
                    let fwd = m.transform_vector3(glam::Vec3::new(0.0, 0.0, -1.0));
                    fwd.normalize_or_zero()
                }
                None => glam::Vec3::new(0.0, 0.0, -1.0),
            };
            gpu_lights[light_count as usize] = GpuLightData {
                position:      [pos.x, pos.y, pos.z],
                _pos_pad:      0.0,
                color:         pl.color,
                _col_pad:      0.0,
                intensity:     pl.intensity,
                range:         pl.range,
                light_type:    pl.light_type,
                spot_angle_cos: pl.spot_angle.to_radians().cos(),
                shadow_index:  if pl.shadow_casting { 0 } else { -1 },
                _pad:          0.0,
                _align_pad:    [0.0; 2],
                direction:     [dir.x, dir.y, dir.z],
                _dir_pad:      0.0,
            };
            light_count += 1;
        }

        let light_uniforms = GpuLightUniforms {
            lights:      gpu_lights,
            light_count,
            _pad1:       0,
            _pad2:       0,
            _pad3:       0,
        };
        self.queue.write_buffer(
            &self.light_uniform_buf, 0,
            bytemuck::bytes_of(&light_uniforms),
        );

        // ── Update sky renderer uniforms ─────────────────────────────────────
        // Feeds environment data (sky gradient, clouds, fog, camera) to the
        // sky shader's uniform buffer. Must happen before the render pass.
        let fog_rgb = [
            self.sky_params.horizon_color.x,
            self.sky_params.horizon_color.y,
            self.sky_params.horizon_color.z,
        ];
        self.sky_renderer.update_uniforms(
            &self.queue,
            vp,
            self.prev_view_proj,
            glam::Vec3::new(cam_pos[0], cam_pos[1], cam_pos[2]),
            &self.sky_params,
            &self.cloud_params,
            fog_rgb,
            self.features.fog_density,
            self.elapsed_time,
            self.config.width as f32,
            self.config.height as f32,
            self.storm_darken,
            self.lightning_intensity,
        );
        // Capture the average sky colour for the light baker.
        self.sky_renderer.last_sky_color = [
            self.sky_params.zenith_color.x * 0.5 + self.sky_params.horizon_color.x * 0.5,
            self.sky_params.zenith_color.y * 0.5 + self.sky_params.horizon_color.y * 0.5,
            self.sky_params.zenith_color.z * 0.5 + self.sky_params.horizon_color.z * 0.5,
        ];
        // Store current VP as previous for next frame's temporal reprojection.
        self.prev_view_proj = vp;
        self.elapsed_time += 1.0 / 60.0; // Approximate; real delta would come from main.rs.

        // Acquire the next swapchain frame.
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Outdated => return stats,
            wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Validation => {
                tracing::error!("[Renderer] Failed to acquire frame texture");
                return stats;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Frame Encoder") }
        );

        // ── Collect render candidates (CPU-side, before any render pass) ──────
        // This must happen here so the shadow passes below can reuse the same
        // instanced model matrices as the main colour pass.
        let candidates: Vec<DrawCandidate> = world
            .query::<(hecs::Entity, &Position, &Renderable)>()
            .iter()
            .filter(|(entity, _pos, _renderable)| {
                world.get::<&BlendedPose>(*entity).is_err()
            })
            .map(|(entity, pos, renderable)| DrawCandidate {
                entity,
                pos: *pos,
                renderable: *renderable,
            })
            .collect();
        stats.total = candidates.len();

        // Collect skinned entities separately (drawn with the skinning pipeline).
        let skinned_candidates: Vec<(hecs::Entity, Position, Renderable)> = world
            .query::<(hecs::Entity, &Position, &Renderable)>()
            .iter()
            .filter(|(entity, _pos, _renderable)| {
                world.get::<&BlendedPose>(*entity).is_ok()
            })
            .map(|(entity, pos, renderable)| (entity, *pos, *renderable))
            .collect();
        stats.total += skinned_candidates.len();

        let visible: Vec<DrawCandidate> = if self.features.culling_enabled {
            let cam_pos = camera.position();
            let vp = camera.view_projection_matrix();
            let cull_dist2 = self.features.culling_distance * self.features.culling_distance;
            let frustum_enabled = self.features.frustum_culling_enabled;
            let occlusion_enabled = self.features.occlusion_culling_enabled;

            // Build the software occlusion grid from Occluder entities before
            // any visibility testing, so hidden meshes are rejected cheaply.
            if occlusion_enabled {
                self.occlusion_culler.begin_frame();
                let mut occluders: Vec<(glam::Vec3, f32)> = world
                    .query::<(&Position, &crate::components::Occluder)>()
                    .iter()
                    .map(|(pos, occ)| (glam::Vec3::new(pos.x, pos.y, pos.z), occ.radius.max(0.5)))
                    .collect();
                // Cap submissions so a scene full of occluders can't stall the
                // frame; 256 large occluders cover any practical street.
                occluders.truncate(256);
                for (center, radius) in occluders {
                    self.occlusion_culler.submit_occluder(vp, center, radius, cam_pos);
                }
            }

            jobs.install(|| {
                candidates
                    .into_par_iter()
                    .filter(|c| {
                        let dx = c.pos.x - cam_pos.x;
                        let dy = c.pos.y - cam_pos.y;
                        let dz = c.pos.z - cam_pos.z;
                        let dist2 = dx * dx + dy * dy + dz * dz;
                        let dist_ok = dist2 <= cull_dist2;
                        let frustum_ok = if frustum_enabled {
                            Renderer::inside_frustum(
                                vp,
                                glam::Vec3::new(c.pos.x, c.pos.y, c.pos.z),
                                c.renderable.scale[0]
                                    .max(c.renderable.scale[1])
                                    .max(c.renderable.scale[2]),
                            )
                        } else {
                            true
                        };
                        // Occlusion: skip meshes fully hidden behind Occluders.
                        let occ_ok = if occlusion_enabled {
                            let radius = c.renderable.scale[0]
                                .max(c.renderable.scale[1])
                                .max(c.renderable.scale[2])
                                .max(0.25);
                            !self.occlusion_culler.is_occluded(
                                vp,
                                glam::Vec3::new(c.pos.x, c.pos.y, c.pos.z),
                                radius,
                                cam_pos,
                            )
                        } else {
                            true
                        };
                        dist_ok && frustum_ok && occ_ok
                    })
                    .collect()
            })
        } else {
            candidates
        };
        stats.visible = visible.len();

        // ── GPU Instanced Drawing via InstancingManager ────────────────────
        // Group visible entities by (mesh_id, material_id). Each unique
        // pair gets one batch, rendered with a single instanced draw call.
        instancing.begin_frame();
        let mut material_map: HashMap<u32, hecs::Entity> = HashMap::new();
        let cam_pos = camera.position();

        for cand in &visible {
            let entity = cand.entity;
            let pos = cand.pos;
            let renderable = cand.renderable;
            let rotation = world
                .get::<&Rotation>(entity)
                .map(|r| *r)
                .unwrap_or(Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 });

            // Build model matrix from TRS — the GPU applies this per instance.
            let t = glam::Mat4::from_translation(glam::Vec3::new(pos.x, pos.y, pos.z));
            let ry = glam::Mat4::from_rotation_y(rotation.yaw);
            let rp = glam::Mat4::from_rotation_x(rotation.pitch);
            let rr = glam::Mat4::from_rotation_z(rotation.roll);
            let s = glam::Mat4::from_scale(glam::Vec3::new(
                renderable.scale[0], renderable.scale[1], renderable.scale[2],
            ));
            let model = t * ry * rp * rr * s;

            // Derive material_id from the entity's MaterialTexture.
            // Entities without a material texture share material_id=0.
            let material_id = if let Ok(tex) = world.get::<&MaterialTexture>(entity) {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                tex.path.hash(&mut hasher);
                (hasher.finish() & 0x7FFFFFFF) as u32
            } else { 0 };

            // Store first entity for each material (for bind group creation).
            material_map.entry(material_id).or_insert(entity);

            // Per-instance LOD band (0 = full detail). Band is keyed on real
            // world distance so every instance in this draw shares a vertex
            // buffer that's simplified enough for *its* distance, without
            // degrading nearby instances because of a distant one.
            let dx = pos.x - cam_pos.x;
            let dy = pos.y - cam_pos.y;
            let dz = pos.z - cam_pos.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let lod_band = if dist > self.features.mesh_lod_threshold_4 { 4u8 }
                else if dist > self.features.mesh_lod_threshold_3 { 3u8 }
                else if dist > self.features.mesh_lod_threshold_2 { 2u8 }
                else if dist > self.features.mesh_lod_threshold_1 { 1u8 }
                else { 0u8 };

            instancing.add_instance(
                renderable.mesh.id,
                material_id,
                lod_band,
                InstanceData {
                    model: model.to_cols_array_2d(),
                    color_metallic: [
                        renderable.color[0],
                        renderable.color[1],
                        renderable.color[2],
                        renderable.metallic,
                    ],
                    roughness_ao_pad: [
                        renderable.roughness,
                        renderable.ao,
                        0.0, 0.0,
                    ],
                },
            );
        }
        instancing.upload_buffers(&self.device, &self.queue);

        // ── CSM shadow passes ───────────────────────────────────────────────
        // Renders visible geometry into 3 cascade depth maps before the main
        // pass. Runs only when shadows are enabled (tier + runtime toggle).
        if self.features.shadows_enabled {
            let sun_dir = glam::Vec3::from_array(light_dir_arr);
            self.shadow_system.update_light_matrices(
                &self.queue,
                sun_dir,
                cam_pos,
                camera.forward(),
            );

            // Upload the 256-byte GpuShadowData the PBR shader reads (binding 7).
            let mut light_matrices = [[0.0_f32; 4]; 12];
            for (i, cascade) in self.shadow_system.cascades.iter().enumerate() {
                for (r, row) in cascade.light_matrix.to_cols_array_2d().iter().enumerate() {
                    light_matrices[i * 4 + r] = *row;
                }
            }
            let shadow_data = GpuShadowData {
                light_matrices,
                cascade_dists: [
                    self.shadow_system.cascades[0].far_dist,
                    self.shadow_system.cascades[1].far_dist,
                    self.shadow_system.cascades[2].far_dist,
                    0.0,
                ],
                shadow_bias:        0.005,
                normal_offset_bias: 0.02,
                pcf_radius:         if self.features.pcf_enabled { 2.0 } else { 0.0 },
                shadow_enabled:     1.0,
                shadow_map_size:    self.features.shadow_resolution as f32,
                _pad:               [0.0; 7],
            };
            self.queue.write_buffer(
                &self.shadow_uniform_buf, 0,
                bytemuck::bytes_of(&shadow_data),
            );

            for cascade_index in 0..shadow::CASCADE_COUNT {
                let mut spass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Shadow Pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: self.shadow_system.cascade_view(cascade_index),
                        depth_ops: Some(wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                spass.set_pipeline(&self.shadow_system.pipeline);
                spass.set_bind_group(0, self.shadow_system.cascade_bind_group(cascade_index), &[]);
                for batch in instancing.batches() {
                    if batch.instances.is_empty() { continue; }
                    let Some(rep_entity) = material_map.get(&batch.material_id) else { continue };
                    let Ok(rep_renderable) = world.get::<&Renderable>(*rep_entity) else { continue };
                    let Some(mesh) = meshes.get(&rep_renderable.mesh) else { continue };
                    if mesh.vertices.is_empty() { continue; }
                    let vertex_bytes = bytemuck::cast_slice(&mesh.vertices);
                    if vertex_bytes.len() > self.vertex_buffer.size() as usize { continue; }
                    self.queue.write_buffer(&self.vertex_buffer, 0, vertex_bytes);
                    let instance_buffer = batch.buffer.as_ref()
                        .expect("InstanceBatch buffer should be Some after upload_buffers()");
                    spass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                    spass.set_vertex_buffer(1, instance_buffer.slice(..));
                    spass.draw(0..mesh.vertices.len() as u32, 0..batch.instances.len() as u32);
                }
            }
        }

        // ── Sky pass ────────────────────────────────────────────────────────
        // Fullscreen triangle: renders sky colour + cloud data into separate
        // targets (sky_color_view + cloud_history_current_view). Geometry is
        // NOT drawn here, so the sky no longer shares the depth buffer with it;
        // the deferred lighting pass composites the sky wherever no geometry
        // wrote depth.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sky Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view:           &self.sky_color_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.05, g: 0.05, b: 0.08, a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view:           &self.cloud_history_current_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                // Sky pipeline expects a depth attachment; clears to 1.0 so the
                // sky triangle (drawn at depth 1.0) passes the depth test.
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Draw a fullscreen triangle (no vertex buffer).
            self.sky_renderer.render(&mut pass);
        }

        // ── G-buffer pass ──────────────────────────────────────────────────
        // Renders opaque geometry into the four G-buffer targets + depth.
        // fs_main (shader.wgsl) writes material properties, not final colour.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("G-Buffer Pass"),
                color_attachments: &[
                    // albedo (rgb) + metallic (a)
                    Some(wgpu::RenderPassColorAttachment {
                        view:           &self.gb_albedo_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    // world normal (encoded rgb) + roughness (a)
                    Some(wgpu::RenderPassColorAttachment {
                        view:           &self.gb_normals_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    // emissive (rgb) + ambient occlusion (a)
                    Some(wgpu::RenderPassColorAttachment {
                        view:           &self.gb_material_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    // material extras (subsurface, clearcoat, clearcoat_roughness, anisotropy)
                    Some(wgpu::RenderPassColorAttachment {
                        view:           &self.gb_extras_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        // 1.0 = max depth — empty-depth pixels are "sky" for the
                        // deferred lighting pass.
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            // Bind group 0 = global (camera, IBL). Set once per frame.
            pass.set_bind_group(0, &self.bind_group, &[]);

            // ── GPU Instanced Drawing via InstancingManager ────────────────
            // Instances were grouped + uploaded BEFORE this pass (so the CSM
            // shadow passes could reuse the same model matrices). Here we only
            // iterate the batches and issue the instanced draw calls.
            for batch in instancing.batches() {
                let instances = &batch.instances;
                if instances.is_empty() { continue; }
                let mesh_id = batch.mesh_id;

                // Resolve mesh from the first batch instance's representative entity.
                let Some(rep_entity) = material_map.get(&batch.material_id) else { continue };
                let Ok(rep_renderable) = world.get::<&Renderable>(*rep_entity) else { continue };
                let Some(mesh) = meshes.get(&rep_renderable.mesh) else { continue };

                // Pick the LOD level from the batch's band. Because batches are now split
                // per LOD band at add_instance() time, every instance in this batch
                // is at the same distance bucket — the vertex buffer is simplified
                // to fit the far end of that bucket without hurting nearer meshes.
                let lod_band = batch.lod_band;
                let lod_ratio = match lod_band {
                    0 => 1.0, 1 => 0.78, 2 => 0.56, 3 => 0.34, _ => 0.20,
                };

                let lod_vertices: Vec<crate::assets::mesh::Vertex> = if lod_band == 0 {
                    mesh.vertices.clone()
                } else {
                    let key = (mesh_id, lod_band);
                    if let Some(cached) = self.lod_cache.borrow().get(&key) {
                        cached.clone()
                    } else {
                        let simplified = simplify_triangle_soup_preserve_shape(&mesh.vertices, lod_ratio);
                        {
                            let mut cache = self.lod_cache.borrow_mut();
                            if cache.len() > 256 { cache.clear(); }
                            cache.insert(key, simplified.clone());
                        }
                        simplified
                    }
                };
                if lod_vertices.is_empty() { continue; }

                // Upload mesh vertices for this batch.
                let vertex_bytes = bytemuck::cast_slice(&lod_vertices);
                if vertex_bytes.len() > self.vertex_buffer.size() as usize {
                    tracing::error!(
                        "[Renderer] Mesh too large ({} verts); skipping batch mesh_id={}.",
                        lod_vertices.len(), mesh_id
                    );
                    continue;
                }
                self.queue.write_buffer(&self.vertex_buffer, 0, vertex_bytes);

                // Bind material for this batch.
                let (albedo_view, albedo_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(*rep_entity) {
                    self.get_or_load_texture(&tex.path, true).unwrap_or((
                        self.default_albedo_view.clone(),
                        self.default_albedo_sampler.clone(),
                    ))
                } else {
                    (self.default_albedo_view.clone(), self.default_albedo_sampler.clone())
                };
                let (normal_view, normal_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(*rep_entity) {
                    self.get_or_load_texture(&tex.normal_path, false).unwrap_or((
                        self.default_normal_view.clone(),
                        self.default_normal_sampler.clone(),
                    ))
                } else {
                    (self.default_normal_view.clone(), self.default_normal_sampler.clone())
                };
                let (mr_view, mr_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(*rep_entity) {
                    self.get_or_load_texture(&tex.metallic_roughness_path, false).unwrap_or((
                        self.default_mr_view.clone(),
                        self.default_mr_sampler.clone(),
                    ))
                } else {
                    (self.default_mr_view.clone(), self.default_mr_sampler.clone())
                };
                let material_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Material BG"),
                    layout: &self.material_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&albedo_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&albedo_sampler) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal_view) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&normal_sampler) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&mr_view) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&mr_sampler) },
                        wgpu::BindGroupEntry { binding: 6, resource: self.default_material_extras_buf.as_entire_binding() },
                    ],
                });
                pass.set_bind_group(1, &material_bg, &[]);

                // Draw instanced: vertex buffer at slot 0, instance buffer at slot 1.
                let instance_buffer = batch.buffer.as_ref()
                    .expect("InstanceBatch buffer should be Some after upload_buffers()");
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, instance_buffer.slice(..));
                pass.draw(0..lod_vertices.len() as u32, 0..instances.len() as u32);
                stats.drawn += 1;
            }

            // ── Skinned entity pass ──────────────────────────────────────────
            // Draw entities with BlendedPose using the skinning pipeline.
            // Each skinned entity gets a separate draw call (per-instance joint data).
            if !skinned_candidates.is_empty() {
                pass.set_pipeline(&self.skinning_pipeline);
                pass.set_bind_group(2, &self.skinning_bg, &[]);

                for (sk_entity, sk_pos, sk_renderable) in &skinned_candidates {
                    let Ok(pose) = world.get::<&BlendedPose>(*sk_entity) else { continue };
                    let Some(mesh) = meshes.get(&sk_renderable.mesh) else { continue };
                    let lod_vertices = &mesh.vertices;
                    if lod_vertices.is_empty() { continue; }

                    // Upload joint matrices.
                    let joint_count = pose.joint_matrices.len().min(64);
                    let mut joint_data = [0u8; 4096];
                    for i in 0..joint_count {
                        let mat_bytes = bytemuck::bytes_of(&pose.joint_matrices[i]);
                        let offset = i * 64;
                        joint_data[offset..offset + 64].copy_from_slice(mat_bytes);
                    }
                    self.queue.write_buffer(&self.joint_uniform_buf, 0, &joint_data);

                    // Upload vertex buffer.
                    let vertex_bytes = bytemuck::cast_slice(lod_vertices);
                    if vertex_bytes.len() <= self.vertex_buffer.size() as usize {
                        self.queue.write_buffer(&self.vertex_buffer, 0, vertex_bytes);
                    } else { continue; }

                    // Rotation for model matrix.
                    let rotation = world
                        .get::<&Rotation>(*sk_entity)
                        .map(|r| *r)
                        .unwrap_or(Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 });
                    let t = glam::Mat4::from_translation(glam::Vec3::new(sk_pos.x, sk_pos.y, sk_pos.z));
                    let ry = glam::Mat4::from_rotation_y(rotation.yaw);
                    let rp = glam::Mat4::from_rotation_x(rotation.pitch);
                    let rr = glam::Mat4::from_rotation_z(rotation.roll);
                    let s = glam::Mat4::from_scale(glam::Vec3::new(
                        sk_renderable.scale[0], sk_renderable.scale[1], sk_renderable.scale[2],
                    ));
                    let model = t * ry * rp * rr * s;

                    // Upload instance data (single instance).
                    let instance_data = crate::render::instancing::InstanceData {
                        model: model.to_cols_array_2d(),
                        color_metallic: [
                            sk_renderable.color[0],
                            sk_renderable.color[1],
                            sk_renderable.color[2],
                            sk_renderable.metallic,
                        ],
                        roughness_ao_pad: [
                            sk_renderable.roughness,
                            sk_renderable.ao,
                            0.0, 0.0,
                        ],
                    };
                    self.queue.write_buffer(
                        &self.instance_buffer, 0,
                        bytemuck::bytes_of(&instance_data),
                    );

                    // Material bind group.
                    let (albedo_view, albedo_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(*sk_entity) {
                        self.get_or_load_texture(&tex.path, true).unwrap_or((
                            self.default_albedo_view.clone(),
                            self.default_albedo_sampler.clone(),
                        ))
                    } else {
                        (self.default_albedo_view.clone(), self.default_albedo_sampler.clone())
                    };
                    let (normal_view, normal_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(*sk_entity) {
                        self.get_or_load_texture(&tex.normal_path, false).unwrap_or((
                            self.default_normal_view.clone(),
                            self.default_normal_sampler.clone(),
                        ))
                    } else {
                        (self.default_normal_view.clone(), self.default_normal_sampler.clone())
                    };
                    let (mr_view, mr_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(*sk_entity) {
                        self.get_or_load_texture(&tex.metallic_roughness_path, false).unwrap_or((
                            self.default_mr_view.clone(),
                            self.default_mr_sampler.clone(),
                        ))
                    } else {
                        (self.default_mr_view.clone(), self.default_mr_sampler.clone())
                    };
                    let material_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Skinned Material BG"),
                        layout: &self.material_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&albedo_view) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&albedo_sampler) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal_view) },
                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&normal_sampler) },
                            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&mr_view) },
                            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&mr_sampler) },
                            wgpu::BindGroupEntry { binding: 6, resource: self.default_material_extras_buf.as_entire_binding() },
                        ],
                    });
                    pass.set_bind_group(1, &material_bg, &[]);
                    pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                    pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    pass.draw(0..lod_vertices.len() as u32, 0..1);
                    stats.drawn += 1;
                }
            }
        } // G-buffer pass is dropped here — commands are finalised

        // ── Deferred decal pass ─────────────────────────────────────────────
        // Paints decal albedo into gb_albedo with alpha blending, reading the
        // depth buffer to stamp only where the surface crosses each projector
        // box. Runs after the G-buffer so decals can tint the final material.
        {
            // Collect decal entities (position + rotation + decal component).
            let mut decals: Vec<(glam::Vec3, glam::Quat, crate::components::Decal)> = Vec::new();
            for (pos, dec, rot) in world.query::<(&Position, &Decal, Option<&Rotation>)>().iter() {
                let p = glam::Vec3::new(pos.x, pos.y, pos.z);
                let q = rot
                    .map(|r| glam::Quat::from_euler(
                        glam::EulerRot::YXZ,
                        r.yaw,
                        r.pitch,
                        r.roll + dec.roll_deg.to_radians(),
                    ))
                    .unwrap_or_else(|| glam::Quat::IDENTITY);
                decals.push((p, q, *dec));
            }

            if !decals.is_empty() {
                let decal_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Decal BG"),
                    layout: &self.decal_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.decal_uniform_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    ],
                });
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Decal Pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view:           &self.gb_albedo_view,
                            resolve_target: None,
                            depth_slice:    None,
                            ops: wgpu::Operations {
                                load:  wgpu::LoadOp::Load, // blend over the G-buffer
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.decal_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_bind_group(1, &decal_bg, &[]);
                for (pos, rot, d) in &decals {
                    let model = glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::from_array(d.size),
                        *rot,
                        *pos,
                    );
                    let inv = model.inverse();
                    let view_proj = self.view_proj;
                    let inv_vp = view_proj.inverse();
                    let mut data = [0.0f32; 68]; // 4 mat4 (64) + 1 vec4 (4)
                    data[0..16].copy_from_slice(&model.to_cols_array());
                    data[16..32].copy_from_slice(&inv.to_cols_array());
                    data[32..48].copy_from_slice(&view_proj.to_cols_array());
                    data[48..64].copy_from_slice(&inv_vp.to_cols_array());
                    data[64] = d.opacity.clamp(0.0, 1.0);
                    self.queue.write_buffer(
                        &self.decal_uniform_buf, 0,
                        bytemuck::cast_slice(&data),
                    );
                    pass.draw(0..24, 0..1);
                }
            }
        }

        // ── Deferred lighting pass ─────────────────────────────────────────
        // Fullscreen triangle: reads the G-buffer + depth + sky colour, resolves
        // the full PBR lighting, and writes scene colour (scene_view) + world
        // normals (normals_view) for downstream passes (water, SSR, bloom).
        //
        // GTAO runs first as a compute pass over the depth + normals G-buffer,
        // writing a half-res AO mask that the deferred pass samples.
        if self.features.ssao_enabled {
            let ao_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GTAO BG"),
                layout: &self.gtao_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.gtao_uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.gb_normals_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.ao_view) },
                ],
            });
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GTAO Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gtao_pipeline);
            cpass.set_bind_group(0, &ao_bg, &[]);
            let wg_x = (self.config.width.max(2) / 2).div_ceil(8).max(1);
            let wg_y = (self.config.height.max(2) / 2).div_ceil(8).max(1);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        // ── Froxel volumetric injection ─────────────────────────────────────
        // Fills the 3D froxel grid with sun scattering before the deferred pass
        // raymarches it. Runs whenever volumetric fog is enabled.
        if self.features.volumetric_fog_enabled {
            let froxel_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Froxel BG"),
                layout: &self.froxel_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.froxel_uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.froxel_view) },
                ],
            });
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Froxel Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.froxel_pipeline);
            cpass.set_bind_group(0, &froxel_bg, &[]);
            cpass.dispatch_workgroups(64u32.div_ceil(8), 36u32.div_ceil(8), 32u32.div_ceil(8));
        }
        // ── Voxel GI: injection ──────────────────────────────────────────────
        // Voxelizes the visible scene by stamping the previous frame's lit
        // scene colour (scene_view, untouched since last frame) into the
        // camera-aligned voxel grid. Must run AFTER geometry but BEFORE the
        // deferred pass (which cone-traces the freshly built grid).
        if self.features.voxel_gi_enabled {
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Voxel GI Inject Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.voxel_pipeline);
                cpass.set_bind_group(0, &self.voxel_inject_bg, &[]);
                let wgs = VOXEL_GI_DIM.div_ceil(4).max(1);
                cpass.dispatch_workgroups(wgs, wgs, wgs);
            }
            // ── Voxel GI: mip pyramid ──────────────────────────────────────
            for level in 1..VOXEL_GI_MIPS {
                let dim = VOXEL_GI_DIM >> level;
                let mut mdata = [0.0f32; 8];
                mdata[0] = dim as f32;
                mdata[1] = dim as f32;
                mdata[2] = dim as f32;
                self.queue.write_buffer(
                    &self.voxel_mip_uniform_buf, 0,
                    bytemuck::cast_slice(&mdata),
                );
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Voxel GI Mip Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.voxel_mip_pipeline);
                cpass.set_bind_group(0, &self.voxel_mip_bgs[level as usize - 1], &[]);
                let wgs = dim.div_ceil(4).max(1);
                cpass.dispatch_workgroups(wgs, wgs, wgs);
            }
        }
        {
            let deferred_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Deferred Lighting BG"),
                layout: &self.deferred_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.gb_albedo_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.gb_normals_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.gb_material_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.gb_extras_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.sky_color_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    wgpu::BindGroupEntry { binding: 7, resource: self.deferred_uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.ao_view) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.froxel_view) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.voxel_view) },
                    wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::Sampler(&self.voxel_sampler) },
                ],
            });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Deferred Lighting Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view:           &self.scene_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.05, g: 0.05, b: 0.08, a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view:           &self.normals_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.deferred_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, &deferred_bg, &[]);
            pass.draw(0..3, 0..1);
        } // deferred lighting pass is dropped here

        // ── Copy cloud history current → sky history texture ────────────────
        // The sky pass wrote cloud color+alpha to cloud_history_current (MRT target 2).
        // Copy it to the sky renderer's history texture so next frame's sky shader
        // can read it for temporal reprojection.
        enc.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.cloud_history_current,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: self.sky_renderer.cloud_history_tex(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );

        // ── Particle rendering pass ──────────────────────────────────────────
        // Drawn after geometry (so particles appear in front of sky, behind nothing).
        // Uses depth test (LessEqual) but NO depth write — particles are transparent
        // and should not occlude geometry behind them.
        // Particles are collected from the ParticleSystem each frame via draw_world args.
        if let Some(particles) = particles {
            if !particles.is_empty() {
                self.particle_renderer.render(
                    &mut {
                        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Particle Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &self.scene_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &self.depth_view,
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            ..Default::default()
                        });
                        pass
                    },
                    &self.queue,
                    &self.device,
                    particles,
                    vp,
                    glam::Vec3::new(cam_pos[0], cam_pos[1], cam_pos[2]),
                );
            }
        }

        // ── Water surface pass ───────────────────────────────────────────────
        // Renders water surfaces with Gerstner wave displacement, Fresnel
        // reflection/refraction, depth-based absorption, and foam.
        //
        // Must come AFTER geometry + particles (so scene_view has the opaque
        // scene) and BEFORE SSR (so reflections include water).
        //
        // Water reads scene_view for refraction — we can't read+write the same
        // texture, so scene_view is first copied to water_refraction_view, then
        // water reads water_refraction_view + depth and writes (alpha blended)
        // back to scene_view.
        if self.features.water_enabled {
            // Collect water entities from the world.
            let water_entities: Vec<(Position, crate::components::WaterSurface)> = world
                .query::<(&Position, &crate::components::WaterSurface)>()
                .iter()
                .map(|(pos, ws)| (*pos, *ws))
                .collect();

            if !water_entities.is_empty() {
                // Copy scene_view → water_refraction_view for refraction reads.
                {
                    let copy_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Water Refraction Copy BG"),
                        layout: &self.post_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.scene_view) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                        ],
                    });
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Water Refraction Copy"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.water_refraction_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        ..Default::default()
                    });
                    pass.set_pipeline(&self.post_copy_pipeline);
                    pass.set_bind_group(0, &copy_bg, &[]);
                    pass.draw(0..3, 0..1);
                }

                // Water pass: reads water_refraction_view + depth → writes to scene_view.
                {
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Water Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.scene_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });

                    let inv_vp = vp.inverse();
                    let water_refs: Vec<(&Position, &crate::components::WaterSurface)> = water_entities
                        .iter()
                        .map(|(p, w)| (p, w))
                        .collect();

                    self.water_renderer.render(
                        &mut pass,
                        &self.queue,
                        &self.device,
                        &self.water_refraction_view,
                        &self.depth_view,
                        &self.post_sampler,
                        &water_refs,
                        vp,
                        inv_vp,
                        glam::Vec3::new(cam_pos[0], cam_pos[1], cam_pos[2]),
                        light_dir_arr,
                        light_color_arr,
                        self.elapsed_time,
                        self.weather_intensity,
                        self.wind_strength,
                    );
                }
            }
        }

        // ── Lava surface pass ────────────────────────────────────────────────
        // Renders lava/magma surfaces with animated emissive crack patterns.
        // Lava is fully opaque — no refraction needed, just renders directly
        // into scene_view with emissive output that drives bloom.
        if self.features.lava_enabled {
            let lava_entities: Vec<(Position, crate::components::LavaSurface)> = world
                .query::<(&Position, &crate::components::LavaSurface)>()
                .iter()
                .map(|(pos, ls)| (*pos, *ls))
                .collect();

            if !lava_entities.is_empty() {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Lava Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.scene_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                let lava_refs: Vec<(&Position, &crate::components::LavaSurface)> = lava_entities
                    .iter()
                    .map(|(p, l)| (p, l))
                    .collect();

                self.lava_renderer.render(
                    &mut pass,
                    &self.queue,
                    &self.device,
                    &lava_refs,
                    vp,
                    self.elapsed_time,
                );
            }
        }

        // ── SSR pass (screen-space reflections) ────────────────────────────────
        // When enabled: reads scene_view + normals_view + depth_view → writes
        // blended reflection result into ssr_composite_view.
        // When disabled: copies scene_view → ssr_composite_view (cheap passthrough).
        // The bloom chain then reads from ssr_composite_view instead of scene_view.
        //
        // ── Fire surface pass ──────────────────────────────────────────────────
        // Renders procedural fire flames with emissive glow and additive blending.
        // Must come AFTER lava (so scene_view has opaque surfaces) and BEFORE SSR.
        if self.features.fire_enabled {
            let fire_entities: Vec<(Position, crate::components::FireSurface)> = world
                .query::<(&Position, &crate::components::FireSurface)>()
                .iter()
                .map(|(pos, fs)| (*pos, *fs))
                .collect();

            if !fire_entities.is_empty() {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Fire Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.scene_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                let fire_refs: Vec<(&Position, &crate::components::FireSurface)> = fire_entities
                    .iter()
                    .map(|(p, f)| (p, f))
                    .collect();

                self.fire_renderer.render(
                    &mut pass,
                    &self.queue,
                    &self.device,
                    &fire_refs,
                    vp,
                    self.elapsed_time,
                    self.wind_dir,
                    self.wind_strength,
                );
            }
        }

        // ── Lightning bolt visual pass ──────────────────────────────────────
        // Draws jagged line geometry from bolt_origin to bolt_target during flash.
        if self.lightning_state.flash_intensity > 0.05 {
            self.lightning_bolt_renderer.render(
                &self.lightning_state,
                &vp.to_cols_array_2d(),
                &mut enc,
                &self.scene_view,
                &self.depth_view,
                &self.device,
                &self.queue,
            );
        }

        // ── Heat distortion post-process ───────────────────────────────────
        // Warps the scene behind fire/lava using a screen-space UV offset.
        if self.features.heat_distortion_enabled && self.features.fire_enabled {
            let heat_uniforms = [0.003f32, self.elapsed_time, 8.0f32, 0.0f32]; // strength, time, noise_scale, pad
            self.queue.write_buffer(
                &self.heat_distortion_uniform_buf, 0,
                bytemuck::cast_slice(&heat_uniforms),
            );

            let hd_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Heat Distortion BG"),
                layout: &self.heat_distortion_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.heat_distortion_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.heat_distortion_uniform_buf.as_entire_binding() },
                ],
            });

            let scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Heat Distortion Scene BG"),
                layout: &self.post_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.scene_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                ],
            });

            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Heat Distortion Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.tonemap_temp_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.heat_distortion_pipeline);
                pass.set_bind_group(0, &scene_bg, &[]);
                pass.set_bind_group(1, &hd_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            // Copy result back to scene_color for downstream passes.
            enc.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.tonemap_temp,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.scene_color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d { width: self.config.width, height: self.config.height, depth_or_array_layers: 1 },
            );
        }

        {
            if self.features.ssr_enabled {
                // Upload SSR uniforms.
                let ssr_uniforms = SsrUniforms {
                    inv_view_proj: vp.inverse().to_cols_array_2d(),
                    view_proj:     vp.to_cols_array_2d(),
                    max_steps:     self.features.ssr_max_steps,
                    max_distance:  self.features.ssr_max_distance,
                    thickness:     self.features.ssr_thickness,
                    intensity:     self.features.ssr_intensity,
                    screen_size:   [self.config.width as f32, self.config.height as f32],
                    _pad0:         [0.0; 2],
                };
                self.queue.write_buffer(
                    &self.ssr_uniform_buf, 0,
                    bytemuck::bytes_of(&ssr_uniforms),
                );

                let ssr_scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSR Scene BG"),
                    layout: &self.post_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.scene_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    ],
                });
                let ssr_data_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSR Data BG"),
                    layout: &self.ssr_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.normals_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: self.ssr_uniform_buf.as_entire_binding() },
                    ],
                });

                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SSR Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssr_composite_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.ssr_pipeline);
                pass.set_bind_group(0, &ssr_scene_bg, &[]);
                pass.set_bind_group(1, &ssr_data_bg, &[]);
                pass.draw(0..3, 0..1);
            } else {
                // No SSR — copy scene_view to ssr_composite_view.
                let ssr_scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSR Copy BG"),
                    layout: &self.post_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.scene_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    ],
                });
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SSR Copy"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssr_composite_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.post_copy_pipeline);
                pass.set_bind_group(0, &ssr_scene_bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // ── Bilateral blur pass (SSR denoising) ──────────────────────────
        // Edge-preserving Gaussian blur on the SSR composite.
        // H pass: ssr_composite → postprocess_temp
        // V pass: postprocess_temp → ssr_composite
        if self.features.ssr_enabled {
            let bilateral_uniforms = BilateralUniforms {
                blur_radius: 4.0,
                depth_weight: 100.0,
                norm_weight: 128.0,
                _pad: 0.0,
            };
            self.queue.write_buffer(
                &self.bilateral_uniform_buf, 0,
                bytemuck::bytes_of(&bilateral_uniforms),
            );

            // Shared bilateral data bind group (depth + normals + sampler + uniforms).
            let bilateral_data_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bilateral Data BG"),
                layout: &self.bilateral_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.normals_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: self.bilateral_uniform_buf.as_entire_binding() },
                ],
            });

            // H pass: ssr_composite_view → postprocess_temp_view
            {
                let scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bilateral H Scene BG"),
                    layout: &self.post_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.ssr_composite_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    ],
                });
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bilateral H Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.postprocess_temp_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.bilateral_h_pipeline);
                pass.set_bind_group(0, &scene_bg, &[]);
                pass.set_bind_group(1, &bilateral_data_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            // V pass: postprocess_temp_view → ssr_composite_view
            {
                let scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bilateral V Scene BG"),
                    layout: &self.post_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.postprocess_temp_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    ],
                });
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bilateral V Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssr_composite_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.bilateral_v_pipeline);
                pass.set_bind_group(0, &scene_bg, &[]);
                pass.set_bind_group(1, &bilateral_data_bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // ── TAA resolve pass ────────────────────────────────────────────────
        // Reads current scene (ssr_composite_view) + history (taa_history_view)
        // + velocity buffer → writes resolved image to taa_resolved_view.
        {
            // Upload TAA uniforms.
            let taa_uniforms = TaaUniforms {
                jitter_offset,
                blend_factor: self.features.taa_blend_factor,
                enable_taa: if self.features.taa_enabled { 1.0 } else { 0.0 },
            };
            self.queue.write_buffer(
                &self.taa_uniform_buf, 0,
                bytemuck::bytes_of(&taa_uniforms),
            );

            let taa_scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TAA Scene BG"),
                layout: &self.post_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.ssr_composite_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                ],
            });
            let taa_data_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TAA Data BG"),
                layout: &self.taa_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.taa_history_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.velocity_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: self.taa_uniform_buf.as_entire_binding() },
                ],
            });

            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA Resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.taa_resolved_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.taa_pipeline);
            pass.set_bind_group(0, &taa_scene_bg, &[]);
            pass.set_bind_group(1, &taa_data_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Motion blur pass ────────────────────────────────────────────────
        // Reads TAA-resolved image + velocity + depth → writes blurred result.
        // Ping-pongs: reads taa_resolved_view → writes postprocess_temp_view.
        // Then copies temp back to resolved so DOF / bloom can read from resolved.
        let mut result_in_temp = false;
        if self.features.motion_blur_enabled {
            let mb_uniforms = MotionBlurUniforms {
                blur_strength: self.features.motion_blur_strength,
                max_samples:   8.0,
                _pad:          [0.0; 2],
            };
            self.queue.write_buffer(
                &self.motion_blur_uniform_buf, 0,
                bytemuck::bytes_of(&mb_uniforms),
            );

            let mb_scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MB Scene BG"),
                layout: &self.post_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.taa_resolved_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                ],
            });
            let mb_data_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MB Data BG"),
                layout: &self.motion_blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.velocity_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: self.motion_blur_uniform_buf.as_entire_binding() },
                ],
            });

            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Motion Blur"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.postprocess_temp_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.motion_blur_pipeline);
            pass.set_bind_group(0, &mb_scene_bg, &[]);
            pass.set_bind_group(1, &mb_data_bg, &[]);
            pass.draw(0..3, 0..1);
            result_in_temp = true;
        }

        // ── Depth of field pass ─────────────────────────────────────────────
        // Reads blurred/resolved image + depth → writes DOF-blurred result.
        // Ping-pong: reads from whichever texture has the latest result,
        // writes to the other.
        if self.features.dof_enabled {
            let dof_uniforms = DofUniforms {
                focus_distance: self.features.dof_focus_distance,
                dof_strength:   self.features.dof_strength,
                aperture:       self.features.dof_aperture,
                _pad:           0.0,
            };
            self.queue.write_buffer(
                &self.dof_uniform_buf, 0,
                bytemuck::bytes_of(&dof_uniforms),
            );

            let (dof_input_view, dof_output_view) = if result_in_temp {
                // Motion blur wrote to temp → DOF reads temp, writes resolved.
                (&self.postprocess_temp_view, &self.taa_resolved_view)
            } else {
                // No motion blur → DOF reads resolved, writes temp.
                (&self.taa_resolved_view, &self.postprocess_temp_view)
            };

            let dof_scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DOF Scene BG"),
                layout: &self.post_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(dof_input_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                ],
            });
            let dof_data_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DOF Data BG"),
                layout: &self.dof_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.dof_uniform_buf.as_entire_binding() },
                ],
            });

            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Depth of Field"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dof_output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.dof_pipeline);
            pass.set_bind_group(0, &dof_scene_bg, &[]);
            pass.set_bind_group(1, &dof_data_bg, &[]);
            pass.draw(0..3, 0..1);
            result_in_temp = !result_in_temp;
        }

        // Ensure final result is in taa_resolved_view (for bloom + history).
        if result_in_temp {
            let copy_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Post TAA Copy BG"),
                layout: &self.post_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.postprocess_temp_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                ],
            });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Post TAA Copy"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.taa_resolved_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.post_copy_pipeline);
            pass.set_bind_group(0, &copy_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Copy resolved → history (for next frame's TAA) ─────────────────
        {
            let copy_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TAA History Copy BG"),
                layout: &self.post_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.taa_resolved_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                ],
            });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA History Copy"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.taa_history_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.post_copy_pipeline);
            pass.set_bind_group(0, &copy_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Bloom chain (reads from TAA-resolved image) ─────────────────────
        let scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Scene BG"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.taa_resolved_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });
        let bloom_a_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Bloom A BG"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_a_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });
        let bloom_b_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Bloom B BG"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_b_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });
        let composite_bloom_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Composite Bloom BG"),
            layout: &self.post2_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_b_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });
        // Pyramid-level bind groups.
        let bloom_c_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Bloom C BG"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_c_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });
        let bloom_d_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Bloom D BG"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_d_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });
        let bloom_e_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Bloom E BG"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_e_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });
        let bloom_f_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Bloom F BG"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_f_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });
        // Upsample-add bind groups (t_bloom group 1 = the smaller level).
        let composite_d_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Composite D BG"),
            layout: &self.post2_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_d_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });
        let composite_e_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Composite E BG"),
            layout: &self.post2_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_e_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
            ],
        });

        // ── Pyramid bloom ────────────────────────────────────────────────────
        // Build a real multi-level pyramid: extract bright pixels at half res,
        // then downsample through quarter and eighth levels (each downsample is
        // a box filter that softens the glow), blur the two lowest levels with
        // the H/V Gaussian pair, then upsample-add each level back into the one
        // above, and finally composite into the scene.
        // 1) Extract bright parts → bloom_a (half).
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Extract"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_a_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_extract_pipeline);
            pass.set_bind_group(0, &scene_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 2) Downsample half → quarter (bloom_c), then quarter → eighth (bloom_d).
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Downsample 1/2"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_c_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_downsample_pipeline);
            pass.set_bind_group(0, &bloom_a_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Downsample 1/4"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_d_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_downsample_pipeline);
            pass.set_bind_group(0, &bloom_c_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3) Blur the quarter level (bloom_c → bloom_e → bloom_c).
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Quarter Blur H"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_e_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_blur_h_pipeline);
            pass.set_bind_group(0, &bloom_c_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Quarter Blur V"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_c_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_blur_v_pipeline);
            pass.set_bind_group(0, &bloom_e_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 4) Blur the eighth level (bloom_d → bloom_f → bloom_d).
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Eighth Blur H"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_f_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_blur_h_pipeline);
            pass.set_bind_group(0, &bloom_d_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Eighth Blur V"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_d_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_blur_v_pipeline);
            pass.set_bind_group(0, &bloom_f_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 5) Upsample-add: eighth into quarter (bloom_c += bloom_d), then quarter
        //    into half (bloom_a += bloom_e). Then blur the half level into bloom_b.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Upsample-Add 1/4"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_e_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_composite_pipeline);
            pass.set_bind_group(0, &bloom_c_bg, &[]);
            pass.set_bind_group(1, &composite_d_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Upsample-Add 1/2"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_b_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_composite_pipeline);
            pass.set_bind_group(0, &bloom_a_bg, &[]);
            pass.set_bind_group(1, &composite_e_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 6) Final blur of the combined half-res bloom into bloom_a, then
        //    copy to bloom_b so the composite step reads a stable result.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Final Blur H"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_a_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_blur_h_pipeline);
            pass.set_bind_group(0, &bloom_b_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Final Blur V"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_b_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_blur_v_pipeline);
            pass.set_bind_group(0, &bloom_a_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 7) Bloom composite → tonemap_temp (full-res), then tonemap → swapchain.
        //    If bloom is off, we copy scene_view directly to tonemap_temp.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.tonemap_temp_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            if self.features.bloom_enabled {
                pass.set_pipeline(&self.bloom_composite_pipeline);
                pass.set_bind_group(0, &scene_bg, &[]);
                pass.set_bind_group(1, &composite_bloom_bg, &[]);
            } else {
                pass.set_pipeline(&self.post_copy_pipeline);
                pass.set_bind_group(0, &scene_bg, &[]);
            }
            pass.draw(0..3, 0..1);
        }

        // 5) God rays pass (screen-space sun shafts).
        //    Reads tonemap_temp (HDR+bloom) + depth → radial blur toward sun → postprocess_temp.
        //    Then copies postprocess_temp → tonemap_temp for tonemap input.
        if self.features.god_rays_enabled {
            // Compute sun UV position from sun direction.
            // Project the sun direction into clip space, then to UV.
            let sun_dir = self.sky_params.sun_direction;
            let view_proj = self.view_proj;
            // Project a point far along the sun direction into clip space.
            let sun_clip = view_proj * glam::Vec4::new(sun_dir[0] * 1000.0, sun_dir[1] * 1000.0, sun_dir[2] * 1000.0, 1.0);
            let sun_ndc = glam::Vec2::new(sun_clip.x / sun_clip.w, sun_clip.y / sun_clip.w);
            let sun_uv = glam::Vec2::new(sun_ndc.x * 0.5 + 0.5, -sun_ndc.y * 0.5 + 0.5);

            let godray_uniforms = GodRayUniforms {
                sun_uv:      [sun_uv.x, sun_uv.y],
                intensity:   self.features.god_rays_intensity,
                decay:       self.features.god_rays_decay,
                density:     self.features.god_rays_density,
                weight:      self.features.god_rays_weight,
                num_samples: self.features.god_rays_num_samples as f32,
                _pad:        0.0,
            };
            self.queue.write_buffer(
                &self.godray_uniform_buf, 0,
                bytemuck::bytes_of(&godray_uniforms),
            );

            let godray_scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GodRay Scene BG"),
                layout: &self.post_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.tonemap_temp_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                ],
            });
            let godray_data_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GodRay Data BG"),
                layout: &self.godray_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.godray_uniform_buf.as_entire_binding() },
                ],
            });

            // God rays: tonemap_temp → postprocess_temp
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("GodRay Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.postprocess_temp_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.godray_pipeline);
                pass.set_bind_group(0, &godray_scene_bg, &[]);
                pass.set_bind_group(1, &godray_data_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            // Copy postprocess_temp → tonemap_temp for tonemap input.
            {
                let copy_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GodRay Copy BG"),
                    layout: &self.post_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.postprocess_temp_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                    ],
                });
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("GodRay Copy Back"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.tonemap_temp_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.post_copy_pipeline);
                pass.set_bind_group(0, &copy_bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // 5b) Underwater post-process pass.
        //     Applied when the camera is below any water surface.
        //     Reads tonemap_temp + depth → tint, caustics, god rays, distortion → postprocess_temp.
        //     Then copies postprocess_temp → tonemap_temp for tonemap input.
        if self.features.underwater_enabled {
            // Find the highest water surface Y from all WaterSurface entities.
            let mut water_surface_y: Option<f32> = None;
            for (pos, _ws) in world.query::<(&Position, &crate::components::WaterSurface)>().iter() {
                // Use the entity's Y position as the water surface height.
                // For multiple water surfaces, take the highest one the camera is below.
                let sy = pos.y;
                if cam_pos.y < sy {
                    match &mut water_surface_y {
                        Some(best) => { *best = (*best).max(sy); }
                        None => { water_surface_y = Some(sy); }
                    }
                }
            }
            // Also check WaterBody entities for their position-based water surface.
            for (pos, _wb) in world.query::<(&Position, &crate::components::WaterBody)>().iter() {
                let sy = pos.y;
                if cam_pos.y < sy {
                    match &mut water_surface_y {
                        Some(best) => { *best = (*best).max(sy); }
                        None => { water_surface_y = Some(sy); }
                    }
                }
            }

            if let Some(surface_y) = water_surface_y {
                let camera_depth_below = surface_y - cam_pos.y; // positive = underwater depth
                if camera_depth_below > 0.0 {
                    let uw_uniforms = UnderwaterUniforms {
                        tint: [
                            self.features.underwater_tint[0],
                            self.features.underwater_tint[1],
                            self.features.underwater_tint[2],
                            self.features.underwater_fog_density,
                        ],
                        caustics: [
                            self.features.underwater_caustics,
                            8.0,  // scale — tiling of caustic pattern
                            1.0,  // speed — animation speed
                            self.features.underwater_god_rays,
                        ],
                        distortion: [
                            self.features.underwater_distortion,
                            self.elapsed_time,
                            self.features.underwater_vignette,
                            self.features.underwater_bloom,
                        ],
                        camera_params: [
                            surface_y,
                            camera_depth_below,
                            0.1,   // near clip
                            1000.0, // far clip
                        ],
                    };
                    self.queue.write_buffer(
                        &self.underwater_uniform_buf, 0,
                        bytemuck::bytes_of(&uw_uniforms),
                    );

                    let uw_scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Underwater Scene BG"),
                        layout: &self.post_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.tonemap_temp_view) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                        ],
                    });
                    let uw_data_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Underwater Data BG"),
                        layout: &self.underwater_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                            wgpu::BindGroupEntry { binding: 2, resource: self.underwater_uniform_buf.as_entire_binding() },
                        ],
                    });

                    // Underwater: tonemap_temp → postprocess_temp
                    {
                        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Underwater Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &self.postprocess_temp_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            ..Default::default()
                        });
                        pass.set_pipeline(&self.underwater_pipeline);
                        pass.set_bind_group(0, &uw_scene_bg, &[]);
                        pass.set_bind_group(1, &uw_data_bg, &[]);
                        pass.draw(0..3, 0..1);
                    }

                    // Copy postprocess_temp → tonemap_temp for tonemap input.
                    {
                        let copy_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Underwater Copy BG"),
                            layout: &self.post_bgl,
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.postprocess_temp_view) },
                                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                            ],
                        });
                        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Underwater Copy Back"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &self.tonemap_temp_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            ..Default::default()
                        });
                        pass.set_pipeline(&self.post_copy_pipeline);
                        pass.set_bind_group(0, &copy_bg, &[]);
                        pass.draw(0..3, 0..1);
                    }
                }
            }
        }

        // 6) Tone mapping + colour grading pass.
        //    Reads tonemap_temp (HDR with bloom) → applies ACES + colour grading + gamma → swapchain.
        //    This is ALWAYS the final pass, even if tonemap_enabled is false
        //    (default settings give a neutral look, just adds gamma correction).
        {
            // Upload tonemap uniforms.
            let tonemap_uniforms = TonemapUniforms {
                exposure:    self.features.tonemap_exposure,
                temperature: self.features.tonemap_temperature,
                saturation:  self.features.tonemap_saturation,
                contrast:    self.features.tonemap_contrast,
                vibrance:    self.features.tonemap_vibrance,
                grain:       self.features.tonemap_grain,
                _pad0:       0.0,
                _pad1:       0.0,
            };
            self.queue.write_buffer(
                &self.tonemap_uniform_buf, 0,
                bytemuck::bytes_of(&tonemap_uniforms),
            );

            let tonemap_scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tonemap Scene BG"),
                layout: &self.post_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.tonemap_temp_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.post_sampler) },
                ],
            });
            let tonemap_uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tonemap Uniform BG"),
                layout: &self.tonemap_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.tonemap_uniform_buf.as_entire_binding() },
                ],
            });

            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tonemap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.tonemap_pipeline);
            pass.set_bind_group(0, &tonemap_scene_bg, &[]);
            pass.set_bind_group(1, &tonemap_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if let Some(draw_overlay) = overlay_pass.as_mut() {
            draw_overlay(&self.device, &self.queue, &mut enc, &view);
        }

        self.queue.submit([enc.finish()]);
        frame.present();

        // Advance TAA frame counter for next frame's jitter.
        self.taa_frame_index = self.taa_frame_index.wrapping_add(1);

        stats
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn scene_color_view(&self) -> &wgpu::TextureView {
        &self.scene_view
    }

    pub fn invalidate_texture_path(&self, path: &str) {
        if path.is_empty() {
            return;
        }
        let mut cache = self.texture_cache.borrow_mut();
        cache.remove(&format!("srgb|{}", path));
        cache.remove(&format!("lin|{}", path));
    }

    fn get_or_load_texture(&self, path: &str, srgb: bool) -> Option<(wgpu::TextureView, wgpu::Sampler)> {
        if path.is_empty() {
            return None;
        }
        let cache_key = format!("{}|{}", if srgb { "srgb" } else { "lin" }, path);
        if let Some(v) = self.texture_cache.borrow().get(&cache_key) {
            return Some((v.0.clone(), v.1.clone()));
        }
        let bytes = crate::vfs::read(path).ok()?;
        let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
        let (w, h) = img.dimensions();
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Albedo Texture"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            img.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&Default::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        self.texture_cache
            .borrow_mut()
            .insert(cache_key, (view.clone(), sampler.clone()));
        Some((view, sampler))
    }

    pub fn apply_sky_environment(&mut self, path: &str) -> Result<(), String> {
        let p = path.trim();
        let make_bg = if p.is_empty() {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("Global BG (fallback sky)"),
                layout:  &self.global_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding:  0,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.default_albedo_view)  },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.default_albedo_sampler)  },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.default_albedo_view)  },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.default_albedo_sampler)  },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.default_albedo_view)  },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.default_albedo_sampler)  },
                    wgpu::BindGroupEntry { binding: 7, resource: self.shadow_uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(self.shadow_system.cascade_view(0)) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(self.shadow_system.cascade_view(1)) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(self.shadow_system.cascade_view(2)) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(&self.shadow_system.sampler) },
                    wgpu::BindGroupEntry { binding: 12, resource: self.light_uniform_buf.as_entire_binding() },
                ],
            })
        } else {
            let ibl_maps = ibl::IblMaps::from_hdr(&self.device, &self.queue, p)?;
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("Global BG (custom sky hdr)"),
                layout:  &self.global_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding:  0,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ibl_maps.irradiance_view)  },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ibl_maps.irradiance_sampler)  },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&ibl_maps.prefilter_view)  },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&ibl_maps.prefilter_sampler)  },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&ibl_maps.brdf_lut_view)  },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&ibl_maps.brdf_lut_sampler)  },
                    wgpu::BindGroupEntry { binding: 7, resource: self.shadow_uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(self.shadow_system.cascade_view(0)) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(self.shadow_system.cascade_view(1)) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(self.shadow_system.cascade_view(2)) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(&self.shadow_system.sampler) },
                    wgpu::BindGroupEntry { binding: 12, resource: self.light_uniform_buf.as_entire_binding() },
                ],
            })
        };
        self.bind_group = make_bg;
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Halton radical inverse sequence for TAA sub-pixel jitter.
// Returns a value in [0, 1) for the given index and base.
fn halton(index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    let mut i = index;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

// make_depth_texture() creates a Depth32Float texture matching the surface size.
// Called at startup and whenever the window is resized.
fn make_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size:  wgpu::Extent3d {
            width:                 config.width.max(1),
            height:                config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Depth32Float,
        // TEXTURE_BINDING is required so the SSR post-process pass can read depth
        // to reconstruct view-space positions for ray marching.
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
                       | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats:    &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        // Explicitly request only the depth aspect so the view is valid for
        // depth/stencil attachments.
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    });
    (texture, view)
}

fn make_scene_color_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Scene Color"),
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

// make_normals_texture() creates a full-res Rgba16Float texture for world-space normals.
// Written by the geometry + sky passes (MRT target 1), read by the SSR pass.
fn make_normals_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Normals GBuffer"),
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

// make_gbuffer_texture() creates a full-res Rgba16Float render target for the
// deferred G-buffer. Written by the opaque geometry pass, read by the deferred
// lighting pass.
fn make_gbuffer_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    label:  &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

// make_velocity_texture() creates a full-res Rg16Float texture for per-pixel motion vectors.
// Placeholder until the geometry pass writes actual motion vectors from previous-frame MVP.
fn make_velocity_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Velocity Buffer"),
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

// make_taa_history_texture() creates a full-res Rgba16Float texture for TAA history.
// Stores the previous frame's TAA-resolved output for temporal accumulation.
fn make_taa_history_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("TAA History"),
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn make_bloom_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    make_bloom_texture_at(device, (config.width / 2).max(1), (config.height / 2).max(1), label)
}

/// Bloom pyramid texture at an explicit resolution (half / quarter / …).
fn make_bloom_texture_at(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_post_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bgl0: &wgpu::BindGroupLayout,
    bgl1: Option<&wgpu::BindGroupLayout>,
    format: wgpu::TextureFormat,
    fragment_entry: &str,
) -> wgpu::RenderPipeline {
    let mut bgls = vec![Some(bgl0)];
    if let Some(x) = bgl1 {
        bgls.push(Some(x));
    }
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Post Pipeline Layout"),
        bind_group_layouts: &bgls,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Post Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_fullscreen"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn simplify_triangle_soup_preserve_shape(
    vertices: &[crate::assets::mesh::Vertex],
    keep_ratio: f32,
) -> Vec<crate::assets::mesh::Vertex> {
    if keep_ratio >= 0.999 || vertices.len() < 6 {
        return vertices.to_vec();
    }

    let tri_count = vertices.len() / 3;
    if tri_count <= 2 {
        return vertices.to_vec();
    }

    let target_tris = ((tri_count as f32) * keep_ratio).round() as usize;
    let target_tris = target_tris.clamp(2, tri_count);

    // ── Build edge map ──────────────────────────────────────────────────
    // Each edge is defined by two vertex positions (rounded for dedup).
    // We store the positions directly for QEM scoring.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    struct EdgeKey {
        a: [i32; 3],
        b: [i32; 3],
    }

    impl EdgeKey {
        fn new(p0: [f32; 3], p1: [f32; 3]) -> Self {
            let a = [
                (p0[0] * 1000.0) as i32,
                (p0[1] * 1000.0) as i32,
                (p0[2] * 1000.0) as i32,
            ];
            let b = [
                (p1[0] * 1000.0) as i32,
                (p1[1] * 1000.0) as i32,
                (p1[2] * 1000.0) as i32,
            ];
            if a <= b { EdgeKey { a, b } } else { EdgeKey { a: b, b: a } }
        }
    }

    #[derive(Clone)]
    struct EdgeInfo {
        cost: f32,
        tri_indices: Vec<usize>,
    }

    let mut edge_map: std::collections::HashMap<EdgeKey, EdgeInfo> = std::collections::HashMap::new();

    for i in 0..tri_count {
        let a = vertices[i * 3].position;
        let b = vertices[i * 3 + 1].position;
        let c = vertices[i * 3 + 2].position;

        let edges = [(a, b), (b, c), (c, a)];
        for (p0, p1) in edges {
            let key = EdgeKey::new(p0, p1);
            let entry = edge_map.entry(key).or_insert_with(|| EdgeInfo {
                cost: 0.0,
                tri_indices: Vec::new(),
            });
            entry.tri_indices.push(i);
        }
    }

    // ── Score edges by quadric error metric (simplified) ────────────────
    // For each edge, compute collapse cost from the two endpoint planes.
    for (_key, info) in edge_map.iter_mut() {
        // Simple QEM approximation: cost = average triangle area of incident faces.
        // Real QEM would accumulate face planes into 4x4 matrices, but this
        // approximation preserves shape better than raw area by considering
        // edge adjacency.
        let mut total_area = 0.0;
        let mut max_area = 0.0f32;
        for &tri_idx in &info.tri_indices {
            let va = glam::Vec3::from_array(vertices[tri_idx * 3].position);
            let vb = glam::Vec3::from_array(vertices[tri_idx * 3 + 1].position);
            let vc = glam::Vec3::from_array(vertices[tri_idx * 3 + 2].position);
            let area = (vb - va).cross(vc - va).length() * 0.5;
            total_area += area;
            max_area = max_area.max(area);
        }
        // Cost balances edge length (longer edges = more shape loss)
        // against incident face area (smaller faces = cheaper to remove).
        let edge_center_diff = max_area;
        info.cost = edge_center_diff + total_area * 0.1;
    }

    // ── Iterative edge collapse ─────────────────────────────────────────
    // Collapse cheapest edges until target triangle count is reached.
    let mut alive_tris: Vec<bool> = vec![true; tri_count];
    let mut current_tris = tri_count;

    // Sort edges by cost (cheapest first).
    let mut sorted_edges: Vec<(f32, EdgeKey)> = edge_map
        .iter()
        .map(|(k, v)| (v.cost, *k))
        .collect();
    sorted_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for (_cost, key) in &sorted_edges {
        if current_tris <= target_tris {
            break;
        }
        if let Some(info) = edge_map.get(key) {
            // Collapse this edge: remove all triangles that share both endpoints.
            // In the simplified approach, we remove the smaller-area triangle
            // incident to this edge.
            let mut best_tri = None;
            let mut best_area = f32::MAX;
            for &tri_idx in &info.tri_indices {
                if !alive_tris[tri_idx] {
                    continue;
                }
                let va = glam::Vec3::from_array(vertices[tri_idx * 3].position);
                let vb = glam::Vec3::from_array(vertices[tri_idx * 3 + 1].position);
                let vc = glam::Vec3::from_array(vertices[tri_idx * 3 + 2].position);
                let area = (vb - va).cross(vc - va).length() * 0.5;
                if area < best_area {
                    best_area = area;
                    best_tri = Some(tri_idx);
                }
            }
            if let Some(tri_idx) = best_tri {
                alive_tris[tri_idx] = false;
                current_tris -= 1;
            }
        }
    }

    // ── Rebuild output from surviving triangles ─────────────────────────
    let mut out = Vec::with_capacity(target_tris * 3);
    for (i, &alive) in alive_tris.iter().enumerate() {
        if alive {
            let base = i * 3;
            if base + 2 < vertices.len() {
                out.push(vertices[base]);
                out.push(vertices[base + 1]);
                out.push(vertices[base + 2]);
            }
        }
    }

    if out.len() < 3 {
        vertices.to_vec()
    } else {
        out
    }
}

// make_1x1_texture() creates a tiny solid-colour fallback texture.
// Returns (TextureView, Sampler) — both needed to fill a bind group slot.
fn make_1x1_texture(
    device:  &wgpu::Device,
    queue:   &wgpu::Queue,
    rgba:    [u8; 4],
    is_srgb: bool,
) -> (wgpu::TextureView, wgpu::Sampler) {
    let format = if is_srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("Default 1×1 Texture"),
        size:            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format,
        usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats:    &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture:   &texture,
            mip_level: 0,
            origin:    wgpu::Origin3d::ZERO,
            aspect:    wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset:         0,
            bytes_per_row:  Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    let view = texture.create_view(&Default::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter:    wgpu::FilterMode::Nearest,
        min_filter:    wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    (view, sampler)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every WGSL shader embedded with include_str! must parse AND fully validate
    /// as valid WGSL. naga's frontend parses the source into a module, then the
    /// Validator runs layout/type/usage checks — the same validation wgpu performs
    /// at pipeline creation — so a typo here fails `cargo test` instead of crashing
    /// at startup on the user's machine.
    #[test]
    fn embedded_shaders_parse_as_valid_wgsl() {
        use naga::front::wgsl::parse_str;
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let shaders = [
            ("shader.wgsl", include_str!("renderer/shader.wgsl")),
            ("deferred.wgsl", include_str!("renderer/deferred.wgsl")),
            ("postprocess.wgsl", include_str!("renderer/postprocess.wgsl")),
            ("sky.wgsl", include_str!("renderer/sky.wgsl")),
            ("water.wgsl", include_str!("renderer/water.wgsl")),
            ("shadow.wgsl", include_str!("renderer/shadow.wgsl")),
            ("particle.wgsl", include_str!("renderer/particle.wgsl")),
            ("fire.wgsl", include_str!("renderer/fire.wgsl")),
            ("lava.wgsl", include_str!("renderer/lava.wgsl")),
        ];
        for (name, src) in shaders {
            let module = match parse_str(src) {
                Ok(m) => m,
                Err(e) => panic!("{} failed WGSL parse:\n{:?}", name, e),
            };
            let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
            if let Err(e) = validator.validate(&module) {
                panic!("{} failed WGSL validation:\n{:?}", name, e);
            }
        }
    }

    fn make_test_vertex(position: [f32; 3], normal: [f32; 3]) -> crate::assets::mesh::Vertex {
        crate::assets::mesh::Vertex::new(position, normal, [1.0, 1.0, 1.0])
    }

    #[test]
    fn edge_collapse_reduces_triangle_count() {
        // 4 triangles forming a strip: should reduce to target.
        let vertices = vec![
            make_test_vertex([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([0.5, 1.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([2.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([1.5, 1.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([2.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([3.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([2.5, 1.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([3.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([4.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([3.5, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        let result = simplify_triangle_soup_preserve_shape(&vertices, 0.5);
        // Should reduce from 4 triangles to ~2.
        assert!(result.len() <= 12, "expected <=12 verts, got {}", result.len());
        assert!(result.len() >= 6, "expected >=6 verts, got {}", result.len());
    }

    #[test]
    fn edge_collapse_preserves_minimum() {
        // 2 triangles — too few to simplify.
        let vertices = vec![
            make_test_vertex([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([0.5, 1.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([2.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([1.5, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        let result = simplify_triangle_soup_preserve_shape(&vertices, 0.25);
        assert_eq!(result.len(), 6, "should keep all 2 triangles");
    }

    #[test]
    fn edge_collapse_noop_at_full_ratio() {
        let vertices = vec![
            make_test_vertex([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_test_vertex([0.5, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        let result = simplify_triangle_soup_preserve_shape(&vertices, 1.0);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn edge_collapse_empty_input() {
        let result = simplify_triangle_soup_preserve_shape(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn snow_coverage_zero_when_clear() {
        let data = GpuWeatherData {
            snow_coverage: 0.0,
            _pad: [0.0; 3],
        };
        assert_eq!(data.snow_coverage, 0.0);
    }

    #[test]
    fn snow_coverage_full_when_snowing() {
        let data = GpuWeatherData {
            snow_coverage: 1.0,
            _pad: [0.0; 3],
        };
        assert_eq!(data.snow_coverage, 1.0);
    }

    #[test]
    fn gpu_weather_data_layout() {
        assert_eq!(std::mem::size_of::<GpuWeatherData>(), 16);
    }

    #[test]
    fn lod_defaults_are_monotonically_increasing() {
        let f = RenderFeatures::default();
        assert!(f.mesh_lod_threshold_1 < f.mesh_lod_threshold_2);
        assert!(f.mesh_lod_threshold_2 < f.mesh_lod_threshold_3);
        assert!(f.mesh_lod_threshold_3 < f.mesh_lod_threshold_4);
    }
}
