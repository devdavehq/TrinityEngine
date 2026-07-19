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
pub mod light_probes;
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
use crate::components::{MaterialTexture, PointLight, Position, Renderable, Rotation};
use crate::jobs::JobSystem;
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
    _pad2:         [f32; 2],
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
    bloom_blur_h_pipeline: wgpu::RenderPipeline,
    bloom_blur_v_pipeline: wgpu::RenderPipeline,
    bloom_composite_pipeline: wgpu::RenderPipeline,
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
    /// ── Multi-light uniform buffer (group 0, binding 12) ───────────────────
    light_uniform_buf: wgpu::Buffer,
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

        // Weather intensity drives water roughness.
        self.weather_intensity = weather.intensity;

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

        // ── Normals GBuffer texture (MRT target 1) ────────────────────────────
        // Rgba16Float gives enough precision for world-space normals without
        // quantisation artefacts.  Written by the geometry + sky passes,
        // read by the SSR pass to reconstruct reflection vectors.
        let (normals_texture, normals_view) = make_normals_texture(&device, &config);

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
                // Shadow fallbacks so shader bindings are valid even before full shadow system hookup.
                wgpu::BindGroupEntry { binding: 7, resource: shadow_fallback_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&shadow_fallback_view) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&shadow_fallback_view) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&shadow_fallback_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(&shadow_fallback_sampler) },
                // Multi-light uniform (binding 12) — populated each frame in draw_world().
                wgpu::BindGroupEntry { binding: 12, resource: light_uniform_buf.as_entire_binding() },
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

        Self {
            _window:      window.clone(),
            surface,
            device,
            queue,
            config,
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
            post_sampler,
            post_bgl,
            post2_bgl,
            post_copy_pipeline,
            bloom_extract_pipeline,
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
            features,
            adapter_info,
            sky_renderer,
            particle_renderer,
            water_renderer,
            lava_renderer,
            fire_renderer,
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
            taa_frame_index: 0,
            light_uniform_buf,
            default_material_extras_buf,
            sky_params: crate::environment::sky::SkyParams::default(),
            cloud_params: crate::environment::clouds::CloudParams::default(),
            storm_darken: 0.0,
            lightning_intensity: 0.0,
            weather_intensity: 0.0,
            elapsed_time: 0.0,
        }
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
        let (tt, ttv) = make_scene_color_texture(&self.device, &self.config);
        self.tonemap_temp = tt;
        self.tonemap_temp_view = ttv;
        let (nt, nv) = make_normals_texture(&self.device, &self.config);
        self.normals_texture = nt;
        self.normals_view = nv;
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
    }

    // draw_world() — renders every entity with a Position + Renderable component.
    // Called once per frame from main.rs.
    pub fn draw_world(
        &mut self,
        world:  &World,
        meshes: &AssetStore<Mesh>,
        camera: &dyn Camera,
        jobs: &JobSystem,
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

        // ── Build multi-light array ────────────────────────────────────────────
        // Populates the LightUniforms buffer with the directional sun light
        // (index 0) plus all PointLight entities from the ECS world (up to 16).
        let mut gpu_lights = [GpuLightData::default(); 16];
        let mut light_count: u32 = 0;

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
            _pad2:         [0.0; 2],
        };
        light_count = 1;

        // Add all PointLight entities (point + spot + additional directional).
        for (pos, pl) in world.query::<(&Position, &PointLight)>().iter() {
            if light_count >= 16 { break; }
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
                _pad2:         [0.0; 2],
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

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        // 1.0 = max depth (everything starts as "lit / in front").
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // ── Sky rendering (before geometry) ──────────────────────────────
            // Draw a fullscreen triangle at depth = 1.0 (far plane).
            // Geometry drawn afterwards overwrites sky pixels via depth test.
            self.sky_renderer.render(&mut pass);

            pass.set_pipeline(&self.pipeline);
            // Bind group 0 = global (camera, IBL). Set once per frame.
            pass.set_bind_group(0, &self.bind_group, &[]);

            let candidates: Vec<DrawCandidate> = world
                .query::<(hecs::Entity, &Position, &Renderable)>()
                .iter()
                .map(|(entity, pos, renderable)| DrawCandidate {
                    entity,
                    pos: *pos,
                    renderable: *renderable,
                })
                .collect();
            stats.total = candidates.len();

            let visible: Vec<DrawCandidate> = if self.features.culling_enabled {
                let cam_pos = camera.position();
                let vp = camera.view_projection_matrix();
                let cull_dist2 = self.features.culling_distance * self.features.culling_distance;
                let frustum_enabled = self.features.frustum_culling_enabled;
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
                            dist_ok && frustum_ok
                        })
                        .collect()
                })
            } else {
                candidates
            };
            stats.visible = visible.len();

            // ── GPU Instanced Drawing ──────────────────────────────────────
            // Group visible entities by mesh_id. Each group is drawn with ONE
            // draw call using instance_count = number of entities sharing that mesh.
            // This replaces the old per-entity CPU transform + draw approach.
            use std::collections::HashMap;
            let cam_pos = camera.position();
            let mut mesh_groups: HashMap<u32, Vec<(&DrawCandidate, glam::Mat4)>> = HashMap::new();

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

                mesh_groups.entry(renderable.mesh.id).or_default().push((cand, model));
            }

            // Sort groups by mesh_id for consistent draw order.
            let mut sorted_groups: Vec<_> = mesh_groups.into_iter().collect();
            sorted_groups.sort_by_key(|(id, _)| *id);

            for (mesh_id, group) in sorted_groups {
                // Use first entity's renderable to look up mesh data.
                let first_renderable = group[0].0.renderable;
                let Some(mesh) = meshes.get(&first_renderable.mesh) else { continue };

                // Pick the highest LOD level needed by any entity in this group.
                // (We use the same LOD vertices for all instances of the same mesh.)
                let max_lod_distance = group.iter().map(|(cand, _)| {
                    let dx = cand.pos.x - cam_pos.x;
                    let dy = cand.pos.y - cam_pos.y;
                    let dz = cand.pos.z - cam_pos.z;
                    let max_scale = cand.renderable.scale[0].abs()
                        .max(cand.renderable.scale[1].abs())
                        .max(cand.renderable.scale[2].abs())
                        .max(0.25);
                    ((dx * dx + dy * dy + dz * dz).sqrt()) / max_scale
                }).fold(0.0f32, f32::max);

                let lod_band = if max_lod_distance > 300.0 { 4u8 }
                    else if max_lod_distance > 180.0 { 3u8 }
                    else if max_lod_distance > 95.0 { 2u8 }
                    else if max_lod_distance > 48.0 { 1u8 }
                    else { 0u8 };
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

                // Upload mesh vertices ONCE for this group.
                let vertex_bytes = bytemuck::cast_slice(&lod_vertices);
                if vertex_bytes.len() > self.vertex_buffer.size() as usize {
                    tracing::error!(
                        "[Renderer] Mesh too large ({} verts); skipping group mesh_id={}.",
                        lod_vertices.len(), mesh_id
                    );
                    continue;
                }
                self.queue.write_buffer(&self.vertex_buffer, 0, vertex_bytes);

                // Build instance data for all entities in this group.
                let instances: Vec<crate::render::instancing::InstanceData> = group
                    .iter()
                    .map(|(cand, model)| {
                        crate::render::instancing::InstanceData {
                            model: model.to_cols_array_2d(),
                            color_metallic: [
                                cand.renderable.color[0],
                                cand.renderable.color[1],
                                cand.renderable.color[2],
                                cand.renderable.metallic,
                            ],
                            roughness_ao_pad: [
                                cand.renderable.roughness,
                                cand.renderable.ao,
                                0.0, 0.0,
                            ],
                        }
                    })
                    .collect();

                let instance_bytes: &[u8] = bytemuck::cast_slice(&instances);
                let instance_buf_size = self.instance_buffer.size() as usize;
                if instance_bytes.len() > instance_buf_size {
                    tracing::error!(
                        "[Renderer] Instance buffer overflow ({} bytes needed, {} available). \
                         Group mesh_id={} with {} instances skipped.",
                        instance_bytes.len(), instance_buf_size, mesh_id, instances.len()
                    );
                    continue;
                }
                self.queue.write_buffer(&self.instance_buffer, 0, instance_bytes);

                // Bind material for the first entity (all instances in a batch share material).
                let first_entity = group[0].0.entity;
                let (albedo_view, albedo_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(first_entity) {
                    self.get_or_load_texture(&tex.path, true).unwrap_or((
                        self.default_albedo_view.clone(),
                        self.default_albedo_sampler.clone(),
                    ))
                } else {
                    (self.default_albedo_view.clone(), self.default_albedo_sampler.clone())
                };
                let (normal_view, normal_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(first_entity) {
                    self.get_or_load_texture(&tex.normal_path, false).unwrap_or((
                        self.default_normal_view.clone(),
                        self.default_normal_sampler.clone(),
                    ))
                } else {
                    (self.default_normal_view.clone(), self.default_normal_sampler.clone())
                };
                let (mr_view, mr_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(first_entity) {
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
                        // Material extras (binding 6) — default all-zeros (no SSS/clearcoat).
                        wgpu::BindGroupEntry { binding: 6, resource: self.default_material_extras_buf.as_entire_binding() },
                    ],
                });
                pass.set_bind_group(1, &material_bg, &[]);

                // Bind both vertex and instance buffers, then draw instanced.
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                pass.draw(0..lod_vertices.len() as u32, 0..instances.len() as u32);
                stats.drawn += 1;
            }
        } // main pass is dropped here — commands are finalised

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

        // 1) Extract bright parts.
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

        // 2) Blur horizontally into bloom_b.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Blur H"),
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
            pass.set_pipeline(&self.bloom_blur_h_pipeline);
            pass.set_bind_group(0, &bloom_a_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3) Blur vertically back into bloom_a, then copy to bloom_b final.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Blur V"),
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
            pass.set_pipeline(&self.bloom_blur_v_pipeline);
            pass.set_bind_group(0, &bloom_b_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 4) Bloom composite → tonemap_temp (full-res), then tonemap → swapchain.
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
                    wgpu::BindGroupEntry { binding: 7, resource: self._shadow_fallback_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self._shadow_fallback_view) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self._shadow_fallback_view) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self._shadow_fallback_view) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(&self._shadow_fallback_sampler) },
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
                    wgpu::BindGroupEntry { binding: 7, resource: self._shadow_fallback_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self._shadow_fallback_view) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self._shadow_fallback_view) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self._shadow_fallback_view) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(&self._shadow_fallback_sampler) },
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
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: (config.width / 2).max(1),
            height: (config.height / 2).max(1),
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

    #[derive(Clone, Copy)]
    struct TriInfo {
        tri_idx: usize,
        score: f32,
    }

    let mut infos: Vec<TriInfo> = Vec::with_capacity(tri_count);
    for i in 0..tri_count {
        let a = vertices[i * 3].position;
        let b = vertices[i * 3 + 1].position;
        let c = vertices[i * 3 + 2].position;
        let va = glam::Vec3::from_array(a);
        let vb = glam::Vec3::from_array(b);
        let vc = glam::Vec3::from_array(c);
        let area2 = (vb - va).cross(vc - va).length();
        // Prefer larger triangles and keep some thin/small ones via sqrt weighting.
        let score = area2 + area2.sqrt() * 0.35;
        infos.push(TriInfo { tri_idx: i, score });
    }

    infos.sort_by(|x, y| y.score.partial_cmp(&x.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = ((tri_count as f32) * keep_ratio).round() as usize;
    keep = keep.clamp(2, tri_count);

    // Take top scored triangles.
    let mut selected: Vec<usize> = infos.iter().take(keep).map(|t| t.tri_idx).collect();

    // Add sparse coverage pass so far zones keep silhouette hints.
    let stride = (tri_count / keep.max(1)).max(1);
    let mut i = 0usize;
    while selected.len() < keep + (keep / 6) && i < tri_count {
        selected.push(i);
        i += stride;
    }
    selected.sort_unstable();
    selected.dedup();

    let mut out = Vec::with_capacity(selected.len() * 3);
    for tri in selected {
        let base = tri * 3;
        if base + 2 < vertices.len() {
            out.push(vertices[base]);
            out.push(vertices[base + 1]);
            out.push(vertices[base + 2]);
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
