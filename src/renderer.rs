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

pub mod ibl;
pub mod light_probes;
pub mod pipeline;
pub mod shadow;

use std::sync::Arc;
use std::collections::HashMap;

use winit::window::Window;

use crate::assets::{AssetStore, Mesh};
use crate::camera::Camera;
use crate::components::{MaterialTexture, PointLight, Position, Renderable, Rotation};
use crate::jobs::JobSystem;
use hecs::World;
use rayon::prelude::*;

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
// Total size: 64 + 16 + 16 + 16 = 112 bytes — must be a multiple of 16.
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
        }
    }

    pub fn balanced() -> Self {
        Self::default()
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

        // Detect low-end GPU automatically.
        let is_low_end = matches!(
            adapter_info.device_type,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
        );
        if is_low_end {
            println!("[Renderer] Integrated GPU detected — using low-end preset");
        }
        let features = if is_low_end {
            RenderFeatures::low_end()
        } else {
            RenderFeatures::default()
        };

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

        // ── Vertex buffer ─────────────────────────────────────────────────
        // Staging buffer overwritten once per mesh draw (see MAX_VERTICES_PER_DRAW).
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Vertices"),
            size:               (MAX_VERTICES_PER_DRAW as u64)
                * (std::mem::size_of::<crate::assets::mesh::Vertex>() as u64),
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Uniform buffer ─────────────────────────────────────────────────
        // std::mem::size_of::<GpuUniforms>() = 112 bytes.
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Camera Uniform"),
            size:               std::mem::size_of::<GpuUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        Self {
            _window:      window,
            surface,
            device,
            queue,
            config,
            pipeline:     render_pipeline,
            vertex_buffer,
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
    }

    // draw_world() — renders every entity with a Position + Renderable component.
    // Called once per frame from main.rs.
    pub fn draw_world(
        &self,
        world:  &World,
        meshes: &AssetStore<Mesh>,
        camera: &dyn Camera,
        jobs: &JobSystem,
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
        let cam_pos = camera.position();
        let uniforms = GpuUniforms {
            view_proj:   vp.to_cols_array_2d(),
            camera_pos:  cam_pos.to_array(),
            _pad0:       0.0,
            light_dir:   {
                let az = self.features.sun_azimuth_deg.to_radians();
                let el = self.features.sun_elevation_deg.to_radians();
                glam::Vec3::new(el.cos() * az.cos(), el.sin(), el.cos() * az.sin())
                    .normalize()
                    .to_array()
            },
            _pad1:       0.0,
            light_color: [self.features.sun_intensity, self.features.sun_intensity, self.features.sun_intensity],
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
        };
        self.queue.write_buffer(
            &self.uniform_buf, 0,
            bytemuck::bytes_of(&uniforms),
        );

        // Acquire the next swapchain frame.
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Outdated => return stats,
            wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("[Renderer] Failed to acquire frame texture");
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
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &self.scene_view,
                    resolve_target: None,
                    depth_slice:    None,
                    ops: wgpu::Operations {
                        // Dark near-black background.
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.08, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
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

            for cand in visible {
                let entity = cand.entity;
                let pos = cand.pos;
                let renderable = cand.renderable;
                let rotation = world
                    .get::<&Rotation>(entity)
                    .map(|r| *r)
                    .unwrap_or(Rotation {
                        pitch: 0.0,
                        yaw: 0.0,
                        roll: 0.0,
                    });
                if let Some(mesh) = meshes.get(&renderable.mesh) {
                    let (albedo_view, albedo_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(entity) {
                        self.get_or_load_texture(&tex.path, true).unwrap_or((
                            self.default_albedo_view.clone(),
                            self.default_albedo_sampler.clone(),
                        ))
                    } else {
                        (
                            self.default_albedo_view.clone(),
                            self.default_albedo_sampler.clone(),
                        )
                    };
                    let (normal_view, normal_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(entity) {
                        self.get_or_load_texture(&tex.normal_path, false).unwrap_or((
                            self.default_normal_view.clone(),
                            self.default_normal_sampler.clone(),
                        ))
                    } else {
                        (
                            self.default_normal_view.clone(),
                            self.default_normal_sampler.clone(),
                        )
                    };
                    let (mr_view, mr_sampler) = if let Ok(tex) = world.get::<&MaterialTexture>(entity) {
                        self.get_or_load_texture(&tex.metallic_roughness_path, false).unwrap_or((
                            self.default_mr_view.clone(),
                            self.default_mr_sampler.clone(),
                        ))
                    } else {
                        (
                            self.default_mr_view.clone(),
                            self.default_mr_sampler.clone(),
                        )
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
                        ],
                    });
                    pass.set_bind_group(1, &material_bg, &[]);

                    let dx = pos.x - cam_pos.x;
                    let dy = pos.y - cam_pos.y;
                    let dz = pos.z - cam_pos.z;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    let max_scale = renderable.scale[0]
                        .abs()
                        .max(renderable.scale[1].abs())
                        .max(renderable.scale[2].abs())
                        .max(0.25);
                    // Scale-aware LOD: larger objects hold detail longer than tiny props.
                    let lod_metric = distance / max_scale;
                    let lod_band = if lod_metric > 300.0 {
                        4u8
                    } else if lod_metric > 180.0 {
                        3u8
                    } else if lod_metric > 95.0 {
                        2u8
                    } else if lod_metric > 48.0 {
                        1u8
                    } else {
                        0u8
                    };
                    let lod_ratio = match lod_band {
                        0 => 1.0,
                        1 => 0.78,
                        2 => 0.56,
                        3 => 0.34,
                        _ => 0.20,
                    };

                    let lod_vertices: Vec<crate::assets::mesh::Vertex> = if lod_band == 0 {
                        mesh.vertices.clone()
                    } else {
                        let key = (renderable.mesh.id, lod_band);
                        if let Some(cached) = self.lod_cache.borrow().get(&key) {
                            cached.clone()
                        } else {
                            let simplified = simplify_triangle_soup_preserve_shape(&mesh.vertices, lod_ratio);
                            {
                                let mut cache = self.lod_cache.borrow_mut();
                                if cache.len() > 256 {
                                    cache.clear();
                                }
                                cache.insert(key, simplified.clone());
                            }
                            simplified
                        }
                    };

                    // Build per-entity vertices: apply scale and world position.
                    // The camera matrix in the shader handles world → clip conversion.
                    let colored_verts: Vec<crate::assets::mesh::Vertex> = lod_vertices
                        .iter()
                        .map(|v| {
                            let mut out = *v;
                            let local = glam::Vec3::new(
                                v.position[0] * renderable.scale[0],
                                v.position[1] * renderable.scale[1],
                                v.position[2] * renderable.scale[2],
                            );
                            let rot = glam::Mat3::from_euler(
                                glam::EulerRot::YXZ,
                                rotation.yaw,
                                rotation.pitch,
                                rotation.roll,
                            );
                            let rotated = rot * local;
                            out.position = [
                                rotated.x + pos.x,
                                rotated.y + pos.y,
                                rotated.z + pos.z,
                            ];
                            out.color     = renderable.color;
                            out.metallic  = renderable.metallic;
                            out.roughness = renderable.roughness;
                            out.ao        = renderable.ao;
                            out
                        })
                        .collect();
                    if colored_verts.len() < 3 {
                        continue;
                    }

                    let bytes = bytemuck::cast_slice(&colored_verts);
                    // Guard: don't write more than the buffer holds.
                    if bytes.len() <= (self.vertex_buffer.size() as usize) {
                        self.queue.write_buffer(&self.vertex_buffer, 0, bytes);
                        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                        pass.draw(0..colored_verts.len() as u32, 0..1);
                        stats.drawn += 1;
                    } else {
                        eprintln!(
                            "[Renderer] Mesh too large ({} verts); max {} per draw. \
                             Import a lower-poly mesh or raise MAX_VERTICES_PER_DRAW in renderer.rs.",
                            colored_verts.len(),
                            MAX_VERTICES_PER_DRAW
                        );
                    }
                }
            }
        } // pass is dropped here — commands are finalised

        let scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Scene BG"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.scene_view) },
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

        // 4) Final composite to swapchain.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Post Composite"),
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
            if self.features.bloom_enabled {
                pass.set_pipeline(&self.bloom_composite_pipeline);
                pass.set_bind_group(0, &scene_bg, &[]);
                pass.set_bind_group(1, &composite_bloom_bg, &[]);
                pass.draw(0..3, 0..1);
            } else {
                pass.set_pipeline(&self.post_copy_pipeline);
                pass.set_bind_group(0, &scene_bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        if let Some(draw_overlay) = overlay_pass.as_mut() {
            draw_overlay(&self.device, &self.queue, &mut enc, &view);
        }

        self.queue.submit([enc.finish()]);
        frame.present();
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
        let bytes = std::fs::read(path).ok()?;
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
                ],
            })
        };
        self.bind_group = make_bg;
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

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
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
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
