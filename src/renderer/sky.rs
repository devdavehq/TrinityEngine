// src/renderer/sky.rs
// Procedural sky rendering pipeline.
// Renders a fullscreen triangle with a procedural sky shader before geometry,
// so that the sky is visible wherever no geometry is drawn.
//
// ── Architecture ─────────────────────────────────────────────────────────────
// The sky is rendered FIRST in the main render pass:
//   1. Sky draws a fullscreen triangle at depth = 1.0 (far plane).
//   2. Geometry is drawn with depth test LessEqual — closer geometry
//      naturally overwrites sky pixels.
//
// This means the sky is only visible where there's no geometry, which is
// exactly what we want. No stencil tricks needed.
//
// ── Data flow ────────────────────────────────────────────────────────────────
// Each frame:
//   1. main.rs calls environment systems (TimeOfDay, SkyParams, WeatherState, CloudParams).
//   2. main.rs calls sky.update_uniforms() with all environment data + camera matrices.
//   3. draw_world() calls sky.render() BEFORE the geometry loop.
//   4. Sky shader computes procedural sky from the uniform buffer.

use std::sync::Arc;
use winit::window::Window;

use crate::environment::sky::{SkyParams, SkyUniformData};
use crate::environment::clouds::{CloudParams, CloudUniformData};

/// GPU-side uniform data for the sky shader.
/// Total: 384 bytes (24 × vec4).
///
/// Layout must match SkyUniforms in sky.wgsl exactly.
/// repr(C) ensures field ordering matches the WGSL struct.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniforms {
    /// Sky gradient colors and sun/moon/atmosphere data.
    pub sky: SkyUniformData,          // 144 bytes
    /// Cloud layer parameters.
    pub cloud: CloudUniformData,       //  64 bytes
    /// Inverse view-projection matrix for ray reconstruction.
    pub inv_view_proj: [[f32; 4]; 4],  //  64 bytes
    /// xyz = camera world position, w = total elapsed time.
    pub camera_pos_time: [f32; 4],     //  16 bytes
    /// xy = screen dimensions, z = fog density, w = unused.
    pub screen_fog: [f32; 4],          //  16 bytes
    /// rgb = fog color (from TimeOfDay), w = unused.
    pub fog_color: [f32; 4],           //  16 bytes
    /// Previous frame's view-projection matrix for temporal reprojection.
    pub prev_view_proj: [[f32; 4]; 4], //  64 bytes
}

impl Default for SkyUniforms {
    fn default() -> Self {
        Self {
            sky: SkyUniformData {
                zenith_color:  [0.1, 0.2, 0.5, 0.0],
                horizon_color: [0.6, 0.7, 0.9, 0.0],
                ground_color:  [0.1, 0.15, 0.1, 0.0],
                sun_direction: [0.5, 0.7, 0.5, 0.0],
                sun_color:     [1.0, 0.95, 0.8, 1.0],
                moon_direction: [0.0, 0.0, 0.0, 0.0],
                atmosphere:    [0.1, 0.001, 0.01, 0.76],
                stars_params:  [0.0, 0.5, 0.5, 4.0],
                sky_visibility: [1.0, 1.0, 0.0, 0.0],
            },
            cloud: CloudUniformData {
                params: [0.0, 2.0, 1.0, 1.0],
                noise: [0.005, 0.4, 0.1, 0.0],
                scroll: [0.0, 0.0, 0.01, 0.0],
                cloud_type: [0.0, 0.0, 0.0, 0.0],
            },
            inv_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos_time: [0.0, 0.0, 0.0, 0.0],
            screen_fog: [1920.0, 1080.0, 0.03, 0.0],
            fog_color: [0.52, 0.60, 0.70, 0.0],
            prev_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

/// Manages the sky rendering pipeline and uniform buffer.
pub struct SkyRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Cloud history texture for temporal reprojection.
    cloud_history_tex: wgpu::Texture,
    cloud_history_view: wgpu::TextureView,
    cloud_history_sampler: wgpu::Sampler,
    /// Average sky colour estimate (set during update_uniforms) — used by the
    /// CPU light baker as the ambient colour for open-sky probe rays.
    pub last_sky_color: [f32; 3],
}

impl SkyRenderer {
    /// Create the sky renderer. Called once at engine startup.
    ///
    /// Parameters:
    /// - `device`: GPU device for creating buffers and pipelines.
    /// - `window`: Required for wgpu 29's Surface<'static> type inference.
    /// - `format`: Surface texture format (must match the main render pass).
    pub fn new(
        device: &wgpu::Device,
        _window: &Arc<Window>,
        _format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        // ── Bind group layout ────────────────────────────────────────────────
        // Binding 0: uniform buffer (SkyUniforms).
        // Binding 1: cloud history texture (previous frame cloud result).
        // Binding 2: sampler for cloud history texture.
        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Sky BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
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
                ],
            },
        );

        // ── Cloud history texture ────────────────────────────────────────────
        // Stores previous frame's cloud result for temporal reprojection.
        let cloud_history_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cloud History"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let cloud_history_view = cloud_history_tex.create_view(&Default::default());
        let cloud_history_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // ── Uniform buffer ──────────────────────────────────────────────────
        // Pre-allocated with default data. Updated every frame.
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sky Uniform"),
            size: std::mem::size_of::<SkyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind group ──────────────────────────────────────────────────────
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&cloud_history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&cloud_history_sampler),
                },
            ],
        });

        // ── Shader module ───────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sky.wgsl").into()),
        });

        // ── Pipeline layout ─────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Sky Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            },
        );

        // ── Render pipeline ─────────────────────────────────────────────────
        // Fullscreen triangle: no vertex buffer, vertex_index generates positions.
        // Depth test: LessEqual — sky writes at depth=1.0, geometry overwrites.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&pipeline_layout),

            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[], // No vertex buffer — fullscreen triangle from vertex_index
            },

            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    // Sky colour rendered linear into a dedicated Rgba16Float
                    // target; the deferred lighting pass samples it with no
                    // sRGB decode and composites it over empty-depth pixels.
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),

            primitive: wgpu::PrimitiveState::default(),

            // Depth test: LessEqual so geometry overwrites sky at depth=1.0.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),

            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buf,
            bind_group,
            cloud_history_tex,
            cloud_history_view,
            cloud_history_sampler,
            last_sky_color: [0.3, 0.5, 0.8],
        }
    }

    /// Recreate the cloud-history texture and bind group at a new window size.
    /// Must be called from the main renderer's resize path, otherwise the
    /// cloud-history copy overruns after the window grows.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let cloud_history_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cloud History"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let cloud_history_view = cloud_history_tex.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&cloud_history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.cloud_history_sampler),
                },
            ],
        });
        self.cloud_history_tex = cloud_history_tex;
        self.cloud_history_view = cloud_history_view;
        self.bind_group = bind_group;
    }

    /// Best-known average sky colour. The CPU light baker falls back to this
    /// for open-sky probe rays before the first frame's sky update runs.
    pub fn average_sky_color_estimate(&self) -> glam::Vec3 {
        glam::Vec3::from(self.last_sky_color)
    }

    /// Update sky uniform data. Call once per frame before rendering.
    ///
    /// Computes the inverse view-projection matrix and packs all environment
    /// data into the GPU uniform buffer.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        prev_view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        sky_params: &SkyParams,
        cloud_params: &CloudParams,
        fog_color_rgb: [f32; 3],
        fog_density: f32,
        elapsed_time: f32,
        screen_width: f32,
        screen_height: f32,
        storm_darken: f32,
        lightning_intensity: f32,
    ) {
        let inv_vp = view_proj.inverse();
        let mut cloud_data = cloud_params.to_uniform_data();
        // Pack storm_darken and lightning into cloud_type.yz (previously unused).
        cloud_data.cloud_type[1] = storm_darken;
        cloud_data.cloud_type[2] = lightning_intensity;

        let uniforms = SkyUniforms {
            sky: sky_params.to_uniform_data(),
            cloud: cloud_data,
            inv_view_proj: inv_vp.to_cols_array_2d(),
            camera_pos_time: [camera_pos.x, camera_pos.y, camera_pos.z, elapsed_time],
            screen_fog: [screen_width, screen_height, fog_density, 0.0],
            fog_color: [fog_color_rgb[0], fog_color_rgb[1], fog_color_rgb[2], 0.0],
            prev_view_proj: prev_view_proj.to_cols_array_2d(),
        };

        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    /// Render the sky. Call inside the main render pass, BEFORE geometry.
    ///
    /// The sky writes at depth = 1.0 (far plane). Any geometry drawn after
    /// this will naturally overwrite sky pixels where geometry is closer.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        // Draw fullscreen triangle (3 vertices, 1 instance).
        pass.draw(0..3, 0..1);
    }

    /// Returns a reference to the cloud history texture (for copy_texture_to_texture).
    pub fn cloud_history_tex(&self) -> &wgpu::Texture {
        &self.cloud_history_tex
    }
}
