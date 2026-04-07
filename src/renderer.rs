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

use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::assets::{AssetStore, Mesh};
use crate::camera::Camera;
use crate::components::{Position, Renderable};
use hecs::World;

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
    depth_texture: wgpu::Texture,
    depth_view:    wgpu::TextureView,
    pub features:  RenderFeatures,
    pub adapter_info: wgpu::AdapterInfo,
}

impl Renderer {
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

        // ── Vertex buffer ─────────────────────────────────────────────────
        // 1024 vertices × size_of Vertex. Overwritten per entity each frame.
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Vertices"),
            size:               (1024 * std::mem::size_of::<crate::assets::mesh::Vertex>()) as u64,
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
            depth_texture,
            depth_view,
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
    }

    // draw_world() — renders every entity with a Position + Renderable component.
    // Called once per frame from main.rs.
    pub fn draw_world(
        &self,
        world:  &World,
        meshes: &AssetStore<Mesh>,
        camera: &dyn Camera,
    ) {
        // Build and upload the per-frame uniform.
        let vp      = camera.view_projection_matrix();
        let cam_pos = camera.position();
        let uniforms = GpuUniforms {
            view_proj:   vp.to_cols_array_2d(),
            camera_pos:  cam_pos.to_array(),
            _pad0:       0.0,
            light_dir:   glam::Vec3::new(0.5, 1.0, 0.8).normalize().to_array(),
            _pad1:       0.0,
            light_color: [1.0, 1.0, 1.0],
            _pad2:       0.0,
        };
        self.queue.write_buffer(
            &self.uniform_buf, 0,
            bytemuck::bytes_of(&uniforms),
        );

        // Acquire the next swapchain frame.
        let frame   = self.surface.get_current_texture()
            .expect("Failed to acquire swapchain texture");
        let view    = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Frame Encoder") }
        );

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
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

            // Iterate every entity that has both a Position and Renderable.
            // hecs 0.11: query() returns an iterator directly; .iter() still works.
            for (_entity, (pos, renderable)) in
                world.query::<(&Position, &Renderable)>().iter()
            {
                if let Some(mesh) = meshes.get(&renderable.mesh) {

                    // Build per-entity vertices: apply scale and world position.
                    // The camera matrix in the shader handles world → clip conversion.
                    let colored_verts: Vec<crate::assets::mesh::Vertex> = mesh
                        .vertices
                        .iter()
                        .map(|v| {
                            let mut out = *v;
                            out.position = [
                                v.position[0] * renderable.scale[0] + pos.x,
                                v.position[1] * renderable.scale[1] + pos.y,
                                v.position[2] * renderable.scale[2] + pos.z,
                            ];
                            out.color     = renderable.color;
                            out.metallic  = renderable.metallic;
                            out.roughness = renderable.roughness;
                            out.ao        = renderable.ao;
                            out
                        })
                        .collect();

                    let bytes = bytemuck::cast_slice(&colored_verts);
                    // Guard: don't write more than the buffer holds.
                    if bytes.len() <= (self.vertex_buffer.size() as usize) {
                        self.queue.write_buffer(&self.vertex_buffer, 0, bytes);
                        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                        pass.draw(0..colored_verts.len() as u32, 0..1);
                    } else {
                        eprintln!(
                            "[Renderer] Mesh too large ({} verts), skipping. \
                             Increase vertex buffer size.",
                            colored_verts.len()
                        );
                    }
                }
            }
        } // pass is dropped here — commands are finalised

        self.queue.submit([enc.finish()]);
        frame.present();
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
        wgpu::ImageCopyTexture {
            texture:   &texture,
            mip_level: 0,
            origin:    wgpu::Origin3d::ZERO,
            aspect:    wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::ImageDataLayout {
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