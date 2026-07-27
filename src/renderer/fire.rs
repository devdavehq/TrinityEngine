// src/renderer/fire.rs
// Procedural fire / flame surface renderer.
//
// ── Architecture ────────────────────────────────────────────────────────────
// FireRenderer handles:
//   - A dedicated render pipeline for fire surfaces (additive + alpha blend)
//   - Fire uniform buffer (flame colours, flicker params, wind time)
//   - A flat mesh that gets vertex displacement from the shader
//   - Semi-transparent rendering with emissive output (drives bloom)
//
// Integration:
//   - draw_world() calls fire_renderer.render() after lava, before SSR
//   - FireSurface component on entities controls colour, intensity, height
//   - Fire is semi-transparent — alpha blending, no depth write

use wgpu::util::DeviceExt;

use crate::components::{FireSurface, Position};

/// GPU uniform data matching FireUniforms in fire.wgsl.
/// Total: 128 bytes (8 × vec4).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFireUniforms {
    pub base_color:  [f32; 4],  // rgb + intensity
    pub tip_color:   [f32; 4],  // rgb + unused
    pub params:      [f32; 4],  // flame_speed, noise_scale, flicker_strength, flame_height
    pub wind_time:   [f32; 4],  // elapsed, wind_x, wind_z, unused
    pub view_proj:   [[f32; 4]; 4],
}

/// Minimal vertex for the fire mesh (position + normal).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FireVertex {
    pub position: [f32; 3],
    pub _pad: f32,
    pub normal: [f32; 3],
    pub _pad2: f32,
}

pub struct FireRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    num_indices: u32,
}

impl FireRenderer {
    /// Create the fire renderer.
    ///
    /// - `device`: wgpu device
    /// - `surf_fmt`: surface texture format (for the render target)
    pub fn new(device: &wgpu::Device, surf_fmt: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fire Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fire.wgsl").into()),
        });

        // Bind group layout: group(0) = uniforms only (no textures needed —
        // fire is fully procedural, no sampling required).
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fire BGL"),
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
            ],
        });

        // Pipeline layout
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fire Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Vertex buffer layout
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FireVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 0, format: wgpu::VertexFormat::Float32x3, offset: 0 },
                wgpu::VertexAttribute { shader_location: 1, format: wgpu::VertexFormat::Float32x3, offset: 16 },
            ],
        };

        // Render pipeline — semi-transparent, alpha blend, no depth write.
        // Fire is always visible through geometry behind it (flames don't occlude).
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fire Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fire"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[vertex_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_fire"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surf_fmt,
                    // Additive + alpha blend: fire glows brighter where flames overlap.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,  // additive
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None, // cross-plane: both sides visible from any angle
                ..Default::default()
            },
            // No depth write — fire is transparent and doesn't occlude anything.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Uniform buffer
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fire Uniforms"),
            size: std::mem::size_of::<GpuFireUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Generate a flat vertical quad mesh for the fire surface.
        let (vertex_buffer, index_buffer, num_indices) = Self::create_fire_quad(device);

        Self {
            pipeline,
            uniform_buf,
            vertex_buffer,
            index_buffer,
            bind_group_layout,
            num_indices,
        }
    }

    /// Create a cross-plane mesh for the fire effect (two intersecting quads at 90°).
    /// This ensures fire looks 3D from any viewing angle.
    /// The vertex shader displaces vertices upward based on noise,
    /// creating the flame shape procedurally.
    fn create_fire_quad(
        device: &wgpu::Device,
    ) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let half_w = 1.0;
        let height = 2.0;
        let segments_h = 8;
        let segments_w = 4;

        let verts_per_quad = (segments_h + 1) * (segments_w + 1);
        let tris_per_quad = segments_h * segments_w * 2;
        let mut vertices = Vec::with_capacity(verts_per_quad * 2);
        let mut indices = Vec::with_capacity(tris_per_quad * 3 * 2);

        // Quad A: faces the camera (XY plane, normal along +Z).
        for y in 0..=segments_h {
            for x in 0..=segments_w {
                let fx = x as f32 / segments_w as f32;
                let fy = y as f32 / segments_h as f32;
                let px = (fx - 0.5) * 2.0 * half_w;
                let py = fy * height;
                vertices.push(FireVertex {
                    position: [px, py, 0.0],
                    _pad: 0.0,
                    normal: [0.0, 0.0, 1.0],
                    _pad2: 0.0,
                });
            }
        }

        // Quad B: rotated 90° around Y axis (ZY plane, normal along +X).
        for y in 0..=segments_h {
            for x in 0..=segments_w {
                let fx = x as f32 / segments_w as f32;
                let fy = y as f32 / segments_h as f32;
                let pz = (fx - 0.5) * 2.0 * half_w;
                let py = fy * height;
                vertices.push(FireVertex {
                    position: [0.0, py, pz],
                    _pad: 0.0,
                    normal: [1.0, 0.0, 0.0],
                    _pad2: 0.0,
                });
            }
        }

        // Generate indices for both quads.
        for quad in 0..2u32 {
            let base = quad * verts_per_quad as u32;
            for y in 0..segments_h {
                for x in 0..segments_w {
                    let tl = base + y as u32 * (segments_w as u32 + 1) + x as u32;
                    let tr = tl + 1;
                    let bl = base + (y as u32 + 1) * (segments_w as u32 + 1) + x as u32;
                    let br = bl + 1;
                    indices.push(tl);
                    indices.push(bl);
                    indices.push(tr);
                    indices.push(tr);
                    indices.push(bl);
                    indices.push(br);
                }
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fire Vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fire Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer, indices.len() as u32)
    }

    /// Render fire surfaces into the scene.
    ///
    /// - `pass`: the active render pass (writes into scene_view)
    /// - `queue`: for uniform uploads
    /// - `device`: for bind group creation
    /// - `fire_entities`: list of (position, FireSurface) for each fire entity
    /// - `vp`: view-projection matrix
    /// - `elapsed`: total elapsed time
    /// - `wind_dir`: normalised wind direction [x, y, z]
    /// - `wind_strength`: wind strength 0..1
    pub fn render(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        fire_entities: &[(&Position, &FireSurface)],
        vp: glam::Mat4,
        elapsed: f32,
        wind_dir: [f32; 3],
        wind_strength: f32,
    ) {
        if fire_entities.is_empty() {
            return;
        }

        // Render each fire entity individually (they have different positions/colours).
        for (_pos, fire_surf) in fire_entities {
            let uniforms = GpuFireUniforms {
                base_color: [
                    fire_surf.base_color[0],
                    fire_surf.base_color[1],
                    fire_surf.base_color[2],
                    fire_surf.intensity,
                ],
                tip_color: [
                    fire_surf.tip_color[0],
                    fire_surf.tip_color[1],
                    fire_surf.tip_color[2],
                    0.0,
                ],
                params: [
                    fire_surf.flame_speed,
                    fire_surf.noise_scale,
                    fire_surf.flicker_strength,
                    fire_surf.flame_height,
                ],
                wind_time: [
                    elapsed,
                    wind_dir[0] * wind_strength,
                    wind_dir[2] * wind_strength,
                    0.0,
                ],
                view_proj: vp.to_cols_array_2d(),
            };
            queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Fire BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                ],
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
    }
}
