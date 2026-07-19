// src/renderer/lava.rs
// Lava / magma surface renderer.
//
// ── Architecture ────────────────────────────────────────────────────────────
// LavaRenderer handles:
//   - A dedicated render pipeline for lava surfaces
//   - Lava uniform buffer (rock/emissive colours, flow params, time)
//   - A flat mesh that gets displacement from the vertex shader
//   - Opaque rendering with emissive output (drives bloom)
//
// Integration:
//   - draw_world() calls lava_renderer.render() after water, before SSR
//   - LavaSurface component on entities controls colour, flow, emissive
//   - Lava is opaque — no alpha blending, no refraction reads

use wgpu::util::DeviceExt;

use crate::components::{LavaSurface, Position};

/// GPU uniform data matching LavaUniforms in lava.wgsl.
/// Total: 128 bytes (8 × vec4).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLavaUniforms {
    pub rock_color:     [f32; 4],  // rgb + opacity
    pub emissive_color: [f32; 4],  // rgb + intensity
    pub params:         [f32; 4],  // flow_speed, crack_scale, crack_threshold, displacement_amp
    pub time:           [f32; 4],  // elapsed, pad
    pub view_proj:      [[f32; 4]; 4],
}

/// Minimal vertex for the lava mesh (position + normal).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LavaVertex {
    pub position: [f32; 3],
    pub _pad: f32,
    pub normal: [f32; 3],
    pub _pad2: f32,
}

pub struct LavaRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    num_indices: u32,
}

impl LavaRenderer {
    /// Create the lava renderer.
    ///
    /// - `device`: wgpu device
    /// - `surf_fmt`: surface texture format (for the render target)
    pub fn new(device: &wgpu::Device, surf_fmt: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lava Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lava.wgsl").into()),
        });

        // Bind group layout: group(0) = uniforms only (no textures needed —
        // lava is fully procedural, no sampling required).
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lava BGL"),
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
            label: Some("Lava Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Vertex buffer layout
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LavaVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 0, format: wgpu::VertexFormat::Float32x3, offset: 0 },
                wgpu::VertexAttribute { shader_location: 1, format: wgpu::VertexFormat::Float32x3, offset: 16 },
            ],
        };

        // Render pipeline — opaque, no blending, depth write ON.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lava Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_lava"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[vertex_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_lava"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surf_fmt,
                    blend: None, // opaque — no blending
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
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

        // Uniform buffer
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lava Uniforms"),
            size: std::mem::size_of::<GpuLavaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Generate a flat mesh — same grid approach as water.
        let (vertex_buffer, index_buffer, num_indices) = Self::create_grid_mesh(device, 64, 60.0);

        Self {
            pipeline,
            uniform_buf,
            vertex_buffer,
            index_buffer,
            bind_group_layout,
            num_indices,
        }
    }

    /// Generate a flat grid mesh for the lava surface.
    fn create_grid_mesh(
        device: &wgpu::Device,
        resolution: u32,
        size: f32,
    ) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let half = size * 0.5;
        let step = size / resolution as f32;
        let vert_count = (resolution + 1) * (resolution + 1);
        let mut vertices = Vec::with_capacity(vert_count as usize);

        for z in 0..=resolution {
            for x in 0..=resolution {
                let px = -half + x as f32 * step;
                let pz = -half + z as f32 * step;
                vertices.push(LavaVertex {
                    position: [px, 0.0, pz],
                    _pad: 0.0,
                    normal: [0.0, 1.0, 0.0],
                    _pad2: 0.0,
                });
            }
        }

        let mut indices = Vec::with_capacity((resolution * resolution * 6) as usize);
        for z in 0..resolution {
            for x in 0..resolution {
                let tl = z * (resolution + 1) + x;
                let tr = tl + 1;
                let bl = (z + 1) * (resolution + 1) + x;
                let br = bl + 1;
                indices.push(tl);
                indices.push(bl);
                indices.push(tr);
                indices.push(tr);
                indices.push(bl);
                indices.push(br);
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lava Vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lava Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer, indices.len() as u32)
    }

    /// Render lava surfaces into the scene.
    ///
    /// - `pass`: the active render pass (writes into scene_view)
    /// - `queue`: for uniform uploads
    /// - `device`: for bind group creation
    /// - `lava_entities`: list of (position, LavaSurface) for each lava entity
    /// - `vp`: view-projection matrix
    /// - `elapsed`: total elapsed time
    pub fn render(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        lava_entities: &[(&Position, &LavaSurface)],
        vp: glam::Mat4,
        elapsed: f32,
    ) {
        if lava_entities.is_empty() {
            return;
        }

        // Merge all lava surfaces into a single draw for now.
        // TODO: per-entity rendering or instancing for many lava bodies.
        let (_, first) = lava_entities[0];

        let uniforms = GpuLavaUniforms {
            rock_color:     [first.rock_color[0], first.rock_color[1], first.rock_color[2], first.opacity],
            emissive_color: [first.emissive_color[0], first.emissive_color[1], first.emissive_color[2], first.emissive_intensity],
            params:         [first.flow_speed, first.crack_scale, first.crack_threshold, first.displacement_amp],
            time:           [elapsed, 0.0, 0.0, 0.0],
            view_proj:      vp.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lava BG"),
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
