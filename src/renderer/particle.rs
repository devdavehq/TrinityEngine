// renderer/particle.rs
// GPU rendering for particles via instanced billboard quads.
//
// Architecture:
//   - ParticleRenderer owns the pipeline, vertex buffer, instance buffer, and bind group.
//   - Each frame: caller provides Vec<GpuParticle>, renderer uploads + draws.
//   - Uses the same fullscreen-quad vertex buffer pattern as the sky renderer,
//     but with a 4-vertex billboard quad instead.
//
// PERFORMANCE:
//   - One draw call for ALL particles (instanced).
//   - Instance buffer grows automatically if needed (doubles capacity).
//   - Alpha blending enabled for transparent particles.

use crate::particles::GpuParticle;
use wgpu::util::DeviceExt;

/// Camera uniform data needed by the particle shader.
/// Minimal subset: just the VP matrix and camera position.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleCameraUniforms {
    view_proj: [[f32; 4]; 4], // 64 bytes
    camera_pos: [f32; 3],     // 12 bytes
    _pad: f32,                //  4 bytes
}

pub struct ParticleRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ParticleRenderer {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("particle.wgsl").into()),
        });

        // ── Bind group layout ──────────────────────────────────────────────
        // group(0): camera uniform buffer
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // ── Vertex buffer: 4 vertices forming a billboard quad ─────────────
        // Two triangles: (0,1,2) and (1,3,2).
        // Vertex positions are local [-0.5, 0.5] — the shader scales by particle size.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ParticleVertex {
            pos: [f32; 2],
        }
        const QUAD_VERTS: [ParticleVertex; 4] = [
            ParticleVertex { pos: [-0.5, -0.5] }, // bottom-left
            ParticleVertex { pos: [ 0.5, -0.5] }, // bottom-right
            ParticleVertex { pos: [-0.5,  0.5] }, // top-left
            ParticleVertex { pos: [ 0.5,  0.5] }, // top-right
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Quad VB"),
            contents: bytemuck::cast_slice(&QUAD_VERTS),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // ── Instance buffer (pre-allocated for 4096 particles) ─────────────
        let initial_cap = 4096;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Instance Buffer"),
            size: (initial_cap * GpuParticle::STRIDE) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Uniform buffer ─────────────────────────────────────────────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Uniforms"),
            size: std::mem::size_of::<ParticleCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle BG"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ── Render pipeline ────────────────────────────────────────────────
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_particle"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[
                    // Slot 0: per-vertex (quad corners)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ParticleVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            shader_location: 0,
                            offset: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    // Slot 1: per-instance (GpuParticle)
                    wgpu::VertexBufferLayout {
                        array_stride: GpuParticle::STRIDE as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute { shader_location: 1, offset: 0,  format: wgpu::VertexFormat::Float32x3 }, // position
                            wgpu::VertexAttribute { shader_location: 2, offset: 12, format: wgpu::VertexFormat::Float32 },   // size
                            wgpu::VertexAttribute { shader_location: 3, offset: 16, format: wgpu::VertexFormat::Float32x4 }, // color
                            wgpu::VertexAttribute { shader_location: 4, offset: 32, format: wgpu::VertexFormat::Float32x3 }, // velocity
                            wgpu::VertexAttribute { shader_location: 5, offset: 44, format: wgpu::VertexFormat::Float32 },   // pad
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_particle"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    // Alpha blending: src * src_alpha + dst * (1 - src_alpha).
                    // Standard transparency for particles.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false), // particles are transparent
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
            vertex_buffer,
            instance_buffer,
            instance_capacity: initial_cap,
            uniform_buf,
            bind_group,
            bind_group_layout,
        }
    }

    /// Draw all particles into the current render pass.
    ///
    /// This must be called WITHIN an active render pass that already has the
    /// depth texture bound. Particles render with depth test but no depth write,
    /// so they correctly appear behind solid geometry.
    pub fn render(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        particles: &[GpuParticle],
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
    ) {
        if particles.is_empty() {
            return;
        }

        // ── Upload camera uniforms ─────────────────────────────────────────
        let uniforms = ParticleCameraUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // ── Grow instance buffer if needed ─────────────────────────────────
        let needed = particles.len() * GpuParticle::STRIDE;
        if needed > self.instance_capacity {
            let new_cap = (self.instance_capacity * 2).max(particles.len());
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Instance Buffer (grow)"),
                size: (new_cap * GpuParticle::STRIDE) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
            // Rebuild bind group if needed (uniform buf didn't change, so no).
        }

        // ── Upload particle instance data ──────────────────────────────────
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(particles),
        );

        // ── Draw instanced quad ────────────────────────────────────────────
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.draw(0..4, 0..particles.len() as u32);
    }
}
