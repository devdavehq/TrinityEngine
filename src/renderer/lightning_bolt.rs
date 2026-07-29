// src/renderer/lightning_bolt.rs
// ──────────────────────────────────────────────────────────────────────────────
// Visual lightning bolt renderer.
//
// Draws the actual lightning bolt geometry from bolt_origin to bolt_target
// using the LightningState bolt data. Renders as a bright, jagged line
// with glow (additive blend) in the main pass.
//
// The bolt is rendered as a series of line segments with random mid-point
// displacement, giving it the classic branching lightning appearance.
//
// Architecture:
//   GpuLightningBolt — vertex data for a single bolt
//   LightningBoltRenderer — creates pipeline, uploads vertices, renders
//
// Called from draw_world() after fire pass, before SSR.
// ──────────────────────────────────────────────────────────────────────────────

use crate::environment::lightning::LightningState;
use std::mem;

// ── Lightning bolt vertex ────────────────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightningBoltVertex {
    position: [f32; 3],
    brightness: f32,
}

impl LightningBoltVertex {
    fn new(pos: [f32; 3], brightness: f32) -> Self {
        Self { position: pos, brightness }
    }
}

// ── Lightning bolt uniforms ──────────────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightningBoltUniforms {
    view_proj: [[f32; 4]; 4],
    bolt_color: [f32; 4],
    glow_intensity: f32,
    bolt_width: f32,
    _pad: [f32; 2],
}

// ── LightningBoltRenderer ────────────────────────────────────────────────────
pub struct LightningBoltRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    max_vertices: usize,
    vertex_count: usize,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl LightningBoltRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let max_vertices = 512;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lightning Bolt Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lightning Bolt BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lightning Bolt Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lightning Bolt Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: mem::size_of::<LightningBoltVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        shader_location: 0,
                        offset: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }, wgpu::VertexAttribute {
                        shader_location: 1,
                        offset: 12,
                        format: wgpu::VertexFormat::Float32,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,  // Additive
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lightning Bolt VB"),
            size: (max_vertices * mem::size_of::<LightningBoltVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lightning Bolt Uniforms"),
            size: mem::size_of::<LightningBoltUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lightning Bolt BG"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            bind_group_layout,
            vertex_buffer,
            max_vertices,
            vertex_count: 0,
            uniform_buf,
            bind_group,
        }
    }

    /// Generate bolt vertices from LightningState and render.
    pub fn render(
        &mut self,
        lightning: &LightningState,
        view_proj: &[[f32; 4]; 4],
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        // Only render when a bolt was just fired
        if lightning.thunder_just_fired || lightning.flash_intensity > 0.05 {
            let verts = self.generate_bolt_vertices(lightning);
            self.vertex_count = verts.len();
            if self.vertex_count == 0 || self.vertex_count > self.max_vertices {
                return;
            }

            let vert_data = bytemuck::cast_slice(&verts);
            queue.write_buffer(&self.vertex_buffer, 0, vert_data);

            let uniforms = LightningBoltUniforms {
                view_proj: *view_proj,
                bolt_color: [0.85, 0.9, 1.0, 1.0],
                glow_intensity: lightning.flash_intensity,
                bolt_width: 3.0,
                _pad: [0.0; 2],
            };
            queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lightning Bolt Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..self.vertex_count as u32, 0..1);
        }
    }

    /// Generate jagged line segments from bolt_origin to bolt_target.
    fn generate_bolt_vertices(&self, lightning: &LightningState) -> Vec<LightningBoltVertex> {
        let origin = lightning.bolt_origin;
        let target = lightning.bolt_target;
        let mut verts = Vec::new();

        // Main bolt: 12 segments with random displacement
        let segments = 12;
        let displacement = 8.0; // max displacement per segment

        let dir = [
            target[0] - origin[0],
            target[1] - origin[1],
            target[2] - origin[2],
        ];
        let length = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if length < 0.01 { return verts; }

        // Generate midpoints with displacement
        let mut points: Vec<[f32; 3]> = Vec::with_capacity(segments + 2);
        points.push(origin);

        for i in 1..segments {
            let t = i as f32 / segments as f32;
            let base = [
                origin[0] + dir[0] * t,
                origin[1] + dir[1] * t,
                origin[2] + dir[2] * t,
            ];

            // Perpendicular displacement using sin-hash pseudo-random
            let hash1 = (t * 137.3 + origin[0] * 0.1).sin() * 43758.5453;
            let hash2 = (t * 251.7 + origin[2] * 0.1).sin() * 43758.5453;
            let disp = displacement * (1.0 - (t - 0.5).abs() * 2.0); // more displacement in middle

            points.push([
                base[0] + (hash1.fract() - 0.5) * disp,
                base[1] + (hash2.fract() - 0.5) * disp * 0.3,
                base[2] + (hash1.fract() - 0.5) * disp,
            ]);
        }
        points.push(target);

        // Convert to line strip vertices
        for (i, pt) in points.iter().enumerate() {
            let brightness = if i == 0 || i == points.len() - 1 { 1.0 } else { 0.85 };
            verts.push(LightningBoltVertex::new(*pt, brightness));
        }

        // Add 2-3 small branches
        for b in 0..3 {
            let branch_start_idx = 2 + b * 3;
            if branch_start_idx >= points.len() { break; }
            let bp = points[branch_start_idx];

            let hash_b = ((b as f32 + 0.5) * 99.1 + origin[0] * 0.07).sin();
            let branch_len = 15.0 + hash_b.abs() * 25.0;
            let branch_dir = [
                (hash_b * 2.1).sin() * branch_len,
                -branch_len * 0.6,
                (hash_b * 3.7).cos() * branch_len,
            ];

            let branch_end = [
                bp[0] + branch_dir[0],
                bp[1] + branch_dir[1],
                bp[2] + branch_dir[2],
            ];

            // Branch midpoint
            let mid = [
                (bp[0] + branch_end[0]) * 0.5 + (hash_b.sin()) * 3.0,
                (bp[1] + branch_end[1]) * 0.5,
                (bp[2] + branch_end[2]) * 0.5 + (hash_b.cos()) * 3.0,
            ];

            verts.push(LightningBoltVertex::new(bp, 0.7));
            verts.push(LightningBoltVertex::new(mid, 0.5));
            verts.push(LightningBoltVertex::new(mid, 0.5));
            verts.push(LightningBoltVertex::new(branch_end, 0.3));
        }

        verts
    }
}

// ── Shader ───────────────────────────────────────────────────────────────────
const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    bolt_color: vec4<f32>,
    glow_intensity: f32,
    bolt_width: f32,
    pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) brightness: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) brightness: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.brightness = in.brightness;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let core = in.brightness * uniforms.glow_intensity;
    let color = uniforms.bolt_color.rgb;
    // Core white, edges blue-ish
    let final_color = mix(color * 0.7, vec3<f32>(1.0, 1.0, 1.0), core);
    return vec4<f32>(final_color * core * 2.0, core);
}
"#;
