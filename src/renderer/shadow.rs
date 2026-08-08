#![allow(dead_code)]

// src/renderer/shadow.rs
// Cascaded Shadow Maps (CSM) for the directional sun light.
//
// ── wgpu 29 changes ─────────────────────────────────────────────────────────
// • DepthStencilState.depth_write_enabled: bool → Option<bool>  (Some(true))
// • DepthStencilState.depth_compare:       CompareFunction → Option<CompareFunction>
// • RenderPipelineDescriptor.multiview → multiview_mask: Option<NonZeroU32>
// • VertexState / FragmentState: entry_point = Some("name"), compilation_options required
// • fragment: None still valid for depth-only pipelines

use glam::{Mat4, Vec3};
use wgpu::{Device, Queue};

// How many cascades. 3 = close/medium/far.
pub const CASCADE_COUNT: usize = 3;

// ShadowCascade holds one cascade's GPU depth texture and CPU light matrix.
// Each cascade owns its own tiny uniform buffer + bind group so the depth
// shader can be re-bound per cascade with a single mat4 (the light matrix).
pub struct ShadowCascade {
    pub texture:      wgpu::Texture,
    pub view:         wgpu::TextureView,
    pub light_matrix: Mat4,
    pub near_dist:    f32,
    pub far_dist:     f32,
    pub uniform_buf:  wgpu::Buffer,
    pub bind_group:   wgpu::BindGroup,
}

// ShadowSystem manages all cascades and the depth-only render pipeline.
pub struct ShadowSystem {
    pub cascades:    Vec<ShadowCascade>,
    pub sampler:     wgpu::Sampler,   // comparison sampler for PCF
    pub pipeline:    wgpu::RenderPipeline,
    pub shadow_bgl:  wgpu::BindGroupLayout,
}

impl ShadowSystem {
    pub fn new(device: &Device, shadow_resolution: u32) -> Self {
        // ── Cascade textures ───────────────────────────────────────────────
        // Distance ranges per cascade.
        let (near, far) = [
            (0.1_f32, 10.0_f32),   // close  — sharp
            (10.0,    40.0),        // medium
            (40.0,   150.0),        // far    — softer
        ][0]; // placeholder; replaced by per-index match below
        let _ = (near, far);

        // ── Bind group layout for shadow pass ─────────────────────────────
        let shadow_bgl = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label:   Some("Shadow BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                }],
            },
        );

        let cascades: Vec<ShadowCascade> = (0..CASCADE_COUNT)
            .map(|i| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Shadow Cascade {}", i)),
                    size:  wgpu::Extent3d {
                        width:                 shadow_resolution,
                        height:                shadow_resolution,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count:    1,
                    dimension:       wgpu::TextureDimension::D2,
                    format:          wgpu::TextureFormat::Depth32Float,
                    // RENDER_ATTACHMENT: rendered into during shadow pass.
                    // TEXTURE_BINDING:   sampled during main pass.
                    usage:  wgpu::TextureUsages::RENDER_ATTACHMENT
                          | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                // DepthOnly aspect is required for use as a depth attachment.
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    aspect: wgpu::TextureAspect::DepthOnly,
                    ..Default::default()
                });
                // Distance ranges per cascade.
                let (near, far) = match i {
                    0 => (0.1_f32,  10.0_f32),  // close  — sharp
                    1 => (10.0,     40.0),        // medium
                    _ => (40.0,    150.0),        // far    — softer
                };
                let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some(&format!("Shadow Cascade {} Uniform", i)),
                    size:               64,
                    usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label:   Some(&format!("Shadow Cascade {} BG", i)),
                    layout:  &shadow_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    }],
                });
                ShadowCascade {
                    texture,
                    view,
                    light_matrix: Mat4::IDENTITY,
                    near_dist: near,
                    far_dist:  far,
                    uniform_buf,
                    bind_group,
                }
            })
            .collect();

        // ── Comparison sampler ─────────────────────────────────────────────
        // CompareFunction::LessEqual enables hardware PCF filtering.
        // The sampler returns 0..1 (fraction of samples that pass the test).
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            // ── wgpu 29: MipmapFilterMode instead of FilterMode ────────────
            mipmap_filter:  wgpu::MipmapFilterMode::Nearest,
            compare:        Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // ── Shadow depth-only pipeline ────────────────────────────────────
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
        });

        // Instance model matrix is read from vertex buffer slot 1
        // (step mode = Instance). It matches InstanceData in instancing.rs.
        const INSTANCE_STRIDE: u64 = 96;
        const MODEL_OFFSETS: [u64; 4] = [0, 16, 32, 48];
        const MODEL_LOCATIONS: [u32; 4] = [1, 2, 3, 4];

        let instance_attrs: Vec<wgpu::VertexAttribute> = (0..4)
            .map(|k| wgpu::VertexAttribute {
                shader_location: MODEL_LOCATIONS[k],
                format:          wgpu::VertexFormat::Float32x4,
                offset:          MODEL_OFFSETS[k],
            })
            .collect();

        let shadow_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:              Some("Shadow Layout"),
                // ── wgpu 29: &[Option<&BindGroupLayout>] ──────────────────
                bind_group_layouts: &[Some(&shadow_bgl)],
                immediate_size:     0,
            },
        );

        let pipeline = device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label:  Some("Shadow Pipeline"),
                layout: Some(&shadow_layout),
                vertex: wgpu::VertexState {
                    module: &shadow_shader,
                    // ── wgpu 29: entry_point is Option<&str> ─────────────
                    entry_point: Some("vs_shadow"),
                    // ── wgpu 29: compilation_options required ─────────────
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            // Stride matches the full Vertex struct in mesh.rs.
                            array_stride: std::mem::size_of::<crate::assets::mesh::Vertex>() as u64,
                            step_mode:    wgpu::VertexStepMode::Vertex,
                            attributes:   &[wgpu::VertexAttribute {
                                shader_location: 0,
                                format:          wgpu::VertexFormat::Float32x3,
                                offset:          0,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: INSTANCE_STRIDE,
                            step_mode:    wgpu::VertexStepMode::Instance,
                            attributes:   &instance_attrs,
                        },
                    ],
                },
                // No fragment shader — GPU writes depth automatically.
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    // Front-face culling: render back faces into the shadow map.
                    // This eliminates shadow acne without needing large bias values.
                    cull_mode: Some(wgpu::Face::Front),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    // ── wgpu 29: depth_write_enabled is Option<bool> ──────
                    depth_write_enabled: Some(true),
                    // ── wgpu 29: depth_compare is Option<CompareFunction> ─
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    // Slope-scaled bias reduces self-shadowing at angles.
                    bias: wgpu::DepthBiasState {
                        constant:    2,
                        slope_scale: 2.0,
                        clamp:       0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                // ── wgpu 29: multiview renamed to multiview_mask ──────────
                multiview_mask: None,
                // ── wgpu 29: cache field added ────────────────────────────
                cache: None,
            },
        );

        Self { cascades, sampler, pipeline, shadow_bgl }
    }

    // update_light_matrices() — call every frame (or when light/camera moves).
    // Computes tight orthographic projections per cascade.
    pub fn update_light_matrices(
        &mut self,
        queue:          &Queue,
        light_dir:      Vec3,
        camera_pos:     Vec3,
        camera_forward: Vec3,
    ) {
        let light_dir = light_dir.normalize();

        // Stable up vector — switch to forward if light is nearly vertical.
        let up = if light_dir.dot(Vec3::Y).abs() > 0.99 { Vec3::Z } else { Vec3::Y };

        for cascade in &mut self.cascades {
            // Frustum centre for this cascade's distance slice.
            let centre = camera_pos
                + camera_forward * (cascade.near_dist + cascade.far_dist) * 0.5;

            // Bounding sphere radius — conservative but stable.
            let radius = (cascade.far_dist - cascade.near_dist) * 0.5 + 2.0;

            // Orthographic projection tightly fitting the cascade sphere.
            let proj = Mat4::orthographic_rh(
                -radius, radius,
                -radius, radius,
                -radius * 2.0 - 50.0, // extend behind camera for back-lit surfaces
                 radius * 2.0,
            );

            // View matrix centred on this cascade's frustum.
            let cascade_view = Mat4::look_at_rh(
                centre - light_dir * (radius + 50.0),
                centre,
                up,
            );

            cascade.light_matrix = proj * cascade_view;
            let matrix_bytes: Vec<u8> = bytemuck::cast_slice(
                &cascade.light_matrix.to_cols_array_2d(),
            ).to_vec();
            queue.write_buffer(
                &cascade.uniform_buf, 0,
                &matrix_bytes,
            );
        }
    }

    // render_shadow_pass() — render the scene into one cascade's depth texture.
    // Call once per cascade before the main colour pass each frame.
    pub fn render_shadow_pass(
        &self,
        encoder:          &mut wgpu::CommandEncoder,
        vertex_buffer:    &wgpu::Buffer,
        vertex_count:     u32,
        cascade_index:    usize,
    ) {
        let cascade = &self.cascades[cascade_index];

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label:             Some(&format!("Shadow Pass {}", cascade_index)),
            color_attachments: &[], // depth-only — no colour output
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &cascade.view,
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(1.0), // 1.0 = max depth
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &cascade.bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..vertex_count, 0..1);
    }

    /// Depth view for a cascade — used when the caller draws instanced geometry.
    pub fn cascade_view(&self, index: usize) -> &wgpu::TextureView {
        &self.cascades[index].view
    }

    /// Per-cascade light-matrix bind group for the depth pipeline.
    pub fn cascade_bind_group(&self, index: usize) -> &wgpu::BindGroup {
        &self.cascades[index].bind_group
    }
}