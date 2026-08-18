// src/renderer/water.rs
// Water surface renderer.
//
// ── Architecture ────────────────────────────────────────────────────────────
// WaterRenderer handles:
//   - A dedicated render pipeline for water surfaces
//   - Water uniform buffer (wave params, colours, lighting)
//   - A fullscreen water mesh that gets displaced by Gerstner waves in the vertex shader
//   - Reading the scene colour + depth buffer for refraction/absorption
//
// Integration:
//   - draw_world() calls water_renderer.render() after particles, before post-processing
//   - WaterSurface component on entities controls wave height, colour, etc.
//   - The water pass renders to scene_view (same as geometry) with alpha blending

use wgpu::util::DeviceExt;


/// GPU uniform data matching WaterUniforms in water.wgsl.
/// Total: 192 bytes (12 × vec4).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuWaterUniforms {
    pub wave_params:  [f32; 4],  // x=height, y=speed, z=choppy, w=time
    pub wave_dir_a:   [f32; 4],  // xyz=direction, w=steepness
    pub wave_dir_b:   [f32; 4],  // xyz=direction, w=steepness
    pub wave_dir_c:   [f32; 4],  // xyz=direction, w=steepness
    pub deep_color:   [f32; 4],  // rgb + unused
    pub shallow_color:[f32; 4],  // rgb + opacity
    pub light_dir:    [f32; 4],  // xyz + unused
    pub light_color:  [f32; 4],  // rgb + specular_power
    pub camera_pos:   [f32; 4],  // xyz + foam_intensity
    pub view_proj:    [[f32; 4]; 4],
    pub inv_view_proj:[[f32; 4]; 4],
}

/// Minimal vertex for the water mesh (position + flat normal).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterVertex {
    pub position: [f32; 3],
    pub _pad: f32,
    pub normal: [f32; 3],
    pub _pad2: f32,
}

pub struct WaterRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    num_indices: u32,
}

impl WaterRenderer {
    /// Create the water renderer.
    ///
    /// - `device`: wgpu device
    /// - `surf_fmt`: surface texture format (for the render target)
    pub fn new(device: &wgpu::Device, surf_fmt: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("water.wgsl").into()),
        });

        // Bind group layout: group(0) = uniforms + scene + depth + samplers
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water BGL"),
            entries: &[
                // binding 0: WaterUniforms
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
                // binding 1: t_scene (scene colour for refraction)
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
                // binding 2: s_scene
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: t_depth
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 4: s_depth
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Pipeline layout
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Vertex buffer layout
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WaterVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 0, format: wgpu::VertexFormat::Float32x3, offset: 0 },
                wgpu::VertexAttribute { shader_location: 1, format: wgpu::VertexFormat::Float32x3, offset: 16 },
            ],
        };

        // Render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_water"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[vertex_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_water"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surf_fmt,
                    // Alpha blending: water is semi-transparent.
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
                cull_mode: None, // render both sides
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false), // water is transparent
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
            label: Some("Water Uniforms"),
            size: std::mem::size_of::<GpuWaterUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Generate a flat water mesh (grid of triangles).
        // The vertex shader applies Gerstner wave displacement.
        let (vertex_buffer, index_buffer, num_indices) = Self::create_grid_mesh(device, 100, 100.0);

        Self {
            pipeline,
            uniform_buf,
            vertex_buffer,
            index_buffer,
            bind_group_layout,
            num_indices,
        }
    }

    /// Generate a flat grid mesh for the water surface.
    /// The grid is centred at origin, extending from -size/2 to +size/2.
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
                vertices.push(WaterVertex {
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
            label: Some("Water Vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer, indices.len() as u32)
    }

    /// Render water surfaces into the scene.
    ///
    /// Weather intensity boosts wave height and speed during storms.
    pub fn render(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        scene_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        surface_sampler: &wgpu::Sampler,
        water_entities: &[(&crate::components::Position, &crate::components::WaterSurface)],
        vp: glam::Mat4,
        inv_vp: glam::Mat4,
        cam_pos: glam::Vec3,
        light_dir: [f32; 3],
        light_color: [f32; 3],
        elapsed: f32,
        weather_intensity: f32,
        wind_strength: f32,
    ) {
        if water_entities.is_empty() {
            return;
        }

        let (_, first) = water_entities[0];

        // Weather-driven water roughness: storms increase wave height and speed.
        let storm_boost = weather_intensity.clamp(0.0, 1.0);
        let wave_height = first.wave_height * (1.0 + storm_boost * 2.0);
        let wave_speed = first.wave_speed * (1.0 + storm_boost * 1.5 + wind_strength * 0.3);

        let uniforms = GpuWaterUniforms {
            wave_params: [wave_height, wave_speed, 1.0, elapsed],
            wave_dir_a:  [1.0, 0.0, 0.0, 0.5],
            wave_dir_b:  [0.0, 1.0, 0.0, 0.3],
            wave_dir_c:  [0.707, 0.707, 0.0, 0.2],
            deep_color:  [first.deep_color[0], first.deep_color[1], first.deep_color[2], 0.0],
            shallow_color: [first.shallow_color[0], first.shallow_color[1], first.shallow_color[2], first.opacity],
            light_dir:   [light_dir[0], light_dir[1], light_dir[2], 0.0],
            light_color: [light_color[0], light_color[1], light_color[2], first.specular_power],
            camera_pos:  [cam_pos.x, cam_pos.y, cam_pos.z, first.foam_intensity],
            view_proj:   vp.to_cols_array_2d(),
            inv_view_proj: inv_vp.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Depth sampler for the depth buffer
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(scene_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(surface_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&depth_sampler) },
            ],
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}
