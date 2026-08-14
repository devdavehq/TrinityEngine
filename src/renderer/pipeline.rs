// src/renderer/pipeline.rs
// Creates the wgpu render pipeline and bind group layouts.
// Split out from renderer.rs so it stays readable as we add more passes.
//
// ── wgpu 29 changes vs older versions ──────────────────────────────────────
// • PipelineLayoutDescriptor.bind_group_layouts is now &[Option<&BindGroupLayout>]
//   — wrap each layout in Some(&bgl).
// • PipelineLayoutDescriptor now has `immediate_size: 0` field (set to 0 unless
//   you are using the IMMEDIATES feature, which we are not).
// • VertexState and FragmentState both require:
//   - entry_point: Some("fn_name")  (was: "fn_name")
//   - compilation_options: wgpu::PipelineCompilationOptions::default()
// • DepthStencilState fields are now Option:
//   - depth_write_enabled: Some(true)
//   - depth_compare: Some(wgpu::CompareFunction::LessEqual)
// • RenderPipelineDescriptor.multiview is now multiview_mask: Option<NonZeroU32>
//   (use None for no multiview)
// • RenderPipelineDescriptor.cache: Option<&PipelineCache> field must be set (None)
// • SamplerDescriptor.mipmap_filter is now wgpu::MipmapFilterMode, not FilterMode

use crate::assets::mesh::Vertex;
use wgpu::Device;

// create_bind_group_layouts() returns two layouts:
//   Group 0 — global, changes once per frame (camera uniform + IBL textures)
//   Group 1 — per-material, changes per entity (albedo, normal, metallic-rough)
pub fn create_bind_group_layouts(
    device: &Device,
) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {

    // Helper closures keep the entry declarations concise.
    let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    };

    let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled:   false,
        },
        count: None,
    };

    let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };

    let depth_texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let comparison_sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
        count: None,
    };

    // Group 0: camera/lights uniform (binding 0) + 3 IBL texture pairs (1–6).
    let global_bgl = device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label:   Some("Global BGL"),
            entries: &[
                uniform_entry(0), // GpuUniforms (camera + light)
                texture_entry(1), // ibl_irradiance
                sampler_entry(2),
                texture_entry(3), // ibl_prefilter
                sampler_entry(4),
                texture_entry(5), // brdf_lut
                sampler_entry(6),
                uniform_entry(7), // ShadowData
                depth_texture_entry(8), // t_shadow0
                depth_texture_entry(9), // t_shadow1
                depth_texture_entry(10), // t_shadow2
                comparison_sampler_entry(11), // s_shadow
                uniform_entry(12), // LightUniforms (multi-light array, up to 16)
                uniform_entry(13), // WeatherData (snow_coverage)
                uniform_entry(14), // ProbeControl (baked probe count)
                wgpu::BindGroupLayoutEntry { // BakedProbes (SH array, storage read)
                    binding: 15,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        },
    );

    // Group 1: albedo + normal + metallic-roughness texture pairs.
    let material_bgl = device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label:   Some("Material BGL"),
            entries: &[
                texture_entry(0), // t_albedo
                sampler_entry(1),
                texture_entry(2), // t_normal
                sampler_entry(3),
                texture_entry(4), // t_metallic_roughness
                sampler_entry(5),
                uniform_entry(6), // MaterialExtras (subsurface, clearcoat, etc.)
            ],
        },
    );

    (global_bgl, material_bgl)
}

// create_pipeline() builds the full PBR render pipeline.
// Called once at startup; reused every frame.
pub fn create_pipeline(
    device:       &Device,
    _surf_fmt:     wgpu::TextureFormat,
    global_bgl:   &wgpu::BindGroupLayout,
    material_bgl: &wgpu::BindGroupLayout,
    shader:       &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {

    // ── wgpu 29: bind_group_layouts is &[Option<&BindGroupLayout>] ───────────
    // Each entry must be wrapped in Some(). None means "leave this slot empty".
    let layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label:              Some("PBR Layout"),
            bind_group_layouts: &[Some(global_bgl), Some(material_bgl)],
            // immediate_size: bytes for var<immediate> in shaders.
            // We don't use IMMEDIATES, so 0.
            immediate_size:     0,
        },
    );

    // Byte offsets for each field inside Vertex (must match mesh.rs exactly):
    //   position      [f32;3]  offset  0  (12 bytes)
    //   normal        [f32;3]  offset 12  (12 bytes)
    //   tangent       [f32;3]  offset 24  (12 bytes)
    //   bitangent     [f32;3]  offset 36  (12 bytes)
    //   color         [f32;3]  offset 48  (12 bytes)
    //   metallic      f32      offset 60  ( 4 bytes)
    //   roughness     f32      offset 64  ( 4 bytes)
    //   ao            f32      offset 68  ( 4 bytes)
    //   bone_indices  [u32;4]  offset 72  (16 bytes)
    //   bone_weights  [f32;4]  offset 88  (16 bytes)
    //   Total: 104 bytes (repr(C) padded to 112)
    let vertex_attributes = [
        wgpu::VertexAttribute { shader_location: 0, format: wgpu::VertexFormat::Float32x3, offset:  0 },
        wgpu::VertexAttribute { shader_location: 1, format: wgpu::VertexFormat::Float32x3, offset: 12 },
        wgpu::VertexAttribute { shader_location: 2, format: wgpu::VertexFormat::Float32x3, offset: 24 },
        wgpu::VertexAttribute { shader_location: 3, format: wgpu::VertexFormat::Float32x3, offset: 36 },
        wgpu::VertexAttribute { shader_location: 4, format: wgpu::VertexFormat::Float32x3, offset: 48 },
        wgpu::VertexAttribute { shader_location: 5, format: wgpu::VertexFormat::Float32,   offset: 60 },
        wgpu::VertexAttribute { shader_location: 6, format: wgpu::VertexFormat::Float32,   offset: 64 },
        wgpu::VertexAttribute { shader_location: 7, format: wgpu::VertexFormat::Float32,   offset: 68 },
        // bone indices and weights — used by skinned mesh pipeline (location 8-9)
        wgpu::VertexAttribute { shader_location: 8,  format: wgpu::VertexFormat::Uint32x4, offset: 72 },
        wgpu::VertexAttribute { shader_location: 9,  format: wgpu::VertexFormat::Float32x4, offset: 88 },
    ];

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some("PBR Pipeline"),
        layout: Some(&layout),

        vertex: wgpu::VertexState {
            module: shader,
            // ── wgpu 29: entry_point is Option<&str> ─────────────────────
            entry_point: Some("vs_main"),
            // ── wgpu 29: compilation_options is now required ──────────────
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[
                // Slot 0: per-vertex data (position, normal, color, metallic, roughness, ao)
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &vertex_attributes,
                },
                // Slot 1: per-instance data (model matrix, color_metallic, roughness_ao)
                wgpu::VertexBufferLayout {
                    array_stride: 96, // InstanceData: mat4(64) + vec4(16) + vec4(16) = 96
                    step_mode:    wgpu::VertexStepMode::Instance,
                    attributes:   &[
                        // model matrix — 4 rows of float32x4
                        wgpu::VertexAttribute { shader_location: 14, format: wgpu::VertexFormat::Float32x4, offset:  0 },
                        wgpu::VertexAttribute { shader_location: 15, format: wgpu::VertexFormat::Float32x4, offset: 16 },
                        wgpu::VertexAttribute { shader_location: 16, format: wgpu::VertexFormat::Float32x4, offset: 32 },
                        wgpu::VertexAttribute { shader_location: 17, format: wgpu::VertexFormat::Float32x4, offset: 48 },
                        // color_metallic (rgb + metallic)
                        wgpu::VertexAttribute { shader_location: 18, format: wgpu::VertexFormat::Float32x4, offset: 64 },
                        // roughness_ao_pad
                        wgpu::VertexAttribute { shader_location: 19, format: wgpu::VertexFormat::Float32x4, offset: 80 },
                    ],
                },
            ],
        },

        fragment: Some(wgpu::FragmentState {
            module:      shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[
                // Deferred G-buffer outputs (see shader.wgsl fs_main):
                // 0 = albedo (rgb) + metallic (a)
                // 1 = world normal (rgb, encoded) + roughness (a)
                // 2 = emissive (rgb) + ambient occlusion (a)
                // 3 = material extras (subsurface, clearcoat, clearcoat_roughness, anisotropy)
                Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),

        primitive: wgpu::PrimitiveState {
            // Back-face culling: skip triangles facing away from camera.
            // Halves fragment work for closed meshes.
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },

        // ── wgpu 29: depth_write_enabled and depth_compare are Option<> ──
        // Must be Some() when format has a depth aspect (Depth32Float does).
        depth_stencil: Some(wgpu::DepthStencilState {
            format:              wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare:       Some(wgpu::CompareFunction::LessEqual),
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState::default(),
        }),

        multisample: wgpu::MultisampleState::default(),

        // ── wgpu 29: multiview renamed to multiview_mask ─────────────────
        // None = not doing VR / multi-layer rendering.
        multiview_mask: None,

        // ── wgpu 29: cache field added ────────────────────────────────────
        // Pipeline cache speeds up shader compilation on Android. None elsewhere.
        cache: None,
    })
}

// create_skinning_bind_group_layout() — group 2 joint matrix buffer for GPU skinning.
// Group 2 binding 0: JointData uniform (64 × mat4 = 4096 bytes).
pub fn create_skinning_bgl(device: &Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Skinning BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

// create_skinning_pipeline() — PBR + GPU skinning render pipeline.
// Uses groups 0 (global), 1 (material), and 2 (joint matrices).
pub fn create_skinning_pipeline(
    device:       &Device,
    _surf_fmt:     wgpu::TextureFormat,
    global_bgl:   &wgpu::BindGroupLayout,
    material_bgl: &wgpu::BindGroupLayout,
    skinning_bgl: &wgpu::BindGroupLayout,
    shader:       &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label:              Some("Skinning PBR Layout"),
            bind_group_layouts: &[Some(global_bgl), Some(material_bgl), Some(skinning_bgl)],
            immediate_size:     0,
        },
    );

    let vertex_attributes = [
        wgpu::VertexAttribute { shader_location: 0, format: wgpu::VertexFormat::Float32x3, offset:  0 },
        wgpu::VertexAttribute { shader_location: 1, format: wgpu::VertexFormat::Float32x3, offset: 12 },
        wgpu::VertexAttribute { shader_location: 2, format: wgpu::VertexFormat::Float32x3, offset: 24 },
        wgpu::VertexAttribute { shader_location: 3, format: wgpu::VertexFormat::Float32x3, offset: 36 },
        wgpu::VertexAttribute { shader_location: 4, format: wgpu::VertexFormat::Float32x3, offset: 48 },
        wgpu::VertexAttribute { shader_location: 5, format: wgpu::VertexFormat::Float32,   offset: 60 },
        wgpu::VertexAttribute { shader_location: 6, format: wgpu::VertexFormat::Float32,   offset: 64 },
        wgpu::VertexAttribute { shader_location: 7, format: wgpu::VertexFormat::Float32,   offset: 68 },
        wgpu::VertexAttribute { shader_location: 8, format: wgpu::VertexFormat::Uint32x4, offset: 72 },
        wgpu::VertexAttribute { shader_location: 9, format: wgpu::VertexFormat::Float32x4, offset: 88 },
    ];

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some("Skinning PBR Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main_skinned"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<crate::assets::mesh::Vertex>() as u64,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &vertex_attributes,
                },
                wgpu::VertexBufferLayout {
                    array_stride: 96,
                    step_mode:    wgpu::VertexStepMode::Instance,
                    attributes:   &[
                        wgpu::VertexAttribute { shader_location: 14, format: wgpu::VertexFormat::Float32x4, offset:  0 },
                        wgpu::VertexAttribute { shader_location: 15, format: wgpu::VertexFormat::Float32x4, offset: 16 },
                        wgpu::VertexAttribute { shader_location: 16, format: wgpu::VertexFormat::Float32x4, offset: 32 },
                        wgpu::VertexAttribute { shader_location: 17, format: wgpu::VertexFormat::Float32x4, offset: 48 },
                        wgpu::VertexAttribute { shader_location: 18, format: wgpu::VertexFormat::Float32x4, offset: 64 },
                        wgpu::VertexAttribute { shader_location: 19, format: wgpu::VertexFormat::Float32x4, offset: 80 },
                    ],
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module:      shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[
                // Deferred G-buffer outputs (same layout as the PBR pipeline).
                Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format:              wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare:       Some(wgpu::CompareFunction::LessEqual),
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState::default(),
        }),
        multisample:    wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache:          None,
    })
}

// create_deferred_bgl() — bind group layout for the deferred lighting pass (group 1).
// Reads the four G-buffer targets, the depth buffer, the separate sky colour
// target, plus a small uniform buffer (inverse VP + screen size) used to
// reconstruct world positions from depth.
pub fn create_deferred_bgl(device: &Device) -> wgpu::BindGroupLayout {
    let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled:   false,
        },
        count: None,
    };
    let depth_texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    };

    device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label:   Some("Deferred G-Buffer BGL"),
            entries: &[
                texture_entry(0), // t_gb_albedo
                texture_entry(1), // t_gb_normal
                texture_entry(2), // t_gb_material
                texture_entry(3), // t_gb_extras
                depth_texture_entry(4), // t_depth
                texture_entry(5), // t_sky (sky colour target)
                sampler_entry(6), // s_gb (shared filtering sampler)
                uniform_entry(7), // DeferredUniforms (inv_view_proj + screen size)
                texture_entry(8), // t_ao (GTAO ambient-occlusion mask)
                froxel_texture_entry(9), // t_froxel (3D volumetric grid)
                froxel_sampler_entry(10), // s_froxel
                froxel_texture_entry(11), // t_voxel (RTGI 3D voxel grid)
                froxel_sampler_entry(12), // s_voxel
            ],
        },
    )
}

// Froxel 3D texture + trilinear sampler entries for the deferred pass.
fn froxel_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled:   false,
        },
        count: None,
    }
}
fn froxel_sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

// create_gtao_bgl() — bind group layout for the GTAO compute pass.
// group(0): uniform params, depth texture, normal G-buffer, storage output.
pub fn create_gtao_bgl(device: &Device) -> wgpu::BindGroupLayout {
    let uniform_entry = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    };
    let depth_entry = wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled:   false,
        },
        count: None,
    };
    let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled:   false,
        },
        count: None,
    };
    let storage_entry = wgpu::BindGroupLayoutEntry {
        binding: 3,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access:          wgpu::StorageTextureAccess::WriteOnly,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            view_dimension:  wgpu::TextureViewDimension::D2,
        },
        count: None,
    };
    device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label:   Some("GTAO BGL"),
            entries: &[uniform_entry, depth_entry, tex_entry(2), storage_entry],
        },
    )
}

// create_gtao_pipeline() — compute pipeline that solves screen-space AO.
pub fn create_gtao_pipeline(
    device: &Device,
    bgl:    &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label:              Some("GTAO Layout"),
            bind_group_layouts: &[Some(bgl)],
            immediate_size:     0,
        },
    );
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:              Some("GTAO Pipeline"),
        layout:             Some(&layout),
        module:             shader,
        entry_point:        Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache:              None,
    })
}

// create_froxel_bgl() — bind group layout for the froxel volumetric-injection
// compute pass. group(0): uniform params, 3D storage texture.
pub fn create_froxel_bgl(device: &Device) -> wgpu::BindGroupLayout {
    let uniform_entry = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    };
    let storage_entry = wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access:          wgpu::StorageTextureAccess::WriteOnly,
            format:          wgpu::TextureFormat::Rgba16Float,
            view_dimension:  wgpu::TextureViewDimension::D3,
        },
        count: None,
    };
    device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label:   Some("Froxel BGL"),
            entries: &[uniform_entry, storage_entry],
        },
    )
}

// create_froxel_pipeline() — compute pipeline that injects sun scattering
// into the 3D froxel grid.
pub fn create_froxel_pipeline(
    device: &Device,
    bgl:    &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label:              Some("Froxel Layout"),
            bind_group_layouts: &[Some(bgl)],
            immediate_size:     0,
        },
    );
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:              Some("Froxel Pipeline"),
        layout:             Some(&layout),
        module:             shader,
        entry_point:        Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache:              None,
    })
}

// create_voxel_gi_bgl() — bind group layout for the RTGI voxel-injection
// compute pass (voxel_gi.wgsl). group(0): uniform params, depth, previous-frame
// lit scene colour, shared sampler, and the write-only voxel grid (level 0).
pub fn create_voxel_gi_bgl(device: &Device) -> wgpu::BindGroupLayout {
    let uniform_entry = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    };
    let depth_entry = wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled:   false,
        },
        count: None,
    };
    let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled:   false,
        },
        count: None,
    };
    let sampler_entry = wgpu::BindGroupLayoutEntry {
        binding: 3,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    let storage_entry = wgpu::BindGroupLayoutEntry {
        binding: 4,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access:          wgpu::StorageTextureAccess::WriteOnly,
            format:          wgpu::TextureFormat::Rgba16Float,
            view_dimension:  wgpu::TextureViewDimension::D3,
        },
        count: None,
    };
    device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label:   Some("Voxel GI BGL"),
            entries: &[uniform_entry, depth_entry, tex_entry(2), sampler_entry, storage_entry],
        },
    )
}

// create_voxel_gi_pipeline() — compute pipeline that voxelizes the visible
// scene into the camera-aligned grid (entry point cs_inject).
pub fn create_voxel_gi_pipeline(
    device: &Device,
    bgl:    &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label:              Some("Voxel GI Layout"),
            bind_group_layouts: &[Some(bgl)],
            immediate_size:     0,
        },
    );
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:              Some("Voxel GI Pipeline"),
        layout:             Some(&layout),
        module:             shader,
        entry_point:        Some("cs_inject"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache:              None,
    })
}

// create_voxel_gi_mip_bgl() — bind group layout for the voxel mip-pyramid
// compute pass (voxel_gi_mip.wgsl). group(0): uniform params, sampled source
// level (L-1), write-only destination level (L).
pub fn create_voxel_gi_mip_bgl(device: &Device) -> wgpu::BindGroupLayout {
    let uniform_entry = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    };
    let tex_entry = wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled:   false,
        },
        count: None,
    };
    let storage_entry = wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access:          wgpu::StorageTextureAccess::WriteOnly,
            format:          wgpu::TextureFormat::Rgba16Float,
            view_dimension:  wgpu::TextureViewDimension::D3,
        },
        count: None,
    };
    device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label:   Some("Voxel GI Mip BGL"),
            entries: &[uniform_entry, tex_entry, storage_entry],
        },
    )
}

// create_voxel_gi_mip_pipeline() — compute pipeline that builds one mip level
// of the voxel pyramid (entry point cs_mip).
pub fn create_voxel_gi_mip_pipeline(
    device: &Device,
    bgl:    &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label:              Some("Voxel GI Mip Layout"),
            bind_group_layouts: &[Some(bgl)],
            immediate_size:     0,
        },
    );
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:              Some("Voxel GI Mip Pipeline"),
        layout:             Some(&layout),
        module:             shader,
        entry_point:        Some("cs_mip"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache:              None,
    })
}

// create_decal_bgl() — bind group layout for the decal volume pass (group 1).
// group(1): decal uniforms, depth texture, linear sampler.
pub fn create_decal_bgl(device: &Device) -> wgpu::BindGroupLayout {
    let uniform_entry = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    };
    let depth_entry = wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled:   false,
        },
        count: None,
    };
    let sampler_entry = wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label:   Some("Decal BGL"),
            entries: &[uniform_entry, depth_entry, sampler_entry],
        },
    )
}

// create_decal_pipeline() — renders decal box volumes into gb_albedo with
// SrcAlpha/OneMinusSrcAlpha blending (only the albedo target is attached).
pub fn create_decal_pipeline(
    device: &Device,
    global_bgl:  &wgpu::BindGroupLayout,
    decal_bgl:   &wgpu::BindGroupLayout,
    shader:      &wgpu::ShaderModule,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label:              Some("Decal Layout"),
            bind_group_layouts: &[Some(global_bgl), Some(decal_bgl)],
            immediate_size:     0,
        },
    );
    let blend = wgpu::BlendState {
        color:     wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation:  wgpu::BlendOperation::Add,
        },
        alpha:     wgpu::BlendComponent::REPLACE,
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:              Some("Decal Pipeline"),
        layout:             Some(&layout),
        vertex:             wgpu::VertexState {
            module:     shader,
            entry_point: Some("vs_decal"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers:    &[],
        },
        fragment:           Some(wgpu::FragmentState {
            module:     shader,
            entry_point: Some("fs_decal"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets:    &[Some(wgpu::ColorTargetState {
                format:     target_format,
                blend:      Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive:          wgpu::PrimitiveState {
            topology:         wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face:       wgpu::FrontFace::Ccw,
            cull_mode:        None,
            unclipped_depth:  false,
            polygon_mode:     wgpu::PolygonMode::Fill,
            conservative:     false,
        },
        depth_stencil:      None,
        multisample:        wgpu::MultisampleState::default(),
        multiview_mask:     None,
        cache:              None,
    })
}

// create_deferred_pipeline() — fullscreen deferred lighting pass.
// Group 0 reuses the global bind group (camera/IBL/lights/shadows), group 1
// holds the G-buffer bindings. Outputs scene colour (surf_fmt) + world normals
// (Rgba16Float, consumed by SSR).
pub fn create_deferred_pipeline(
    device:       &Device,
    surf_fmt:     wgpu::TextureFormat,
    global_bgl:   &wgpu::BindGroupLayout,
    deferred_bgl: &wgpu::BindGroupLayout,
    shader:       &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label:              Some("Deferred Lighting Layout"),
            bind_group_layouts: &[Some(global_bgl), Some(deferred_bgl)],
            immediate_size:     0,
        },
    );

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some("Deferred Lighting Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module:                shader,
            entry_point:           Some("vs_deferred"),
            compilation_options:   wgpu::PipelineCompilationOptions::default(),
            buffers:               &[], // fullscreen triangle from vertex_index
        },
        fragment: Some(wgpu::FragmentState {
            module:                shader,
            entry_point:           Some("fs_deferred"),
            compilation_options:   wgpu::PipelineCompilationOptions::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format:     surf_fmt,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        primitive:    wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample:  wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache:        None,
    })
}

// create_shadow_pipeline() — vertex-only pipeline for depth-only shadow pass.
// Uses the same vertex layout but no fragment shader and front-face culling
// (the standard trick to eliminate shadow acne without large depth bias).
#[allow(dead_code)]
pub fn create_shadow_pipeline(
    device:     &Device,
    shader:     &wgpu::ShaderModule,
    shadow_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {

    let layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label:              Some("Shadow Layout"),
            bind_group_layouts: &[Some(shadow_bgl)],
            immediate_size:     0,
        },
    );

    // Shadow pass only needs position — we stride the full Vertex but only
    // read the first attribute. This avoids maintaining a separate smaller buffer.
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some("Shadow Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module:      shader,
            entry_point: Some("vs_shadow"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<crate::assets::mesh::Vertex>() as u64,
                step_mode:    wgpu::VertexStepMode::Vertex,
                attributes:   &[wgpu::VertexAttribute {
                    shader_location: 0,
                    format:          wgpu::VertexFormat::Float32x3,
                    offset:          0,
                }],
            }],
        },
        // No fragment shader — GPU fills depth automatically.
        fragment: None,
        primitive: wgpu::PrimitiveState {
            // Front-face culling eliminates shadow acne:
            // back faces are rendered into the shadow map, so the comparison
            // during the main pass happens on front faces which are slightly farther.
            cull_mode: Some(wgpu::Face::Front),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format:              wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare:       Some(wgpu::CompareFunction::LessEqual),
            stencil:             wgpu::StencilState::default(),
            // Slope-scaled depth bias reduces shadow acne on angled surfaces.
            bias: wgpu::DepthBiasState {
                constant:    2,
                slope_scale: 2.0,
                clamp:       0.0,
            },
        }),
        multisample:    wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache:          None,
    })
}