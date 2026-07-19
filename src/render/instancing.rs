// src/render/instancing.rs
// ──────────────────────────────────────────────────────────────────────────────
// GPU Instancing System
//
// WHY IT EXISTS:
//   Currently the renderer copies vertex data into a staging buffer per entity
//   per frame. That means 100 cubes = 100 separate vertex buffer uploads + 100
//   draw calls. That's catastrophic for performance.
//
//   GPU instancing lets us draw 100 cubes with ONE draw call. The GPU reads
//   the mesh data once, then applies per-instance transforms from an instance
//   buffer. This is how every professional engine handles repeated geometry:
//   foliage, rocks, grass, buildings, particles, debris, etc.
//
// HOW IT WORKS:
//   1. Group entities by (mesh_id, material_id) — same mesh + same material.
//   2. For each group, build an instance buffer containing per-instance data:
//      transform matrix, color tint, metallic, roughness, ao.
//   3. Issue ONE draw call per group with instance_count = number of entities.
//   4. The vertex shader reads instance data from @builtin(instance_index).
//
// LOW-END PC SUPPORT:
//   Instancing is MORE important on low-end, not less. Fewer draw calls = less
//   CPU overhead = better framerates on integrated GPUs. We always use
//   instancing; the quality tier only affects WHAT we render, not HOW we batch.
//
// PERFORMANCE:
//   CPU cost: O(n) to build instance buffers (one matrix copy per entity).
//   GPU cost: O(n) to process instances, but mesh loaded from VRAM once.
//   Net win: massive. 500 identical entities goes from ~500 draw calls to 1.
// ──────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;

// ── Per-instance data uploaded to GPU ─────────────────────────────────────────
// This struct MUST match the @group(0) instance input in the shader.
// repr(C) ensures predictable memory layout for bytemuck.
//
// Size: 64 (model) + 16 (color+metallic) + 12 (roughness+ao+pad) = 92 bytes
// Padded to 96 for alignment.

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    /// 4x4 model (world) transform matrix. Row-major, matching glam::Mat4.
    pub model: [[f32; 4]; 4], // 64 bytes
    /// RGB color tint + metallic in alpha channel.
    pub color_metallic: [f32; 4], // 16 bytes
    /// x = roughness, y = ao, zw = padding for future use.
    pub roughness_ao_pad: [f32; 4], // 16 bytes
}

impl InstanceData {
    /// Total bytes for one instance. Used for buffer sizing.
    pub const STRIDE: usize = std::mem::size_of::<Self>(); // 96 bytes

    pub fn new(
        position: [f32; 3],
        rotation_pitch: f32,
        rotation_yaw: f32,
        rotation_roll: f32,
        scale: [f32; 3],
        color: [f32; 3],
        metallic: f32,
        roughness: f32,
        ao: f32,
    ) -> Self {
        // Build transform matrix from TRS (translation, rotation, scale).
        let t = glam::Mat4::from_translation(glam::Vec3::new(position[0], position[1], position[2]));
        let ry = glam::Mat4::from_rotation_y(rotation_yaw);
        let rp = glam::Mat4::from_rotation_x(rotation_pitch);
        let rr = glam::Mat4::from_rotation_z(rotation_roll);
        let s = glam::Mat4::from_scale(glam::Vec3::new(scale[0], scale[1], scale[2]));
        let model = (t * ry * rp * rr * s).to_cols_array_2d();

        Self {
            model,
            color_metallic: [color[0], color[1], color[2], metallic],
            roughness_ao_pad: [roughness, ao, 0.0, 0.0],
        }
    }
}

// ── Instance Batch ────────────────────────────────────────────────────────────
// All entities sharing the same (mesh_id, material_id) go into one batch.
// The batch owns the GPU buffer for instance data.

pub struct InstanceBatch {
    /// Which mesh to draw.
    pub mesh_id: u32,
    /// Which material to use.
    pub material_id: u32,
    /// CPU-side instance data. Rebuilt each frame (or when dirty).
    pub instances: Vec<InstanceData>,
    /// GPU buffer. Created/resized when instance count changes.
    pub buffer: Option<wgpu::Buffer>,
    /// True if instances changed this frame and buffer needs upload.
    pub dirty: bool,
}

impl InstanceBatch {
    pub fn new(mesh_id: u32, material_id: u32) -> Self {
        Self {
            mesh_id,
            material_id,
            instances: Vec::new(),
            buffer: None,
            dirty: true,
        }
    }
}

// ── InstancingManager ─────────────────────────────────────────────────────────
// Owns all instance batches. Each frame:
//   1. clear_and_rebuild() — re-group all entities into batches
//   2. upload_buffers() — push dirty batches to GPU
//   3. render_batch() — issue one draw call per batch

pub struct InstancingManager {
    /// Key: (mesh_id, material_id) -> batch index
    batch_index: HashMap<(u32, u32), usize>,
    /// All batches, indexed by batch_index.
    batches: Vec<InstanceBatch>,
    /// Maximum instances we've seen in any single batch (for buffer sizing).
    max_instances_per_batch: usize,
}

impl InstancingManager {
    pub fn new() -> Self {
        Self {
            batch_index: HashMap::new(),
            batches: Vec::new(),
            max_instances_per_batch: 256,
        }
    }

    /// Begin a new frame. Clears all instance data but keeps GPU buffers.
    pub fn begin_frame(&mut self) {
        self.batch_index.clear();
        for batch in &mut self.batches {
            batch.instances.clear();
            batch.dirty = true;
        }
    }

    /// Add an entity to the appropriate batch.
    pub fn add_instance(
        &mut self,
        mesh_id: u32,
        material_id: u32,
        instance: InstanceData,
    ) {
        let key = (mesh_id, material_id);
        let idx = *self.batch_index.entry(key).or_insert_with(|| {
            let idx = self.batches.len();
            self.batches.push(InstanceBatch::new(mesh_id, material_id));
            idx
        });
        self.batches[idx].instances.push(instance);
        self.batches[idx].dirty = true;
    }

    /// Upload dirty instance buffers to GPU. Call after begin_frame + add_instances.
    pub fn upload_buffers(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for batch in &mut self.batches {
            if !batch.dirty || batch.instances.is_empty() {
                continue;
            }

            let needed_size =
                (batch.instances.len() * InstanceData::STRIDE) as wgpu::BufferAddress;

            // Reuse buffer if large enough, otherwise recreate.
            let too_small = batch
                .buffer
                .as_ref()
                .map_or(true, |b| b.size() < needed_size);

            if too_small {
                // Allocate with 2x headroom to avoid frequent reallocation.
                let alloc_size = (needed_size * 2).max(256 * InstanceData::STRIDE as u64);
                batch.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Instance Buffer"),
                    size: alloc_size,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                // Update max for diagnostics.
                self.max_instances_per_batch = self
                    .max_instances_per_batch
                    .max(batch.instances.len());
            }

            if let Some(buffer) = &batch.buffer {
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&batch.instances));
            }

            batch.dirty = false;
        }
    }

    /// Get all batches for rendering.
    pub fn batches(&self) -> &[InstanceBatch] {
        &self.batches
    }

    /// Get total instance count across all batches (for profiler).
    pub fn total_instances(&self) -> usize {
        self.batches.iter().map(|b| b.instances.len()).sum()
    }

    /// Get batch count (for profiler / debug).
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Reset everything (for project switch).
    pub fn clear(&mut self) {
        self.batch_index.clear();
        self.batches.clear();
    }
}

// ── Shader integration ────────────────────────────────────────────────────────
// The WGSL shader needs to read instance data. Here's what to add to the
// vertex shader:
//
// ```wgsl
// // Instance data input
// @group(2) @binding(0) var<storage> instances: array<InstanceData>;
//
// struct InstanceData {
//     model: mat4x4<f32>,
//     color_metallic: vec4<f32>,
//     roughness_ao_pad: vec4<f32>,
// };
//
// @vertex
// fn vs_main(in: VertIn) -> VertOut {
//     let instance = instances[instance_index];
//     var out: VertOut;
//     out.clip_pos = uniforms.view_proj * instance.model * vec4<f32>(in.position, 1.0);
//     out.world_pos = (instance.model * vec4<f32>(in.position, 1.0)).xyz;
//     out.normal = (instance.model * vec4<f32>(in.normal, 0.0)).xyz;
//     out.color = in.color * instance.color_metallic.rgb;
//     out.metallic = instance.color_metallic.a;
//     out.roughness = instance.roughness_ao_pad.x;
//     out.ao = instance.roughness_ao_pad.y;
//     return out;
// }
// ```
//
// Or using vertex buffer step mode (simpler, works on all GPUs):
//
// ```wgsl
// @vertex
// fn vs_main(
//     @location(0) position: vec3<f32>,
//     @location(1) normal: vec3<f32>,
//     // ... other mesh attributes ...
//     @location(6) i_model_0: vec4<f32>,  // instance column 0
//     @location(7) i_model_1: vec4<f32>,  // instance column 1
//     @location(8) i_model_2: vec4<f32>,  // instance column 2
//     @location(9) i_model_3: vec4<f32>,  // instance column 3
//     @location(10) i_color_metallic: vec4<f32>,
//     @location(11) i_roughness_ao: vec4<f32>,
// ) -> VertOut { ... }
// ```
