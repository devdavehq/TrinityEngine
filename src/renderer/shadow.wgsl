// shadow.wgsl — vertex-only shader for CSM depth rendering.
// Transforms instanced geometry into each cascade's light clip space.
// The GPU automatically writes depth — no fragment shader needed.

struct ShadowUniforms {
    light_matrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> shadow: ShadowUniforms;

// Note: position comes from vertex buffer slot 0. The instance model matrix
// comes from vertex buffer slot 1 (step mode = Instance), matching the
// InstanceData layout in instancing.rs (column-major 4x4).
struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) model_row0: vec4<f32>,
    @location(2) model_row1: vec4<f32>,
    @location(3) model_row2: vec4<f32>,
    @location(4) model_row3: vec4<f32>,
};

@vertex
fn vs_shadow(in: VertIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(
        in.model_row0,
        in.model_row1,
        in.model_row2,
        in.model_row3,
    );
    return shadow.light_matrix * model * vec4<f32>(in.position, 1.0);
}