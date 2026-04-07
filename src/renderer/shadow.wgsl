// shadow.wgsl — vertex-only shader for depth rendering.
// Transforms vertices into the light's clip space.
// The GPU automatically writes depth — no fragment shader needed.

struct ShadowUniforms {
    // All cascade matrices packed together.
    // We use push constants or a cascade index to select which one.
    // For simplicity: upload per cascade, same uniform slot.
    light_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> shadow: ShadowUniforms;

// Only position — we ignore everything else.
// This matches offset 0 in our vertex buffer layout.
struct VertIn {
    @location(0) position: vec3<f32>,
}

@vertex
fn vs_shadow(in: VertIn) -> @builtin(position) vec4<f32> {
    // Transform to light clip space.
    // The GPU writes this as depth automatically.
    return shadow.light_matrix * vec4<f32>(in.position, 1.0);
}

