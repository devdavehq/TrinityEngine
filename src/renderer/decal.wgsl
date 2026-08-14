// src/renderer/decal.wgsl
// Deferred decals — paint albedo onto the G-buffer after the geometry pass.
//
// Each decal is a unit cube placed at the decal's transform. The fragment
// shader reconstructs the scene world position from the depth buffer, converts
// it into decal-local space, and only stamps colour where the surface actually
// crosses the box. This is the classic "box volume" decal: the cube is a
// projector, not geometry — every pixel of the stored depth that falls inside
// the box gets decal albedo blended in with alpha blending.
//
// Writes to gb_albedo only (SrcAlpha/OneMinusSrcAlpha blend) so normals and
// material channels stay intact — bullet holes, warning stripes, dirt spills.

struct DecalUniforms {
    model:       mat4x4<f32>, // decal box world transform (position + rotation + scale)
    inv_model:   mat4x4<f32>, // inverse — world pos → decal-local [-0.5, 0.5]
    view_proj:   mat4x4<f32>,
    inv_view_proj: mat4x4<f32>, // depth buffer → world space
    params:      vec4<f32>,   // x = opacity, yzw = unused
}
@group(1) @binding(0) var<uniform> decal: DecalUniforms;
@group(1) @binding(1) var t_depth:  texture_depth_2d;
@group(1) @binding(2) var s_lin:    sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_decal(@builtin(vertex_index) vi: u32) -> VsOut {
    // Unit cube (12 triangles) corners.
    let corners = array<vec3<f32>, 24>(
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>( 0.5, -0.5, -0.5), vec3<f32>( 0.5,  0.5, -0.5),
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>( 0.5,  0.5, -0.5), vec3<f32>(-0.5,  0.5, -0.5),
        vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>(-0.5,  0.5,  0.5), vec3<f32>( 0.5,  0.5,  0.5),
        vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>( 0.5,  0.5,  0.5), vec3<f32>( 0.5, -0.5,  0.5),
        vec3<f32>(-0.5,  0.5, -0.5), vec3<f32>( 0.5,  0.5, -0.5), vec3<f32>( 0.5,  0.5,  0.5),
        vec3<f32>(-0.5,  0.5, -0.5), vec3<f32>( 0.5,  0.5,  0.5), vec3<f32>(-0.5,  0.5,  0.5),
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>( 0.5, -0.5,  0.5),
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>( 0.5, -0.5,  0.5), vec3<f32>( 0.5, -0.5, -0.5),
    );
    let p = corners[vi];
    let world = decal.model * vec4<f32>(p, 1.0);
    let clip = decal.view_proj * world;
    // Camera-facing UV (from the box's top/front) so the decal maps sensibly.
    let uv = p.xz + vec2<f32>(0.5);
    var out: VsOut;
    out.clip_pos = clip;
    out.uv = uv;
    return out;
}

@fragment
fn fs_decal(in: VsOut) -> @location(0) vec4<f32> {
    // Reconstruct the scene world position at this pixel from depth.
    let depth = textureSample(t_depth, s_lin, in.uv);
    if (depth >= 0.9998) { return vec4<f32>(0.0); }
    let ndc = vec3<f32>(in.uv * 2.0 - vec2<f32>(1.0), depth * 2.0 - 1.0);
    let world_h = decal.inv_view_proj * vec4<f32>(ndc, 1.0);
    let world = world_h.xyz / world_h.w;
    // world → decal-local space.
    let local = (decal.inv_model * vec4<f32>(world, 1.0)).xyz;
    if (any(abs(local) > vec3<f32>(0.5))) { return vec4<f32>(0.0); }

    // Fade the decal out near the box edges so it looks painted, not cut.
    let edge = 1.0 - smoothstep(0.42, 0.5, max(abs(local.x), abs(local.y)));
    // Colour comes from the GPU constant; texture variants can sample t_decal.
    let tint = vec4<f32>(0.9, 0.2, 0.2, 1.0); // placeholder bright red
    return vec4<f32>(tint.rgb, tint.a * edge * decal.params.x);
}