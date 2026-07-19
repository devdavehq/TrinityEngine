// src/renderer/lava.wgsl
// Lava / magma surface rendering shader.
//
// Features:
//   - Animated emissive crack patterns (two scrolling noise layers)
//   - Dark rocky base with bright molten cracks
//   - Emissive glow that drives bloom
//   - Simple vertex displacement for heat shimmer
//   - No transparency — fully opaque
//
// ── How it works ────────────────────────────────────────────────────────────
// 1. Vertex shader applies gentle sin-wave displacement (heat shimmer).
// 2. Fragment shader generates a procedural crack pattern by scrolling two
//    noise layers in opposite directions and taking the absolute difference.
// 3. The crack pattern is thresholded to create sharp molten cracks.
// 4. Rock base colour blends into emissive colour at cracks.
// 5. Final emissive output drives bloom post-processing.

struct LavaUniforms {
    rock_color:      vec4<f32>,  // rgb = cooled rock, a = opacity
    emissive_color:  vec4<f32>,  // rgb = molten glow, a = intensity
    params:          vec4<f32>,  // x=flow_speed, y=crack_scale, z=crack_threshold, w=displacement_amp
    time:            vec4<f32>,  // x=elapsed, yzw=unused
    view_proj:       mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> lava: LavaUniforms;

// ── Vertex input/output ─────────────────────────────────────────────────────
struct LavaVertIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
}

struct LavaVertOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       uv:        vec2<f32>,
}

@vertex
fn vs_lava(in: LavaVertIn) -> LavaVertOut {
    var out: LavaVertOut;

    let t = lava.time.x;

    // Gentle heat shimmer displacement along normal.
    let shimmer = sin(in.position.x * 2.0 + t * 3.0)
                * cos(in.position.z * 1.5 + t * 2.3)
                * lava.params.w;
    var world_pos = in.position + in.normal * shimmer;

    out.clip_pos = lava.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    // Tile UVs: world XZ scaled by crack_scale.
    out.uv = world_pos.xz * lava.params.y;

    return out;
}

// ── Procedural noise (simple hash-based value noise) ────────────────────────
// Fast, deterministic — good enough for lava cracks.
fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q = q + dot(q, q + 45.32);
    return fract(q.x * q.y);
}

// Smooth value noise with bilinear interpolation.
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Smoothstep for interpolation.
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractional Brownian Motion — 4 octaves for detail.
fn fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var value = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 4; i++) {
        value += amp * noise(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return value;
}

@fragment
fn fs_lava(in: LavaVertOut) -> @location(0) vec4<f32> {
    let t = lava.time.x;
    let speed = lava.params.x;
    let threshold = lava.params.z;

    // ── Two scrolling noise layers in opposite directions ───────────────────
    // The absolute difference between two layers creates sharp crack lines.
    let uv_a = in.uv + vec2<f32>(t * speed, t * speed * 0.7);
    let uv_b = in.uv + vec2<f32>(-t * speed * 0.8, t * speed * 0.5);

    let n1 = fbm(uv_a);
    let n2 = fbm(uv_b);

    // Absolute difference creates a crack pattern where layers cross.
    let crack = abs(n1 - n2);

    // ── Threshold the crack pattern ─────────────────────────────────────────
    // Smoothstep creates a sharp transition from rock to molten crack.
    let crack_mask = smoothstep(threshold - 0.08, threshold + 0.08, crack);

    // ── Blend rock and emissive ─────────────────────────────────────────────
    let rock = lava.rock_color.rgb;
    let glow = lava.emissive_color.rgb;

    var color = mix(rock, glow, crack_mask);

    // ── Emissive output ─────────────────────────────────────────────────────
    // Multiply by intensity — values > 1.0 will trigger bloom.
    let intensity = lava.emissive_color.a;
    color *= intensity;

    // ── Subtle heat shimmer glow ────────────────────────────────────────────
    // Areas with high noise value glow slightly even without cracks.
    let heat = fbm(in.uv * 0.5 + vec2<f32>(t * 0.1));
    let heat_glow = smoothstep(0.55, 0.75, heat) * 0.3 * glow;
    color += heat_glow;

    return vec4<f32>(color, lava.rock_color.a);
}
