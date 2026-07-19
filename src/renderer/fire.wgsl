// src/renderer/fire.wgsl
// Procedural fire / flame rendering shader — no textures needed.
//
// Features:
//   - Animated flame shape from scrolling 3D FBM noise
//   - Heat-to-colour gradient: white-hot base → orange → red tips → dark smoke
//   - Emissive output that drives bloom
//   - Flickering wind displacement for organic movement
//   - Semi-transparent with alpha blending
//
// ── How it works ────────────────────────────────────────────────────────────
// 1. Vertex shader displaces a flat quad upward using flame shape noise.
// 2. Fragment shader samples 2D FBM noise at two scroll speeds to create
//    a flickering flame mask with sharp internal detail.
// 3. Height-based colour gradient: bright at base, dark at tips.
// 4. Alpha = flame mask × height fade for soft transparent edges.
// 5. Emissive output drives bloom post-processing.

struct FireUniforms {
    base_color:      vec4<f32>,  // rgb = base flame colour (white-hot), a = intensity
    tip_color:       vec4<f32>,  // rgb = flame tip colour (orange/red), a = unused
    params:          vec4<f32>,  // x=flame_speed, y=noise_scale, z=flicker_strength, w=flame_height
    wind_time:       vec4<f32>,  // x=elapsed, y=wind_x, z=wind_z, w=unused
    view_proj:       mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> fire: FireUniforms;

// ── Vertex input/output ─────────────────────────────────────────────────────
struct FireVertIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
}

struct FireVertOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       height:    f32,  // normalised height for colour gradient
    @location(2)       flame_uv:  vec2<f32>,  // UV for noise sampling
}

@vertex
fn vs_fire(in: FireVertIn) -> FireVertOut {
    var out: FireVertOut;

    let t = fire.wind_time.x;
    let flame_h = fire.params.w;

    // Flame UV: world XZ scaled by noise_scale.
    out.flame_uv = in.position.xz * fire.params.y;

    // Height ratio: 0 at base, 1 at top of flame.
    out.height = clamp(in.position.y / max(flame_h, 0.01), 0.0, 1.0);

    // Flickering wind displacement: sways the flame sideways.
    let wind_x = fire.wind_time.y;
    let wind_z = fire.wind_time.z;
    let flicker = fire.params.z;
    let sway = sin(t * 5.0 + in.position.y * 3.0) * flicker;
    let wind_offset = vec3<f32>(
        wind_x * out.height * 0.5 + sway * out.height,
        0.0,
        wind_z * out.height * 0.5,
    );
    var world_pos = in.position + wind_offset;

    out.clip_pos = fire.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    return out;
}

// ── Procedural noise ───────────────────────────────────────────────────────
fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q = q + dot(q, q + 45.32);
    return fract(q.x * q.y);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var value = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 5; i++) {
        value += amp * noise(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return value;
}

// ── Fragment shader ─────────────────────────────────────────────────────────
@fragment
fn fs_fire(in: FireVertOut) -> @location(0) vec4<f32> {
    let t = fire.wind_time.x;
    let speed = fire.params.x;
    let h = in.height;

    // ── Two scrolling noise layers for flickering flame shape ────────────
    // Layer A scrolls upward (flame rising).
    let uv_a = in.flame_uv + vec2<f32>(t * speed * 0.3, -t * speed);
    // Layer B scrolls upward at different speed for internal detail.
    let uv_b = in.flame_uv * 1.5 + vec2<f32>(-t * speed * 0.2, -t * speed * 0.7);

    let n1 = fbm(uv_a);
    let n2 = fbm(uv_b);

    // Combine layers: average creates organic flame texture.
    let flame_mask = (n1 + n2) * 0.5;

    // ── Height-based flame shape ─────────────────────────────────────────
    // Flames are wider at the base and taper to a point at the top.
    // This simulates the natural shape of a candle / campfire flame.
    let taper = smoothstep(0.8, 0.1, h);
    let flame = flame_mask * taper;

    // ── Colour gradient ──────────────────────────────────────────────────
    // Base (h=0): white-hot bright colour
    // Mid (h=0.3): orange
    // Top (h=0.8): dark red / smoke
    let base = fire.base_color.rgb;
    let tip  = fire.tip_color.rgb;

    // Three-stop gradient: base → mid → tip.
    let mid = mix(base, tip, 0.5);
    var color: vec3<f32>;
    if h < 0.3 {
        color = mix(base, mid, h / 0.3);
    } else {
        color = mix(mid, tip, (h - 0.3) / 0.7);
    }

    // ── Brightness variation ─────────────────────────────────────────────
    // Brighter at the core (low noise value = dense flame = bright).
    let brightness = mix(1.5, 0.8, flame_mask);
    color *= brightness;

    // ── Emissive output ──────────────────────────────────────────────────
    let intensity = fire.base_color.a;
    color *= intensity;

    // ── Alpha ────────────────────────────────────────────────────────────
    // Flame alpha: high at base, fades at tips.
    let alpha = flame * smoothstep(0.0, 0.15, 1.0 - h) * 0.9;
    let final_alpha = clamp(alpha, 0.0, 0.95);

    return vec4<f32>(color, final_alpha);
}
