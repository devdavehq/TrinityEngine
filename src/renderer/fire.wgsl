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

    // ── Multiple scrolling noise layers for volumetric depth ─────────────
    // Layer A: primary flame rising motion.
    let uv_a = in.flame_uv + vec2<f32>(t * speed * 0.3, -t * speed);
    // Layer B: secondary detail at different scale/speed.
    let uv_b = in.flame_uv * 1.5 + vec2<f32>(-t * speed * 0.2, -t * speed * 0.7);
    // Layer C: fine internal turbulence (small scale, fast scroll).
    let uv_c = in.flame_uv * 3.0 + vec2<f32>(t * speed * 0.15, -t * speed * 1.3);
    // Layer D: slow large-scale undulation for overall flame sway.
    let uv_d = in.flame_uv * 0.5 + vec2<f32>(-t * speed * 0.1, -t * speed * 0.4);

    let n1 = fbm(uv_a);
    let n2 = fbm(uv_b);
    let n3 = fbm(uv_c);
    let n4 = fbm(uv_d);

    // Combine: primary + secondary + fine turbulence + sway.
    let flame_mask = (n1 * 0.4 + n2 * 0.3 + n3 * 0.15 + n4 * 0.15);

    // ── Volumetric depth effect ──────────────────────────────────────────
    // Simulate internal glow by creating a bright core surrounded by
    // dimmer outer flames — gives the illusion of 3D volume.
    let core_mask = smoothstep(0.3, 0.6, flame_mask);
    let outer_mask = smoothstep(0.0, 0.4, flame_mask);
    let volume = core_mask * 0.6 + outer_mask * 0.4;

    // ── Height-based flame shape ─────────────────────────────────────────
    // Wider at base, tapers to point. Multiple taper stages for realism.
    let taper_base = smoothstep(0.9, 0.0, h);         // Main taper
    let taper_inner = smoothstep(0.6, 0.05, h) * 0.3; // Inner bright core taper
    let taper = taper_base + taper_inner;

    let flame = volume * taper;

    // ── Colour gradient (4-stop for realism) ─────────────────────────────
    // Base (h=0.0): white-hot bright core
    // Lower (h=0.15): yellow-orange
    // Mid (h=0.4): deep orange
    // Tip (h=0.8): dark red → smoke
    let base = fire.base_color.rgb;
    let tip  = fire.tip_color.rgb;
    let white_hot = vec3<f32>(1.0, 0.95, 0.85);
    let yellow = mix(base, vec3<f32>(1.0, 0.8, 0.1), 0.3);

    var color: vec3<f32>;
    if h < 0.15 {
        color = mix(white_hot, yellow, h / 0.15);
    } else if h < 0.4 {
        color = mix(yellow, base, (h - 0.15) / 0.25);
    } else if h < 0.7 {
        color = mix(base, tip, (h - 0.4) / 0.3);
    } else {
        // Fade to dark smoke at the very tips.
        let smoke = tip * 0.3;
        color = mix(tip, smoke, (h - 0.7) / 0.3);
    }

    // ── Core brightness boost ────────────────────────────────────────────
    // The inner core of the flame is much brighter than the outer shell.
    let core_boost = mix(0.8, 2.0, core_mask);
    color *= core_boost;

    // ── Brightness variation from flame density ──────────────────────────
    let density_var = mix(1.2, 0.7, flame_mask);
    color *= density_var;

    // ── Emissive output ──────────────────────────────────────────────────
    let intensity = fire.base_color.a;
    color *= intensity;

    // ── Alpha ────────────────────────────────────────────────────────────
    // High at base core, soft fade at tips and edges.
    let alpha_core = flame * smoothstep(0.0, 0.1, 1.0 - h) * 0.95;
    let alpha_edge = outer_mask * 0.3;
    let alpha = max(alpha_core, alpha_edge * taper);
    let final_alpha = clamp(alpha, 0.0, 0.95);

    return vec4<f32>(color, final_alpha);
}
