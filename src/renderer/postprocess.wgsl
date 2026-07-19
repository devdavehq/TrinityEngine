struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let xy = p[vi];
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv = xy * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

// ── Tone mapping + colour grading uniforms ────────────────────────────────
// Loaded via group(1) binding 0 on the tonemap pipeline.
// Contains all the parameters for the ACES tone mapper and colour adjustments.
struct TonemapUniforms {
    exposure:    f32,  // Exposure compensation — multiplies HDR luminance.
    temperature: f32,  // Colour temperature shift: -1 = cool/blue, +1 = warm/orange.
    saturation:  f32,  // Global saturation multiplier: 0 = greyscale, 1 = normal, >1 = vivid.
    contrast:    f32,  // Contrast adjustment: 0 = flat, 1 = normal, >1 = punchy.
    vibrance:    f32,  // Selective saturation boost on less-saturated colours.
    grain:       f32,  // Film grain intensity (0 = off).
    _pad0:       f32,
    _pad1:       f32,
}
@group(1) @binding(0) var<uniform> tonemap: TonemapUniforms;

@fragment
fn fs_copy(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t_src, s_src, in.uv);
}

@fragment
fn fs_bloom_extract(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(t_src, s_src, in.uv).rgb;
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Lower threshold catches emissive objects (HDR colors > 1.0 from
    // material_extras.emissive_strength). Higher ceiling keeps the ramp
    // smooth so bright emissive surfaces bloom naturally.
    let k = smoothstep(0.4, 1.2, luma);
    return vec4<f32>(c * k, 1.0);
}

@fragment
fn fs_blur_h(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_src, 0));
    let px = vec2<f32>(1.0 / tex_size.x, 0.0);
    var c = textureSample(t_src, s_src, in.uv).rgb * 0.227027;
    c += textureSample(t_src, s_src, in.uv + px * 1.384615).rgb * 0.316216;
    c += textureSample(t_src, s_src, in.uv - px * 1.384615).rgb * 0.316216;
    c += textureSample(t_src, s_src, in.uv + px * 3.230769).rgb * 0.070270;
    c += textureSample(t_src, s_src, in.uv - px * 3.230769).rgb * 0.070270;
    return vec4<f32>(c, 1.0);
}

@fragment
fn fs_blur_v(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_src, 0));
    let px = vec2<f32>(0.0, 1.0 / tex_size.y);
    var c = textureSample(t_src, s_src, in.uv).rgb * 0.227027;
    c += textureSample(t_src, s_src, in.uv + px * 1.384615).rgb * 0.316216;
    c += textureSample(t_src, s_src, in.uv - px * 1.384615).rgb * 0.316216;
    c += textureSample(t_src, s_src, in.uv + px * 3.230769).rgb * 0.070270;
    c += textureSample(t_src, s_src, in.uv - px * 3.230769).rgb * 0.070270;
    return vec4<f32>(c, 1.0);
}

@group(1) @binding(0) var t_bloom: texture_2d<f32>;
@group(1) @binding(1) var s_bloom: sampler;

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let base = textureSample(t_src, s_src, in.uv).rgb;
    let bloom = textureSample(t_bloom, s_bloom, in.uv).rgb;
    return vec4<f32>(base + bloom, 1.0);
}

// ── Tone mapping + colour grading pass ────────────────────────────────────
// Applied as the FINAL post-process step, after bloom composite.
// Reads the full-resolution scene (with bloom) and outputs to the swapchain.
//
// WHY ACES INSTEAD OF REINHARD?
// Reinhard maps all HDR values into [0,1] but desaturates bright colours.
// ACES (Academy Color Encoding System) is the film industry standard —
// it preserves colour saturation in bright highlights and gives a cinematic
// roll-off that looks natural. RDR2, Cyberpunk, and virtually every AAA
// game uses ACES or a variant.

// ACES fitted curve (Stephen Hill's approximation).
// Input: linear HDR value. Output: LDR value in [0,1].
fn aces_film(x: vec3<f32>) -> vec3<f32> {
    // Curve coefficients — these map the ACES response curve into a
    // simple rational function that runs entirely on the GPU.
    let a = x * (x * 2.51 + vec3<f32>(0.03));
    let b = x * (x * 2.43 + vec3<f32>(0.59)) + vec3<f32>(0.14);
    return clamp(a / b, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Colour temperature shift.
// Shifts the white balance by adding a blue↔orange tint.
// Negative values cool the image (blue hour / overcast).
// Positive values warm it (golden hour / interior lighting).
fn apply_temperature(color: vec3<f32>, temp: f32) -> vec3<f32> {
    // Map temperature from [-1,1] to a visible tint.
    let t = clamp(temp, -1.0, 1.0) * 0.15;
    return color + vec3<f32>(t, 0.0, -t);
}

// Saturation adjustment.
// Desaturates toward luminance (greyscale) or boosts away from it.
fn apply_saturation(color: vec3<f32>, sat: f32) -> vec3<f32> {
    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(vec3<f32>(lum), color, sat);
}

// Contrast adjustment.
// Scales around midpoint (0.18 middle grey) for perceptually correct contrast.
fn apply_contrast(color: vec3<f32>, con: f32) -> vec3<f32> {
    let midpoint = vec3<f32>(0.18);
    return (color - midpoint) * con + midpoint;
}

// Vibrance — selective saturation.
// Boosts less-saturated colours more than already-saturated ones,
// preventing skin tones from going neon while making muted colours pop.
fn apply_vibrance(color: vec3<f32>, vib: f32) -> vec3<f32> {
    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let luma_color = vec3<f32>(lum);
    let sat_color = color - luma_color;
    // Measure how saturated the pixel already is.
    let current_sat = length(sat_color);
    // Pixels below 50% saturation get full vibrance boost;
    // already-saturated pixels get less (prevents blowout).
    let boost = vib * (1.0 - smoothstep(0.0, 0.5, current_sat));
    return color + sat_color * boost;
}

// Simple film grain from UV + time — adds subtle texture and masks
// colour banding in smooth gradients.
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_tonemap(in: VsOut) -> @location(0) vec4<f32> {
    var color = textureSample(t_src, s_src, in.uv).rgb;

    // 1) Exposure — scale HDR brightness before tone mapping.
    color *= exp(tonemap.exposure);

    // 2) Temperature — subtle colour temperature shift.
    color = apply_temperature(color, tonemap.temperature);

    // 3) ACES tone mapping — HDR → LDR with cinematic roll-off.
    color = aces_film(color);

    // 4) Contrast — applied after tone mapping in LDR space.
    let contrast_val = mix(1.0, 1.0 + tonemap.contrast, step(0.001, abs(tonemap.contrast)));
    color = apply_contrast(color, contrast_val);

    // 5) Saturation — global saturation.
    let sat_val = 1.0 + tonemap.saturation;
    color = apply_saturation(color, sat_val);

    // 6) Vibrance — selective saturation for already-muted colours.
    color = apply_vibrance(color, tonemap.vibrance * 0.5);

    // 7) Film grain — subtle dithering to mask banding.
    if (tonemap.grain > 0.001) {
        let grain_noise = hash21(in.uv * 1000.0) * 2.0 - 1.0;
        color += grain_noise * tonemap.grain * 0.03;
    }

    // 8) Gamma correction — linear → sRGB for display.
    color = pow(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Screen-Space Reflections (SSR)
// ═══════════════════════════════════════════════════════════════════════════════
// Hierarchical-Z ray marching in screen space.
// For each pixel:
//   1. Reconstruct view-space position from depth buffer.
//   2. Compute reflection vector R = reflect(-V, N).
//   3. March along R in screen space, comparing depths at each step.
//   4. On hit, sample scene colour at the hit UV and blend via Fresnel.
//   5. Fade reflections near screen edges to prevent hard borders.
//
// ── Bindings ────────────────────────────────────────────────────────────────
// Group(0) reuse: t_src = scene colour, s_src = sampler.
// Group(1) SSR:
@group(1) @binding(0) var t_normals:   texture_2d<f32>;
@group(1) @binding(1) var t_depth:     texture_depth_2d;
@group(1) @binding(2) var s_ssr:       sampler;

struct SsrUniforms {
    inv_view_proj: mat4x4<f32>,
    view_proj:     mat4x4<f32>,
    max_steps:     u32,
    max_distance:  f32,
    thickness:     f32,
    intensity:     f32,
    screen_size:   vec2<f32>,
    _pad0:         vec2<f32>,
}
@group(1) @binding(3) var<uniform> ssr: SsrUniforms;

// Reconstruct view-space position from UV + depth.
fn ssr_reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view_h = ssr.inv_view_proj * ndc;
    return view_h.xyz / view_h.w;
}

// Project view-space position to screen UV via the VP matrix.
fn ssr_project_to_uv(pos: vec3<f32>) -> vec2<f32> {
    let clip = ssr.view_proj * vec4<f32>(pos, 1.0);
    return (clip.xy / clip.w) * 0.5 + vec2<f32>(0.5, 0.5);
}

@fragment
fn fs_ssr(in: VsOut) -> @location(0) vec4<f32> {
    let scene_color = textureSample(t_src, s_src, in.uv).rgb;
    let normal_encoded = textureSample(t_normals, s_ssr, in.uv).rgb;
    let depth = textureSampleDepth(t_depth, s_ssr, in.uv);

    // Decode normals from [0,1] → [-1,1].
    let N = normalize(normal_encoded * 2.0 - vec3<f32>(1.0));

    // Sky pixels (depth ≈ 1.0) skip SSR.
    if (depth >= 0.9999) {
        return vec4<f32>(scene_color, 1.0);
    }

    // Reconstruct view-space position and view direction.
    let view_pos = ssr_reconstruct_view_pos(in.uv, depth);
    let V = normalize(-view_pos);

    // Reflection vector in view space.
    let R = reflect(-V, N);

    // Early-out if reflection goes behind camera.
    if (R.z >= 0.0) {
        return vec4<f32>(scene_color, 1.0);
    }

    // ── Hi-Z screen-space ray marching ────────────────────────────────────
    let step_size = ssr.max_distance / f32(ssr.max_steps);
    var ray_pos = view_pos;
    var hit_uv = vec2<f32>(0.0);
    var hit = false;

    for (var i = 0u; i < ssr.max_steps; i = i + 1u) {
        ray_pos = ray_pos + R * step_size;

        // Project ray position to screen UV via VP matrix.
        let ray_uv = ssr_project_to_uv(ray_pos);

        // Out of screen → stop.
        if (ray_uv.x < 0.0 || ray_uv.x > 1.0 || ray_uv.y < 0.0 || ray_uv.y > 1.0) {
            break;
        }

        // Sample depth buffer at the ray's screen position.
        let sample_depth = textureSampleDepth(t_depth, s_ssr, ray_uv);

        // Reconstruct the surface's view-space position at this UV.
        let sample_view = ssr_reconstruct_view_pos(ray_uv, sample_depth);

        // Ray z is more negative (further from camera) as it marches forward.
        // If ray is behind the surface (ray_z > surface_z) but within thickness → hit.
        let depth_diff = ray_pos.z - sample_view.z;

        if (depth_diff > 0.0 && depth_diff < ssr.thickness) {
            hit_uv = ray_uv;
            hit = true;
            break;
        }
    }

    if (!hit) {
        return vec4<f32>(scene_color, 1.0);
    }

    // ── Blend reflection ──────────────────────────────────────────────────
    let reflection = textureSample(t_src, s_src, hit_uv).rgb;

    // Fresnel: more reflection at grazing angles.
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0);
    let reflection_strength = mix(0.04, 1.0, fresnel) * ssr.intensity;

    // Edge fade: fade reflections near screen borders.
    let edge_fade = smoothstep(0.0, 0.1, hit_uv.x)
                  * smoothstep(1.0, 0.9, hit_uv.x)
                  * smoothstep(0.0, 0.1, hit_uv.y)
                  * smoothstep(1.0, 0.9, hit_uv.y);

    let final_color = mix(scene_color, reflection, reflection_strength * edge_fade);
    return vec4<f32>(final_color, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Temporal Anti-Aliasing (TAA)
// ═══════════════════════════════════════════════════════════════════════════════
// Exponential moving average with motion-based history rejection.
// Reprojects the previous frame using per-pixel velocity, clamps the
// history sample to a 3×3 colour bounding box of the current frame
// to prevent ghosting, and blends based on pixel motion.
//
// ── Bindings ────────────────────────────────────────────────────────────────
// Group(0) reuse: t_src = current scene colour, s_src = sampler.
// Group(1) TAA:
struct TaaUniforms {
    jitter_offset: vec2<f32>,
    blend_factor: f32,
    enable_taa: f32,
};
@group(1) @binding(0) var t_history:  texture_2d<f32>;
@group(1) @binding(1) var t_velocity: texture_2d<f32>;
@group(1) @binding(2) var s_hist:     sampler;
@group(1) @binding(3) var<uniform> taa: TaaUniforms;

@fragment
fn fs_taa(in: VsOut) -> @location(0) vec4<f32> {
    if (taa.enable_taa < 0.5) {
        return textureSample(t_src, s_src, in.uv);
    }

    let current = textureSample(t_src, s_src, in.uv);
    let velocity = textureSample(t_velocity, s_hist, in.uv).rg;

    // Reproject UV to where this pixel was in the previous frame.
    let history_uv = in.uv - velocity;

    // Sample history at the reprojected position.
    let history = textureSample(t_history, s_hist, history_uv);

    // ── 3×3 colour bounding box (current frame neighbourhood) ────────────
    let tex_size = vec2<f32>(textureDimensions(t_src, 0));
    let px = vec2<f32>(1.0 / tex_size.x, 1.0 / tex_size.y);

    var min_color = current.rgb;
    var max_color = current.rgb;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor = textureSample(t_src, s_src,
                in.uv + vec2<f32>(f32(dx), f32(dy)) * px).rgb;
            min_color = min(min_color, neighbor);
            max_color = max(max_color, neighbor);
        }
    }

    // Clamp history to the bounding box to reject ghosts / disoccluded regions.
    let clamped_history = clamp(history.rgb, min_color, max_color);

    // Blend: more current-frame contribution when moving (anti-ghost),
    // more history when stationary (maximum temporal smoothing).
    let motion_length = length(velocity);
    let blend = select(taa.blend_factor * 0.5, taa.blend_factor, motion_length > 0.001);

    let result = mix(clamped_history, current.rgb, blend);
    return vec4<f32>(result, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Per-Object Motion Blur
// ═══════════════════════════════════════════════════════════════════════════════
// Reads per-pixel velocity and scatters along the motion vector with
// linear weighting. Very distant objects (depth ≈ far plane) skip blur.
//
// ── Bindings ────────────────────────────────────────────────────────────────
// Group(0) reuse: t_src = scene colour, s_src = sampler.
// Group(1) Motion Blur:
struct MotionBlurUniforms {
    blur_strength: f32,
    max_samples: f32,
    _pad: vec2<f32>,
};
@group(1) @binding(0) var t_mb_velocity: texture_2d<f32>;
@group(1) @binding(1) var t_mb_depth:    texture_depth_2d;
@group(1) @binding(2) var s_mb:          sampler;
@group(1) @binding(3) var<uniform> mb:   MotionBlurUniforms;

@fragment
fn fs_motion_blur(in: VsOut) -> @location(0) vec4<f32> {
    let velocity = textureSample(t_mb_velocity, s_mb, in.uv).rg;
    let depth = textureSampleDepth(t_mb_depth, s_mb, in.uv);

    // Skip blur for very distant objects (sky / far plane).
    if (depth > 0.99) {
        return textureSample(t_src, s_src, in.uv);
    }

    let scaled_velocity = velocity * mb.blur_strength;
    let speed = length(scaled_velocity);

    // Skip if motion is negligible.
    if (speed < 0.0001) {
        return textureSample(t_src, s_src, in.uv);
    }

    let samples = i32(mb.max_samples);
    var color = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0; i < samples; i++) {
        let t = f32(i) / f32(samples) - 0.5;
        let offset = scaled_velocity * t;
        let sample_color = textureSample(t_src, s_src, in.uv + offset).rgb;
        // Linear weight: center samples weighted more.
        let weight = 1.0 - abs(t) * 2.0;
        color += sample_color * weight;
        total_weight += weight;
    }

    return vec4<f32>(color / total_weight, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Depth of Field (Circle of Confusion)
// ═══════════════════════════════════════════════════════════════════════════════
// Computes a per-pixel circle of confusion from the depth buffer, then
// samples a 13-tap Poisson disc kernel weighted by CoC radius.
// Pixels closer than the focal plane (foreground) blur toward background.
//
// ── Bindings ────────────────────────────────────────────────────────────────
// Group(0) reuse: t_src = scene colour, s_src = sampler.
// Group(1) DOF:
struct DofUniforms {
    focus_distance: f32,
    dof_strength: f32,
    aperture: f32,
    _pad: f32,
};
@group(1) @binding(0) var t_dof_depth: texture_depth_2d;
@group(1) @binding(1) var s_dof:       sampler;
@group(1) @binding(2) var<uniform> dof: DofUniforms;

// 13-tap Poisson disc sample offsets (unit circle, pre-rotated).
const DOF_TAPS: array<vec2<f32>, 13> = array<vec2<f32>, 13>(
    vec2<f32>( 0.0,      0.0),
    vec2<f32>( 0.2380,   0.4250),
    vec2<f32>(-0.6230,   0.2140),
    vec2<f32>( 0.3570,  -0.5940),
    vec2<f32>(-0.1730,  -0.2870),
    vec2<f32>( 0.6820,   0.1470),
    vec2<f32>(-0.3920,   0.6480),
    vec2<f32>( 0.1140,  -0.7830),
    vec2<f32>(-0.7150,  -0.5210),
    vec2<f32>( 0.4930,   0.6730),
    vec2<f32>(-0.1260,   0.1980),
    vec2<f32>( 0.8160,  -0.3420),
    vec2<f32>(-0.5470,  -0.0760),
);

@fragment
fn fs_dof(in: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSampleDepth(t_dof_depth, s_dof, in.uv);

    // Circle of confusion.
    let coc = abs(depth - dof.focus_distance) * dof.aperture
            / max(dof.focus_distance, 0.001);
    let clamped_coc = clamp(coc, 0.0, dof.dof_strength);

    // Skip if in focus (CoC < 0.5 pixels).
    if (clamped_coc < 0.5) {
        return textureSample(t_src, s_src, in.uv);
    }

    let tex_size = vec2<f32>(textureDimensions(t_src, 0));
    let radius = clamped_coc / max(tex_size.x, tex_size.y);

    var color = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0; i < 13; i++) {
        let offset = DOF_TAPS[i] * radius;
        let sample_color = textureSample(t_src, s_src, in.uv + offset).rgb;
        // Weight by inverse distance from centre (centre tap has weight 1).
        let dist = length(DOF_TAPS[i]);
        let weight = 1.0 / (1.0 + dist * 2.0);
        color += sample_color * weight;
        total_weight += weight;
    }

    return vec4<f32>(color / total_weight, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bilateral Blur (edge-preserving blur for SSR denoising)
// ═══════════════════════════════════════════════════════════════════════════════
// Separable Gaussian blur weighted by depth and normal similarity.
// Smooths reflections within surfaces but preserves edges at depth/normal
// discontinuities. Essential for clean SSR without excessive ray march steps.
//
// Bindings:
// Group(0): t_src = SSR composite colour, s_src = sampler.
// Group(1):
@group(1) @binding(0) var t_bilateral_depth:   texture_depth_2d;
@group(1) @binding(1) var t_bilateral_normals: texture_2d<f32>;
@group(1) @binding(2) var s_bilateral:         sampler;
@group(1) @binding(3) var<uniform> bilateral:  BilateralUniforms;

struct BilateralUniforms {
    blur_radius:  f32,   // blur radius in pixels (4-8 typical)
    depth_weight: f32,   // depth similarity weight (higher = more edge preservation)
    norm_weight:  f32,   // normal similarity weight
    _pad:         f32,
};

@fragment
fn fs_bilateral_h(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_src, 0));
    let px = vec2<f32>(1.0 / tex_size.x, 0.0);

    let center_color = textureSample(t_src, s_src, in.uv).rgb;
    let center_depth = textureSampleDepth(t_bilateral_depth, s_bilateral, in.uv);
    let center_norm  = textureSample(t_bilateral_normals, s_bilateral, in.uv).rgb;

    var total_color = center_color;
    var total_weight = 1.0;

    let radius = i32(bilateral.blur_radius);

    for (var i = -12; i <= 12; i++) {
        if (i == 0 || abs(i) > radius) { continue; }

        let offset = vec2<f32>(f32(i), 0.0) * px;
        let sample_uv = in.uv + offset;

        let sample_color = textureSample(t_src, s_src, sample_uv).rgb;
        let sample_depth = textureSampleDepth(t_bilateral_depth, s_bilateral, sample_uv);
        let sample_norm  = textureSample(t_bilateral_normals, s_bilateral, sample_uv).rgb;

        // Spatial weight: Gaussian falloff with distance.
        let spatial_w = exp(-0.5 * f32(i * i) / (bilateral.blur_radius * bilateral.blur_radius * 0.25));

        // Depth weight: reject samples at very different depths (edge preservation).
        let depth_diff = abs(center_depth - sample_depth) * bilateral.depth_weight;
        let depth_w = exp(-depth_diff * depth_diff);

        // Normal weight: reject samples with different surface orientation.
        let norm_diff = 1.0 - max(dot(center_norm, sample_norm), 0.0);
        let norm_w = exp(-norm_diff * norm_diff * bilateral.norm_weight);

        let w = spatial_w * depth_w * norm_w;
        total_color += sample_color * w;
        total_weight += w;
    }

    return vec4<f32>(total_color / total_weight, 1.0);
}

@fragment
fn fs_bilateral_v(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_src, 0));
    let px = vec2<f32>(0.0, 1.0 / tex_size.y);

    let center_color = textureSample(t_src, s_src, in.uv).rgb;
    let center_depth = textureSampleDepth(t_bilateral_depth, s_bilateral, in.uv);
    let center_norm  = textureSample(t_bilateral_normals, s_bilateral, in.uv).rgb;

    var total_color = center_color;
    var total_weight = 1.0;

    let radius = i32(bilateral.blur_radius);

    for (var i = -12; i <= 12; i++) {
        if (i == 0 || abs(i) > radius) { continue; }

        let offset = vec2<f32>(0.0, f32(i)) * px;
        let sample_uv = in.uv + offset;

        let sample_color = textureSample(t_src, s_src, sample_uv).rgb;
        let sample_depth = textureSampleDepth(t_bilateral_depth, s_bilateral, sample_uv);
        let sample_norm  = textureSample(t_bilateral_normals, s_bilateral, sample_uv).rgb;

        let spatial_w = exp(-0.5 * f32(i * i) / (bilateral.blur_radius * bilateral.blur_radius * 0.25));
        let depth_diff = abs(center_depth - sample_depth) * bilateral.depth_weight;
        let depth_w = exp(-depth_diff * depth_diff);
        let norm_diff = 1.0 - max(dot(center_norm, sample_norm), 0.0);
        let norm_w = exp(-norm_diff * norm_diff * norm_diff * bilateral.norm_weight);

        let w = spatial_w * depth_w * norm_w;
        total_color += sample_color * w;
        total_weight += w;
    }

    return vec4<f32>(total_color / total_weight, 1.0);
}

// ── Screen-space god rays ────────────────────────────────────────────────
// Radial blur from the sun's projected screen position.
// Depth-masked: rays are blocked by geometry (simulates shafts through
// trees, buildings, and terrain).
//
// Algorithm:
//   1. For each pixel, march toward the sun in screen space.
//   2. At each sample, check depth — if sample is behind geometry, it's
//      occluded and doesn't contribute to rays.
//   3. Accumulate scene colour weighted by density and decay.
//   4. Add the result to the scene (additive blend).

struct GodRayUniforms {
    sun_uv:      vec2<f32>, // Sun position in screen UV space [0,1].
    intensity:   f32,       // Overall ray brightness (0 = off, 1 = strong).
    decay:       f32,       // How fast rays fade from sun outward (0.9 = subtle, 0.98 = long).
    density:     f32,       // Sample spacing (0.5 = dense, 2.0 = sparse).
    weight:      f32,       // Base weight per sample (0.01 = subtle, 0.1 = strong).
    num_samples: f32,       // Number of ray-march samples (8-64, stored as float).
    _pad:        f32,
}

@group(1) @binding(0) var t_godray_depth: texture_depth_2d;
@group(1) @binding(1) var s_godray:       sampler;
@group(1) @binding(2) var<uniform> godray: GodRayUniforms;

@fragment
fn fs_god_rays(in: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSample(t_src, s_src, in.uv).rgb;

    // Direction from this pixel toward the sun in UV space.
    let delta = godray.sun_uv - in.uv;
    let dist_to_sun = length(delta);
    let dir = delta / max(dist_to_sun, 0.001);

    // Screen-space dimensions for depth comparison.
    let screen_size = vec2<f32>(textureDimensions(t_src, 0));
    let px = 1.0 / screen_size;

    // Current pixel depth (linear, from depth buffer).
    let center_depth = textureSampleDepth(t_godray_depth, s_godray, in.uv);

    // Ray-march accumulation.
    var illumination_decay = 1.0;
    var ray_color = vec3<f32>(0.0, 0.0, 0.0);
    let samples = i32(godray.num_samples);

    // Weight per sample is the base weight scaled by number of samples.
    let sample_weight = godray.weight / godray.num_samples;

    for (var i = 0; i < samples; i++) {
        // Step along the ray toward the sun.
        let sample_uv = in.uv + dir * f32(i) * godray.density * px;

        // Clamp to screen bounds.
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            break;
        }

        // Depth test: only accumulate if the sample is in front of geometry
        // (i.e., the ray is not occluded at this point).
        let sample_depth = textureSampleDepth(t_godray_depth, s_godray, sample_uv);

        // If the sample depth is farther than center_depth, this sample is
        // "behind" the geometry from the camera's perspective — occluded.
        // We allow a small bias to avoid self-occlusion at edges.
        let occlusion = step(center_depth * 0.998, sample_depth);

        // Sample scene colour at this point (this is what glows).
        let sample_color = textureSample(t_src, s_src, sample_uv).rgb;

        // Accumulate with occlusion mask and distance decay.
        ray_color += sample_color * illumination_decay * sample_weight * occlusion;

        // Decay as we move away from the sun (farther = dimmer).
        illumination_decay *= godray.decay;
    }

    // Add god rays to the original scene.
    let result = scene + ray_color * godray.intensity;
    return vec4<f32>(result, 1.0);
}
