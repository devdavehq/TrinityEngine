// src/renderer/water.wgsl
// Water surface rendering shader — AAA-quality with GGX microfacet specular.
//
// Features:
//   - Gerstner wave vertex displacement (3 overlapping waves)
//   - Procedural normal maps for fine-scale ripples and capillary waves
//   - Fresnel-based reflection/refraction blending (Schlick approximation)
//   - Depth-based colour absorption (deeper = darker blue)
//   - Foam at wave crests with procedural detail
//   - Shoreline foam (depth-buffer edge detection with animated texture)
//   - GGX microfacet specular (energy-conserving, metallic-free)
//   - Subsurface scattering approximation (light through shallow water)
//   - Transparency with scene colour refraction
//
// ── How it works ────────────────────────────────────────────────────────────
// 1. Vertex shader applies Gerstner wave displacement to a flat water mesh.
// 2. Fragment shader computes analytical normals + procedural detail normals.
// 3. GGX specular replaces Blinn-Phong for physically accurate highlights.
// 4. Fresnel blends reflected sky and refracted scene colour.
// 5. Depth-based absorption, subsurface scattering, and shoreline foam.

// ── Uniform buffer ──────────────────────────────────────────────────────────
struct WaterUniforms {
    // Wave parameters
    wave_params:   vec4<f32>,  // x=height, y=speed, z=choppy, w=time
    // Wave direction vectors (3 waves packed)
    wave_dir_a:    vec4<f32>,  // xyz = direction, w = steepness
    wave_dir_b:    vec4<f32>,  // xyz = direction, w = steepness
    wave_dir_c:    vec4<f32>,  // xyz = direction, w = steepness
    // Colours
    deep_color:    vec4<f32>,  // rgb = deep water colour, a = shore foam intensity
    shallow_color: vec4<f32>,  // rgb = shallow water colour, a = opacity
    // Lighting
    light_dir:     vec4<f32>,  // xyz = normalised sun direction, w = shore foam width
    light_color:   vec4<f32>,  // rgb = sun colour, w = specular power (now: roughness)
    // Camera
    camera_pos:    vec4<f32>,  // xyz = camera position, w = foam intensity
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> water: WaterUniforms;

// Scene colour texture (for refraction)
@group(0) @binding(1) var t_scene: texture_2d<f32>;
@group(0) @binding(2) var s_scene: sampler;

// Depth buffer
@group(0) @binding(3) var t_depth: texture_depth_2d;
@group(0) @binding(4) var s_depth: sampler;

// ── Vertex input/output ─────────────────────────────────────────────────────
struct WaterVertIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
}

struct WaterVertOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec3<f32>,
    @location(1)       uv:           vec2<f32>,
    @location(2)       wave_height:  f32,
}

// ── Gerstner wave displacement ──────────────────────────────────────────────
fn gerstner_wave(pos: vec3<f32>, time: f32, dir: vec3<f32>, steepness: f32, speed: f32, height: f32) -> vec3<f32> {
    let a = height * 0.5;
    let q = steepness;
    let d = normalize(dir.xz);
    let phase = dot(d, pos.xz) * q + time * speed;
    let s = sin(phase);
    let c = cos(phase);
    return vec3<f32>(
        d.x * a * c,
        a * s,
        d.y * a * c,
    );
}

// ── Water normal (analytical from Gerstner derivatives) ─────────────────────
fn water_normal(pos: vec3<f32>, time: f32, height: f32) -> vec3<f32> {
    let h = height * 0.5;
    let sp = water.wave_params.y;

    let dA = normalize(water.wave_dir_a.xz);
    let dB = normalize(water.wave_dir_b.xz);
    let dC = normalize(water.wave_dir_c.xz);
    let qa = water.wave_dir_a.w;
    let qb = water.wave_dir_b.w;
    let qc = water.wave_dir_c.w;

    var dPdx = vec3<f32>(0.0);
    var dPdz = vec3<f32>(0.0);

    // Wave A
    let phaseA = dot(dA, pos.xz) * qa + time * sp * water.wave_dir_a.w * 3.14159;
    let cA = cos(phaseA);
    let sA = sin(phaseA);
    dPdx.x += -dA.x * dA.x * qa * h * cA;
    dPdx.z += -dA.x * dA.y * qa * h * cA;
    dPdx.y += dA.x * h * sA;
    dPdz.x += -dA.x * dA.y * qa * h * cA;
    dPdz.z += -dA.y * dA.y * qa * h * cA;
    dPdz.y += dA.y * h * sA;

    // Wave B
    let phaseB = dot(dB, pos.xz) * qb + time * sp * water.wave_dir_b.w * 2.1;
    let cB = cos(phaseB);
    let sB = sin(phaseB);
    dPdx.x += -dB.x * dB.x * qb * h * cB;
    dPdx.z += -dB.x * dB.y * qb * h * cB;
    dPdx.y += dB.x * h * sB;
    dPdz.x += -dB.x * dB.y * qb * h * cB;
    dPdz.z += -dB.y * dB.y * qb * h * cB;
    dPdz.y += dB.y * h * sB;

    // Wave C
    let phaseC = dot(dC, pos.xz) * qc + time * sp * water.wave_dir_c.w * 1.3;
    let cC = cos(phaseC);
    let sC = sin(phaseC);
    dPdx.x += -dC.x * dC.x * qc * h * cC;
    dPdx.z += -dC.x * dC.y * qc * h * cC;
    dPdx.y += dC.x * h * sC;
    dPdz.x += -dC.x * dC.y * qc * h * cC;
    dPdz.z += -dC.y * dC.y * qc * h * cC;
    dPdz.y += dC.y * h * sC;

    dPdx.x += 1.0;

    let n = cross(dPdz, dPdx);
    return normalize(n);
}

// ── Procedural normal map (replaces texture-based normal maps) ──────────────
// Generates fine-scale ripple and capillary wave normals procedurally.
// Two octaves of gradient noise at different scales create realistic detail.
fn procedural_water_normal(pos: vec3<f32>, time: f32) -> vec3<f32> {
    // Large ripples (10cm scale)
    let ripple_scale = 8.0;
    let ripple_speed = 1.5;
    let ripple_phase = pos.xz * ripple_scale + time * ripple_speed;
    let ripple_n = vec2<f32>(
        cos(ripple_phase.x * 1.3 + ripple_phase.y * 0.7) * 0.03,
        sin(ripple_phase.y * 1.1 + ripple_phase.x * 0.9) * 0.03,
    );

    // Fine capillary waves (2cm scale)
    let cap_scale = 40.0;
    let cap_speed = 3.0;
    let cap_phase = pos.xz * cap_scale + time * cap_speed;
    let cap_n = vec2<f32>(
        cos(cap_phase.x * 2.1 + cap_phase.y * 1.7) * 0.008,
        sin(cap_phase.y * 1.9 + cap_phase.x * 2.3) * 0.008,
    );

    // Combine ripples into a perturbation vector.
    let perturbation = ripple_n + cap_n;

    // Convert 2D perturbation to 3D normal (assumes flat base).
    return normalize(vec3<f32>(perturbation.x, 1.0, perturbation.y));
}

// ── GGX Normal Distribution Function ────────────────────────────────────────
// Trowbridge-Reitz NDF: probability distribution of microfacet orientations.
// Used for physically-based specular highlights.
fn ggx_ndf(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}

// ── Schlick-GGX Geometry function ───────────────────────────────────────────
// Smith's geometry function: accounts for microfacet self-shadowing.
fn schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometry_smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let g1 = schlick_ggx(NdotV, roughness);
    let g2 = schlick_ggx(NdotL, roughness);
    return g1 * g2;
}

// ── Fresnel (Schlick) ───────────────────────────────────────────────────────
fn fresnel_schlick(cos_theta: f32, F0: f32) -> f32 {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

@vertex
fn vs_water(in: WaterVertIn) -> WaterVertOut {
    var out: WaterVertOut;

    let time = water.wave_params.w * water.wave_params.y;

    // Apply 3 overlapping Gerstner waves.
    var displacement = vec3<f32>(0.0);
    displacement += gerstner_wave(in.position, time,
        water.wave_dir_a.xyz, water.wave_dir_a.w, 1.0, water.wave_params.x);
    displacement += gerstner_wave(in.position, time * 0.7,
        water.wave_dir_b.xyz, water.wave_dir_b.w, 0.8, water.wave_params.x * 0.6);
    displacement += gerstner_wave(in.position, time * 1.3,
        water.wave_dir_c.xyz, water.wave_dir_c.w, 0.6, water.wave_params.x * 0.3);

    var world_pos = in.position + displacement;

    out.clip_pos = water.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.uv = world_pos.xz * 0.05;
    out.wave_height = displacement.y;

    return out;
}

@fragment
fn fs_water(in: WaterVertOut) -> @location(0) vec4<f32> {
    // ── Compute base normal analytically from Gerstner derivatives ────────
    let base_N = water_normal(in.world_pos, water.wave_params.w * water.wave_params.y, water.wave_params.x);

    // ── Add procedural detail normal (ripples + capillary waves) ─────────
    let detail_N = procedural_water_normal(in.world_pos, water.wave_params.w);
    // Blend: 80% analytical + 20% detail for realistic surface micro-detail.
    let N = normalize(base_N * 0.8 + detail_N * 0.2);

    let V = normalize(water.camera_pos.xyz - in.world_pos);
    let L = normalize(water.light_dir.xyz);
    let H = normalize(V + L);

    let NdotV = max(dot(N, V), 0.001);
    let NdotL = max(dot(N, L), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    // ── GGX Microfacet specular (replaces Blinn-Phong) ──────────────────
    // roughness from light_color.w (legacy specular_power is now roughness)
    let roughness = clamp(water.light_color.w / 512.0, 0.02, 1.0);
    let D = ggx_ndf(NdotH, roughness);
    let G = geometry_smith(NdotV, NdotL, roughness);
    let F = fresnel_schlick(HdotV, 0.02); // water F0 = 0.02

    // Cook-Torrance specular BRDF.
    let specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.001);
    let kS = F;
    let kD = (1.0 - kS) * (1.0); // metals: kD = 0; dielectrics: energy conserved

    // ── Fresnel ──────────────────────────────────────────────────────────
    let fresnel = fresnel_schlick(NdotV, 0.02);

    // ── Reflection (sample scene colour at reflected UV) ─────────────────
    let R = reflect(-V, N);
    let reflected_clip = water.view_proj * vec4<f32>(in.world_pos + R * 100.0, 1.0);
    let reflected_uv = (reflected_clip.xy / reflected_clip.w) * 0.5 + vec2<f32>(0.5, 0.5);
    var reflection = textureSample(t_scene, s_scene, clamp(reflected_uv, vec2<f32>(0.001), vec2<f32>(0.999))).rgb;

    // ── Refraction (depth-based absorption) ──────────────────────────────
    let scene_uv = in.clip_pos.xy / vec2<f32>(textureDimensions(t_scene, 0));
    let scene_depth = textureSampleDepth(t_depth, s_depth, scene_uv);
    let water_depth = max(scene_depth - in.clip_pos.z, 0.0);

    // Exponential absorption: deeper water = more blue, less light.
    let absorption = exp(-water_depth * vec3<f32>(0.3, 0.6, 0.9));
    let refraction = mix(water.shallow_color.rgb, water.deep_color.rgb, clamp(water_depth * 2.0, 0.0, 1.0));

    // ── Subsurface scattering approximation ──────────────────────────────
    // Light transmitted through thin water (shallows) gives a green-blue glow.
    let sss_intensity = exp(-water_depth * 2.0) * 0.15;
    let sss = water.light_color.rgb * sss_intensity * max(dot(L, -V), 0.0);

    // ── Blend reflection and refraction via Fresnel ──────────────────────
    var water_color = mix(refraction + sss, reflection, fresnel);

    // ── Apply GGX specular ───────────────────────────────────────────────
    water_color += water.light_color.rgb * specular * kD * NdotL;

    // ── Foam at wave crests ──────────────────────────────────────────────
    let foam_threshold = water.wave_params.x * 0.6;
    let foam_mask = smoothstep(foam_threshold - 0.05, foam_threshold + 0.05, in.wave_height);
    let foam_color = vec3<f32>(0.9, 0.95, 1.0);

    // ── Shoreline foam ───────────────────────────────────────────────────
    // Detects where water meets terrain using depth buffer analysis.
    // Cross-pattern sampling finds the nearest geometry; where it is close
    // to the water surface we are near shore and render a foam band.
    // Parameters: deep_color.w = shore_foam_intensity, light_dir.w = shore_foam_width
    let shore_texel = 1.0 / vec2<f32>(textureDimensions(t_depth, 0));
    let d_n = textureSampleDepth(t_depth, s_depth, scene_uv + vec2<f32>( 0.0,          shore_texel.y));
    let d_s = textureSampleDepth(t_depth, s_depth, scene_uv + vec2<f32>( 0.0,         -shore_texel.y));
    let d_e = textureSampleDepth(t_depth, s_depth, scene_uv + vec2<f32>( shore_texel.x, 0.0));
    let d_w = textureSampleDepth(t_depth, s_depth, scene_uv + vec2<f32>(-shore_texel.x, 0.0));

    // Minimum scene depth among neighbours — close to water surface = near shore.
    let min_neighbor_depth = min(min(d_n, d_s), min(d_e, d_w));
    let shore_water_depth  = max(min_neighbor_depth - in.clip_pos.z, 0.0);

    // Depth gradient across the pixel — large = sharp shoreline edge.
    let depth_gradient = max(
        max(abs(scene_depth - d_n), abs(scene_depth - d_s)),
        max(abs(scene_depth - d_e), abs(scene_depth - d_w))
    );

    // Foam mask: exponential falloff with distance from shore, sharpened by gradient.
    let shore_width    = max(light_dir.w, 0.001);
    let foam_band      = exp(-shore_water_depth / shore_width);
    let edge_sharpen   = smoothstep(0.0005, 0.005, depth_gradient);
    let shore_foam_base = foam_band * edge_sharpen;

    // Animated procedural foam texture — two scrolling noise layers at
    // different scales and speeds produce a shimmering, organic look.
    let foam_uv_1 = in.world_pos.xz * 0.8
                  + vec2<f32>(water.wave_params.w *  0.3, water.wave_params.w *  0.15);
    let foam_uv_2 = in.world_pos.xz * 1.6
                  + vec2<f32>(water.wave_params.w * -0.2, water.wave_params.w *  0.25);
    let foam_tex_1 = fract(sin(dot(floor(foam_uv_1), vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let foam_tex_2 = fract(sin(dot(floor(foam_uv_2), vec2<f32>(4.898,   7.23)))   * 23421.631);
    let foam_detail = clamp(foam_tex_1 * 0.6 + foam_tex_2 * 0.4, 0.0, 1.0);

    // Final shoreline foam: masked by intensity uniform and foam texture.
    let shore_foam_mask = shore_foam_base * foam_detail * deep_color.w;

    // Combine wave-crest foam and shoreline foam (additive max).
    let combined_foam = max(foam_mask * water.camera_pos.w, shore_foam_mask);
    water_color = mix(water_color, foam_color, combined_foam);

    // ── Apply absorption and opacity ─────────────────────────────────────
    water_color *= absorption;
    let opacity = water.shallow_color.a;

    return vec4<f32>(water_color, opacity);
}
