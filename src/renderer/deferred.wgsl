// src/renderer/deferred.wgsl
// Deferred lighting pass.
//
// The G-buffer pass (shader.wgsl fs_main) writes material properties into four
// MRT targets. This pass is a fullscreen triangle that reconstructs the world
// position from the depth buffer and resolves the full PBR lighting (sun +
// shadows, point/spot lights, IBL, SSAO approximation, emissive, voxel GI,
// snow accumulation, volumetric fog) into scene color + normals (for SSR).
//
// Sky pixels (depth == 1.0, no geometry) composite the sky pass output which
// was rendered separately into the sky_color texture.

// ── Bind Group 0: global (camera + IBL + lights + shadows) ────────────────
// Identical layout to shader.wgsl group(0) so the pipeline reuses the global
// bind group (global_bgl). Declared here as a separate module.
struct Uniforms {
    view_proj:   mat4x4<f32>,
    camera_pos:  vec3<f32>,
    _pad0:       f32,
    light_dir:   vec3<f32>,
    _pad1:       f32,
    light_color: vec3<f32>,
    _pad2:       f32,
    point_light_pos_range: vec4<f32>,
    point_light_color_intensity: vec4<f32>,
    post_params0: vec4<f32>, // x=bloom_enabled, y=bloom_strength, z=ssao_enabled, w=ssao_strength
    post_params1: vec4<f32>, // x=fog_enabled, y=fog_density, z=voxel_enabled, w=voxel_strength
    fog_color:    vec4<f32>, // rgb = dynamic fog color, w = elapsed time
    wind_dir_strength: vec4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct WeatherData {
    snow_coverage: f32,
    _pad: vec3<f32>,
}
@group(0) @binding(13) var<uniform> weather: WeatherData;

@group(0) @binding(1) var ibl_irradiance:         texture_2d<f32>;
@group(0) @binding(2) var ibl_irradiance_sampler:  sampler;
@group(0) @binding(3) var ibl_prefilter:           texture_2d<f32>;
@group(0) @binding(4) var ibl_prefilter_sampler:   sampler;
@group(0) @binding(5) var brdf_lut:                texture_2d<f32>;
@group(0) @binding(6) var brdf_lut_sampler:        sampler;

const MAX_LIGHTS: u32 = 16u;

struct LightData {
    position: vec3<f32>,
    _pos_pad: f32,
    color: vec3<f32>,
    _col_pad: f32,
    intensity: f32,
    range: f32,
    light_type: f32,
    spot_angle_cos: f32,
    shadow_index: i32,
    _pad: f32,
    _align_pad: vec2<f32>,
    direction: vec3<f32>,
    _dir_pad: f32,
};

struct LightUniforms {
    lights: array<LightData, 16>,
    light_count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};
@group(0) @binding(12) var<uniform> light_uniforms: LightUniforms;

struct ShadowData {
    light_matrices: array<mat4x4<f32>, 3>,
    cascade_dists:  vec4<f32>,
    shadow_bias:        f32,
    normal_offset_bias: f32,
    pcf_radius:         f32,
    shadow_enabled:     f32,
    shadow_map_size:    f32,
}
@group(0) @binding(7)  var<uniform>    shadow_data:     ShadowData;
@group(0) @binding(8)  var            t_shadow0:        texture_depth_2d;
@group(0) @binding(9)  var            t_shadow1:        texture_depth_2d;
@group(0) @binding(10) var            t_shadow2:        texture_depth_2d;
@group(0) @binding(11) var            s_shadow:         sampler_comparison;

// ── Baked light probes ──────────────────────────────────────────────────────
struct ProbeControl {
    count: u32,
    _pad: vec3<u32>,
}
@group(0) @binding(14) var<uniform> probe_control: ProbeControl;
// Probe layout: 10 × vec4 each = position/r + 9 SH coeffs (rgb).
@group(0) @binding(15) var<storage, read> probe_data: array<vec4<f32>>;

// ── Bind Group 1: deferred G-buffer bindings ───────────────────────────────
struct DeferredUniforms {
    inv_view_proj: mat4x4<f32>, // inverse of the (jittered) view-projection
    screen_size:   vec4<f32>,   // xy = render target size in pixels
    voxel_origin:  vec4<f32>,   // xyz = voxel-grid world origin, w = voxel size
    voxel_dims:    vec4<f32>,   // xyz = grid dimensions (voxels per axis)
}
@group(1) @binding(0) var t_gb_albedo:   texture_2d<f32>;
@group(1) @binding(1) var t_gb_normal:   texture_2d<f32>;
@group(1) @binding(2) var t_gb_material: texture_2d<f32>;
@group(1) @binding(3) var t_gb_extras:   texture_2d<f32>;
@group(1) @binding(4) var t_depth:       texture_depth_2d;
@group(1) @binding(5) var t_sky:         texture_2d<f32>;
@group(1) @binding(6) var s_gb:          sampler;
@group(1) @binding(7) var<uniform> deferred: DeferredUniforms;
@group(1) @binding(8) var t_ao:          texture_2d<f32>;
@group(1) @binding(9) var t_froxel:      texture_3d<f32>;
@group(1) @binding(10) var s_froxel:     sampler;
@group(1) @binding(11) var t_voxel:      texture_3d<f32>;
@group(1) @binding(12) var s_voxel:      sampler;

// Material extras are reconstructed from the gb_extras target.
struct MaterialExtras {
    subsurface:          f32,
    clearcoat:           f32,
    clearcoat_roughness: f32,
    anisotropy:          f32,
    emissive_strength:   f32,
    _pad: vec3<f32>,
};

// ── Fullscreen triangle ────────────────────────────────────────────────────
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Evaluate baked L2 SH irradiance for a surface normal direction.
// Reads the module-scope `probe_data` storage array. Layout per probe
// (base = index*10 + n): n=0 position.xyz+range, n=1..9 coeffs rgb.
fn eval_sh(base: u32, normal: vec3<f32>) -> vec3<f32> {
    let d = normalize(normal);
    var result: vec3<f32> = vec3<f32>(0.0);

    // Band 0.
    result += probe_data[base + 1u].xyz * 0.282095;
    // Band 1.
    result += probe_data[base + 2u].xyz * 0.488603 * d.y;
    result += probe_data[base + 3u].xyz * 0.488603 * d.z;
    result += probe_data[base + 4u].xyz * 0.488603 * d.x;
    // Band 2.
    result += probe_data[base + 5u].xyz * 1.092548 * d.x * d.y;
    result += probe_data[base + 6u].xyz * 1.092548 * d.y * d.z;
    result += probe_data[base + 7u].xyz * 0.315392 * (3.0 * d.z * d.z - 1.0);
    result += probe_data[base + 8u].xyz * 1.092548 * d.x * d.z;
    result += probe_data[base + 9u].xyz * 0.546274 * (d.x * d.x - d.y * d.y);
    return max(result, vec3<f32>(0.0));
}

@vertex
fn vs_deferred(@builtin(vertex_index) vi: u32) -> VsOut {
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

// ── Utility: equirectangular direction → UV ────────────────────────────────
fn dir_to_equirect(dir: vec3<f32>) -> vec2<f32> {
    let n = normalize(dir);
    let uv = vec2<f32>(
        atan2(n.z, n.x) / (2.0 * 3.14159265) + 0.5,
        asin(n.y) / 3.14159265 + 0.5,
    );
    return uv;
}

// ── PCF shadow sampling ────────────────────────────────────────────────────
fn sample_shadow_pcf(
    shadow_coord: vec3<f32>,
    cascade_idx: i32,
    bias: f32,
    pcf_radius: f32,
) -> f32 {
    let uv  = shadow_coord.xy * 0.5 + 0.5;
    let uv2 = vec2<f32>(uv.x, 1.0 - uv.y);
    let ref_depth = shadow_coord.z - bias;

    let offsets = array<vec2<f32>, 9>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 0.0, -1.0), vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  0.0), vec2<f32>( 0.0,  0.0), vec2<f32>( 1.0,  0.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 0.0,  1.0), vec2<f32>( 1.0,  1.0),
    );

    let texel_size = pcf_radius / shadow_data.shadow_map_size;

    var shadow_sum = 0.0;
    for (var i = 0; i < 9; i++) {
        let sample_uv = uv2 + offsets[i] * texel_size;
        let lit = select(
            textureSampleCompareLevel(t_shadow0, s_shadow, sample_uv, ref_depth),
            select(
                textureSampleCompareLevel(t_shadow1, s_shadow, sample_uv, ref_depth),
                textureSampleCompareLevel(t_shadow2, s_shadow, sample_uv, ref_depth),
                cascade_idx == 2,
            ),
            cascade_idx == 0,
        );
        shadow_sum += lit;
    }
    return shadow_sum / 9.0;
}

fn get_cascade_index(view_z: f32) -> i32 {
    if view_z < shadow_data.cascade_dists.x { return 0; }
    if view_z < shadow_data.cascade_dists.y { return 1; }
    return 2;
}

fn compute_shadow(world_pos: vec3<f32>, N: vec3<f32>, view_z: f32) -> f32 {
    if shadow_data.shadow_enabled < 0.5 { return 1.0; }
    let cascade = get_cascade_index(abs(view_z));
    let normal_bias_amount = shadow_data.normal_offset_bias
        * (1.0 - max(dot(N, uniforms.light_dir), 0.0));
    let biased_pos = world_pos + N * normal_bias_amount;
    let light_clip = shadow_data.light_matrices[cascade] * vec4<f32>(biased_pos, 1.0);
    let shadow_coord = light_clip.xyz / light_clip.w;
    if any(shadow_coord.xy < vec2<f32>(-0.99)) || any(shadow_coord.xy > vec2<f32>(0.99)) {
        return 1.0;
    }
    if shadow_coord.z < 0.0 || shadow_coord.z > 1.0 {
        return 1.0;
    }
    return sample_shadow_pcf(
        shadow_coord,
        cascade,
        shadow_data.shadow_bias,
        shadow_data.pcf_radius,
    );
}

// ── PBR functions ──────────────────────────────────────────────────────────
const PI: f32 = 3.14159265358979;

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a      = roughness * roughness;
    let a2     = a * a;
    let NdotH  = max(dot(N, H), 0.0);
    let denom  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness)
         * geometry_schlick_ggx(NdotL, roughness);
}

fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cosTheta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let smoother = max(vec3<f32>(1.0 - roughness), F0);
    return F0 + (smoother - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn compute_directional_light(light: LightData, N: vec3<f32>, V: vec3<f32>, albedo: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    let L = normalize(-light.position);
    let H = normalize(V + L);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    let D = distribution_ggx(N, H, roughness);
    let G = geometry_smith(N, V, L, roughness);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0001);
    let spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.0001);
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    return (kD * albedo / PI + spec) * light.color * light.intensity * NdotL;
}

fn compute_point_light(light: LightData, N: vec3<f32>, V: vec3<f32>, world_pos: vec3<f32>, albedo: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    let L_raw = light.position - world_pos;
    let dist = length(L_raw);
    let L = L_raw / max(dist, 0.001);
    let H = normalize(V + L);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    let D = distribution_ggx(N, H, roughness);
    let G = geometry_smith(N, V, L, roughness);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0001);
    let spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.0001);
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    return (kD * albedo / PI + spec) * light.color * light.intensity * NdotL;
}

fn compute_spot_light(light: LightData, N: vec3<f32>, V: vec3<f32>, world_pos: vec3<f32>, albedo: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    var contrib = compute_point_light(light, N, V, world_pos, albedo, metallic, roughness);
    let L = normalize(light.position - world_pos);
    var axis = light.direction;
    if length(axis) < 0.001 {
        axis = vec3<f32>(0.0, 0.0, -1.0);
    }
    let spot_cos = dot(-L, normalize(axis));
    let spot_atten = smoothstep(light.spot_angle_cos, light.spot_angle_cos + 0.01, spot_cos);
    return contrib * spot_atten;
}

// ── Real-time voxel GI: Voxel Cone Tracing ────────────────────────────────
// Marches a cone through the mip-mapped voxel grid (voxel_gi.wgsl + mip pass)
// and integrates the stored radiance. The mip level is picked from the cone
// footprint radius, so each step is one blurred fetch. `a` (occupancy) gates
// empty voxels out, and the solid-angle weighting keeps the result in
// radiance units regardless of how many steps we take.
fn trace_voxel_cone(
    origin:    vec3<f32>,
    dir:       vec3<f32>,
    half_angle: f32,
    max_dist:   f32,
    steps:      u32,
) -> vec3<f32> {
    let voxel_size = deferred.voxel_origin.w;
    let mip_count  = f32(textureNumLevels(t_voxel));
    let tan_a = tan(half_angle);
    let step_len = max_dist / f32(steps);
    var acc = vec3<f32>(0.0);
    var t = voxel_size * 1.5; // start beyond the surface shell to avoid self-hit
    for (var i = 0u; i < steps; i = i + 1u) {
        let sample_pos = origin + dir * t;
        let radius = t * tan_a;
        let mip = clamp(
            log2(max(radius / max(voxel_size, 1e-5), 1e-4)),
            0.0, mip_count - 1.0,
        );
        // World → normalized voxel-grid coordinates.
        let voxel_uv = (sample_pos - deferred.voxel_origin.xyz)
            / (deferred.voxel_dims.xyz * voxel_size);
        let vox = textureSampleLevel(t_voxel, s_voxel, voxel_uv, mip);
        // Solid-angle footprint of this step's sphere ≈ π r² / t², gated by occupancy.
        let footprint = PI * (radius * radius) / max(t * t, 1e-4);
        acc += vox.rgb * footprint * vox.a;
        t += step_len;
    }
    return acc / f32(steps);
}

// Indirect diffuse: 6 cones spanning a hemisphere around the normal. The cone
// directions come from the standard VCT set (Crassin et al.), rotated into a
// tangent basis built from the surface normal.
fn trace_voxel_gi_diffuse(origin: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    let up = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(N.y) > 0.99,
    );
    let T = normalize(cross(up, N));
    let B = cross(N, T);

    // (direction, tan(half-angle)) pairs — 4 side cones at ~30° + up/down at ~42°.
    let cones = array<vec4<f32>, 6>(
        vec4<f32>(1.0, 0.0, 1.0, 0.577350269),
        vec4<f32>(-1.0, 0.0, 1.0, 0.577350269),
        vec4<f32>(1.0, 0.0, -1.0, 0.577350269),
        vec4<f32>(-1.0, 0.0, -1.0, 0.577350269),
        vec4<f32>(0.0, 1.0, 0.0, 0.9),
        vec4<f32>(0.0, -1.0, 0.0, 0.9),
    );
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < 6u; i = i + 1u) {
        let local = normalize(cones[i].xyz);
        let world = normalize(T * local.x + N * local.y + B * local.z);
        let half_angle = atan(cones[i].w);
        sum += trace_voxel_cone(origin, world, half_angle, 8.0, 10u);
    }
    // Scale so the accumulated bounce stays comfortably below the direct light
    // (keeps the GI feedback loop convergent across frames).
    return sum * 0.35;
}

// Indirect specular: one tight cone along the reflection direction, widening
// with roughness.
fn trace_voxel_gi_specular(origin: vec3<f32>, dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    let half_angle = clamp(0.02 + roughness * 0.35, 0.02, 0.6);
    return trace_voxel_cone(origin, dir, half_angle, 6.0, 10u);
}

// ── Volumetric fog ─────────────────────────────────────────────────────────
fn hash31(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453123);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash31(i);
    let b = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let f2 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let g = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let h = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    return mix(
        mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
        mix(mix(e, f2, u.x), mix(g, h, u.x), u.y),
        u.z,
    );
}

// Raymarch the froxel volume along the view ray. Each sample holds
// rgb = in-scattered sun radiance at that cell and a = extinction density,
// injected by the froxel compute pass (froxel.wgsl). This replaces the old
// per-pixel analytic march with a cheap 3D-texture lookup per step.
fn compute_volumetric_fog(world_pos: vec3<f32>, uv: vec2<f32>) -> vec4<f32> {
    const NEAR: f32 = 0.1;
    const FAR: f32 = 200.0;
    let surf_dist = length(world_pos - uniforms.camera_pos);
    // Avoid marching beyond the surface (clamp so we always sample at least once).
    let max_t = clamp((log(surf_dist / NEAR) / log(FAR / NEAR)), 0.05, 0.98);

    const STEPS: u32 = 24u;
    var transmittance = 1.0;
    var scattered = vec3<f32>(0.0);
    var prev_depth = 0.0;

    for (var i = 0u; i < STEPS; i = i + 1u) {
        let t = (f32(i) + 0.5) / f32(STEPS);
        let depth = NEAR * pow(FAR / NEAR, t);
        if t > max_t { break; }
        let step_len = max(depth - prev_depth, 0.001);
        prev_depth = depth;

        let cell = textureSample(t_froxel, s_froxel, vec3<f32>(uv, t));
        let density = cell.a;
        let slice_trans = exp(-density * step_len);
        // Radiance emitted from this cell × how much view transmittance remains.
        scattered += transmittance * cell.rgb * step_len;
        transmittance *= slice_trans;
    }

    // Ambient fill so fog never looks completely black.
    let ambient = uniforms.fog_color.rgb * (1.0 - transmittance) * 0.6;
    return vec4<f32>(scattered + ambient, transmittance);
}

// ── Fragment shader ────────────────────────────────────────────────────────
struct DeferredOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

@fragment
fn fs_deferred(in: VsOut) -> DeferredOutput {
    let uv = in.uv;

    // ── Reconstruct world position from depth ──────────────────────────────
    let depth = textureSample(t_depth, s_gb, uv);

    // Sky pixels: geometry never wrote depth (cleared to 1.0 in the G-buffer
    // pass). Composite the separately-rendered sky colour. The normal target
    // gets a neutral value — SSR early-outs on sky pixels anyway.
    if (depth >= 0.9998) {
        return DeferredOutput(
            textureSample(t_sky, s_gb, uv),
            vec4<f32>(0.5, 0.5, 1.0, 1.0),
        );
    }

    let ndc = vec3<f32>(uv * 2.0 - vec2<f32>(1.0), depth * 2.0 - 1.0);
    let world_h = deferred.inv_view_proj * vec4<f32>(ndc, 1.0);
    let world_pos = world_h.xyz / world_h.w;

    // ── Read material properties from the G-buffer ────────────────────────
    let albedo_metallic = textureSample(t_gb_albedo, s_gb, uv);
    let albedo  = albedo_metallic.rgb;
    let metallic = albedo_metallic.a;

    let normal_encoded = textureSample(t_gb_normal, s_gb, uv);
    let N = normalize(normal_encoded.rgb * 2.0 - vec3<f32>(1.0));
    let roughness  = clamp(normal_encoded.a, 0.04, 1.0);

    let material = textureSample(t_gb_material, s_gb, uv);
    let emissive = material.rgb;
    var ao        = material.a;

    let extras = textureSample(t_gb_extras, s_gb, uv);
    let material_extras = MaterialExtras(
        extras.r, // subsurface
        extras.g, // clearcoat
        extras.b, // clearcoat_roughness
        extras.a, // anisotropy
        0.0,      // emissive_strength — emissive is already premultiplied into gb_material
        vec3<f32>(0.0),
    );

    let V  = normalize(uniforms.camera_pos - world_pos);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    let NdotV  = max(dot(N, V), 0.0);
    // Reconstruct clip-space depth the same way the forward vertex shader did:
    // view_z drives CSM cascade selection for shadow sampling.
    let clip = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    let view_z = abs(clip.z / max(clip.w, 0.0001));

    // ── Multi-light loop ────────────────────────────────────────────────────
    var total_lighting = vec3<f32>(0.0);

    for (var i = 0u; i < light_uniforms.light_count; i++) {
        let light = light_uniforms.lights[i];
        var contrib = vec3<f32>(0.0);

        if (light.light_type == 0.0) {
            contrib = compute_directional_light(light, N, V, albedo, metallic, roughness);
            if (light.shadow_index >= 0) {
                contrib *= compute_shadow(world_pos, N, view_z);
            }
        } else if (light.light_type == 1.0) {
            let attenuation = 1.0 - saturate(length(light.position - world_pos) / max(light.range, 0.001));
            contrib = compute_point_light(light, N, V, world_pos, albedo, metallic, roughness)
                    * attenuation * attenuation;
        } else {
            let attenuation = 1.0 - saturate(length(light.position - world_pos) / max(light.range, 0.001));
            contrib = compute_spot_light(light, N, V, world_pos, albedo, metallic, roughness)
                    * attenuation * attenuation;
        }

        // ── Subsurface scattering (per-light) ──────────────────────────────
        if (material_extras.subsurface > 0.01) {
            var L_sss: vec3<f32>;
            if (light.light_type == 0.0) {
                L_sss = normalize(-light.position);
            } else {
                L_sss = normalize(light.position - world_pos);
            }
            let wrap = 0.5;
            let wrap_lighting = max(0.0, (dot(N, L_sss) + wrap) / (1.0 + wrap));
            let translucency = pow(max(0.0, dot(V, -L_sss)), 2.0);
            let sss = albedo * (wrap_lighting + translucency * 0.5) * material_extras.subsurface;
            contrib += sss * light.color * light.intensity;
        }

        // ── Clearcoat (per-light) ───────────────────────────────────────────
        if (material_extras.clearcoat > 0.01) {
            var L_cc: vec3<f32>;
            if (light.light_type == 0.0) {
                L_cc = normalize(-light.position);
            } else {
                L_cc = normalize(light.position - world_pos);
            }
            let H_cc = normalize(V + L_cc);
            let F0_cc = vec3<f32>(0.04);
            let D_cc = distribution_ggx(N, H_cc, material_extras.clearcoat_roughness);
            let G_cc = geometry_smith(N, V, L_cc, material_extras.clearcoat_roughness);
            let F_cc = fresnel_schlick(max(dot(H_cc, V), 0.0), F0_cc);
            let NdotL_cc = max(dot(N, L_cc), 0.0);
            let NdotV_cc = max(dot(N, V), 0.0001);
            let cc_spec = (D_cc * G_cc * F_cc) / max(4.0 * NdotV_cc * NdotL_cc, 0.0001);
            contrib += cc_spec * light.color * light.intensity * NdotL_cc * material_extras.clearcoat;
        }

        total_lighting += contrib;
    }

    // ── IBL: diffuse irradiance ────────────────────────────────────────────
    let irr_uv      = dir_to_equirect(N);
    let irradiance  = textureSample(ibl_irradiance, ibl_irradiance_sampler, irr_uv).rgb;
    let F_ibl       = fresnel_schlick_roughness(NdotV, F0, roughness);
    let kD_ibl      = (vec3<f32>(1.0) - F_ibl) * (1.0 - metallic);
    let diffuse_ibl = irradiance * albedo;

    // ── IBL: specular reflection ───────────────────────────────────────────
    let R           = reflect(-V, N);
    let refl_uv     = dir_to_equirect(R);
    let mip_count   = f32(textureNumLevels(ibl_prefilter));
    let prefiltered = textureSampleLevel(
        ibl_prefilter, ibl_prefilter_sampler, refl_uv,
        roughness * (mip_count - 1.0),
    ).rgb;
    let brdf     = textureSample(brdf_lut, brdf_lut_sampler, vec2<f32>(NdotV, roughness)).rg;
    let spec_ibl = prefiltered * (F_ibl * brdf.x + brdf.y);

    // ── Combine ambient + direct lighting ─────────────────────────────────
    // GTAO: sample the half-res ambient-occlusion mask produced by the GTAO
    // compute pass (deferred.wgsl binding 8). UV is scaled by 0.5 because the
    // AO texture is half resolution.
    if uniforms.post_params0.z > 0.5 {
        let ao_uv = uv * vec2<f32>(0.5);
        let gtao_ao = textureSample(t_ao, s_gb, ao_uv).r;
        let view_dist = length(uniforms.camera_pos - world_pos);
        let distance_fade = 1.0 - smoothstep(10.0, 80.0, view_dist);
        let approx = clamp(
            (1.0 - gtao_ao) * 0.9 * distance_fade * uniforms.post_params0.w,
            0.0, 0.85,
        );
        ao *= 1.0 - approx;
        ao = max(ao, 0.18);
    }
    let ambient = (kD_ibl * diffuse_ibl + spec_ibl) * ao;

    var color = ambient + total_lighting + emissive;

    // ── Indirect light: real-time voxel GI + baked probe hybrid ─────────────
    // Horizon Forbidden West style: dynamic one-bounce GI comes from the
    // camera-aligned voxel clipmap (cone-traced), while baked SH irradiance
    // volumes add stable distant fill. Accumulated into `gi` so it survives
    // the snow re-blend below.
    var gi = vec3<f32>(0.0);

    if uniforms.post_params1.z > 0.5 {
        // Cone-trace the voxel grid for indirect diffuse (6 cones) + specular.
        let voxel_diff  = trace_voxel_gi_diffuse(world_pos, N);
        let voxel_spec  = trace_voxel_gi_specular(world_pos, R, roughness);
        let diffuse_gi  = voxel_diff * albedo * (1.0 - metallic);
        let specular_gi = voxel_spec * F_ibl;
        gi += (diffuse_gi + specular_gi) * uniforms.post_params1.w;
    }

    if probe_control.count > 0u {
        // Nearest-probe IDW blend — matching LightProbeGrid::interpolate.
        var best_w = -1.0;
        var best_i = 0u;
        for (var i = 0u; i < min(probe_control.count, 32u); i = i + 1u) {
            let p = probe_data[i * 10u + 0u];
            let d = distance(p.xyz, world_pos);
            if d < p.w {
                let w = 1.0 / (d * d + 0.01);
                if w > best_w {
                    best_w = w;
                    best_i = i;
                }
            }
        }
        var probe_irr = vec3<f32>(0.0);
        if best_w >= 0.0 {
            probe_irr = eval_sh(best_i * 10u, N);
        } else {
            // Fall back to the nearest probe globally if none is in range.
            var nd = 1e9;
            var ni = 0u;
            for (var i = 0u; i < min(probe_control.count, 32u); i = i + 1u) {
                let d = distance(probe_data[i * 10u + 0u].xyz, world_pos);
                if d < nd {
                    nd = d;
                    ni = i;
                }
            }
            probe_irr = eval_sh(ni * 10u, N);
        }
        let bake_strength = select(
            0.25, uniforms.post_params1.w,
            uniforms.post_params1.z > 0.5,
        );
        gi += probe_irr * albedo * bake_strength;
    }

    // Snow accumulation: blend snow albedo/roughness on upward-facing surfaces.
    let snow_up = max(dot(N, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
    let snow_mask = smoothstep(0.4, 0.8, snow_up) * weather.snow_coverage;
    let snow_albedo = vec3<f32>(0.92, 0.95, 0.98);
    let snow_roughness = 0.25;
    let snow_metallic = 0.0;
    let final_albedo = mix(albedo, snow_albedo, snow_mask);
    let final_roughness = mix(roughness, snow_roughness, snow_mask);
    let final_metallic = mix(metallic, snow_metallic, snow_mask);

    // Recompute ambient with snow-blended material properties for visible accumulation.
    let F0_snow = mix(vec3<f32>(0.04), final_albedo, final_metallic);
    let F_ibl_snow = fresnel_schlick_roughness(NdotV, F0_snow, final_roughness);
    let kD_ibl_snow = (vec3<f32>(1.0) - F_ibl_snow) * (1.0 - final_metallic);
    let diffuse_ibl_snow = irradiance * final_albedo;
    let prefiltered_snow = textureSampleLevel(
        ibl_prefilter, ibl_prefilter_sampler, refl_uv,
        final_roughness * (mip_count - 1.0),
    ).rgb;
    let brdf_snow = textureSample(brdf_lut, brdf_lut_sampler, vec2<f32>(NdotV, final_roughness)).rg;
    let spec_ibl_snow = prefiltered_snow * (F_ibl_snow * brdf_snow.x + brdf_snow.y);
    let ambient_snow = (kD_ibl_snow * diffuse_ibl_snow + spec_ibl_snow) * ao;

    // Blend ambient toward snow-lit version where snow covers the surface.
    let ambient_blended = mix(ambient, ambient_snow, snow_mask);
    color = ambient_blended + gi + (total_lighting * (1.0 - snow_mask * 0.3)) + emissive;

    // Volumetric fog: ray-marches from camera to surface.
    if uniforms.post_params1.x > 0.5 {
        let vf = compute_volumetric_fog(world_pos, uv);
        color = color * vf.w + vf.xyz;
    }

    // Output raw HDR linear + world-space normals (MRT target 1, for SSR).
    // Normals encoded as (N * 0.5 + 0.5); tone mapping is applied by the
    // post-process tonemap pass.
    return DeferredOutput(
        vec4<f32>(color, 1.0),
        vec4<f32>(N * 0.5 + vec3<f32>(0.5), 1.0),
    );
}