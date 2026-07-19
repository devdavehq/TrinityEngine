// ── Bind Group 0: global (camera + IBL) ───────────────────────────────────
// This group changes once per frame.

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
    fog_color:    vec4<f32>, // rgb = dynamic fog color from TimeOfDay, w = elapsed time
    wind_dir_strength: vec4<f32>, // xyz = wind direction (normalised), w = wind strength [0..1]
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// IBL textures. We use equirectangular maps sampled with a direction vector.
// A full implementation would use cubemaps — equirectangular is simpler to load.
@group(0) @binding(1) var ibl_irradiance:         texture_2d<f32>;
@group(0) @binding(2) var ibl_irradiance_sampler:  sampler;
@group(0) @binding(3) var ibl_prefilter:           texture_2d<f32>;
@group(0) @binding(4) var ibl_prefilter_sampler:   sampler;
@group(0) @binding(5) var brdf_lut:                texture_2d<f32>;
@group(0) @binding(6) var brdf_lut_sampler:        sampler;

// ── Multi-light uniform buffer (group 0, binding 12) ────────────────────────
// Up to 16 lights: directional sun, point lights, spot lights.
// Populated each frame from the ECS world.
const MAX_LIGHTS: u32 = 16u;

struct LightData {
    position: vec3<f32>,       // world-space position (point/spot) or direction (directional)
    _pos_pad: f32,
    color: vec3<f32>,          // light colour
    _col_pad: f32,
    intensity: f32,            // brightness multiplier
    range: f32,                // attenuation range (0 = infinite / directional)
    light_type: f32,           // 0 = directional, 1 = point, 2 = spot
    spot_angle_cos: f32,       // cos of spot cone half-angle
    shadow_index: i32,         // index into shadow cascade array (-1 = no shadow)
    _pad: f32,
};

struct LightUniforms {
    lights: array<LightData, 16>,
    light_count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};
@group(0) @binding(12) var<uniform> light_uniforms: LightUniforms;

// ── Bind Group 1: per-material textures + extras ────────────────────────────
// This group changes per entity (each entity has its own material).
@group(1) @binding(0) var t_albedo:             texture_2d<f32>;
@group(1) @binding(1) var s_albedo:             sampler;
@group(1) @binding(2) var t_normal:             texture_2d<f32>;
@group(1) @binding(3) var s_normal:             sampler;
@group(1) @binding(4) var t_metallic_roughness: texture_2d<f32>;
@group(1) @binding(5) var s_metallic_roughness: sampler;

// ── Per-object material extras (subsurface, clearcoat, etc.) ─────────────────
struct MaterialExtras {
    subsurface:          f32,  // 0 = none, 1 = full SSS (leaves, skin, cloth)
    clearcoat:           f32,  // 0-1 clearcoat layer strength (car paint, lacquer)
    clearcoat_roughness: f32,  // 0-1 clearcoat roughness
    anisotropy:          f32,  // anisotropic highlight stretch (brushed metal, hair)
    emissive_strength:   f32,  // self-illumination multiplier
    _pad: vec3<f32>,
};
@group(1) @binding(6) var<uniform> material_extras: MaterialExtras;

// ── Vertex input/output ────────────────────────────────────────────────────
struct VertIn {
    @location(0) position:  vec3<f32>,
    @location(1) normal:    vec3<f32>,
    @location(2) tangent:   vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) color:     vec3<f32>,
    @location(5) metallic:  f32,
    @location(6) roughness: f32,
    @location(7) ao:        f32,
}

// ── Per-instance data (matches InstanceData in instancing.rs) ──────────────
// Two buffers: slot 0 = per-vertex, slot 1 = per-instance.
struct InstanceIn {
    @location(8)  model_row0:     vec4<f32>,
    @location(9)  model_row1:     vec4<f32>,
    @location(10) model_row2:     vec4<f32>,
    @location(11) model_row3:     vec4<f32>,
    @location(12) color_metallic: vec4<f32>,
    @location(13) roughness_ao:   vec4<f32>,
}

struct VertOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       normal:    vec3<f32>,
    @location(2)       tangent:   vec3<f32>,
    @location(3)       bitangent: vec3<f32>,
    @location(4)       color:     vec3<f32>,
    @location(5)       metallic:  f32,
    @location(6)       roughness: f32,
    @location(7)       ao:        f32,
}

@vertex
fn vs_main(in: VertIn, instance: InstanceIn) -> VertOut {
    // Reconstruct model matrix from instance data.
    let model = mat4x4<f32>(
        instance.model_row0,
        instance.model_row1,
        instance.model_row2,
        instance.model_row3,
    );

    // Transform vertex position by instance model matrix.
    var world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;

    // ── Wind displacement ──────────────────────────────────────────────────
    // Wind sways vertices based on their local-space height (Y position).
    // Vertices at or below y=0 are anchored (ground level).
    // Vertices above y=0 sway proportionally to height — taller parts of
    // trees/plants move more, matching how real vegetation behaves.
    // Two sine waves at different frequencies create organic-looking motion
    // instead of a robotic back-and-forth.
    {
        let wind = uniforms.wind_dir_strength;
        let strength = wind.w;
        if (strength > 0.001) {
            let height_factor = max(in.position.y, 0.0);
            // Primary sway wave — driven by elapsed time.
            let phase = uniforms.fog_color.w * 2.0
                      + in.position.x * 0.5
                      + in.position.z * 0.3;
            // Secondary wave at different frequency for organic feel.
            let phase2 = uniforms.fog_color.w * 3.3
                       + in.position.x * 0.2
                       + in.position.z * 0.7;
            let sway = vec3<f32>(
                sin(phase)  * strength * height_factor,
                sin(phase2) * strength * height_factor * 0.15,
                cos(phase * 0.7) * strength * height_factor * 0.5,
            );
            world_pos += sway;
        }
    }

    // Transform normal by the upper-left 3x3 of the model matrix.
    // For uniform scale this is just the rotation part; for non-uniform
    // scale it should use the inverse transpose, but we approximate here.
    let normal_matrix = mat3x3<f32>(
        instance.model_row0.xyz,
        instance.model_row1.xyz,
        instance.model_row2.xyz,
    );
    var world_normal = normalize(normal_matrix * in.normal);

    // Transform tangent and bitangent by the normal matrix (upper-left 3x3).
    // These define the TBN frame in world space for tangent-space normal mapping.
    var world_tangent   = normalize(normal_matrix * in.tangent);
    var world_bitangent = normalize(normal_matrix * in.bitangent);
    // Re-orthogonalize tangent w.r.t. normal (Gram-Schmidt).
    world_tangent = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
    // Recompute bitangent to maintain right-handed frame.
    world_bitangent = cross(world_normal, world_tangent);

    var out: VertOut;
    out.clip_pos  = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.normal    = world_normal;
    out.tangent   = world_tangent;
    out.bitangent = world_bitangent;
    // Use instance color/metallic/roughness/ao if present (non-zero color).
    out.color     = select(in.color, instance.color_metallic.xyz, instance.color_metallic.xyz != vec3<f32>(0.0));
    out.metallic  = select(in.metallic, instance.color_metallic.w, instance.color_metallic.w >= 0.0);
    out.roughness = select(in.roughness, instance.roughness_ao.x, instance.roughness_ao.x > 0.0);
    out.ao        = select(in.ao, instance.roughness_ao.y, instance.roughness_ao.y > 0.0);
    return out;
}

// ── Utility: equirectangular direction → UV ────────────────────────────────
// Converts a 3D direction vector into UV coordinates for sampling
// an equirectangular (lat-long) environment map.
// This is how we use a flat HDR image as a spherical environment.
fn dir_to_equirect(dir: vec3<f32>) -> vec2<f32> {
    let n = normalize(dir);
    // atan2 gives the longitude (horizontal angle around Y axis).
    // asin gives the latitude (vertical angle).
    let uv = vec2<f32>(
        atan2(n.z, n.x) / (2.0 * 3.14159265) + 0.5,
        asin(n.y) / 3.14159265 + 0.5,
    );
    return uv;
}


// ── Shadow map uniforms ────────────────────────────────────────────────────
// Three cascade shadow maps and their light-space matrices.
// group(0) binding 7-12 (after IBL bindings).

struct ShadowData {
    light_matrices: array<mat4x4<f32>, 3>,  // one per cascade
    cascade_dists:  vec4<f32>,              // xyz = cascade far distances
    // Shadow settings
    shadow_bias:        f32,
    normal_offset_bias: f32,
    pcf_radius:         f32,
    shadow_enabled:     f32,  // 0 or 1 — toggleable on old hardware
    shadow_map_size:    f32,  // resolution of the shadow map (e.g. 2048.0)
}

@group(0) @binding(7)  var<uniform>    shadow_data:     ShadowData;
@group(0) @binding(8)  var            t_shadow0:        texture_depth_2d;
@group(0) @binding(9)  var            t_shadow1:        texture_depth_2d;
@group(0) @binding(10) var            t_shadow2:        texture_depth_2d;
@group(0) @binding(11) var            s_shadow:         sampler_comparison;

// ── PCF shadow sampling ────────────────────────────────────────────────────
// PCF = Percentage Closer Filtering.
// Instead of one depth comparison (hard shadow), we take N samples in a
// small neighbourhood around the shadow map coordinate and average the results.
// This gives soft shadow edges that match how real shadows look.
//
// shadow_coord: position in light clip space (after light matrix transform).
// shadow_texture: which cascade's shadow map to sample.
// bias: how much to offset depth comparison (prevents self-shadowing acne).
// pcf_radius: how many texels to spread samples over (larger = softer).
fn sample_shadow_pcf(
    shadow_coord: vec3<f32>,
    cascade_idx: i32,
    bias: f32,
    pcf_radius: f32,
) -> f32 {
    // Convert from NDC [-1,1] to UV [0,1] space.
    let uv  = shadow_coord.xy * 0.5 + 0.5;
    // Flip Y: NDC has Y up, texture has Y down.
    let uv2 = vec2<f32>(uv.x, 1.0 - uv.y);
    // Reference depth with bias applied.
    // Subtracting bias means the comparison passes for surfaces
    // slightly closer to the light than the shadow map says —
    // this prevents a surface from shadowing itself.
    let ref_depth = shadow_coord.z - bias;

    // PCF kernel: 3×3 = 9 samples, or 5×5 = 25 for PCSS.
    // We use a 3×3 Poisson disk for performance.
    // Poisson disk = samples are well-distributed (not on a grid) for better quality.
    let offsets = array<vec2<f32>, 9>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 0.0, -1.0), vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  0.0), vec2<f32>( 0.0,  0.0), vec2<f32>( 1.0,  0.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 0.0,  1.0), vec2<f32>( 1.0,  1.0),
    );

    // Texel size in UV space: 1/resolution.
    // Uses shadow_data.shadow_map_size which is set from RenderFeatures.shadow_resolution.
    let texel_size = pcf_radius / shadow_data.shadow_map_size;

    var shadow_sum = 0.0;
    for (var i = 0; i < 9; i++) {
        let sample_uv = uv2 + offsets[i] * texel_size;

        // textureSampleCompare: samples the depth texture and compares against ref_depth.
        // Returns 1.0 if the stored depth >= ref_depth (lit), 0.0 if < (shadowed).
        // With the linear sampler, this bilinearly interpolates between 4 comparison results.
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

    // Average: 0.0 = fully in shadow, 1.0 = fully lit.
    return shadow_sum / 9.0;
}

// ── Select cascade based on distance from camera ───────────────────────────
fn get_cascade_index(view_z: f32) -> i32 {
    // shadow_data.cascade_dists.xyz = far distances of cascades 0, 1, 2.
    if view_z < shadow_data.cascade_dists.x { return 0; }
    if view_z < shadow_data.cascade_dists.y { return 1; }
    return 2;
}

// ── Compute shadow factor ──────────────────────────────────────────────────
// This is the main shadow function called from the fragment shader.
// Returns 0.0 = fully in shadow, 1.0 = fully lit.
//
// world_pos: fragment's world space position.
// N: surface normal — used for normal offset bias (light leak fix).
fn compute_shadow(world_pos: vec3<f32>, N: vec3<f32>, view_z: f32) -> f32 {
    // Early out if shadows are disabled.
    if shadow_data.shadow_enabled < 0.5 { return 1.0; }

    // Select which cascade covers this fragment's depth.
    let cascade = get_cascade_index(abs(view_z));

    // Normal offset bias — THIS IS THE LIGHT LEAK FIX.
    // The problem: at grazing angles, the shadow comparison point is slightly
    // inside the surface (due to floating point), making it compare against
    // the wrong depth. The surface appears to shadow itself.
    //
    // The fix: offset the world position along the surface normal before
    // projecting into shadow space. This moves the comparison point off the
    // surface entirely, into open air — no self-intersection.
    //
    // The offset amount scales with the angle between N and the light:
    // surfaces nearly parallel to the light get a larger offset.
    let normal_bias_amount = shadow_data.normal_offset_bias
        * (1.0 - max(dot(N, uniforms.light_dir), 0.0));
    let biased_pos = world_pos + N * normal_bias_amount;

    // Project the biased position into the cascade's light clip space.
    let light_clip = shadow_data.light_matrices[cascade] * vec4<f32>(biased_pos, 1.0);

    // Perspective divide: convert from homogeneous to NDC.
    let shadow_coord = light_clip.xyz / light_clip.w;

    // Check if this point is outside the shadow map's coverage area.
    // If so, return "lit" — we don't have shadow data here.
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





// ── PBR functions (same as before) ────────────────────────────────────────
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

// Fresnel with roughness bias — used for IBL specular.
// Rougher surfaces have less sharp Fresnel transitions.
fn fresnel_schlick_roughness(cosTheta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let smoother = max(vec3<f32>(1.0 - roughness), F0);
    return F0 + (smoother - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ── Per-light PBR Cook-Torrance functions ───────────────────────────────────
// These compute the direct lighting contribution from a single light,
// used inside the multi-light loop.

fn compute_directional_light(light: LightData, N: vec3<f32>, V: vec3<f32>, albedo: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    // Directional light: position field stores the light DIRECTION.
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
    // Cone attenuation: sharp falloff at the edge of the spot cone.
    let L = normalize(light.position - world_pos);
    // light.position doubles as spot direction for spot lights.
    let spot_cos = dot(-L, normalize(light.position));
    let spot_atten = smoothstep(light.spot_angle_cos, light.spot_angle_cos + 0.01, spot_cos);
    return contrib * spot_atten;
}

// ── Fragment shader ────────────────────────────────────────────────────────
@fragment
fn fs_main(in: VertOut) -> (@location(0) vec4<f32>, @location(1) vec4<f32>) {

    // ── Sample textures ────────────────────────────────────────────────────
    // Sample albedo texture at the vertex's UV coordinates.
    // Multiply by the vertex color (tint). Usually tint = white = no change.
    let world_uv = in.world_pos.xz * 0.12;
    let albedo_sample = textureSample(t_albedo, s_albedo, world_uv).rgb;
    let albedo = albedo_sample * in.color;

    // Sample normal map.
    // Normal maps store directions as RGB where (0.5, 0.5, 1.0) = flat.
    // Unpacking: multiply by 2 and subtract 1 maps (0..1) → (-1..1).
    let normal_sample = textureSample(t_normal, s_normal, world_uv).rgb;
    // Tangent-space normal from the map.
    let normal_tangent = normal_sample * 2.0 - vec3<f32>(1.0);
    // Build TBN matrix: transforms from tangent space to world space.
    let TBN = mat3x3<f32>(in.tangent, in.bitangent, in.normal);
    let N = normalize(TBN * normal_tangent);

    // ── Sample metallic-roughness texture ─────────────────────────────────
    // glTF convention: B channel = metallic, G channel = roughness.
    let mr_sample  = textureSample(t_metallic_roughness, s_metallic_roughness, world_uv);
    // Take the higher of texture and vertex value — allows scene-level override.
    let metallic   = max(mr_sample.b, in.metallic);
    let roughness_raw = max(mr_sample.g, in.roughness);
    // Clamp roughness — pure 0.0 causes division by zero in GGX.
    let roughness  = clamp(roughness_raw, 0.04, 1.0);

    let V  = normalize(uniforms.camera_pos - in.world_pos);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    let NdotV  = max(dot(N, V), 0.0);
    let view_z = abs(in.clip_pos.z / max(in.clip_pos.w, 0.0001));

    // ── Multi-light loop ────────────────────────────────────────────────────
    // Iterates over all active lights (directional, point, spot) and accumulates
    // their PBR Cook-Torrance contributions.  Includes per-light SSS and clearcoat
    // contributions when material_extras enables them.
    var total_lighting = vec3<f32>(0.0);

    for (var i = 0u; i < light_uniforms.light_count; i++) {
        let light = light_uniforms.lights[i];
        var contrib = vec3<f32>(0.0);

        if (light.light_type == 0.0) {
            // ── Directional light (sun) ─────────────────────────────────
            contrib = compute_directional_light(light, N, V, albedo, metallic, roughness);
            // Apply cascaded shadow maps for the directional light.
            if (light.shadow_index >= 0) {
                contrib *= compute_shadow(in.world_pos, N, view_z);
            }
        } else if (light.light_type == 1.0) {
            // ── Point light with range-based attenuation ────────────────
            let attenuation = 1.0 - saturate(length(light.position - in.world_pos) / max(light.range, 0.001));
            contrib = compute_point_light(light, N, V, in.world_pos, albedo, metallic, roughness)
                    * attenuation * attenuation;
        } else {
            // ── Spot light ─────────────────────────────────────────────
            let attenuation = 1.0 - saturate(length(light.position - in.world_pos) / max(light.range, 0.001));
            contrib = compute_spot_light(light, N, V, in.world_pos, albedo, metallic, roughness)
                    * attenuation * attenuation;
        }

        // ── Subsurface scattering (per-light) ──────────────────────────────
        // Wrap lighting + view-dependent translucency for thin surfaces
        // (leaves, cloth, skin).  Blends with standard diffuse via subsurface.
        if (material_extras.subsurface > 0.01) {
            var L_sss: vec3<f32>;
            if (light.light_type == 0.0) {
                L_sss = normalize(-light.position);
            } else {
                L_sss = normalize(light.position - in.world_pos);
            }
            let wrap = 0.5;
            let wrap_lighting = max(0.0, (dot(N, L_sss) + wrap) / (1.0 + wrap));
            let translucency = pow(max(0.0, dot(V, -L_sss)), 2.0);
            let sss = albedo * (wrap_lighting + translucency * 0.5) * material_extras.subsurface;
            contrib += sss * light.color * light.intensity;
        }

        // ── Clearcoat (per-light) ───────────────────────────────────────────
        // Thin dielectric layer on top of the base material (car paint, lacquer).
        // Uses a second GGX lobe with F0 = 0.04 (glass-like) and its own roughness.
        if (material_extras.clearcoat > 0.01) {
            var L_cc: vec3<f32>;
            if (light.light_type == 0.0) {
                L_cc = normalize(-light.position);
            } else {
                L_cc = normalize(light.position - in.world_pos);
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
    var ao = in.ao;
    if uniforms.post_params0.z > 0.5 {
        let horizon = 1.0 - max(N.y, 0.0);
        let dist_occ = clamp(view_z * 0.03, 0.0, 1.0);
        let approx = clamp((horizon * 0.6 + dist_occ * 0.4) * uniforms.post_params0.w, 0.0, 0.9);
        ao *= (1.0 - approx);
    }
    let ambient = (kD_ibl * diffuse_ibl + spec_ibl) * ao;

    // ── Emissive self-illumination ─────────────────────────────────────────
    var emissive = vec3<f32>(0.0);
    if material_extras.emissive_strength > 0.01 {
        emissive = albedo * material_extras.emissive_strength;
    }

    var color = ambient + total_lighting + emissive;

    // Voxel GI prototype (cheap approximation) — gives blocky bounced fill style.
    if uniforms.post_params1.z > 0.5 {
        let voxel_cell = floor(in.world_pos * 0.5);
        let hash = fract(sin(dot(voxel_cell, vec3<f32>(12.9898, 78.233, 39.425))) * 43758.5453);
        let voxel_bounce = mix(vec3<f32>(0.03, 0.04, 0.05), albedo * 0.25, hash);
        color += voxel_bounce * uniforms.post_params1.w;
    }

    // Volumetric fog approximation.
    if uniforms.post_params1.x > 0.5 {
        let fog = 1.0 - exp(-view_z * uniforms.post_params1.y);
        let fog_color = uniforms.fog_color.rgb;
        color = mix(color, fog_color, clamp(fog, 0.0, 0.9));
    }

    // Output raw HDR linear + world-space normals (MRT target 1).
    // Normals are encoded as (N * 0.5 + 0.5) so negative values map to [0,0.5]
    // and positive values map to [0.5,1.0] — suitable for Rgba16Float.
    // Tone mapping and gamma are applied by the post-process tonemap pass.
    return (vec4<f32>(color, 1.0), vec4<f32>(N * 0.5 + vec3<f32>(0.5), 1.0));
}

