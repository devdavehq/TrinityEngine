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

// ── Bind Group 1: per-material textures ───────────────────────────────────
// This group changes per entity (each entity has its own material).
@group(1) @binding(0) var t_albedo:             texture_2d<f32>;
@group(1) @binding(1) var s_albedo:             sampler;
@group(1) @binding(2) var t_normal:             texture_2d<f32>;
@group(1) @binding(3) var s_normal:             sampler;
@group(1) @binding(4) var t_metallic_roughness: texture_2d<f32>;
@group(1) @binding(5) var s_metallic_roughness: sampler;

// ── Vertex input/output ────────────────────────────────────────────────────
struct VertIn {
    @location(0) position:  vec3<f32>,
    @location(1) normal:    vec3<f32>,
    @location(2) color:     vec3<f32>,
    @location(3) metallic:  f32,
    @location(4) roughness: f32,
    @location(5) ao:        f32,
}

struct VertOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       normal:    vec3<f32>,
    @location(2)       color:     vec3<f32>,
    @location(3)       metallic:  f32,
    @location(4)       roughness: f32,
    @location(5)       ao:        f32,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    out.clip_pos  = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.normal    = in.normal;
    out.color     = in.color;
    out.metallic  = in.metallic;
    out.roughness = in.roughness;
    out.ao        = in.ao;
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
    // We hardcode 2048 to match SHADOW_MAP_SIZE in shadow.rs.
    let texel_size = pcf_radius / 2048.0;

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

// ── Fragment shader ────────────────────────────────────────────────────────
@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {

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
    // Unpack: (0,0,1) in texture = (0,0,1) in tangent space = flat surface.
    // We're using object-space normals here for simplicity.
    // A full implementation uses tangent-space normals + TBN matrix.
    let N_from_map = normalize(normal_sample * 2.0 - vec3<f32>(1.0));

    // Blend between geometry normal and normal map.
    // For now weight toward geometry normal for stability.
    // A full TBN implementation would use N_from_map directly.
    let N = normalize(in.normal * 0.3 + N_from_map * 0.7);

    // Sample metallic-roughness texture.
    // glTF convention: B channel = metallic, G channel = roughness.
    let mr_sample  = textureSample(t_metallic_roughness, s_metallic_roughness, world_uv);
    // Multiply by component values from vertex — allows scene-level override.
    let metallic   = mr_sample.b * in.metallic  + (1.0 - in.metallic)  * mr_sample.b;
    let roughness_raw  = mr_sample.g * in.roughness + (1.0 - in.roughness) * mr_sample.g;
    // Clamp roughness — pure 0.0 causes division by zero in GGX.
    let roughness  = clamp(roughness_raw, 0.04, 1.0);

    let V  = normalize(uniforms.camera_pos - in.world_pos);
    let L  = normalize(uniforms.light_dir);
    let H  = normalize(V + L);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    // ── Direct lighting (same Cook-Torrance as before) ─────────────────────
    let D      = distribution_ggx(N, H, roughness);
    let G      = geometry_smith(N, V, L, roughness);
    let F_dir  = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let NdotL  = max(dot(N, L), 0.0);
    let NdotV  = max(dot(N, V), 0.0);
    let spec   = (D * G * F_dir) / max(4.0 * NdotV * NdotL, 0.0001);
    let kS_dir = F_dir;
    let kD_dir = (vec3<f32>(1.0) - kS_dir) * (1.0 - metallic);
   // IBL is ambient — it comes from everywhere so shadows don't apply.
    let view_z = abs(in.clip_pos.z / max(in.clip_pos.w, 0.0001));
    let shadow = compute_shadow(in.world_pos, N, view_z);
    let Lo = (kD_dir * albedo / PI + spec) * uniforms.light_color * 3.0 * NdotL * shadow;


    // ── IBL: diffuse irradiance ────────────────────────────────────────────
    // Sample the irradiance map in the surface normal direction.
    // The irradiance map tells us: "total light arriving from the hemisphere
    // around direction N." This is the diffuse contribution from the environment.
    let irr_uv      = dir_to_equirect(N);
    let irradiance  = textureSample(ibl_irradiance, ibl_irradiance_sampler, irr_uv).rgb;
    let F_ibl       = fresnel_schlick_roughness(NdotV, F0, roughness);
    let kD_ibl      = (vec3<f32>(1.0) - F_ibl) * (1.0 - metallic);
    let diffuse_ibl = irradiance * albedo;

    // ── IBL: specular reflection ───────────────────────────────────────────
    // Reflect the view direction off the surface to get the reflection direction.
    // Sample the prefiltered env at a mip level matching roughness.
    let R           = reflect(-V, N);
    let refl_uv     = dir_to_equirect(R);
    // Select mip level based on roughness: rougher = blurrier reflection.
    // textureNumLevels gives the total mip count.
    let mip_count   = f32(textureNumLevels(ibl_prefilter));
    let prefiltered = textureSampleLevel(
        ibl_prefilter,
        ibl_prefilter_sampler,
        refl_uv,
        roughness * (mip_count - 1.0),
    ).rgb;

    // BRDF LUT lookup: gives scale and bias for the specular integral.
    // X = NdotV (clamped to valid range), Y = roughness.
    let brdf        = textureSample(brdf_lut, brdf_lut_sampler,
                          vec2<f32>(NdotV, roughness)).rg;
    let spec_ibl    = prefiltered * (F_ibl * brdf.x + brdf.y);

    // Combine IBL diffuse + specular with ambient occlusion.
    var ao = in.ao; // vertex AO — later add AO texture
    if uniforms.post_params0.z > 0.5 {
        // Cheap SSAO-like approximation from normal orientation and distance.
        let horizon = 1.0 - max(N.y, 0.0);
        let dist_occ = clamp(view_z * 0.03, 0.0, 1.0);
        let approx = clamp((horizon * 0.6 + dist_occ * 0.4) * uniforms.post_params0.w, 0.0, 0.9);
        ao *= (1.0 - approx);
    }
    let ambient     = (kD_ibl * diffuse_ibl + spec_ibl) * ao;

    // Single movable point light.
    let pl_pos = uniforms.point_light_pos_range.xyz;
    let pl_range = max(uniforms.point_light_pos_range.w, 0.001);
    let pl_color = uniforms.point_light_color_intensity.xyz;
    let pl_intensity = uniforms.point_light_color_intensity.w;
    let to_pl = pl_pos - in.world_pos;
    let pl_dist = length(to_pl);
    let pl_dir = normalize(to_pl);
    let pl_nl = max(dot(N, pl_dir), 0.0);
    let pl_att = clamp(1.0 - (pl_dist / pl_range), 0.0, 1.0);
    let point_diff = (albedo / PI) * pl_color * (pl_nl * pl_att * pl_intensity);

    // ── Final combination ──────────────────────────────────────────────────
    var color = ambient + Lo + point_diff;

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
        let fog_color = vec3<f32>(0.52, 0.60, 0.70);
        color = mix(color, fog_color, clamp(fog, 0.0, 0.9));
    }

    // Tone mapping (Reinhard) — HDR → LDR.
    color = color / (color + vec3<f32>(1.0));

    // Gamma correction — linear → sRGB for display.
    color = pow(color, vec3<f32>(1.0 / 2.2));

    // Bloom approximation: boost highlights after tone map.
    if uniforms.post_params0.x > 0.5 {
        let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
        let glow = smoothstep(0.7, 1.0, lum) * uniforms.post_params0.y;
        color += color * glow;
    }

    return vec4<f32>(color, 1.0);
}

