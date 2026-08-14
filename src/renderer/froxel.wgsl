// src/renderer/froxel.wgsl
// Froxel volumetric lighting — inject sun light scattering into a 3D texture.
//
// A "froxel" is a frustum-aligned voxel: XY maps to screen UV, Z maps to an
// exponential depth slice (more slices near the camera, where the eye cares).
// For each froxel we reconstruct the world-space position, compute the smoke /
// haze density (height falloff + animated value noise), then march a few steps
// toward the sun to accumulate in-scattered radiance and the transmittance of
// the sun shaft reaching this cell.
//
// The deferred pass later raymarches this 3D texture along the view ray so the
// fog both thickens with distance and glows toward the sun (volumetric light
// shafts) without doing a per-pixel march inside the lighting shader.

struct FroxelUniforms {
    inv_view_proj: mat4x4<f32>, // inverse view-projection to unproject froxels
    camera_pos:    vec3<f32>,
    _p0:           f32,
    sun_dir:       vec3<f32>,
    _p1:           f32,
    sun_color:     vec3<f32>,
    _p2:           f32,
    fog:           vec4<f32>,   // x = density, y = near, z = far, w = elapsed time
    grid:          vec4<f32>,   // xyz = grid resolution, w = sun intensity scale
}
@group(0) @binding(0) var<uniform> froxel: FroxelUniforms;

@group(0) @binding(1) var t_froxel_out: texture_storage_3d<rgba16float, write>;

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

// Compute smoke density at a world-space point.
fn cloud_density(p: vec3<f32>) -> f32 {
    let time = froxel.fog.w;
    let height = max(0.0, 8.0 - p.y);
    let noise = value_noise(p * 0.25 + vec3<f32>(time * 0.02, 0.0, time * 0.013));
    return froxel.fog.x * (height * 0.12 + 0.25) * (0.6 + 0.4 * noise);
}

@compute @workgroup_size(8, 8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(textureDimensions(t_froxel_out));
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }

    // XY → screen UV, Z → exponential depth slice.
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims.xy);
    let t = (f32(gid.z) + 0.5) / f32(dims.z);
    let near = froxel.fog.y;
    let far = froxel.fog.z;
    let view_depth = near * pow(far / near, t);

    // Reconstruct the world-space ray through this pixel.
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let p_near = froxel.inv_view_proj * vec4<f32>(ndc_xy, 0.0, 1.0);
    let p_far = froxel.inv_view_proj * vec4<f32>(ndc_xy, 1.0, 1.0);
    let ray_origin = p_near.xyz / p_near.w;
    let ray_far = p_far.xyz / p_far.w;
    let ray_dir = normalize(ray_far - ray_origin);
    let world_pos = ray_origin + ray_dir * view_depth;

    // Haze density at this cell.
    let density = cloud_density(world_pos);

    // March a few steps toward the sun to get in-scattered radiance and how
    // much of the sun shaft survives to reach this cell.
    let sun_dir = normalize(froxel.sun_dir);
    let phase = 0.25 + 0.75 * pow(0.5 + 0.5 * dot(ray_dir, sun_dir), 2.0);
    const SUN_STEPS: u32 = 8u;
    let sstep = 4.0;
    var sun_trans = 1.0;
    var scattered = vec3<f32>(0.0);
    for (var i = 0u; i < SUN_STEPS; i = i + 1u) {
        let q = world_pos + sun_dir * f32(i) * sstep;
        let qd = cloud_density(q);
        let st = exp(-qd * sstep);
        scattered += sun_trans * froxel.sun_color * phase * (1.0 - st) * 0.35;
        sun_trans *= st;
    }
    scattered *= froxel.grid.w;

    // rgb = in-scattered sun radiance, a = extinction density (for view march).
    textureStore(t_froxel_out, gid, vec4<f32>(scattered, density));
}