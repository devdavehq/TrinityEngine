// src/renderer/voxel_gi.wgsl
// Real-time Global Illumination — voxel injection (compute pass).
//
// The technique is the Decima-style hybrid from Horizon Forbidden West:
// dynamic one-bounce GI stored in a camera-aligned voxel clipmap, hybridized
// with the baked SH irradiance volumes. Since wgpu has no hardware ray
// tracing, the ray-traced component is replaced with Voxel Cone Tracing:
//
//   1. This pass "voxelizes" the visible scene by gathering the G-buffer:
//      every voxel in a 128³ grid projects to screen, and if it lies within
//      half a voxel of the visible surface shell it is stamped with the
//      PREVIOUS frame's fully-lit HDR scene radiance. The one-frame lag is
//      what turns the lighting into a feedback loop (bounce light that
//      re-bounces), which is exactly how real-time GI accumulates energy.
//   2. voxel_gi_mip.wgsl builds a summed mip pyramid of the grid.
//   3. deferred.wgsl cone-traces the pyramid for indirect diffuse (6 cones)
//      and indirect specular (1 cone).
//
// Gathering per voxel (instead of scattering per pixel) means no atomics and
// no clear pass — every voxel is written exactly once per frame.

struct VoxelUniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    // xyz = camera position.
    camera_pos:     vec4<f32>,
    // x = voxel size (world units per voxel).
    voxel_size:     vec4<f32>,
    // xyz = world-space corner of the grid (voxel (0,0,0)).
    grid_origin:    vec4<f32>,
    // xyz = grid dimensions (voxels per axis).
    grid_dims:      vec4<f32>,
    // xy = render target size in pixels.
    screen_size:    vec4<f32>,
}

@group(0) @binding(0) var<uniform> vparams: VoxelUniforms;
@group(0) @binding(1) var t_depth:  texture_depth_2d;
@group(0) @binding(2) var t_scene:  texture_2d<f32>;
@group(0) @binding(3) var s_scene:  sampler;
@group(0) @binding(4) var voxel_out: texture_storage_3d<rgba16float, write>;

// Reconstruct the world position of a pixel using the same (jittered) inverse
// view-projection the deferred lighting pass uses, so the voxel shell lines
// up exactly with the lit pixels we stamp into it.
fn reconstruct_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv * 2.0 - vec2<f32>(1.0), depth * 2.0 - 1.0);
    let clip = vec4<f32>(ndc, 1.0);
    let w = vparams.inv_view_proj * clip;
    return w.xyz / w.w;
}

@compute @workgroup_size(4, 4, 4)
fn cs_inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = u32(vparams.grid_dims.x);
    if any(gid >= vec3<u32>(dim)) {
        return;
    }

    // Voxel centre in world space.
    let voxel = vec3<f32>(gid) + vec3<f32>(0.5);
    let pos = vparams.grid_origin.xyz + voxel * vparams.voxel_size.x;

    // Project to screen space.
    let clip = vparams.view_proj * vec4<f32>(pos, 1.0);
    if clip.w <= 0.0001 {
        textureStore(voxel_out, vec3<i32>(gid), vec4<f32>(0.0));
        return;
    }
    let ndc = clip.xyz / clip.w;
    if ndc.z < 0.0 || ndc.z > 1.0 {
        textureStore(voxel_out, vec3<i32>(gid), vec4<f32>(0.0));
        return;
    }
    let uv = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        textureStore(voxel_out, vec3<i32>(gid), vec4<f32>(0.0));
        return;
    }

    // Depth at this pixel — sky pixels carry no geometry to bounce.
    let depth = textureLoad(t_depth, vec2<i32>(uv * vparams.screen_size.xy), 0);
    if depth >= 0.9998 {
        textureStore(voxel_out, vec3<i32>(gid), vec4<f32>(0.0));
        return;
    }

    // Reconstruct the visible surface point at this pixel.
    let surf = reconstruct_world(uv, depth);

    // Is this voxel on the visible surface shell (within half a voxel along
    // the view axis)? If so, stamp it with the previous frame's lit radiance.
    let view_dir = normalize(surf - vparams.camera_pos.xyz);
    let aligned = dot(pos - surf, view_dir);
    if abs(aligned) < vparams.voxel_size.x * 0.5 {
        let rad = textureSampleLevel(t_scene, s_scene, uv, 0.0).rgb;
        textureStore(voxel_out, vec3<i32>(gid), vec4<f32>(rad, 1.0));
    } else {
        textureStore(voxel_out, vec3<i32>(gid), vec4<f32>(0.0));
    }
}