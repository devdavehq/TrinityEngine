// src/renderer/voxel_gi_mip.wgsl
// Real-time Global Illumination — voxel mip pyramid (compute pass).
//
// Downsamples the injected voxel grid (voxel_gi.wgsl level 0) into a summed
// mip pyramid. The cone tracer in deferred.wgsl picks the mip level from the
// cone footprint radius, so distant / wide cones sample blurred aggregate
// radiance in a single fetch — this is what makes cone tracing cheap.
//
// Run once per mip level (L = 1..7). The read view is bound to level L-1 and
// the write view to level L.

struct MipParams {
    // xyz = dimensions of the DESTINATION (current) level.
    dst_dims: vec4<f32>,
    _pad:     vec4<f32>,
}

@group(0) @binding(0) var<uniform> mp: MipParams;
@group(0) @binding(1) var voxel_in:  texture_3d<f32>;
@group(0) @binding(2) var voxel_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn cs_mip(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = vec3<u32>(mp.dst_dims.xyz);
    if any(gid >= dim) {
        return;
    }

    // Average the 8 child voxels from the previous level.
    let base = vec3<i32>(gid) * 2;
    var acc = vec4<f32>(0.0);
    var count = 0.0;
    for (var z = 0u; z < 2u; z = z + 1u) {
        for (var y = 0u; y < 2u; y = y + 1u) {
            for (var x = 0u; x < 2u; x = x + 1u) {
                let c = textureLoad(voxel_in, base + vec3<i32>(i32(x), i32(y), i32(z)), 0);
                acc += c;
                count += 1.0;
            }
        }
    }
    textureStore(voxel_out, vec3<i32>(gid), acc / count);
}