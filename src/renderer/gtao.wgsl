// src/renderer/gtao.wgsl
// Ground-Truth Ambient Occlusion (GTAO) — compute pass.
//
// Reads the depth buffer + world-space normal G-buffer, ray-marches a small
// number of directions in the tangent plane around each normal and finds the
// highest occluder horizon, then writes an AO mask (1.0 = fully lit, 0.0 =
// fully occluded) into a half-res texture that the deferred lighting pass
// samples. This replaces the old per-pixel "horizon term" SSAO hack with a
// real screen-space occlusion solve.

struct GtaoParams {
    // VP + inverse VP so we can round-trip world <-> screen in the march.
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // xyz = camera position, w = occlusion radius (world units).
    camera_pos_radius: vec4<f32>,
    // xy = render target size, zw = 1/size (UV step per pixel).
    screen_size:   vec4<f32>,
    // x = strength (0..1), y = intensity slider, z = unused, w = time (noise).
    params:        vec4<f32>,
}

@group(0) @binding(0) var<uniform> gtao: GtaoParams;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var t_normal: texture_2d<f32>;
@group(0) @binding(3) var ao_out: texture_storage_2d<rgba8unorm, write>;

const DIRECTIONS: u32 = 8u;
const STEPS: u32 = 6u;

fn hash2(uv: vec2<f32>) -> f32 {
    let s = sin(dot(uv, vec2<f32>(127.1, 311.7)) + gtao.params.w * 0.01) * 43758.5453;
    return s - floor(s);
}

// Reconstruct the world position of a pixel from its depth value.
fn reconstruct_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
    let clip = vec4<f32>(ndc, 1.0);
    let w = gtao.inv_view_proj * clip;
    return w.xyz / w.w;
}

// Project a world position back to UV space (for depth sampling).
fn world_to_uv(p: vec3<f32>) -> vec2<f32> {
    let clip = gtao.view_proj * vec4<f32>(p, 1.0);
    if clip.w <= 0.0001 {
        return vec2<f32>(-1.0, -1.0);
    }
    let ndc = clip.xyz / clip.w;
    return vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = gtao.screen_size.xy;
    if (any(vec2<f32>(gid.xy) >= size)) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * gtao.screen_size.zw;

    let depth = textureLoad(t_depth, vec2<i32>(gid.xy), 0);
    if (depth >= 1.0) {
        // Sky — no geometry to occlude.
        textureStore(ao_out, vec2<i32>(gid.xy), vec4<f32>(1.0));
        return;
    }

    let P = reconstruct_world(uv, depth);
    let N = normalize(textureLoad(t_normal, vec2<i32>(gid.xy), 0).xyz * 2.0 - 1.0);

    // Build an orthonormal tangent basis around the surface normal.
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(N.y) > 0.99);
    let T = normalize(cross(up, N));
    let B = cross(N, T);

    let radius = gtao.camera_pos_radius.w;
    let noise = hash2(vec2<f32>(f32(gid.x), f32(gid.y)));

    var total: f32 = 0.0;
    for (var d: u32 = 0u; d < DIRECTIONS; d = d + 1u) {
        let theta = (f32(d) + noise) * (6.2831853 / f32(DIRECTIONS));
        let dir = normalize(T * cos(theta) + B * sin(theta));

        // March outward from the pixel, tracking the highest occluder horizon.
        var horizon_sin: f32 = 0.0;
        for (var s: u32 = 1u; s <= STEPS; s = s + 1u) {
            let t = f32(s) / f32(STEPS);
            let Q = P + dir * (t * radius);
            let q_uv = world_to_uv(Q);
            if (any(q_uv < vec2<f32>(0.0)) || any(q_uv > vec2<f32>(1.0))) {
                continue;
            }
            let q_depth = textureLoad(t_depth, vec2<i32>(q_uv * size), 0);
            if (q_depth >= 1.0) {
                continue;
            }
            let R = reconstruct_world(q_uv, q_depth);
            let to_r = R - P;
            let dist = length(to_r);
            if (dist < 0.001) {
                continue;
            }
            // Elevation of this occluder above the tangent plane.
            let elev = dot(N, to_r / dist);
            horizon_sin = max(horizon_sin, elev);
        }
        // Clamp to the visible hemisphere and integrate the visible sky wedge.
        horizon_sin = clamp(horizon_sin, 0.0, 1.0);
        // Visibility ≈ sqrt(1 - h²) integrated over the wedge (simplified GTAO).
        let vis = sqrt(1.0 - horizon_sin * horizon_sin);
        total += vis;
    }

    let ao = total / f32(DIRECTIONS);
    // Strength / intensity shaping: raise AO (darkening) as strength grows.
    let s = clamp(gtao.params.x, 0.0, 1.0);
    let occlusion = 1.0 - ao;
    let out_ao = 1.0 - occlusion * s * gtao.params.y;
    textureStore(ao_out, vec2<i32>(gid.xy), vec4<f32>(clamp(out_ao, 0.0, 1.0)));
}
