// src/renderer/sky.wgsl
// Procedural sky rendering shader — AAA quality with volumetric clouds.
// Renders a fullscreen triangle and computes sky color from environment uniforms.
// Features: gradient sky (zenith→horizon→ground), sun disc, moon, stars,
//           Rayleigh/Mie atmospheric scattering, volumetric cloud ray marching,
//           storm cloud darkening, and lightning flash support.
//
// ── How it works ─────────────────────────────────────────────────────────────
// 1. Generate a fullscreen triangle from vertex_index (no vertex buffer needed).
// 2. Reconstruct world-space ray direction from clip-space position using
//    the inverse view-projection matrix.
// 3. Compute sky color based on the ray's elevation angle:
//    - Gradient between zenith, horizon, and ground colors
//    - Sun disc + glow based on angle to sun direction
//    - Atmospheric scattering (Rayleigh for blue sky, Mie for haze)
//    - Star field at night (hash-based procedural stars)
//    - Volumetric cloud layer via ray marching through 3D noise
//    - Storm darkening and lightning flash
// 4. Output HDR color (no tone mapping — the main pass handles that).

// ── Uniform buffer ──────────────────────────────────────────────────────────
// Matches SkyUniforms in sky.rs. Total: 304 bytes.
struct SkyUniforms {
    // Sky gradient colors (from environment::sky)
    sky_zenith:       vec4<f32>,  // rgb = zenith color (top of sky)
    sky_horizon:      vec4<f32>,  // rgb = horizon color
    sky_ground:       vec4<f32>,  // rgb = ground color (below horizon)
    sky_sun_dir:      vec4<f32>,  // xyz = sun direction, w = unused
    sky_sun_color:    vec4<f32>,  // rgb = sun color, w = sun intensity
    sky_moon_dir:     vec4<f32>,  // xyz = moon direction, w = moon intensity
    sky_atmosphere:   vec4<f32>,  // x = rayleigh_density, y = mie_scatter, z = mie_density, w = mie_direction
    sky_stars:        vec4<f32>,  // x = star_intensity, y = star_density, z = sun_disc_radius_deg, w = sun_halo_falloff

    // Cloud parameters (from environment::clouds)
    cloud_params:     vec4<f32>,  // x = coverage, y = base_altitude, z = thickness, w = type_scale
    cloud_noise:      vec4<f32>,  // x = noise_scale, y = density_threshold, z = density_smoothness, w = precipitation
    cloud_scroll:     vec4<f32>,  // xy = uv_offset, z = speed, w = unused
    cloud_type:       vec4<f32>,  // x = cloud type (0=none, 1=cirrus, 2=stratus, 3=cumulus), y = storm_darken, z = lightning_intensity

    // Camera / screen
    inv_view_proj:    mat4x4<f32>,
    camera_pos_time:  vec4<f32>,   // xyz = camera position, w = total elapsed time
    screen_fog:       vec4<f32>,   // x = screen width, y = screen height, z = fog_density, w = unused
    fog_color:        vec4<f32>,   // rgb = fog color, w = unused
}

@group(0) @binding(0) var<uniform> sky: SkyUniforms;
// Cloud history texture: previous frame's cloud result for temporal reprojection.
@group(0) @binding(1) var t_cloud_history: texture_2d<f32>;
@group(0) @binding(2) var s_cloud_history: sampler;

// ── Fullscreen triangle generation ──────────────────────────────────────────
struct SkyVertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) vertex_index: u32) -> SkyVertOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    var out: SkyVertOut;
    out.clip_pos = vec4<f32>(pos[vertex_index], 1.0, 1.0);
    out.uv = pos[vertex_index] * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

// ── Utility functions ────────────────────────────────────────────────────────

fn hash31(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453);
}

fn hash21(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

// 3D value noise for volumetric clouds — 8-cube interpolation for temporal coherence.
fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f); // smoothstep interpolation

    let a = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let b = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let ff = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let g = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let h2 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

    return mix(
        mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
        mix(mix(e, ff, u.x), mix(g, h2, u.x), u.y),
        u.z,
    );
}

// Fractional Brownian Motion — 6 octaves for detailed cloud shapes.
// Each octave doubles frequency and halves amplitude.
fn fbm6(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    for (var i = 0u; i < 6u; i++) {
        value += amplitude * noise3d(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// ── Procedural sky gradient ──────────────────────────────────────────────────
fn sky_gradient(ray_dir: vec3<f32>) -> vec3<f32> {
    let t = ray_dir.y;
    if t > 0.0 {
        let blend = pow(clamp(t, 0.0, 1.0), 0.5);
        return mix(sky.sky_horizon.rgb, sky.sky_zenith.rgb, blend);
    } else {
        let blend = pow(clamp(-t, 0.0, 1.0), 0.4);
        return mix(sky.sky_horizon.rgb, sky.sky_ground.rgb, blend);
    }
}

// ── Atmospheric scattering ──────────────────────────────────────────────────
fn atmospheric_scattering(ray_dir: vec3<f32>) -> vec3<f32> {
    let sun_dir = normalize(sky.sky_sun_dir.xyz);
    let rayleigh_density = sky.sky_atmosphere.x;
    let mie_scatter = sky.sky_atmosphere.y;
    let mie_direction = sky.sky_atmosphere.w;

    let sun_elevation = max(sun_dir.y, 0.0);
    let cos_theta = dot(ray_dir, sun_dir);
    let rayleigh_phase = 3.0 / (16.0 * 3.14159265) * (1.0 + cos_theta * cos_theta);
    let g2 = mie_direction * mie_direction;
    let mie_phase = (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * mie_direction * cos_theta, 1.5));
    let scatter_intensity = pow(sun_elevation, 0.4);

    let rayleigh_color = vec3<f32>(0.15, 0.35, 0.75) * rayleigh_phase * rayleigh_density * scatter_intensity;
    let mie_color = vec3<f32>(0.8, 0.6, 0.3) * mie_phase * mie_scatter * scatter_intensity;

    return rayleigh_color + mie_color;
}

// ── Sun disc and glow ───────────────────────────────────────────────────────
fn sun_disc(ray_dir: vec3<f32>) -> vec3<f32> {
    let sun_dir = normalize(sky.sky_sun_dir.xyz);
    let cos_angle = dot(ray_dir, sun_dir);
    let radius_cos = cos(sky.sky_stars.z * 3.14159265 / 180.0);
    let halo_cos = cos(sky.sky_stars.z * 3.14159265 / 180.0 * sky.sky_stars.w);
    let disc = smoothstep(radius_cos, radius_cos + 0.001, cos_angle);
    let glow = pow(clamp((cos_angle - halo_cos) / (1.0 - halo_cos), 0.0, 1.0), 2.0);
    let sun_rgb = sky.sky_sun_color.rgb * sky.sky_sun_color.w;
    return sun_rgb * (disc + glow * 0.4);
}

// ── Moon ─────────────────────────────────────────────────────────────────────
fn moon_render(ray_dir: vec3<f32>) -> vec3<f32> {
    let moon_dir = normalize(sky.sky_moon_dir.xyz);
    let moon_intensity = sky.sky_moon_dir.w;
    if moon_intensity < 0.01 { return vec3<f32>(0.0); }
    let cos_angle = dot(ray_dir, moon_dir);
    let disc = smoothstep(0.999, 0.9995, cos_angle);
    let glow = pow(clamp(cos_angle, 0.0, 1.0), 32.0) * 0.3;
    let moon_color = vec3<f32>(0.8, 0.85, 1.0) * moon_intensity;
    return moon_color * (disc + glow);
}

// ── Stars ────────────────────────────────────────────────────────────────────
fn star_field(ray_dir: vec3<f32>) -> f32 {
    let intensity = sky.sky_stars.x;
    if intensity < 0.01 { return 0.0; }
    let density = sky.sky_stars.y;
    var star_val = 0.0;

    let cell_large = floor(ray_dir.xz * density * 30.0);
    let h_large = hash21(cell_large);
    if h_large > 0.97 {
        let center = (cell_large + 0.5) / (density * 30.0);
        let dist = length(ray_dir.xz - center);
        star_val += smoothstep(0.02, 0.0, dist) * 1.5;
    }
    let cell_med = floor(ray_dir.xz * density * 60.0);
    let h_med = hash21(cell_med);
    if h_med > 0.95 {
        let center = (cell_med + 0.5) / (density * 60.0);
        let dist = length(ray_dir.xz - center);
        star_val += smoothstep(0.015, 0.0, dist) * 0.8;
    }
    let cell_small = floor(ray_dir.xz * density * 120.0);
    let h_small = hash21(cell_small);
    if h_small > 0.92 {
        let center = (cell_small + 0.5) / (density * 120.0);
        let dist = length(ray_dir.xz - center);
        star_val += smoothstep(0.01, 0.0, dist) * 0.4;
    }
    let twinkle = sin(hash21(floor(ray_dir.xz * density * 50.0)) * 6.28 + sky.camera_pos_time.w * 2.0) * 0.3 + 0.7;
    return star_val * intensity * twinkle;
}

// ── Volumetric cloud layer ───────────────────────────────────────────────────
// Ray marches through a 3D cloud volume for realistic volumetric clouds.
// Uses 6-octave FBM noise for detailed cloud shapes with smooth edges.
// Storm darkening and internal scattering are computed per-sample.
//
// How it works:
// 1. Ray intersects the cloud volume (base altitude to top = base + thickness).
// 2. Step along the ray through the volume (32 steps for performance).
// 3. At each step, sample 3D FBM noise to determine cloud density.
// 4. Accumulate density along the ray (Beer-Lambert absorption).
// 5. Apply sun lighting with darkening toward cloud base (internal scattering).
// 6. Storm darkening reduces cloud brightness; lightning flash adds a pulse.
fn cloud_layer(ray_dir: vec3<f32>) -> vec4<f32> {
    // Returns vec4: rgb = cloud color, a = cloud density (for blending).
    if sky.cloud_type.x < 0.5 || sky.cloud_params.x < 0.01 {
        return vec4<f32>(0.0);
    }
    if ray_dir.y < 0.01 {
        return vec4<f32>(0.0);
    }

    let coverage = sky.cloud_params.x;
    let base_alt = sky.cloud_params.y;
    let thickness = sky.cloud_params.z;
    let type_scale = sky.cloud_params.w;
    let noise_scale = sky.cloud_noise.x;
    let threshold = sky.cloud_noise.y;
    let smoothness = sky.cloud_noise.z;
    let storm_darken = sky.cloud_type.y;
    let lightning = sky.cloud_type.z;
    let scroll = sky.cloud_scroll.xy;
    let speed = sky.cloud_scroll.z;
    let time = sky.camera_pos_time.w;

    // Intersect ray with cloud volume (bottom and top planes).
    let t_bottom = base_alt / max(ray_dir.y, 0.001);
    let t_top = (base_alt + thickness) / max(ray_dir.y, 0.001);

    // Only march if ray hits the volume.
    if t_bottom < 0.0 { return vec4<f32>(0.0); }

    // March parameters: 32 steps for quality/performance balance.
    let march_end = t_top;
    let march_dist = march_end - max(t_bottom, 0.0);
    let step_count = 32u;
    let step_size = march_dist / f32(step_count);

    // Temporal jitter: offset march start by a per-frame random amount to break
    // temporal aliasing. Combined with TAA, this gives smooth clouds without
    // flicker — the same technique used by UE4/5 and Horizon Zero Dawn.
    let jitter = hash31(floor(ray_dir * 100.0) + vec3<f32>(0.0, 0.0, sky.camera_pos_time.w * 7.13)) * 2.0 - 1.0;
    let march_start = max(t_bottom, 0.0) + jitter * step_size * 0.5;

    // Camera position for 3D noise coherence.
    let cam_pos = sky.camera_pos_time.xyz;

    var accumulated_density = 0.0;
    var accumulated_color = vec3<f32>(0.0);
    var transmittance = 1.0;

    let sun_dir = normalize(sky.sky_sun_dir.xyz);

    for (var i = 0u; i < 32u; i++) {
        if transmittance < 0.01 { break; }

        let t = march_start + f32(i) * step_size;
        let sample_pos = ray_dir * t;

        // 3D world position for noise (add wind scroll).
        let noise_pos = vec3<f32>(
            (sample_pos.x + scroll.x * 100.0) * noise_scale,
            sample_pos.y * noise_scale * 0.5,
            (sample_pos.z + scroll.y * 100.0) * noise_scale,
        );

        // 6-octave FBM for detailed cloud shapes.
        var density = fbm6(noise_pos);

        // Apply coverage and threshold.
        density = smoothstep(threshold - smoothness, threshold + smoothness, density);
        density *= coverage * type_scale;

        // Height gradient: denser at middle of cloud volume, fades at edges.
        let height_ratio = (sample_pos.y - base_alt) / thickness;
        let height_gradient = smoothstep(0.0, 0.2, height_ratio) * smoothstep(1.0, 0.6, height_ratio);
        density *= height_gradient;

        if density > 0.001 {
            // Beer-Lambert: light attenuates exponentially through the cloud.
            let step_absorption = exp(-density * step_size * 0.03);

            // Cloud lighting: sun-facing side bright, base darkened (internal scattering).
            let sun_wrap = max(dot(normalize(sun_dir + vec3<f32>(0.0, 0.3, 0.0)), vec3<f32>(0.0, 1.0, 0.0)), 0.0);
            let light_intensity = mix(0.4, 1.2, sun_wrap);

            // Storm darkening: storms darken cloud bases more.
            let darken = mix(1.0, 0.4, storm_darken * (1.0 - height_ratio));

            // Base color: white at day, gray at night.
            let daylight = sky.sky_sun_color.w;
            let cloud_day = vec3<f32>(1.0, 1.0, 1.0);
            let cloud_night = vec3<f32>(0.15, 0.17, 0.22);
            let base_color = mix(cloud_night, cloud_day, clamp(daylight, 0.0, 1.0));

            let sample_color = base_color * light_intensity * darken;

            // Integrate: density × transmittance gives contribution.
            let contribution = density * transmittance;
            accumulated_color += sample_color * contribution;
            accumulated_density += contribution;

            // Update transmittance (Beer-Lambert absorption).
            transmittance *= step_absorption;
        }
    }

    // Lightning flash: uniform bright pulse through all cloud samples.
    if lightning > 0.01 {
        accumulated_color += vec3<f32>(0.9, 0.95, 1.0) * lightning * 3.0 * transmittance;
        accumulated_density = max(accumulated_density, lightning * 0.5);
    }

    return vec4<f32>(accumulated_color, clamp(accumulated_density, 0.0, 0.95));
}

// ── Fragment shader ──────────────────────────────────────────────────────────
// Temporal reprojection for clouds: reprojects the previous frame's cloud
// result using the previous VP matrix, then blends with the current frame
// based on motion magnitude. This eliminates cloud flicker at low step counts.
@fragment
fn fs_sky(in: SkyVertOut) -> (@location(0) vec4<f32>, @location(1) vec4<f32>, @location(2) vec4<f32>) {
    let ndc = vec4<f32>(in.uv * 2.0 - 1.0, 1.0, 1.0);
    let world_pos_h = sky.inv_view_proj * ndc;
    let ray_dir = normalize(world_pos_h.xyz / world_pos_h.w);

    var color = sky_gradient(ray_dir);
    color += atmospheric_scattering(ray_dir);
    color += sun_disc(ray_dir);
    color += moon_render(ray_dir);
    let stars = star_field(ray_dir);
    color += vec3<f32>(stars);

    // ── Volumetric cloud layer ─────────────────────────────────────────────
    let cloud = cloud_layer(ray_dir);

    // ── Temporal reprojection blend ────────────────────────────────────────
    // Reproject current pixel's world position to previous frame's screen UV.
    let prev_clip = sky.prev_view_proj * vec4<f32>(world_pos_h.xyz, 1.0);
    let prev_uv = prev_clip.xy / prev_clip.w * 0.5 + vec2<f32>(0.5, 0.5);
    let history = textureSample(t_cloud_history, s_cloud_history, prev_uv);

    // Motion magnitude (in UV space) drives the blend factor.
    let motion = length(in.uv - prev_uv);
    // Stationary pixels trust history more (0.15 blend = 85% history).
    // Moving pixels trust current more (0.8 blend = 80% current).
    let temporal_blend = mix(0.15, 0.8, clamp(motion * 30.0, 0.0, 1.0));

    // Blend cloud alpha and color separately for stability.
    let blended_alpha = mix(history.a, cloud.a, temporal_blend);
    let blended_color = mix(history.rgb, cloud.rgb,
        select(temporal_blend, 1.0, blended_alpha < 0.01));

    // Blend volumetric clouds with sky using the temporally-smoothed alpha.
    color = mix(color, blended_color, blended_alpha);

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(100.0));
    return (vec4<f32>(color, 1.0), vec4<f32>(ray_dir * 0.5 + vec3<f32>(0.5), 1.0), vec4<f32>(blended_color, blended_alpha));
}
