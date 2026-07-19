// renderer/particle.wgsl
// GPU particle rendering via instanced quads.
//
// Each particle is a camera-facing billboard quad, drawn as one instance.
// The vertex shader offsets the quad by the particle position and scales by size.
// The fragment shader creates a soft circular falloff for a natural look.
// Velocity data drives streak elongation (rain streaks, not dots).
//
// BIND GROUPS:
//   group(0) = per-frame uniforms (camera VP matrix, camera position)

// ── Uniforms ─────────────────────────────────────────────────────────────────
struct CameraUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// ── Per-vertex data (the billboard quad) ─────────────────────────────────────
struct ParticleVertexIn {
    @location(0) quad_pos: vec2<f32>, // [-0.5..0.5] quad corners
}

// ── Per-instance data (one GpuParticle per instance) ────────────────────────
struct ParticleInstanceIn {
    @location(1) particle_pos: vec3<f32>,
    @location(2) particle_size: f32,
    @location(3) particle_color: vec4<f32>,
    @location(4) particle_vel: vec3<f32>,
    @location(5) _pad_inst: f32,
}

struct ParticleVertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,       // quad UV [0,1]
    @location(1) color: vec4<f32>,    // particle RGBA
    @location(2) dist_to_center: f32, // for soft circle
}

// ── Billboard vertex shader ─────────────────────────────────────────────────
// Builds a camera-facing quad. Velocity magnitude drives elongation:
// fast particles (rain) stretch along their velocity vector for a natural streak.
@vertex
fn vs_particle(in: ParticleVertexIn, instance: ParticleInstanceIn) -> ParticleVertOut {
    let to_cam = normalize(camera.camera_pos - instance.particle_pos);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(up, to_cam));
    let billboard_up = cross(to_cam, right);

    // Velocity-based elongation: fast particles stretch along their velocity.
    let speed = length(instance.particle_vel);
    // Elongation factor: 0 = no stretch (slow), up to 3x stretch for fast particles.
    // Threshold: only stretch above 2 m/s to avoid stretching stationary embers.
    let stretch = clamp((speed - 2.0) * 0.3, 0.0, 2.0);

    // Compute elongation axis: velocity direction projected onto billboard plane.
    var elongation = vec3<f32>(0.0);
    if speed > 0.5 {
        let vel_dir = instance.particle_vel / speed;
        // Project velocity onto billboard plane (remove component toward camera).
        elongation = vel_dir - to_cam * dot(vel_dir, to_cam);
        if length(elongation) > 0.001 {
            elongation = normalize(elongation) * stretch;
        } else {
            elongation = vec3<f32>(0.0);
        }
    }

    // Offset quad corners: stretch in the elongation direction, shrink perpendicular.
    // stretch_factor < 1 makes the quad thinner in the perpendicular direction (raindrop shape).
    let stretch_factor = 1.0 / (1.0 + stretch * 0.5);
    let world_pos = instance.particle_pos
        + right * in.quad_pos.x * instance.particle_size * stretch_factor
        + billboard_up * in.quad_pos.y * instance.particle_size
        + elongation * instance.particle_size * in.quad_pos.y; // stretch along velocity

    var out: ParticleVertOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = in.quad_pos + vec2<f32>(0.5);
    out.color = instance.particle_color;
    out.dist_to_center = length(in.quad_pos);
    return out;
}

// ── Fragment shader: soft circle / streak ────────────────────────────────────
// Creates smooth circular particles with soft edges.
// For stretched particles (rain), the fragment uses the UV to create an
// elongated falloff along the stretch axis.
@fragment
fn fs_particle(in: ParticleVertOut) -> @location(0) vec4<f32> {
    // Distance from center (0 = center, 1 = edge of quad diagonal).
    let d = in.dist_to_center * 1.414;

    // Soft circle: full opacity in the inner 40%, smooth falloff to 0 at the edge.
    let alpha = smoothstep(1.0, 0.3, d) * in.color.a;

    // Premultiply alpha for correct blending.
    return vec4<f32>(in.color.rgb * alpha, alpha);
}
