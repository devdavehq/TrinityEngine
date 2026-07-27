// src/physics/particle_collision.rs
// ──────────────────────────────────────────────────────────────────────────────
// Per-particle collision system.
//
// Checks each particle against terrain height and optional object AABBs.
// On collision, particles bounce (reflect velocity with damping) or die.
//
// Integration:
//   Called from ParticleSystem::update() after physics step.
//   Uses terrain height lookup and optional collision volumes.
//
// Behavior:
//   - Terrain collision: particle.y < terrain_height → bounce or die
//   - Object collision: particle inside AABB → bounce
//   - Resting friction: slows horizontal velocity after ground contact
//   - Maximum bounces before death (default: 2)
// ──────────────────────────────────────────────────────────────────────────────

use glam::Vec3;
use crate::particles::Particle;

/// Collision result for a single particle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleCollisionResult {
    /// No collision — particle continues.
    None,
    /// Particle hit terrain/ground.
    Ground,
    /// Particle hit an object AABB.
    Object,
    /// Particle should die (too many bounces).
    Die,
}

/// Settings for particle collision behavior.
#[derive(Clone, Debug)]
pub struct ParticleCollisionConfig {
    /// How much velocity is retained after bounce (0 = no bounce, 1 = perfect bounce).
    pub bounce_damping: f32,
    /// How much horizontal velocity is reduced on ground contact.
    pub friction: f32,
    /// Maximum bounces before the particle dies.
    pub max_bounces: u8,
    /// Minimum velocity to trigger a bounce (below this, particle dies).
    pub min_bounce_velocity: f32,
    /// If true, particles rest on the ground after bouncing (no further physics).
    pub rest_on_ground: bool,
}

impl Default for ParticleCollisionConfig {
    fn default() -> Self {
        Self {
            bounce_damping: 0.4,
            friction: 0.8,
            max_bounces: 2,
            min_bounce_velocity: 1.0,
            rest_on_ground: true,
        }
    }
}

/// Per-particle collision state (stored alongside the particle pool).
#[derive(Clone, Debug, Default)]
pub struct ParticleCollisionState {
    pub bounces: u8,
    pub on_ground: bool,
}

/// AABB for object collision volumes.
#[derive(Clone, Copy, Debug)]
pub struct CollisionVolume {
    pub min: Vec3,
    pub max: Vec3,
}

impl CollisionVolume {
    pub fn from_aabb(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Check if a point is inside this AABB.
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x
            && point.y >= self.min.y && point.y <= self.max.y
            && point.z >= self.min.z && point.z <= self.max.z
    }

    /// Find the closest point on the AABB surface to a point.
    pub fn closest_point(&self, point: Vec3) -> Vec3 {
        Vec3::new(
            point.x.clamp(self.min.x, self.max.x),
            point.y.clamp(self.min.y, self.max.y),
            point.z.clamp(self.min.z, self.max.z),
        )
    }
}

/// Get terrain height at a world position.
/// This is a placeholder — wire this to your actual terrain height lookup.
fn get_terrain_height(_x: f32, _z: f32) -> f32 {
    // Default: flat ground at y=0
    // In production, sample from TerrainWorld/ChunkGrid
    0.0
}

/// Process collision for a single particle.
/// Returns the collision result and modifies velocity/position accordingly.
pub fn process_particle_collision(
    particle: &mut Particle,
    state: &mut ParticleCollisionState,
    config: &ParticleCollisionConfig,
    dt: f32,
    collision_volumes: &[CollisionVolume],
    terrain_height_fn: Option<fn(f32, f32) -> f32>,
) -> ParticleCollisionResult {
    // Already resting on ground
    if state.on_ground && config.rest_on_ground {
        // Zero out vertical velocity, apply friction to horizontal
        particle.velocity.x *= 1.0 - config.friction * dt * 10.0;
        particle.velocity.z *= 1.0 - config.friction * dt * 10.0;
        particle.velocity.y = 0.0;
        return ParticleCollisionResult::Ground;
    }

    // Check terrain collision
    let terrain_height = terrain_height_fn
        .map(|f| f(particle.position.x, particle.position.z))
        .unwrap_or_else(|| get_terrain_height(particle.position.x, particle.position.z));

    if particle.position.y <= terrain_height {
        // Ground collision
        if state.bounces >= config.max_bounces || particle.velocity.y.abs() < config.min_bounce_velocity {
            state.on_ground = true;
            particle.position.y = terrain_height;
            particle.velocity.y = 0.0;
            particle.velocity.x *= 1.0 - config.friction;
            particle.velocity.z *= 1.0 - config.friction;
            state.bounces += 1;
            return if state.bounces >= config.max_bounces {
                ParticleCollisionResult::Die
            } else {
                ParticleCollisionResult::Ground
            };
        }

        // Bounce
        particle.position.y = terrain_height;
        particle.velocity.y = -particle.velocity.y * config.bounce_damping;
        particle.velocity.x *= 1.0 - config.friction * 0.5;
        particle.velocity.z *= 1.0 - config.friction * 0.5;
        state.bounces += 1;

        // If bounce is too small, rest
        if particle.velocity.y.abs() < config.min_bounce_velocity * 0.5 {
            state.on_ground = true;
            particle.velocity.y = 0.0;
        }

        return ParticleCollisionResult::Ground;
    }

    // Check object collision volumes
    for vol in collision_volumes {
        if vol.contains(particle.position) {
            let closest = vol.closest_point(particle.position);
            let normal = (particle.position - closest).normalize_or_zero();

            if state.bounces >= config.max_bounces {
                return ParticleCollisionResult::Die;
            }

            // Reflect velocity
            let vel = particle.velocity;
            let dot = vel.dot(normal);
            if dot < 0.0 {
                particle.velocity = vel - normal * (2.0 * dot) * config.bounce_damping;
            }

            // Push particle out of volume
            particle.position = closest + normal * 0.01;
            state.bounces += 1;

            return ParticleCollisionResult::Object;
        }
    }

    ParticleCollisionResult::None
}

/// Process collision for a batch of particles.
/// Modifies particles and states in-place.
pub fn process_batch_collision(
    particles: &mut [Particle],
    states: &mut [ParticleCollisionState],
    config: &ParticleCollisionConfig,
    dt: f32,
    collision_volumes: &[CollisionVolume],
    terrain_height_fn: Option<fn(f32, f32) -> f32>,
) {
    for (particle, state) in particles.iter_mut().zip(states.iter_mut()) {
        process_particle_collision(particle, state, config, dt, collision_volumes, terrain_height_fn);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_particle_at_y(y: f32, vy: f32) -> Particle {
        Particle {
            position: Vec3::new(0.0, y, 0.0),
            velocity: Vec3::new(1.0, vy, 0.5),
            age: 0.0,
            lifetime: 10.0,
            size: 0.1,
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        }
    }

    #[test]
    fn no_collision_above_ground() {
        let mut p = make_particle_at_y(5.0, -2.0);
        let mut s = ParticleCollisionState::default();
        let config = ParticleCollisionConfig::default();
        let result = process_particle_collision(&mut p, &mut s, &config, 0.016, &[], None);
        assert_eq!(result, ParticleCollisionResult::None);
    }

    #[test]
    fn ground_collision_bounce() {
        let mut p = make_particle_at_y(0.0, -5.0); // At ground, moving down
        let mut s = ParticleCollisionState::default();
        let config = ParticleCollisionConfig { bounce_damping: 0.5, ..Default::default() };
        let result = process_particle_collision(&mut p, &mut s, &config, 0.016, &[], None);
        assert_eq!(result, ParticleCollisionResult::Ground);
        assert!(p.velocity.y > 0.0, "should bounce upward");
        assert_eq!(s.bounces, 1);
    }

    #[test]
    fn max_bounces_dies() {
        let mut p = make_particle_at_y(0.0, -10.0);
        let mut s = ParticleCollisionState { bounces: 2, on_ground: false };
        let config = ParticleCollisionConfig { max_bounces: 2, min_bounce_velocity: 0.0, ..Default::default() };
        let result = process_particle_collision(&mut p, &mut s, &config, 0.016, &[], None);
        assert_eq!(result, ParticleCollisionResult::Die);
    }

    #[test]
    fn aabb_collision() {
        let mut p = Particle {
            position: Vec3::new(0.3, 0.3, 0.3),
            velocity: Vec3::new(-5.0, 0.0, 0.0),
            age: 0.0, lifetime: 10.0, size: 0.1,
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        };
        let mut s = ParticleCollisionState::default();
        let config = ParticleCollisionConfig::default();
        let vol = CollisionVolume::from_aabb(Vec3::ZERO, Vec3::splat(0.5));

        let result = process_particle_collision(&mut p, &mut s, &config, 0.016, &[vol], None);
        assert_eq!(result, ParticleCollisionResult::Object);
    }

    #[test]
    fn aabb_contains() {
        let vol = CollisionVolume::from_aabb(Vec3::new(1.0, 1.0, 1.0), Vec3::splat(0.5));
        assert!(vol.contains(Vec3::new(1.0, 1.0, 1.0)));
        assert!(!vol.contains(Vec3::new(2.0, 1.0, 1.0)));
    }

    #[test]
    fn rest_on_ground() {
        let mut p = make_particle_at_y(0.0, -1.0);
        let mut s = ParticleCollisionState { bounces: 0, on_ground: false };
        let config = ParticleCollisionConfig { rest_on_ground: true, min_bounce_velocity: 2.0, bounce_damping: 0.1, ..Default::default() };
        let result = process_particle_collision(&mut p, &mut s, &config, 0.016, &[], None);
        assert_eq!(result, ParticleCollisionResult::Ground);
        assert!(s.on_ground, "should rest on ground");
        assert_eq!(p.velocity.y, 0.0);
    }

    #[test]
    fn batch_collision() {
        let mut particles = vec![
            make_particle_at_y(0.0, -3.0),
            make_particle_at_y(5.0, -1.0),
            make_particle_at_y(0.0, -10.0),
        ];
        let mut states = vec![
            ParticleCollisionState::default(),
            ParticleCollisionState::default(),
            ParticleCollisionState::default(),
        ];
        let config = ParticleCollisionConfig::default();
        process_batch_collision(&mut particles, &mut states, &config, 0.016, &[], None);
        assert_eq!(states[0].bounces, 1);
        assert_eq!(states[1].bounces, 0); // still above ground
    }
}
