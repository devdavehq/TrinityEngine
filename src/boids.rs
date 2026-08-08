// src/boids.rs
// ──────────────────────────────────────────────────────────────────────────────
// Lightweight flocking / boids system for ambient creatures (birds, fish,
// herds).  Pure CPU, no dependencies beyond glam.
//
// Classic Reynolds steering with three rules:
//   separation  — avoid crowding neighbours
//   alignment   — match neighbour velocity
//   cohesion    — steer toward the flock centroid
// plus optional per-agent wander and goal seeking so herds drift naturally.
//
// The engine stores one Flock per group (e.g. "birds", "deer").  The system
// updates positions each frame and returns them for ECS/render consumption.
// ──────────────────────────────────────────────────────────────────────────────

use glam::Vec3;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Boid {
    pub position: Vec3,
    pub velocity: Vec3,
}

/// Per-agent steering weights + a small per-agent RNG stream (seeded) so
/// behaviour differs slightly between members of a flock.
#[derive(Clone, Debug)]
pub struct FlockParams {
    pub separation_weight: f32,
    pub alignment_weight: f32,
    pub cohesion_weight: f32,
    pub wander_weight: f32,
    pub perception_radius: f32,
    pub max_speed: f32,
    pub max_steer: f32,
    /// World-space bounds the flock stays inside (min_x, min_y, min_z, max_x, max_y, max_z).
    pub bounds: Option<[f32; 6]>,
    /// Optional goal point the whole flock gently seeks.
    pub goal: Option<Vec3>,
    pub goal_weight: f32,
}

impl Default for FlockParams {
    fn default() -> Self {
        Self {
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            wander_weight: 0.3,
            perception_radius: 6.0,
            max_speed: 4.0,
            max_steer: 1.5,
            bounds: None,
            goal: None,
            goal_weight: 0.4,
        }
    }
}

/// A flock of boids.  `rng` is a per-flock seed for deterministic wander.
#[derive(Clone, Debug)]
pub struct Flock {
    pub boids: Vec<Boid>,
    pub params: FlockParams,
    seed: u64,
}

impl Flock {
    pub fn new(seed: u64, params: FlockParams) -> Self {
        Self { boids: Vec::new(), params, seed }
    }

    /// Add a boid at a position with an optional initial velocity.
    pub fn add(&mut self, position: Vec3, velocity: Vec3) {
        self.boids.push(Boid { position, velocity });
    }

    /// Advance the flock by `dt`.  Updates all boid positions in place.
    pub fn update(&mut self, dt: f32) {
        let n = self.boids.len();
        if n == 0 {
            return;
        }
        // Neighbourhood is computed once per boid per frame (O(n²) — fine for
        // ambient flock sizes up to a few hundred).
        for i in 0..n {
            let me = self.boids[i];
            let mut sep = Vec3::ZERO;
            let mut align = Vec3::ZERO;
            let mut coh = Vec3::ZERO;
            let mut count = 0usize;
            let r2 = self.params.perception_radius * self.params.perception_radius;

            for j in 0..n {
                if j == i {
                    continue;
                }
                let d = self.boids[j].position - me.position;
                if d.length_squared() < r2 {
                    // Separation: push away, scaled by inverse distance.
                    let dist = d.length().max(0.001);
                    sep -= d / dist / dist;
                    align += self.boids[j].velocity;
                    coh += self.boids[j].position;
                    count += 1;
                }
            }

            let mut steer = Vec3::ZERO;
            if count > 0 {
                align /= count as f32;
                coh /= count as f32;
                let desired = align.normalize_or_zero() * self.params.max_speed;
                steer += (desired - me.velocity) * self.params.alignment_weight;

                let to_center = coh - me.position;
                let desired = to_center.normalize_or_zero() * self.params.max_speed;
                steer += (desired - me.velocity) * self.params.cohesion_weight;

                let desired = sep.normalize_or_zero() * self.params.max_speed;
                steer += (desired - me.velocity) * self.params.separation_weight;
            }

            // Wander: add a small deterministic random jitter.
            self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let jitter = Vec3::new(
                ((self.seed >> 33) as f32) / u64::MAX as f32 * 2.0 - 1.0,
                ((self.seed >> 17) as f32) / u64::MAX as f32 * 2.0 - 1.0,
                (self.seed as f32) / u64::MAX as f32 * 2.0 - 1.0,
            );
            steer += jitter * self.params.wander_weight;

            // Goal seeking.
            if let Some(g) = self.params.goal {
                let to_goal = g - me.position;
                let desired = to_goal.normalize_or_zero() * self.params.max_speed;
                steer += (desired - me.velocity) * self.params.goal_weight;
            }

            // Apply steering with max_steer clamping, then integrate.
            let steer_len = steer.length();
            let steer = if steer_len > self.params.max_steer {
                steer / steer_len * self.params.max_steer
            } else {
                steer
            };
            let mut vel = me.velocity + steer * dt;
            let speed = vel.length();
            if speed > self.params.max_speed {
                vel = vel / speed * self.params.max_speed;
            }
            let pos = me.position + vel * dt;

            // Keep within bounds.
            let (pos, vel) = if let Some(b) = self.params.bounds {
                let mut p = pos;
                let mut v = vel;
                if p.x < b[0] { p.x = b[0]; v.x = v.x.abs(); }
                if p.x > b[3] { p.x = b[3]; v.x = -v.x.abs(); }
                if p.y < b[1] { p.y = b[1]; v.y = v.y.abs(); }
                if p.y > b[4] { p.y = b[4]; v.y = -v.y.abs(); }
                if p.z < b[2] { p.z = b[2]; v.z = v.z.abs(); }
                if p.z > b[5] { p.z = b[5]; v.z = -v.z.abs(); }
                (p, v)
            } else {
                (pos, vel)
            };

            self.boids[i] = Boid { position: pos, velocity: vel };
        }
    }

    pub fn len(&self) -> usize {
        self.boids.len()
    }
}

// ── BoidRegistry ─────────────────────────────────────────────────────────────
// Holds one Flock per named group ("birds", "fish", "deer").  The game loop
// calls update_boids() each frame; boids_system() then writes each member's
// position into a matching ECS entity so flocks render and react to the world.

/// Registry of named flocks.  Named groups are the unit of authoring:
/// `boids.*` Lua API and ambient-creature systems address a group by name.
#[derive(Clone, Debug)]
pub struct BoidRegistry {
    /// group name → flock
    pub groups: HashMap<String, Flock>,
    /// Gaps a fresh RNG seed for newly created groups.
    next_seed: u64,
}

impl BoidRegistry {
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            next_seed: 1,
        }
    }

    /// Ensure a group exists (creating it empty if not).  Returns true if it
    /// was freshly created.
    pub fn ensure_group(&mut self, name: &str) -> bool {
        if self.groups.contains_key(name) {
            return false;
        }
        let seed = self.next_seed;
        self.next_seed = self.next_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.groups
            .insert(name.to_string(), Flock::new(seed, FlockParams::default()));
        true
    }

    pub fn group(&self, name: &str) -> Option<&Flock> {
        self.groups.get(name)
    }

    pub fn group_mut(&mut self, name: &str) -> Option<&mut Flock> {
        self.groups.get_mut(name)
    }

    /// Append a boid to a group, creating the group if needed.
    pub fn add_boid(&mut self, name: &str, position: Vec3, velocity: Vec3) {
        self.ensure_group(name);
        if let Some(g) = self.groups.get_mut(name) {
            g.add(position, velocity);
        }
    }

    /// Remove a group and return the number of boids it held.
    pub fn remove_group(&mut self, name: &str) -> usize {
        self.groups.remove(name).map_or(0, |g| g.boids.len())
    }

    /// Clear all boids in a group (keep the group).
    pub fn clear_group(&mut self, name: &str) {
        if let Some(g) = self.groups.get_mut(name) {
            g.boids.clear();
        }
    }

    /// Advance every enabled group by `dt`.
    pub fn update(&mut self, dt: f32) {
        for group in self.groups.values_mut() {
            group.update(dt);
        }
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Total boids across all groups.
    pub fn total_boids(&self) -> usize {
        self.groups.values().map(|g| g.boids.len()).sum()
    }
}

impl Default for BoidRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// The ECS hook component that ties a flock member to an entity.
// Add `BoidMember { group, index }` to an entity to have boids_system drive its
// transform from the flock's simulation.
pub struct BoidMember {
    pub group: String,
    pub index: usize,
}

/// Run the flocking simulation and sync flock member positions into the ECS
/// world for entities carrying a `BoidMember` component.
pub fn boids_system(world: &mut hecs::World, registry: &mut BoidRegistry, dt: f32) {
    registry.update(dt);

    // Collect (entity, group, index) first to satisfy the borrow checker.
    let members: Vec<(hecs::Entity, String, usize)> = world
        .query::<(hecs::Entity, &BoidMember)>()
        .iter()
        .map(|(e, b)| (e, b.group.clone(), b.index))
        .collect();

    for (entity, group, index) in members {
        let Some(flock) = registry.group(&group) else {
            continue;
        };
        let Some(boid) = flock.boids.get(index) else {
            continue;
        };
        if let Ok(mut pos) = world.get::<&mut crate::components::Position>(entity) {
            pos.x = boid.position.x;
            pos.y = boid.position.y;
            pos.z = boid.position.z;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boids_move_without_collapsing() {
        let mut flock = Flock::new(42, FlockParams::default());
        for i in 0..5 {
            flock.add(Vec3::new(i as f32, 2.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        }
        let before = flock.boids[0].position;
        flock.update(0.016);
        let after = flock.boids[0].position;
        assert_ne!(before, after, "boid should have moved");
    }

    #[test]
    fn stays_inside_bounds() {
        let mut params = FlockParams::default();
        params.bounds = Some([-10.0, 0.0, -10.0, 10.0, 10.0, 10.0]);
        let mut flock = Flock::new(7, params);
        flock.add(Vec3::new(0.0, 5.0, 0.0), Vec3::new(50.0, 0.0, 0.0));
        for _ in 0..100 {
            flock.update(0.016);
        }
        let p = flock.boids[0].position;
        assert!(p.x >= -10.0 && p.x <= 10.0, "out of bounds x: {}", p.x);
    }

    #[test]
    fn registry_manages_named_groups() {
        let mut reg = BoidRegistry::new();
        assert!(reg.ensure_group("birds"));
        assert!(!reg.ensure_group("birds"), "existing group returns false");

        reg.add_boid("birds", Vec3::new(0.0, 3.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        reg.add_boid("birds", Vec3::new(1.0, 3.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(reg.group("birds").unwrap().len(), 2);
        assert_eq!(reg.total_boids(), 2);

        reg.update(0.016);
        // Boids should have moved from their start position.
        let p = reg.group("birds").unwrap().boids[0].position;
        assert!(p != Vec3::new(0.0, 3.0, 0.0));

        reg.clear_group("birds");
        assert_eq!(reg.group("birds").unwrap().len(), 0);

        assert_eq!(reg.remove_group("birds"), 0);
        assert!(reg.group("birds").is_none());
    }

    #[test]
    fn boids_system_syncs_ecs_positions() {
        use crate::components::Position;

        let mut reg = BoidRegistry::new();
        reg.add_boid("fish", Vec3::new(0.0, 1.0, 0.0), Vec3::new(2.0, 0.0, 0.0));

        let mut world = hecs::World::new();
        // Harvest the ECS component from the module scope (kept separate from Flock::Boid).
        world.spawn((
            Position { x: 0.0, y: 0.0, z: 0.0 },
            super::BoidMember { group: "fish".to_string(), index: 0 },
        ));

        boids_system(&mut world, &mut reg, 0.016);
        let mut found = false;
        for pos in world.query::<&Position>().iter() {
            // Boid started at x=0 and has velocity +2 in X → should move right.
            if pos.x > 0.001 {
                found = true;
            }
        }
        assert!(found, "flock member position should be applied to the entity");
    }
}