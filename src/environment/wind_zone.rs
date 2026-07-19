// src/environment/wind_zone.rs
// ── Wind Zone System ─────────────────────────────────────────────────────
//
// WHY IT EXISTS:
//   Global wind affects everything uniformly. But near a waterfall there might
//   be outward draft, near a cliff there might be updraft, and buildings create
//   wind shadows. WindZone components define local wind overrides.
//
// ARCHITECTURE:
//   WindZone is a pure component on entities. This module provides a system
//   function that evaluates all zones and returns effective wind for a position.
//
// DATA FLOW:
//   WindZone (ECS) → evaluate_wind_at(pos) → (direction, strength)
//
// PERFORMANCE:
//   O(n) where n = number of WindZone entities. Typically <10 per level.

use crate::components::{Position, WindZone};

/// Effective wind at a world position, blending global wind with local WindZones.
///
/// Returns (direction_x, direction_y, direction_z, strength).
/// Direction is normalized. Strength is the effective wind speed.
pub fn evaluate_wind_at(
    world: &hecs::World,
    global_direction: [f32; 3],
    global_strength: f32,
    position: [f32; 3],
) -> ([f32; 3], f32) {
    let mut zone_influence: f32 = 0.0;
    let mut result_dir = global_direction;
    let mut result_str = global_strength;

    for (pos, zone) in world.query::<(&Position, &WindZone)>().iter() {
        if !zone.active { continue; }

        let dx = position[0] - pos.x;
        let dy = position[1] - pos.y;
        let dz = position[2] - pos.z;
        let dist = (dx*dx + dy*dy + dz*dz).sqrt();

        if dist > zone.radius + zone.falloff { continue; }

        // Blend weight: 1.0 at center, ramps to 0.0 at edge.
        let blend = if dist <= zone.radius {
            1.0
        } else if zone.falloff > 0.0 {
            ((zone.radius + zone.falloff - dist) / zone.falloff).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Accumulate: weighted blend of global and zone wind.
        // Each zone only contributes to the remaining un-influenced portion.
        let w = blend * (1.0 - zone_influence);
        result_dir[0] += (zone.direction[0] - result_dir[0]) * w;
        result_dir[1] += (zone.direction[1] - result_dir[1]) * w;
        result_dir[2] += (zone.direction[2] - result_dir[2]) * w;
        result_str += (zone.strength - result_str) * w;
        zone_influence += w;
    }

    // Normalize direction.
    let len = (result_dir[0]*result_dir[0] + result_dir[1]*result_dir[1] + result_dir[2]*result_dir[2]).sqrt();
    if len > 0.001 {
        let inv = 1.0 / len;
        result_dir[0] *= inv;
        result_dir[1] *= inv;
        result_dir[2] *= inv;
    }

    (result_dir, result_str)
}

/// Find the strongest WindZone affecting a position.
/// Returns (direction, strength, distance) or None.
pub fn nearest_wind_zone_at(
    world: &hecs::World,
    position: [f32; 3],
) -> Option<([f32; 3], f32, f32)> {
    let mut best: Option<([f32; 3], f32, f32)> = None;
    let mut best_dist = f32::MAX;

    for (pos, zone) in world.query::<(&Position, &WindZone)>().iter() {
        if !zone.active { continue; }

        let dx = position[0] - pos.x;
        let dy = position[1] - pos.y;
        let dz = position[2] - pos.z;
        let dist = (dx*dx + dy*dy + dz*dz).sqrt();

        if dist < zone.radius + zone.falloff && dist < best_dist {
            best_dist = dist;
            best = Some((zone.direction, zone.strength, dist));
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_zones_returns_global() {
        let world = hecs::World::new();
        let (dir, str) = evaluate_wind_at(&world, [1.0, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0]);
        assert!((dir[0] - 1.0).abs() < 0.001);
        assert!((str - 0.5).abs() < 0.001);
    }

    #[test]
    fn zone_inside_overrides() {
        let mut world = hecs::World::new();
        world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 }, WindZone {
            direction: [0.0, 0.0, 1.0],
            strength: 2.0,
            radius: 50.0,
            falloff: 10.0,
            active: true,
        }));
        let (dir, str) = evaluate_wind_at(&world, [1.0, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0]);
        // Should be mostly the zone's direction (0,0,1) and strength (2.0)
        assert!(dir[2] > 0.8);
        assert!(str > 1.5);
    }

    #[test]
    fn zone_outside_ignored() {
        let mut world = hecs::World::new();
        world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 }, WindZone {
            direction: [0.0, 0.0, 1.0],
            strength: 2.0,
            radius: 10.0,
            falloff: 5.0,
            active: true,
        }));
        // 20 units away — beyond radius + falloff = 15
        let (dir, str) = evaluate_wind_at(&world, [1.0, 0.0, 0.0], 0.5, [20.0, 0.0, 0.0]);
        assert!((dir[0] - 1.0).abs() < 0.001);
        assert!((str - 0.5).abs() < 0.001);
    }

    #[test]
    fn inactive_zone_ignored() {
        let mut world = hecs::World::new();
        world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 }, WindZone {
            direction: [0.0, 0.0, 1.0],
            strength: 2.0,
            radius: 50.0,
            falloff: 10.0,
            active: false,
        }));
        let (dir, str) = evaluate_wind_at(&world, [1.0, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0]);
        assert!((dir[0] - 1.0).abs() < 0.001);
        assert!((str - 0.5).abs() < 0.001);
    }
}
