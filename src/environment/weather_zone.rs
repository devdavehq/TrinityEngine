// src/environment/weather_zone.rs
// ── Weather Zone System ──────────────────────────────────────────────────
//
// WHY IT EXISTS:
//   Different parts of the world can have different weather. A volcano might
//   have ash storms while a nearby valley is clear. WeatherZone components
//   on entities define local weather overrides that blend with the global
//   weather state based on player proximity.
//
// ARCHITECTURE:
//   WeatherZone is a pure component on entities. This module provides a
//   system function that evaluates all zones and returns the effective
//   weather for a given world position.
//
// DATA FLOW:
//   WeatherZone (ECS) → evaluate_weather_at(pos) → WeatherState (effective)
//
// USAGE:
//   Place an entity with Position + WeatherZone in the world.
//   Call evaluate_weather_at() with the player's position each frame.
//   The returned WeatherState blends the global weather with the nearest zone.
//
// PERFORMANCE:
//   O(n) where n = number of WeatherZone entities. Typically <10 per level.

use crate::components::{Position, WeatherZone};
use crate::environment::weather::{WeatherCondition, WeatherState};

/// Evaluate the effective weather at a world position, considering both
/// the global weather and any nearby WeatherZone components.
///
/// Returns the blended weather state. Zones override global weather when
/// the position is within their radius (with optional falloff at edges).
pub fn evaluate_weather_at(
    world: &hecs::World,
    global_weather: &WeatherState,
    position: [f32; 3],
) -> WeatherState {
    let mut best_override: Option<(WeatherCondition, f32, f32)> = None; // (condition, intensity, blend_weight)

    for (pos, zone) in world.query::<(&Position, &WeatherZone)>().iter() {
        if !zone.active { continue; }

        let dx = position[0] - pos.x;
        let dy = position[1] - pos.y;
        let dz = position[2] - pos.z;
        let dist = (dx*dx + dy*dy + dz*dz).sqrt();

        if dist > zone.radius + zone.falloff { continue; }

        // Compute blend weight: 1.0 inside zone core, ramps to 0.0 at falloff edge.
        let blend = if dist <= zone.radius - zone.falloff || (zone.falloff <= 0.0 && dist <= zone.radius) {
            1.0
        } else if zone.falloff > 0.0 && dist <= zone.radius + zone.falloff {
            ((zone.radius + zone.falloff - dist) / zone.falloff).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Keep the zone with the highest blend weight (closest zone wins).
        match &best_override {
            Some((_, _, existing_weight)) if blend <= *existing_weight => {}
            _ => {
                best_override = Some((zone.condition, zone.intensity, blend));
            }
        }
    }

    match best_override {
        Some((condition, intensity, blend)) if blend > 0.01 => {
            let mut result = global_weather.clone();
            // When blend > 0.5, fully adopt the zone's condition.
            // Blend the intensity smoothly between global and zone.
            if blend > 0.5 {
                result.condition = condition;
                result.intensity = result.intensity * (1.0 - blend) + intensity * blend;
            } else {
                // Partial influence: shift intensity toward zone but don't change condition.
                result.intensity = result.intensity * (1.0 - blend) + intensity * blend * 0.3;
            }
            result
        }
        _ => global_weather.clone(),
    }
}

/// Find the strongest WeatherZone affecting a position.
/// Returns (condition, intensity, distance_to_center, zone_radius) or None.
pub fn nearest_zone_at(
    world: &hecs::World,
    position: [f32; 3],
) -> Option<(WeatherCondition, f32, f32, f32)> {
    let mut best: Option<(WeatherCondition, f32, f32, f32)> = None;
    let mut best_dist = f32::MAX;

    for (pos, zone) in world.query::<(&Position, &WeatherZone)>().iter() {
        if !zone.active { continue; }

        let dx = position[0] - pos.x;
        let dy = position[1] - pos.y;
        let dz = position[2] - pos.z;
        let dist = (dx*dx + dy*dy + dz*dz).sqrt();

        if dist < zone.radius + zone.falloff && dist < best_dist {
            best_dist = dist;
            best = Some((zone.condition, zone.intensity, dist, zone.radius));
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::weather::WeatherState;

    #[test]
    fn no_zones_returns_global() {
        let world = hecs::World::new();
        let global = WeatherState::clear();
        let result = evaluate_weather_at(&world, &global, [0.0, 0.0, 0.0]);
        assert_eq!(result.condition, WeatherCondition::Clear);
    }

    #[test]
    fn zone_inside_overrides() {
        let mut world = hecs::World::new();
        world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 }, WeatherZone {
            condition: WeatherCondition::Storm,
            intensity: 1.0,
            radius: 50.0,
            falloff: 10.0,
            active: true,
        }));
        let global = WeatherState::clear();
        let result = evaluate_weather_at(&world, &global, [0.0, 0.0, 0.0]);
        assert_eq!(result.condition, WeatherCondition::Storm);
        assert!(result.intensity > 0.8);
    }

    #[test]
    fn zone_outside_falloff() {
        let mut world = hecs::World::new();
        world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 }, WeatherZone {
            condition: WeatherCondition::Storm,
            intensity: 1.0,
            radius: 10.0,
            falloff: 5.0,
            active: true,
        }));
        let global = WeatherState::clear();
        // 20 units away — beyond radius (10) + falloff (5) = 15
        let result = evaluate_weather_at(&world, &global, [20.0, 0.0, 0.0]);
        assert_eq!(result.condition, WeatherCondition::Clear);
    }

    #[test]
    fn inactive_zone_ignored() {
        let mut world = hecs::World::new();
        world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 }, WeatherZone {
            condition: WeatherCondition::Storm,
            intensity: 1.0,
            radius: 50.0,
            falloff: 10.0,
            active: false,
        }));
        let global = WeatherState::clear();
        let result = evaluate_weather_at(&world, &global, [0.0, 0.0, 0.0]);
        assert_eq!(result.condition, WeatherCondition::Clear);
    }
}
