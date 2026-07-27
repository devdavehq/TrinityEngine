use crate::components::Position;
use hecs::World;

/// Dynamic flood system — allows water level to rise/flood terrain.
///
/// Water level rises over time when flooding is active. Entities with
/// components::Position below the water surface are considered submerged.
/// The renderer reads water_level to set the water plane height.
pub struct FloodSystem {
    /// Current water level (Y height in world space).
    pub water_level: f32,
    /// Target water level to flood towards.
    pub target_level: f32,
    /// Rate of water rise (units per second).
    pub rise_rate: f32,
    /// Rate of water fall (units per second).
    pub fall_rate: f32,
    /// Whether flooding is active.
    pub active: bool,
    /// Maximum water level cap.
    pub max_level: f32,
    /// Minimum water level.
    pub min_level: f32,
    /// Per-entity flood effects (splash, buoyancy).
    pub splash_active: bool,
}

impl Default for FloodSystem {
    fn default() -> Self {
        Self {
            water_level: 0.0,
            target_level: 0.0,
            rise_rate: 0.5,
            fall_rate: 0.3,
            active: false,
            max_level: 50.0,
            min_level: 0.0,
            splash_active: true,
        }
    }
}

impl FloodSystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin flooding towards a target water level.
    pub fn start_flood(&mut self, target: f32) {
        self.target_level = target.clamp(self.min_level, self.max_level);
        self.active = true;
    }

    /// Stop flooding — water holds at current level.
    pub fn stop_flood(&mut self) {
        self.active = false;
    }

    /// Advance the water level by dt seconds.
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }
        if (self.water_level - self.target_level).abs() < 0.001 {
            self.water_level = self.target_level;
            return;
        }
        let rate = if self.water_level < self.target_level {
            self.rise_rate
        } else {
            self.fall_rate
        };
        let step = rate * dt;
        if self.water_level < self.target_level {
            self.water_level = (self.water_level + step).min(self.target_level);
        } else {
            self.water_level = (self.water_level - step).max(self.target_level);
        }
        self.water_level = self.water_level.clamp(self.min_level, self.max_level);
    }

    /// Returns true if the entity is below the water surface.
    pub fn is_submerged(&self, entity_y: f32) -> bool {
        entity_y < self.water_level
    }

    /// How far below the surface the entity is (0 if above water).
    pub fn depth_below_surface(&self, entity_y: f32) -> f32 {
        (self.water_level - entity_y).max(0.0)
    }
}

/// System function — call each frame to advance flood state and log newly
/// submerged entities.
pub fn flood_system(flood: &mut FloodSystem, world: &mut World, dt: f32) {
    flood.update(dt);
    if !flood.splash_active {
        return;
    }
    for (_entity, pos) in world.query_mut::<(hecs::Entity, &Position)>() {
        if flood.is_submerged(pos.y) {
            tracing::trace!(
                "[Flood] Entity at y={:.2} submerged (water={:.2})",
                pos.y,
                flood.water_level
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_water_level_is_zero() {
        let f = FloodSystem::new();
        assert_eq!(f.water_level, 0.0);
        assert!(!f.active);
    }

    #[test]
    fn start_flood_sets_target_and_active() {
        let mut f = FloodSystem::new();
        f.start_flood(10.0);
        assert!(f.active);
        assert_eq!(f.target_level, 10.0);
    }

    #[test]
    fn update_rises_toward_target() {
        let mut f = FloodSystem::new();
        f.start_flood(5.0);
        f.update(1.0);
        assert!((f.water_level - 0.5).abs() < 0.001);
    }

    #[test]
    fn update_falls_toward_target() {
        let mut f = FloodSystem::new();
        f.water_level = 5.0;
        f.start_flood(0.0);
        f.update(1.0);
        assert!((f.water_level - 4.7).abs() < 0.01);
    }

    #[test]
    fn clamp_at_max_level() {
        let mut f = FloodSystem::new();
        f.max_level = 2.0;
        f.start_flood(100.0);
        f.update(1000.0);
        assert!((f.water_level - 2.0).abs() < 0.001);
    }

    #[test]
    fn clamp_at_min_level() {
        let mut f = FloodSystem::new();
        f.water_level = 0.5;
        f.min_level = 0.0;
        f.start_flood(-5.0);
        f.update(1000.0);
        assert!((f.water_level).abs() < 0.001);
    }

    #[test]
    fn stop_flood_freezes() {
        let mut f = FloodSystem::new();
        f.start_flood(10.0);
        f.update(1.0);
        let level = f.water_level;
        f.stop_flood();
        f.update(10.0);
        assert!((f.water_level - level).abs() < 0.001);
    }

    #[test]
    fn is_submerged() {
        let f = FloodSystem { water_level: 3.0, ..FloodSystem::new() };
        assert!(f.is_submerged(2.0));
        assert!(!f.is_submerged(4.0));
        assert!(!f.is_submerged(3.0));
    }

    #[test]
    fn depth_below_surface() {
        let f = FloodSystem { water_level: 5.0, ..FloodSystem::new() };
        assert!((f.depth_below_surface(2.0) - 3.0).abs() < 0.001);
        assert!((f.depth_below_surface(7.0)).abs() < 0.001);
    }
}
