// src/levels/portal.rs
// ──────────────────────────────────────────────────────────────────────────────
// Portal / trigger zone system for level transitions.
//
// A LevelPortal is a sphere in world space. When the player enters the
// trigger radius, it fires a load event for the target level. Optionally,
// it can also unload the source level (useful for one-way dungeon entrances
// where you don't need the overworld loaded anymore).
//
// Portals are placed in the persistent level (or any loaded level) as
// regular entities with a LevelPortal component. The portal system checks
// the player position against each active portal each frame.
// ──────────────────────────────────────────────────────────────────────────────

/// A portal that loads/unloads levels when the player enters its trigger zone.
///
/// Place this as a component on an entity in the persistent level.
/// The portal system (or game code) calls check_portal_trigger() each frame.
pub struct LevelPortal {
    /// World-space position of the portal trigger center.
    pub position: [f32; 3],
    /// Radius of the trigger zone (in world units).
    pub trigger_radius: f32,
    /// Level ID to load when entering the portal.
    pub target_level_id: u32,
    /// Whether to unload the source level after loading the target.
    /// Useful for one-way transitions (dungeon entrance, etc.).
    pub unload_source_level: bool,
    /// Whether this portal is active. Disabled portals don't trigger.
    pub active: bool,
    /// Whether the player is currently inside the trigger.
    /// Used to prevent re-triggering while inside.
    pub player_inside: bool,
}

impl LevelPortal {
    /// Create a new portal at a position, targeting a specific level.
    pub fn new(position: [f32; 3], target_level_id: u32) -> Self {
        Self {
            position,
            trigger_radius: 3.0,
            target_level_id,
            unload_source_level: false,
            active: true,
            player_inside: false,
        }
    }

    /// Set the trigger radius.
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.trigger_radius = radius;
        self
    }

    /// Mark this portal to unload the source level when entering.
    pub fn with_unload_source(mut self) -> Self {
        self.unload_source_level = true;
        self
    }
}

/// Result of a portal trigger check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortalEvent {
    /// The entity just entered the portal trigger zone.
    Entered,
    /// The entity just exited the portal trigger zone.
    Exited,
    /// No change — the entity's state relative to the portal hasn't changed.
    None,
}

/// Check if an entity has entered or exited a portal trigger zone.
///
/// Call this each frame for the player entity against each active portal.
/// The portal's player_inside flag is updated to track edge transitions.
pub fn check_portal_trigger(
    portal: &mut LevelPortal,
    entity_pos: [f32; 3],
) -> PortalEvent {
    if !portal.active {
        return PortalEvent::None;
    }

    // Calculate distance from entity to portal center.
    let dx = entity_pos[0] - portal.position[0];
    let dy = entity_pos[1] - portal.position[1];
    let dz = entity_pos[2] - portal.position[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    let is_inside = dist <= portal.trigger_radius;

    let event = if is_inside && !portal.player_inside {
        // Just entered the trigger zone.
        PortalEvent::Entered
    } else if !is_inside && portal.player_inside {
        // Just exited the trigger zone.
        PortalEvent::Exited
    } else {
        PortalEvent::None
    };

    // Update tracking state for next frame's edge detection.
    portal.player_inside = is_inside;

    event
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portal_enter_and_exit() {
        let mut portal = LevelPortal::new([0.0, 0.0, 0.0], 1)
            .with_radius(5.0);

        // Entity far away — no event.
        let event = check_portal_trigger(&mut portal, [10.0, 0.0, 0.0]);
        assert_eq!(event, PortalEvent::None);
        assert!(!portal.player_inside);

        // Entity moves inside trigger zone.
        let event = check_portal_trigger(&mut portal, [2.0, 0.0, 0.0]);
        assert_eq!(event, PortalEvent::Entered);
        assert!(portal.player_inside);

        // Entity still inside — no re-trigger.
        let event = check_portal_trigger(&mut portal, [3.0, 0.0, 0.0]);
        assert_eq!(event, PortalEvent::None);

        // Entity exits trigger zone.
        let event = check_portal_trigger(&mut portal, [10.0, 0.0, 0.0]);
        assert_eq!(event, PortalEvent::Exited);
        assert!(!portal.player_inside);
    }

    #[test]
    fn test_portal_disabled() {
        let mut portal = LevelPortal::new([0.0, 0.0, 0.0], 1)
            .with_radius(5.0);
        portal.active = false;

        // Even inside the trigger, a disabled portal fires no events.
        let event = check_portal_trigger(&mut portal, [0.0, 0.0, 0.0]);
        assert_eq!(event, PortalEvent::None);
        assert!(!portal.player_inside);
    }

    #[test]
    fn test_portal_builder() {
        let portal = LevelPortal::new([10.0, 20.0, 30.0], 42)
            .with_radius(15.0)
            .with_unload_source();

        assert_eq!(portal.position, [10.0, 20.0, 30.0]);
        assert_eq!(portal.target_level_id, 42);
        assert_eq!(portal.trigger_radius, 15.0);
        assert!(portal.unload_source_level);
    }
}
