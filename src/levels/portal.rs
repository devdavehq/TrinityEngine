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

/// The trigger volume shape of a portal. Like a UE5 trigger volume, the
/// portal fires when the player's position enters its volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortalShape {
    /// A ball of `trigger_radius` centered on the position.
    Sphere,
    /// An axis-aligned box with half-extents `box_extents` centered on the
    /// position.
    Box,
    /// A vertical capsule (cylinder + hemispherical caps) of `capsule_radius`
    /// and half-height `capsule_half_height` (the cylinder spans
    /// `position.y ± capsule_half_height`).
    Capsule,
}

impl Default for PortalShape {
    fn default() -> Self {
        Self::Sphere
    }
}

impl PortalShape {
    /// Parse a shape name from a manifest entry ("sphere"/"box"/"capsule").
    /// Unknown names fall back to sphere so stale manifests stay usable.
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "box" => Self::Box,
            "capsule" => Self::Capsule,
            _ => Self::Sphere,
        }
    }

    /// Canonical name for serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sphere => "sphere",
            Self::Box => "box",
            Self::Capsule => "capsule",
        }
    }
}

/// A portal that loads/unloads levels when the player enters its trigger zone.
///
/// Place this as a component on an entity in the persistent level.
/// The portal system (or game code) calls check_portal_trigger() each frame.
#[derive(Clone)]
pub struct LevelPortal {
    /// Shape of the trigger volume.
    pub shape: PortalShape,
    /// World-space position of the portal trigger center.
    pub position: [f32; 3],
    /// Trigger radius (sphere radius; used only when shape == Sphere).
    pub trigger_radius: f32,
    /// Half-extents of the box trigger (used only when shape == Box).
    pub box_extents: [f32; 3],
    /// Capsule radius + half-height (used only when shape == Capsule).
    pub capsule_radius: f32,
    pub capsule_half_height: f32,
    /// Level ID to load when entering the portal.
    pub target_level_id: u32,
    /// Level ID to unload when the portal fires (0 = keep everything loaded).
    /// Set this for one-way transitions (dungeon entrance, etc.).
    pub source_level_id: u32,
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
    /// Create a new spherical portal at a position, targeting a specific level.
    pub fn new(position: [f32; 3], target_level_id: u32) -> Self {
        Self {
            shape: PortalShape::Sphere,
            position,
            trigger_radius: 3.0,
            box_extents: [3.0, 3.0, 3.0],
            capsule_radius: 2.0,
            capsule_half_height: 2.0,
            target_level_id,
            source_level_id: 0,
            unload_source_level: false,
            active: true,
            player_inside: false,
        }
    }

    /// Set the sphere trigger radius.
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.trigger_radius = radius;
        self
    }

    /// Convert to a box trigger with the given half-extents.
    pub fn with_box(mut self, extents: [f32; 3]) -> Self {
        self.shape = PortalShape::Box;
        self.box_extents = extents;
        self
    }

    /// Convert to a capsule trigger with the given radius and half-height.
    pub fn with_capsule(mut self, radius: f32, half_height: f32) -> Self {
        self.shape = PortalShape::Capsule;
        self.capsule_radius = radius;
        self.capsule_half_height = half_height;
        self
    }

    /// Mark this portal to unload the source level when entering.
    pub fn with_unload_source(mut self) -> Self {
        self.unload_source_level = true;
        self
    }

    /// Build a portal from a manifest entry once level names have been
    /// resolved to IDs. If either referenced level doesn't exist the portal
    /// is disabled so a stale manifest can't fire into the void.
    pub fn from_entry(
        entry: &crate::levels::manifest::PortalEntry,
        level_manager: &crate::levels::LevelManager,
    ) -> Self {
        let target = level_manager
            .find_by_name(&entry.target_level)
            .map(|l| l.id)
            .unwrap_or(0);
        let source = level_manager
            .find_by_name(&entry.source_level)
            .map(|l| l.id)
            .unwrap_or(0);
        let shape = PortalShape::from_str(&entry.shape);
        Self {
            shape,
            position: entry.position,
            trigger_radius: entry.trigger_radius.max(0.1),
            box_extents: entry.box_extents,
            capsule_radius: entry.capsule_radius.max(0.1),
            capsule_half_height: entry.capsule_half_height.max(0.1),
            target_level_id: target,
            source_level_id: source,
            unload_source_level: !entry.source_level.is_empty(),
            active: entry.active && target != 0,
            player_inside: false,
        }
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

    let is_inside = match portal.shape {
        PortalShape::Sphere => {
            let dx = entity_pos[0] - portal.position[0];
            let dy = entity_pos[1] - portal.position[1];
            let dz = entity_pos[2] - portal.position[2];
            (dx * dx + dy * dy + dz * dz) <= portal.trigger_radius * portal.trigger_radius
        }
        PortalShape::Box => {
            let dx = (entity_pos[0] - portal.position[0]).abs();
            let dy = (entity_pos[1] - portal.position[1]).abs();
            let dz = (entity_pos[2] - portal.position[2]).abs();
            dx <= portal.box_extents[0]
                && dy <= portal.box_extents[1]
                && dz <= portal.box_extents[2]
        }
        PortalShape::Capsule => {
            // Distance from the point to the vertical segment
            // [position.y - half_height, position.y + half_height].
            let cy = (entity_pos[1] - portal.position[1])
                .clamp(-portal.capsule_half_height, portal.capsule_half_height);
            let dx = entity_pos[0] - portal.position[0];
            let dz = entity_pos[2] - portal.position[2];
            let dy = entity_pos[1] - portal.position[1] - cy;
            (dx * dx + dy * dy + dz * dz) <= portal.capsule_radius * portal.capsule_radius
        }
    };

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

    #[test]
    fn test_box_trigger() {
        let mut portal = LevelPortal::new([0.0, 0.0, 0.0], 1)
            .with_box([5.0, 2.0, 4.0]);

        // Inside the box.
        let event = check_portal_trigger(&mut portal, [4.9, 1.9, -3.9]);
        assert_eq!(event, PortalEvent::Entered);

        // Outside along the Y axis (thin box).
        let event = check_portal_trigger(&mut portal, [0.0, 3.0, 0.0]);
        assert_eq!(event, PortalEvent::Exited);

        // Outside on the corner.
        let event = check_portal_trigger(&mut portal, [6.0, 0.0, 5.0]);
        assert_eq!(event, PortalEvent::None);
    }

    #[test]
    fn test_capsule_trigger() {
        let mut portal = LevelPortal::new([0.0, 0.0, 0.0], 1)
            .with_capsule(2.0, 3.0);

        // Inside near the top hemisphere.
        let event = check_portal_trigger(&mut portal, [1.0, 4.5, 0.0]);
        assert_eq!(event, PortalEvent::Entered);

        // Inside the cylinder at full half-height.
        let event = check_portal_trigger(&mut portal, [0.0, 3.0, 1.9]);
        assert_eq!(event, PortalEvent::None); // still inside

        // Beyond the top cap.
        let event = check_portal_trigger(&mut portal, [0.0, 6.0, 0.0]);
        assert_eq!(event, PortalEvent::Exited);

        // Far to the side.
        let event = check_portal_trigger(&mut portal, [0.0, 0.0, 4.0]);
        assert_eq!(event, PortalEvent::None);
    }

    #[test]
    fn test_shape_parsing() {
        assert_eq!(PortalShape::from_str("box"), PortalShape::Box);
        assert_eq!(PortalShape::from_str("CAPSULE"), PortalShape::Capsule);
        assert_eq!(PortalShape::from_str("sphere"), PortalShape::Sphere);
        assert_eq!(PortalShape::from_str("garbage"), PortalShape::Sphere);
        assert_eq!(PortalShape::Box.as_str(), "box");
        assert_eq!(PortalShape::Capsule.as_str(), "capsule");
    }
}
