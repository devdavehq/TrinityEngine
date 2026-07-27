// src/levels/state.rs
// ──────────────────────────────────────────────────────────────────────────────
// Persistent world state across level loads.
//
// When a level is unloaded and later reloaded, entities are freshly spawned
// from the .scene file. Any runtime state (health, killed flags, collected
// items) is lost unless we save it here.
//
// WorldStateManager stores per-entity state keyed by (level_name, entity_name).
// Before unloading a level, the game code should save entity states here.
// When reloading, it checks this manager to restore state (e.g., don't respawn
// an NPC the player already killed).
//
// This is a lightweight precursor to a full save/load system. It holds just
// enough data to maintain continuity between level loads/unloads.
// ──────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;

/// Per-entity state that persists between level loads.
///
/// Stored in WorldStateManager and consulted when re-loading a level
/// to determine if an entity should be modified from its default state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntityState {
    /// The entity's name (matches SceneMeta.name from the .scene file).
    pub entity_name: String,
    /// Last known position [x, y, z].
    pub position: [f32; 3],
    /// Optional health value (None if the entity doesn't have health).
    pub health: Option<i32>,
    /// Whether this entity is alive. If false, it won't be re-spawned
    /// when the level is loaded again.
    pub is_alive: bool,
    /// Arbitrary key-value flags for custom state.
    /// E.g., "door_opened" => "true", "quest_completed" => "1".
    pub custom_flags: HashMap<String, String>,
}

/// Manages persistent state for all entities across level loads.
///
/// Keyed by level name, then by entity name within that level.
/// Call save_entity() before unloading, get_entity() when reloading.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorldStateManager {
    /// level_name → list of entity states for that level.
    states: HashMap<String, Vec<EntityState>>,
}

impl WorldStateManager {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    /// Save the state of an entity within a level.
    /// If an entity with the same name already exists, it's replaced.
    pub fn save_entity(&mut self, level_name: &str, state: EntityState) {
        let level_states = self.states
            .entry(level_name.to_string())
            .or_default();
        // Replace existing entry with same name, or push new.
        if let Some(existing) = level_states.iter_mut().find(|s| s.entity_name == state.entity_name) {
            *existing = state;
        } else {
            level_states.push(state);
        }
    }

    /// Get saved state for an entity by name within a level.
    pub fn get_entity(&self, level_name: &str, entity_name: &str) -> Option<&EntityState> {
        self.states.get(level_name)?.iter().find(|s| s.entity_name == entity_name)
    }

    /// Check if an entity was killed (saved as not alive) before.
    pub fn is_entity_dead(&self, level_name: &str, entity_name: &str) -> bool {
        self.get_entity(level_name, entity_name)
            .map_or(false, |s| !s.is_alive)
    }

    /// Get all saved states for a level.
    pub fn get_level_states(&self, level_name: &str) -> &[EntityState] {
        self.states.get(level_name).map_or(&[], |v| v)
    }

    /// Clear all saved state for a specific level.
    pub fn clear_level(&mut self, level_name: &str) {
        self.states.remove(level_name);
    }

    /// Clear all saved state across all levels.
    pub fn clear_all(&mut self) {
        self.states.clear();
    }

    /// Get total number of saved entities across all levels.
    pub fn total_entities(&self) -> usize {
        self.states.values().map(|v| v.len()).sum()
    }

    /// Set a custom flag on a saved entity.
    pub fn set_flag(&mut self, level_name: &str, entity_name: &str, key: &str, value: &str) {
        if let Some(state) = self.states
            .get_mut(level_name)
            .and_then(|v| v.iter_mut().find(|s| s.entity_name == entity_name))
        {
            state.custom_flags.insert(key.to_string(), value.to_string());
        }
    }

    /// Get a custom flag from a saved entity.
    pub fn get_flag(&self, level_name: &str, entity_name: &str, key: &str) -> Option<&str> {
        self.get_entity(level_name, entity_name)?
            .custom_flags.get(key)
            .map(|s| s.as_str())
    }

    /// Save the entire state manager to a JSON file.
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    /// Load a state manager from a JSON file.
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&json).map_err(|e| e.to_string())
    }
}

impl Default for WorldStateManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_get_entity() {
        let mut mgr = WorldStateManager::new();
        let state = EntityState {
            entity_name: "Goblin_01".to_string(),
            position: [10.0, 0.0, 5.0],
            health: Some(50),
            is_alive: true,
            custom_flags: HashMap::new(),
        };
        mgr.save_entity("Forest", state);

        let retrieved = mgr.get_entity("Forest", "Goblin_01");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().health, Some(50));
        assert!(retrieved.unwrap().is_alive);
    }

    #[test]
    fn test_entity_dead_tracking() {
        let mut mgr = WorldStateManager::new();
        let state = EntityState {
            entity_name: "Boss".to_string(),
            position: [0.0, 0.0, 0.0],
            health: Some(0),
            is_alive: false,
            custom_flags: HashMap::new(),
        };
        mgr.save_entity("Dungeon", state);

        assert!(mgr.is_entity_dead("Dungeon", "Boss"));
        assert!(!mgr.is_entity_dead("Dungeon", "NonExistent"));
    }

    #[test]
    fn test_custom_flags() {
        let mut mgr = WorldStateManager::new();
        let state = EntityState {
            entity_name: "Door".to_string(),
            position: [0.0, 0.0, 0.0],
            health: None,
            is_alive: true,
            custom_flags: HashMap::new(),
        };
        mgr.save_entity("Dungeon", state);

        mgr.set_flag("Dungeon", "Door", "opened", "true");
        assert_eq!(mgr.get_flag("Dungeon", "Door", "opened"), Some("true"));
        assert_eq!(mgr.get_flag("Dungeon", "Door", "missing"), None);
    }

    #[test]
    fn test_clear_level() {
        let mut mgr = WorldStateManager::new();
        let state = EntityState {
            entity_name: "NPC".to_string(),
            position: [0.0; 3],
            health: Some(100),
            is_alive: true,
            custom_flags: HashMap::new(),
        };
        mgr.save_entity("Village", state);
        assert_eq!(mgr.total_entities(), 1);

        mgr.clear_level("Village");
        assert_eq!(mgr.total_entities(), 0);
        assert!(mgr.get_entity("Village", "NPC").is_none());
    }

    #[test]
    fn test_save_replaces_existing() {
        let mut mgr = WorldStateManager::new();
        let state1 = EntityState {
            entity_name: "Chest".to_string(),
            position: [0.0; 3],
            health: None,
            is_alive: true,
            custom_flags: HashMap::new(),
        };
        mgr.save_entity("Room", state1);

        let mut flags = HashMap::new();
        flags.insert("looted".to_string(), "true".to_string());
        let state2 = EntityState {
            entity_name: "Chest".to_string(),
            position: [1.0; 3],
            health: None,
            is_alive: true,
            custom_flags: flags,
        };
        mgr.save_entity("Room", state2);

        // Should be replaced, not duplicated.
        let entity = mgr.get_entity("Room", "Chest").unwrap();
        assert_eq!(entity.position, [1.0; 3]);
        assert_eq!(entity.custom_flags.get("looted").map(|s| s.as_str()), Some("true"));
        // Only one entity for this level.
        assert_eq!(mgr.get_level_states("Room").len(), 1);
    }
}
