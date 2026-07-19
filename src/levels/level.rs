// src/levels/level.rs
// ──────────────────────────────────────────────────────────────────────────────
// Core level types.
//
// A Level represents a collection of entities loaded from a .scene file.
// Unlike the old SceneManager::build() approach which called world.clear()
// and destroyed everything, levels coexist in memory. The "persistent level"
// is always loaded (think of it as the "always-loaded base world" in UE5),
// while streaming levels load/unload dynamically around the player.
//
// Design notes:
// - Each Level tracks which hecs::Entity IDs belong to it so we can
//   selectively despawn only that level's entities on unload.
// - Levels have an origin offset (world-space position) and streaming
//   distances for distance-based automatic loading.
// - The persistent level can never be unloaded — it holds the player,
//   UI, and other permanent state.
// ──────────────────────────────────────────────────────────────────────────────

use std::path::PathBuf;

/// Represents one loaded level (scene) within the game world.
///
/// Multiple Levels coexist in the hecs World. Each Level tracks its own
/// entities so we can load/unload them independently without touching
/// entities belonging to other levels.
pub struct Level {
    /// Unique ID for this level instance.
    pub id: u32,
    /// Display name (e.g., "Forest_01", "Dungeon_Boss").
    pub name: String,
    /// Path to the .scene file on disk.
    pub file_path: PathBuf,
    /// Whether this level is currently loaded in memory.
    pub loaded: bool,
    /// Whether this level is visible (rendered). You can unload rendering
    /// without unloading the entities — useful for hiding distant levels
    /// while keeping their logic running.
    pub visible: bool,
    /// Whether this is the persistent (always-loaded) level.
    /// The persistent level is never streamed out. It typically contains
    /// the player character, global managers, and UI elements.
    pub persistent: bool,
    /// Entity IDs that belong to this level. Used for selective despawn
    /// when unloading — we only remove these entities, not everything.
    pub entities: Vec<hecs::Entity>,
    /// World-space origin offset for this level. When loading a level,
    /// entities can be placed relative to this origin, allowing levels
    /// to be positioned anywhere in the world.
    pub origin: [f32; 3],
    /// Streaming distance — how close the player must be to trigger loading.
    pub streaming_distance: f32,
    /// Unloading distance — how far before the level unloads.
    /// Must be >= streaming_distance to create a hysteresis band that
    /// prevents load/unload oscillation at the boundary.
    pub unloading_distance: f32,
}

impl Level {
    /// Create a new level with default settings.
    /// The level starts unloaded and non-persistent.
    pub fn new(id: u32, name: &str, file_path: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            file_path: PathBuf::from(file_path),
            loaded: false,
            visible: true,
            persistent: false,
            entities: Vec::new(),
            origin: [0.0; 3],
            streaming_distance: 100.0,
            unloading_distance: 200.0,
        }
    }

    /// Set the world-space origin offset for this level.
    pub fn with_origin(mut self, x: f32, y: f32, z: f32) -> Self {
        self.origin = [x, y, z];
        self
    }

    /// Set streaming distances (load distance, unload distance).
    pub fn with_streaming(mut self, load_dist: f32, unload_dist: f32) -> Self {
        self.streaming_distance = load_dist;
        self.unloading_distance = unload_dist;
        self
    }

    /// Mark this level as persistent (always loaded, never unloaded).
    pub fn with_persistent(mut self) -> Self {
        self.persistent = true;
        self.loaded = true;
        self
    }
}

/// Manages all levels in the game world.
///
/// LevelManager is the central registry. It doesn't spawn/despawn entities
/// itself — that's done by the level loading system which reads the .scene
/// file and spawns entities, then records them in the Level's entities vec.
pub struct LevelManager {
    /// All registered levels (both loaded and unloaded).
    pub levels: Vec<Level>,
    /// Next available level ID (auto-incrementing).
    next_id: u32,
    /// The persistent level index (always loaded).
    persistent_index: Option<usize>,
}

impl LevelManager {
    pub fn new() -> Self {
        Self {
            levels: Vec::new(),
            next_id: 1,
            persistent_index: None,
        }
    }

    /// Register a new level (does NOT load it yet).
    /// Returns the assigned level ID.
    pub fn register_level(&mut self, name: &str, file_path: &str) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.levels.push(Level::new(id, name, file_path));
        id
    }

    /// Set a level as the persistent level (always loaded, never unloaded).
    pub fn set_persistent(&mut self, level_id: u32) {
        if let Some(level) = self.levels.iter_mut().find(|l| l.id == level_id) {
            level.persistent = true;
            level.loaded = true;
            self.persistent_index = self.levels.iter().position(|l| l.id == level_id);
        }
    }

    /// Get a level by ID (immutable borrow).
    pub fn get(&self, id: u32) -> Option<&Level> {
        self.levels.iter().find(|l| l.id == id)
    }

    /// Get a level by ID (mutable borrow).
    pub fn get_mut(&mut self, id: u32) -> Option<&mut Level> {
        self.levels.iter_mut().find(|l| l.id == id)
    }

    /// Find a level by name.
    pub fn find_by_name(&self, name: &str) -> Option<&Level> {
        self.levels.iter().find(|l| l.name == name)
    }

    /// Get all currently loaded levels.
    pub fn loaded_levels(&self) -> Vec<&Level> {
        self.levels.iter().filter(|l| l.loaded).collect()
    }

    /// Get the persistent level (if one exists).
    pub fn persistent_level(&self) -> Option<&Level> {
        self.persistent_index.and_then(|i| self.levels.get(i))
    }

    /// Load a level by ID — marks it as loaded.
    /// Actual entity spawning happens externally (in the level loading system).
    pub fn load_level(&mut self, id: u32) -> bool {
        if let Some(level) = self.levels.iter_mut().find(|l| l.id == id) {
            level.loaded = true;
            true
        } else {
            false
        }
    }

    /// Unload a level by ID — marks it as unloaded and clears its entity list.
    /// The persistent level cannot be unloaded.
    pub fn unload_level(&mut self, id: u32) -> bool {
        if let Some(level) = self.levels.iter_mut().find(|l| l.id == id) {
            if level.persistent {
                return false; // Can't unload the persistent level.
            }
            level.loaded = false;
            level.entities.clear();
            true
        } else {
            false
        }
    }

    /// Toggle level visibility without unloading its entities.
    pub fn set_visible(&mut self, id: u32, visible: bool) {
        if let Some(level) = self.levels.iter_mut().find(|l| l.id == id) {
            level.visible = visible;
        }
    }

    /// Check if a level is loaded.
    pub fn is_loaded(&self, id: u32) -> bool {
        self.levels
            .iter()
            .find(|l| l.id == id)
            .map_or(false, |l| l.loaded)
    }

    /// Get the count of loaded levels.
    pub fn loaded_count(&self) -> usize {
        self.levels.iter().filter(|l| l.loaded).count()
    }

    /// Get total entity count across all loaded levels.
    pub fn total_entities(&self) -> usize {
        self.levels
            .iter()
            .filter(|l| l.loaded)
            .map(|l| l.entities.len())
            .sum()
    }

    /// Remove all levels (for full reset). Clears everything including
    /// the persistent level designation.
    pub fn clear_all(&mut self) {
        self.levels.clear();
        self.next_id = 1;
        self.persistent_index = None;
    }
}

impl Default for LevelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_load_level() {
        let mut mgr = LevelManager::new();
        let id = mgr.register_level("Forest", "scenes/forest.scene");
        assert_eq!(id, 1);
        assert!(!mgr.is_loaded(id));

        assert!(mgr.load_level(id));
        assert!(mgr.is_loaded(id));
        assert_eq!(mgr.loaded_count(), 1);
    }

    #[test]
    fn test_unload_level() {
        let mut mgr = LevelManager::new();
        let id = mgr.register_level("Dungeon", "scenes/dungeon.scene");
        mgr.load_level(id);
        assert!(mgr.unload_level(id));
        assert!(!mgr.is_loaded(id));
        assert_eq!(mgr.loaded_count(), 0);
    }

    #[test]
    fn test_persistent_level_cannot_be_unloaded() {
        let mut mgr = LevelManager::new();
        let id = mgr.register_level("Persistent", "scenes/main.scene");
        mgr.set_persistent(id);
        assert!(mgr.is_loaded(id));
        // Persistent level should refuse unload.
        assert!(!mgr.unload_level(id));
        assert!(mgr.is_loaded(id));
    }

    #[test]
    fn test_level_builder() {
        let level = Level::new(1, "Test", "test.scene")
            .with_origin(10.0, 0.0, 20.0)
            .with_streaming(50.0, 100.0)
            .with_persistent();

        assert_eq!(level.origin, [10.0, 0.0, 20.0]);
        assert_eq!(level.streaming_distance, 50.0);
        assert_eq!(level.unloading_distance, 100.0);
        assert!(level.persistent);
        assert!(level.loaded); // persistent levels start loaded
    }

    #[test]
    fn test_find_by_name() {
        let mut mgr = LevelManager::new();
        mgr.register_level("Forest", "scenes/forest.scene");
        mgr.register_level("Dungeon", "scenes/dungeon.scene");

        assert!(mgr.find_by_name("Forest").is_some());
        assert!(mgr.find_by_name("Dungeon").is_some());
        assert!(mgr.find_by_name("Ocean").is_none());
    }

    #[test]
    fn test_total_entities() {
        use hecs::World;
        let mut mgr = LevelManager::new();
        let id1 = mgr.register_level("Level1", "a.scene");
        let id2 = mgr.register_level("Level2", "b.scene");
        mgr.load_level(id1);
        mgr.load_level(id2);
        // Spawn real entities in a temporary world to get valid handles.
        let mut w = World::new();
        let e1 = w.spawn(());
        let e2 = w.spawn(());
        let e3 = w.spawn(());
        mgr.get_mut(id1).unwrap().entities.push(e1);
        mgr.get_mut(id1).unwrap().entities.push(e2);
        mgr.get_mut(id2).unwrap().entities.push(e3);
        assert_eq!(mgr.total_entities(), 3);
    }
}
