// src/levels/cross_ref.rs
// ──────────────────────────────────────────────────────────────────────────────
// Cross-level entity references.
//
// Allows an entity in one level to reference an entity in another level.
// When levels load/unload, references are remapped using entity names.
//
// Design:
//   CrossLevelRef stores a source (level + entity) and target (level + entity name).
//   CrossRefManager resolves these references at runtime by looking up resolved
//   entity handles in a HashMap keyed by (level_name, entity_name).
//   When a level loads, the manager can resolve all pending references that
//   target entities in that level. When a level unloads, those resolved entries
//   are cleared so stale handles are never returned.
// ──────────────────────────────────────────────────────────────────────────────

use hecs::Entity;
use std::collections::HashMap;

/// A reference from an entity in one level to a named entity in another level.
///
/// The source is identified by both level name and entity handle (runtime).
/// The target is identified by level name and entity name (string), so it
/// can survive level reloads where entity handles change.
#[derive(Debug, Clone)]
pub struct CrossLevelRef {
    pub source_level: String,
    pub source_entity: Entity,
    pub target_level: String,
    pub target_entity_name: String,
}

/// Manages cross-level entity references.
///
/// References are registered as (level, entity_name) pairs. When a level is
/// loaded, `on_level_loaded` resolves all pending references whose target is
/// in that level. When a level is unloaded, `on_level_unloaded` clears the
/// resolved entries for that level so stale handles don't linger.
pub struct CrossRefManager {
    /// All registered cross-level references.
    refs: Vec<CrossLevelRef>,
    /// Maps (level_name, entity_name) -> Entity for resolved references.
    resolved: HashMap<(String, String), Entity>,
}

impl CrossRefManager {
    pub fn new() -> Self {
        Self {
            refs: Vec::new(),
            resolved: HashMap::new(),
        }
    }

    /// Register a new cross-level reference.
    pub fn register_ref(&mut self, r: CrossLevelRef) {
        self.refs.push(r);
    }

    /// Resolve a reference by level name and entity name.
    /// Returns the live entity handle if it has been resolved.
    pub fn resolve(&self, level: &str, name: &str) -> Option<Entity> {
        self.resolved.get(&(level.to_string(), name.to_string())).copied()
    }

    /// Called when a level is loaded. Iterates all registered references whose
    /// target is in `level` and resolves them by scanning the world for the
    /// named entity.
    pub fn on_level_loaded(&mut self, level: &str, world: &hecs::World) {
        // Collect target names we need to resolve for this level.
        let targets: Vec<String> = self.refs.iter()
            .filter(|r| r.target_level == level)
            .map(|r| r.target_entity_name.clone())
            .collect();

        // Build a temporary name -> entity map by querying the world.
        // We query all entities that have a Name component (any T that is a String).
        // Since we don't know the exact component type, we use hecs::QueryOne or
        // a broader approach. For now, we rely on the caller to provide the world
        // and we iterate all entities looking for a Name-like component.
        // In practice, levels should register entity names via a component.
        // We'll do a simple linear scan matching against a common Name component.
        // We need to import or reference the Name component — but to keep this
        // decoupled, we accept that the user stores entity names as a component.

        // For a general solution, we iterate the world's entity set and attempt
        // to match by entity name. This requires the target entities to have a
        // component we can read. We use a heuristic: try to find entities with
        // a &str or String component that matches.
        //
        // NOTE: This is a simplified resolution. In a full engine, you'd query
        // a specific `Name(String)` component. Here we do an archetypal scan
        // looking for (&str,) or (&String,) tuples.

        // We can't do arbitrary component queries without knowing the type,
        // so we store the world's full entity list and let the user call
        // `resolve_after_spawn` manually, or we do a name-based lookup
        // using a separate mapping.
        //
        // Simple approach: iterate the world's entities and look for (&String,)
        // as a common pattern for named entities.

        for target_name in &targets {
            if self.resolved.contains_key(&(level.to_string(), target_name.clone())) {
                continue; // Already resolved.
            }
            // Match against the engine's entity name component (SceneMeta).
            for (entity, meta) in world.query::<(Entity, &crate::components::SceneMeta)>().iter() {
                if meta.name == target_name.as_str() {
                    self.resolved.insert(
                        (level.to_string(), target_name.clone()),
                        entity,
                    );
                    break;
                }
            }
        }
    }

    /// Called when a level is unloaded. Clears all resolved entries for that
    /// level so stale entity handles are never returned.
    pub fn on_level_unloaded(&mut self, level: &str) {
        self.resolved.retain(|(lvl, _), _| lvl != level);
    }

    /// Get all registered references.
    pub fn refs(&self) -> &[CrossLevelRef] {
        &self.refs
    }

    /// Remove all references (for full reset).
    pub fn clear(&mut self) {
        self.refs.clear();
        self.resolved.clear();
    }
}

impl Default for CrossRefManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_resolve() {
        let mut mgr = CrossRefManager::new();
        let mut world = hecs::World::new();

        let entity = world.spawn((crate::components::SceneMeta {
            name: "Player".to_string(),
            mesh_path: String::new(),
        },));
        let source_entity = world.spawn(());

        mgr.register_ref(CrossLevelRef {
            source_level: "Level_A".to_string(),
            source_entity,
            target_level: "Level_B".to_string(),
            target_entity_name: "Player".to_string(),
        });

        // Not yet resolved.
        assert!(mgr.resolve("Level_B", "Player").is_none());

        // After loading the level, the entity should resolve.
        mgr.on_level_loaded("Level_B", &world);
        assert_eq!(mgr.resolve("Level_B", "Player"), Some(entity));
    }

    #[test]
    fn test_unload_clears_resolved() {
        let mut mgr = CrossRefManager::new();
        let mut world = hecs::World::new();
        let entity = world.spawn((crate::components::SceneMeta {
            name: "NPC".to_string(),
            mesh_path: String::new(),
        },));
        let source_entity = world.spawn(());

        mgr.register_ref(CrossLevelRef {
            source_level: "Level_A".to_string(),
            source_entity,
            target_level: "Level_C".to_string(),
            target_entity_name: "NPC".to_string(),
        });

        mgr.on_level_loaded("Level_C", &world);
        assert!(mgr.resolve("Level_C", "NPC").is_some());

        mgr.on_level_unloaded("Level_C");
        assert!(mgr.resolve("Level_C", "NPC").is_none());
    }

    #[test]
    fn test_clear() {
        let mut mgr = CrossRefManager::new();
        let mut world = hecs::World::new();
        let entity = world.spawn((crate::components::SceneMeta {
            name: "X".to_string(),
            mesh_path: String::new(),
        },));
        let source_entity = world.spawn(());

        mgr.register_ref(CrossLevelRef {
            source_level: "L1".to_string(),
            source_entity,
            target_level: "L2".to_string(),
            target_entity_name: "X".to_string(),
        });

        mgr.on_level_loaded("L2", &world);
        assert!(mgr.resolve("L2", "X").is_some());

        mgr.clear();
        assert!(mgr.resolve("L2", "X").is_none());
        assert!(mgr.refs().is_empty());
    }
}
