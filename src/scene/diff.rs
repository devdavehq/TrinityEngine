// src/scene/diff.rs
// ──────────────────────────────────────────────────────────────────────────────
// Scene diffing — compare two scenes and produce a delta of changes.
//
// Only loads/reloads changed assets instead of doing a full scene reload.
// Supports:
//   - Entity diff (added/removed/modified entities)
//   - Component diff (which components changed on each entity)
//   - Material diff (which materials changed)
//   - Mesh diff (which meshes need reloading)
//   - Asset fingerprinting (hash-based change detection)
//
// Usage:
//   let delta = diff_scenes(&old_scene, &new_scene);
//   apply_delta(world, delta); // only processes changes
// ──────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Hash-based fingerprint for detecting changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AssetFingerprint(pub u64);

impl AssetFingerprint {
    pub fn compute(data: &[u8]) -> Self {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        Self(hasher.finish())
    }

    pub fn compute_str(s: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        Self(hasher.finish())
    }
}

/// Entity representation for diffing purposes.
#[derive(Clone, Debug)]
pub struct DiffEntity {
    pub entity_id: u64,
    pub name: String,
    pub components: HashMap<String, String>, // component_name → serialized data
    pub fingerprint: AssetFingerprint,
}

/// A single change in the diff.
#[derive(Clone, Debug)]
pub enum DiffChange {
    EntityAdded { entity_id: u64, name: String },
    EntityRemoved { entity_id: u64, name: String },
    EntityModified { entity_id: u64, name: String, component_changes: Vec<ComponentChange> },
    MaterialChanged { path: String, old_fingerprint: AssetFingerprint, new_fingerprint: AssetFingerprint },
    MeshChanged { path: String, old_fingerprint: AssetFingerprint, new_fingerprint: AssetFingerprint },
    ScriptChanged { path: String },
}

/// A component-level change.
#[derive(Clone, Debug)]
pub struct ComponentChange {
    pub component_name: String,
    pub old_data: String,
    pub new_data: String,
}

/// Full scene diff result.
#[derive(Clone, Debug)]
pub struct SceneDiff {
    pub changes: Vec<DiffChange>,
    pub added_count: usize,
    pub removed_count: usize,
    pub modified_count: usize,
    pub asset_changes: usize,
}

impl SceneDiff {
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    pub fn summary(&self) -> String {
        format!(
            "{} added, {} removed, {} modified, {} asset changes",
            self.added_count, self.removed_count, self.modified_count, self.asset_changes
        )
    }
}

/// Compare two sets of entities and produce a diff.
pub fn diff_entities(old_entities: &[DiffEntity], new_entities: &[DiffEntity]) -> SceneDiff {
    let mut changes = Vec::new();
    let mut added = 0;
    let mut removed = 0;
    let mut modified = 0;
    let mut asset_changes = 0;

    // Build lookup maps
    let old_map: HashMap<u64, &DiffEntity> = old_entities.iter().map(|e| (e.entity_id, e)).collect();
    let new_map: HashMap<u64, &DiffEntity> = new_entities.iter().map(|e| (e.entity_id, e)).collect();

    // Find removed entities
    for (id, old) in &old_map {
        if !new_map.contains_key(id) {
            changes.push(DiffChange::EntityRemoved {
                entity_id: *id,
                name: old.name.clone(),
            });
            removed += 1;
        }
    }

    // Find added and modified entities
    for (id, new_entity) in &new_map {
        match old_map.get(id) {
            None => {
                // Entity added
                changes.push(DiffChange::EntityAdded {
                    entity_id: *id,
                    name: new_entity.name.clone(),
                });
                added += 1;
            }
            Some(old) => {
                // Check if modified
                if old.fingerprint != new_entity.fingerprint {
                    let mut component_changes = Vec::new();

                    // Find changed components
                    for (comp_name, new_data) in &new_entity.components {
                        match old.components.get(comp_name) {
                            None => {
                                component_changes.push(ComponentChange {
                                    component_name: comp_name.clone(),
                                    old_data: String::new(),
                                    new_data: new_data.clone(),
                                });
                            }
                            Some(old_data) => {
                                if old_data != new_data {
                                    component_changes.push(ComponentChange {
                                        component_name: comp_name.clone(),
                                        old_data: old_data.clone(),
                                        new_data: new_data.clone(),
                                    });
                                }
                            }
                        }
                    }

                    // Find removed components
                    for comp_name in old.components.keys() {
                        if !new_entity.components.contains_key(comp_name) {
                            component_changes.push(ComponentChange {
                                component_name: comp_name.clone(),
                                old_data: old.components[comp_name].clone(),
                                new_data: String::new(),
                            });
                        }
                    }

                    if !component_changes.is_empty() {
                        changes.push(DiffChange::EntityModified {
                            entity_id: *id,
                            name: new_entity.name.clone(),
                            component_changes: component_changes.clone(),
                        });
                        asset_changes += component_changes.len();
                        modified += 1;
                    }
                }
            }
        }
    }

    SceneDiff {
        changes,
        added_count: added,
        removed_count: removed,
        modified_count: modified,
        asset_changes,
    }
}

/// Compare material fingerprints.
pub fn diff_materials(
    old: &[(String, AssetFingerprint)],
    new: &[(String, AssetFingerprint)],
) -> Vec<DiffChange> {
    let old_map: HashMap<&str, AssetFingerprint> = old.iter().map(|(p, f)| (p.as_str(), *f)).collect();
    let new_map: HashMap<&str, AssetFingerprint> = new.iter().map(|(p, f)| (p.as_str(), *f)).collect();

    let mut changes = Vec::new();

    for (path, new_fp) in &new_map {
        match old_map.get(path) {
            None => {
                changes.push(DiffChange::MaterialChanged {
                    path: path.to_string(),
                    old_fingerprint: AssetFingerprint(0),
                    new_fingerprint: *new_fp,
                });
            }
            Some(old_fp) => {
                if old_fp != new_fp {
                    changes.push(DiffChange::MaterialChanged {
                        path: path.to_string(),
                        old_fingerprint: *old_fp,
                        new_fingerprint: *new_fp,
                    });
                }
            }
        }
    }

    changes
}

/// Compute a fingerprint for a serialized entity component set.
pub fn fingerprint_entity(components: &HashMap<String, String>) -> AssetFingerprint {
    let mut combined = String::new();
    let mut keys: Vec<&String> = components.keys().collect();
    keys.sort();
    for key in keys {
        combined.push_str(key);
        combined.push('=');
        combined.push_str(&components[key]);
        combined.push('\n');
    }
    AssetFingerprint::compute_str(&combined)
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_deterministic() {
        let mut comps = HashMap::new();
        comps.insert("Position".to_string(), "1,2,3".to_string());
        comps.insert("Health".to_string(), "100".to_string());
        let f1 = fingerprint_entity(&comps);
        let f2 = fingerprint_entity(&comps);
        assert_eq!(f1, f2);
    }

    #[test]
    fn fingerprint_changes_with_data() {
        let mut comps = HashMap::new();
        comps.insert("Health".to_string(), "100".to_string());
        let f1 = fingerprint_entity(&comps);
        comps.insert("Health".to_string(), "50".to_string());
        let f2 = fingerprint_entity(&comps);
        assert_ne!(f1, f2);
    }

    #[test]
    fn diff_detects_added_entities() {
        let old = vec![];
        let new = vec![
            DiffEntity { entity_id: 1, name: "new_entity".to_string(), components: HashMap::new(), fingerprint: AssetFingerprint(42) },
        ];
        let diff = diff_entities(&old, &new);
        assert_eq!(diff.added_count, 1);
        assert_eq!(diff.removed_count, 0);
    }

    #[test]
    fn diff_detects_removed_entities() {
        let old = vec![
            DiffEntity { entity_id: 1, name: "old_entity".to_string(), components: HashMap::new(), fingerprint: AssetFingerprint(42) },
        ];
        let new = vec![];
        let diff = diff_entities(&old, &new);
        assert_eq!(diff.added_count, 0);
        assert_eq!(diff.removed_count, 1);
    }

    #[test]
    fn diff_detects_modified_entities() {
        let mut comps1 = HashMap::new();
        comps1.insert("Health".to_string(), "100".to_string());
        let mut comps2 = HashMap::new();
        comps2.insert("Health".to_string(), "50".to_string());

        let old = vec![
            DiffEntity { entity_id: 1, name: "player".to_string(), components: comps1, fingerprint: AssetFingerprint(10) },
        ];
        let new = vec![
            DiffEntity { entity_id: 1, name: "player".to_string(), components: comps2, fingerprint: AssetFingerprint(20) },
        ];
        let diff = diff_entities(&old, &new);
        assert_eq!(diff.modified_count, 1);
        assert!(!diff.is_empty());
    }

    #[test]
    fn diff_detects_component_changes() {
        let mut comps1 = HashMap::new();
        comps1.insert("Position".to_string(), "1,2,3".to_string());
        comps1.insert("Health".to_string(), "100".to_string());
        let mut comps2 = HashMap::new();
        comps2.insert("Position".to_string(), "1,2,3".to_string());
        comps2.insert("Health".to_string(), "75".to_string());

        let old = vec![
            DiffEntity { entity_id: 1, name: "e".to_string(), components: comps1, fingerprint: AssetFingerprint(1) },
        ];
        let new = vec![
            DiffEntity { entity_id: 1, name: "e".to_string(), components: comps2, fingerprint: AssetFingerprint(2) },
        ];
        let diff = diff_entities(&old, &new);
        assert_eq!(diff.modified_count, 1);
        if let DiffChange::EntityModified { component_changes, .. } = &diff.changes[0] {
            assert_eq!(component_changes.len(), 1);
            assert_eq!(component_changes[0].component_name, "Health");
        } else {
            panic!("expected EntityModified");
        }
    }

    #[test]
    fn summary_format() {
        let diff = SceneDiff {
            changes: vec![], added_count: 3, removed_count: 1, modified_count: 2, asset_changes: 5,
        };
        assert_eq!(diff.summary(), "3 added, 1 removed, 2 modified, 5 asset changes");
    }
}
