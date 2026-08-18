// src/levels/manifest.rs
// ──────────────────────────────────────────────────────────────────────────────
// Level streaming manifest.
//
// A `levels.json` beside the project root describes which .scene files are
// registered as levels and how they stream. Each entry decides per level
// whether it participates in distance streaming at all:
//
//   • `persistent = true`   → always loaded, never streamed out. Use this for
//     small levels / the base world you always want in memory.
//   • `persistent = false`   → streamed: loads when the player gets within
//     `streaming_distance` of the level origin and unloads past
//     `unloading_distance` (the gap is the hysteresis band).
//
// The editor's Levels panel edits this file live via the LevelManager; the
// running game just reads the manifest at project load.
// ──────────────────────────────────────────────────────────────────────────────

use serde::{Deserialize, Serialize};

/// One streamed / persistent level declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LevelEntry {
    /// Display name (shown in the Levels panel).
    pub name: String,
    /// Path to the .scene file, relative to the project root.
    pub file: String,
    /// World-space origin the level's entities are placed at.
    pub origin: [f32; 3],
    /// Distance from the level origin at which the player triggers loading.
    pub streaming_distance: f32,
    /// Distance beyond which the level unloads (>= streaming_distance).
    pub unloading_distance: f32,
    /// `true` = always loaded, never streamed (the per-level "no streaming"
    /// switch — perfect for small levels).
    pub persistent: bool,
}

impl Default for LevelEntry {
    fn default() -> Self {
        Self {
            name: "Level".to_string(),
            file: "Content/Scenes/level.scene".to_string(),
            origin: [0.0, 0.0, 0.0],
            streaming_distance: 100.0,
            unloading_distance: 200.0,
            persistent: false,
        }
    }
}

/// A portal (Decima-style transition trigger): a volume that loads a target
/// level when the player walks into it — for doorways, cave entrances, and
/// one-way transitions. Optionally unloads the source level afterwards.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PortalEntry {
    /// Display name (shown in debug/tooling).
    pub name: String,
    /// World-space center of the trigger volume.
    pub position: [f32; 3],
    /// Trigger volume shape: "sphere", "box", or "capsule".
    pub shape: String,
    /// Sphere radius (used when shape == "sphere").
    pub trigger_radius: f32,
    /// Box half-extents (used when shape == "box").
    pub box_extents: [f32; 3],
    /// Capsule radius + half-height (used when shape == "capsule").
    pub capsule_radius: f32,
    pub capsule_half_height: f32,
    /// Level to load (must match a LevelEntry name).
    pub target_level: String,
    /// Level to unload when the portal fires (empty = keep everything).
    /// Use this for one-way transitions (dungeon entrance, etc.).
    pub source_level: String,
    /// Whether this portal is active. Disabled portals never fire.
    pub active: bool,
}

impl Default for PortalEntry {
    fn default() -> Self {
        Self {
            name: "Portal".to_string(),
            position: [0.0, 0.0, 0.0],
            shape: "sphere".to_string(),
            trigger_radius: 3.0,
            box_extents: [3.0, 3.0, 3.0],
            capsule_radius: 2.0,
            capsule_half_height: 2.0,
            target_level: String::new(),
            source_level: String::new(),
            active: true,
        }
    }
}

/// Full set of level declarations for a project.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LevelManifest {
    pub levels: Vec<LevelEntry>,
    /// Trigger portals that load/unload levels by proximity.
    pub portals: Vec<PortalEntry>,
}

impl Default for LevelManifest {
    fn default() -> Self {
        Self { levels: Vec::new(), portals: Vec::new() }
    }
}

impl LevelManifest {
    /// Load a manifest from a JSON file. A missing file is `Ok` with no levels.
    pub fn load(path: &str) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).map_err(|e| format!("bad level manifest: {}", e)),
            Err(_) => Ok(Self::default()),
        }
    }

    /// Write the manifest to a JSON file (pretty-printed for hand editing).
    pub fn save(&self, path: &str) -> Result<(), String> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trip() {
        let m = LevelManifest {
            levels: vec![
                LevelEntry {
                    name: "Village".into(),
                    file: "Content/Scenes/village.scene".into(),
                    origin: [10.0, 0.0, 20.0],
                    streaming_distance: 80.0,
                    unloading_distance: 160.0,
                    persistent: true,
                },
                LevelEntry::default(),
            ],
            portals: vec![],
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: LevelManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.levels.len(), 2);
        assert_eq!(back.levels[0].name, "Village");
        assert!(back.levels[0].persistent);
        assert_eq!(back.levels[1].streaming_distance, 100.0);
    }

    #[test]
    fn manifest_missing_file_is_empty() {
        let m = LevelManifest::load("definitely/not/here.json").unwrap();
        assert!(m.levels.is_empty());
    }
}
