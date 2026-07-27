// src/scene.rs — scene module root
// ──────────────────────────────────────────────────────────────────────────────
// Scene system: loading, saving, and managing .scene files.
//
// Sub-modules:
//   loader     — .scene file parser (INI-like format)
//   prefab     — reusable entity templates (.prefab files)
//   subscene   — load scenes INSIDE the current scene with position offsets
//   transition — fade-to-black visual effect for scene switches
// ──────────────────────────────────────────────────────────────────────────────
pub mod loader;
pub mod prefab;
pub mod subscene;
pub mod transition;
pub mod diff;
pub use loader::parse_scene;
pub use prefab::{Prefab, PrefabRegistry};
pub use subscene::SubSceneManager;
pub use transition::SceneTransition;
pub use diff::{SceneDiff, diff_entities, diff_materials, AssetFingerprint};

use crate::assets::{AssetStore, Handle, Mesh};
use crate::components::{Position, Renderable, Rotation, Script, SceneMeta, RigidBody, PointLight};
use hecs::World;
use loader::EntityDesc;
use std::collections::VecDeque;
use std::path::PathBuf;

// SceneManager owns the scene file path and the current list of entities.
// When the file changes, rebuild() clears the world and respawns everything.
pub struct SceneManager {
    pub scene_path: String,
    /// Whether the scene has been modified since last save.
    pub dirty: bool,
    /// Scene name shown in the title bar.
    pub scene_name: String,
    /// Recent scene paths (most recent first).
    recent: VecDeque<PathBuf>,
    /// Maximum number of recent scenes to track.
    max_recent: usize,
}

impl SceneManager {
    pub fn new(scene_path: &str) -> Self {
        let name = PathBuf::from(scene_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Scene")
            .to_string();
        let mut recent = VecDeque::new();
        recent.push_front(PathBuf::from(scene_path));
        Self {
            scene_path: scene_path.to_string(),
            dirty: false,
            scene_name: name,
            recent,
            max_recent: 10,
        }
    }

    // build() reads the scene file and spawns all described entities.
    // Called at startup and whenever the scene file changes.
    //
    // Why clear the whole world?
    // Simple and correct. For a production engine you'd diff the scene
    // and only respawn changed entities. For now, full rebuild is fine.
    pub fn build(
        &self,
        world: &mut World,
        meshes: &mut AssetStore<Mesh>,
        // We pass a mutable cache so meshes loaded for one entity
        // can be reused by the next entity with the same path.
        mesh_cache: &mut std::collections::HashMap<String, Handle<Mesh>>,
        // Optional prefab registry for resolving prefab references.
        prefabs: Option<&prefab::PrefabRegistry>,
    ) -> Result<(), String> {
        // Clear ALL entities from the world.
        // World::clear() drops every entity and component.
        world.clear();

        // Parse the scene file.
        let entities = parse_scene(&self.scene_path)?;

        tracing::info!(
            "[Scene] Loading {} entities from {}",
            entities.len(),
            self.scene_path
        );

        for desc in entities {
            // ── Prefab Resolution ──────────────────────────────────────────
            // If this entity references a prefab, load it and merge defaults.
            // Scene fields override prefab defaults (only non-default values win).
            let resolved = if let Some(prefab_path) = &desc.prefab {
                if let Some(registry) = prefabs {
                    if let Some(pf) = registry.get_by_path(prefab_path)
                        .or_else(|| registry.get_by_name(prefab_path))
                    {
                        // Merge: scene desc overrides prefab defaults.
                        // Only override fields that differ from EntityDesc defaults.
                        let mut merged = desc.clone();
                        let default_desc = EntityDesc::default();

                        // Mesh comes from prefab if scene didn't specify a different one
                        if merged.mesh == default_desc.mesh && merged.mesh != pf.mesh {
                            merged.mesh = pf.mesh.clone();
                        }
                        // Name comes from prefab if scene used default
                        if merged.name == default_desc.name {
                            merged.name = pf.name.clone();
                        }
                        // Material comes from prefab if scene didn't specify one
                        if merged.material.is_none() {
                            merged.material = pf.material.clone();
                        }
                        // Color/metallic/roughness: only override if prefab has non-default values
                        if pf.color != [1.0, 1.0, 1.0] && merged.color == default_desc.color {
                            merged.color = pf.color;
                        }
                        if (pf.metallic - 0.0).abs() > 0.001 && (merged.metallic - 0.0).abs() < 0.001 {
                            merged.metallic = pf.metallic;
                        }
                        if (pf.roughness - 0.5).abs() > 0.001 && (merged.roughness - 0.5).abs() < 0.001 {
                            merged.roughness = pf.roughness;
                        }
                        if (pf.ao - 1.0).abs() > 0.001 && (merged.ao - 1.0).abs() < 0.001 {
                            merged.ao = pf.ao;
                        }
                        // Rigidbody from prefab if scene didn't specify
                        if merged.rigidbody.is_none() {
                            merged.rigidbody = pf.rigidbody;
                        }
                        // Light from prefab if scene didn't specify
                        if merged.light.is_none() {
                            if let Some((ref ltype, color, intensity, range)) = pf.light {
                                merged.light = Some(loader::LightDesc {
                                    light_type: ltype.clone(),
                                    color,
                                    intensity,
                                    range,
                                });
                            }
                        }
                        // Script from prefab if scene didn't specify
                        if merged.script.is_none() {
                            merged.script = pf.script.clone();
                        }
                        merged
                    } else {
                        tracing::error!("[Scene] Prefab not found: {}", prefab_path);
                        desc
                    }
                } else {
                    desc
                }
            } else {
                desc
            };
            // Load (or reuse) the mesh for this entity.
            // mesh_cache maps file path → Handle so we don't load the
            // same .obj twice. Meshes are shared — entities just hold handles.
            let mesh_handle = if let Some(handle) = mesh_cache.get(&resolved.mesh) {
                *handle // Dereference because Handle is Copy
            } else {
                let mesh =
                    Mesh::load(&resolved.mesh).map_err(|e| format!("Scene mesh error: {}", e))?;
                let handle = meshes.add(mesh);
                mesh_cache.insert(resolved.mesh.clone(), handle);
                handle
            };

            // Build the Renderable component with PBR values from the scene file.
            // If a material name is specified, the color/metallic/roughness values
            // are ignored — the material library will fill them in later.
            let renderable = Renderable {
                mesh: mesh_handle,
                color: resolved.color,
                metallic: resolved.metallic,
                roughness: resolved.roughness,
                ao: resolved.ao,
                scale: resolved.scale,
            };

            // Rotation — Euler angles converted from degrees to radians.
            let rotation = Rotation {
                pitch: resolved.rotation[0].to_radians(),
                yaw:   resolved.rotation[1].to_radians(),
                roll:  resolved.rotation[2].to_radians(),
            };

            // Spawn the entity with all applicable components.
            // hecs doesn't support dynamic component lists, so we branch
            // on which optional components are present.
            if resolved.script.is_some() {
                let mut body = resolved.rigidbody.map(|mass| {
                    let mut b = RigidBody::dynamic();
                    if mass <= 0.0 { b = RigidBody::static_body(); } else { b.mass = mass; }
                    b
                });
                let mut light = resolved.light.map(|l| PointLight {
                    color: l.color, intensity: l.intensity, range: l.range,
                    light_type: 1.0, spot_angle: 45.0, shadow_casting: false,
                });
                let ent = world.spawn((
                    Position { x: resolved.position[0], y: resolved.position[1], z: resolved.position[2] },
                    renderable,
                    rotation,
                    SceneMeta { name: resolved.name.clone(), mesh_path: resolved.mesh.clone() },
                    Script { path: resolved.script.unwrap() },
                ));
                if let Some(b) = body.take() {
                    let _ = world.insert(ent, (b,));
                }
                if let Some(l) = light.take() {
                    let _ = world.insert(ent, (l,));
                }
            } else {
                let mut body = resolved.rigidbody.map(|mass| {
                    let mut b = RigidBody::dynamic();
                    if mass <= 0.0 { b = RigidBody::static_body(); } else { b.mass = mass; }
                    b
                });
                let mut light = resolved.light.map(|l| PointLight {
                    color: l.color, intensity: l.intensity, range: l.range,
                    light_type: 1.0, spot_angle: 45.0, shadow_casting: false,
                });
                let ent = world.spawn((
                    Position { x: resolved.position[0], y: resolved.position[1], z: resolved.position[2] },
                    renderable,
                    rotation,
                    SceneMeta { name: resolved.name.clone(), mesh_path: resolved.mesh.clone() },
                ));
                if let Some(b) = body.take() {
                    let _ = world.insert(ent, (b,));
                }
                if let Some(l) = light.take() {
                    let _ = world.insert(ent, (l,));
                }
            }

            tracing::info!("[Scene]   Spawned: {}", resolved.name);
        }

        Ok(())
    }

    // ── Scene Management Methods ─────────────────────────────────────────

    /// Create a new empty scene. Returns the action to take.
    pub fn new_scene(&mut self) -> SceneAction {
        self.scene_path = String::new();
        self.dirty = false;
        self.scene_name = "Untitled".to_string();
        SceneAction::ClearAll
    }

    /// Load a scene from a file path.
    pub fn load_scene(&mut self, path: &str) -> SceneAction {
        let pb = PathBuf::from(path);
        let name = pb
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Scene")
            .to_string();

        self.add_recent(&pb);
        self.scene_path = path.to_string();
        self.dirty = false;
        self.scene_name = name;

        SceneAction::LoadScene(path.to_string())
    }

    /// Save the current scene. Returns the path to save to.
    pub fn save(&mut self, path: Option<&str>) -> Option<String> {
        let save_path = if let Some(p) = path {
            p.to_string()
        } else if !self.scene_path.is_empty() {
            self.scene_path.clone()
        } else {
            return None;
        };

        self.dirty = false;
        self.scene_path = save_path.clone();
        self.scene_name = PathBuf::from(&save_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Scene")
            .to_string();
        self.add_recent(&PathBuf::from(&save_path));

        Some(save_path)
    }

    /// Save as a new file path.
    pub fn save_as(&mut self, path: &str) -> Option<String> {
        self.save(Some(path))
    }

    /// Navigate back to the previous scene.
    ///
    /// Returns the path to the previous scene if available.
    /// Uses the recent scenes list: index 0 is current, index 1 is previous.
    /// Useful for "go back" UI buttons or keyboard shortcuts.
    pub fn go_back(&mut self) -> Option<String> {
        if self.recent.len() > 1 {
            // Skip current scene (index 0), go to index 1.
            let prev = self.recent[1].clone();
            tracing::info!(
                "[Scene] Going back from '{}' to '{}'",
                self.scene_name,
                prev.to_string_lossy()
            );
            // Re-order the recent list so the previous scene becomes current.
            let prev_path = prev.to_string_lossy().to_string();
            self.load_scene(&prev_path);
            Some(prev_path)
        } else {
            tracing::info!("[Scene] No previous scene to go back to");
            None
        }
    }

    /// Get the previous scene path (if any), without navigating to it.
    pub fn previous_scene(&self) -> Option<&std::path::PathBuf> {
        self.recent.get(1)
    }

    /// Mark the scene as modified.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Get the list of recent scenes.
    pub fn recent_scenes(&self) -> impl Iterator<Item = &PathBuf> {
        self.recent.iter()
    }

    /// Get the title bar string (includes dirty indicator).
    pub fn title(&self) -> String {
        let dirty_marker = if self.dirty { " *" } else { "" };
        if !self.scene_path.is_empty() {
            format!(
                "{} - {}{}",
                self.scene_name,
                self.scene_path,
                dirty_marker
            )
        } else {
            format!("{}{}", self.scene_name, dirty_marker)
        }
    }

    /// List all .scene files in a directory.
    pub fn list_scene_files(dir: &str) -> Vec<String> {
        let mut scenes = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "scene").unwrap_or(false) {
                    if let Some(s) = path.to_str() {
                        scenes.push(s.to_string());
                    }
                }
            }
        }
        scenes.sort();
        scenes
    }

    fn add_recent(&mut self, path: &PathBuf) {
        self.recent.retain(|p| p != path);
        self.recent.push_front(path.clone());
        while self.recent.len() > self.max_recent {
            self.recent.pop_back();
        }
    }
}

/// Actions the main loop should take in response to scene operations.
#[derive(Debug, Clone)]
pub enum SceneAction {
    /// Clear all entities from the world.
    ClearAll,
    /// Load a scene from the given path.
    LoadScene(String),
    /// No action needed.
    None,
}

/// Save all entities in the world back to a .scene file.
/// Each entity needs a SceneMeta + Position + Renderable component.
/// Entities without SceneMeta are skipped (they weren't loaded from a file).
pub fn save_scene(
    scene_path: &str,
    world: &mut World,
) -> Result<(), String> {
    use crate::components::SceneMeta;

    let mut lines: Vec<String> = Vec::new();
    let mut count: usize = 0;

    for (pos, renderable, meta) in
        world.query::<(&Position, &Renderable, &SceneMeta)>().iter()
    {
        lines.push("[entity]".to_string());
        lines.push(format!("name = {}", meta.name));
        lines.push(format!("mesh = {}", meta.mesh_path));
        lines.push(format!("position = {} {} {}", pos.x, pos.y, pos.z));
        lines.push(format!(
            "scale = {} {} {}",
            renderable.scale[0], renderable.scale[1], renderable.scale[2]
        ));
        lines.push(format!(
            "color = {} {} {}",
            renderable.color[0], renderable.color[1], renderable.color[2]
        ));
        lines.push(format!("metallic = {}", renderable.metallic));
        lines.push(format!("roughness = {}", renderable.roughness));
        lines.push(format!("ao = {}", renderable.ao));
        lines.push(String::new());
        count += 1;
    }

    // Second pass: collect scripts to inject into entity blocks.
    let mut scripts: Vec<(f32, f32, f32, String)> = Vec::new();
    for (pos, script) in world.query_mut::<(&Position, &Script)>() {
        scripts.push((pos.x, pos.y, pos.z, script.path.clone()));
    }
    for (sx, sy, sz, script_path) in &scripts {
        if let Some(idx) = lines.iter().position(|l| {
            l.starts_with("position = ")
                && l.contains(&format!("{} {} {}", sx, sy, sz))
        }) {
            lines.insert(idx + 1, format!("script = {}", script_path));
        }
    }

    let content = lines.join("\n");
    std::fs::write(scene_path, content)
        .map_err(|e| format!("Failed to save scene {}: {}", scene_path, e))?;

    tracing::info!("[Scene] Saved {} entities to {}", count, scene_path);
    Ok(())
}
