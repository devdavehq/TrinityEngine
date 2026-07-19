// src/scene/subscene.rs
// ──────────────────────────────────────────────────────────────────────────────
// Sub-scene System — load scenes INSIDE the current scene.
//
// WHY:
//   SceneManager::build() clears the entire world before loading a new scene.
//   Sub-scenes let you compose a level from multiple .scene files without
//   destroying the parent. Each sub-scene spawns its own entities at a world-
//   space offset, and can be loaded/unloaded/toggled independently.
//
// USE CASES:
//   • Streaming: load nearby rooms, unload distant ones
//   • Modding: let users drop .scene files into a folder as addons
//   • Level-of-detail: swap a high-poly room for a low-poly one at distance
//   • Game logic: dynamically spawn encounter arenas at runtime
//
// DATA FLOW:
//   .scene file → SubSceneManager::load_sub_scene() → hecs::World (with offset)
//
// Each SubScene tracks its spawned entity IDs so it can remove them cleanly
// when unloaded. The offset is applied to every entity's position at spawn
// time — the original .scene file coordinates are preserved relative to the
// sub-scene origin.
// ──────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::path::PathBuf;

use crate::assets::{AssetStore, Handle, Mesh};
use crate::components::*;
use crate::scene::loader::EntityDesc;
use crate::scene::prefab::PrefabRegistry;

/// A loaded sub-scene instance within the current scene.
///
/// Each instance represents one .scene file that was spawned into the world
/// at a specific offset. The entity_ids list lets us remove them later without
/// touching entities from the parent scene or other sub-scenes.
pub struct SubScene {
    /// Display name derived from the .scene file stem (e.g. "arena_01").
    pub name: String,
    /// Path to the .scene file this was loaded from.
    pub scene_path: String,
    /// World-space offset where this sub-scene was placed.
    /// Added to every entity's position at spawn time.
    pub offset: [f32; 3],
    /// Entity IDs spawned by this sub-scene.
    /// Used for targeted despawn on unload.
    pub entity_ids: Vec<hecs::Entity>,
    /// Whether this sub-scene is currently active/visible.
    /// Inactive sub-scenes have their entities hidden but not despawned.
    pub active: bool,
}

/// Manages all sub-scenes loaded into the current scene.
///
/// The parent scene's SceneManager handles the primary scene. SubSceneManager
/// handles additional scenes composited on top via load_sub_scene().
pub struct SubSceneManager {
    /// All loaded sub-scene instances.
    pub instances: Vec<SubScene>,
}

impl SubSceneManager {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
        }
    }

    /// Load a .scene file as a sub-scene at the given world-space offset.
    ///
    /// This parses the .scene file, resolves prefabs, loads meshes, and spawns
    /// every entity with `position += offset`. Unlike SceneManager::build(),
    /// this does NOT clear the world — existing entities are preserved.
    ///
    /// Returns the number of entities spawned on success.
    pub fn load_sub_scene(
        &mut self,
        scene_path: &str,
        offset: [f32; 3],
        world: &mut hecs::World,
        meshes: &mut AssetStore<Mesh>,
        mesh_cache: &mut HashMap<String, Handle<Mesh>>,
        prefabs: Option<&PrefabRegistry>,
    ) -> Result<usize, String> {
        // Parse the .scene file into entity descriptors.
        let entities = crate::scene::parse_scene(scene_path)?;

        // Derive a display name from the file stem (e.g. "arena_01" from "arena_01.scene").
        let name = PathBuf::from(scene_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("SubScene")
            .to_string();

        tracing::info!(
            "[SubScene] Loading '{}' from {} ({} entities, offset {:?})",
            name,
            scene_path,
            entities.len(),
            offset
        );

        let mut entity_ids = Vec::new();

        // ── Spawn each entity with offset ────────────────────────────────
        // This mirrors SceneManager::build() logic: prefab resolution,
        // mesh caching, material/RigidBody/Script/Light attachment.
        for desc in entities {
            // ── Prefab Resolution ────────────────────────────────────────
            // If this entity references a prefab, merge its defaults.
            // Scene fields override prefab defaults (only non-default values win).
            let resolved = if let Some(prefab_path) = &desc.prefab {
                if let Some(registry) = prefabs {
                    if let Some(pf) = registry
                        .get_by_path(prefab_path)
                        .or_else(|| registry.get_by_name(prefab_path))
                    {
                        let mut merged = desc.clone();
                        let default_desc = EntityDesc::default();

                        if merged.mesh == default_desc.mesh && merged.mesh != pf.mesh {
                            merged.mesh = pf.mesh.clone();
                        }
                        if merged.name == default_desc.name {
                            merged.name = pf.name.clone();
                        }
                        if merged.material.is_none() {
                            merged.material = pf.material.clone();
                        }
                        if pf.color != [1.0, 1.0, 1.0] && merged.color == default_desc.color {
                            merged.color = pf.color;
                        }
                        if (pf.metallic - 0.0).abs() > 0.001
                            && (merged.metallic - 0.0).abs() < 0.001
                        {
                            merged.metallic = pf.metallic;
                        }
                        if (pf.roughness - 0.5).abs() > 0.001
                            && (merged.roughness - 0.5).abs() < 0.001
                        {
                            merged.roughness = pf.roughness;
                        }
                        if (pf.ao - 1.0).abs() > 0.001 && (merged.ao - 1.0).abs() < 0.001 {
                            merged.ao = pf.ao;
                        }
                        if merged.rigidbody.is_none() {
                            merged.rigidbody = pf.rigidbody;
                        }
                        if merged.light.is_none() {
                            if let Some((ref ltype, color, intensity, range)) = pf.light {
                                merged.light = Some(crate::scene::loader::LightDesc {
                                    light_type: ltype.clone(),
                                    color,
                                    intensity,
                                    range,
                                });
                            }
                        }
                        if merged.script.is_none() {
                            merged.script = pf.script.clone();
                        }
                        merged
                    } else {
                        tracing::error!("[SubScene] Prefab not found: {}", prefab_path);
                        desc
                    }
                } else {
                    desc
                }
            } else {
                desc
            };

            // ── Mesh Loading (cached) ────────────────────────────────────
            // Reuse meshes already loaded by the parent scene or this sub-scene.
            let mesh_handle = if let Some(handle) = mesh_cache.get(&resolved.mesh) {
                *handle
            } else {
                let mesh = Mesh::load(&resolved.mesh)
                    .map_err(|e| format!("SubScene mesh error: {}", e))?;
                let handle = meshes.add(mesh);
                mesh_cache.insert(resolved.mesh.clone(), handle);
                handle
            };

            // ── Build Renderable with PBR values ────────────────────────
            let renderable = Renderable {
                mesh: mesh_handle,
                color: resolved.color,
                metallic: resolved.metallic,
                roughness: resolved.roughness,
                ao: resolved.ao,
                scale: resolved.scale,
            };

            // ── Rotation (degrees → radians) ────────────────────────────
            let rotation = Rotation {
                pitch: resolved.rotation[0].to_radians(),
                yaw: resolved.rotation[1].to_radians(),
                roll: resolved.rotation[2].to_radians(),
            };

            // ── Position WITH offset applied ─────────────────────────────
            // The key difference from SceneManager::build(): every entity
            // position is shifted by the sub-scene's world-space offset.
            let position = Position {
                x: resolved.position[0] + offset[0],
                y: resolved.position[1] + offset[1],
                z: resolved.position[2] + offset[2],
            };

            // ── Spawn entity with optional components ────────────────────
            // hecs doesn't support dynamic component lists, so we branch
            // on which optional components are present (same pattern as build()).
            if resolved.script.is_some() {
                let mut body = resolved.rigidbody.map(|mass| {
                    let mut b = RigidBody::dynamic();
                    if mass <= 0.0 {
                        b = RigidBody::static_body();
                    } else {
                        b.mass = mass;
                    }
                    b
                });
                let mut light = resolved.light.map(|l| PointLight {
                    color: l.color,
                    intensity: l.intensity,
                    range: l.range,
                    light_type: 1.0,
                    spot_angle: 45.0,
                    shadow_casting: false,
                });
                let ent = world.spawn((
                    position,
                    renderable,
                    rotation,
                    SceneMeta {
                        name: resolved.name.clone(),
                        mesh_path: resolved.mesh.clone(),
                    },
                    Script {
                        path: resolved.script.unwrap(),
                    },
                ));
                if let Some(b) = body.take() {
                    let _ = world.insert(ent, (b,));
                }
                if let Some(l) = light.take() {
                    let _ = world.insert(ent, (l,));
                }
                entity_ids.push(ent);
            } else {
                let mut body = resolved.rigidbody.map(|mass| {
                    let mut b = RigidBody::dynamic();
                    if mass <= 0.0 {
                        b = RigidBody::static_body();
                    } else {
                        b.mass = mass;
                    }
                    b
                });
                let mut light = resolved.light.map(|l| PointLight {
                    color: l.color,
                    intensity: l.intensity,
                    range: l.range,
                    light_type: 1.0,
                    spot_angle: 45.0,
                    shadow_casting: false,
                });
                let ent = world.spawn((
                    position,
                    renderable,
                    rotation,
                    SceneMeta {
                        name: resolved.name.clone(),
                        mesh_path: resolved.mesh.clone(),
                    },
                ));
                if let Some(b) = body.take() {
                    let _ = world.insert(ent, (b,));
                }
                if let Some(l) = light.take() {
                    let _ = world.insert(ent, (l,));
                }
                entity_ids.push(ent);
            }

            tracing::info!("[SubScene]   Spawned: {} @ {:?}", resolved.name, offset);
        }

        // ── Register the sub-scene instance ──────────────────────────────
        let count = entity_ids.len();
        self.instances.push(SubScene {
            name,
            scene_path: scene_path.to_string(),
            offset,
            entity_ids,
            active: true,
        });

        tracing::info!(
            "[SubScene] Loaded '{}' — {} entities",
            self.instances.last().unwrap().name,
            count
        );

        Ok(count)
    }

    /// Remove a sub-scene by index, despawning all its entities.
    ///
    /// This permanently removes the sub-scene and its entities from the world.
    /// To temporarily hide without destroying, use toggle_sub_scene() instead.
    pub fn unload_sub_scene(&mut self, index: usize, world: &mut hecs::World) {
        if index < self.instances.len() {
            let sub = self.instances.remove(index);
            let name = sub.name.clone();
            let count = sub.entity_ids.len();
            for entity in sub.entity_ids {
                let _ = world.despawn(entity);
            }
            tracing::info!(
                "[SubScene] Unloaded '{}' — {} entities despawned",
                name,
                count
            );
        }
    }

    /// Unload a sub-scene by name instead of index.
    pub fn unload_by_name(&mut self, name: &str, world: &mut hecs::World) {
        if let Some(idx) = self.instances.iter().position(|s| s.name == name) {
            self.unload_sub_scene(idx, world);
        } else {
            tracing::warn!("[SubScene] No sub-scene named '{}'", name);
        }
    }

    /// Toggle a sub-scene's active state (show/hide).
    ///
    /// When toggled off, all entities from this sub-scene are despawned but
    /// the SubScene metadata is kept so it can be reloaded later.
    /// When toggled on, the sub-scene is re-parsed and re-spawned at its
    /// original offset.
    pub fn toggle_sub_scene(
        &mut self,
        index: usize,
        world: &mut hecs::World,
        meshes: &mut AssetStore<Mesh>,
        mesh_cache: &mut HashMap<String, Handle<Mesh>>,
        prefabs: Option<&PrefabRegistry>,
    ) {
        if index >= self.instances.len() {
            return;
        }

        // Read the data we need before taking the mutable borrow.
        let was_active = self.instances[index].active;
        let path = self.instances[index].scene_path.clone();
        let offset = self.instances[index].offset;

        if was_active {
            // ── Deactivating: despawn all entities ────────────────────────
            self.instances[index].active = false;
            let entity_count = self.instances[index].entity_ids.len();
            let entity_ids: Vec<hecs::Entity> =
                self.instances[index].entity_ids.drain(..).collect();
            for entity in entity_ids {
                let _ = world.despawn(entity);
            }
            tracing::info!(
                "[SubScene] Hidden '{}' — {} entities despawned",
                self.instances[index].name,
                entity_count
            );
        } else {
            // ── Activating: re-parse and re-spawn at the same offset ─────
            // First mark active, then spawn (avoids borrow conflict).
            self.instances[index].active = true;
            match self.spawn_sub_scene_entities(
                &path,
                offset,
                world,
                meshes,
                mesh_cache,
                prefabs,
            ) {
                Ok(entity_ids) => {
                    let name = self.instances[index].name.clone();
                    self.instances[index].entity_ids = entity_ids;
                    tracing::info!(
                        "[SubScene] Shown '{}' — {} entities spawned",
                        name,
                        self.instances[index].entity_ids.len()
                    );
                }
                Err(e) => {
                    tracing::error!(
                        "[SubScene] Failed to re-spawn '{}': {}",
                        self.instances[index].name,
                        e
                    );
                    // Mark as inactive since we couldn't spawn the entities.
                    self.instances[index].active = false;
                }
            }
        }
    }

    /// Internal helper: parse and spawn entities without creating a new SubScene entry.
    /// Returns the list of spawned entity IDs.
    fn spawn_sub_scene_entities(
        &self,
        scene_path: &str,
        offset: [f32; 3],
        world: &mut hecs::World,
        meshes: &mut AssetStore<Mesh>,
        mesh_cache: &mut HashMap<String, Handle<Mesh>>,
        prefabs: Option<&PrefabRegistry>,
    ) -> Result<Vec<hecs::Entity>, String> {
        let entities = crate::scene::parse_scene(scene_path)?;
        let mut entity_ids = Vec::new();

        for desc in entities {
            // ── Prefab Resolution (same as load_sub_scene) ───────────────
            let resolved = if let Some(prefab_path) = &desc.prefab {
                if let Some(registry) = prefabs {
                    if let Some(pf) = registry
                        .get_by_path(prefab_path)
                        .or_else(|| registry.get_by_name(prefab_path))
                    {
                        let mut merged = desc.clone();
                        let default_desc = EntityDesc::default();
                        if merged.mesh == default_desc.mesh && merged.mesh != pf.mesh {
                            merged.mesh = pf.mesh.clone();
                        }
                        if merged.name == default_desc.name {
                            merged.name = pf.name.clone();
                        }
                        if merged.material.is_none() {
                            merged.material = pf.material.clone();
                        }
                        if pf.color != [1.0, 1.0, 1.0] && merged.color == default_desc.color {
                            merged.color = pf.color;
                        }
                        if (pf.metallic - 0.0).abs() > 0.001
                            && (merged.metallic - 0.0).abs() < 0.001
                        {
                            merged.metallic = pf.metallic;
                        }
                        if (pf.roughness - 0.5).abs() > 0.001
                            && (merged.roughness - 0.5).abs() < 0.001
                        {
                            merged.roughness = pf.roughness;
                        }
                        if (pf.ao - 1.0).abs() > 0.001 && (merged.ao - 1.0).abs() < 0.001 {
                            merged.ao = pf.ao;
                        }
                        if merged.rigidbody.is_none() {
                            merged.rigidbody = pf.rigidbody;
                        }
                        if merged.light.is_none() {
                            if let Some((ref ltype, color, intensity, range)) = pf.light {
                                merged.light = Some(crate::scene::loader::LightDesc {
                                    light_type: ltype.clone(),
                                    color,
                                    intensity,
                                    range,
                                });
                            }
                        }
                        if merged.script.is_none() {
                            merged.script = pf.script.clone();
                        }
                        merged
                    } else {
                        desc
                    }
                } else {
                    desc
                }
            } else {
                desc
            };

            // ── Mesh (cached) ────────────────────────────────────────────
            let mesh_handle = if let Some(handle) = mesh_cache.get(&resolved.mesh) {
                *handle
            } else {
                let mesh = Mesh::load(&resolved.mesh)
                    .map_err(|e| format!("SubScene mesh error: {}", e))?;
                let handle = meshes.add(mesh);
                mesh_cache.insert(resolved.mesh.clone(), handle);
                handle
            };

            let renderable = Renderable {
                mesh: mesh_handle,
                color: resolved.color,
                metallic: resolved.metallic,
                roughness: resolved.roughness,
                ao: resolved.ao,
                scale: resolved.scale,
            };

            let rotation = Rotation {
                pitch: resolved.rotation[0].to_radians(),
                yaw: resolved.rotation[1].to_radians(),
                roll: resolved.rotation[2].to_radians(),
            };

            let position = Position {
                x: resolved.position[0] + offset[0],
                y: resolved.position[1] + offset[1],
                z: resolved.position[2] + offset[2],
            };

            // ── Spawn with optional components ───────────────────────────
            if resolved.script.is_some() {
                let mut body = resolved.rigidbody.map(|mass| {
                    let mut b = RigidBody::dynamic();
                    if mass <= 0.0 {
                        b = RigidBody::static_body();
                    } else {
                        b.mass = mass;
                    }
                    b
                });
                let mut light = resolved.light.map(|l| PointLight {
                    color: l.color,
                    intensity: l.intensity,
                    range: l.range,
                    light_type: 1.0,
                    spot_angle: 45.0,
                    shadow_casting: false,
                });
                let ent = world.spawn((
                    position,
                    renderable,
                    rotation,
                    SceneMeta {
                        name: resolved.name.clone(),
                        mesh_path: resolved.mesh.clone(),
                    },
                    Script {
                        path: resolved.script.unwrap(),
                    },
                ));
                if let Some(b) = body.take() {
                    let _ = world.insert(ent, (b,));
                }
                if let Some(l) = light.take() {
                    let _ = world.insert(ent, (l,));
                }
                entity_ids.push(ent);
            } else {
                let mut body = resolved.rigidbody.map(|mass| {
                    let mut b = RigidBody::dynamic();
                    if mass <= 0.0 {
                        b = RigidBody::static_body();
                    } else {
                        b.mass = mass;
                    }
                    b
                });
                let mut light = resolved.light.map(|l| PointLight {
                    color: l.color,
                    intensity: l.intensity,
                    range: l.range,
                    light_type: 1.0,
                    spot_angle: 45.0,
                    shadow_casting: false,
                });
                let ent = world.spawn((
                    position,
                    renderable,
                    rotation,
                    SceneMeta {
                        name: resolved.name.clone(),
                        mesh_path: resolved.mesh.clone(),
                    },
                ));
                if let Some(b) = body.take() {
                    let _ = world.insert(ent, (b,));
                }
                if let Some(l) = light.take() {
                    let _ = world.insert(ent, (l,));
                }
                entity_ids.push(ent);
            }
        }

        Ok(entity_ids)
    }

    /// Get the count of loaded sub-scenes.
    pub fn count(&self) -> usize {
        self.instances.len()
    }

    /// Get info about a sub-scene by index.
    pub fn get(&self, index: usize) -> Option<&SubScene> {
        self.instances.get(index)
    }

    /// Check if a sub-scene is active.
    pub fn is_active(&self, index: usize) -> bool {
        self.instances.get(index).map_or(false, |s| s.active)
    }

    /// Get all sub-scene names.
    pub fn names(&self) -> Vec<&str> {
        self.instances.iter().map(|s| s.name.as_str()).collect()
    }
}
