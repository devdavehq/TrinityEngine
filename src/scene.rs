// src/scene.rs — scene module root
pub mod loader;
pub use loader::parse_scene;

use crate::assets::{AssetStore, Handle, Mesh};
use crate::components::{Position, Renderable, Script};
use hecs::World;
// SceneManager owns the scene file path and the current list of entities.
// When the file changes, rebuild() clears the world and respawns everything.
pub struct SceneManager {
    pub scene_path: String,
}

impl SceneManager {
    pub fn new(scene_path: &str) -> Self {
        Self {
            scene_path: scene_path.to_string(),
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
    ) -> Result<(), String> {
        // Clear ALL entities from the world.
        // World::clear() drops every entity and component.
        world.clear();

        // Parse the scene file.
        let entities = parse_scene(&self.scene_path)?;

        println!(
            "[Scene] Loading {} entities from {}",
            entities.len(),
            self.scene_path
        );

        for desc in entities {
            // Load (or reuse) the mesh for this entity.
            // mesh_cache maps file path → Handle so we don't load the
            // same .obj twice. Meshes are shared — entities just hold handles.
            let mesh_handle = if let Some(handle) = mesh_cache.get(&desc.mesh) {
                *handle // Dereference because Handle is Copy
            } else {
                let mesh =
                    Mesh::load(&desc.mesh).map_err(|e| format!("Scene mesh error: {}", e))?;
                let handle = meshes.add(mesh);
                mesh_cache.insert(desc.mesh.clone(), handle);
                handle
            };

            // Build the Renderable component with PBR values from the scene file.
            let renderable = Renderable {
                mesh: mesh_handle,
                color: desc.color,
                metallic: desc.metallic,
                roughness: desc.roughness,
                ao: desc.ao,
                scale: desc.scale,
            };

            // Spawn differently depending on whether a script is attached.
            if let Some(script_path) = desc.script {
                // Entity with script.
                world.spawn((
                    Position {
                        x: desc.position[0],
                        y: desc.position[1],
                        z: desc.position[2],
                    },
                    renderable,
                    Script { path: script_path },
                ));
            } else {
                // Static entity — no script.
                world.spawn((
                    Position {
                        x: desc.position[0],
                        y: desc.position[1],
                        z: desc.position[2],
                    },
                    renderable,
                ));
            }

            println!("[Scene]   Spawned: {}", desc.name);
        }

        Ok(())
    }
}
