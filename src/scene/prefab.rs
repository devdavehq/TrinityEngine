// src/scene/prefab.rs
// Prefab System — reusable entity templates loaded from .prefab files.
//
// WHY:
//   Without prefabs, every entity in a scene must be defined from scratch.
//   Prefabs let designers create a "wooden crate" template once, then place
//   100 instances in the scene with different positions/rotations/scales.
//
// DATA FLOW:
//   .prefab file → PrefabRegistry → scene builder → ECS World
//
//   A .prefab file has the same format as an [entity] block in .scene files,
//   but with an added "prefab" name at the top. Scene files can reference
//   prefabs with `prefab = path/to/file.prefab` instead of defining all
//   fields inline.
//
// FILE FORMAT:
//   # Wooden crate prefab
//   name     = wooden_crate
//   mesh     = meshes/cube.obj
//   material = dark_wood
//   scale    = 1.0  1.0  1.0
//   rigidbody = 2.0
//
//   The scene file then uses:
//   [entity]
//   prefab  = Content/Prefabs/wooden_crate.prefab
//   position = 5.0  1.0  3.0
//   rotation = 0.0  45.0  0.0
//
//   Position/rotation/scale in the scene override the prefab defaults.
// ─────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A loaded prefab template. Stores the default component values
/// that can be overridden when placing in a scene.
#[derive(Debug, Clone)]
pub struct Prefab {
    /// Friendly name (e.g. "wooden_crate").
    pub name: String,
    /// Mesh file path.
    pub mesh: String,
    /// Material instance name (from MaterialLibrary).
    pub material: Option<String>,
    /// Default position (overridden by scene placement).
    pub position: [f32; 3],
    /// Default rotation in degrees (overridden by scene placement).
    pub rotation: [f32; 3],
    /// Default scale (overridden by scene placement).
    pub scale: [f32; 3],
    /// Default base color.
    pub color: [f32; 3],
    /// Default metallic value.
    pub metallic: f32,
    /// Default roughness value.
    pub roughness: f32,
    /// Default AO value.
    pub ao: f32,
    /// Optional rigidbody mass (0 = static).
    pub rigidbody: Option<f32>,
    /// Optional point light.
    pub light: Option<(String, [f32; 3], f32, f32)>, // type, color, intensity, range
    /// Optional script path.
    pub script: Option<String>,
    /// Source file path (for debugging and hot-reload).
    pub file_path: PathBuf,
}

/// Registry of loaded prefabs, keyed by file path or name.
pub struct PrefabRegistry {
    /// path → Prefab
    by_path: HashMap<String, Prefab>,
    /// name → path (for name-based lookups)
    by_name: HashMap<String, String>,
}

impl PrefabRegistry {
    pub fn new() -> Self {
        Self {
            by_path: HashMap::new(),
            by_name: HashMap::new(),
        }
    }

    /// Scan a directory for .prefab files and load them all.
    /// Enumerates through the VFS so prefabs ship inside a packed game.pak too.
    pub fn load_from_directory(&mut self, dir: impl AsRef<Path>) {
        let dir = dir.as_ref().to_string_lossy().to_string();
        if !crate::vfs::exists(&dir) {
            return;
        }
        let mut files: Vec<String> = crate::vfs::walk_dir(&dir)
            .unwrap_or_default()
            .into_iter()
            .filter(|rel| rel.ends_with(".prefab"))
            .map(|rel| format!("{}/{}", dir.trim_end_matches('/'), rel))
            .collect();
        files.sort();
        for path_str in files {
            if let Err(e) = self.load_file(Path::new(&path_str)) {
                tracing::error!("[Prefab] Failed to load {}: {}", path_str, e);
            }
        }
        tracing::info!(
            "[Prefabs] Loaded {} prefabs from {:?}",
            self.by_path.len(),
            dir
        );
    }

    /// Load a single .prefab file.
    pub fn load_file(&mut self, path: &Path) -> Result<(), String> {
        let path_str = path.to_string_lossy().to_string();
        let contents =
            crate::vfs::read_to_string(&path_str).map_err(|e| format!("Read failed: {}", e))?;
        let prefab = parse_prefab(&contents, path)?;
        let path_str = path.to_string_lossy().to_string();
        self.by_name
            .insert(prefab.name.clone(), path_str.clone());
        self.by_path.insert(path_str, prefab);
        Ok(())
    }

    /// Reload a prefab file that changed on disk (hot-reload).
    pub fn reload_file(&mut self, path: &Path) -> Result<(), String> {
        self.load_file(path)
    }

    /// Look up a prefab by file path.
    pub fn get_by_path(&self, path: &str) -> Option<&Prefab> {
        self.by_path.get(path)
    }

    /// Look up a prefab by name.
    pub fn get_by_name(&self, name: &str) -> Option<&Prefab> {
        self.by_name
            .get(name)
            .and_then(|p| self.by_path.get(p))
    }

    pub fn count(&self) -> usize {
        self.by_path.len()
    }
}

/// Parse a .prefab file into a Prefab struct.
fn parse_prefab(contents: &str, file_path: &Path) -> Result<Prefab, String> {
    let mut name = String::new();
    let mut mesh = "meshes/cube.obj".to_string();
    let mut material: Option<String> = None;
    let mut position = [0.0f32; 3];
    let mut rotation = [0.0f32; 3];
    let mut scale = [1.0f32; 3];
    let mut color = [1.0f32; 3];
    let mut metallic = 0.0f32;
    let mut roughness = 0.5f32;
    let mut ao = 1.0f32;
    let mut rigidbody: Option<f32> = None;
    let mut light: Option<(String, [f32; 3], f32, f32)> = None;
    let mut script: Option<String> = None;

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        let mut parts = line.splitn(2, '=');
        let key = parts.next().unwrap_or("").trim();
        let value = parts.next().unwrap_or("").trim();

        match key {
            "name" => name = value.trim_matches('"').to_string(),
            "mesh" => mesh = value.trim_matches('"').to_string(),
            "material" => material = Some(value.trim_matches('"').to_string()),
            "script" => script = Some(value.trim_matches('"').to_string()),
            "metallic" => metallic = parse_f32(value),
            "roughness" => roughness = parse_f32(value),
            "ao" => ao = parse_f32(value),
            "position" => {
                if let Some(arr) = parse_f32_array(value) {
                    if arr.len() >= 3 {
                        position = [arr[0], arr[1], arr[2]];
                    }
                }
            }
            "rotation" => {
                if let Some(arr) = parse_f32_array(value) {
                    if arr.len() >= 3 {
                        rotation = [arr[0], arr[1], arr[2]];
                    } else if arr.len() == 1 {
                        rotation = [0.0, arr[0], 0.0];
                    }
                }
            }
            "scale" => {
                if let Some(arr) = parse_f32_array(value) {
                    if arr.len() >= 3 {
                        scale = [arr[0], arr[1], arr[2]];
                    } else if arr.len() == 1 {
                        scale = [arr[0], arr[0], arr[0]];
                    }
                }
            }
            "color" => {
                if let Some(arr) = parse_f32_array(value) {
                    if arr.len() >= 3 {
                        color = [arr[0], arr[1], arr[2]];
                    }
                }
            }
            "rigidbody" => rigidbody = Some(parse_f32(value)),
            "light" => {
                let tokens: Vec<&str> = value.split_whitespace().collect();
                if tokens.len() >= 6 {
                    light = Some((
                        tokens[0].to_string(),
                        [
                            tokens[1].parse().unwrap_or(1.0),
                            tokens[2].parse().unwrap_or(1.0),
                            tokens[3].parse().unwrap_or(1.0),
                        ],
                        tokens[4].parse().unwrap_or(1.0),
                        tokens[5].parse().unwrap_or(10.0),
                    ));
                }
            }
            _ => {}
        }
    }

    if name.is_empty() {
        return Err(format!(
            "Prefab {:?} missing 'name' field",
            file_path
        ));
    }

    Ok(Prefab {
        name,
        mesh,
        material,
        position,
        rotation,
        scale,
        color,
        metallic,
        roughness,
        ao,
        rigidbody,
        light,
        script,
        file_path: file_path.to_path_buf(),
    })
}

fn parse_f32(s: &str) -> f32 {
    s.trim().trim_matches('"').parse::<f32>().unwrap_or(0.0)
}

fn parse_f32_array(s: &str) -> Option<Vec<f32>> {
    let s = s.trim();
    // Try bracketed format: [1.0, 2.0, 3.0]
    if let Some(inner) = s.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        return Some(
            inner
                .split(',')
                .map(|tok| tok.trim().parse::<f32>().unwrap_or(0.0))
                .collect(),
        );
    }
    // Fall back to space-separated: 1.0 2.0 3.0
    let tokens: Vec<f32> = s
        .split_whitespace()
        .filter_map(|tok| tok.parse::<f32>().ok())
        .collect();
    if tokens.is_empty() { None } else { Some(tokens) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_prefab() {
        let toml = r#"
name = "test_crate"
mesh = meshes/cube.obj
material = dark_wood
position = 1.0 2.0 3.0
rotation = 0.0 45.0 0.0
scale = 2.0
rigidbody = 3.0
"#;
        let prefab = parse_prefab(toml, Path::new("test.prefab")).unwrap();
        assert_eq!(prefab.name, "test_crate");
        assert_eq!(prefab.mesh, "meshes/cube.obj");
        assert_eq!(prefab.material.as_deref(), Some("dark_wood"));
        assert_eq!(prefab.position, [1.0, 2.0, 3.0]);
        assert_eq!(prefab.rotation, [0.0, 45.0, 0.0]);
        assert_eq!(prefab.scale, [2.0, 2.0, 2.0]);
        assert_eq!(prefab.rigidbody, Some(3.0));
    }

    #[test]
    fn parse_light_prefab() {
        let toml = r#"
name = "lantern"
mesh = meshes/cube.obj
scale = 0.2 0.2 0.2
light = point 1.0 0.9 0.8 3.0 12.0
"#;
        let prefab = parse_prefab(toml, Path::new("lantern.prefab")).unwrap();
        let light = prefab.light.unwrap();
        assert_eq!(light.0, "point");
        assert!((light.3 - 12.0).abs() < 0.001);
    }
}
