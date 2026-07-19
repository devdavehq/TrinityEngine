use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::components::Renderable;

// ── Material Data Structures ────────────────────────────────────────────────
//
// WHY:
//   Previously materials were hardcoded in Rust structs. This made it impossible
//   for designers to tweak colors/metallic/roughness without recompiling.
//
//   Now materials are TOML files that the engine hot-reloads. The format is:
//
//   [material]
//   name = "rusty_metal"
//   base_color = [0.65, 0.25, 0.15]
//   metallic = 0.8
//   roughness = 0.6
//   ao = 1.0
//
//   Material instances reference a master and apply multipliers:
//
//   [instance]
//   name = "rusted_plate"
//   master = "master_metal"
//   color_tint = [0.75, 0.30, 0.18]
//   metallic_mul = 0.95
//   roughness_mul = 1.3
//   ao_mul = 1.0

/// A base material definition. Multiple instances can reference this.
#[derive(Clone, Debug)]
pub struct MasterMaterial {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
}

/// A material instance that references a master and applies multipliers.
#[derive(Clone, Debug)]
pub struct MaterialInstance {
    pub master: String,
    pub color_tint: [f32; 3],
    pub metallic_mul: f32,
    pub roughness_mul: f32,
    pub ao_mul: f32,
}

/// Parsed representation of a .material TOML file.
/// Can be either a master material or a material instance.
#[derive(Clone, Debug)]
pub enum MaterialFile {
    Master(MasterMaterial),
    Instance(MaterialInstance),
}

pub struct MaterialLibrary {
    masters: HashMap<String, MasterMaterial>,
    instances: HashMap<String, MaterialInstance>,
    /// Track file paths so we can hot-reload.
    file_sources: HashMap<String, PathBuf>,
}

impl MaterialLibrary {
    pub fn new_defaults() -> Self {
        let mut lib = Self {
            masters: HashMap::new(),
            instances: HashMap::new(),
            file_sources: HashMap::new(),
        };
        lib.insert_hardcoded_defaults();
        lib
    }

    /// Register the built-in fallback materials.
    fn insert_hardcoded_defaults(&mut self) {
        self.masters.insert(
            "master_surface".into(),
            MasterMaterial {
                base_color: [1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.7,
                ao: 1.0,
            },
        );
        self.masters.insert(
            "master_metal".into(),
            MasterMaterial {
                base_color: [0.82, 0.82, 0.82],
                metallic: 1.0,
                roughness: 0.3,
                ao: 1.0,
            },
        );
        self.masters.insert(
            "master_foliage".into(),
            MasterMaterial {
                base_color: [0.18, 0.42, 0.20],
                metallic: 0.0,
                roughness: 0.9,
                ao: 1.0,
            },
        );

        self.instances.insert(
            "matte_black".into(),
            MaterialInstance {
                master: "master_surface".into(),
                color_tint: [0.12, 0.12, 0.12],
                metallic_mul: 0.0,
                roughness_mul: 1.1,
                ao_mul: 1.0,
            },
        );
        self.instances.insert(
            "silver_brushed".into(),
            MaterialInstance {
                master: "master_metal".into(),
                color_tint: [0.92, 0.92, 0.94],
                metallic_mul: 1.0,
                roughness_mul: 1.2,
                ao_mul: 1.0,
            },
        );
        self.instances.insert(
            "foliage_leaf".into(),
            MaterialInstance {
                master: "master_foliage".into(),
                color_tint: [0.20, 0.55, 0.22],
                metallic_mul: 0.0,
                roughness_mul: 1.0,
                ao_mul: 1.0,
            },
        );
    }

    // ── Data-Driven Loading ─────────────────────────────────────────────────

    /// Scan a directory for .material files and load them.
    /// This is the main data-driven entry point.
    pub fn load_from_directory(&mut self, dir: impl AsRef<Path>) {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            tracing::info!(
                "[Materials] Directory {:?} not found, using hardcoded defaults.",
                dir
            );
            return;
        }
        self.visit_directory(dir);
        tracing::info!(
            "[Materials] Loaded {} masters, {} instances from disk.",
            self.masters.len(),
            self.instances.len()
        );
    }

    fn visit_directory(&mut self, dir: &Path) {
        let Ok(read) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in read.flatten() {
            let path = entry.path();
            if path.is_dir() {
                self.visit_directory(&path);
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if ext == "material" || ext == "mat" {
                self.load_material_file(&path);
            }
        }
    }

    /// Parse a single .material TOML file and register it.
    pub fn load_material_file(&mut self, path: &Path) {
        let path_str = path.to_string_lossy().to_string();
        let contents = match crate::vfs::read_to_string(&path_str) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(
                    "[Materials] Failed to read {:?}: {}",
                    path, e
                );
                return;
            }
        };
        match parse_material_toml(&contents) {
            Ok((name, mat)) => {
                match mat {
                    MaterialFile::Master(m) => {
                        tracing::info!("[Materials] Loaded master '{}' from {:?}", name, path);
                        self.file_sources
                            .insert(format!("master:{}", name), path.to_path_buf());
                        self.masters.insert(name, m);
                    }
                    MaterialFile::Instance(inst) => {
                        tracing::info!("[Materials] Loaded instance '{}' from {:?}", name, path);
                        self.file_sources
                            .insert(format!("instance:{}", name), path.to_path_buf());
                        self.instances.insert(name, inst);
                    }
                }
            }
            Err(e) => {
                tracing::error!("[Materials] Parse error in {:?}: {}", path, e);
            }
        }
    }

    /// Reload a material file that changed on disk.
    pub fn reload_material_file(&mut self, path: &Path) -> Result<(), String> {
        let path_str = path.to_string_lossy().to_string();
        let contents =
            crate::vfs::read_to_string(&path_str).map_err(|e| format!("Read failed: {}", e))?;
        let (name, mat) = parse_material_toml(&contents)?;
        match mat {
            MaterialFile::Master(m) => {
                self.masters.insert(name, m);
            }
            MaterialFile::Instance(inst) => {
                self.instances.insert(name, inst);
            }
        }
        Ok(())
    }

    // ── Query ───────────────────────────────────────────────────────────────

    pub fn instance_names(&self) -> Vec<String> {
        let mut v: Vec<_> = self.instances.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn master_names(&self) -> Vec<String> {
        let mut v: Vec<_> = self.masters.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn has_instance(&self, name: &str) -> bool {
        self.instances.contains_key(name)
    }

    pub fn has_master(&self, name: &str) -> bool {
        self.masters.contains_key(name)
    }

    pub fn get_master(&self, name: &str) -> Option<&MasterMaterial> {
        self.masters.get(name)
    }

    pub fn get_instance(&self, name: &str) -> Option<&MaterialInstance> {
        self.instances.get(name)
    }

    pub fn print_help() {
        tracing::info!("[Materials] Master + Instance workflow (data-driven):");
        tracing::info!("  Place .material files in Content/Materials/ directory.");
        tracing::info!("  Format: TOML with [material] section.");
        tracing::info!("  Keys: name, base_color, metallic, roughness, ao");
        tracing::info!("  1 = apply 'matte_black' to selected entity");
        tracing::info!("  2 = apply 'silver_brushed' to selected entity");
        tracing::info!("  3 = apply 'foliage_leaf' to selected entity");
        tracing::info!("  N / M = select previous/next renderable entity");
    }

    pub fn apply_instance(
        &self,
        name: &str,
        renderable: &mut Renderable,
    ) -> Result<(), String> {
        let inst = self
            .instances
            .get(name)
            .ok_or_else(|| format!("Material instance '{}' not found", name))?;
        let master = self
            .masters
            .get(&inst.master)
            .ok_or_else(|| format!("Master material '{}' not found", inst.master))?;

        renderable.color = [
            (master.base_color[0] * inst.color_tint[0]).clamp(0.0, 1.0),
            (master.base_color[1] * inst.color_tint[1]).clamp(0.0, 1.0),
            (master.base_color[2] * inst.color_tint[2]).clamp(0.0, 1.0),
        ];
        renderable.metallic = (master.metallic * inst.metallic_mul).clamp(0.0, 1.0);
        renderable.roughness = (master.roughness * inst.roughness_mul).clamp(0.02, 1.0);
        renderable.ao = (master.ao * inst.ao_mul).clamp(0.0, 1.0);
        Ok(())
    }

    /// Apply a master material directly (no instance multiplier).
    pub fn apply_master(
        &self,
        name: &str,
        renderable: &mut Renderable,
    ) -> Result<(), String> {
        let master = self
            .masters
            .get(name)
            .ok_or_else(|| format!("Master material '{}' not found", name))?;

        renderable.color = master.base_color;
        renderable.metallic = master.metallic;
        renderable.roughness = master.roughness;
        renderable.ao = master.ao;
        Ok(())
    }
}

// ── TOML Parser ─────────────────────────────────────────────────────────────
// Minimal TOML parser for .material files.
// Supports [material] sections with key = value and key = [array] syntax.
// No external TOML crate needed — this keeps the engine dependency-free for
// simple asset formats.

fn parse_material_toml(contents: &str) -> Result<(String, MaterialFile), String> {
    let mut name = String::new();
    let mut is_instance = false;

    // Master fields
    let mut base_color = [1.0f32; 3];
    let mut metallic = 0.0f32;
    let mut roughness = 0.5f32;
    let mut ao = 1.0f32;

    // Instance fields
    let mut master_ref = String::new();
    let mut color_tint = [1.0f32; 3];
    let mut metallic_mul = 1.0f32;
    let mut roughness_mul = 1.0f32;
    let mut ao_mul = 1.0f32;

    let mut in_material_section = false;

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        // Section headers
        if line == "[material]" || line == "[master]" {
            in_material_section = true;
            is_instance = false;
            continue;
        }
        if line == "[instance]" {
            in_material_section = true;
            is_instance = true;
            continue;
        }

        // Skip lines outside any section
        if !in_material_section {
            continue;
        }

        // Parse key = value
        let mut parts = line.splitn(2, '=');
        let key = parts.next().unwrap_or("").trim();
        let value = parts.next().unwrap_or("").trim();

        match key {
            "name" => name = value.trim_matches('"').to_string(),
            "master" => master_ref = value.trim_matches('"').to_string(),

            // Float values
            "metallic" => metallic = parse_f32(value),
            "roughness" => roughness = parse_f32(value),
            "ao" => ao = parse_f32(value),
            "metallic_mul" => metallic_mul = parse_f32(value),
            "roughness_mul" => roughness_mul = parse_f32(value),
            "ao_mul" => ao_mul = parse_f32(value),

            // Array values: base_color = [0.5, 0.3, 0.1]
            "base_color" => {
                if let Some(arr) = parse_f32_array(value) {
                    if arr.len() >= 3 {
                        base_color = [arr[0], arr[1], arr[2]];
                    }
                }
            }
            "color_tint" => {
                if let Some(arr) = parse_f32_array(value) {
                    if arr.len() >= 3 {
                        color_tint = [arr[0], arr[1], arr[2]];
                    }
                }
            }

            _ => {} // Unknown keys ignored for forward compatibility
        }
    }

    if name.is_empty() {
        return Err("Missing 'name' field in material file".into());
    }

    let mat = if is_instance {
        if master_ref.is_empty() {
            return Err(format!(
                "Instance '{}' missing 'master' field",
                name
            ));
        }
        MaterialFile::Instance(MaterialInstance {
            master: master_ref,
            color_tint,
            metallic_mul,
            roughness_mul,
            ao_mul,
        })
    } else {
        MaterialFile::Master(MasterMaterial {
            base_color,
            metallic,
            roughness,
            ao,
        })
    };

    Ok((name, mat))
}

fn parse_f32(s: &str) -> f32 {
    let s = s.trim().trim_matches('"');
    s.parse::<f32>().unwrap_or(0.0)
}

fn parse_f32_array(s: &str) -> Option<Vec<f32>> {
    let s = s.trim();
    // Handle [1.0, 2.0, 3.0] format
    let inner = s
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))?;
    Some(
        inner
            .split(',')
            .map(|tok| tok.trim().parse::<f32>().unwrap_or(0.0))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_master_material() {
        let toml = r#"
[material]
name = "test_master"
base_color = [0.5, 0.3, 0.1]
metallic = 0.8
roughness = 0.4
ao = 0.9
"#;
        let (name, mat) = parse_material_toml(toml).unwrap();
        assert_eq!(name, "test_master");
        match mat {
            MaterialFile::Master(m) => {
                assert_eq!(m.base_color, [0.5, 0.3, 0.1]);
                assert!((m.metallic - 0.8).abs() < 0.001);
                assert!((m.roughness - 0.4).abs() < 0.001);
                assert!((m.ao - 0.9).abs() < 0.001);
            }
            _ => panic!("Expected master material"),
        }
    }

    #[test]
    fn parse_material_instance() {
        let toml = r#"
[instance]
name = "test_instance"
master = "test_master"
color_tint = [0.7, 0.8, 0.9]
metallic_mul = 1.1
roughness_mul = 0.9
ao_mul = 1.0
"#;
        let (name, mat) = parse_material_toml(toml).unwrap();
        assert_eq!(name, "test_instance");
        match mat {
            MaterialFile::Instance(inst) => {
                assert_eq!(inst.master, "test_master");
                assert_eq!(inst.color_tint, [0.7, 0.8, 0.9]);
                assert!((inst.metallic_mul - 1.1).abs() < 0.001);
                assert!((inst.roughness_mul - 0.9).abs() < 0.001);
            }
            _ => panic!("Expected material instance"),
        }
    }

    #[test]
    fn hardcoded_defaults_still_work() {
        let lib = MaterialLibrary::new_defaults();
        assert!(lib.has_master("master_surface"));
        assert!(lib.has_master("master_metal"));
        assert!(lib.has_master("master_foliage"));
        assert!(lib.has_instance("matte_black"));
        assert!(lib.has_instance("silver_brushed"));
        assert!(lib.has_instance("foliage_leaf"));
    }

    #[test]
    fn apply_instance_works() {
        use crate::assets::AssetStore;
        let lib = MaterialLibrary::new_defaults();
        let mut store = AssetStore::<crate::assets::Mesh>::new();
        let dummy_mesh = crate::assets::Mesh {
            vertices: vec![],
        };
        let handle = store.add(dummy_mesh);
        let mut rend = Renderable {
            mesh: handle,
            color: [0.0; 3],
            metallic: 0.0,
            roughness: 0.0,
            ao: 0.0,
            scale: [1.0; 3],
        };
        lib.apply_instance("matte_black", &mut rend).unwrap();
        // master_surface.base_color * matte_black.color_tint = [1,1,1] * [0.12,0.12,0.12]
        assert!((rend.color[0] - 0.12).abs() < 0.001);
        assert!((rend.color[1] - 0.12).abs() < 0.001);
        assert!((rend.color[2] - 0.12).abs() < 0.001);
    }
}
