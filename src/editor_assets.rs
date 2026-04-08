use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub struct AssetEntry {
    pub path: String,
    pub kind: String,
    pub modified_unix_secs: u64,
}

#[derive(Default)]
pub struct AssetMetadataDb {
    pub entries: Vec<AssetEntry>,
}

impl AssetMetadataDb {
    pub fn scan_content_root(&mut self, root: &str) {
        self.entries.clear();
        let root_path = Path::new(root);
        self.visit(root_path);
    }

    fn visit(&mut self, dir: &Path) {
        let Ok(read) = fs::read_dir(dir) else { return; };
        for entry in read.flatten() {
            let path = entry.path();
            if path.is_dir() {
                self.visit(&path);
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            let kind = match ext.as_str() {
                "png" | "jpg" | "jpeg" => "texture",
                "obj" | "gltf" | "glb" => "mesh",
                "prefab" => "prefab",
                "lua" => "script",
                "mat" | "material" => "material",
                "fol" | "foliage" => "foliage",
                _ => "other",
            };
            let modified = fs::metadata(&path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            self.entries.push(AssetEntry {
                path: path.to_string_lossy().to_string(),
                kind: kind.to_string(),
                modified_unix_secs: modified,
            });
        }
    }
}

#[derive(Default)]
pub struct IconRegistry {
    pub icons: HashMap<String, PathBuf>,
}

impl IconRegistry {
    pub fn load_from_dir(&mut self, dir: &str) {
        self.icons.clear();
        let Ok(entries) = fs::read_dir(dir) else { return; };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if ext != "png" && ext != "svg" {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                self.icons.insert(stem.to_string(), path);
            }
        }
    }
}
