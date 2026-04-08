//! Hub project list persisted next to other Trinity app data.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::editor_persist;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectEntry {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub last_opened_unix: u64,
    /// Engine build last used when this project was opened (informational).
    #[serde(default)]
    pub engine_version_at_last_open: String,
}

#[derive(Default, Serialize, Deserialize)]
pub struct ProjectRegistry {
    pub projects: Vec<ProjectEntry>,
}

fn registry_path() -> PathBuf {
    editor_persist::trinity_data_dir().join("hub_projects.toml")
}

impl ProjectRegistry {
    pub fn load() -> Self {
        let path = registry_path();
        let Ok(text) = fs::read_to_string(&path) else {
            return Self::default();
        };
        toml::from_str(&text).unwrap_or_default()
    }

    pub fn save(&self) {
        let dir = editor_persist::trinity_data_dir();
        let _ = fs::create_dir_all(&dir);
        if let Ok(s) = toml::to_string(self) {
            let _ = fs::write(registry_path(), s);
        }
    }

    pub fn upsert_opened(
        &mut self,
        project_dir: &Path,
        display_name: Option<&str>,
        engine_version: &str,
    ) {
        let path_s = project_dir.to_string_lossy().to_string();
        let name = display_name
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                project_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("Project")
                    .to_string()
            });
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Some(e) = self.projects.iter_mut().find(|e| e.path == path_s) {
            e.last_opened_unix = now;
            e.name = name;
            e.engine_version_at_last_open = engine_version.to_string();
        } else {
            self.projects.push(ProjectEntry {
                name,
                path: path_s,
                last_opened_unix: now,
                engine_version_at_last_open: engine_version.to_string(),
            });
        }
        self.projects
            .sort_by(|a, b| b.last_opened_unix.cmp(&a.last_opened_unix));
    }

    pub fn remove_index(&mut self, i: usize) {
        if i < self.projects.len() {
            self.projects.remove(i);
        }
    }
}
