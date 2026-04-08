//! Cross-session editor preferences (window geometry, shared data paths).

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Per-machine app data: `%LOCALAPPDATA%\TrinityEngine` on Windows, else `~/.local/share/TrinityEngine`.
pub fn trinity_data_dir() -> PathBuf {
    #[cfg(windows)]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local).join("TrinityEngine");
        }
    }
    #[cfg(not(windows))]
    {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".local/share/TrinityEngine");
        }
    }
    PathBuf::from(".trinity")
}

pub fn editor_dock_layout_path() -> PathBuf {
    trinity_data_dir().join("editor_dock_layout.json")
}

fn window_prefs_path() -> PathBuf {
    trinity_data_dir().join("editor_window.toml")
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EditorWindowPrefs {
    pub width: u32,
    pub height: u32,
    pub pos_x: Option<i32>,
    pub pos_y: Option<i32>,
}

pub fn load_window_prefs() -> Option<EditorWindowPrefs> {
    let s = fs::read_to_string(window_prefs_path()).ok()?;
    toml::from_str(&s).ok()
}

pub fn save_window_prefs(prefs: &EditorWindowPrefs) {
    let dir = trinity_data_dir();
    let _ = fs::create_dir_all(&dir);
    if let Ok(s) = toml::to_string(prefs) {
        let _ = fs::write(window_prefs_path(), s);
    }
}
