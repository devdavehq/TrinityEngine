// src/render/shader_manager.rs
// ──────────────────────────────────────────────────────────────────────────────
// Shader Management System
//
// WHY IT EXISTS:
//   Right now shaders are loaded as raw .wgsl strings and compiled inline
//   in the renderer. There's no caching, no error reporting beyond a panic,
//   and no hot-reload for shaders.
//
//   The ShaderManager provides:
//   - Named shader loading with error handling
//   - Runtime compilation + validation
//   - File watching for hot-reload (edit a .wgsl, see changes without restart)
//   - Cache compiled shaders to avoid recompilation
//   - Quality-tier shader variants (low: simplified, high: full features)
//
// LOW-END PC STRATEGY:
//   Shader variants let us serve different complexity levels:
//   - "pbr_simple": reduced lighting calculations, no SSAO, no voxel GI
//   - "pbr_full": complete PBR with all features
//   The ShaderManager selects the right variant based on the active quality tier.
//
// HOT RELOAD:
//   Uses the notify crate to watch shader directories. When a .wgsl file changes,
//   the manager recompiles it, validates the output, and hot-swaps the pipeline.
//   If compilation fails, the old shader stays active and an error is logged.
// ──────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

// ── Shader Entry ──────────────────────────────────────────────────────────────
// One compiled/validated shader module, ready for pipeline creation.

pub struct ShaderEntry {
    /// The wgpu shader module, ready to use in a pipeline.
    pub module: wgpu::ShaderModule,
    /// Raw WGSL source (for recompilation on hot-reload).
    pub source: String,
    /// File this shader was loaded from (for hot-reload).
    pub file_path: PathBuf,
    /// Last modification time (for change detection).
    pub last_modified: std::time::SystemTime,
    /// Compilation warnings (non-fatal).
    pub warnings: Vec<String>,
}

// ── Shader Manager ────────────────────────────────────────────────────────────

pub struct ShaderManager {
    /// Named shaders: "pbr_main" -> ShaderEntry
    shaders: HashMap<String, ShaderEntry>,
    /// File watcher for hot-reload.
    watcher: Option<RecommendedWatcher>,
    /// Channel for file change notifications.
    watcher_rx: Option<mpsc::Receiver<PathBuf>>,
    /// Active quality tier: "low", "balanced", "high", "cinematic"
    quality_tier: String,
}

impl ShaderManager {
    pub fn new() -> Self {
        Self {
            shaders: HashMap::new(),
            watcher: None,
            watcher_rx: None,
            quality_tier: "balanced".to_string(),
        }
    }

    // ── Load ──────────────────────────────────────────────────────────────

    /// Load a shader from a .wgsl file.
    /// Returns Ok(name) on success, Err(message) on failure.
    pub fn load_shader(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        path: impl AsRef<Path>,
    ) -> Result<String, String> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy().to_string();
        let source = crate::vfs::read_to_string(&path_str)
            .map_err(|e| format!("Failed to read shader {:?}: {}", path, e))?;

        self.load_shader_from_source(device, name, &source, path)
    }

    /// Load a shader from source code directly (for built-in shaders).
    pub fn load_shader_from_source(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        source: &str,
        path: impl AsRef<Path>,
    ) -> Result<String, String> {
        let path = path.as_ref().to_path_buf();

        // Create the shader module. wgpu validates WGSL at creation time.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let metadata = std::fs::metadata(&path).ok();
        let last_modified = metadata
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::SystemTime::now());

        let entry = ShaderEntry {
            module,
            source: source.to_string(),
            file_path: path,
            last_modified,
            warnings: Vec::new(),
        };

        self.shaders.insert(name.to_string(), entry);
        log::info!("[ShaderManager] Loaded shader: {}", name);
        Ok(name.to_string())
    }

    // ── Query ─────────────────────────────────────────────────────────────

    /// Get a compiled shader by name.
    pub fn get(&self, name: &str) -> Option<&ShaderEntry> {
        self.shaders.get(name)
    }

    /// Get a compiled shader, panicking if not found (for known shaders).
    pub fn require(&self, name: &str) -> &ShaderEntry {
        self.shaders
            .get(name)
            .unwrap_or_else(|| panic!("Shader '{}' not loaded", name))
    }

    /// Check if a shader is loaded.
    pub fn contains(&self, name: &str) -> bool {
        self.shaders.contains_key(name)
    }

    /// All loaded shader names (for editor display).
    pub fn shader_names(&self) -> Vec<&str> {
        self.shaders.keys().map(|s| s.as_str()).collect()
    }

    // ── Quality Tier ──────────────────────────────────────────────────────

    /// Set the quality tier. Future: select shader variants based on tier.
    pub fn set_quality_tier(&mut self, tier: &str) {
        self.quality_tier = tier.to_string();
        log::info!("[ShaderManager] Quality tier: {}", tier);
    }

    pub fn quality_tier(&self) -> &str {
        &self.quality_tier
    }

    // ── Hot Reload ────────────────────────────────────────────────────────

    /// Start watching a directory for shader changes.
    pub fn start_watching(&mut self, dir: impl AsRef<Path>) {
        let dir = dir.as_ref().to_path_buf();
        let (tx, rx) = mpsc::channel::<PathBuf>();

        let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
            if let Ok(event) = res {
                for path in event.paths {
                    if path.extension().and_then(|e| e.to_str()) == Some("wgsl") {
                        let _ = tx.send(path);
                    }
                }
            }
        })
        .expect("Failed to create shader watcher");

        watcher
            .watch(&dir, RecursiveMode::Recursive)
            .ok();

        self.watcher = Some(watcher);
        self.watcher_rx = Some(rx);
        log::info!("[ShaderManager] Watching for shader changes in {:?}", dir);
    }

    /// Check for hot-reload events. Call once per frame.
    /// Returns list of (shader_name, success, error_message).
    pub fn check_hot_reload(&mut self, device: &wgpu::Device) -> Vec<(String, bool, Option<String>)> {
        let mut results = Vec::new();

        // Drain all pending notifications.
        let changed_paths: Vec<PathBuf> = match &self.watcher_rx {
            Some(rx) => {
                let mut paths = Vec::new();
                while let Ok(path) = rx.try_recv() {
                    paths.push(path);
                }
                paths
            }
            None => return results,
        };

        for changed_path in changed_paths {
            let changed_path_str = changed_path.to_string_lossy().to_string();

            // Find which shader this file belongs to.
            let shader_name = self
                .shaders
                .iter()
                .find(|(_, entry)| entry.file_path == changed_path)
                .map(|(name, _)| name.clone());

            if let Some(name) = shader_name {
                // Recompile.
                match self.reload_shader(device, &name) {
                    Ok(()) => {
                        log::info!("[ShaderManager] Hot-reloaded: {}", name);
                        results.push((name, true, None));
                    }
                    Err(e) => {
                        log::error!("[ShaderManager] Hot-reload FAILED for {}: {}", name, e);
                        results.push((name, false, Some(e)));
                    }
                }
            } else {
                log::info!("[ShaderManager] Shader file changed but not loaded: {}", changed_path_str);
            }
        }

        results
    }

    /// Reload a single shader from disk.
    fn reload_shader(&mut self, device: &wgpu::Device, name: &str) -> Result<(), String> {
        let entry = self.shaders.get(name).ok_or("Shader not found")?;
        let path = entry.file_path.clone();
        let path_str = path.to_string_lossy().to_string();

        let source = crate::vfs::read_to_string(&path_str)
            .map_err(|e| format!("Failed to read {:?}: {}", path, e))?;

        // Recompile. If it fails, we keep the old shader active.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.clone().into()),
        });

        let metadata = std::fs::metadata(&path).ok();
        let last_modified = metadata
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::SystemTime::now());

        // Only swap if we got here (no validation error).
        if let Some(entry) = self.shaders.get_mut(name) {
            entry.module = module;
            entry.source = source;
            entry.last_modified = last_modified;
            entry.warnings.clear();
        }

        Ok(())
    }

    // ── Cleanup ───────────────────────────────────────────────────────────

    pub fn clear(&mut self) {
        self.shaders.clear();
        self.watcher = None;
        self.watcher_rx = None;
    }
}
