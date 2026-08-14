//! Rust hot reload: swap `game_plugin.dll` at runtime so engine code can be
//! iterated on without restarting the engine process.
//!
//! How it works
//! ------------
//! - `game_plugin/` is a separate crate compiled as a `cdylib` exposing
//!   `plugin_version()` and `game_tick(&mut FrameCtx)`.
//! - A `notify` watcher watches `game_plugin/` + `game_api/`. On any `.rs` or
//!   `Cargo.toml` change a worker thread runs `cargo build --release -p
//!   game_plugin`. Compile errors keep the previous DLL loaded and are logged.
//! - On Windows a loaded DLL is locked, so the freshly built DLL is first
//!   copied to a unique name and *that* copy is loaded; old copies are pruned.
//! - `FrameCtx` crosses as `#[repr(C)]` plain data (see the `game_api` crate).
//!
//! What hot reloads: only code inside `game_plugin/`. The renderer, physics,
//! scene, and scripting systems are engine code — changing them still needs a
//! normal rebuild.

#![cfg(feature = "hotreload")]

use std::ffi::{c_char, CStr};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use game_api::{FrameCtx, GameTickFn, PLUGIN_VERSION};
use libloading::Library;
use notify::Watcher;

const REBUILD_PROFILE: &str = "release";

struct PluginState {
    lib: Option<Library>,
    tick: Option<GameTickFn>,
    version: u64,
    generation: u32,
    last_error: Option<String>,
}

impl PluginState {
    fn idle() -> Self {
        Self {
            lib: None,
            tick: None,
            version: 0,
            generation: 0,
            last_error: None,
        }
    }
}

/// Log bridge from plugin -> engine (`tracing` + the editor console).
pub extern "C" fn plugin_log(level: u8, msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    let s = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    match level {
        0 => log::info!("[GamePlugin] {}", s),
        1 => log::warn!("[GamePlugin] {}", s),
        _ => log::error!("[GamePlugin] {}", s),
    }
}

#[cfg(target_os = "windows")]
fn lib_ext() -> &'static str {
    "dll"
}
#[cfg(target_os = "macos")]
fn lib_ext() -> &'static str {
    "dylib"
}
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn lib_ext() -> &'static str {
    "so"
}

/// Walk upward from the current working dir until we find the workspace root
/// (the folder that contains the `game_plugin` crate). Falls back to `cwd`.
pub fn find_project_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    loop {
        if dir.join("game_plugin").is_dir() && dir.join("Cargo.toml").is_file() {
            break;
        }
        if !dir.pop() {
            break;
        }
    }
    dir
}

pub struct RustHotReloader {
    state: Arc<Mutex<PluginState>>,
    reset_pending: Arc<AtomicBool>,
    notify_tx: Sender<()>,
    enabled: bool,
}

impl RustHotReloader {
    /// Create a reloader rooted at `project_root` (the workspace root).
    /// Nothing is built until a change is detected or `kick()` is called.
    pub fn new(project_root: PathBuf) -> Self {
        let state = Arc::new(Mutex::new(PluginState::idle()));
        let reset_pending = Arc::new(AtomicBool::new(false));
        let building = Arc::new(AtomicBool::new(false));

        // One channel: watcher + kick -> builder.
        let (notify_tx, builder_rx) = mpsc::channel::<()>();
        spawn_builder(
            builder_rx,
            Arc::clone(&state),
            Arc::clone(&reset_pending),
            Arc::clone(&building),
            project_root.clone(),
        );
        for dir in ["game_plugin", "game_api"] {
            let dir = project_root.join(dir);
            if !dir.is_dir() {
                continue;
            }
            let tx = notify_tx.clone();
            let mut watcher = match notify::recommended_watcher(
                move |res: notify::Result<notify::Event>| {
                    if let Ok(event) = res {
                        let interesting = event.paths.iter().any(|p| {
                            let ext = p.extension().and_then(|e| e.to_str());
                            ext == Some("rs") || ext == Some("toml")
                        });
                        if interesting {
                            let _ = tx.send(());
                        }
                    }
                },
            ) {
                Ok(w) => w,
                Err(e) => {
                    log::warn!("[HotReload] watcher for {} failed: {}", dir.display(), e);
                    return Self {
                            state,
                            reset_pending,
                            notify_tx,
                            enabled: true,
                        };
                }
            };
            if let Err(e) = watcher.watch(&dir, notify::RecursiveMode::Recursive) {
                log::warn!("[HotReload] cannot watch {}: {}", dir.display(), e);
            }
        }

        Self {
            state,
            reset_pending,
            notify_tx,
            enabled: true,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// True once a plugin has been compiled and mapped successfully.
    pub fn is_loaded(&self) -> bool {
        let st = self.state.lock().unwrap();
        st.tick.is_some()
    }

    pub fn generation(&self) -> u32 {
        self.state.lock().unwrap().generation
    }

    pub fn last_error(&self) -> Option<String> {
        self.state.lock().unwrap().last_error.clone()
    }

    /// Force an eager build (e.g. right after startup).
    pub fn kick(&self) {
        let _ = self.notify_tx.send(());
    }

    /// Per-frame hook: runs the current plugin with a fresh context.
    pub fn tick(&mut self, ctx: &mut FrameCtx) {
        if !self.enabled {
            return;
        }
        ctx.reset = if self.reset_pending.swap(false, Ordering::SeqCst) { 1 } else { 0 };
        ctx.log = plugin_log;

        let st = self.state.lock().unwrap();
        let tick = match st.tick {
            Some(f) => f,
            None => return,
        };
        let ptr = ctx as *mut FrameCtx;
        unsafe { tick(ptr) };
    }
}

// ── Build + swap worker ──────────────────────────────────────────────────────

fn spawn_builder(
    rx: Receiver<()>,
    state: Arc<Mutex<PluginState>>,
    reset_pending: Arc<AtomicBool>,
    building: Arc<AtomicBool>,
    project_root: PathBuf,
) {
    std::thread::spawn(move || loop {
        if rx.recv().is_err() {
            return;
        }
        // Debounce: slurp any further change notifications for ~150 ms.
        while rx.recv_timeout(Duration::from_millis(150)).is_ok() {}

        if building.swap(true, Ordering::SeqCst) {
            continue;
        }
        let result = build_and_swap(&project_root, &state, &reset_pending);
        building.store(false, Ordering::SeqCst);

        match result {
            Ok(reload_gen) => {
                log::info!("[HotReload] game_plugin swapped (generation {})", reload_gen)
            }
            Err(e) => {
                let mut st = state.lock().unwrap();
                st.last_error = Some(e.clone());
                log::error!(
                    "[HotReload] {} (keeping the previously loaded plugin)",
                    e
                );
            }
        }
    });
}

fn build_and_swap(
    root: &Path,
    state: &Arc<Mutex<PluginState>>,
    reset_pending: &Arc<AtomicBool>,
) -> Result<u32, String> {
    let output = Command::new("cargo")
        .args(["build", "--release", "-p", "game_plugin"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to launch cargo: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut tail: Vec<&str> = stderr.lines().rev().take(30).collect();
        tail.reverse();
        return Err(format!("compile error:\n{}", tail.join("\n")));
    }

    let built = root
        .join("target")
        .join(REBUILD_PROFILE)
        .join(format!("game_plugin.{}", lib_ext()));
    if !built.exists() {
        return Err(format!(
            "cargo succeeded but {:?} was not produced — cdylib build failed?",
            built
        ));
    }

    // Windows locks mapped DLLs; map a private copy so the next build can
    // overwrite game_plugin.dll on disk.
    let reload_gen = {
        let st = state.lock().unwrap();
        st.generation + 1
    };
    let copy_dir = root
        .join("target")
        .join(REBUILD_PROFILE)
        .join("hotreload");
    std::fs::create_dir_all(&copy_dir)
        .map_err(|e| format!("cannot create {}: {}", copy_dir.display(), e))?;
    let copy = copy_dir.join(format!("game_plugin_v{}.{}", reload_gen, lib_ext()));
    std::fs::copy(&built, &copy)
        .map_err(|e| format!("copy {} -> {} failed: {}", built.display(), copy.display(), e))?;

    // Load + verify version BEFORE dropping anything old.
    let lib = unsafe { Library::new(&copy) }
        .map_err(|e| format!("load {} failed: {}", copy.display(), e))?;
    let version_sym = unsafe { lib.get::<extern "C" fn() -> u64>(b"plugin_version") }
        .map_err(|e| format!("missing plugin_version symbol: {}", e))?;
    let version = version_sym();
    drop(version_sym);
    if version != PLUGIN_VERSION {
        return Err(format!(
            "plugin ABI version {} != engine {} — bump game_api::PLUGIN_VERSION and rebuild",
            version, PLUGIN_VERSION
        ));
    }
    let tick_sym = unsafe { lib.get::<GameTickFn>(b"game_tick") }
        .map_err(|e| format!("missing game_tick symbol: {}", e))?;
    let tick: GameTickFn = *tick_sym;
    drop(tick_sym);

    // Swap: store the new Library first, then drop the old one. Never unload
    // the code we are about to call.
    {
        let mut st = state.lock().unwrap();
        st.lib = Some(lib);
        st.tick = Some(tick);
        st.version = version;
        st.generation = reload_gen;
        st.last_error = None;
        reset_pending.store(true, Ordering::SeqCst);
    }

    // Prune old copies (keep a few generations past the current one).
    if let Ok(entries) = std::fs::read_dir(&copy_dir) {
        let mut stale: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some(lib_ext()))
            .filter(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.rsplit('_').next())
                    .and_then(|n| n.parse::<u32>().ok())
                    .map(|n| n + 4 < reload_gen)
                    .unwrap_or(false)
            })
            .collect();
        for p in stale.drain(..) {
            let _ = std::fs::remove_file(&p);
        }
    }

    Ok(reload_gen)
}