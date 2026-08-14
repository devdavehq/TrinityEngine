//! Shared, ABI-stable bridge between the engine (host) and the hot-reloadable
//! Rust game plugin (`game_plugin`).
//!
//! Every type in this crate is plain `#[repr(C)]` data with a fixed layout.
//! The engine and the plugin each compile their own copy of this crate; they
//! interoperate because the layout is C-stable and both sides are built by the
//! same toolchain inside the same workspace. That is what lets the engine
//! unload and reload `game_plugin.dll` without restarting.
//!
//! # Rules
//! - Never add `String`/`Vec`/`&str` here; use fixed arrays + length fields.
//! - Bump `PLUGIN_VERSION` when the layout changes so an old DLL refuses to
//!   load instead of misinterpreting the context.

use core::ffi::c_char;

/// Bump this whenever `FrameCtx` (or any other type here) changes layout.
pub const PLUGIN_VERSION: u64 = 1;

/// Log callback the engine installs into `FrameCtx`. `level`: 0=info, 1=warn,
/// 2+=error. The message is a NUL-terminated UTF-8 string.
pub type LogFn = extern "C" fn(level: u8, msg: *const c_char);

/// The per-frame entry point the engine calls into the plugin.
/// Invalid (NULL) contexts are ignored; a new pointer is handed out each frame.
pub type GameTickFn = unsafe extern "C" fn(ctx: *mut FrameCtx);

/// Snapshot of engine state handed to the plugin every frame.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FrameCtx {
    /// Seconds since engine start.
    pub time: f32,
    /// Frame delta seconds (clamped by the engine).
    pub dt: f32,
    /// Monotonic frame counter.
    pub frame_index: u64,

    /// Viewport size in physical pixels.
    pub width: f32,
    pub height: f32,

    /// Editor/player camera in world space.
    pub cam_pos: [f32; 3],
    /// Forward direction (normalised) of the camera.
    pub cam_forward: [f32; 3],

    /// Gamepad left stick (deadzone applied), [-1, 1].
    pub move_x: f32,
    pub move_y: f32,
    /// Reserved axis (0 unless wired to a source).
    pub look_x: f32,
    pub look_y: f32,

    // Keyboard / mouse bit flags (0 or 1).
    pub key_w: u8,
    pub key_a: u8,
    pub key_s: u8,
    pub key_d: u8,
    pub key_space: u8,
    pub key_shift: u8,
    pub key_e: u8,
    pub key_r: u8,
    pub key_f: u8,
    pub key_q: u8,
    pub mouse_l: u8,
    pub mouse_r: u8,
    pub mouse_m: u8,

    /// Set to 1 for the first frame after a hot reload so the plugin can
    /// reinitialise its state. The plugin should clear it back to 0.
    pub reset: u8,
    /// Padding to keep the struct 16-byte aligned overall.
    pub _pad: [u8; 3],

    /// Callback for writing to the engine console. Always installed.
    pub log: LogFn,

    /// Scratch value the plugin may write; the editor can surface it.
    pub debug_value: f32,
    /// Scratch text the plugin may fill (NUL-terminated).
    pub debug_text: [u8; 64],
    pub _pad2: [u8; 4],
}

impl FrameCtx {
    /// Convenience: forward a message to the engine console with level 0.
    #[inline]
    pub fn log_info(&self, msg: &str) {
        self.log_with(0, msg);
    }

    #[inline]
    pub fn log_with(&self, level: u8, msg: &str) {
        use std::ffi::CString;
        let log = self.log;
        if let Ok(c) = CString::new(msg) {
            log(level, c.as_ptr());
        }
    }
}

/// Helper that a plugin can call from Rust-unsafe-free context to deref a ctx
/// pointer safely (used by generated stubs if any).
#[inline]
pub unsafe fn ctx_mut<'a>(ptr: *mut FrameCtx) -> Option<&'a mut FrameCtx> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { &mut *ptr })
    }
}