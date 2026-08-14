//! Hot-reloadable Rust game logic.
//!
//! The engine (`Triengine`) loads this crate as a `cdylib` and calls
//! `game_tick` every frame. When you save any `.rs` file in this crate (or the
//! shared `game_api` contract), the engine rebuilds it with `cargo` and swaps
//! the DLL live — the engine never restarts.
//!
//! Experimental: shared state does NOT survive a reload; use `ctx.reset` to
//! reinitialise persistent plugin data. Keep builds small; only code in this
//! crate is hot-swapped, never the engine itself.

use game_api::FrameCtx;

#[unsafe(no_mangle)]
pub extern "C" fn plugin_version() -> u64 {
    game_api::PLUGIN_VERSION
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn game_tick(ctx: *mut FrameCtx) {
    // Prefer the safe helper; bail on a NULL/invalid pointer.
    let Some(ctx) = (unsafe { game_api::ctx_mut(ctx) }) else {
        return;
    };

    if ctx.reset == 1 {
        ctx.debug_value = 0.0;
        ctx.debug_text = [0u8; 64];
        ctx.log_info("rust plugin (re)loaded");
        ctx.reset = 0;
    }

    // ── Demo game logic ────────────────────────────────────────────────
    // Edit this file and save: you'll see the new behaviour apply within a
    // second or two, without restarting the engine.
    ctx.debug_value += ctx.dt * (if ctx.move_y > 0.0 { 3.0 } else { 1.0 });

    let text = format!("frames={} debug={:.2}", ctx.frame_index, ctx.debug_value);
    fill_text(&mut ctx.debug_text, &text);

    if ctx.key_space == 1 {
        ctx.debug_value = 0.0;
    }
    if ctx.frame_index % 300 == 0 {
        ctx.log_info("rust plugin tick: edit game_plugin/src/lib.rs to change me");
    }
}

fn fill_text(buf: &mut [u8; 64], s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(buf.len().saturating_sub(1));
    buf[..n].copy_from_slice(&bytes[..n]);
    buf[n] = 0;
}