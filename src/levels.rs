// src/levels.rs
// ──────────────────────────────────────────────────────────────────────────────
// Level system for TrinEngine.
//
// A "level" (or sub-level) is a collection of entities loaded from a .scene
// file that coexist in the game world alongside other levels. Multiple levels
// can be loaded/unloaded independently, enabling streaming worlds, additive
// scene composition, and portal-based transitions.
//
// Sub-modules:
//   level     — core Level and LevelManager types
//   streaming — distance-based automatic level loading/unloading
//   portal    — trigger-zone portals that load/unload levels
//   state     — persistent per-entity world state across level loads
// ──────────────────────────────────────────────────────────────────────────────

pub mod level;
pub mod streaming;
pub mod portal;
pub mod state;
pub mod cross_ref;
pub mod loading_screen;

pub use level::*;
pub use streaming::*;
pub use state::*;
pub use loading_screen::*;
