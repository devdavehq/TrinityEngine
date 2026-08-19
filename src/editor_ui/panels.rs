// src/editor_ui/panels.rs
// ──────────────────────────────────────────────────────────────────────────────
// Editor panel sub-modules.
//
// Modules:
//   content_browser — asset browsing, folder/file creation, material/foliage editors
//   details         — docked property inspector for selected entities
//   outliner        — world hierarchy tree with drag-drop reparent
//   viewport        — 3D viewport with gizmos, terrain painting, entity picking
//   navmesh         — rebuild trigger + stats for the baked AI navmesh
//   decals          — place/tune decal projector boxes
//   save_slots      — save-game slot browser (save/load/delete)
// ──────────────────────────────────────────────────────────────────────────────

#[path = "panels/content_browser.rs"]
pub mod content_browser;
#[path = "panels/details.rs"]
pub mod details;
#[path = "panels/outliner.rs"]
pub mod outliner;
#[path = "panels/viewport.rs"]
pub mod viewport;
#[path = "panels/bt_editor.rs"]
pub mod bt_editor;
#[path = "panels/anim_graph_editor.rs"]
pub mod anim_graph_editor;
#[path = "panels/script_editor.rs"]
pub mod script_editor;
#[path = "panels/levels.rs"]
pub mod levels;
#[path = "panels/navmesh.rs"]
pub mod navmesh;
#[path = "panels/decals.rs"]
pub mod decals;
#[path = "panels/save_slots.rs"]
pub mod save_slots;

pub use content_browser::render_content_browser_panel;
pub use details::render_details_panel;
pub use outliner::render_outliner_panel;
pub use viewport::render_viewport_panel;
pub use levels::render_levels_panel;
pub use script_editor::render_script_editor_panel;
pub use navmesh::render_navmesh_panel;
pub use decals::render_decals_panel;
pub use save_slots::render_save_slots_panel;
