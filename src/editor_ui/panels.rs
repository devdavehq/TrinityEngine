// src/editor_ui/panels.rs
// ──────────────────────────────────────────────────────────────────────────────
// Editor panel sub-modules.
//
// Modules:
//   content_browser — asset browsing, folder/file creation, material/foliage editors
//   details         — docked property inspector for selected entities
//   outliner        — world hierarchy tree with drag-drop reparent
//   viewport        — 3D viewport with gizmos, terrain painting, entity picking
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

pub use content_browser::render_content_browser_panel;
pub use details::render_details_panel;
pub use outliner::render_outliner_panel;
pub use viewport::render_viewport_panel;
pub use levels::render_levels_panel;
pub use script_editor::render_script_editor_panel;
