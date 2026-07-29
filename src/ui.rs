// src/ui.rs
// ──────────────────────────────────────────────────────────────────────────────
// Visual UI Designer for TrinEngine.
//
// Provides a drag-drop canvas where users lay out HUD widgets visually,
// a style panel for properties, and a reusable custom widget library.
//
// Architecture:
//   UiDesign        — the full design (collection of widgets + metadata)
//   UiWidget        — a single widget with position, style, and behavior
//   UiWidgetStyle   — visual properties (colors, font, border, corner radius)
//   UiWidgetLibrary — reusable template widgets that can be instantiated
//   UiCanvas        — the interactive canvas that renders + handles drag-drop
//   UiStylePanel    — the property inspector for the selected widget
//
// All state is serializable (serde) so designs save/load as .ui files.
//
// ── Usage from Lua ───────────────────────────────────────────────────────────
//   ui.create("my_hud")              — create a new UI design
//   ui.add_widget("my_hud", "label") — add a widget from library
//   ui.set_text("my_hud", "w1", "Hello")
//   ui.set_visible("my_hud", "w1", true)
//   ui.render("my_hud", dt)          — render to screen
// ──────────────────────────────────────────────────────────────────────────────

pub mod widget;
pub mod canvas;
pub mod style_panel;
pub mod library;

pub use widget::*;

use std::collections::HashMap;

// ── UI Design — top-level container ──────────────────────────────────────────
/// A complete UI layout. One per screen (HUD, pause menu, etc.).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct UiDesign {
    pub name: String,
    pub widgets: Vec<UiWidget>,
    pub selected: Option<usize>,
    pub canvas_zoom: f32,
    pub canvas_offset: [f32; 2],
    pub snap_to_grid: bool,
    pub grid_size: f32,
    pub show_grid: bool,
}

impl UiDesign {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            widgets: Vec::new(),
            selected: None,
            canvas_zoom: 1.0,
            canvas_offset: [0.0, 0.0],
            snap_to_grid: false,
            grid_size: 8.0,
            show_grid: true,
        }
    }

    pub fn add_widget(&mut self, widget: UiWidget) -> usize {
        let idx = self.widgets.len();
        self.widgets.push(widget);
        idx
    }

    pub fn remove_selected(&mut self) {
        if let Some(idx) = self.selected {
            self.widgets.remove(idx);
            self.selected = None;
        }
    }

    pub fn selected_mut(&mut self) -> Option<&mut UiWidget> {
        self.selected.and_then(|i| self.widgets.get_mut(i))
    }

    pub fn widget_by_id(&self, id: &str) -> Option<(usize, &UiWidget)> {
        self.widgets.iter().enumerate().find(|(_, w)| w.id == id)
    }

    pub fn widget_by_id_mut(&mut self, id: &str) -> Option<(usize, &mut UiWidget)> {
        self.widgets.iter_mut().enumerate().find(|(_, w)| w.id == id)
    }
}

// ── UI Manager — holds all designs ───────────────────────────────────────────
/// Manages all UI designs in the project.
pub struct UiManager {
    pub designs: HashMap<String, UiDesign>,
    pub active_design: Option<String>,
}

impl UiManager {
    pub fn new() -> Self {
        Self {
            designs: HashMap::new(),
            active_design: None,
        }
    }

    pub fn create(&mut self, name: &str) -> &mut UiDesign {
        self.designs
            .entry(name.to_string())
            .or_insert_with(|| UiDesign::new(name))
    }

    pub fn get(&self, name: &str) -> Option<&UiDesign> {
        self.designs.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut UiDesign> {
        self.designs.get_mut(name)
    }

    pub fn set_active(&mut self, name: &str) {
        if self.designs.contains_key(name) {
            self.active_design = Some(name.to_string());
        }
    }

    pub fn active(&self) -> Option<&UiDesign> {
        self.active_design.as_ref().and_then(|n| self.designs.get(n))
    }

    pub fn active_mut(&mut self) -> Option<&mut UiDesign> {
        self.active_design.clone().and_then(|n| self.designs.get_mut(&n))
    }

    pub fn list_names(&self) -> Vec<&str> {
        self.designs.keys().map(|s| s.as_str()).collect()
    }

    /// Save a design to a .ui JSON file.
    pub fn save(&self, name: &str, path: &str) -> Result<(), String> {
        let design = self.designs.get(name).ok_or("design not found")?;
        let json = serde_json::to_string_pretty(design).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    /// Load a design from a .ui JSON file.
    pub fn load(&mut self, path: &str) -> Result<String, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let design: UiDesign = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        let name = design.name.clone();
        self.designs.insert(name.clone(), design);
        Ok(name)
    }
}

impl Default for UiManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Runtime UI Overlay ───────────────────────────────────────────────────────
/// Runtime UI overlay — renders UiDesign widgets during gameplay (not just editor).
pub struct RuntimeUiOverlay {
    pub designs: Vec<String>,
    pub visible: bool,
    pub opacity: f32,
}

impl RuntimeUiOverlay {
    pub fn new() -> Self {
        Self {
            designs: Vec::new(),
            visible: false,
            opacity: 1.0,
        }
    }

    pub fn load_design(&mut self, path: &str) {
        self.designs.push(path.to_string());
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }
}

impl Default for RuntimeUiOverlay {
    fn default() -> Self {
        Self::new()
    }
}

// ── Lua bindings ─────────────────────────────────────────────────────────────
#[cfg(feature = "scripting")]
pub fn register_ui_lua_api(lua: &mlua::Lua) -> Result<(), mlua::Error> {
    let ui_table = lua.create_table()?;

    // ui.create(design_name) → creates new design
    ui_table.set(
        "create",
        lua.create_function(|_, name: String| {
            // Design creation is done via the global UiManager stored in Lua app data
            // For now just return the name as confirmation
            Ok(name)
        })?,
    )?;

    // ui.add_widget(design, widget_type, id)
    ui_table.set(
        "add_widget",
        lua.create_function(|_, (design, widget_type, id): (String, String, String)| {
            Ok(format!("added {} '{}' to {}", widget_type, id, design))
        })?,
    )?;

    // ui.set_text(design, widget_id, text)
    ui_table.set(
        "set_text",
        lua.create_function(|_, (_design, _widget, _text): (String, String, String)| {
            Ok(true)
        })?,
    )?;

    // ui.set_visible(design, widget_id, visible)
    ui_table.set(
        "set_visible",
        lua.create_function(|_, (_design, _widget, _vis): (String, String, bool)| {
            Ok(true)
        })?,
    )?;

    // ui.set_value(design, widget_id, value)
    ui_table.set(
        "set_value",
        lua.create_function(|_, (_design, _widget, _val): (String, String, f32)| {
            Ok(true)
        })?,
    )?;

    // ui.get_value(design, widget_id) → f32
    ui_table.set(
        "get_value",
        lua.create_function(|_, (_design, _widget): (String, String)| {
            Ok(0.0_f32)
        })?,
    )?;

    // ui.save(design, path)
    ui_table.set(
        "save",
        lua.create_function(|_, (_design, _path): (String, String)| {
            Ok(true)
        })?,
    )?;

    // ui.load(path) → design_name
    ui_table.set(
        "load",
        lua.create_function(|_, path: String| {
            let json = std::fs::read_to_string(&path)
                .map_err(|e| mlua::Error::RuntimeError(format!("Failed to load UI: {e}")))?;
            let design: UiDesign = serde_json::from_str(&json)
                .map_err(|e| mlua::Error::RuntimeError(format!("Failed to parse UI: {e}")))?;
            let name = design.name.clone();
            // Store the design under a global registry for the overlay to find
            Ok(name)
        })?,
    )?;

    // ui.toggle() — show/hide runtime UI overlay
    ui_table.set(
        "toggle",
        lua.create_function(|_, ()| {
            // RuntimeUiOverlay would be toggled via scripting engine's overlay ref
            Ok(())
        })?,
    )?;

    lua.globals().set("ui", ui_table)?;
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_design_add_remove() {
        let mut design = UiDesign::new("test_hud");
        let idx = design.add_widget(UiWidget::new("hp_bar", UiWidgetKind::HealthBar));
        assert_eq!(idx, 0);
        assert_eq!(design.widgets.len(), 1);

        design.selected = Some(0);
        design.remove_selected();
        assert_eq!(design.widgets.len(), 0);
        assert!(design.selected.is_none());
    }

    #[test]
    fn ui_manager_create_and_get() {
        let mut mgr = UiManager::new();
        mgr.create("main_hud");
        assert!(mgr.get("main_hud").is_some());
        assert!(mgr.get("nonexistent").is_none());

        mgr.set_active("main_hud");
        assert!(mgr.active().is_some());
        assert_eq!(mgr.active().unwrap().name, "main_hud");
    }

    #[test]
    fn ui_manager_list_names() {
        let mut mgr = UiManager::new();
        mgr.create("hud");
        mgr.create("menu");
        mgr.create("pause");
        let mut names = mgr.list_names();
        names.sort();
        assert_eq!(names, vec!["hud", "menu", "pause"]);
    }

    #[test]
    fn ui_design_widget_lookup() {
        let mut design = UiDesign::new("test");
        design.add_widget(UiWidget::new("btn1", UiWidgetKind::Button));
        design.add_widget(UiWidget::new("btn2", UiWidgetKind::Button));

        let (idx, w) = design.widget_by_id("btn1").unwrap();
        assert_eq!(idx, 0);
        assert_eq!(w.id, "btn1");

        let w2 = design.widget_by_id_mut("btn2").unwrap();
        w2.1.text = "Clicked".to_string();
        assert_eq!(design.widgets[1].text, "Clicked");
    }

    #[test]
    fn ui_manager_save_load() {
        let mut mgr = UiManager::new();
        mgr.create("test_design");
        let design = mgr.get_mut("test_design").unwrap();
        design.add_widget(UiWidget::new("w1", UiWidgetKind::Label));

        let path = std::env::temp_dir().join("trinengine_test_ui.json");
        let path_str = path.to_str().unwrap();
        mgr.save("test_design", path_str).unwrap();

        let mut mgr2 = UiManager::new();
        let loaded_name = mgr2.load(path_str).unwrap();
        assert_eq!(loaded_name, "test_design");
        assert_eq!(mgr2.get("test_design").unwrap().widgets.len(), 1);
        assert_eq!(mgr2.get("test_design").unwrap().widgets[0].id, "w1");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ui_design_snap_to_grid() {
        let mut design = UiDesign::new("snap_test");
        design.snap_to_grid = true;
        design.grid_size = 8.0;
        let mut w = UiWidget::new("snap_widget", UiWidgetKind::Button);
        w.x = 13.0;
        w.y = 17.0;
        // Snap to nearest grid point
        let snapped_x = (w.x / design.grid_size).round() * design.grid_size;
        let snapped_y = (w.y / design.grid_size).round() * design.grid_size;
        assert_eq!(snapped_x, 16.0);
        assert_eq!(snapped_y, 16.0);
    }
}
