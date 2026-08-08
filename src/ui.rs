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
pub use crate::scripting_api::*;

#[cfg(feature = "scripting")]
use mlua::prelude::*;

#[cfg(feature = "scripting")]
use std::sync::{Arc, Mutex};

/// Parse a widget kind from a Lua string (e.g. "HealthBar", "label").
#[cfg(feature = "scripting")]
fn widget_kind_from_str(s: &str) -> Option<UiWidgetKind> {
    let normalized = s.trim().to_ascii_lowercase().replace([' ', '_', '-'], "");
    match normalized.as_str() {
        "label" => Some(UiWidgetKind::Label),
        "button" => Some(UiWidgetKind::Button),
        "healthbar" | "hpbar" => Some(UiWidgetKind::HealthBar),
        "manabar" => Some(UiWidgetKind::ManaBar),
        "staminabar" => Some(UiWidgetKind::StaminaBar),
        "counter" => Some(UiWidgetKind::Counter),
        "slider" => Some(UiWidgetKind::Slider),
        "toggle" => Some(UiWidgetKind::Toggle),
        "panel" => Some(UiWidgetKind::Panel),
        "progressring" => Some(UiWidgetKind::ProgressRing),
        "meter" => Some(UiWidgetKind::Meter),
        "image" => Some(UiWidgetKind::Image),
        "tooltip" => Some(UiWidgetKind::Tooltip),
        "minimap" => Some(UiWidgetKind::Minimap),
        "damagenumber" => Some(UiWidgetKind::DamageNumber),
        _ => None,
    }
}

/// UI ScriptPlugin — mounts the `ui.*` Lua namespace backed by a real
/// `UiManager`. The manager is shared (Arc<Mutex>) so the engine can render
/// the active design each frame.
#[cfg(feature = "scripting")]
pub struct UiScriptPlugin {
    manager: Arc<Mutex<UiManager>>,
}

#[cfg(feature = "scripting")]
impl UiScriptPlugin {
    pub fn new() -> Self {
        Self {
            manager: Arc::new(Mutex::new(UiManager::new())),
        }
    }

    pub fn manager(&self) -> &Arc<Mutex<UiManager>> {
        &self.manager
    }

    pub fn manager_clone(&self) -> Arc<Mutex<UiManager>> {
        Arc::clone(&self.manager)
    }
}

#[cfg(feature = "scripting")]
impl ScriptPlugin for UiScriptPlugin {
    fn name(&self) -> &'static str {
        "ui"
    }

    fn register(&self, registry: &mut ApiRegistry) -> LuaResult<()> {
        let manager = Arc::clone(&self.manager);

        // ui.create(design_name) → creates a new design.
        registry.register_namespaced("ui", "create", move |_, name: String| {
            manager.lock().unwrap().create(&name);
            Ok(name)
        })?;

        // ui.add_widget(design, widget_kind, id) → adds a widget.
        let manager = Arc::clone(&self.manager);
        registry.register_namespaced(
            "ui",
            "add_widget",
            move |_, (design, kind, id): (String, String, String)| {
                let kind = widget_kind_from_str(&kind).ok_or_else(|| {
                    mlua::Error::RuntimeError(format!(
                        "ui.add_widget: unknown widget kind '{kind}'"
                    ))
                })?;
                let mut mgr = manager.lock().unwrap();
                let design = mgr
                    .get_mut(&design)
                    .ok_or_else(|| {
                        mlua::Error::RuntimeError(format!(
                            "ui.add_widget: design '{design}' not found"
                        ))
                    })?;
                design.add_widget(UiWidget::new(&id, kind));
                Ok(())
            },
        )?;

        // ui.set_text(design, widget_id, text)
        let manager = Arc::clone(&self.manager);
        registry.register_namespaced(
            "ui",
            "set_text",
            move |_, (design, widget, text): (String, String, String)| {
                let mut mgr = manager.lock().unwrap();
                let design = mgr
                    .get_mut(&design)
                    .ok_or_else(|| {
                        mlua::Error::RuntimeError(format!(
                            "ui.set_text: design '{design}' not found"
                        ))
                    })?;
                let (_, w) = design
                    .widget_by_id_mut(&widget)
                    .ok_or_else(|| {
                        mlua::Error::RuntimeError(format!(
                            "ui.set_text: widget '{widget}' not found"
                        ))
                    })?;
                w.text = text;
                Ok(())
            },
        )?;

        // ui.set_visible(design, widget_id, visible)
        let manager = Arc::clone(&self.manager);
        registry.register_namespaced(
            "ui",
            "set_visible",
            move |_, (design, widget, vis): (String, String, bool)| {
                let mut mgr = manager.lock().unwrap();
                let design = mgr
                    .get_mut(&design)
                    .ok_or_else(|| {
                        mlua::Error::RuntimeError(format!(
                            "ui.set_visible: design '{design}' not found"
                        ))
                    })?;
                let (_, w) = design
                    .widget_by_id_mut(&widget)
                    .ok_or_else(|| {
                        mlua::Error::RuntimeError(format!(
                            "ui.set_visible: widget '{widget}' not found"
                        ))
                    })?;
                w.visible = vis;
                Ok(())
            },
        )?;

        // ui.set_value(design, widget_id, value)
        let manager = Arc::clone(&self.manager);
        registry.register_namespaced(
            "ui",
            "set_value",
            move |_, (design, widget, val): (String, String, f32)| {
                let mut mgr = manager.lock().unwrap();
                let design = mgr
                    .get_mut(&design)
                    .ok_or_else(|| {
                        mlua::Error::RuntimeError(format!(
                            "ui.set_value: design '{design}' not found"
                        ))
                    })?;
                let (_, w) = design
                    .widget_by_id_mut(&widget)
                    .ok_or_else(|| {
                        mlua::Error::RuntimeError(format!(
                            "ui.set_value: widget '{widget}' not found"
                        ))
                    })?;
                w.value = val;
                Ok(())
            },
        )?;

        // ui.get_value(design, widget_id) → f32
        let manager = Arc::clone(&self.manager);
        registry.register_namespaced(
            "ui",
            "get_value",
            move |_, (design, widget): (String, String)| {
                let mgr = manager.lock().unwrap();
                let design = mgr
                    .get(&design)
                    .ok_or_else(|| {
                        mlua::Error::RuntimeError(format!(
                            "ui.get_value: design '{design}' not found"
                        ))
                    })?;
                let (_, w) = design
                    .widget_by_id(&widget)
                    .ok_or_else(|| {
                        mlua::Error::RuntimeError(format!(
                            "ui.get_value: widget '{widget}' not found"
                        ))
                    })?;
                Ok(w.value)
            },
        )?;

        // ui.save(design, path)
        let manager = Arc::clone(&self.manager);
        registry.register_namespaced(
            "ui",
            "save",
            move |_, (design, path): (String, String)| {
                let mgr = manager.lock().unwrap();
                mgr.save(&design, &path).map_err(|e| {
                    mlua::Error::RuntimeError(format!("ui.save: {e}"))
                })?;
                Ok(true)
            },
        )?;

        // ui.load(path) → design_name
        let manager = Arc::clone(&self.manager);
        registry.register_namespaced(
            "ui",
            "load",
            move |_, path: String| {
                let mut mgr = manager.lock().unwrap();
                let name = mgr.load(&path).map_err(|e| {
                    mlua::Error::RuntimeError(format!("ui.load: {e}"))
                })?;
                Ok(name)
            },
        )?;

        // ui.toggle() — show/hide the runtime UI overlay.
        registry.register_namespaced(
            "ui",
            "toggle",
            |_, ()| {
                // Toggling is handled by the engine overlay which owns
                // visibility; this is a no-op hook for Lua callers.
                Ok(())
            },
        )?;

        // ui.list() → table of design names.
        let manager = Arc::clone(&self.manager);
        registry.register_namespaced(
            "ui",
            "list",
            move |lua, ()| {
                let mgr = manager.lock().unwrap();
                let names = mgr.list_names();
                Ok(lua.create_table_from(names.iter().map(|s| (s.to_string(), true)))?)
            },
        )?;

        Ok(())
    }
}

#[cfg(feature = "scripting")]
pub fn register_ui_lua_api(lua: &Lua) -> Result<(), mlua::Error> {
    let plugin = UiScriptPlugin::new();
    let mut registry = ApiRegistry::new(lua);
    plugin.register(&mut registry)?;
    registry.apply()
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

    #[cfg(feature = "scripting")]
    #[test]
    fn ui_plugin_lua_api_roundtrip() -> LuaResult<()> {
        use crate::scripting_api::mount_plugins;

        let lua = Lua::new();
        let plugin = UiScriptPlugin::new();
        mount_plugins(&lua, &[&plugin])?;

        // Build a design + widget from Lua, mutate it, and read it back.
        lua.load(
            r#"
            ui.create("hud")
            ui.add_widget("hud", "healthbar", "hp")
            ui.set_text("hud", "hp", "100/100")
            ui.set_value("hud", "hp", 0.75)
            ui.set_visible("hud", "hp", false)
            "#,
        )
        .exec()?;

        let mgr = plugin.manager().lock().unwrap();
        let design = mgr.get("hud").unwrap();
        let (_, w) = design.widget_by_id("hp").unwrap();
        assert_eq!(w.text, "100/100");
        assert_eq!(w.value, 0.75);
        assert!(!w.visible);
        drop(mgr);

        // Read value back through Lua too.
        let val: f64 = lua
            .load("return ui.get_value('hud', 'hp')")
            .eval()?;
        assert_eq!(val, 0.75);
        Ok(())
    }
}
