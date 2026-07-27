// src/ui/library.rs
// ──────────────────────────────────────────────────────────────────────────────
// Reusable widget library — predefined widget templates that users can
// instantiate onto any UI design. Acts like a component palette.
//
// Templates are stored as UiWidget instances with default styling.
// Users can browse the library, click "Add" to instantiate a widget
// onto the active design, then customize it in the style panel.
//
// Built-in presets:
//   - Player HUD (health/mana/stamina bars + coin counter)
//   - Damage popup (floating damage numbers)
//   - Minimap (top-right corner minimap)
//   - Boss health bar (large centered bar)
//   - Crosshair (centered aiming reticle)
//   - Interaction prompt (bottom-center "Press E" text)
//   - Ammo counter (bottom-right ammo display)
//   - Loading bar (centered progress bar)
// ──────────────────────────────────────────────────────────────────────────────

use super::*;

/// A library template: name + a factory function that produces a widget.
pub struct WidgetTemplate {
    pub name: &'static str,
    pub description: &'static str,
    pub category: WidgetCategory,
    pub factory: fn() -> UiWidget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WidgetCategory {
    Bars,
    Text,
    Indicators,
    Interactive,
    Layout,
    Effects,
}

impl WidgetCategory {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bars => "Bars",
            Self::Text => "Text",
            Self::Indicators => "Indicators",
            Self::Interactive => "Interactive",
            Self::Layout => "Layout",
            Self::Effects => "Effects",
        }
    }

    pub fn all() -> &'static [WidgetCategory] {
        &[Self::Bars, Self::Text, Self::Indicators, Self::Interactive, Self::Layout, Self::Effects]
    }
}

/// Get all built-in widget templates.
pub fn builtin_templates() -> Vec<WidgetTemplate> {
    vec![
        WidgetTemplate {
            name: "Health Bar",
            description: "Classic red health bar with value text",
            category: WidgetCategory::Bars,
            factory: || UiWidget { id: "health_bar".to_string(), kind: UiWidgetKind::HealthBar,
                anchor: UiAnchor::TopLeft, x: 24.0, y: 24.0, w: 280.0, h: 16.0,
                style: UiWidgetStyle::bar_style([0.8, 0.2, 0.2, 1.0]),
                text: "100".to_string(), max_value: 100.0, ..UiWidget::new("_", UiWidgetKind::HealthBar) },
        },
        WidgetTemplate {
            name: "Mana Bar",
            description: "Blue mana/energy bar",
            category: WidgetCategory::Bars,
            factory: || UiWidget { id: "mana_bar".to_string(), kind: UiWidgetKind::ManaBar,
                anchor: UiAnchor::TopLeft, x: 24.0, y: 48.0, w: 280.0, h: 16.0,
                style: UiWidgetStyle::bar_style([0.2, 0.4, 0.9, 1.0]),
                text: "100".to_string(), max_value: 100.0, ..UiWidget::new("_", UiWidgetKind::ManaBar) },
        },
        WidgetTemplate {
            name: "Stamina Bar",
            description: "Yellow stamina bar",
            category: WidgetCategory::Bars,
            factory: || UiWidget { id: "stamina_bar".to_string(), kind: UiWidgetKind::StaminaBar,
                anchor: UiAnchor::TopLeft, x: 24.0, y: 72.0, w: 280.0, h: 16.0,
                style: UiWidgetStyle::bar_style([0.9, 0.8, 0.2, 1.0]),
                text: "100".to_string(), max_value: 100.0, ..UiWidget::new("_", UiWidgetKind::StaminaBar) },
        },
        WidgetTemplate {
            name: "Boss Health Bar",
            description: "Large centered boss health bar",
            category: WidgetCategory::Bars,
            factory: || UiWidget { id: "boss_health".to_string(), kind: UiWidgetKind::HealthBar,
                anchor: UiAnchor::TopCenter, x: 0.0, y: 40.0, w: 500.0, h: 24.0,
                style: UiWidgetStyle { bar_fill_color: [0.9, 0.1, 0.1, 1.0], bar_corner_radius: 6.0,
                    border_width: 2.0, border_color: [0.6, 0.1, 0.1, 0.9],
                    glow_enabled: true, glow_color: [0.8, 0.0, 0.0, 0.4], glow_radius: 8.0,
                    ..UiWidgetStyle::default() },
                text: "BOSS HP".to_string(), max_value: 1000.0, ..UiWidget::new("_", UiWidgetKind::HealthBar) },
        },
        WidgetTemplate {
            name: "Counter",
            description: "Text counter (coins, score, etc.)",
            category: WidgetCategory::Text,
            factory: || UiWidget { id: "counter".to_string(), kind: UiWidgetKind::Counter,
                anchor: UiAnchor::TopRight, x: 24.0, y: 24.0, w: 180.0, h: 24.0,
                text: "0".to_string(), ..UiWidget::new("_", UiWidgetKind::Counter) },
        },
        WidgetTemplate {
            name: "Label",
            description: "Static text label",
            category: WidgetCategory::Text,
            factory: || UiWidget { id: "label".to_string(), kind: UiWidgetKind::Label,
                text: "Text".to_string(), ..UiWidget::new("_", UiWidgetKind::Label) },
        },
        WidgetTemplate {
            name: "Crosshair",
            description: "Centered aiming crosshair",
            category: WidgetCategory::Indicators,
            factory: || UiWidget { id: "crosshair".to_string(), kind: UiWidgetKind::Label,
                anchor: UiAnchor::Center, x: -1.0, y: -1.0, w: 24.0, h: 24.0,
                style: UiWidgetStyle { text_color: [1.0, 1.0, 1.0, 0.8], bg_color: [0.0, 0.0, 0.0, 0.0],
                    font_size: 20.0, ..UiWidgetStyle::default() },
                text: "+".to_string(), ..UiWidget::new("_", UiWidgetKind::Label) },
        },
        WidgetTemplate {
            name: "Interaction Prompt",
            description: "Bottom-center 'Press E to interact' prompt",
            category: WidgetCategory::Interactive,
            factory: || UiWidget { id: "interact_prompt".to_string(), kind: UiWidgetKind::Button,
                anchor: UiAnchor::BottomCenter, x: 0.0, y: 80.0, w: 260.0, h: 40.0,
                style: UiWidgetStyle { bg_color: [0.0, 0.0, 0.0, 0.6], corner_radius: 8.0,
                    border_width: 1.0, border_color: [0.5, 0.5, 0.5, 0.5],
                    text_color: [1.0, 0.9, 0.6, 1.0], font_size: 16.0,
                    ..UiWidgetStyle::default() },
                text: "Press E to interact".to_string(), visible: false,
                ..UiWidget::new("_", UiWidgetKind::Button) },
        },
        WidgetTemplate {
            name: "Ammo Counter",
            description: "Bottom-right ammo display (30/90)",
            category: WidgetCategory::Indicators,
            factory: || UiWidget { id: "ammo_counter".to_string(), kind: UiWidgetKind::Counter,
                anchor: UiAnchor::BottomRight, x: 24.0, y: 24.0, w: 120.0, h: 32.0,
                style: UiWidgetStyle { font_size: 24.0, text_color: [1.0, 1.0, 1.0, 0.9],
                    bg_color: [0.0, 0.0, 0.0, 0.3], corner_radius: 4.0, ..UiWidgetStyle::default() },
                text: "30/90".to_string(), ..UiWidget::new("_", UiWidgetKind::Counter) },
        },
        WidgetTemplate {
            name: "Progress Ring",
            description: "Circular progress indicator",
            category: WidgetCategory::Indicators,
            factory: || UiWidget { id: "progress_ring".to_string(), kind: UiWidgetKind::ProgressRing,
                anchor: UiAnchor::BottomLeft, x: 24.0, y: 24.0, w: 80.0, h: 80.0,
                ..UiWidget::new("_", UiWidgetKind::ProgressRing) },
        },
        WidgetTemplate {
            name: "Damage Number",
            description: "Floating damage popup with glow",
            category: WidgetCategory::Effects,
            factory: || UiWidget { id: "damage".to_string(), kind: UiWidgetKind::DamageNumber,
                anchor: UiAnchor::Center, x: 0.0, y: -50.0, w: 100.0, h: 24.0,
                style: UiWidgetStyle::damage_number_style(),
                text: "-0".to_string(), visible: false,
                ..UiWidget::new("_", UiWidgetKind::DamageNumber) },
        },
        WidgetTemplate {
            name: "Minimap",
            description: "Rounded corner minimap",
            category: WidgetCategory::Layout,
            factory: || UiWidget { id: "minimap".to_string(), kind: UiWidgetKind::Minimap,
                anchor: UiAnchor::TopRight, x: 16.0, y: 60.0, w: 160.0, h: 160.0,
                style: UiWidgetStyle { corner_radius: 8.0, border_width: 2.0,
                    border_color: [0.4, 0.6, 0.8, 0.8], bg_color: [0.1, 0.15, 0.2, 0.7],
                    ..UiWidgetStyle::default() },
                ..UiWidget::new("_", UiWidgetKind::Minimap) },
        },
        WidgetTemplate {
            name: "Loading Bar",
            description: "Centered loading progress bar",
            category: WidgetCategory::Layout,
            factory: || UiWidget { id: "loading_bar".to_string(), kind: UiWidgetKind::HealthBar,
                anchor: UiAnchor::BottomCenter, x: 0.0, y: 60.0, w: 400.0, h: 20.0,
                style: UiWidgetStyle { bar_fill_color: [0.3, 0.6, 1.0, 1.0], bar_corner_radius: 10.0,
                    border_width: 1.0, border_color: [0.5, 0.5, 0.5, 0.5],
                    ..UiWidgetStyle::default() },
                text: "Loading...".to_string(), max_value: 1.0, value: 0.0,
                ..UiWidget::new("_", UiWidgetKind::HealthBar) },
        },
        WidgetTemplate {
            name: "Button",
            description: "Generic clickable button",
            category: WidgetCategory::Interactive,
            factory: || UiWidget { id: "button".to_string(), kind: UiWidgetKind::Button,
                text: "Button".to_string(), ..UiWidget::new("_", UiWidgetKind::Button) },
        },
        WidgetTemplate {
            name: "Slider",
            description: "0-1 value slider",
            category: WidgetCategory::Interactive,
            factory: || UiWidget { id: "slider".to_string(), kind: UiWidgetKind::Slider,
                value: 0.5, ..UiWidget::new("_", UiWidgetKind::Slider) },
        },
        WidgetTemplate {
            name: "Toggle",
            description: "On/off toggle switch",
            category: WidgetCategory::Interactive,
            factory: || UiWidget { id: "toggle".to_string(), kind: UiWidgetKind::Toggle,
                value: 0.0, ..UiWidget::new("_", UiWidgetKind::Toggle) },
        },
        WidgetTemplate {
            name: "Panel",
            description: "Background panel container",
            category: WidgetCategory::Layout,
            factory: || UiWidget { id: "panel".to_string(), kind: UiWidgetKind::Panel,
                style: UiWidgetStyle { corner_radius: 8.0, bg_color: [0.08, 0.08, 0.1, 0.85],
                    border_width: 1.0, border_color: [0.25, 0.25, 0.3, 0.6], ..UiWidgetStyle::default() },
                ..UiWidget::new("_", UiWidgetKind::Panel) },
        },
        WidgetTemplate {
            name: "Meter",
            description: "Segmented bar meter",
            category: WidgetCategory::Indicators,
            factory: || UiWidget { id: "meter".to_string(), kind: UiWidgetKind::Meter,
                value: 0.7, ..UiWidget::new("_", UiWidgetKind::Meter) },
        },
    ]
}

/// Render the library panel.
/// Returns Some(UiWidget) if the user clicked "Add" on a template.
pub fn render_library_panel(ui: &mut egui::Ui, active_category: &mut WidgetCategory) -> Option<UiWidget> {
    let mut add_widget = None;

    ui.label(egui::RichText::new("Widget Library").strong());
    ui.separator();

    // Category tabs
    ui.horizontal(|ui| {
        for cat in WidgetCategory::all() {
            let is_active = *active_category == *cat;
            let btn = egui::Button::new(
                egui::RichText::new(cat.name())
                    .color(if is_active { egui::Color32::BLACK } else { egui::Color32::WHITE })
            )
            .fill(if is_active { egui::Color32::from_rgb(60, 140, 255) } else { egui::Color32::from_rgb(50, 50, 55) });
            if ui.add(btn).clicked() {
                *active_category = *cat;
            }
        }
    });

    ui.separator();

    // Template list
    let templates = builtin_templates();
    for template in &templates {
        if template.category != *active_category {
            continue;
        }
        ui.group(|ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(template.name).strong());
            });
            ui.label(egui::RichText::new(template.description).color(egui::Color32::GRAY).small());
            if ui.small_button("Add to Design").clicked() {
                add_widget = Some((template.factory)());
            }
        });
    }

    add_widget
}
