// src/ui/style_panel.rs
// ──────────────────────────────────────────────────────────────────────────────
// Property inspector panel for the selected widget.
//
// Shows all editable properties of the currently selected widget:
//   - ID, kind, visibility, locked state
//   - Position (x, y), size (w, h)
//   - Anchor selector (9-point grid)
//   - Style properties (colors, font, border, corner radius, shadow, glow)
//   - Widget-specific fields (min/max value, text, lua hook)
//   - Preset button to load from library
// ──────────────────────────────────────────────────────────────────────────────

use super::*;

/// Render the style panel for the selected widget.
/// Returns true if any property was changed.
pub fn render_style_panel(ui: &mut egui::Ui, design: &mut UiDesign) -> bool {
    let mut changed = false;

    let selected_idx = match design.selected {
        Some(i) if i < design.widgets.len() => i,
        _ => {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("No widget selected").color(egui::Color32::GRAY));
            });
            return false;
        }
    };

    // Get widget info for header
    let (kind, id_clone, _visible, _locked) = {
        let w = &design.widgets[selected_idx];
        (w.kind, w.id.clone(), w.visible, w.locked)
    };

    // ── Header ──────────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(format!("{}", kind)).strong());
        ui.label(egui::RichText::new(&id_clone).color(egui::Color32::GRAY).monospace());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("🗑").on_hover_text("Delete widget").clicked() {
                design.remove_selected();
                return;
            }
        });
    });
    ui.separator();

    // ── Basic properties ────────────────────────────────────────────────
    ui.collapsing("Properties", |ui| {
        let w = &mut design.widgets[selected_idx];

        ui.horizontal(|ui| {
            ui.label("ID:");
            let id_response = ui.text_edit_singleline(&mut w.id);
            if id_response.changed() { changed = true; }
        });

        ui.horizontal(|ui| {
            ui.label("Visible:");
            if ui.checkbox(&mut w.visible, "").changed() { changed = true; }
        });
        ui.horizontal(|ui| {
            ui.label("Locked:");
            if ui.checkbox(&mut w.locked, "").changed() { changed = true; }
        });

        ui.horizontal(|ui| {
            ui.label("Z-Order:");
            if ui.add(egui::DragValue::new(&mut w.z_order).range(-100..=100)).changed() { changed = true; }
        });
    });

    // ── Transform ───────────────────────────────────────────────────────
    ui.collapsing("Transform", |ui| {
        if design.widgets[selected_idx].locked { return; }
        let w = &mut design.widgets[selected_idx];

        ui.horizontal(|ui| {
            ui.label("X:");
            if ui.add(egui::DragValue::new(&mut w.x).speed(1.0).suffix("px")).changed() { changed = true; }
            ui.label("Y:");
            if ui.add(egui::DragValue::new(&mut w.y).speed(1.0).suffix("px")).changed() { changed = true; }
        });
        ui.horizontal(|ui| {
            ui.label("W:");
            if ui.add(egui::DragValue::new(&mut w.w).speed(1.0).range(10.0..=2000.0).suffix("px")).changed() { changed = true; }
            ui.label("H:");
            if ui.add(egui::DragValue::new(&mut w.h).speed(1.0).range(8.0..=1200.0).suffix("px")).changed() { changed = true; }
        });

        // Anchor selector (3×3 grid)
        ui.label("Anchor:");
        ui.horizontal(|ui| {
            for anchor in UiAnchor::all() {
                let is_selected = w.anchor == *anchor;
                let btn = egui::Button::new(
                    egui::RichText::new(anchor.short_name())
                        .monospace()
                        .color(if is_selected { egui::Color32::BLACK } else { egui::Color32::WHITE })
                )
                .fill(if is_selected { egui::Color32::from_rgb(60, 140, 255) } else { egui::Color32::from_rgb(50, 50, 55) });
                if ui.add(btn).clicked() {
                    w.anchor = *anchor;
                    changed = true;
                }
            }
        });
    });

    // ── Style ───────────────────────────────────────────────────────────
    ui.collapsing("Style", |ui| {
        let w = &mut design.widgets[selected_idx];

        ui.label("Colors:");
        changed |= color_edit(ui, "Text", &mut w.style.text_color);
        changed |= color_edit(ui, "Background", &mut w.style.bg_color);
        changed |= color_edit(ui, "Border", &mut w.style.border_color);

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Border Width:");
            if ui.add(egui::DragValue::new(&mut w.style.border_width).speed(0.5).range(0.0..=20.0)).changed() { changed = true; }
        });
        ui.horizontal(|ui| {
            ui.label("Corner Radius:");
            if ui.add(egui::DragValue::new(&mut w.style.corner_radius).speed(1.0).range(0.0..=50.0)).changed() { changed = true; }
        });
        ui.horizontal(|ui| {
            ui.label("Font Size:");
            if ui.add(egui::DragValue::new(&mut w.style.font_size).speed(1.0).range(6.0..=72.0)).changed() { changed = true; }
        });
        ui.horizontal(|ui| {
            ui.label("Opacity:");
            if ui.add(egui::Slider::new(&mut w.style.opacity, 0.0..=1.0)).changed() { changed = true; }
        });

        ui.separator();
        ui.label("Shadow:");
        ui.horizontal(|ui| {
            ui.label("Offset X:");
            if ui.add(egui::DragValue::new(&mut w.style.shadow_offset[0]).speed(0.5)).changed() { changed = true; }
            ui.label("Offset Y:");
            if ui.add(egui::DragValue::new(&mut w.style.shadow_offset[1]).speed(0.5)).changed() { changed = true; }
        });
        changed |= color_edit(ui, "Shadow Color", &mut w.style.shadow_color);

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Glow:");
            if ui.checkbox(&mut w.style.glow_enabled, "").changed() { changed = true; }
        });
        if w.style.glow_enabled {
            changed |= color_edit(ui, "Glow Color", &mut w.style.glow_color);
            ui.horizontal(|ui| {
                ui.label("Glow Radius:");
                if ui.add(egui::DragValue::new(&mut w.style.glow_radius).speed(0.5).range(0.0..=30.0)).changed() { changed = true; }
            });
        }
    });

    // ── Bar-specific style ──────────────────────────────────────────────
    let is_bar = matches!(design.widgets[selected_idx].kind,
        UiWidgetKind::HealthBar | UiWidgetKind::ManaBar | UiWidgetKind::StaminaBar);
    if is_bar {
        ui.collapsing("Bar Style", |ui| {
            let w = &mut design.widgets[selected_idx];
            changed |= color_edit(ui, "Fill Color", &mut w.style.bar_fill_color);
            changed |= color_edit(ui, "Bar Background", &mut w.style.bar_bg_color);
            ui.horizontal(|ui| {
                ui.label("Bar Corner Radius:");
                if ui.add(egui::DragValue::new(&mut w.style.bar_corner_radius).speed(0.5).range(0.0..=20.0)).changed() { changed = true; }
            });
        });
    }

    // ── Widget-specific data ────────────────────────────────────────────
    ui.collapsing("Data", |ui| {
        let w = &mut design.widgets[selected_idx];

        ui.horizontal(|ui| {
            ui.label("Text:");
            if ui.text_edit_singleline(&mut w.text).changed() { changed = true; }
        });

        if matches!(w.kind, UiWidgetKind::HealthBar | UiWidgetKind::ManaBar | UiWidgetKind::StaminaBar |
                             UiWidgetKind::ProgressRing | UiWidgetKind::Meter | UiWidgetKind::Slider) {
            ui.horizontal(|ui| {
                ui.label("Value:");
                if ui.add(egui::DragValue::new(&mut w.value).speed(0.1)).changed() { changed = true; }
            });
            ui.horizontal(|ui| {
                ui.label("Min:");
                if ui.add(egui::DragValue::new(&mut w.min_value).speed(0.1)).changed() { changed = true; }
                ui.label("Max:");
                if ui.add(egui::DragValue::new(&mut w.max_value).speed(0.1)).changed() { changed = true; }
            });
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Lua Hook:");
            if ui.text_edit_singleline(&mut w.lua_hook).changed() { changed = true; }
        });
        ui.label(egui::RichText::new("Called when widget is clicked/changed").color(egui::Color32::GRAY).small());
    });

    changed
}

/// Color picker helper. Returns true if changed.
fn color_edit(ui: &mut egui::Ui, label: &str, color: &mut [f32; 4]) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        let r = ui.color_edit_button_rgba_premultiplied(color);
        r.changed()
    }).inner
}
