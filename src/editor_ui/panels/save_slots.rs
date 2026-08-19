// src/editor_ui/panels/save_slots.rs
// ──────────────────────────────────────────────────────────────────────────────
// Save/Load panel.
//
// The save-slot system (save_slots.rs) already existed and is solid — this
// panel is the missing editor-side view onto it. Actually writing/reading a
// slot needs full `&mut App` (world rebuild, WorldStateManager restore), which
// this panel doesn't have, so it just sets a request flag the main loop
// consumes next frame — same pattern as "Bake Lighting" / "Rebuild NavMesh".
// ──────────────────────────────────────────────────────────────────────────────

use crate::editor_ui::UiFrameArgs;
use crate::save_slots::{AUTOSAVE_SLOT, FIRST_MANUAL_SLOT, MAX_SLOTS};

pub fn render_save_slots_panel(ui: &mut egui::Ui, args: &mut UiFrameArgs) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(16, 19, 25))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(37, 44, 55)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("Save / Load")
                    .strong()
                    .color(egui::Color32::from_rgb(229, 232, 238)),
            );
            ui.label(
                egui::RichText::new("Slot 0 is the rolling autosave/checkpoint; slots 1+ are explicit saves.")
                    .small()
                    .color(egui::Color32::from_rgb(144, 154, 170)),
            );
            ui.separator();

            let mut new_label = ui.data_mut(|d| d.get_temp::<String>("save_slot_new_label".into())).unwrap_or_default();
            let mut new_slot = ui.data_mut(|d| d.get_temp::<u32>("save_slot_new_index".into())).unwrap_or_default();
            if new_slot == 0 {
                new_slot = FIRST_MANUAL_SLOT;
            }

            ui.horizontal(|ui| {
                ui.label("Label");
                ui.add(egui::TextEdit::singleline(&mut new_label).desired_width(160.0).hint_text("save name..."));
                ui.label("Slot");
                ui.add(egui::DragValue::new(&mut new_slot).range(FIRST_MANUAL_SLOT..=MAX_SLOTS - 1));
                if ui.button("Save")
                    .on_hover_text("Writes the current world to this slot. The actual write happens next frame.")
                    .clicked()
                {
                    let label = if new_label.trim().is_empty() { format!("Slot {}", new_slot) } else { new_label.clone() };
                    *args.save_slot_requested = Some((new_slot, label));
                }
            });
            ui.data_mut(|d| {
                d.insert_temp("save_slot_new_label".into(), new_label);
                d.insert_temp("save_slot_new_index".into(), new_slot);
            });

            ui.separator();
            let mut entries = args.save_slots.list();
            entries.sort_by_key(|e| e.slot);
            if entries.is_empty() {
                ui.label(
                    egui::RichText::new("No saves yet.")
                        .italics()
                        .color(egui::Color32::from_rgb(121, 131, 145)),
                );
            }

            egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                for entry in &entries {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            let tag = if entry.slot == AUTOSAVE_SLOT { "[autosave]" } else { "" };
                            ui.label(format!("Slot {} {} — {}", entry.slot, tag, entry.meta.label));
                        });
                        ui.label(
                            egui::RichText::new(format!(
                                "{} · {} entities · {}",
                                entry.meta.scene, entry.meta.entity_count, entry.meta.timestamp_string()
                            ))
                            .small()
                            .color(egui::Color32::from_rgb(144, 154, 170)),
                        );
                        ui.horizontal(|ui| {
                            if ui.button("Load").clicked() {
                                *args.load_slot_requested = Some(entry.slot);
                            }
                            if ui.button("Delete").clicked() {
                                let _ = args.save_slots.delete(entry.slot);
                            }
                        });
                    });
                }
            });
        });
}
