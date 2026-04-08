use crate::components;
use crate::editor_ui::UiFrameArgs;
use egui::{Color32, RichText};

fn header_panel(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(16, 19, 25))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(37, 44, 55)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.label(RichText::new(title).strong().color(Color32::from_rgb(229, 232, 238)));
            ui.label(RichText::new(subtitle).small().color(Color32::from_rgb(142, 152, 168)));
        });
}

pub fn render_outliner_panel(ui: &mut egui::Ui, args: &mut UiFrameArgs<'_>) {
    let entities: Vec<hecs::Entity> = args
        .world
        .query::<(hecs::Entity, &components::Renderable)>()
        .iter()
        .map(|(e, _)| e)
        .collect();

    let mut search = ui
        .data_mut(|d| d.get_temp::<String>("outliner_search".into()))
        .unwrap_or_default();

    header_panel(
        ui,
        "World Outliner",
        &format!("{} actors in current scene", entities.len()),
    );
    ui.add_space(8.0);

    egui::Frame::new()
        .fill(Color32::from_rgb(14, 17, 22))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(33, 39, 49)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Search").small().color(Color32::from_rgb(147, 158, 172)));
                ui.add(
                    egui::TextEdit::singleline(&mut search)
                        .desired_width(f32::INFINITY)
                        .hint_text("Entity id"),
                );
            });
        });
    ui.data_mut(|d| d.insert_temp("outliner_search".into(), search.clone()));
    ui.add_space(8.0);

    egui::Frame::new()
        .fill(Color32::from_rgb(12, 15, 20))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(32, 38, 48)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(6))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Type").small().strong().color(Color32::from_rgb(168, 176, 188)));
                ui.add_space(14.0);
                ui.label(RichText::new("Label").small().strong().color(Color32::from_rgb(168, 176, 188)));
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                let search_lower = search.to_ascii_lowercase();
                for e in entities {
                    let label = format!("Entity {:?}", e);
                    if !search_lower.is_empty() && !label.to_ascii_lowercase().contains(&search_lower) {
                        continue;
                    }
                    let selected = args.selected_renderable.map(|s| s == e).unwrap_or(false);
                    let fill = if selected {
                        Color32::from_rgb(42, 62, 92)
                    } else {
                        Color32::from_rgb(20, 24, 31)
                    };
                    let stroke = if selected {
                        Color32::from_rgb(92, 124, 178)
                    } else {
                        Color32::from_rgb(33, 39, 49)
                    };
                    egui::Frame::new()
                        .fill(fill)
                        .stroke(egui::Stroke::new(1.0, stroke))
                        .corner_radius(6.0)
                        .inner_margin(egui::Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let icon_rect = ui
                                    .allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover())
                                    .0;
                                ui.painter().rect_filled(icon_rect, 3.0, Color32::from_rgb(58, 76, 103));
                                ui.painter().text(
                                    icon_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "SM",
                                    egui::FontId::proportional(9.5),
                                    Color32::from_rgb(232, 236, 241),
                                );
                                if ui
                                    .selectable_label(selected, label)
                                    .clicked()
                                {
                                    *args.selected_renderable = Some(e);
                                }
                            });
                        });
                    ui.add_space(3.0);
                }
            });
        });
}
