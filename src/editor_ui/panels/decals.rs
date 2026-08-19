// src/editor_ui/panels/decals.rs
// ──────────────────────────────────────────────────────────────────────────────
// Decal placement panel.
//
// A Decal is a projector box (see components::Decal) that paints albedo onto
// the G-buffer after the geometry pass — bullet holes, dirt, wetness, warning
// stripes. The renderer and the deferred decal pass already existed; this
// panel is the missing piece that lets you actually place and tune one
// without writing code.
// ──────────────────────────────────────────────────────────────────────────────

use crate::components;
use crate::editor_ui::UiFrameArgs;

fn spawn_decal(args: &mut UiFrameArgs, pos: [f32; 3]) {
    args.world.spawn((
        components::Position { x: pos[0], y: pos[1], z: pos[2] },
        components::Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
        components::Decal::default(),
    ));
}

pub fn render_decals_panel(ui: &mut egui::Ui, args: &mut UiFrameArgs) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(16, 19, 25))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(37, 44, 55)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("Decals")
                    .strong()
                    .color(egui::Color32::from_rgb(229, 232, 238)),
            );
            ui.label(
                egui::RichText::new("Projector boxes painted onto the G-buffer — bullet holes, dirt, wetness, markings.")
                    .small()
                    .color(egui::Color32::from_rgb(144, 154, 170)),
            );
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Add at Camera").clicked() {
                    let p = args.camera.position();
                    spawn_decal(args, [p.x, p.y, p.z]);
                }
                if ui.button("Add at Selection").clicked() {
                    let selected = args.selected_renderable.as_ref().copied();
                    let placed = selected.and_then(|e| {
                        args.world.get::<&components::Position>(e).ok().map(|pos| [pos.x, pos.y, pos.z])
                    });
                    if let Some(p) = placed {
                        spawn_decal(args, p);
                    }
                }
            });

            ui.separator();
            let mut to_remove: Option<hecs::Entity> = None;
            let decal_entities: Vec<hecs::Entity> = args
                .world
                .query::<(hecs::Entity, &components::Decal)>()
                .iter()
                .map(|(e, _)| e)
                .collect();

            if decal_entities.is_empty() {
                ui.label(
                    egui::RichText::new("No decals placed yet.")
                        .italics()
                        .color(egui::Color32::from_rgb(121, 131, 145)),
                );
            }

            egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                for entity in decal_entities {
                    let Ok(mut dec) = args.world.get::<&mut components::Decal>(entity) else { continue };
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("Decal {:?}", entity));
                            if ui.small_button("Select").clicked() {
                                *args.selected_renderable = Some(entity);
                            }
                            if ui.small_button("Remove").clicked() {
                                to_remove = Some(entity);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Color");
                            let mut c = [dec.color[0], dec.color[1], dec.color[2]];
                            if ui.color_edit_button_rgb(&mut c).changed() {
                                dec.color = c;
                            }
                        });
                        ui.add(egui::Slider::new(&mut dec.opacity, 0.0..=1.0).text("Opacity"));
                        ui.add(egui::Slider::new(&mut dec.roll_deg, 0.0..=360.0).text("Roll (deg)"));
                        ui.horizontal(|ui| {
                            ui.label("Size");
                            ui.add(egui::DragValue::new(&mut dec.size[0]).speed(0.05).prefix("w:"));
                            ui.add(egui::DragValue::new(&mut dec.size[1]).speed(0.05).prefix("h:"));
                            ui.add(egui::DragValue::new(&mut dec.size[2]).speed(0.05).prefix("d:"));
                        });
                    });
                }
            });

            if let Some(e) = to_remove {
                let _ = args.world.despawn(e);
                if *args.selected_renderable == Some(e) {
                    *args.selected_renderable = None;
                }
            }
        });
}
