// src/editor_ui/panels/levels.rs
// ──────────────────────────────────────────────────────────────────────────────
// Streaming / level management panel.
//
// Shows every registered level with its per-level settings and lets you:
//   • flip the "always loaded" (persistent) switch per level — the per-level
//     choice between streaming and not streaming,
//   • edit the level origin and streaming / unloading distances,
//   • load / unload a level by hand to preview it,
//   • add / remove levels,
//   • save the current configuration back to Content/levels.json, and
//   • reload the manifest from disk.
//
// The distance check itself runs in the main loop (levels::check_streaming)
// while the game is playing — this panel just configures it.
// ──────────────────────────────────────────────────────────────────────────────

use crate::editor_ui::UiFrameArgs;
use crate::levels::LevelEntry;

const LEVEL_MANIFEST_PATH: &str = "Content/levels.json";

fn cell_label(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).small().color(egui::Color32::from_rgb(150, 158, 172)));
}

pub fn render_levels_panel(ui: &mut egui::Ui, args: &mut UiFrameArgs) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(16, 19, 25))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(37, 44, 55)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Levels & Streaming")
                        .strong()
                        .color(egui::Color32::from_rgb(229, 232, 238)),
                );
                ui.separator();
                let loaded = args.levels.level_manager.loaded_count();
                let total = args.levels.level_manager.levels.len();
                let ents = args.levels.level_manager.total_entities();
                ui.label(
                    egui::RichText::new(format!("{} loaded / {} registered · {} entities", loaded, total, ents))
                        .small()
                        .color(egui::Color32::from_rgb(144, 154, 170)),
                );
            });
            ui.label(
                egui::RichText::new(
                    "Persistent levels are always loaded (never streamed) — use them for small \
                     levels. Streamed levels load when the player gets within the streaming \
                     distance and unload beyond the unloading distance.",
                )
                .small()
                .weak(),
            );
            ui.separator();

            // ── Streaming check configuration ────────────────────────────────
            ui.horizontal(|ui| {
                cell_label(ui, "Streaming check interval (s)");
                ui.add(
                    egui::DragValue::new(&mut args.levels.streaming_config.check_interval)
                        .speed(0.05)
                        .range(0.05..=5.0),
                );
            });
            ui.separator();

            // ── Add level ────────────────────────────────────────────────────
            ui.horizontal(|ui| {
                let mut name = ui
                    .data_mut(|d| d.get_temp::<String>(egui::Id::new("levels_new_name")))
                    .unwrap_or_default();
                let mut file = ui
                    .data_mut(|d| d.get_temp::<String>(egui::Id::new("levels_new_file")))
                    .unwrap_or_default();
                ui.label("New:");
                let changed = ui
                    .add(egui::TextEdit::singleline(&mut name).desired_width(120.0).hint_text("name"))
                    .changed();
                let fchanged = ui
                    .add(
                        egui::TextEdit::singleline(&mut file)
                            .desired_width(220.0)
                            .hint_text("Content/Scenes/level.scene"),
                    )
                    .changed();
                if changed {
                    ui.data_mut(|d| d.insert_temp(egui::Id::new("levels_new_name"), name.clone()));
                }
                if fchanged {
                    ui.data_mut(|d| d.insert_temp(egui::Id::new("levels_new_file"), file.clone()));
                }
                if ui.button("Add level").clicked() && !name.is_empty() && !file.is_empty() {
                    args.levels.level_manager.register_level(&name, &file);
                    ui.data_mut(|d| {
                        d.insert_temp(egui::Id::new("levels_new_name"), String::new());
                        d.insert_temp(egui::Id::new("levels_new_file"), String::new());
                    });
                }
            });
            ui.separator();

            // ── Per-level rows ───────────────────────────────────────────────
            // Snapshot ids so we can freely &mut the manager below.
            let level_ids: Vec<u32> = args
                .levels
                .level_manager
                .levels
                .iter()
                .map(|l| l.id)
                .collect();

            if level_ids.is_empty() {
                ui.label(
                    egui::RichText::new("No levels registered. Add one above, or create a Content/levels.json.")
                        .italics()
                        .weak(),
                );
            }

            for id in &level_ids {
                let id = *id;
                let (name, file) = {
                    let lv = args.levels.level_manager.get(id);
                    (
                        lv.map(|l| l.name.clone()).unwrap_or_default(),
                        lv.map(|l| l.file_path.to_string_lossy().to_string()).unwrap_or_default(),
                    )
                };

                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(22, 25, 32))
                    .corner_radius(5.0)
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let loaded = args.levels.level_manager.is_loaded(id);
                            let state = if args.levels.level_manager.get(id).map_or(false, |l| l.persistent) {
                                egui::RichText::new("ALWAYS LOADED")
                                    .small()
                                    .strong()
                                    .color(egui::Color32::from_rgb(120, 180, 130))
                            } else if loaded {
                                egui::RichText::new("LOADED")
                                    .small()
                                    .strong()
                                    .color(egui::Color32::from_rgb(120, 160, 220))
                            } else {
                                egui::RichText::new("STREAMED")
                                    .small()
                                    .color(egui::Color32::from_rgb(160, 150, 120))
                            };
                            ui.label(state);
                            ui.label(egui::RichText::new(&name).strong());
                            ui.label(
                                egui::RichText::new(&file)
                                    .small()
                                    .weak(),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("Remove").clicked() {
                                    args.levels.level_manager.remove_level(id, args.world);
                                }
                if loaded {
                                    if ui.button("Unload").clicked() {
                                        if !args.levels.level_manager.despawn_level(id, args.world) {
                                            args.levels.level_manager.unload_level(id);
                                        }
                                        let _ = args.levels.level_manager.release_level_meshes(
                                            id,
                                            args.meshes,
                                            args.mesh_cache,
                                        );
                                        if let Some(name) = args
                                            .levels
                                            .level_manager
                                            .get(id)
                                            .map(|l| l.name.clone())
                                        {
                                            args.levels.cross_refs.on_level_unloaded(&name);
                                        }
                                    }
                                } else if ui.button("Load").clicked() {
                                    if args.levels.level_manager.load_level(id) {
                                        let _ = args.levels.level_manager.spawn_level(
                                            id,
                                            args.world,
                                            args.meshes,
                                            args.mesh_cache,
                                            None,
                                        );
                                        if let Some(name) = args
                                            .levels
                                            .level_manager
                                            .get(id)
                                            .map(|l| l.name.clone())
                                        {
                                            if let Ok(wsm) = args.levels.world_state.lock() {
                                                wsm.apply_to_world(args.world, &name);
                                            }
                                            args.levels.cross_refs.on_level_loaded(&name, args.world);
                                        }
                                    }
                                }
                            });
                        });

                        let mut origin = args.levels.level_manager.get(id).map(|l| l.origin).unwrap_or([0.0; 3]);
                        let mut streaming = args
                            .levels
                            .level_manager
                            .get(id)
                            .map(|l| l.streaming_distance)
                            .unwrap_or(100.0);
                        let mut unloading = args
                            .levels
                            .level_manager
                            .get(id)
                            .map(|l| l.unloading_distance)
                            .unwrap_or(200.0);
                        let mut persistent = args
                            .levels
                            .level_manager
                            .get(id)
                            .map(|l| l.persistent)
                            .unwrap_or(false);

                        let mut changed = false;
                        ui.horizontal(|ui| {
                            cell_label(ui, "origin");
                            changed |= ui.add(egui::DragValue::new(&mut origin[0]).speed(0.5)).changed();
                            changed |= ui.add(egui::DragValue::new(&mut origin[1]).speed(0.5)).changed();
                            changed |= ui.add(egui::DragValue::new(&mut origin[2]).speed(0.5)).changed();
                            ui.separator();
                            cell_label(ui, "stream @");
                            changed |= ui.add(egui::DragValue::new(&mut streaming).speed(1.0).range(0.0..=10000.0)).changed();
                            cell_label(ui, "unload >");
                            changed |= ui.add(egui::DragValue::new(&mut unloading).speed(1.0).range(0.0..=20000.0)).changed();
                            ui.separator();
                            changed |= ui.checkbox(&mut persistent, "Always loaded").changed();
                        });

                        if changed {
                            if let Some(lv) = args.levels.level_manager.get_mut(id) {
                                lv.origin = origin;
                                lv.streaming_distance = streaming;
                                lv.unloading_distance = unloading;
                            }
                            if persistent {
                                args.levels.level_manager.set_persistent(id);
                            } else if args.levels.level_manager.get(id).map_or(false, |l| l.persistent) {
                                args.levels.level_manager.set_non_persistent(id);
                            }
                        }
                    });
                ui.add_space(4.0);
            }

            ui.separator();

            // ── Portal triggers ────────────────────────────────────────────────
            let level_names: Vec<String> = args
                .levels
                .level_manager
                .levels
                .iter()
                .map(|l| l.name.clone())
                .collect();
            let id_of_name = |lm: &crate::levels::LevelManager, name: &str| -> u32 {
                lm.find_by_name(name).map(|l| l.id).unwrap_or(0)
            };
            let name_of_id = |lm: &crate::levels::LevelManager, id: u32| -> String {
                lm.levels
                    .iter()
                    .find(|l| l.id == id)
                    .map(|l| l.name.clone())
                    .unwrap_or_else(|| format!("#{}", id))
            };

            ui.label(
                egui::RichText::new("Portals (transition triggers)")
                    .strong()
                    .color(egui::Color32::from_rgb(229, 232, 238)),
            );
            ui.label(
                egui::RichText::new(
                    "Portals load a target level when the player enters their radius — \
                     doorways, cave entrances, one-way transitions. Optionally unload a \
                     source level for true one-way doors. Saved to the manifest with the \
                     levels.",
                )
                .small()
                .weak(),
            );

            if args.levels.portals.is_empty() {
                ui.label(
                    egui::RichText::new("No portals defined.")
                        .italics()
                        .weak(),
                );
            }

            let portal_count = args.levels.portals.len();
            let mut remove_portal: Option<usize> = None;
            for i in 0..portal_count {
                let target_name = name_of_id(&args.levels.level_manager, args.levels.portals[i].target_level_id);
                let source_name = name_of_id(&args.levels.level_manager, args.levels.portals[i].source_level_id);
                let mut target = target_name.clone();
                let mut source = source_name.clone();

                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(22, 25, 32))
                    .corner_radius(5.0)
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("#{}", i + 1))
                                    .small()
                                    .color(egui::Color32::from_rgb(120, 160, 220)),
                            );
                            let active = args.levels.portals[i].active;
                            let mut toggled = active;
                            ui.checkbox(&mut toggled, "active");
                            if toggled != active {
                                args.levels.portals[i].active = toggled;
                            }
                            ui.separator();
                            cell_label(ui, "target");
                            egui::ComboBox::from_id_salt(egui::Id::new(("levels_portal_target", i)))
                                .width(140.0)
                                .selected_text(if target.is_empty() { "(none)" } else { &target })
                                .show_ui(ui, |ui| {
                                    for n in &level_names {
                                        ui.selectable_value(&mut target, n.clone(), n);
                                    }
                                });
                            cell_label(ui, "unload source");
                            egui::ComboBox::from_id_salt(egui::Id::new(("levels_portal_source", i)))
                                .width(140.0)
                                .selected_text(if source.is_empty() { "(none)" } else { &source })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut source, String::new(), "(none)");
                                    for n in &level_names {
                                        ui.selectable_value(&mut source, n.clone(), n);
                                    }
                                });
                        });

                        let mut shape = args.levels.portals[i].shape;

                        ui.horizontal(|ui| {
                            cell_label(ui, "shape");
                            egui::ComboBox::from_id_salt(egui::Id::new(("levels_portal_shape", i)))
                                .width(90.0)
                                .selected_text(shape.as_str())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut shape,
                                        crate::levels::portal::PortalShape::Sphere,
                                        "sphere",
                                    );
                                    ui.selectable_value(
                                        &mut shape,
                                        crate::levels::portal::PortalShape::Box,
                                        "box",
                                    );
                                    ui.selectable_value(
                                        &mut shape,
                                        crate::levels::portal::PortalShape::Capsule,
                                        "capsule",
                                    );
                                });
                            ui.separator();
                            cell_label(ui, "position");
                            ui.add(
                                egui::DragValue::new(&mut args.levels.portals[i].position[0])
                                    .speed(0.5),
                            );
                            ui.add(
                                egui::DragValue::new(&mut args.levels.portals[i].position[1])
                                    .speed(0.5),
                            );
                            ui.add(
                                egui::DragValue::new(&mut args.levels.portals[i].position[2])
                                    .speed(0.5),
                            );
                            ui.separator();
                            match shape {
                                crate::levels::portal::PortalShape::Sphere => {
                                    cell_label(ui, "radius");
                                    ui.add(
                                        egui::DragValue::new(&mut args.levels.portals[i].trigger_radius)
                                            .speed(0.1)
                                            .range(0.1..=500.0),
                                    );
                                }
                                crate::levels::portal::PortalShape::Box => {
                                    cell_label(ui, "extents");
                                    ui.add(
                                        egui::DragValue::new(&mut args.levels.portals[i].box_extents[0])
                                            .speed(0.1)
                                            .range(0.1..=1000.0),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut args.levels.portals[i].box_extents[1])
                                            .speed(0.1)
                                            .range(0.1..=1000.0),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut args.levels.portals[i].box_extents[2])
                                            .speed(0.1)
                                            .range(0.1..=1000.0),
                                    );
                                }
                                crate::levels::portal::PortalShape::Capsule => {
                                    cell_label(ui, "cap r/h");
                                    ui.add(
                                        egui::DragValue::new(&mut args.levels.portals[i].capsule_radius)
                                            .speed(0.1)
                                            .range(0.1..=500.0),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut args.levels.portals[i].capsule_half_height)
                                            .speed(0.1)
                                            .range(0.1..=1000.0),
                                    );
                                }
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("Remove").clicked() {
                                    remove_portal = Some(i);
                                }
                            });
                        });

                        if shape != args.levels.portals[i].shape {
                            args.levels.portals[i].shape = shape;
                        }

                        if target != target_name {
                            args.levels.portals[i].target_level_id = id_of_name(&args.levels.level_manager, &target);
                        }
                        if source != source_name {
                            let tid = id_of_name(&args.levels.level_manager, &source);
                            args.levels.portals[i].source_level_id = tid;
                            args.levels.portals[i].unload_source_level = tid != 0;
                        }
                    });
                ui.add_space(4.0);
            }

            ui.horizontal(|ui| {
                let mut pname = ui
                    .data_mut(|d| d.get_temp::<String>(egui::Id::new("levels_new_portal")))
                    .unwrap_or_default();
                let mut ptarget = ui
                    .data_mut(|d| d.get_temp::<String>(egui::Id::new("levels_new_portal_target")))
                    .unwrap_or_else(|| level_names.first().cloned().unwrap_or_default());
                ui.label("New portal:");
                if ui
                    .add(
                        egui::TextEdit::singleline(&mut pname)
                            .desired_width(120.0)
                            .hint_text("name"),
                    )
                    .changed()
                {
                    ui.data_mut(|d| d.insert_temp(egui::Id::new("levels_new_portal"), pname.clone()));
                }
                egui::ComboBox::from_id_salt("levels_new_portal_target_combo")
                    .width(140.0)
                    .selected_text(if ptarget.is_empty() { "(none)" } else { &ptarget })
                    .show_ui(ui, |ui| {
                        for n in &level_names {
                            ui.selectable_value(&mut ptarget, n.clone(), n);
                        }
                    });
                ui.data_mut(|d| d.insert_temp(egui::Id::new("levels_new_portal_target"), ptarget.clone()));
                if ui.button("Add portal").clicked() && !ptarget.is_empty() {
                    args.levels.portals.push(
                        crate::levels::portal::LevelPortal::new(
                            [0.0; 3],
                            id_of_name(&args.levels.level_manager, &ptarget),
                        ),
                    );
                    ui.data_mut(|d| d.insert_temp(egui::Id::new("levels_new_portal"), String::new()));
                }
            });
            if let Some(i) = remove_portal {
                args.levels.portals.remove(i);
            }

            ui.separator();

            // ── Persistence actions ──────────────────────────────────────────
            ui.horizontal(|ui| {
                if ui.button("Save manifest").clicked() {
                    let manifest = crate::levels::LevelManifest {
                        levels: args
                            .levels
                            .level_manager
                            .levels
                            .iter()
                            .map(|l| LevelEntry {
                                name: l.name.clone(),
                                file: l.file_path.to_string_lossy().to_string(),
                                origin: l.origin,
                                streaming_distance: l.streaming_distance,
                                unloading_distance: l.unloading_distance,
                                persistent: l.persistent,
                            })
                            .collect(),
                        portals: args
                            .levels
                            .portals
                            .iter()
                            .map(|p| crate::levels::PortalEntry {
                                name: String::new(),
                                position: p.position,
                                shape: p.shape.as_str().to_string(),
                                trigger_radius: p.trigger_radius,
                                box_extents: p.box_extents,
                                capsule_radius: p.capsule_radius,
                                capsule_half_height: p.capsule_half_height,
                                target_level: name_of_id(&args.levels.level_manager, p.target_level_id),
                                source_level: name_of_id(&args.levels.level_manager, p.source_level_id),
                                active: p.active,
                            })
                            .collect(),
                    };
                    match manifest.save(LEVEL_MANIFEST_PATH) {
                        Ok(()) => args.error_log.push(format!(
                            "[Streaming] saved manifest to {}",
                            LEVEL_MANIFEST_PATH
                        )),
                        Err(e) => args.error_log.push(format!("[Streaming] save failed: {}", e)),
                    }
                }
                if ui.button("Reload from disk").clicked() {
                    // Unload every streamed level, drop all registrations, then
                    // re-register from the manifest, rebuild portals, and
                    // respawn persistent ones.
                    let ids: Vec<u32> = args
                        .levels
                        .level_manager
                        .levels
                        .iter()
                        .map(|l| l.id)
                        .collect();
                    for id in ids {
                        let _ = args.levels.level_manager.despawn_level(id, args.world);
                        let _ = args.levels.level_manager.release_level_meshes(
                            id,
                            args.meshes,
                            args.mesh_cache,
                        );
                    }
                    args.levels.level_manager.clear_all();
                    let manifest = crate::levels::LevelManifest::load(LEVEL_MANIFEST_PATH)
                        .unwrap_or_default();
                    args.levels.level_manager.register_manifest(&manifest);
                    args.levels.portals = manifest
                        .portals
                        .iter()
                        .map(|e| {
                            crate::levels::portal::LevelPortal::from_entry(
                                e,
                                &args.levels.level_manager,
                            )
                        })
                        .collect();
                    match args.levels.level_manager.spawn_persistent_levels(
                        args.world,
                        args.meshes,
                        args.mesh_cache,
                        None,
                    ) {
                        Ok(n) => args.error_log.push(format!(
                            "[Streaming] reloaded {} level(s) + {} portal(s) from disk ({} persistent spawned)",
                            args.levels.level_manager.levels.len(),
                            args.levels.portals.len(),
                            n
                        )),
                        Err(e) => args.error_log.push(format!("[Streaming] reload failed: {}", e)),
                    }
                }
            });
        });
}
