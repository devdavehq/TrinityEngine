use std::collections::HashMap;
use std::fs;

use crate::editor_assets::AssetMetadataDb;
use crate::editor_ui::texture_thumbnail;

fn kind_color(kind: &str) -> egui::Color32 {
    match kind {
        "mesh" => egui::Color32::from_rgb(96, 122, 164),
        "prefab" => egui::Color32::from_rgb(92, 142, 110),
        "script" => egui::Color32::from_rgb(126, 100, 168),
        "texture" => egui::Color32::from_rgb(160, 138, 84),
        "material" => egui::Color32::from_rgb(90, 124, 148),
        "foliage" => egui::Color32::from_rgb(90, 148, 102),
        _ => egui::Color32::from_rgb(78, 82, 94),
    }
}

pub fn render_content_browser_panel(
    ui: &mut egui::Ui,
    asset_db: &AssetMetadataDb,
    asset_search: &mut String,
    asset_kind_filter: &mut String,
    asset_sort_desc: &mut bool,
    texture_selected: &mut Option<String>,
    mesh_selected: &mut Option<String>,
    texture_dragging: &mut Option<String>,
    _content_new_folder: &mut String,
    _content_new_file: &mut String,
    texture_thumbnail_cache: &mut HashMap<String, egui::TextureHandle>,
    icon_texture_cache: &HashMap<String, egui::TextureHandle>,
    preferred_script_editor: &str,
    show_material_editor: &mut bool,
    show_foliage_editor: &mut bool,
    error_log: &mut Vec<String>,
    ctx: &egui::Context,
    bake_requested: &mut bool,
) {
    let mut grid_mode = ui
        .data_mut(|d| d.get_temp::<bool>("content_browser_grid_mode".into()))
        .unwrap_or(true);

    egui::Frame::new()
        .fill(egui::Color32::from_rgb(16, 19, 25))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(37, 44, 55)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Content Browser").strong().color(egui::Color32::from_rgb(229, 232, 238)));
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("{} assets indexed", asset_db.entries.len()))
                        .small()
                        .color(egui::Color32::from_rgb(144, 154, 170)),
                );
                if ui.button(egui::RichText::new("Bake Lighting").small()).clicked() {
                    *bake_requested = true;
                    error_log.push("[Lighting] Bake requested".to_string());
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.selectable_value(&mut grid_mode, false, "List");
                    ui.selectable_value(&mut grid_mode, true, "Grid");
                });
            });
        });
    ui.data_mut(|d| d.insert_temp("content_browser_grid_mode".into(), grid_mode));
    ui.add_space(8.0);

    egui::Frame::new()
        .fill(egui::Color32::from_rgb(14, 17, 22))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(33, 39, 49)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Search");
                ui.add(
                    egui::TextEdit::singleline(asset_search)
                        .desired_width(220.0)
                        .hint_text("Name or path"),
                );
                ui.separator();
                ui.label("Type");
                for kind in ["all", "texture", "mesh", "prefab", "script", "material", "foliage"] {
                    let selected = asset_kind_filter == kind;
                    if ui.selectable_label(selected, kind).clicked() {
                        *asset_kind_filter = kind.to_string();
                    }
                }
                ui.separator();
                ui.checkbox(asset_sort_desc, "Newest first");
            });
        });
    ui.add_space(8.0);

    egui::Frame::new()
        .fill(egui::Color32::from_rgb(14, 17, 22))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(33, 39, 49)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Scene").small().strong().color(egui::Color32::from_rgb(168, 176, 188)));
                ui.separator();
                if ui.button(egui::RichText::new("New Scene").small().color(egui::Color32::from_rgb(147, 158, 172))).clicked() {
                    // Signal: clear the world. Main loop will handle it.
                    ui.data_mut(|d| d.insert_temp("scene_action_new".into(), true));
                }
                if ui.button(egui::RichText::new("Load Scene").small().color(egui::Color32::from_rgb(147, 158, 172))).clicked() {
                    // List .scene files and show a popup.
                    ui.data_mut(|d| d.insert_temp("scene_action_load".into(), true));
                }
            });

            // Show recent scenes if any.
            let show_recent = ui
                .data_mut(|d| d.get_temp::<bool>("scene_action_load".into()))
                .unwrap_or(false);

            if show_recent {
                ui.separator();
                ui.label(egui::RichText::new("Recent Scenes:").small().color(egui::Color32::from_rgb(147, 158, 172)));
                // Scan Content/ for .scene files.
                let scenes = crate::scene::SceneManager::list_scene_files("Content");
                for scene_path in &scenes {
                    let name = std::path::Path::new(scene_path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or(scene_path);
                    if ui.button(name).clicked() {
                        ui.data_mut(|d| d.insert_temp("scene_load_path".into(), scene_path.clone()));
                        ui.data_mut(|d| d.insert_temp("scene_action_load".into(), false));
                    }
                }
                if ui.button("Cancel").clicked() {
                    ui.data_mut(|d| d.insert_temp("scene_action_load".into(), false));
                }
            }
        });
    ui.add_space(8.0);

    let search = asset_search.to_ascii_lowercase();
    let mut entries: Vec<_> = asset_db
        .entries
        .iter()
        .filter(|e| {
            (*asset_kind_filter == "all" || e.kind == *asset_kind_filter)
                && (search.is_empty() || e.path.to_ascii_lowercase().contains(&search))
        })
        .collect();
    entries.sort_by_key(|e| e.modified_unix_secs);
    if *asset_sort_desc {
        entries.reverse();
    }

    egui::Frame::new()
        .fill(egui::Color32::from_rgb(12, 15, 20))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(32, 38, 48)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                if grid_mode {
                    ui.horizontal_wrapped(|ui| {
                        for e in entries {
                            let color = kind_color(&e.kind);
                            egui::Frame::new()
                                .fill(egui::Color32::from_rgb(18, 22, 29))
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(36, 43, 53)))
                                .corner_radius(8.0)
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    ui.set_width(184.0);
                                    let preview_size = egui::vec2(168.0, 92.0);
                                    if e.kind == "texture" {
                                        if let Some(tex) = texture_thumbnail(ctx, texture_thumbnail_cache, &e.path) {
                                            let img = egui::Image::new((tex.id(), preview_size)).sense(egui::Sense::click());
                                            if ui.add(img).clicked() {
                                                *texture_selected = Some(e.path.clone());
                                            }
                                        }
                                    } else if let Some(icon) = icon_texture_cache.get(match e.kind.as_str() {
                                        "mesh" => "mesh",
                                        "prefab" => "prefab",
                                        "script" => "script",
                                        "material" => "material",
                                        "foliage" => "foliage",
                                        _ => "file",
                                    }) {
                                        let _ = ui.add(egui::Image::new((icon.id(), preview_size)));
                                    } else {
                                        let (r, _) = ui.allocate_exact_size(preview_size, egui::Sense::hover());
                                        ui.painter().rect_filled(r, 6.0, color);
                                    }
                                    ui.add_space(6.0);
                                    ui.horizontal(|ui| {
                                        ui.colored_label(color, egui::RichText::new("■").strong());
                                        ui.monospace(&e.kind);
                                    });
                                    let name = std::path::Path::new(&e.path)
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or(&e.path);
                                    let resp = ui.add_sized(
                                        [168.0, 24.0],
                                        egui::Button::new(name).wrap(),
                                    );
                                    if resp.clicked() {
                                        if e.kind == "texture" {
                                            *texture_selected = Some(e.path.clone());
                                        } else if e.kind == "mesh" {
                                            *mesh_selected = Some(e.path.clone());
                                        }
                                    }
                                    if resp.double_clicked() {
                                        if e.kind == "script" {
                                            if let Err(err) = open_external_editor(preferred_script_editor, &e.path) {
                                                error_log.push(format!("[Content] Open script failed: {}", err));
                                            }
                                        } else if e.kind == "material" {
                                            *show_material_editor = true;
                                        } else if e.kind == "foliage" {
                                            *show_foliage_editor = true;
                                        }
                                    }
                                    ui.horizontal_wrapped(|ui| {
                                        if e.kind == "texture" && ui.button("Drag").clicked() {
                                            *texture_dragging = Some(e.path.clone());
                                        }
                                        if e.kind == "mesh" && ui.button("Prefab").clicked() {
                                            let stem = std::path::Path::new(&e.path)
                                                .file_stem()
                                                .and_then(|s| s.to_str())
                                                .unwrap_or("mesh");
                                            let _ = fs::create_dir_all("Content/Prefabs");
                                            let prefab_path = format!("Content/Prefabs/{}.prefab", stem);
                                            let prefab_text = format!(
                                                "mesh={}\npos=0 0 0\nrot=0 0 0\nscale=1 1 1\ncolor=1 1 1\nmetallic=0\nroughness=1\nao=1\nscript=\nalbedo_tex=\nnormal_tex=\nmr_tex=\n",
                                                e.path
                                            );
                                            match fs::write(&prefab_path, prefab_text) {
                                                Ok(()) => error_log.push(format!("[Content] Prefab created: {}", prefab_path)),
                                                Err(err) => error_log.push(format!("[Content] Prefab create failed: {}", err)),
                                            }
                                        }
                                        if ui.button("Delete").clicked() {
                                            match fs::remove_file(&e.path) {
                                                Ok(()) => error_log.push(format!("[Content] Deleted asset: {}", e.path)),
                                                Err(err) => error_log.push(format!("[Content] Delete failed ({}): {}", e.path, err)),
                                            }
                                        }
                                    });
                                });
                        }
                    });
                } else {
                    for e in entries {
                        let color = kind_color(&e.kind);
                        egui::Frame::new()
                            .fill(egui::Color32::from_rgb(18, 22, 29))
                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(36, 43, 53)))
                            .corner_radius(6.0)
                            .inner_margin(egui::Margin::symmetric(8, 6))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.colored_label(color, egui::RichText::new("■").strong());
                                    ui.monospace(format!("{:>8}", e.kind));
                                    ui.separator();
                                    let name = std::path::Path::new(&e.path)
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or(&e.path);
                                    let resp = ui.selectable_label(false, name);
                                    if resp.clicked() && e.kind == "mesh" {
                                        *mesh_selected = Some(e.path.clone());
                                    }
                                    if resp.clicked() && e.kind == "texture" {
                                        *texture_selected = Some(e.path.clone());
                                    }
                                    ui.separator();
                                    if ui.button("Open").clicked() {
                                        if e.kind == "script" {
                                            let _ = open_external_editor(preferred_script_editor, &e.path);
                                        } else if e.kind == "material" {
                                            *show_material_editor = true;
                                        } else if e.kind == "foliage" {
                                            *show_foliage_editor = true;
                                        }
                                    }
                                    if ui.button("Delete").clicked() {
                                        match fs::remove_file(&e.path) {
                                            Ok(()) => error_log.push(format!("[Content] Deleted asset: {}", e.path)),
                                            Err(err) => error_log.push(format!("[Content] Delete failed ({}): {}", e.path, err)),
                                        }
                                    }
                                });
                            });
                        ui.add_space(3.0);
                    }
                }
            });
        });
}

fn open_external_editor(template: &str, file_path: &str) -> Result<(), String> {
    let cmd = if template.trim().is_empty() {
        format!("code -r \"{}\"", file_path)
    } else {
        template.replace("{file}", file_path)
    };
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", &cmd])
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::process::Command::new("sh")
            .args(["-lc", &cmd])
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
