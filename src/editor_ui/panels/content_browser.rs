use std::collections::HashMap;
use std::fs;

use crate::editor_assets::AssetMetadataDb;
use crate::editor_ui::texture_thumbnail;

pub fn render_content_browser_panel(
    ui: &mut egui::Ui,
    asset_db: &AssetMetadataDb,
    asset_search: &mut String,
    asset_kind_filter: &mut String,
    asset_sort_desc: &mut bool,
    texture_selected: &mut Option<String>,
    mesh_selected: &mut Option<String>,
    texture_dragging: &mut Option<String>,
    content_new_folder: &mut String,
    content_new_file: &mut String,
    texture_thumbnail_cache: &mut HashMap<String, egui::TextureHandle>,
    ctx: &egui::Context,
) {
    ui.horizontal(|ui| {
        ui.label("Search");
        ui.text_edit_singleline(asset_search);
        ui.separator();
        ui.label("Type");
        for kind in ["all", "texture", "mesh", "prefab", "script"] {
            if ui.selectable_label(asset_kind_filter == kind, kind).clicked() {
                *asset_kind_filter = kind.to_string();
            }
        }
        ui.checkbox(asset_sort_desc, "Newest first");
    });
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Folder");
        ui.text_edit_singleline(content_new_folder);
        if ui.button("Create").clicked() {
            let p = format!("Content/{}", content_new_folder.trim());
            let _ = fs::create_dir_all(p);
        }
        ui.label("File");
        ui.text_edit_singleline(content_new_file);
        if ui.button("Create").clicked() {
            let p = format!("Content/{}", content_new_file.trim());
            let _ = fs::write(p, "");
        }
    });
    ui.separator();

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

    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            for e in entries {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.set_width(170.0);
                    if e.kind == "texture" {
                        if let Some(tex) = texture_thumbnail(ctx, texture_thumbnail_cache, &e.path) {
                            let img = egui::Image::new((tex.id(), egui::vec2(150.0, 84.0))).sense(egui::Sense::click());
                            if ui.add(img).clicked() {
                                *texture_selected = Some(e.path.clone());
                            }
                        }
                    } else {
                        let (rect, _) = ui.allocate_exact_size(egui::vec2(150.0, 84.0), egui::Sense::hover());
                        let col = match e.kind.as_str() {
                            "mesh" => egui::Color32::from_rgb(92, 112, 140),
                            "prefab" => egui::Color32::from_rgb(90, 132, 102),
                            "script" => egui::Color32::from_rgb(112, 96, 142),
                            _ => egui::Color32::from_rgb(72, 72, 78),
                        };
                        ui.painter().rect_filled(rect, 6.0, col);
                    }
                    ui.monospace(&e.kind);
                    let name = std::path::Path::new(&e.path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(&e.path);
                    if ui.button(name).clicked() {
                        if e.kind == "texture" {
                            *texture_selected = Some(e.path.clone());
                        } else if e.kind == "mesh" {
                            *mesh_selected = Some(e.path.clone());
                        }
                    }
                    if e.kind == "texture" && ui.button("Drag Texture").clicked() {
                        *texture_dragging = Some(e.path.clone());
                    }
                });
            }
        });
    });
}
