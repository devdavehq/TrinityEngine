// src/editor_ui/panels/outliner.rs
// ── World Outliner Panel ─────────────────────────────────────────────────
// Tree view with parent-child hierarchy, groups, search, and drag-drop reparent.

use crate::components;
use crate::animation::anim_graph::AnimGraphComponent;
use crate::core::hierarchy::{build_hierarchy, set_parent};
use crate::editor_ui::UiFrameArgs;
use egui::{Color32, RichText};
use std::collections::HashMap;

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

/// Spawn a shape entity with the given name and scale.
fn spawn_shape(world: &mut hecs::World, name: &str, scale: [f32; 3]) {
    let e = world.spawn((
        components::Position { x: 0.0, y: scale[1] * 0.5, z: 0.0 },
        components::Renderable {
            mesh: crate::assets::Handle::new(0),
            color: match name {
                "Cube" => [0.7, 0.7, 0.7],
                "Sphere" => [0.6, 0.8, 1.0],
                "Cylinder" => [0.8, 0.7, 0.6],
                "Plane" => [0.4, 0.7, 0.4],
                "Quad" => [0.9, 0.9, 0.9],
                _ => [0.8, 0.8, 0.8],
            },
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            scale,
        },
        components::Rotation::default(),
        components::GroupNode::new(name),
    ));
    tracing::info!("[Outliner] Spawned {} entity", name);
    let _ = e;
}

/// Render a single entity row in the tree, with drag-drop support for reparenting.
fn render_entity_row(
    ui: &mut egui::Ui,
    entity: hecs::Entity,
    label: &str,
    depth: usize,
    is_parent: bool,
    selected: bool,
    args: &mut UiFrameArgs<'_>,
) {
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

    // Check if this entity is the current drag source.
    let drag_source = ui
        .data(|d| d.get_temp::<u64>("outliner_drag_source".into()))
        .unwrap_or(0);
    let my_bits = entity.to_bits().get();
    let is_drag_target = drag_source != 0 && drag_source != my_bits;
    let is_drop_hovered = is_drag_target && ui.rect_contains_pointer(ui.max_rect());

    let mut frame_fill = fill;
    let mut frame_stroke = stroke;
    if is_drop_hovered {
        frame_fill = Color32::from_rgb(52, 72, 102);
        frame_stroke = Color32::from_rgb(120, 160, 220);
    }

    egui::Frame::new()
        .fill(frame_fill)
        .stroke(egui::Stroke::new(1.0, frame_stroke))
        .corner_radius(6.0)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                // Indent based on depth.
                let indent = depth as f32 * 16.0;
                ui.add_space(indent);

                // Icon: expand/collapse arrow if has children, otherwise entity icon.
                let icon_text = if is_parent { "▶" } else { "•" };
                let icon_color = if is_parent {
                    Color32::from_rgb(168, 176, 188)
                } else {
                    Color32::from_rgb(88, 104, 132)
                };

                ui.label(
                    RichText::new(icon_text)
                        .color(icon_color)
                        .size(10.0)
                        .monospace(),
                );

                // Entity type icon.
                let icon_rect = ui
                    .allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover())
                    .0;

                let icon_bg = Color32::from_rgb(58, 76, 103);
                ui.painter().rect_filled(icon_rect, 3.0, icon_bg);
                ui.painter().text(
                    icon_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "SM",
                    egui::FontId::proportional(9.5),
                    Color32::from_rgb(232, 236, 241),
                );

                // Entity label — selectable, draggable, droppable.
                let response = ui
                    .selectable_label(selected, label)
                    .interact(egui::Sense::click_and_drag());

                // ── Drag Source ──────────────────────────────────────────
                if response.drag_started() {
                    ui.data_mut(|d| d.insert_temp("outliner_drag_source".into(), entity.to_bits().get()));
                }

                // ── Drop Target ──────────────────────────────────────────
                if is_drag_target && response.hovered() && ui.input(|i| i.pointer.any_released()) {
                    // Reparent the dragged entity under this entity.
                    let target = entity;
                    if let Some(dragged_bits) = ui.data(|d| d.get_temp::<u64>("outliner_drag_source".into())) {
                        if let Some(dragged_entity) = hecs::Entity::from_bits(dragged_bits) {
                            set_parent(args.world, dragged_entity, target);
                            args.error_log.push(format!(
                                "[Hierarchy] Reparented {:?} under {:?}",
                                dragged_entity, target
                            ));
                        }
                    }
                    ui.data_mut(|d| d.remove_temp::<u64>("outliner_drag_source".into()));
                }

                // Clear drag on mouse release anywhere.
                if response.drag_stopped() {
                    ui.data_mut(|d| d.remove_temp::<u64>("outliner_drag_source".into()));
                }

                if response.clicked() {
                    *args.selected_renderable = Some(entity);
                }
            });
        });
    ui.add_space(3.0);
}

/// Recursive tree traversal for rendering the hierarchy.
fn render_tree(
    ui: &mut egui::Ui,
    entity: hecs::Entity,
    depth: usize,
    parent_to_children: &HashMap<hecs::Entity, Vec<hecs::Entity>>,
    entity_labels: &HashMap<hecs::Entity, String>,
    search: &str,
    args: &mut UiFrameArgs<'_>,
) {
    let label = entity_labels
        .get(&entity)
        .cloned()
        .unwrap_or_else(|| format!("Entity {:?}", entity));

    // Filter by search.
    let search_lower = search.to_ascii_lowercase();
    let matches_search = search_lower.is_empty() || label.to_ascii_lowercase().contains(&search_lower);

    let has_children = parent_to_children
        .get(&entity)
        .map(|c| !c.is_empty())
        .unwrap_or(false);

    let selected = args.selected_renderable.map(|s| s == entity).unwrap_or(false);

    // Only render if matches search or has matching descendants.
    let has_matching_descendant = if !matches_search && !search_lower.is_empty() {
        parent_to_children.get(&entity).map_or(false, |children| {
            children.iter().any(|child| {
                let child_label = entity_labels
                    .get(child)
                    .cloned()
                    .unwrap_or_else(|| format!("Entity {:?}", child));
                child_label.to_ascii_lowercase().contains(&search_lower)
                    || has_descendant_matching(*child, parent_to_children, entity_labels, &search_lower)
            })
        })
    } else {
        false
    };

    if matches_search || has_matching_descendant || search_lower.is_empty() {
        render_entity_row(ui, entity, &label, depth, has_children, selected, args);

        // Recurse into children.
        if let Some(children) = parent_to_children.get(&entity) {
            for &child in children {
                render_tree(
                    ui,
                    child,
                    depth + 1,
                    parent_to_children,
                    entity_labels,
                    search,
                    args,
                );
            }
        }
    }
}

/// Check if any descendant matches the search.
fn has_descendant_matching(
    entity: hecs::Entity,
    parent_to_children: &HashMap<hecs::Entity, Vec<hecs::Entity>>,
    entity_labels: &HashMap<hecs::Entity, String>,
    search_lower: &str,
) -> bool {
    if let Some(children) = parent_to_children.get(&entity) {
        for child in children {
            let label = entity_labels
                .get(child)
                .cloned()
                .unwrap_or_else(|| format!("Entity {:?}", child));
            if label.to_ascii_lowercase().contains(search_lower) {
                return true;
            }
            if has_descendant_matching(*child, parent_to_children, entity_labels, search_lower) {
                return true;
            }
        }
    }
    false
}

/// Import an external asset (mesh/texture/audio) into this project's Content/
/// directory. The file is copied under Content/Meshes (or Content/Textures)
/// with a unique name, then immediately loaded into the asset store so it can
/// be spawned/assigned in the current scene.
///
/// Returns `Some(Ok(()))` on success, `Some(Err(msg))` on failure, and `None`
/// if the user cancelled the dialog.
fn import_asset_to_content(
    external_path: &std::path::Path,
    args: &mut UiFrameArgs<'_>,
) -> Option<Result<(), String>> {
    let ext = external_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let (target_dir, is_mesh) = match ext.as_str() {
        "obj" | "gltf" | "glb" => ("Content/Meshes", true),
        "png" | "jpg" | "jpeg" | "tga" => ("Content/Textures", false),
        "wav" | "ogg" | "mp3" | "flac" => ("Content/Audio", false),
        _ => ("Content/Imported", false),
    };

    if std::fs::create_dir_all(target_dir).is_err() {
        return Some(Err(format!(
            "Could not create import directory {}",
            target_dir
        )));
    }

    let file_name = external_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("asset")
        .to_string();

    // De-duplicate: append a numeric suffix if the name already exists.
    let mut dest_name = file_name.clone();
    let mut counter = 1;
    while std::path::Path::new(target_dir).join(&dest_name).exists() {
        let stem = std::path::Path::new(&file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("asset");
        dest_name = format!("{}_{}.{}", stem, counter, ext);
        counter += 1;
    }

    let dest_path = std::path::Path::new(target_dir).join(&dest_name);
    if let Err(e) = std::fs::copy(external_path, &dest_path) {
        return Some(Err(format!("Failed to copy {} -> {}: {}", external_path.display(), dest_path.display(), e)));
    }

    let dest_str = dest_path.to_string_lossy().to_string();
    tracing::info!("[Import] Copied {} -> {}", external_path.display(), dest_str);

    if is_mesh {
        // Load immediately into the mesh store + cache so the user can assign
        // it to a Renderable without restarting.
        match crate::assets::mesh::Mesh::load(&dest_str) {
            Ok(mesh) => {
                let handle = args.meshes.add(mesh);
                args.mesh_cache.insert(dest_str.clone(), handle);
                Some(Ok(()))
            }
            Err(e) => Some(Err(format!("Imported mesh failed to parse: {}", e))),
        }
    } else {
        // Textures/audio are picked up by the existing hot-reload watcher and
        // asset browser scan on the next frame; no store registration needed.
        Some(Ok(()))
    }
}

pub fn render_outliner_panel(ui: &mut egui::Ui, args: &mut UiFrameArgs<'_>) {
    let (roots, parent_to_children) = build_hierarchy(args.world);
    let total_entities = args.world.query::<hecs::Entity>().iter().count();

    // Build entity labels map.
    let mut entity_labels: HashMap<hecs::Entity, String> = HashMap::new();
    for (entity, _) in args.world.query::<(hecs::Entity, &components::Renderable)>().iter() {
        entity_labels.insert(entity, format!("Entity {:?}", entity));
    }
    // Also add group nodes.
    for (entity, group) in args.world.query::<(hecs::Entity, &components::GroupNode)>().iter() {
        entity_labels.insert(entity, format!("[{}] ", group.name_str()));
    }

    let mut search = ui
        .data_mut(|d| d.get_temp::<String>("outliner_search".into()))
        .unwrap_or_default();

    header_panel(
        ui,
        "World Outliner",
        &format!("{} actors in current scene", total_entities),
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
                        .hint_text("Entity name"),
                );
            });
        });
    ui.data_mut(|d| d.insert_temp("outliner_search".into(), search.clone()));
    ui.add_space(8.0);

    // Add Entity menu — dropdown with shapes, lights, cameras, imports.
    let add_menu_id = egui::Id::new("add_entity_menu");
    egui::Frame::new()
        .fill(Color32::from_rgb(14, 17, 22))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(33, 39, 49)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                // ── Add Entity dropdown ──────────────────────────────────
                let add_btn = ui.button(
                    RichText::new("➕ Add").small().color(Color32::from_rgb(180, 220, 140)),
                );
                if add_btn.clicked() {
                    ui.memory_mut(|m| m.open_popup(add_menu_id));
                }

                egui::Area::new(egui::Id::new("add_entity_menu"))
                    .fixed_pos(add_btn.rect.left_bottom())
                    .order(egui::Order::Foreground)
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.set_min_width(180.0);

                            // ── Shapes ──────────────────────────────────
                            ui.label(RichText::new("Shapes").strong().color(Color32::from_rgb(140, 180, 220)));
                            if ui.button("Cube").clicked() {
                                spawn_shape(args.world, "Cube", [1.0, 1.0, 1.0]);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            if ui.button("Sphere").clicked() {
                                spawn_shape(args.world, "Sphere", [1.0, 1.0, 1.0]);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            if ui.button("Cylinder").clicked() {
                                spawn_shape(args.world, "Cylinder", [1.0, 1.0, 1.0]);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            if ui.button("Plane").clicked() {
                                spawn_shape(args.world, "Plane", [5.0, 0.1, 5.0]);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            if ui.button("Quad").clicked() {
                                spawn_shape(args.world, "Quad", [2.0, 2.0, 1.0]);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            ui.separator();

                            // ── Lights ──────────────────────────────────
                            ui.label(RichText::new("Lights").strong().color(Color32::from_rgb(220, 200, 120)));
                            if ui.button("Point Light").clicked() {
                                let e = args.world.spawn((
                                    components::Position { x: 0.0, y: 3.0, z: 0.0 },
                                    components::PointLight {
                                        color: [1.0, 0.9, 0.7],
                                        intensity: 1.0,
                                        range: 15.0,
                                        light_type: 1.0,
                                        spot_angle: 45.0,
                                        shadow_casting: true,
                                    },
                                    components::GroupNode::new("Point Light"),
                                ));
                                *args.selected_renderable = Some(e);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            if ui.button("Directional Light").clicked() {
                                let e = args.world.spawn((
                                    components::Position { x: 0.0, y: 10.0, z: 0.0 },
                                    components::PointLight {
                                        color: [1.0, 0.95, 0.8],
                                        intensity: 2.0,
                                        range: 1000.0,
                                        light_type: 0.0,
                                        spot_angle: 0.0,
                                        shadow_casting: true,
                                    },
                                    components::GroupNode::new("Directional Light"),
                                ));
                                *args.selected_renderable = Some(e);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            if ui.button("Spot Light").clicked() {
                                let e = args.world.spawn((
                                    components::Position { x: 0.0, y: 4.0, z: 0.0 },
                                    components::PointLight {
                                        color: [1.0, 1.0, 1.0],
                                        intensity: 3.0,
                                        range: 20.0,
                                        light_type: 2.0,
                                        spot_angle: 30.0,
                                        shadow_casting: true,
                                    },
                                    components::GroupNode::new("Spot Light"),
                                ));
                                *args.selected_renderable = Some(e);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            ui.separator();

                            // ── Objects ──────────────────────────────────
                            ui.label(RichText::new("Objects").strong().color(Color32::from_rgb(180, 140, 220)));
                            if ui.button("Camera").clicked() {
                                let e = args.world.spawn((
                                    components::Position { x: 0.0, y: 2.0, z: 5.0 },
                                    components::Rotation::default(),
                                    components::GroupNode::new("Camera"),
                                ));
                                *args.selected_renderable = Some(e);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            if ui.button("Empty Entity").clicked() {
                                let e = args.world.spawn((
                                    components::Position { x: 0.0, y: 0.0, z: 0.0 },
                                    components::GroupNode::new("Empty"),
                                ));
                                *args.selected_renderable = Some(e);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            if ui.button("Group Node").clicked() {
                                let e = args.world.spawn((
                                    components::GroupNode::new("New Group"),
                                    components::Children::new(),
                                ));
                                *args.selected_renderable = Some(e);
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            if ui.button("Animated Character").clicked() {
                                let e = args.world.spawn((
                                    components::Position { x: 0.0, y: 0.0, z: 0.0 },
                                    components::Rotation::default(),
                                    components::Renderable {
                                        mesh: crate::assets::Handle::new(0),
                                        color: [0.6, 0.7, 0.9],
                                        metallic: 0.0,
                                        roughness: 0.6,
                                        ao: 1.0,
                                        scale: [1.0, 1.0, 1.0],
                                    },
                                    components::GroupNode::new("Animated Character"),
                                    AnimGraphComponent::new(),
                                ));
                                *args.selected_renderable = Some(e);
                                tracing::info!("[Outliner] Spawned Animated Character with AnimGraphComponent");
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                            ui.separator();

                            // ── Import ──────────────────────────────────
                            ui.label(RichText::new("Import").strong().color(Color32::from_rgb(140, 200, 180)));
                            if ui.button("Import Mesh...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("3D Models", &["gltf", "glb", "obj"])
                                    .add_filter("All Files", &["*"])
                                    .pick_file()
                                {
                                    if let Some(result) = import_asset_to_content(&path, args) {
                                        ui.memory_mut(|m| m.close_popup(add_menu_id));
                                        if let Err(e) = result {
                                            args.error_log.push(format!("[Import] {}", e));
                                        }
                                    }
                                }
                                ui.memory_mut(|m| m.close_popup(add_menu_id));
                            }
                        });
                    });
            });
        });
    ui.add_space(4.0);

    egui::Frame::new()
        .fill(Color32::from_rgb(12, 15, 20))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(32, 38, 48)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(6))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Type")
                        .small()
                        .strong()
                        .color(Color32::from_rgb(168, 176, 188)),
                );
                ui.add_space(14.0);
                ui.label(
                    RichText::new("Label")
                        .small()
                        .strong()
                        .color(Color32::from_rgb(168, 176, 188)),
                );
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                for &root in &roots {
                    render_tree(
                        ui,
                        root,
                        0,
                        &parent_to_children,
                        &entity_labels,
                        &search,
                        args,
                    );
                }

                // Show orphaned entities (entities not in any parent-child relationship).
                // These are entities with Renderable that have no Parent and are not in roots.
                let orphans: Vec<hecs::Entity> = args
                    .world
                    .query::<(hecs::Entity, &components::Renderable)>()
                    .iter()
                    .map(|(e, _)| e)
                    .filter(|e| !roots.contains(e))
                    .collect();

                for entity in orphans {
                    let label = entity_labels
                        .get(&entity)
                        .cloned()
                        .unwrap_or_else(|| format!("Entity {:?}", entity));
                    let selected = args.selected_renderable.map(|s| s == entity).unwrap_or(false);
                    render_entity_row(ui, entity, &label, 0, false, selected, args);
                }
            });
        });
}
