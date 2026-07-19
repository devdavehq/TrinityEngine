// src/editor_ui/panels/outliner.rs
// ── World Outliner Panel ─────────────────────────────────────────────────
// Tree view with parent-child hierarchy, groups, search, and drag-drop reparent.

use crate::components;
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

    // Group node creation buttons.
    egui::Frame::new()
        .fill(Color32::from_rgb(14, 17, 22))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(33, 39, 49)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button(RichText::new("+ Group").small().color(Color32::from_rgb(147, 158, 172)))
                    .clicked()
                {
                    // Spawn a new group node.
                    let group = args.world.spawn((
                        components::GroupNode::new("New Group"),
                        components::Children::new(),
                    ));
                    *args.selected_renderable = Some(group);
                }
                if ui
                    .button(RichText::new("+ Empty").small().color(Color32::from_rgb(147, 158, 172)))
                    .clicked()
                {
                    // Spawn a new empty entity.
                    let entity = args.world.spawn((
                        components::Position { x: 0.0, y: 0.0, z: 0.0 },
                        components::Renderable {
                            mesh: crate::assets::Handle::new(0),
                            color: [1.0; 3],
                            metallic: 0.0,
                            roughness: 0.5,
                            ao: 1.0,
                            scale: [1.0; 3],
                        },
                    ));
                    *args.selected_renderable = Some(entity);
                }
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
