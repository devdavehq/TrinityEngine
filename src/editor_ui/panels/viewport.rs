use crate::camera::Camera;
use crate::components;
use crate::editor_ui::{
    draw_transform_gizmo, pick_entity_in_viewport, project_to_screen, screen_to_plane_world, GizmoAxis,
    GizmoDragState, GizmoMode, GizmoSpace, UiFrameArgs,
};

// "light_probes" / "probe_selected_index" keys are shared with the Details
// panel so the hide toggle and selection survive across frames and tabs.
const KEY_PROBES_HIDDEN: &str = "probes_hidden";
const KEY_PROBE_SELECTED: &str = "probe_selected_index";

fn probes_hidden(ui: &egui::Ui) -> bool {
    ui.data(|d| d.get_temp::<bool>(KEY_PROBES_HIDDEN.into()))
        .unwrap_or(false)
}

fn overlay_chip(p: &egui::Painter, min: egui::Pos2, text: &str, width: f32) {
    let rect = egui::Rect::from_min_size(min, egui::vec2(width, 24.0));
    p.rect_filled(
        rect,
        6.0,
        egui::Color32::from_rgba_unmultiplied(12, 14, 18, 210),
    );
    p.rect_stroke(
        rect,
        6.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(62, 70, 82, 200)),
        egui::StrokeKind::Middle,
    );
    p.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(11.5),
        egui::Color32::from_rgb(223, 227, 234),
    );
}

pub fn render_viewport_panel(
    ui: &mut egui::Ui,
    args: &mut UiFrameArgs<'_>,
    scene_texture_id: Option<egui::TextureId>,
    gizmo_mode: &mut GizmoMode,
    gizmo_drag: &mut Option<GizmoDragState>,
    gizmo_space: GizmoSpace,
    gizmo_axis_lock: Option<GizmoAxis>,
    terrain_mode: bool,
    terrain_brush_mode: components::TerrainBrushMode,
    terrain_brush_radius: f32,
    snap_enabled: bool,
    snap_translate: f32,
    snap_rotate_deg: f32,
    snap_scale: f32,
) {
    let desired = ui.available_size();
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());
    let p = ui.painter();

    p.rect_filled(rect, 8.0, egui::Color32::from_rgb(7, 9, 13));
    if let Some(texture_id) = scene_texture_id {
        p.image(
            texture_id,
            rect.shrink(1.0),
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }
    p.rect_stroke(
        rect,
        8.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 44, 54)),
        egui::StrokeKind::Middle,
    );

    let title_bar = egui::Rect::from_min_size(rect.min + egui::vec2(10.0, 10.0), egui::vec2(270.0, 28.0));
    p.rect_filled(
        title_bar,
        7.0,
        egui::Color32::from_rgba_unmultiplied(10, 12, 16, 220),
    );
    p.rect_stroke(
        title_bar,
        7.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(54, 62, 74, 220)),
        egui::StrokeKind::Middle,
    );
    p.text(
        title_bar.left_center() + egui::vec2(12.0, 0.0),
        egui::Align2::LEFT_CENTER,
        "Perspective   Lit   Realtime",
        egui::FontId::proportional(12.0),
        egui::Color32::from_rgb(228, 232, 238),
    );

    let toolbar = egui::Rect::from_min_size(
        rect.min + egui::vec2(10.0, 46.0),
        egui::vec2(350.0, 26.0),
    );
    p.rect_filled(
        toolbar,
        7.0,
        egui::Color32::from_rgba_unmultiplied(10, 12, 16, 208),
    );
    p.rect_stroke(
        toolbar,
        7.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(54, 62, 74, 200)),
        egui::StrokeKind::Middle,
    );
    p.text(
        toolbar.left_center() + egui::vec2(10.0, 0.0),
        egui::Align2::LEFT_CENTER,
        "Show   Camera   View Mode   Exposure   Performance",
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(198, 205, 215),
    );

    let right = rect.right() - 10.0;
    let top = rect.top() + 10.0;
    let terrain_label = if terrain_mode {
        format!("T {} [{}]", terrain_brush_mode.label(), terrain_brush_radius as u32)
    } else {
        String::new()
    };
    let snap_label = format!(
        "Snap {}",
        if snap_enabled {
            format!("T {:.2}", snap_translate)
        } else {
            "Off".to_string()
        }
    );
    let rot_label = format!("Rot {:.1}", snap_rotate_deg);
    let scale_label = format!("Scale {:.2}", snap_scale);
    let cam_label = format!("Cam x{:.1}", args.camera_nav_speed);
    let mut chips: Vec<(&str, f32)> = Vec::new();
    if !terrain_label.is_empty() {
        chips.push((&terrain_label, 120.0));
    }
    chips.push(("W / E / R", 82.0));
    chips.push((&snap_label, 92.0));
    chips.push((&rot_label, 76.0));
    chips.push((&scale_label, 84.0));
    chips.push((&cam_label, 84.0));
    let mut cursor = right;
    for (label, width) in &chips {
        cursor -= width;
        overlay_chip(p, egui::pos2(cursor, top), label, *width);
        cursor -= 6.0;
    }

    let footer = egui::Rect::from_min_size(
        egui::pos2(rect.left() + 10.0, rect.bottom() - 36.0),
        egui::vec2(312.0, 26.0),
    );
    p.rect_filled(
        footer,
        7.0,
        egui::Color32::from_rgba_unmultiplied(10, 12, 16, 220),
    );
    p.rect_stroke(
        footer,
        7.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(54, 62, 74, 210)),
        egui::StrokeKind::Middle,
    );
    let selected = args
        .selected_renderable
        .map(|e| format!("{:?}", e))
        .unwrap_or_else(|| "None".to_string());
    p.text(
        footer.left_center() + egui::vec2(10.0, 0.0),
        egui::Align2::LEFT_CENTER,
        &format!("Selected {selected}   |   Preview {}", if *args.game_preview_mode { "On" } else { "Off" }),
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(205, 210, 218),
    );

    if response.drag_started() {
        if let Some(pointer) = response.interact_pointer_pos() {
            if !probes_hidden(ui) && !*args.game_preview_mode {
                if let Some(idx) = probe_hit_index(args, rect, pointer) {
                    ui.data_mut(|d| d.insert_temp(KEY_PROBE_SELECTED.into(), Some(idx)));
                    *args.selected_renderable = None;
                    // Record drag start so we move this probe (or its whole
                    // group) by the pointer's world-space delta each frame.
                    if idx < args.renderer.light_probes.probes.len() {
                        let start = args.renderer.light_probes.probes[idx].position;
                        ui.data_mut(|d| {
                            d.insert_temp("probe_drag_state".into(), Some((idx, pointer, start)))
                        });
                    }
                    return;
                }
            }
            if let Some(hit) = pick_entity_in_viewport(args.world, args.camera, rect, pointer) {
                *args.selected_renderable = Some(hit);
            }
        }
    } else if response.dragged() {
        // Move the probe (and its group) under the drag.
        if !probes_hidden(ui) && !*args.game_preview_mode {
            let drag_state = ui
                .data(|d| d.get_temp::<Option<(usize, egui::Pos2, glam::Vec3)>>("probe_drag_state".into()))
                .flatten();
            if let Some((idx, prev_pointer, start)) = drag_state {
                if let Some(pointer) = response.interact_pointer_pos() {
                    if idx < args.renderer.light_probes.probes.len() {
                        let probe_pos = args.renderer.light_probes.probes[idx].position;
                        let forward = args.camera.forward();
                        let plane_n = glam::Vec3::new(forward.x, forward.y, forward.z);
                        // Project both pointers onto the plane through the probe.
                        if let (Some(prev_w), Some(cur_w)) = (
                            screen_to_plane_world(args.camera, rect, prev_pointer, probe_pos, plane_n),
                            screen_to_plane_world(args.camera, rect, pointer, probe_pos, plane_n),
                        ) {
                            let delta = cur_w - prev_w;
                            let new_world = start + delta;
                            // Move the probe or whole group to absolute new pos.
                            let g = args.renderer.light_probes.probes[idx].group;
                            if g == 0 {
                                args.renderer.light_probes.probes[idx].position = new_world;
                            } else {
                                let shift = new_world - probe_pos;
                                args.renderer.light_probes.move_probe_or_group(idx, shift);
                            }
                        }
                    }
                    ui.data_mut(|d| {
                        d.insert_temp("probe_drag_state".into(), Some((idx, pointer, start)))
                    });
                }
            }
        }
    } else if response.clicked() {
        if let Some(pointer) = response.interact_pointer_pos() {
            if !probes_hidden(ui) && !*args.game_preview_mode {
                if let Some(idx) = probe_hit_index(args, rect, pointer) {
                    ui.data_mut(|d| d.insert_temp(KEY_PROBE_SELECTED.into(), Some(idx)));
                    // Selection belongs to a probe now, not an entity.
                    *args.selected_renderable = None;
                    return;
                }
            }
            if let Some(hit) = pick_entity_in_viewport(args.world, args.camera, rect, pointer) {
                *args.selected_renderable = Some(hit);
            }
        }
    }
    if response.drag_stopped() {
        ui.data_mut(|d| d.insert_temp("probe_drag_state".into(), Option::<(usize, egui::Pos2, glam::Vec3)>::None));
    }

    if !probes_hidden(ui) && !*args.game_preview_mode {
        draw_volume_gizmos(ui, args, rect);
        draw_probe_gizmos(ui, args, rect);
    }

    if let Some(entity) = args.selected_renderable.as_ref().copied() {
        draw_transform_gizmo(
            ui,
            args,
            entity,
            rect,
            response.interact_pointer_pos(),
            gizmo_mode,
            gizmo_drag,
            gizmo_space,
            gizmo_axis_lock,
            snap_enabled,
            snap_translate,
            snap_rotate_deg,
            snap_scale,
        );
    }
}

/// Returns the index of the probe whose screen-space icon is under `pointer`,
/// or None if the click misses every probe icon.
fn probe_hit_index(args: &UiFrameArgs<'_>, rect: egui::Rect, pointer: egui::Pos2) -> Option<usize> {
    let camera: &dyn Camera = args.camera;
    let mut best: Option<(usize, f32)> = None;
    for (i, probe) in args.renderer.light_probes.probes.iter().enumerate() {
        if let Some(screen) = project_to_screen(camera, rect, probe.position) {
            let dx = screen.x - pointer.x;
            let dy = screen.y - pointer.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist <= 12.0 {
                if best.map_or(true, |(_, bd)| dist < bd) {
                    best = Some((i, dist));
                }
            }
        }
    }
    best.map(|b| b.0)
}

/// Draws small light-bulb markers for every light probe in the scene, with a
/// highlight ring around the currently selected one. Hidden via the
/// "Hide probes in viewport" toggle in the Details panel.
fn draw_probe_gizmos(ui: &egui::Ui, args: &UiFrameArgs<'_>, rect: egui::Rect) {
    let camera: &dyn Camera = args.camera;
    let selected = ui
        .data(|d| d.get_temp::<Option<usize>>(KEY_PROBE_SELECTED.into()))
        .flatten();
    let p = ui.painter();

    for (i, probe) in args.renderer.light_probes.probes.iter().enumerate() {
        let Some(screen) = project_to_screen(camera, rect, probe.position) else {
            continue;
        };
        let is_selected = selected == Some(i);
        // A small glowing bulb: outer ring, filled core, bright centre dot.
        p.circle_stroke(
            screen,
            if is_selected { 11.0 } else { 8.0 },
            egui::Stroke::new(
                if is_selected { 2.2 } else { 1.2 },
                if is_selected {
                    egui::Color32::from_rgb(255, 214, 102)
                } else {
                    egui::Color32::from_rgba_unmultiplied(110, 190, 255, 220)
                },
            ),
        );
        p.circle_filled(
            screen,
            if is_selected { 6.5 } else { 5.0 },
            egui::Color32::from_rgba_unmultiplied(60, 120, 200, 210),
        );
        p.circle_filled(screen, 1.8, egui::Color32::from_rgb(235, 244, 255));
        if is_selected {
            // A soft halo so the selected probe stands out.
            p.circle_stroke(
                screen,
                16.0,
                egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_unmultiplied(255, 214, 102, 120),
                ),
            );
        }
    }
}

/// Draws the probe-volume boxes as translucent wireframes so you can see
/// exactly which space each box fills with probes.
fn draw_volume_gizmos(ui: &egui::Ui, args: &UiFrameArgs<'_>, rect: egui::Rect) {
    let camera: &dyn Camera = args.camera;
    let p = ui.painter();
    let vols = args.renderer.light_probes.volumes.len();
    for (vi, vol) in args.renderer.light_probes.volumes.iter().enumerate() {
        let min = vol.center - vol.size * 0.5;
        let max = vol.center + vol.size * 0.5;
        // 8 corners.
        let corners = [
            glam::Vec3::new(min.x, min.y, min.z),
            glam::Vec3::new(max.x, min.y, min.z),
            glam::Vec3::new(max.x, min.y, max.z),
            glam::Vec3::new(min.x, min.y, max.z),
            glam::Vec3::new(min.x, max.y, min.z),
            glam::Vec3::new(max.x, max.y, min.z),
            glam::Vec3::new(max.x, max.y, max.z),
            glam::Vec3::new(min.x, max.y, max.z),
        ];
        let edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];
        let color = if vi == vols - 1 {
            egui::Color32::from_rgba_unmultiplied(140, 220, 255, 170)
        } else {
            egui::Color32::from_rgba_unmultiplied(110, 180, 230, 130)
        };
        let stroke = egui::Stroke::new(1.4, color);
        for (a, b) in edges {
            let Some(sa) = project_to_screen(camera, rect, corners[a]) else { continue };
            let Some(sb) = project_to_screen(camera, rect, corners[b]) else { continue };
            p.line_segment([sa, sb], stroke);
        }
        // Small label with the volume index + probe count.
        let Some(center) = project_to_screen(camera, rect, vol.center) else { continue };
        p.text(
            center + egui::vec2(6.0, -6.0),
            egui::Align2::LEFT_BOTTOM,
            format!("V{vi}"),
            egui::FontId::proportional(11.0),
            egui::Color32::from_rgb(190, 235, 255),
        );
    }
}
