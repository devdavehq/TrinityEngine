use crate::components;
use crate::editor_ui::{
    draw_transform_gizmo, pick_entity_in_viewport, GizmoAxis, GizmoDragState, GizmoMode, GizmoSpace,
    UiFrameArgs,
};

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
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
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

    if response.clicked() {
        if let Some(pointer) = response.interact_pointer_pos() {
            if let Some(hit) = pick_entity_in_viewport(args.world, args.camera, rect, pointer) {
                *args.selected_renderable = Some(hit);
            }
        }
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
