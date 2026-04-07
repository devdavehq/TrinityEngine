use crate::editor_ui::{draw_transform_gizmo, pick_entity_in_viewport, GizmoDragState, GizmoMode, GizmoSpace, UiFrameArgs};

pub fn render_viewport_panel(
    ui: &mut egui::Ui,
    args: &mut UiFrameArgs<'_>,
    scene_texture_id: Option<egui::TextureId>,
    gizmo_mode: &mut GizmoMode,
    gizmo_drag: &mut Option<GizmoDragState>,
    gizmo_space: GizmoSpace,
    snap_enabled: bool,
    snap_translate: f32,
    snap_rotate_deg: f32,
    snap_scale: f32,
) {
    let desired = ui.available_size();
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
    if let Some(texture_id) = scene_texture_id {
        ui.painter().image(
            texture_id,
            rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }
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
            snap_enabled,
            snap_translate,
            snap_rotate_deg,
            snap_scale,
        );
    }
}
