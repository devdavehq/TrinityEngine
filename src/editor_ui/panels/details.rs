use crate::components;
use crate::editor_ui::UiFrameArgs;

pub fn render_details_panel(ui: &mut egui::Ui, args: &mut UiFrameArgs<'_>) {
    if let Some(entity) = args.selected_renderable.as_ref().copied() {
        if let Ok(mut pos) = args.world.get::<&mut components::Position>(entity) {
            ui.label("Position");
            ui.add(egui::DragValue::new(&mut pos.x).prefix("X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut pos.y).prefix("Y ").speed(0.05));
            ui.add(egui::DragValue::new(&mut pos.z).prefix("Z ").speed(0.05));
        }
    } else {
        ui.label("Select an entity.");
    }
}
