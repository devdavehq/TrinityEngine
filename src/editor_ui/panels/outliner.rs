use crate::components;
use crate::editor_ui::UiFrameArgs;

pub fn render_outliner_panel(ui: &mut egui::Ui, args: &mut UiFrameArgs<'_>) {
    let entities: Vec<hecs::Entity> = args
        .world
        .query::<(hecs::Entity, &components::Renderable)>()
        .iter()
        .map(|(e, _)| e)
        .collect();
    for e in entities {
        let selected = args.selected_renderable.map(|s| s == e).unwrap_or(false);
        if ui.selectable_label(selected, format!("Entity {:?}", e)).clicked() {
            *args.selected_renderable = Some(e);
        }
    }
}
