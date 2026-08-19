// src/editor_ui/panels/navmesh.rs
// ──────────────────────────────────────────────────────────────────────────────
// NavMesh panel.
//
// The navmesh itself is derived data (baked from the terrain heightmap by
// `navmesh::NavMesh::from_terrain`) — there's nothing here to hand-author,
// just a rebuild trigger and read-only stats so you can tell whether it's
// stale after sculpting terrain.
// ──────────────────────────────────────────────────────────────────────────────

use crate::editor_ui::UiFrameArgs;

pub fn render_navmesh_panel(ui: &mut egui::Ui, args: &mut UiFrameArgs) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(16, 19, 25))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(37, 44, 55)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("NavMesh")
                    .strong()
                    .color(egui::Color32::from_rgb(229, 232, 238)),
            );
            ui.label(
                egui::RichText::new("Baked from the terrain heightmap — walkability for AI pathfinding.")
                    .small()
                    .color(egui::Color32::from_rgb(144, 154, 170)),
            );
            ui.separator();

            let tris = args.navmesh.triangle_count();
            let (ex, ez) = args.navmesh.extents();
            ui.label(format!("Triangles: {}", tris));
            ui.label(format!("Extents: {:.0} x {:.0} m", ex * 2.0, ez * 2.0));
            if tris == 0 {
                ui.colored_label(
                    egui::Color32::from_rgb(224, 168, 82),
                    "Empty — rebuild after the terrain has any walkable ground.",
                );
            }

            ui.separator();
            if ui.button("Rebuild NavMesh")
                .on_hover_text("Re-derives walkability from the current terrain heights. Do this after sculpting — the navmesh doesn't update itself.")
                .clicked()
            {
                *args.nav_rebuild_requested = true;
            }
            ui.label(
                egui::RichText::new("Rebuild runs on the next frame; AI agents re-path automatically once it's ready.")
                    .small()
                    .color(egui::Color32::from_rgb(121, 131, 145)),
            );
        });
}
