#![allow(deprecated)]

use std::fs;

use crate::editor;
use crate::materials::MaterialLibrary;
use crate::profiler::FrameProfiler;
use crate::renderer::Renderer;
use crate::settings::{EngineSettings, RenderPreset};
use crate::terrain::{remove_nearby_foliage, spawn_foliage_ring, TerrainGrid};
use crate::navigation::NavGrid;
use crate::scripting::ScriptEngine;
use crate::camera::Camera;
use crate::{assets, components};
use egui::{Color32, RichText};
use hecs::World;
use winit::window::Window;

#[derive(Clone, Copy, PartialEq, Eq)]
enum UiWidgetKind {
    HealthBar,
    Counter,
    Label,
}

#[derive(Clone)]
struct UiWidgetSpec {
    id: String,
    kind: UiWidgetKind,
    x: f32,
    y: f32,
    w: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GizmoMode {
    Move,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GizmoAxis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy)]
struct GizmoDragState {
    axis: GizmoAxis,
    pointer_start: egui::Pos2,
    pos_start: [f32; 3],
    rot_start: [f32; 3],
    scale_start: [f32; 3],
}

pub struct EditorUi {
    pub visible: bool,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    pending: Option<PreparedUi>,
    lua_selected: Option<String>,
    lua_buffer: String,
    lua_dirty: bool,
    texture_selected: Option<String>,
    texture_dragging: Option<String>,
    mesh_selected: Option<String>,
    content_new_folder: String,
    content_new_file: String,
    toasts: Vec<(String, f32)>,
    undock_hierarchy: bool,
    undock_inspector: bool,
    undock_asset_browser: bool,
    undock_viewport: bool,
    widget_specs: Vec<UiWidgetSpec>,
    widget_new_id: String,
    widget_new_kind: UiWidgetKind,
    show_project_launcher: bool,
    gizmo_mode: GizmoMode,
    gizmo_drag: Option<GizmoDragState>,
}

struct PreparedUi {
    paint_jobs: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    screen_desc: egui_wgpu::ScreenDescriptor,
}

pub struct UiFrameArgs<'a> {
    pub world: &'a mut World,
    pub settings: &'a mut EngineSettings,
    pub renderer: &'a mut Renderer,
    pub camera: &'a dyn Camera,
    pub profiler: &'a FrameProfiler,
    pub mesh_cache: &'a mut std::collections::HashMap<String, assets::Handle<assets::Mesh>>,
    pub meshes: &'a mut assets::AssetStore<assets::Mesh>,
    pub materials: &'a mut MaterialLibrary,
    pub selected_renderable: &'a mut Option<hecs::Entity>,
    pub terrain: &'a mut TerrainGrid,
    pub terrain_cursor_x: usize,
    pub terrain_cursor_z: usize,
    pub app_time_seconds: f32,
    pub sim_paused: &'a mut bool,
    pub sim_step_once: &'a mut bool,
    pub game_preview_mode: &'a mut bool,
    pub mouse_look_latched: &'a mut bool,
    pub error_log: &'a mut Vec<String>,
    pub nav_grid: &'a mut NavGrid,
    pub nav_rebuild_requested: &'a mut bool,
    pub scripts: &'a mut ScriptEngine,
    pub scripts_dir: &'a str,
    pub script_hot_reload_enabled: &'a mut bool,
    pub preferred_script_editor: &'a mut String,
    pub asset_hot_reload_enabled: &'a mut bool,
}

impl EditorUi {
    pub fn new(window: &Window, renderer: &Renderer) -> Self {
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &renderer.device,
            renderer.surface_format(),
            egui_wgpu::RendererOptions::default(),
        );
        Self {
            visible: true,
            egui_ctx,
            egui_state,
            egui_renderer,
            pending: None,
            lua_selected: None,
            lua_buffer: String::new(),
            lua_dirty: false,
            texture_selected: None,
            texture_dragging: None,
            mesh_selected: None,
            content_new_folder: "NewFolder".to_string(),
            content_new_file: "new_asset.txt".to_string(),
            toasts: Vec::new(),
            undock_hierarchy: false,
            undock_inspector: false,
            undock_asset_browser: false,
            undock_viewport: false,
            widget_specs: vec![
                UiWidgetSpec { id: "player_health".to_string(), kind: UiWidgetKind::HealthBar, x: 24.0, y: 24.0, w: 280.0 },
                UiWidgetSpec { id: "coins".to_string(), kind: UiWidgetKind::Counter, x: 24.0, y: 56.0, w: 180.0 },
            ],
            widget_new_id: "new_widget".to_string(),
            widget_new_kind: UiWidgetKind::Label,
            show_project_launcher: true,
            gizmo_mode: GizmoMode::Move,
            gizmo_drag: None,
        }
    }

    pub fn push_toast(&mut self, message: String, now_seconds: f32) {
        self.toasts.push((message, now_seconds + 3.0));
    }

    pub fn on_window_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        let response = self.egui_state.on_window_event(window, event);
        response.consumed
    }

    pub fn begin_and_build(&mut self, window: &Window, args: &mut UiFrameArgs<'_>) {
        if !self.visible {
            self.pending = None;
            return;
        }

        self.apply_theme();
        let raw_input = self.egui_state.take_egui_input(window);
        let output = self.egui_ctx.run_ui(raw_input, |ctx| {
            build_ui(
                ctx,
                args,
                &mut self.lua_selected,
                &mut self.lua_buffer,
                &mut self.lua_dirty,
                &mut self.texture_selected,
                &mut self.texture_dragging,
                &mut self.mesh_selected,
                &mut self.content_new_folder,
                &mut self.content_new_file,
                &mut self.toasts,
                &mut self.undock_hierarchy,
                &mut self.undock_inspector,
                &mut self.undock_asset_browser,
                &mut self.undock_viewport,
                &mut self.widget_specs,
                &mut self.widget_new_id,
                &mut self.widget_new_kind,
                &mut self.show_project_launcher,
                &mut self.gizmo_mode,
                &mut self.gizmo_drag,
            );
        });
        self.egui_state
            .handle_platform_output(window, output.platform_output);

        let paint_jobs = self
            .egui_ctx
            .tessellate(output.shapes, output.pixels_per_point);
        let size = window.inner_size();
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: output.pixels_per_point,
        };
        self.pending = Some(PreparedUi {
            paint_jobs,
            textures_delta: output.textures_delta,
            screen_desc,
        });
    }

    pub fn paint_on(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let Some(ui) = self.pending.take() else {
            return;
        };

        for (id, delta) in &ui.textures_delta.set {
            self.egui_renderer.update_texture(device, queue, *id, delta);
        }

        self.egui_renderer.update_buffers(
            device,
            queue,
            encoder,
            &ui.paint_jobs,
            &ui.screen_desc,
        );

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Editor UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            let mut pass_static = pass.forget_lifetime();
            self.egui_renderer
                .render(&mut pass_static, &ui.paint_jobs, &ui.screen_desc);
        }

        for id in &ui.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }

    fn apply_theme(&self) {
        let mut style = (*self.egui_ctx.global_style()).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.window_fill = Color32::from_rgb(18, 18, 22);
        style.visuals.panel_fill = Color32::from_rgb(20, 20, 24);
        style.visuals.faint_bg_color = Color32::from_rgb(26, 26, 32);
        style.visuals.extreme_bg_color = Color32::from_rgb(10, 10, 14);
        style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(24, 24, 28);
        style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(32, 32, 38);
        style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(64, 64, 70);
        style.visuals.widgets.active.bg_fill = Color32::from_rgb(94, 94, 100);
        style.visuals.selection.bg_fill = Color32::from_rgb(170, 170, 176);
        self.egui_ctx.set_global_style(style);
    }
}

fn build_ui(
    ctx: &egui::Context,
    args: &mut UiFrameArgs<'_>,
    lua_selected: &mut Option<String>,
    lua_buffer: &mut String,
    lua_dirty: &mut bool,
    texture_selected: &mut Option<String>,
    texture_dragging: &mut Option<String>,
    mesh_selected: &mut Option<String>,
    content_new_folder: &mut String,
    content_new_file: &mut String,
    toasts: &mut Vec<(String, f32)>,
    undock_hierarchy: &mut bool,
    undock_inspector: &mut bool,
    undock_asset_browser: &mut bool,
    undock_viewport: &mut bool,
    widget_specs: &mut Vec<UiWidgetSpec>,
    widget_new_id: &mut String,
    widget_new_kind: &mut UiWidgetKind,
    show_project_launcher: &mut bool,
    gizmo_mode: &mut GizmoMode,
    gizmo_drag: &mut Option<GizmoDragState>,
) {
    if args.app_time_seconds < 2.2 {
        egui::Area::new("startup_splash".into())
            .fixed_pos([0.0, 0.0])
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                ui.painter().rect_filled(rect, 0.0, Color32::from_rgb(4, 6, 10));
                let center = rect.center();
                draw_triangle_logo(ui.painter(), center + egui::vec2(0.0, -40.0), 120.0);
                ui.painter().text(
                    center + egui::vec2(0.0, 90.0),
                    egui::Align2::CENTER_CENTER,
                    "TRINITY",
                    egui::FontId::proportional(42.0),
                    Color32::from_rgb(220, 224, 230),
                );
                ui.painter().text(
                    center + egui::vec2(0.0, 130.0),
                    egui::Align2::CENTER_CENTER,
                    "STUDIOS",
                    egui::FontId::proportional(20.0),
                    Color32::from_rgb(170, 176, 186),
                );
            });
        return;
    }
    ctx.input(|i| {
        if i.key_pressed(egui::Key::W) {
            *gizmo_mode = GizmoMode::Move;
        } else if i.key_pressed(egui::Key::E) {
            *gizmo_mode = GizmoMode::Rotate;
        } else if i.key_pressed(egui::Key::R) {
            *gizmo_mode = GizmoMode::Scale;
        }
    });

    egui::Panel::top("top_toolbar").show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Triengine Editor").strong().color(Color32::from_rgb(212, 212, 218)));
            ui.separator();
            ui.label("Preset");
            preset_combo(ui, &mut args.settings.render.preset);
            ui.separator();
            if ui.button(if *args.sim_paused { "Play" } else { "Pause" }).clicked() {
                *args.sim_paused = !*args.sim_paused;
            }
            if ui.button("Step").clicked() {
                *args.sim_step_once = true;
                *args.sim_paused = true;
            }
            ui.separator();
            ui.toggle_value(args.mouse_look_latched, "Mouse Look Lock");
            if ui.button("Apply Preset").clicked() {
                // Re-load to apply preset logic path while preserving file defaults behavior.
                let mut copy = args.settings.clone();
                copy.render.preset = args.settings.render.preset;
                *args.settings = copy;
            }
            ui.separator();
            ui.toggle_value(&mut args.renderer.features.bloom_enabled, "Bloom");
            ui.toggle_value(&mut args.renderer.features.ssao_enabled, "SSAO");
            ui.toggle_value(&mut args.renderer.features.volumetric_fog_enabled, "VFog");
            ui.toggle_value(&mut args.renderer.features.voxel_gi_enabled, "Voxel");
            ui.separator();
            ui.toggle_value(undock_hierarchy, "Undock Hierarchy");
            ui.toggle_value(undock_inspector, "Undock Details");
            ui.toggle_value(undock_asset_browser, "Undock Assets");
            ui.toggle_value(undock_viewport, "Undock Viewport");
            ui.separator();
            ui.label("Gizmo");
            ui.selectable_value(gizmo_mode, GizmoMode::Move, "W Move");
            ui.selectable_value(gizmo_mode, GizmoMode::Rotate, "E Rotate");
            ui.selectable_value(gizmo_mode, GizmoMode::Scale, "R Scale");
            if ui.button("Project Launcher").clicked() {
                *show_project_launcher = true;
            }
            if ui.button("Dock All").clicked() {
                *undock_hierarchy = false;
                *undock_inspector = false;
                *undock_asset_browser = false;
                *undock_viewport = false;
            }
        });
    });

    let mut draw_hierarchy_panel = |ui: &mut egui::Ui| {
        ui.heading("Hierarchy");
        ui.label("Professional editor icons");
        ui.separator();
        icon_row(ui, IconKind::Camera, "Main Camera");
        icon_row(ui, IconKind::Light, "Directional Light");
        icon_row(ui, IconKind::Sky, "Skylight / Sky Background");
        ui.separator();
        let entities: Vec<hecs::Entity> = args
            .world
            .query::<(hecs::Entity, &components::Renderable)>()
            .iter()
            .map(|(e, _)| e)
            .collect();
        for entity in entities {
            let selected = args.selected_renderable.map(|e| e == entity).unwrap_or(false);
            if ui.selectable_label(selected, format!("mesh entity {:?}", entity)).clicked() {
                *args.selected_renderable = Some(entity);
                if let Some(tex) = texture_dragging.clone() {
                    upsert_albedo_texture(args.world, entity, tex);
                    *texture_dragging = None;
                }
            }
        }
        let light_entities: Vec<hecs::Entity> = args
            .world
            .query::<(hecs::Entity, &components::PointLight)>()
            .iter()
            .map(|(e, _)| e)
            .collect();
        for entity in light_entities {
            let selected = args.selected_renderable.map(|e| e == entity).unwrap_or(false);
            if ui.selectable_label(selected, format!("point light {:?}", entity)).clicked() {
                *args.selected_renderable = Some(entity);
            }
        }
        let player_starts: Vec<hecs::Entity> = args
            .world
            .query::<(hecs::Entity, &components::PlayerStart)>()
            .iter()
            .map(|(e, _)| e)
            .collect();
        for entity in player_starts {
            let selected = args.selected_renderable.map(|e| e == entity).unwrap_or(false);
            if ui.selectable_label(selected, format!("player start {:?}", entity)).clicked() {
                *args.selected_renderable = Some(entity);
            }
        }
    };
    if !*undock_hierarchy {
        egui::Panel::left("hierarchy_panel")
            .resizable(true)
            .default_size(260.0)
            .show(ctx, |ui| draw_hierarchy_panel(ui));
    } else {
        egui::Window::new("Hierarchy")
            .default_pos([20.0, 70.0])
            .default_size([320.0, 520.0])
            .resizable(true)
            .show(ctx, |ui| draw_hierarchy_panel(ui));
    }

    let mut draw_inspector_panel = |ui: &mut egui::Ui| {
        ui.heading("Inspector");
        ui.label("Sky background is auto-present when engine starts.");
        ui.label("Skylight (IBL) is enabled by default in presets.");
        ui.collapsing("Lighting", |ui| {
            ui.checkbox(&mut args.renderer.features.ibl_enabled, "Sky Light (IBL)");
            ui.checkbox(&mut args.renderer.features.shadows_enabled, "Directional Shadows");
            ui.checkbox(&mut args.renderer.features.pcss_enabled, "Soft shadows (PCSS)");
            ui.add(egui::Slider::new(&mut args.renderer.features.bloom_strength, 0.0..=2.0).text("Bloom"));
            ui.add(egui::Slider::new(&mut args.renderer.features.ssao_strength, 0.0..=1.0).text("SSAO"));
            ui.add(egui::Slider::new(&mut args.renderer.features.fog_density, 0.0..=0.20).text("Fog density"));
            ui.separator();
            ui.label("Sun Direction (real-time day cycle)");
            ui.add(egui::Slider::new(&mut args.renderer.features.sun_azimuth_deg, 0.0..=360.0).text("Sun azimuth"));
            ui.add(egui::Slider::new(&mut args.renderer.features.sun_elevation_deg, -5.0..=89.0).text("Sun elevation"));
            ui.add(egui::Slider::new(&mut args.renderer.features.sun_intensity, 0.1..=2.0).text("Sun intensity"));
            ui.horizontal(|ui| {
                if ui.button("Morning").clicked() {
                    args.renderer.features.sun_azimuth_deg = 55.0;
                    args.renderer.features.sun_elevation_deg = 22.0;
                    args.renderer.features.sun_intensity = 0.85;
                }
                if ui.button("Noon").clicked() {
                    args.renderer.features.sun_azimuth_deg = 130.0;
                    args.renderer.features.sun_elevation_deg = 72.0;
                    args.renderer.features.sun_intensity = 1.2;
                }
                if ui.button("Evening").clicked() {
                    args.renderer.features.sun_azimuth_deg = 300.0;
                    args.renderer.features.sun_elevation_deg = 15.0;
                    args.renderer.features.sun_intensity = 0.8;
                }
                if ui.button("Night").clicked() {
                    args.renderer.features.sun_elevation_deg = -4.0;
                    args.renderer.features.sun_intensity = 0.2;
                }
            });
            ui.separator();
            ui.label("Movable Point Light");
            if let Some(entity) = args.selected_renderable.as_ref().copied() {
                if let Ok(mut p) = args.world.get::<&mut components::PointLight>(entity) {
                    ui.add(egui::Slider::new(&mut p.color[0], 0.0..=2.0).text("R"));
                    ui.add(egui::Slider::new(&mut p.color[1], 0.0..=2.0).text("G"));
                    ui.add(egui::Slider::new(&mut p.color[2], 0.0..=2.0).text("B"));
                    ui.add(egui::Slider::new(&mut p.intensity, 0.0..=4.0).text("Intensity"));
                    ui.add(egui::Slider::new(&mut p.range, 0.5..=60.0).text("Range"));
                } else if ui.button("Add Point Light To Selected").clicked() {
                    let _ = args.world.insert(
                        entity,
                        (components::PointLight {
                            color: [1.0, 0.95, 0.85],
                            intensity: 1.5,
                            range: 12.0,
                        },),
                    );
                }
            }
            if ui.button("Create New Point Light").clicked() {
                let e = args.world.spawn((
                    components::Position { x: 0.0, y: 2.5, z: 0.0 },
                    components::PointLight {
                        color: [1.0, 0.92, 0.82],
                        intensity: 1.8,
                        range: 14.0,
                    },
                ));
                *args.selected_renderable = Some(e);
            }
        });
        ui.collapsing("Gameplay Spawn", |ui| {
            if let Some(entity) = args.selected_renderable.as_ref().copied() {
                if ui.button("Set PlayerStart From Selected").clicked() {
                    set_player_start_from_selected(args, entity);
                }
            }
            if ui.button("Create PlayerStart At Origin").clicked() {
                clear_player_starts(args.world);
                let _ = args.world.spawn((
                    components::Position { x: 0.0, y: 0.5, z: 0.0 },
                    components::PlayerStart,
                ));
            }
        });
        ui.collapsing("Terrain Auto Material", |ui| {
            ui.label("Grass on flats, dirt transitions, rock on steep/high areas.");
            ui.add(
                egui::Slider::new(&mut args.terrain.material.slope_rock_start, 0.1..=1.6)
                    .text("Rock from slope"),
            );
            ui.add(
                egui::Slider::new(&mut args.terrain.material.height_rock_start, 0.0..=6.0)
                    .text("Rock from height"),
            );
            let world_x = args.terrain_cursor_x as f32 * args.terrain.cell_size - 32.0;
            let world_z = args.terrain_cursor_z as f32 * args.terrain.cell_size - 32.0;
            let preview = args.terrain.auto_surface_color_world(world_x, world_z);
            let col = Color32::from_rgb(
                (preview[0] * 255.0) as u8,
                (preview[1] * 255.0) as u8,
                (preview[2] * 255.0) as u8,
            );
            ui.colored_label(col, "Preview at cursor");
            ui.label("When terrain is raised, slope/height blending updates automatically.");
        });
        ui.collapsing("Navigation Foundation", |ui| {
            ui.label("Foundation = base walkable grid + path query, before full navmesh bake.");
            ui.label(format!(
                "Walkable cells: {} / {}",
                args.nav_grid.walkable_count(),
                args.nav_grid.width * args.nav_grid.depth
            ));
            if ui.button("Rebuild Nav Grid").clicked() {
                *args.nav_rebuild_requested = true;
            }
            let start = (args.terrain_cursor_x.min(args.nav_grid.width.saturating_sub(1)), args.terrain_cursor_z.min(args.nav_grid.depth.saturating_sub(1)));
            let goal = (
                (start.0 + 8).min(args.nav_grid.width.saturating_sub(1)),
                (start.1 + 6).min(args.nav_grid.depth.saturating_sub(1)),
            );
            let path_len = args
                .nav_grid
                .find_path(start, goal)
                .map(|p| p.len())
                .unwrap_or(0);
            let smooth_len = args
                .nav_grid
                .find_path(start, goal)
                .map(|p| args.nav_grid.smooth_path(&p).len())
                .unwrap_or(0);
            ui.label(format!("Test path length from cursor: {}", path_len));
            ui.label(format!("Smoothed path length: {}", smooth_len));
            ui.label(format!(
                "Contours: {} | Regions: {}",
                args.nav_grid.contour_edges.len(),
                args.nav_grid.region_count
            ));
        });
        ui.collapsing("Material Tools", |ui| {
            if let Some(entity) = args.selected_renderable.as_ref().copied() {
                if ui.button("Apply matte_black").clicked() {
                    if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
                        let _ = args.materials.apply_instance("matte_black", &mut rend);
                    }
                }
                if ui.button("Apply silver_brushed").clicked() {
                    if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
                        let _ = args.materials.apply_instance("silver_brushed", &mut rend);
                    }
                }
                if ui.button("Apply foliage_leaf").clicked() {
                    if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
                        let _ = args.materials.apply_instance("foliage_leaf", &mut rend);
                    }
                }
                ui.separator();
                ui.label("Texture slots");
                let (albedo_label, normal_label, mr_label) =
                    if let Ok(mt) = args.world.get::<&components::MaterialTexture>(entity) {
                        (
                            if mt.path.is_empty() { "<default>".to_string() } else { mt.path.clone() },
                            if mt.normal_path.is_empty() { "<default flat>".to_string() } else { mt.normal_path.clone() },
                            if mt.metallic_roughness_path.is_empty() {
                                "<default>".to_string()
                            } else {
                                mt.metallic_roughness_path.clone()
                            },
                        )
                    } else {
                        (
                            "<default>".to_string(),
                            "<default flat>".to_string(),
                            "<default>".to_string(),
                        )
                    };
                ui.monospace(format!(
                    "Albedo: {}",
                    albedo_label
                ));
                ui.monospace(format!(
                    "Normal: {}",
                    normal_label
                ));
                ui.monospace(format!(
                    "Metal/Rough: {}",
                    mr_label
                ));
                if let Some(tex) = texture_selected.clone() {
                    ui.horizontal(|ui| {
                        if ui.button("Set Albedo from Selected Texture").clicked() {
                            upsert_albedo_texture(args.world, entity, tex.clone());
                        }
                        if ui.button("Set Normal from Selected Texture").clicked() {
                            upsert_normal_texture(args.world, entity, tex.clone());
                        }
                        if ui.button("Set Metallic+Roughness from Selected Texture").clicked() {
                            upsert_mr_texture(args.world, entity, tex.clone());
                        }
                    });
                } else {
                    ui.label("Select a texture in Asset Browser first.");
                }
                if ui.button("Clear All Texture Slots").clicked() {
                    let _ = args.world.remove_one::<components::MaterialTexture>(entity);
                }
            } else {
                ui.label("Select an entity in Hierarchy first.");
            }
        });
        ui.collapsing("Transform (Move / Rotate / Scale)", |ui| {
            if let Some(entity) = args.selected_renderable.as_ref().copied() {
                ui.horizontal(|ui| {
                    ui.label("Gizmo Modes:");
                    let _ = ui.button("Move");
                    let _ = ui.button("Rotate");
                    let _ = ui.button("Scale");
                });
                if let Ok(mut pos) = args.world.get::<&mut components::Position>(entity) {
                    ui.label("Position");
                    ui.add(egui::DragValue::new(&mut pos.x).speed(0.05).prefix("X "));
                    ui.add(egui::DragValue::new(&mut pos.y).speed(0.05).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut pos.z).speed(0.05).prefix("Z "));
                }
                if let Ok(mut rot) = args.world.get::<&mut components::Rotation>(entity) {
                    ui.label("Rotation (deg)");
                    let mut pitch = rot.pitch.to_degrees();
                    let mut yaw = rot.yaw.to_degrees();
                    let mut roll = rot.roll.to_degrees();
                    ui.add(egui::DragValue::new(&mut pitch).speed(0.5).prefix("P "));
                    ui.add(egui::DragValue::new(&mut yaw).speed(0.5).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut roll).speed(0.5).prefix("R "));
                    rot.pitch = pitch.to_radians();
                    rot.yaw = yaw.to_radians();
                    rot.roll = roll.to_radians();
                } else if ui.button("Add Rotation Component").clicked() {
                    let _ = args.world.insert(
                        entity,
                        (components::Rotation {
                            pitch: 0.0,
                            yaw: 0.0,
                            roll: 0.0,
                        },),
                    );
                }
                if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
                    ui.label("Scale");
                    ui.add(egui::DragValue::new(&mut rend.scale[0]).speed(0.05).prefix("X "));
                    ui.add(egui::DragValue::new(&mut rend.scale[1]).speed(0.05).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut rend.scale[2]).speed(0.05).prefix("Z "));
                }
            } else {
                ui.label("Select a mesh entity to edit transform.");
            }
        });
    };
    if !*undock_inspector {
        egui::Panel::right("inspector_panel")
            .resizable(true)
            .default_size(340.0)
            .show(ctx, |ui| draw_inspector_panel(ui));
    } else {
        egui::Window::new("Details")
            .default_pos([940.0, 70.0])
            .default_size([380.0, 620.0])
            .resizable(true)
            .show(ctx, |ui| draw_inspector_panel(ui));
    }

    let mut draw_asset_browser_panel = |ui: &mut egui::Ui| {
            ui.heading("Asset Browser");
            ui.label("Created files/folders go into Content/ in this project.");
            ui.horizontal(|ui| {
                ui.label("New folder:");
                ui.text_edit_singleline(content_new_folder);
                if ui.button("Create Folder").clicked() {
                    let folder = content_new_folder.trim();
                    if !folder.is_empty() {
                        let p = format!("Content/{}", folder);
                        if let Err(err) = fs::create_dir_all(&p) {
                            args.error_log.push(format!("[Content] Create folder failed {}: {}", p, err));
                        }
                    }
                }
                ui.label("New file:");
                ui.text_edit_singleline(content_new_file);
                if ui.button("Create File").clicked() {
                    let file = content_new_file.trim();
                    if !file.is_empty() {
                        let p = format!("Content/{}", file);
                        if let Some(parent) = std::path::Path::new(&p).parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        if let Err(err) = fs::write(&p, b"") {
                            args.error_log.push(format!("[Content] Create file failed {}: {}", p, err));
                        }
                    }
                }
            });
            ui.label("Thumbnail cards with dark preview backgrounds");
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                thumbnail_card(ui, "MAT", "matte_black", Some(Color32::from_rgb(18, 18, 18)));
                thumbnail_card(ui, "MAT", "silver_brushed", Some(Color32::from_rgb(180, 180, 190)));
                thumbnail_card(ui, "MAT", "foliage_leaf", Some(Color32::from_rgb(48, 120, 58)));
                if let Ok(entries) = fs::read_dir(args.scripts_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().map(|e| e == "lua").unwrap_or(false) {
                            thumbnail_card(
                                ui,
                                "LUA",
                                &path.file_name().unwrap_or_default().to_string_lossy(),
                                Some(Color32::from_rgb(36, 46, 72)),
                            );
                        }
                    }
                }
                if let Ok(entries) = fs::read_dir("Content/Textures") {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let p = path.to_string_lossy().to_string();
                        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
                        if ["png", "jpg", "jpeg"].contains(&ext.as_str()) {
                            let selected = texture_selected.as_ref().map(|s| s == &p).unwrap_or(false);
                            if ui.selectable_label(selected, format!("TEX {}", path.file_name().unwrap_or_default().to_string_lossy())).clicked() {
                                *texture_selected = Some(p);
                            }
                        }
                    }
                }
                if let Ok(entries) = fs::read_dir("Content/Meshes") {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let p = path.to_string_lossy().to_string();
                        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
                        if ext == "obj" {
                            let selected = mesh_selected.as_ref().map(|s| s == &p).unwrap_or(false);
                            if ui
                                .selectable_label(
                                    selected,
                                    format!("MESH {}", path.file_name().unwrap_or_default().to_string_lossy()),
                                )
                                .clicked()
                            {
                                *mesh_selected = Some(p);
                            }
                        }
                    }
                }
                if let Some(tex) = texture_selected.clone() {
                    if ui.button("Drag Texture To Mesh (then click mesh in Hierarchy)").clicked() {
                        *texture_dragging = Some(tex);
                    }
                }
                if let Some(active) = texture_dragging.as_ref() {
                    ui.colored_label(Color32::from_rgb(150, 200, 255), format!("Dragging: {}", active));
                }
            });

            ui.separator();
            ui.horizontal(|ui| {
                if let Some(mesh_path) = mesh_selected.clone() {
                    if ui.button("Add Selected Mesh To Scene").clicked() {
                        let key = mesh_path.replace('\\', "/");
                        let handle = if let Some(h) = args.mesh_cache.get(&key).copied() {
                            h
                        } else {
                            match assets::Mesh::load(&key) {
                                Ok(mesh) => {
                                    let h = args.meshes.add(mesh);
                                    args.mesh_cache.insert(key.clone(), h);
                                    h
                                }
                                Err(err) => {
                                    args.error_log.push(format!("[Content] Mesh load failed {}: {}", key, err));
                                    return;
                                }
                            }
                        };
                        let c = terrain_color_at_cursor(args);
                        let _ = spawn_mesh_entity(args, handle, [1.0, 1.0, 1.0], c, false);
                    }
                }
                if ui.button("Add foliage ring").clicked() {
                    if let Some(handle) = args.mesh_cache.get("meshes/cube.obj").copied() {
                        spawn_foliage_ring(
                            args.world,
                            handle,
                            args.terrain_cursor_x as f32 * args.terrain.cell_size - 32.0,
                            args.terrain_cursor_z as f32 * args.terrain.cell_size - 32.0,
                            4.0,
                            24,
                            true,
                        );
                    }
                }
                if ui.button("Remove nearby foliage").clicked() {
                    let _ = remove_nearby_foliage(
                        args.world,
                        args.terrain_cursor_x as f32 * args.terrain.cell_size - 32.0,
                        args.terrain_cursor_z as f32 * args.terrain.cell_size - 32.0,
                        4.5,
                    );
                }
                if ui.button("Paint foliage patch").clicked() {
                    editor::add_foliage_patch(args.world, args.meshes, args.mesh_cache);
                }
                if ui.button("Add Cube").clicked() {
                    let _ = spawn_primitive(args, [1.0, 1.0, 1.0], [0.78, 0.80, 0.84], false);
                }
                if ui.button("Add Plane").clicked() {
                    let c = terrain_color_at_cursor(args);
                    let mesh = assets::Mesh::make_plane(1.0, 1.0);
                    let handle = args.meshes.add(mesh);
                    let _ = spawn_mesh_entity(args, handle, [6.0, 1.0, 6.0], c, false);
                }
                if ui.button("Add Capsule").clicked() {
                    let c = terrain_color_at_cursor(args);
                    let mesh = assets::Mesh::make_capsule(0.35, 0.55, 12, 20);
                    let handle = args.meshes.add(mesh);
                    let _ = spawn_mesh_entity(args, handle, [1.0, 1.0, 1.0], c, true);
                }
                if ui.button("Add Floor").clicked() {
                    let c = terrain_color_at_cursor(args);
                    let _ = spawn_primitive(args, [4.0, 0.2, 4.0], c, false);
                }
                if ui.button("Add Physics Cube").clicked() {
                    let _ = spawn_primitive(args, [1.0, 1.0, 1.0], [0.72, 0.75, 0.80], true);
                }
                if ui.button("Apply Selected Texture -> Selected Mesh").clicked() {
                    if let (Some(entity), Some(tex)) = (args.selected_renderable.as_ref().copied(), texture_selected.clone()) {
                        upsert_albedo_texture(args.world, entity, tex.clone());
                        if let Ok(img) = image::open(&tex) {
                            let rgb = img.to_rgb8();
                            let mut acc = [0u64; 3];
                            let mut n = 0u64;
                            for px in rgb.pixels().step_by(16) {
                                acc[0] += px[0] as u64;
                                acc[1] += px[1] as u64;
                                acc[2] += px[2] as u64;
                                n += 1;
                            }
                            if n > 0 {
                                if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
                                    rend.color = [
                                        (acc[0] as f32 / n as f32) / 255.0,
                                        (acc[1] as f32 / n as f32) / 255.0,
                                        (acc[2] as f32 / n as f32) / 255.0,
                                    ];
                                }
                            }
                        }
                    }
                }
            });
        };
    if !*undock_asset_browser {
        egui::Panel::bottom("asset_panel")
            .resizable(true)
            .default_size(180.0)
            .show(ctx, |ui| draw_asset_browser_panel(ui));
    } else {
        egui::Window::new("Asset Browser")
            .default_pos([20.0, 560.0])
            .default_size([900.0, 260.0])
            .resizable(true)
            .show(ctx, |ui| draw_asset_browser_panel(ui));
    }

    let mut draw_viewport_panel = |ui: &mut egui::Ui| {
        ui.heading("Viewport");
        ui.label("Click in viewport to select. If a texture is dragging, click mesh to drop it.");
        let desired = egui::vec2(ui.available_width(), (ui.available_height() - 8.0).max(120.0));
        let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
        ui.painter().rect_filled(rect, 6.0, Color32::from_rgb(12, 14, 18));
        ui.painter().text(
            rect.left_top() + egui::vec2(10.0, 10.0),
            egui::Align2::LEFT_TOP,
            "Scene View",
            egui::FontId::proportional(16.0),
            Color32::from_rgb(210, 214, 222),
        );
        if let Some(active) = texture_dragging.as_ref() {
            ui.painter().text(
                rect.left_top() + egui::vec2(10.0, 30.0),
                egui::Align2::LEFT_TOP,
                format!("Drop armed: {}", active),
                egui::FontId::proportional(13.0),
                Color32::from_rgb(130, 190, 255),
            );
        }
        if response.clicked() {
            if let Some(pointer) = response.interact_pointer_pos() {
                if let Some(hit) = pick_entity_in_viewport(args.world, args.camera, rect, pointer) {
                    *args.selected_renderable = Some(hit);
                    if let Some(tex) = texture_dragging.clone() {
                        upsert_albedo_texture(args.world, hit, tex);
                        *texture_dragging = None;
                    }
                }
            }
        }
        if let Some(entity) = args.selected_renderable.as_ref().copied() {
            draw_transform_gizmo(ui, args, entity, rect, response.interact_pointer_pos(), gizmo_mode, gizmo_drag);
        }
    };
    if !*undock_viewport {
        egui::CentralPanel::default().show(ctx, |ui| draw_viewport_panel(ui));
    } else {
        egui::Window::new("Viewport")
            .default_pos([350.0, 70.0])
            .default_size([700.0, 460.0])
            .resizable(true)
            .show(ctx, |ui| draw_viewport_panel(ui));
    }

    toasts.retain(|(_, until)| *until >= args.app_time_seconds);
    if !toasts.is_empty() {
        egui::Area::new("toast_area".into())
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-16.0, 16.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                for (msg, _) in toasts.iter() {
                    egui::Frame::new()
                        .fill(Color32::from_rgba_unmultiplied(24, 28, 34, 235))
                        .corner_radius(6.0)
                        .show(ui, |ui| {
                            ui.colored_label(Color32::from_rgb(190, 230, 255), msg);
                        });
                }
            });
    }

    egui::Window::new("UI Widgets")
        .default_pos([990.0, 520.0])
        .default_size([340.0, 260.0])
        .resizable(true)
        .show(ctx, |ui| {
            ui.label("Create HUD widgets driven by Lua values.");
            ui.horizontal(|ui| {
                ui.label("Id");
                ui.text_edit_singleline(widget_new_id);
            });
            ui.horizontal(|ui| {
                ui.selectable_value(widget_new_kind, UiWidgetKind::HealthBar, "Health Bar");
                ui.selectable_value(widget_new_kind, UiWidgetKind::Counter, "Counter");
                ui.selectable_value(widget_new_kind, UiWidgetKind::Label, "Label");
            });
            if ui.button("Add Widget").clicked() {
                if !widget_new_id.trim().is_empty() {
                    widget_specs.push(UiWidgetSpec {
                        id: widget_new_id.trim().to_string(),
                        kind: *widget_new_kind,
                        x: 30.0,
                        y: 90.0 + widget_specs.len() as f32 * 26.0,
                        w: 240.0,
                    });
                }
            }
            ui.separator();
            let mut remove_idx: Option<usize> = None;
            for (i, w) in widget_specs.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.monospace(&w.id);
                    ui.add(egui::DragValue::new(&mut w.x).prefix("x ").speed(1.0));
                    ui.add(egui::DragValue::new(&mut w.y).prefix("y ").speed(1.0));
                    if ui.button("X").clicked() {
                        remove_idx = Some(i);
                    }
                });
            }
            if let Some(i) = remove_idx {
                widget_specs.remove(i);
            }
            ui.separator();
            ui.label("Lua API: set_ui_value(\"player_health\", 0.75), set_ui_text(\"coins\", \"42\")");
        });

    for w in widget_specs.iter() {
        egui::Area::new(format!("hud_widget_{}", w.id).into())
            .fixed_pos([w.x, w.y])
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                match w.kind {
                    UiWidgetKind::HealthBar => {
                        let v = args.scripts.ui_value(&w.id).clamp(0.0, 1.0);
                        ui.add(egui::ProgressBar::new(v).desired_width(w.w).text(format!("{}: {:.0}%", w.id, v * 100.0)));
                    }
                    UiWidgetKind::Counter => {
                        let txt = args.scripts.ui_text(&w.id).unwrap_or_else(|| {
                            format!("{:.0}", args.scripts.ui_value(&w.id))
                        });
                        ui.label(RichText::new(format!("{}: {}", w.id, txt)).strong());
                    }
                    UiWidgetKind::Label => {
                        let txt = args.scripts.ui_text(&w.id).unwrap_or_else(|| w.id.clone());
                        ui.label(txt);
                    }
                }
            });
    }

    if *show_project_launcher {
        egui::Window::new("Project Launcher")
            .default_pos([420.0, 120.0])
            .default_size([480.0, 210.0])
            .resizable(true)
            .show(ctx, |ui| {
                ui.label("Create or open project folders (app-style workflow).");
                if ui.button("Open Current Project Folder").clicked() {
                    let _ = open_external_editor("explorer \"{file}\"", ".");
                }
                if ui.button("Create New Project Folder In Documents").clicked() {
                    if let Some(home) = std::env::var_os("USERPROFILE") {
                        let p = std::path::PathBuf::from(home).join("Documents").join("TriengineProject");
                        let _ = std::fs::create_dir_all(p.join("Content"));
                        let _ = std::fs::create_dir_all(p.join("scenes"));
                        let _ = std::fs::write(p.join("engine_settings.toml"), std::fs::read_to_string("engine_settings.toml").unwrap_or_default());
                        ui.ctx().copy_text(p.to_string_lossy().to_string());
                    }
                }
                if ui.button("Close Launcher").clicked() {
                    *show_project_launcher = false;
                }
                ui.separator();
                ui.label("Windows startup behavior");
                let startup_on = is_launch_on_startup_enabled();
                ui.label(if startup_on {
                    "Launch on startup: ON"
                } else {
                    "Launch on startup: OFF"
                });
                ui.horizontal(|ui| {
                    if ui.button("Enable Startup Launch").clicked() {
                        if let Err(err) = set_launch_on_startup(true) {
                            args.error_log.push(format!("[Startup] {}", err));
                        }
                    }
                    if ui.button("Disable Startup Launch").clicked() {
                        if let Err(err) = set_launch_on_startup(false) {
                            args.error_log.push(format!("[Startup] {}", err));
                        }
                    }
                });
                ui.separator();
                ui.label("Tip: in installed app, use tools/install_app.ps1 output folder.");
            });
    }

    egui::Window::new("Scene Mode")
        .default_pos([500.0, 18.0])
        .default_size([260.0, 88.0])
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(args.game_preview_mode, false, "Editor Scene");
                ui.selectable_value(args.game_preview_mode, true, "Game Preview");
            });
            ui.label(if *args.game_preview_mode {
                "Game Preview: scripts + physics + animation run."
            } else {
                "Editor Scene: safe edit mode (simulation off)."
            });
        });

    egui::Window::new("Profiler")
        .default_pos([20.0, 60.0])
        .show(ctx, |ui| {
            if let Some(text) = args.profiler.overlay_text() {
                ui.label(text);
            } else {
                ui.label("Profiler warming up...");
            }
        });

    egui::Window::new("Errors")
        .default_pos([20.0, 250.0])
        .default_size([560.0, 180.0])
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Runtime/Lua/Scene errors: {}", args.error_log.len()));
                if ui.button("Clear").clicked() {
                    args.error_log.clear();
                }
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                for line in args.error_log.iter().rev().take(50) {
                    ui.colored_label(Color32::from_rgb(245, 130, 130), line);
                }
                if args.error_log.is_empty() {
                    ui.label("No captured errors.");
                }
            });
        });

    egui::Window::new("Lua Scripts")
        .default_pos([620.0, 60.0])
        .default_size([640.0, 420.0])
        .show(ctx, |ui| {
            ui.label(format!("Content scripts folder: {}", args.scripts_dir));
            ui.checkbox(args.script_hot_reload_enabled, "Hot reload scripts (watch folder)");
            ui.checkbox(args.asset_hot_reload_enabled, "Hot reload meshes/textures (Content)");
            ui.horizontal(|ui| {
                ui.label("External editor:");
                ui.text_edit_singleline(args.preferred_script_editor);
            });
            ui.label("Example: code -r \"{file}\"");
            ui.separator();
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.set_min_width(180.0);
                    if ui.button("New Script").clicked() {
                        let path = format!("{}/new_script.lua", args.scripts_dir);
                        let _ = fs::write(
                            &path,
                            "function update(entity, dt)\n    -- TODO: your logic\nend\n",
                        );
                    }
                    egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                        if let Ok(entries) = fs::read_dir(args.scripts_dir) {
                            for e in entries.flatten() {
                                let p = e.path();
                                if p.extension().map(|x| x == "lua").unwrap_or(false) {
                                    let path = p.to_string_lossy().to_string();
                                    let selected = lua_selected.as_ref().map(|s| s == &path).unwrap_or(false);
                                    if ui.selectable_label(selected, p.file_name().unwrap_or_default().to_string_lossy()).clicked() {
                                        *lua_selected = Some(path.clone());
                                        match fs::read_to_string(&path) {
                                            Ok(t) => {
                                                *lua_buffer = t;
                                                *lua_dirty = false;
                                            }
                                            Err(err) => args.error_log.push(format!("[LuaUI] Read error: {}", err)),
                                        }
                                    }
                                }
                            }
                        }
                    });
                });
                ui.separator();
                ui.vertical(|ui| {
                    if let Some(path) = lua_selected.clone() {
                        ui.label(path.clone());
                        let edit = egui::TextEdit::multiline(lua_buffer)
                            .desired_rows(18)
                            .desired_width(420.0);
                        if ui.add(edit).changed() {
                            *lua_dirty = true;
                        }
                        ui.horizontal(|ui| {
                            if ui.button("Save").clicked() {
                                match fs::write(&path, lua_buffer.as_bytes()) {
                                    Ok(_) => *lua_dirty = false,
                                    Err(err) => args.error_log.push(format!("[LuaUI] Save error: {}", err)),
                                }
                            }
                            if ui.button("Save + Reload").clicked() {
                                match fs::write(&path, lua_buffer.as_bytes()) {
                                    Ok(_) => {
                                        *lua_dirty = false;
                                        if let Err(err) = args.scripts.reload_script(&path) {
                                            args.error_log.push(format!("[LuaUI] Reload error: {}", err));
                                        }
                                    }
                                    Err(err) => args.error_log.push(format!("[LuaUI] Save error: {}", err)),
                                }
                            }
                            if ui.button("Open External").clicked() {
                                if let Err(err) = open_external_editor(args.preferred_script_editor, &path) {
                                    args.error_log.push(format!("[LuaUI] External open error: {}", err));
                                }
                            }
                            if *lua_dirty {
                                ui.colored_label(Color32::from_rgb(230, 180, 90), "Unsaved");
                            }
                        });
                        if let Some(entity) = args.selected_renderable.as_ref().copied() {
                            if ui.button("Attach Script To Selected Mesh").clicked() {
                                let _ = args.world.insert(entity, (components::Script { path: path.clone() },));
                            }
                        }
                    } else {
                        ui.label("Select a .lua script from Content/Scripts");
                    }
                });
            });
        });

    egui::Window::new("Prefabs")
        .default_pos([620.0, 500.0])
        .default_size([640.0, 190.0])
        .show(ctx, |ui| {
            let prefab_dir = "Content/Prefabs";
            let _ = fs::create_dir_all(prefab_dir);
            ui.horizontal(|ui| {
                if ui.button("Save Selected As Prefab").clicked() {
                    if let Some(entity) = args.selected_renderable.as_ref().copied() {
                        match save_selected_as_prefab(args, entity, prefab_dir) {
                            Ok(path) => ui.ctx().copy_text(path),
                            Err(err) => args.error_log.push(format!("[Prefab] Save error: {}", err)),
                        }
                    }
                }
            });
            ui.separator();
            egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                if let Ok(entries) = fs::read_dir(prefab_dir) {
                    for e in entries.flatten() {
                        let p = e.path();
                        if p.extension().map(|x| x == "prefab").unwrap_or(false) {
                            ui.horizontal(|ui| {
                                ui.label(p.file_name().unwrap_or_default().to_string_lossy());
                                if ui.button("Spawn").clicked() {
                                    if let Err(err) = spawn_prefab(args, &p.to_string_lossy()) {
                                        args.error_log.push(format!("[Prefab] Spawn error: {}", err));
                                    }
                                }
                            });
                        }
                    }
                }
            });
        });
}

fn terrain_color_at_cursor(args: &UiFrameArgs<'_>) -> [f32; 3] {
    let world_x = args.terrain_cursor_x as f32 * args.terrain.cell_size - 32.0;
    let world_z = args.terrain_cursor_z as f32 * args.terrain.cell_size - 32.0;
    args.terrain.auto_surface_color_world(world_x, world_z)
}

enum IconKind {
    Camera,
    Light,
    Sky,
}

fn icon_row(ui: &mut egui::Ui, kind: IconKind, label: &str) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
        let p = ui.painter();
        p.rect_filled(rect, 3.0, Color32::from_rgb(28, 30, 36));
        match kind {
            IconKind::Camera => {
                let r = rect.shrink(3.0);
                p.rect_stroke(r, 2.0, egui::Stroke::new(1.2, Color32::from_rgb(170, 188, 230)), egui::StrokeKind::Middle);
                p.circle_stroke(r.center(), 2.8, egui::Stroke::new(1.2, Color32::from_rgb(170, 188, 230)));
            }
            IconKind::Light => {
                let c = rect.center();
                p.circle_filled(c, 3.0, Color32::from_rgb(237, 206, 105));
                p.line_segment([c + egui::vec2(0.0, -6.0), c + egui::vec2(0.0, -4.0)], egui::Stroke::new(1.0, Color32::from_rgb(237, 206, 105)));
                p.line_segment([c + egui::vec2(0.0, 6.0), c + egui::vec2(0.0, 4.0)], egui::Stroke::new(1.0, Color32::from_rgb(237, 206, 105)));
            }
            IconKind::Sky => {
                let r = rect.shrink(3.0);
                p.circle_stroke(r.center(), 4.0, egui::Stroke::new(1.2, Color32::from_rgb(122, 172, 235)));
                p.line_segment([r.left_center(), r.right_center()], egui::Stroke::new(1.0, Color32::from_rgb(122, 172, 235)));
            }
        }
        ui.label(label);
    });
}

fn draw_triangle_logo(p: &egui::Painter, center: egui::Pos2, size: f32) {
    let h = size;
    let a = center + egui::vec2(0.0, -h * 0.62);
    let b = center + egui::vec2(-h * 0.58, h * 0.42);
    let c = center + egui::vec2(h * 0.58, h * 0.42);
    p.line_segment([a, b], egui::Stroke::new(7.0, Color32::from_rgb(208, 214, 224)));
    p.line_segment([b, c], egui::Stroke::new(7.0, Color32::from_rgb(196, 202, 212)));
    p.line_segment([c, a], egui::Stroke::new(7.0, Color32::from_rgb(224, 228, 236)));
}

fn pick_entity_in_viewport(
    world: &World,
    camera: &dyn Camera,
    rect: egui::Rect,
    pointer: egui::Pos2,
) -> Option<hecs::Entity> {
    let vp = camera.view_projection_matrix();
    let mut best: Option<(hecs::Entity, f32)> = None;
    for (e, pos, rend) in world.query::<(hecs::Entity, &components::Position, &components::Renderable)>().iter() {
        let p = glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
        let clip = vp * p;
        if clip.w <= 0.0001 {
            continue;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.x.abs() > 1.2 || ndc.y.abs() > 1.2 {
            continue;
        }
        let sx = rect.left() + ((ndc.x + 1.0) * 0.5) * rect.width();
        let sy = rect.top() + ((1.0 - (ndc.y + 1.0) * 0.5) * rect.height());
        let dx = pointer.x - sx;
        let dy = pointer.y - sy;
        let d2 = dx * dx + dy * dy;
        let radius_px = (rend.scale[0].max(rend.scale[1]).max(rend.scale[2]) * 22.0).clamp(10.0, 64.0);
        if d2 <= radius_px * radius_px {
            match best {
                Some((_, best_d2)) if d2 >= best_d2 => {}
                _ => best = Some((e, d2)),
            }
        }
    }
    best.map(|b| b.0)
}

fn project_to_screen(camera: &dyn Camera, rect: egui::Rect, world: glam::Vec3) -> Option<egui::Pos2> {
    let clip = camera.view_projection_matrix() * glam::Vec4::new(world.x, world.y, world.z, 1.0);
    if clip.w <= 0.0001 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    let sx = rect.left() + ((ndc.x + 1.0) * 0.5) * rect.width();
    let sy = rect.top() + ((1.0 - (ndc.y + 1.0) * 0.5) * rect.height());
    Some(egui::pos2(sx, sy))
}

fn nearest_axis_hit(pointer: egui::Pos2, origin: egui::Pos2, x_end: egui::Pos2, y_end: egui::Pos2, z_end: egui::Pos2) -> Option<GizmoAxis> {
    let d = |a: egui::Pos2, b: egui::Pos2| ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
    let dx = d(pointer, x_end);
    let dy = d(pointer, y_end);
    let dz = d(pointer, z_end);
    let min_d = dx.min(dy.min(dz));
    if min_d > 22.0 || d(pointer, origin) < 6.0 {
        return None;
    }
    if min_d == dx {
        Some(GizmoAxis::X)
    } else if min_d == dy {
        Some(GizmoAxis::Y)
    } else {
        Some(GizmoAxis::Z)
    }
}

fn draw_transform_gizmo(
    ui: &mut egui::Ui,
    args: &mut UiFrameArgs<'_>,
    entity: hecs::Entity,
    rect: egui::Rect,
    pointer: Option<egui::Pos2>,
    mode: &mut GizmoMode,
    drag: &mut Option<GizmoDragState>,
) {
    let Ok(pos) = args.world.get::<&components::Position>(entity) else { return; };
    let origin_world = glam::Vec3::new(pos.x, pos.y, pos.z);
    let axis_len = 1.0;
    let Some(origin) = project_to_screen(args.camera, rect, origin_world) else { return; };
    let Some(x_end) = project_to_screen(args.camera, rect, origin_world + glam::Vec3::X * axis_len) else { return; };
    let Some(y_end) = project_to_screen(args.camera, rect, origin_world + glam::Vec3::Y * axis_len) else { return; };
    let Some(z_end) = project_to_screen(args.camera, rect, origin_world + glam::Vec3::Z * axis_len) else { return; };

    let p = ui.painter();
    p.line_segment([origin, x_end], egui::Stroke::new(2.2, Color32::from_rgb(230, 70, 70)));
    p.line_segment([origin, y_end], egui::Stroke::new(2.2, Color32::from_rgb(80, 220, 90)));
    p.line_segment([origin, z_end], egui::Stroke::new(2.2, Color32::from_rgb(95, 150, 255)));
    p.circle_filled(x_end, 4.5, Color32::from_rgb(230, 70, 70));
    p.circle_filled(y_end, 4.5, Color32::from_rgb(80, 220, 90));
    p.circle_filled(z_end, 4.5, Color32::from_rgb(95, 150, 255));

    if let Some(ptr) = pointer {
        let primary_down = ui.ctx().input(|i| i.pointer.primary_down());
        if drag.is_none() && primary_down {
            if let Some(axis) = nearest_axis_hit(ptr, origin, x_end, y_end, z_end) {
                let rot = args.world.get::<&components::Rotation>(entity).ok();
                let rend = args.world.get::<&components::Renderable>(entity).ok();
                *drag = Some(GizmoDragState {
                    axis,
                    pointer_start: ptr,
                    pos_start: [pos.x, pos.y, pos.z],
                    rot_start: [
                        rot.as_ref().map(|r| r.pitch).unwrap_or(0.0),
                        rot.as_ref().map(|r| r.yaw).unwrap_or(0.0),
                        rot.as_ref().map(|r| r.roll).unwrap_or(0.0),
                    ],
                    scale_start: [
                        rend.as_ref().map(|r| r.scale[0]).unwrap_or(1.0),
                        rend.as_ref().map(|r| r.scale[1]).unwrap_or(1.0),
                        rend.as_ref().map(|r| r.scale[2]).unwrap_or(1.0),
                    ],
                });
            }
        }
        if let Some(d) = *drag {
            if primary_down {
                let delta_px = (ptr.x - d.pointer_start.x) + (d.pointer_start.y - ptr.y);
                match *mode {
                    GizmoMode::Move => {
                        if let Ok(mut p) = args.world.get::<&mut components::Position>(entity) {
                            let amt = delta_px * 0.01;
                            match d.axis {
                                GizmoAxis::X => p.x = d.pos_start[0] + amt,
                                GizmoAxis::Y => p.y = d.pos_start[1] + amt,
                                GizmoAxis::Z => p.z = d.pos_start[2] + amt,
                            }
                        }
                    }
                    GizmoMode::Rotate => {
                        if let Ok(mut r) = args.world.get::<&mut components::Rotation>(entity) {
                            let amt = delta_px * 0.01;
                            match d.axis {
                                GizmoAxis::X => r.pitch = d.rot_start[0] + amt,
                                GizmoAxis::Y => r.yaw = d.rot_start[1] + amt,
                                GizmoAxis::Z => r.roll = d.rot_start[2] + amt,
                            }
                        }
                    }
                    GizmoMode::Scale => {
                        if let Ok(mut r) = args.world.get::<&mut components::Renderable>(entity) {
                            let amt = (delta_px * 0.01).max(-0.95);
                            match d.axis {
                                GizmoAxis::X => r.scale[0] = (d.scale_start[0] + amt).max(0.05),
                                GizmoAxis::Y => r.scale[1] = (d.scale_start[1] + amt).max(0.05),
                                GizmoAxis::Z => r.scale[2] = (d.scale_start[2] + amt).max(0.05),
                            }
                        }
                    }
                }
            } else {
                *drag = None;
            }
        }
    } else {
        *drag = None;
    }
}

fn upsert_albedo_texture(world: &mut World, entity: hecs::Entity, path: String) {
    if let Ok(mut m) = world.get::<&mut components::MaterialTexture>(entity) {
        m.path = path;
    } else {
        let _ = world.insert(
            entity,
            (components::MaterialTexture {
                path,
                normal_path: String::new(),
                metallic_roughness_path: String::new(),
            },),
        );
    }
}

fn upsert_normal_texture(world: &mut World, entity: hecs::Entity, path: String) {
    if let Ok(mut m) = world.get::<&mut components::MaterialTexture>(entity) {
        m.normal_path = path;
    } else {
        let _ = world.insert(
            entity,
            (components::MaterialTexture {
                path: String::new(),
                normal_path: path,
                metallic_roughness_path: String::new(),
            },),
        );
    }
}

fn upsert_mr_texture(world: &mut World, entity: hecs::Entity, path: String) {
    if let Ok(mut m) = world.get::<&mut components::MaterialTexture>(entity) {
        m.metallic_roughness_path = path;
    } else {
        let _ = world.insert(
            entity,
            (components::MaterialTexture {
                path: String::new(),
                normal_path: String::new(),
                metallic_roughness_path: path,
            },),
        );
    }
}

fn clear_player_starts(world: &mut World) {
    let ids: Vec<hecs::Entity> = world
        .query::<(hecs::Entity, &components::PlayerStart)>()
        .iter()
        .map(|(e, _)| e)
        .collect();
    for e in ids {
        let _ = world.despawn(e);
    }
}

fn set_player_start_from_selected(args: &mut UiFrameArgs<'_>, entity: hecs::Entity) {
    let p = if let Ok(pos) = args.world.get::<&components::Position>(entity) {
        [pos.x, pos.y, pos.z]
    } else {
        return;
    };
    clear_player_starts(args.world);
    let _ = args.world.spawn((
        components::Position { x: p[0], y: p[1], z: p[2] },
        components::PlayerStart,
    ));
}

fn preset_combo(ui: &mut egui::Ui, preset: &mut RenderPreset) {
    egui::ComboBox::from_id_salt("render_preset")
        .selected_text(format!("{:?}", preset))
        .show_ui(ui, |ui| {
            ui.selectable_value(preset, RenderPreset::Mobile, "Mobile");
            ui.selectable_value(preset, RenderPreset::Balanced, "Balanced");
            ui.selectable_value(preset, RenderPreset::Cinematic, "Cinematic");
            ui.selectable_value(preset, RenderPreset::Custom, "Custom");
        });
}

fn thumbnail_card(ui: &mut egui::Ui, kind: &str, name: &str, swatch: Option<Color32>) {
    egui::Frame::new()
        .fill(Color32::from_rgb(30, 30, 36))
        .corner_radius(6.0)
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(96.0, 54.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 4.0, Color32::from_rgb(16, 16, 20));
                if let Some(c) = swatch {
                    let inner = rect.shrink2(egui::vec2(16.0, 14.0));
                    ui.painter().rect_filled(inner, 4.0, c);
                }
                ui.label(RichText::new(kind).color(Color32::from_rgb(180, 180, 190)).small());
                ui.label(RichText::new(name).strong());
            });
        });
}

fn spawn_primitive(
    args: &mut UiFrameArgs<'_>,
    scale: [f32; 3],
    color: [f32; 3],
    with_physics: bool,
) -> Result<(), String> {
    let mesh_path = "meshes/cube.obj".to_string();
    let handle = if let Some(h) = args.mesh_cache.get(&mesh_path).copied() {
        h
    } else {
        let mesh = assets::Mesh::load(&mesh_path).map_err(|e| e.to_string())?;
        let h = args.meshes.add(mesh);
        args.mesh_cache.insert(mesh_path, h);
        h
    };

    spawn_mesh_entity(args, handle, scale, color, with_physics)
}

fn spawn_mesh_entity(
    args: &mut UiFrameArgs<'_>,
    handle: assets::Handle<assets::Mesh>,
    scale: [f32; 3],
    color: [f32; 3],
    with_physics: bool,
) -> Result<(), String> {
    let x = args.terrain_cursor_x as f32 * args.terrain.cell_size - 32.0;
    let z = args.terrain_cursor_z as f32 * args.terrain.cell_size - 32.0;
    let y = args.terrain.sample_height_world(x, z) + 0.5;
    let entity = args.world.spawn((
        components::Position { x, y, z },
        components::Rotation {
            pitch: 0.0,
            yaw: 0.0,
            roll: 0.0,
        },
        components::Renderable {
            mesh: handle,
            color,
            metallic: 0.0,
            roughness: 0.72,
            ao: 1.0,
            scale,
        },
    ));
    if with_physics {
        let _ = args.world.insert(
            entity,
            (
                components::RigidBody {
                    velocity_x: 0.0,
                    velocity_y: 0.0,
                    _velocity_z: 0.0,
                    on_ground: false,
                    use_gravity: true,
                },
                components::Collider {
                    half_w: scale[0].abs() * 0.5,
                    half_h: scale[1].abs() * 0.5,
                    half_d: scale[2].abs() * 0.5,
                },
            ),
        );
    }
    *args.selected_renderable = Some(entity);
    Ok(())
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

fn is_launch_on_startup_enabled() -> bool {
    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("reg")
            .args([
                "query",
                r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                "/v",
                "Triengine",
            ])
            .output();
        return output.map(|o| o.status.success()).unwrap_or(false);
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

fn set_launch_on_startup(enable: bool) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        if enable {
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            let status = std::process::Command::new("reg")
                .args([
                    "add",
                    r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                    "/v",
                    "Triengine",
                    "/t",
                    "REG_SZ",
                    "/d",
                    &format!("\"{}\"", exe.to_string_lossy()),
                    "/f",
                ])
                .status()
                .map_err(|e| e.to_string())?;
            if !status.success() {
                return Err("Could not enable startup launch".to_string());
            }
        } else {
            let status = std::process::Command::new("reg")
                .args([
                    "delete",
                    r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                    "/v",
                    "Triengine",
                    "/f",
                ])
                .status()
                .map_err(|e| e.to_string())?;
            if !status.success() {
                return Err("Could not disable startup launch".to_string());
            }
        }
        Ok(())
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = enable;
        Err("Startup toggle is currently implemented on Windows only.".to_string())
    }
}

fn save_selected_as_prefab(
    args: &mut UiFrameArgs<'_>,
    entity: hecs::Entity,
    prefab_dir: &str,
) -> Result<String, String> {
    let pos = args
        .world
        .get::<&components::Position>(entity)
        .map_err(|_| "No Position".to_string())?;
    let rend = args
        .world
        .get::<&components::Renderable>(entity)
        .map_err(|_| "No Renderable".to_string())?;
    let rot = args.world.get::<&components::Rotation>(entity).ok();
    let script = args.world.get::<&components::Script>(entity).ok();
    let tex = args.world.get::<&components::MaterialTexture>(entity).ok();
    let mesh_path = args
        .mesh_cache
        .iter()
        .find_map(|(p, h)| if h.id == rend.mesh.id { Some(p.clone()) } else { None })
        .unwrap_or_else(|| "meshes/cube.obj".to_string());
    let prefab_path = format!("{}/prefab_{}.prefab", prefab_dir, entity.to_bits().get());
    let rp = rot.as_ref().map(|r| r.pitch).unwrap_or(0.0);
    let ry = rot.as_ref().map(|r| r.yaw).unwrap_or(0.0);
    let rr = rot.as_ref().map(|r| r.roll).unwrap_or(0.0);
    let text = format!(
        "mesh={}\npos={} {} {}\nrot={} {} {}\nscale={} {} {}\ncolor={} {} {}\nmetallic={}\nroughness={}\nao={}\nscript={}\nalbedo_tex={}\nnormal_tex={}\nmr_tex={}\n",
        mesh_path,
        pos.x, pos.y, pos.z,
        rp, ry, rr,
        rend.scale[0], rend.scale[1], rend.scale[2],
        rend.color[0], rend.color[1], rend.color[2],
        rend.metallic, rend.roughness, rend.ao,
        script.map(|s| s.path.clone()).unwrap_or_default(),
        tex.as_ref().map(|t| t.path.clone()).unwrap_or_default(),
        tex.as_ref().map(|t| t.normal_path.clone()).unwrap_or_default(),
        tex.as_ref().map(|t| t.metallic_roughness_path.clone()).unwrap_or_default(),
    );
    fs::write(&prefab_path, text).map_err(|e| e.to_string())?;
    Ok(prefab_path)
}

fn spawn_prefab(args: &mut UiFrameArgs<'_>, path: &str) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut kv = std::collections::HashMap::<String, String>::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            kv.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    let mesh_path = kv.get("mesh").cloned().unwrap_or_else(|| "meshes/cube.obj".to_string());
    let handle = if let Some(h) = args.mesh_cache.get(&mesh_path).copied() {
        h
    } else {
        let mesh = assets::Mesh::load(&mesh_path).map_err(|e| e.to_string())?;
        let h = args.meshes.add(mesh);
        args.mesh_cache.insert(mesh_path.clone(), h);
        h
    };
    let parse3 = |s: Option<&String>, d: [f32; 3]| -> [f32; 3] {
        if let Some(v) = s {
            let p: Vec<f32> = v.split_whitespace().filter_map(|t| t.parse::<f32>().ok()).collect();
            if p.len() >= 3 {
                return [p[0], p[1], p[2]];
            }
        }
        d
    };
    let pos = parse3(kv.get("pos"), [0.0, 0.5, 0.0]);
    let rot = parse3(kv.get("rot"), [0.0, 0.0, 0.0]);
    let scale = parse3(kv.get("scale"), [1.0, 1.0, 1.0]);
    let color = parse3(kv.get("color"), [0.8, 0.8, 0.8]);
    let metallic = kv.get("metallic").and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
    let roughness = kv.get("roughness").and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.7);
    let ao = kv.get("ao").and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
    let e = args.world.spawn((
        components::Position { x: pos[0], y: pos[1], z: pos[2] },
        components::Rotation {
            pitch: rot[0],
            yaw: rot[1],
            roll: rot[2],
        },
        components::Renderable {
            mesh: handle,
            color,
            metallic,
            roughness,
            ao,
            scale,
        },
    ));
    if let Some(script) = kv.get("script") {
        if !script.is_empty() {
            let _ = args.world.insert(e, (components::Script { path: script.clone() },));
        }
    }
    let albedo_tex = kv.get("albedo_tex").cloned().unwrap_or_default();
    let normal_tex = kv.get("normal_tex").cloned().unwrap_or_default();
    let mr_tex = kv.get("mr_tex").cloned().unwrap_or_default();
    if !albedo_tex.is_empty() || !normal_tex.is_empty() || !mr_tex.is_empty() {
        let _ = args.world.insert(
            e,
            (components::MaterialTexture {
                path: albedo_tex,
                normal_path: normal_tex,
                metallic_roughness_path: mr_tex,
            },),
        );
    }
    *args.selected_renderable = Some(e);
    Ok(())
}
