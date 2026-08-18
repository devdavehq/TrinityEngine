#![allow(deprecated)]

mod panels;

use std::fs;
use std::collections::HashMap;

use crate::editor;
use crate::editor_assets::{AssetMetadataDb, IconRegistry};
use crate::editor_persist;
use crate::project_registry::ProjectRegistry;
use crate::materials::MaterialLibrary;
use crate::profiler::FrameProfiler;
use crate::renderer::{RenderFeatures, Renderer};
use crate::settings::{EngineSettings, RenderPreset};
use crate::terrain::{remove_nearby_foliage, spawn_foliage_ring, TerrainWorld};
use crate::navigation::NavGrid;
use crate::scripting::ScriptEngine;
use crate::camera::Camera;
use crate::{assets, components};
use egui::{Color32, RichText};
use egui_dock::{DockArea, DockState, NodeIndex, Style as DockStyle, TabViewer};
use hecs::World;
use winit::window::Window;

const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
const SPLASH_LOGO_PATH: &str = "assets/branding/splash_logo.png";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UiWidgetKind {
    HealthBar,
    Counter,
    Label,
    Button,
    Slider,
    Toggle,
    Panel,
    ProgressRing,
    Meter,
}

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum UiAnchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl Default for UiAnchor {
    fn default() -> Self { Self::TopLeft }
}

#[derive(Clone)]
struct UiWidgetSpec {
    id: String,
    kind: UiWidgetKind,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    visible: bool,
    z_order: i32,
    color: [f32; 4],
    bg_color: [f32; 4],
    font_size: f32,
    anchor: UiAnchor,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GizmoMode {
    Move,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GizmoAxis {
    X,
    Y,
    Z,
    XY,
    YZ,
    ZX,
    Uniform,
}

#[derive(Clone, Copy)]
pub(crate) struct GizmoDragState {
    axis: GizmoAxis,
    pointer_start: egui::Pos2,
    pos_start: [f32; 3],
    rot_start: [f32; 3],
    scale_start: [f32; 3],
    axis_dir_screen: egui::Vec2,
    axis_dir_screen_2: egui::Vec2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GizmoSpace {
    Local,
    World,
}

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
enum EditorDockTab {
    Viewport,
    Outliner,
    Details,
    ContentBrowser,
    Profiler,
    Console,
    ScriptEditor,
    Levels,
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
    gizmo_space: GizmoSpace,
    gizmo_axis_lock: Option<GizmoAxis>,
    terrain_mode: bool,
    terrain_brush_mode: components::TerrainBrushMode,
    terrain_brush_radius: f32,
    snap_translate: f32,
    snap_rotate_deg: f32,
    snap_scale: f32,
    snap_enabled: bool,
    texture_thumbnail_cache: HashMap<String, egui::TextureHandle>,
    mesh_thumbnail_color_cache: HashMap<String, Color32>,
    scene_texture_id: Option<egui::TextureId>,
    asset_db: AssetMetadataDb,
    icon_registry: IconRegistry,
    dock_state: DockState<EditorDockTab>,
    workspace_preset: String,
    asset_search: String,
    asset_kind_filter: String,
    asset_sort_desc: bool,
    icon_texture_cache: HashMap<String, egui::TextureHandle>,
    hub_add_path: String,
    hub_new_project_name: String,
    console_filter: String,
    show_material_editor: bool,
    show_foliage_editor: bool,
    show_icon_debug: bool,
    show_perf_safety_check: bool,
    show_scene_manager: bool,
    splash_logo_texture: Option<egui::TextureHandle>,
    scene_picker_choice: String,
    scene_create_name: String,
    scene_rename_name: String,
    scene_duplicate_name: String,
    asset_index_dirty: bool,
    icon_registry_dirty: bool,
    undo_stack: Vec<EntityEditCommand>,
    redo_stack: Vec<EntityEditCommand>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct EditorDockLayoutFile {
    workspace_preset: String,
    dock_state: DockState<EditorDockTab>,
}

struct PreparedUi {
    paint_jobs: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    screen_desc: egui_wgpu::ScreenDescriptor,
}

#[derive(Clone)]
struct EntityEditState {
    entity: hecs::Entity,
    position: Option<components::Position>,
    rotation: Option<components::Rotation>,
    renderable: Option<components::Renderable>,
    rigid_body: Option<components::RigidBody>,
    collider: Option<components::Collider>,
    obb_collider: Option<components::OrientedBoxCollider>,
    hinge_joint: Option<components::HingeJoint>,
    fixed_joint: Option<components::FixedJoint>,
    spring_joint: Option<components::SpringJoint>,
    rope_constraint: Option<components::RopeConstraint>,
    material_texture: Option<components::MaterialTexture>,
}

#[derive(Clone)]
struct EntityEditCommand {
    before: Option<EntityEditState>,
    after: Option<EntityEditState>,
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
    pub terrain_world: &'a mut TerrainWorld,
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
    pub return_to_hub: &'a mut bool,
    pub scene_path: &'a mut String,
    pub available_scene_paths: &'a [String],
    pub requested_scene_switch: &'a mut Option<String>,
    pub camera_nav_speed: f32,
    pub time_of_day: &'a mut crate::environment::time_of_day::TimeOfDay,
    pub weather: &'a mut crate::environment::weather::WeatherState,
    pub sky: &'a mut crate::environment::sky::SkyParams,
    pub audio: &'a mut Option<crate::audio::AudioSystem>,
    /// Set true by the "Bake Lighting" button; main loop performs the bake.
    pub bake_requested: &'a mut bool,
    /// Level manager + streaming config, edited by the Levels panel.
    pub levels: &'a mut crate::engine_subsystems::LevelState,
}

fn nearly_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5
}

fn state_same(a: &Option<EntityEditState>, b: &Option<EntityEditState>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(sa), Some(sb)) => {
            if sa.entity != sb.entity {
                return false;
            }
            let cmp_pos = |pa: Option<components::Position>, pb: Option<components::Position>| match (pa, pb) {
                (None, None) => true,
                (Some(a), Some(b)) => nearly_eq(a.x, b.x) && nearly_eq(a.y, b.y) && nearly_eq(a.z, b.z),
                _ => false,
            };
            let cmp_rot = |ra: Option<components::Rotation>, rb: Option<components::Rotation>| match (ra, rb) {
                (None, None) => true,
                (Some(a), Some(b)) => nearly_eq(a.pitch, b.pitch) && nearly_eq(a.yaw, b.yaw) && nearly_eq(a.roll, b.roll),
                _ => false,
            };
            let cmp_rend = |ra: Option<components::Renderable>, rb: Option<components::Renderable>| match (ra, rb) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.mesh.id == b.mesh.id
                        && nearly_eq(a.color[0], b.color[0])
                        && nearly_eq(a.color[1], b.color[1])
                        && nearly_eq(a.color[2], b.color[2])
                        && nearly_eq(a.metallic, b.metallic)
                        && nearly_eq(a.roughness, b.roughness)
                        && nearly_eq(a.ao, b.ao)
                        && nearly_eq(a.scale[0], b.scale[0])
                        && nearly_eq(a.scale[1], b.scale[1])
                        && nearly_eq(a.scale[2], b.scale[2])
                }
                _ => false,
            };
            let cmp_rb = |ra: Option<components::RigidBody>, rb: Option<components::RigidBody>| match (ra, rb) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.body_type == b.body_type
                        && nearly_eq(a.mass, b.mass)
                        && nearly_eq(a.restitution, b.restitution)
                        && nearly_eq(a.friction, b.friction)
                        && nearly_eq(a.linear_damping, b.linear_damping)
                        && nearly_eq(a.velocity_x, b.velocity_x)
                        && nearly_eq(a.velocity_y, b.velocity_y)
                        && nearly_eq(a._velocity_z, b._velocity_z)
                        && nearly_eq(a.angular_velocity, b.angular_velocity)
                        && nearly_eq(a.angular_damping, b.angular_damping)
                        && nearly_eq(a.torque, b.torque)
                        && a.on_ground == b.on_ground
                        && a.use_gravity == b.use_gravity
                        && nearly_eq(a.inertia, b.inertia)
                        && a.lock_rotation == b.lock_rotation
                        && a.can_sleep == b.can_sleep
                        && a.sleeping == b.sleeping
                }
                _ => false,
            };
            let cmp_col = |ca: Option<components::Collider>, cb: Option<components::Collider>| match (ca, cb) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    nearly_eq(a.half_w, b.half_w)
                        && nearly_eq(a.half_h, b.half_h)
                        && nearly_eq(a.half_d, b.half_d)
                        && a.layer == b.layer
                        && a.mask == b.mask
                }
                _ => false,
            };
            let cmp_obb =
                |oa: Option<components::OrientedBoxCollider>, ob: Option<components::OrientedBoxCollider>| match (oa, ob) {
                    (None, None) => true,
                    (Some(a), Some(b)) => {
                        nearly_eq(a.half_w, b.half_w)
                            && nearly_eq(a.half_h, b.half_h)
                            && nearly_eq(a.half_d, b.half_d)
                            && nearly_eq(a.angle_rad, b.angle_rad)
                            && a.layer == b.layer
                            && a.mask == b.mask
                    }
                    _ => false,
                };
            let cmp_hinge = |ja: Option<components::HingeJoint>, jb: Option<components::HingeJoint>| match (ja, jb) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.connected == b.connected
                        && nearly_eq(a.rest_length, b.rest_length)
                        && nearly_eq(a.stiffness, b.stiffness)
                        && nearly_eq(a.anchor_a[0], b.anchor_a[0])
                        && nearly_eq(a.anchor_a[1], b.anchor_a[1])
                        && nearly_eq(a.anchor_a[2], b.anchor_a[2])
                        && nearly_eq(a.anchor_b[0], b.anchor_b[0])
                        && nearly_eq(a.anchor_b[1], b.anchor_b[1])
                        && nearly_eq(a.anchor_b[2], b.anchor_b[2])
                }
                _ => false,
            };
            let cmp_fixed = |ja: Option<components::FixedJoint>, jb: Option<components::FixedJoint>| match (ja, jb) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.connected == b.connected
                        && nearly_eq(a.offset_x, b.offset_x)
                        && nearly_eq(a.offset_y, b.offset_y)
                        && nearly_eq(a.stiffness, b.stiffness)
                        && nearly_eq(a.anchor_a[0], b.anchor_a[0])
                        && nearly_eq(a.anchor_a[1], b.anchor_a[1])
                        && nearly_eq(a.anchor_a[2], b.anchor_a[2])
                        && nearly_eq(a.anchor_b[0], b.anchor_b[0])
                        && nearly_eq(a.anchor_b[1], b.anchor_b[1])
                        && nearly_eq(a.anchor_b[2], b.anchor_b[2])
                }
                _ => false,
            };
            let cmp_spring = |ja: Option<components::SpringJoint>, jb: Option<components::SpringJoint>| match (ja, jb) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.connected == b.connected
                        && nearly_eq(a.rest_length, b.rest_length)
                        && nearly_eq(a.stiffness, b.stiffness)
                        && nearly_eq(a.damping, b.damping)
                        && nearly_eq(a.anchor_a[0], b.anchor_a[0])
                        && nearly_eq(a.anchor_a[1], b.anchor_a[1])
                        && nearly_eq(a.anchor_a[2], b.anchor_a[2])
                        && nearly_eq(a.anchor_b[0], b.anchor_b[0])
                        && nearly_eq(a.anchor_b[1], b.anchor_b[1])
                        && nearly_eq(a.anchor_b[2], b.anchor_b[2])
                }
                _ => false,
            };
            let cmp_rope =
                |ja: Option<components::RopeConstraint>, jb: Option<components::RopeConstraint>| match (ja, jb) {
                    (None, None) => true,
                    (Some(a), Some(b)) => {
                        a.connected == b.connected
                            && nearly_eq(a.max_length, b.max_length)
                            && nearly_eq(a.stiffness, b.stiffness)
                            && nearly_eq(a.anchor_a[0], b.anchor_a[0])
                            && nearly_eq(a.anchor_a[1], b.anchor_a[1])
                            && nearly_eq(a.anchor_a[2], b.anchor_a[2])
                            && nearly_eq(a.anchor_b[0], b.anchor_b[0])
                            && nearly_eq(a.anchor_b[1], b.anchor_b[1])
                            && nearly_eq(a.anchor_b[2], b.anchor_b[2])
                    }
                    _ => false,
                };
            let cmp_tex = |ta: &Option<components::MaterialTexture>, tb: &Option<components::MaterialTexture>| match (ta, tb) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.path == b.path
                        && a.normal_path == b.normal_path
                        && a.metallic_roughness_path == b.metallic_roughness_path
                }
                _ => false,
            };
            cmp_pos(sa.position, sb.position)
                && cmp_rot(sa.rotation, sb.rotation)
                && cmp_rend(sa.renderable, sb.renderable)
                && cmp_rb(sa.rigid_body, sb.rigid_body)
                && cmp_col(sa.collider, sb.collider)
                && cmp_obb(sa.obb_collider, sb.obb_collider)
                && cmp_hinge(sa.hinge_joint, sb.hinge_joint)
                && cmp_fixed(sa.fixed_joint, sb.fixed_joint)
                && cmp_spring(sa.spring_joint, sb.spring_joint)
                && cmp_rope(sa.rope_constraint, sb.rope_constraint)
                && cmp_tex(&sa.material_texture, &sb.material_texture)
        }
        _ => false,
    }
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
        let mut dock_state = DockState::new(vec![EditorDockTab::Viewport]);
        let [main, left] = dock_state
            .main_surface_mut()
            .split_left(NodeIndex::root(), 0.20, vec![EditorDockTab::Outliner]);
        let [_main, _right] = dock_state
            .main_surface_mut()
            .split_right(main, 0.26, vec![EditorDockTab::Details]);
        dock_state
            .main_surface_mut()
            .split_below(left, 0.60, vec![EditorDockTab::ContentBrowser, EditorDockTab::Console, EditorDockTab::Profiler, EditorDockTab::ScriptEditor, EditorDockTab::Levels]);
        let mut workspace_preset = "Default".to_string();
        if let Some((p, st)) = load_saved_dock_layout() {
            workspace_preset = p;
            dock_state = st;
        }
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
                UiWidgetSpec { id: "player_health".to_string(), kind: UiWidgetKind::HealthBar, x: 24.0, y: 24.0, w: 280.0, h: 24.0, visible: true, z_order: 0, color: [1.0, 1.0, 1.0, 1.0], bg_color: [0.0, 0.0, 0.0, 0.5], font_size: 14.0, anchor: UiAnchor::TopLeft },
                UiWidgetSpec { id: "coins".to_string(), kind: UiWidgetKind::Counter, x: 24.0, y: 56.0, w: 180.0, h: 24.0, visible: true, z_order: 0, color: [1.0, 1.0, 1.0, 1.0], bg_color: [0.0, 0.0, 0.0, 0.5], font_size: 14.0, anchor: UiAnchor::TopLeft },
            ],
            widget_new_id: "new_widget".to_string(),
            widget_new_kind: UiWidgetKind::Label,
            show_project_launcher: false,
            gizmo_mode: GizmoMode::Move,
            gizmo_drag: None,
            gizmo_space: GizmoSpace::World,
            gizmo_axis_lock: None,
            terrain_mode: false,
            terrain_brush_mode: components::TerrainBrushMode::Raise,
            terrain_brush_radius: 5.0,
            snap_translate: 0.25,
            snap_rotate_deg: 5.0,
            snap_scale: 0.1,
            snap_enabled: true,
            texture_thumbnail_cache: HashMap::new(),
            mesh_thumbnail_color_cache: HashMap::new(),
            scene_texture_id: None,
            asset_db: AssetMetadataDb::default(),
            icon_registry: IconRegistry::default(),
            dock_state,
            workspace_preset,
            asset_search: String::new(),
            asset_kind_filter: "all".to_string(),
            asset_sort_desc: true,
            icon_texture_cache: HashMap::new(),
            hub_add_path: String::new(),
            hub_new_project_name: "MyProject".to_string(),
            console_filter: String::new(),
            show_material_editor: false,
            show_foliage_editor: false,
            show_icon_debug: false,
            show_perf_safety_check: false,
            show_scene_manager: false,
            splash_logo_texture: None,
            scene_picker_choice: String::new(),
            scene_create_name: "new.scene".to_string(),
            scene_rename_name: String::new(),
            scene_duplicate_name: String::new(),
            asset_index_dirty: true,
            icon_registry_dirty: true,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    fn capture_selected_state(&self, world: &World, selected: Option<hecs::Entity>) -> Option<EntityEditState> {
        let e = selected?;
        Some(EntityEditState {
            entity: e,
            position: world.get::<&components::Position>(e).ok().map(|v| *v),
            rotation: world.get::<&components::Rotation>(e).ok().map(|v| *v),
            renderable: world.get::<&components::Renderable>(e).ok().map(|v| *v),
            rigid_body: world.get::<&components::RigidBody>(e).ok().map(|v| *v),
            collider: world.get::<&components::Collider>(e).ok().map(|v| *v),
            obb_collider: world
                .get::<&components::OrientedBoxCollider>(e)
                .ok()
                .map(|v| *v),
            hinge_joint: world.get::<&components::HingeJoint>(e).ok().map(|v| *v),
            fixed_joint: world.get::<&components::FixedJoint>(e).ok().map(|v| *v),
            spring_joint: world.get::<&components::SpringJoint>(e).ok().map(|v| *v),
            rope_constraint: world.get::<&components::RopeConstraint>(e).ok().map(|v| *v),
            material_texture: world
                .get::<&components::MaterialTexture>(e)
                .ok()
                .map(|v| components::MaterialTexture {
                    path: v.path.clone(),
                    normal_path: v.normal_path.clone(),
                    metallic_roughness_path: v.metallic_roughness_path.clone(),
                }),
        })
    }

    fn apply_state(world: &mut World, state: &Option<EntityEditState>) -> Option<hecs::Entity> {
        let s = state.as_ref()?;
        let e = s.entity;
        let _ = world.insert(
            e,
            (
                s.position.unwrap_or(components::Position { x: 0.0, y: 0.0, z: 0.0 }),
                s.rotation.unwrap_or(components::Rotation {
                    pitch: 0.0,
                    yaw: 0.0,
                    roll: 0.0,
                }),
            ),
        );
        if let Some(r) = s.renderable {
            let _ = world.insert(e, (r,));
        } else {
            let _ = world.remove_one::<components::Renderable>(e);
        }
        if let Some(rb) = s.rigid_body {
            let _ = world.insert(e, (rb,));
        } else {
            let _ = world.remove_one::<components::RigidBody>(e);
        }
        if let Some(c) = s.collider {
            let _ = world.insert(e, (c,));
        } else {
            let _ = world.remove_one::<components::Collider>(e);
        }
        if let Some(c) = s.obb_collider {
            let _ = world.insert(e, (c,));
        } else {
            let _ = world.remove_one::<components::OrientedBoxCollider>(e);
        }
        if let Some(j) = s.hinge_joint {
            let _ = world.insert(e, (j,));
        } else {
            let _ = world.remove_one::<components::HingeJoint>(e);
        }
        if let Some(j) = s.fixed_joint {
            let _ = world.insert(e, (j,));
        } else {
            let _ = world.remove_one::<components::FixedJoint>(e);
        }
        if let Some(j) = s.spring_joint {
            let _ = world.insert(e, (j,));
        } else {
            let _ = world.remove_one::<components::SpringJoint>(e);
        }
        if let Some(j) = s.rope_constraint {
            let _ = world.insert(e, (j,));
        } else {
            let _ = world.remove_one::<components::RopeConstraint>(e);
        }
        if let Some(mt) = &s.material_texture {
            let _ = world.insert(
                e,
                (components::MaterialTexture {
                    path: mt.path.clone(),
                    normal_path: mt.normal_path.clone(),
                    metallic_roughness_path: mt.metallic_roughness_path.clone(),
                },),
            );
        } else {
            let _ = world.remove_one::<components::MaterialTexture>(e);
        }
        Some(e)
    }

    fn try_undo(&mut self, args: &mut UiFrameArgs<'_>) {
        let Some(cmd) = self.undo_stack.pop() else {
            args.error_log.push("[Undo] Nothing to undo.".to_string());
            return;
        };
        if let Some(e) = Self::apply_state(args.world, &cmd.before) {
            *args.selected_renderable = Some(e);
        }
        self.redo_stack.push(cmd);
        if self.redo_stack.len() > 256 {
            self.redo_stack.remove(0);
        }
    }

    fn try_redo(&mut self, args: &mut UiFrameArgs<'_>) {
        let Some(cmd) = self.redo_stack.pop() else {
            args.error_log.push("[Redo] Nothing to redo.".to_string());
            return;
        };
        if let Some(e) = Self::apply_state(args.world, &cmd.after) {
            *args.selected_renderable = Some(e);
        }
        self.undo_stack.push(cmd);
        if self.undo_stack.len() > 256 {
            self.undo_stack.remove(0);
        }
    }

    pub fn begin_project_hub(&mut self, window: &Window, elapsed: f32) -> Option<std::path::PathBuf> {
        self.apply_theme();
        let mut open_project: Option<std::path::PathBuf> = None;
        let mut hub_registry = ProjectRegistry::load();
        if window.scale_factor() >= 0.0 {
            let raw_input = self.egui_state.take_egui_input(window);
            let output = self.egui_ctx.run_ui(raw_input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let rect = ui.max_rect();
                    ui.painter().rect_filled(rect, 0.0, Color32::from_rgb(8, 10, 14));
                    ui.painter().rect_filled(
                        egui::Rect::from_min_max(rect.min, egui::pos2(rect.right(), rect.top() + rect.height() * 0.42)),
                        0.0,
                        Color32::from_rgb(15, 19, 28),
                    );
                    ui.painter().circle_filled(
                        rect.left_top() + egui::vec2(rect.width() * 0.18, rect.height() * 0.18),
                        rect.width() * 0.16,
                        Color32::from_rgba_unmultiplied(32, 64, 112, 36),
                    );
                    ui.painter().circle_filled(
                        rect.right_top() + egui::vec2(-rect.width() * 0.12, rect.height() * 0.08),
                        rect.width() * 0.12,
                        Color32::from_rgba_unmultiplied(118, 168, 255, 18),
                    );

                    ui.add_space(22.0);
                    ui.horizontal(|ui| {
                        ui.add_space(22.0);
                        ui.label(
                            RichText::new("TRINITY HUB")
                                .strong()
                                .size(15.0)
                                .color(Color32::from_rgb(224, 228, 235)),
                        );
                        ui.separator();
                        ui.label(
                            RichText::new(format!("Engine v{ENGINE_VERSION}"))
                                .small()
                                .color(Color32::from_rgb(132, 145, 164)),
                        );
                        ui.separator();
                        ui.label(
                            RichText::new(format!("Session {:.1}s", elapsed))
                                .small()
                                .color(Color32::from_rgb(132, 145, 164)),
                        );
                    });

                    ui.add_space(18.0);
                    ui.columns(2, |columns| {
                        columns[0].add_space(18.0);
                        egui::Frame::new()
                            .fill(Color32::from_rgba_unmultiplied(12, 16, 22, 220))
                            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(38, 48, 64)))
                            .corner_radius(14.0)
                            .inner_margin(egui::Margin::same(20))
                            .show(&mut columns[0], |ui| {
                                ui.set_min_height((rect.height() - 110.0).max(520.0));
                                ui.horizontal(|ui| {
                                    draw_triangle_logo(ui.painter(), ui.min_rect().min + egui::vec2(68.0, 62.0), 44.0);
                                    ui.add_space(84.0);
                                    ui.vertical(|ui| {
                                        ui.label(
                                            RichText::new("Project hub first")
                                                .size(28.0)
                                                .strong()
                                                .color(Color32::from_rgb(232, 236, 242)),
                                        );
                                        ui.label(
                                            RichText::new(
                                                "Splash screen, then the hub, then the editor after project creation or opening. Project management stays inside the hub before the editor opens.",
                                            )
                                            .size(14.0)
                                            .color(Color32::from_rgb(154, 165, 180)),
                                        );
                                    });
                                });
                                ui.add_space(20.0);
                                for line in [
                                    "Dark launcher shell with a cleaner studio-style first impression.",
                                    "Recent projects are presented as cards with version visibility and direct open actions.",
                                    "Blank project creation sets up Content, scenes, prefabs, and cache, then opens the redesigned editor.",
                                ] {
                                    egui::Frame::new()
                                        .fill(Color32::from_rgb(17, 22, 30))
                                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(35, 45, 58)))
                                        .corner_radius(10.0)
                                        .inner_margin(egui::Margin::same(12))
                                        .show(ui, |ui| {
                                            ui.label(
                                                RichText::new(line)
                                                    .size(13.5)
                                                    .color(Color32::from_rgb(204, 210, 220)),
                                            );
                                        });
                                    ui.add_space(8.0);
                                }
                                ui.add_space(10.0);
                                ui.horizontal(|ui| {
                                    for (title, value) in [
                                        ("Recent", hub_registry.projects.len().to_string()),
                                        ("Flow", "Splash > Hub > Editor".to_string()),
                                        ("Layouts", format!("{:?}", editor_persist::trinity_data_dir())),
                                    ] {
                                        egui::Frame::new()
                                            .fill(Color32::from_rgb(12, 18, 25))
                                            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(34, 44, 58)))
                                            .corner_radius(999.0)
                                            .inner_margin(egui::Margin::symmetric(12, 8))
                                            .show(ui, |ui| {
                                                ui.label(
                                                    RichText::new(format!("{title}: {value}"))
                                                        .small()
                                                        .color(Color32::from_rgb(176, 188, 204)),
                                                );
                                            });
                                    }
                                });
                            });

                        columns[1].add_space(18.0);
                        columns[1].vertical(|ui| {
                            egui::Frame::new()
                                .fill(Color32::from_rgba_unmultiplied(10, 13, 18, 238))
                                .stroke(egui::Stroke::new(1.0, Color32::from_rgb(42, 48, 57)))
                                .corner_radius(14.0)
                                .inner_margin(egui::Margin::same(18))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new("Projects")
                                                .size(20.0)
                                                .strong()
                                                .color(Color32::from_rgb(228, 231, 236)),
                                        );
                                        ui.separator();
                                        ui.label(
                                            RichText::new("Open, register, or create")
                                                .small()
                                                .color(Color32::from_rgb(132, 145, 164)),
                                        );
                                    });
                                    ui.add_space(12.0);

                                    egui::Frame::new()
                                        .fill(Color32::from_rgb(14, 18, 24))
                                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(36, 43, 54)))
                                        .corner_radius(10.0)
                                        .inner_margin(egui::Margin::same(14))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.label(RichText::new("Recent Projects").strong());
                                                ui.separator();
                                                ui.label(
                                                    RichText::new(format!("{} saved", hub_registry.projects.len()))
                                                        .small()
                                                        .color(Color32::from_rgb(130, 140, 156)),
                                                );
                                            });
                                            ui.add_space(8.0);
                                            egui::ScrollArea::vertical().max_height(250.0).show(ui, |ui| {
                                                if hub_registry.projects.is_empty() {
                                                    ui.label(
                                                        RichText::new("No saved projects yet. Create one below or register an existing folder.")
                                                            .color(Color32::from_rgb(165, 172, 182)),
                                                    );
                                                } else {
                                                    let mut remove_idx: Option<usize> = None;
                                                    for (i, p) in hub_registry.projects.iter().enumerate() {
                                                        let mismatch = !p.engine_version_at_last_open.is_empty()
                                                            && p.engine_version_at_last_open != ENGINE_VERSION;
                                                        egui::Frame::new()
                                                            .fill(Color32::from_rgb(18, 23, 31))
                                                            .stroke(egui::Stroke::new(
                                                                1.0,
                                                                if mismatch {
                                                                    Color32::from_rgb(110, 76, 34)
                                                                } else {
                                                                    Color32::from_rgb(34, 44, 58)
                                                                },
                                                            ))
                                                            .corner_radius(10.0)
                                                            .inner_margin(egui::Margin::same(12))
                                                            .show(ui, |ui| {
                                                                ui.horizontal(|ui| {
                                                                    ui.vertical(|ui| {
                                                                        ui.label(
                                                                            RichText::new(&p.name)
                                                                                .strong()
                                                                                .color(Color32::from_rgb(226, 230, 236)),
                                                                        );
                                                                        ui.label(
                                                                            RichText::new(&p.path)
                                                                                .small()
                                                                                .color(Color32::from_rgb(133, 145, 160)),
                                                                        );
                                                                    });
                                                                    ui.with_layout(
                                                                        egui::Layout::right_to_left(egui::Align::Center),
                                                                        |ui| {
                                                                            if ui.button("Remove").clicked() {
                                                                                remove_idx = Some(i);
                                                                            }
                                                                            if ui.button("Open").clicked() {
                                                                                open_project = Some(std::path::PathBuf::from(&p.path));
                                                                            }
                                                                        },
                                                                    );
                                                                });
                                                                ui.add_space(6.0);
                                                                ui.horizontal_wrapped(|ui| {
                                                                    let last_engine = if p.engine_version_at_last_open.is_empty() {
                                                                        "unknown".to_string()
                                                                    } else {
                                                                        p.engine_version_at_last_open.clone()
                                                                    };
                                                                    ui.label(
                                                                        RichText::new(format!("Last engine: v{last_engine}"))
                                                                            .small()
                                                                            .color(Color32::from_rgb(157, 167, 180)),
                                                                    );
                                                                    if mismatch {
                                                                        ui.separator();
                                                                        ui.label(
                                                                            RichText::new("Version differs from current build")
                                                                                .small()
                                                                                .color(Color32::from_rgb(244, 184, 102)),
                                                                        );
                                                                    }
                                                                });
                                                            });
                                                        ui.add_space(8.0);
                                                    }
                                                    if let Some(i) = remove_idx {
                                                        hub_registry.remove_index(i);
                                                        hub_registry.save();
                                                    }
                                                }
                                            });
                                        });

                                    ui.add_space(12.0);
                                    egui::Frame::new()
                                        .fill(Color32::from_rgb(14, 18, 24))
                                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(36, 43, 54)))
                                        .corner_radius(10.0)
                                        .inner_margin(egui::Margin::same(14))
                                        .show(ui, |ui| {
                                            ui.label(RichText::new("Register Existing Project").strong());
                                            ui.add_space(8.0);
                                            ui.add(
                                                egui::TextEdit::singleline(&mut self.hub_add_path)
                                                    .desired_width(f32::INFINITY)
                                                    .hint_text("C:\\Dev\\MyGame or relative path"),
                                            );
                                            ui.add_space(8.0);
                                            if ui.button("Register and Open").clicked() {
                                                let p = std::path::PathBuf::from(self.hub_add_path.trim());
                                                if p.is_dir() {
                                                    hub_registry.upsert_opened(&p, None, ENGINE_VERSION);
                                                    hub_registry.save();
                                                    open_project = Some(p);
                                                }
                                            }
                                        });

                                    ui.add_space(12.0);
                                    egui::Frame::new()
                                        .fill(Color32::from_rgb(14, 18, 24))
                                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(36, 43, 54)))
                                        .corner_radius(10.0)
                                        .inner_margin(egui::Margin::same(14))
                                        .show(ui, |ui| {
                                            ui.label(RichText::new("Create Blank Project").strong());
                                            ui.add_space(8.0);
                                            ui.add(
                                                egui::TextEdit::singleline(&mut self.hub_new_project_name)
                                                    .desired_width(f32::INFINITY)
                                                    .hint_text("Project name"),
                                            );
                                            ui.add_space(8.0);
                                            ui.label(
                                                RichText::new("Creates Content, scenes, prefabs, and cache in Documents/TrinityProjects.")
                                                    .small()
                                                    .color(Color32::from_rgb(135, 146, 162)),
                                            );
                                            ui.add_space(10.0);
                                            if ui.button("Create Project and Open Editor").clicked() {
                                                let name = self.hub_new_project_name.trim();
                                                if !name.is_empty() {
                                                    if let Some(home) = std::env::var_os("USERPROFILE") {
                                                        let root = std::path::PathBuf::from(home)
                                                            .join("Documents")
                                                            .join("TrinityProjects")
                                                            .join(name);
                                                        let _ = std::fs::create_dir_all(root.join("Content/Meshes"));
                                                        let _ = std::fs::create_dir_all(root.join("Content/Textures"));
                                                        let _ = std::fs::create_dir_all(root.join("Content/Scripts"));
                                                        let _ = std::fs::create_dir_all(root.join("Content/Prefabs"));
                                                        let _ = std::fs::create_dir_all(root.join(crate::scene::SCENE_DIR));
                                                        let _ = std::fs::create_dir_all(root.join(".trinity/cache/thumbnails"));
                                                        let scene = root.join(crate::scene::SCENE_DIR).join("main.scene");
                                                        if !scene.exists() {
                                                            let _ = std::fs::write(&scene, "");
                                                        }
                                                        hub_registry.upsert_opened(&root, Some(name), ENGINE_VERSION);
                                                        hub_registry.save();
                                                        open_project = Some(root);
                                                    }
                                                }
                                            }
                                        });

                                    ui.add_space(12.0);
                                    ui.horizontal(|ui| {
                                        if ui.button("Open current working folder").clicked() {
                                            if let Ok(p) = std::env::current_dir() {
                                                hub_registry.upsert_opened(&p, None, ENGINE_VERSION);
                                                hub_registry.save();
                                                open_project = Some(p);
                                            }
                                        }
                                        if ui.button("Reveal in Explorer").clicked() {
                                            let _ = open_external_editor("explorer \"{file}\"", ".");
                                        }
                                    });
                                });
                        });
                    });
                });
            });
            self.egui_state
                .handle_platform_output(window, output.platform_output);
            let paint_jobs = self
                .egui_ctx
                .tessellate(output.shapes, output.pixels_per_point);
            let size = window.inner_size();
            self.pending = Some(PreparedUi {
                paint_jobs,
                textures_delta: output.textures_delta,
                screen_desc: egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [size.width, size.height],
                    pixels_per_point: output.pixels_per_point,
                },
            });
            if let Some(ref p) = open_project {
                hub_registry.upsert_opened(p, None, ENGINE_VERSION);
                hub_registry.save();
            }
            return open_project;
        }
        let raw_input = self.egui_state.take_egui_input(window);
        let output = self.egui_ctx.run_ui(raw_input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let rect = ui.max_rect();
                ui.painter().rect_filled(rect, 0.0, Color32::from_rgb(10, 10, 12));
                draw_triangle_logo(ui.painter(), rect.center_top() + egui::vec2(0.0, 96.0), 88.0);
                ui.add_space(190.0);
                ui.vertical_centered(|ui| {
                    ui.heading(RichText::new("TrinityEngine Hub").color(Color32::from_rgb(214, 216, 220)));
                    ui.label(
                        RichText::new(format!("Engine v{ENGINE_VERSION} Â· Integrated GPU: use editor toolbar preset if this PC struggles."))
                            .color(Color32::from_rgb(150, 155, 164)),
                    );
                    ui.add_space(16.0);
                    ui.set_max_width(720.0);
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.label(RichText::new("Recent projects").strong());
                        ui.add_space(6.0);
                        egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                            if hub_registry.projects.is_empty() {
                                ui.label("No saved projects yet â€” add or create one below.");
                            } else {
                                let mut remove_idx: Option<usize> = None;
                                for (i, p) in hub_registry.projects.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        ui.label(RichText::new(&p.name).strong());
                                        let ev = if p.engine_version_at_last_open.is_empty() {
                                            "â€”".to_string()
                                        } else {
                                            p.engine_version_at_last_open.clone()
                                        };
                                        ui.label(
                                            RichText::new(format!("opened with v{ev}"))
                                                .small()
                                                .color(Color32::from_rgb(130, 140, 155)),
                                        );
                                        if !p.engine_version_at_last_open.is_empty()
                                            && p.engine_version_at_last_open != ENGINE_VERSION
                                        {
                                            ui.label(
                                                RichText::new("different engine")
                                                    .small()
                                                    .color(Color32::from_rgb(255, 170, 90)),
                                            );
                                        }
                                        ui.label(RichText::new(&p.path).small().italics());
                                        if ui.button("Open / Preview editor").clicked() {
                                            open_project = Some(std::path::PathBuf::from(&p.path));
                                        }
                                        if ui.button("Remove from list").clicked() {
                                            remove_idx = Some(i);
                                        }
                                    });
                                }
                                if let Some(i) = remove_idx {
                                    hub_registry.remove_index(i);
                                    hub_registry.save();
                                }
                            }
                        });
                    });
                    ui.add_space(10.0);
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.label("Add existing project folder");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.hub_add_path)
                                    .desired_width(420.0)
                                    .hint_text("C:\\Dev\\MyGame or relative path"),
                            );
                            if ui.button("Register & open").clicked() {
                                let p = std::path::PathBuf::from(self.hub_add_path.trim());
                                if p.is_dir() {
                                    hub_registry.upsert_opened(&p, None, ENGINE_VERSION);
                                    hub_registry.save();
                                    open_project = Some(p);
                                }
                            }
                        });
                    });
                    ui.add_space(8.0);
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.label("Create blank project");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.hub_new_project_name)
                                    .desired_width(220.0)
                                    .hint_text("Project name"),
                            );
                            if ui.button("Create under Documents/TrinityProjects").clicked() {
                                let name = self.hub_new_project_name.trim();
                                if !name.is_empty() {
                                    if let Some(home) = std::env::var_os("USERPROFILE") {
                                        let root = std::path::PathBuf::from(home)
                                            .join("Documents")
                                            .join("TrinityProjects")
                                            .join(name);
                                        let _ = std::fs::create_dir_all(root.join("Content/Meshes"));
                                        let _ = std::fs::create_dir_all(root.join("Content/Textures"));
                                        let _ = std::fs::create_dir_all(root.join("Content/Scripts"));
                                        let _ = std::fs::create_dir_all(root.join("Content/Prefabs"));
                                        let _ = std::fs::create_dir_all(root.join(crate::scene::SCENE_DIR));
                                        let _ = std::fs::create_dir_all(root.join(".trinity/cache/thumbnails"));
                                        let scene = root.join(crate::scene::SCENE_DIR).join("main.scene");
                                        if !scene.exists() {
                                            let _ = std::fs::write(&scene, "");
                                        }
                                        hub_registry.upsert_opened(&root, Some(name), ENGINE_VERSION);
                                        hub_registry.save();
                                        open_project = Some(root);
                                    }
                                }
                            }
                        });
                    });
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        if ui.button("Open current working folder").clicked() {
                            if let Ok(p) = std::env::current_dir() {
                                hub_registry.upsert_opened(&p, None, ENGINE_VERSION);
                                hub_registry.save();
                                open_project = Some(p);
                            }
                        }
                        if ui.button("Reveal in Explorer").clicked() {
                            let _ = open_external_editor("explorer \"{file}\"", ".");
                        }
                    });
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(format!(
                            "Session {:.1}s Â· layouts save to {:?}",
                            elapsed,
                            editor_persist::trinity_data_dir()
                        ))
                        .small(),
                    );
                });
            });
        });
        self.egui_state
            .handle_platform_output(window, output.platform_output);
        let paint_jobs = self
            .egui_ctx
            .tessellate(output.shapes, output.pixels_per_point);
        let size = window.inner_size();
        self.pending = Some(PreparedUi {
            paint_jobs,
            textures_delta: output.textures_delta,
            screen_desc: egui_wgpu::ScreenDescriptor {
                size_in_pixels: [size.width, size.height],
                pixels_per_point: output.pixels_per_point,
            },
        });
        if let Some(ref p) = open_project {
            hub_registry.upsert_opened(p, None, ENGINE_VERSION);
            hub_registry.save();
        }
        open_project
    }

    pub fn begin_editor_loading(&mut self, window: &Window, elapsed: f32) {
        self.apply_theme();
        if window.scale_factor() >= 0.0 {
            const SPLASH_SECS: f32 = 0.90;
            ensure_splash_logo_texture(&self.egui_ctx, &mut self.splash_logo_texture);
            let raw_input = self.egui_state.take_egui_input(window);
            let output = self.egui_ctx.run_ui(raw_input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let rect = ui.max_rect();
                    ui.painter().rect_filled(rect, 0.0, Color32::from_rgb(6, 8, 12));
                    ui.painter().rect_filled(
                        egui::Rect::from_min_max(rect.min, egui::pos2(rect.right(), rect.top() + rect.height() * 0.56)),
                        0.0,
                        Color32::from_rgb(11, 16, 24),
                    );
                    ui.painter().circle_filled(
                        rect.center_top() + egui::vec2(0.0, rect.height() * 0.20),
                        rect.width() * 0.18,
                        Color32::from_rgba_unmultiplied(74, 126, 214, 24),
                    );
                    if let Some(tex) = &self.splash_logo_texture {
                        let logo_w = (rect.width() * 0.42).clamp(240.0, 580.0);
                        let logo_h = logo_w;
                        let logo_rect = egui::Rect::from_center_size(
                            rect.center_top() + egui::vec2(0.0, rect.height() * 0.22),
                            egui::vec2(logo_w, logo_h),
                        );
                        ui.painter().image(
                            tex.id(),
                            logo_rect,
                            egui::Rect::from_min_max(egui::Pos2::new(0.0, 0.0), egui::Pos2::new(1.0, 1.0)),
                            Color32::WHITE,
                        );
                    } else {
                        draw_triangle_logo(ui.painter(), rect.center_top() + egui::vec2(0.0, rect.height() * 0.18), 72.0);
                    }
                    ui.vertical_centered(|ui| {
                        ui.add_space(rect.height() * 0.44);
                        ui.heading(
                            RichText::new(format!("Trinity Editor v{ENGINE_VERSION}"))
                                .size(28.0)
                                .color(Color32::from_rgb(222, 226, 232)),
                        );
                        ui.label(
                            RichText::new("Preparing project workspace")
                                .size(14.0)
                                .color(Color32::from_rgb(150, 160, 175)),
                        );
                        ui.add_space(10.0);
                        egui::Frame::new()
                            .fill(Color32::from_rgb(10, 13, 18))
                            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(34, 44, 58)))
                            .corner_radius(10.0)
                            .inner_margin(egui::Margin::same(14))
                            .show(ui, |ui| {
                                ui.set_width(360.0);
                                ui.label(
                                    RichText::new("Loading")
                                        .small()
                                        .color(Color32::from_rgb(154, 166, 182)),
                                );
                                ui.add(
                                    egui::ProgressBar::new((elapsed / SPLASH_SECS).clamp(0.0, 1.0))
                                        .desired_width(332.0)
                                        .show_percentage(),
                                );
                            });
                    });
                });
            });
            self.egui_state
                .handle_platform_output(window, output.platform_output);
            let paint_jobs = self.egui_ctx.tessellate(output.shapes, output.pixels_per_point);
            let size = window.inner_size();
            self.pending = Some(PreparedUi {
                paint_jobs,
                textures_delta: output.textures_delta,
                screen_desc: egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [size.width, size.height],
                    pixels_per_point: output.pixels_per_point,
                },
            });
            return;
        }
        const SPLASH_SECS: f32 = 0.85;
        ensure_splash_logo_texture(&self.egui_ctx, &mut self.splash_logo_texture);
        let raw_input = self.egui_state.take_egui_input(window);
        let output = self.egui_ctx.run_ui(raw_input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let rect = ui.max_rect();
                ui.painter().rect_filled(rect, 0.0, Color32::from_rgb(8, 8, 10));
                if let Some(tex) = &self.splash_logo_texture {
                    let logo_w = (rect.width() * 0.46).clamp(240.0, 620.0);
                    let logo_h = logo_w;
                    let logo_rect = egui::Rect::from_center_size(
                        rect.center_top() + egui::vec2(0.0, rect.height() * 0.24),
                        egui::vec2(logo_w, logo_h),
                    );
                    ui.painter().image(
                        tex.id(),
                        logo_rect,
                        egui::Rect::from_min_max(egui::Pos2::new(0.0, 0.0), egui::Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );
                } else {
                    draw_triangle_logo(ui.painter(), rect.center_top() + egui::vec2(0.0, rect.height() * 0.18), 72.0);
                }
                ui.vertical_centered(|ui| {
                    ui.add_space(rect.height() * 0.38);
                    ui.heading(
                        RichText::new(format!("Trinity Editor v{ENGINE_VERSION}"))
                            .color(Color32::from_rgb(210, 212, 218)),
                    );
                    ui.label(
                        RichText::new("Loading workspaceâ€¦")
                            .color(Color32::from_rgb(150, 155, 165)),
                    );
                    ui.add_space(10.0);
                    ui.add(
                        egui::ProgressBar::new((elapsed / SPLASH_SECS).clamp(0.0, 1.0))
                            .desired_width(320.0)
                            .show_percentage(),
                    );
                });
            });
        });
        self.egui_state
            .handle_platform_output(window, output.platform_output);
        let paint_jobs = self.egui_ctx.tessellate(output.shapes, output.pixels_per_point);
        let size = window.inner_size();
        self.pending = Some(PreparedUi {
            paint_jobs,
            textures_delta: output.textures_delta,
            screen_desc: egui_wgpu::ScreenDescriptor {
                size_in_pixels: [size.width, size.height],
                pixels_per_point: output.pixels_per_point,
            },
        });
    }

    pub fn push_toast(&mut self, message: String, now_seconds: f32) {
        self.toasts.push((message, now_seconds + 3.0));
    }

    pub fn mark_content_dirty(&mut self) {
        self.asset_index_dirty = true;
    }

    pub fn mark_icons_dirty(&mut self) {
        self.icon_registry_dirty = true;
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
        if self.scene_picker_choice.is_empty() {
            self.scene_picker_choice = args.scene_path.clone();
        }
        if self.asset_index_dirty {
            self.asset_db.scan_content_root("Content");
            self.asset_index_dirty = false;
        }
        if self.icon_registry_dirty {
            self.icon_registry.load_from_dir("assets/icons");
            refresh_icon_textures(&mut self.icon_texture_cache, &self.icon_registry, &self.egui_ctx);
            apply_icon_alias_fallbacks(&mut self.icon_texture_cache);
            self.icon_registry_dirty = false;
        }
        if let Some(tex_id) = self.scene_texture_id {
            self.egui_renderer.update_egui_texture_from_wgpu_texture(
                &args.renderer.device,
                args.renderer.scene_color_view(),
                wgpu::FilterMode::Linear,
                tex_id,
            );
        } else {
            let id = self.egui_renderer.register_native_texture(
                &args.renderer.device,
                args.renderer.scene_color_view(),
                wgpu::FilterMode::Linear,
            );
            self.scene_texture_id = Some(id);
        }
        let before_state = self.capture_selected_state(args.world, *args.selected_renderable);
        let mut undo_requested = false;
        let mut redo_requested = false;
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
                &mut self.gizmo_space,
                &mut self.gizmo_axis_lock,
                &mut self.terrain_mode,
                &mut self.terrain_brush_mode,
                &mut self.terrain_brush_radius,
                &mut self.snap_enabled,
                &mut self.snap_translate,
                &mut self.snap_rotate_deg,
                &mut self.snap_scale,
                &mut self.texture_thumbnail_cache,
                &mut self.mesh_thumbnail_color_cache,
                self.scene_texture_id,
                &self.asset_db,
                &self.icon_registry,
                &mut self.dock_state,
                &mut self.workspace_preset,
                &mut self.asset_search,
                &mut self.asset_kind_filter,
                &mut self.asset_sort_desc,
                &self.icon_texture_cache,
                &mut self.console_filter,
                &mut self.show_material_editor,
                &mut self.show_foliage_editor,
                &mut self.show_icon_debug,
                &mut self.show_perf_safety_check,
                &mut self.show_scene_manager,
                &mut self.scene_picker_choice,
                &mut self.scene_create_name,
                &mut self.scene_rename_name,
                &mut self.scene_duplicate_name,
                &mut undo_requested,
                &mut redo_requested,
            );
        });
        if undo_requested {
            self.try_undo(args);
        } else if redo_requested {
            self.try_redo(args);
        }
        let after_state = self.capture_selected_state(args.world, *args.selected_renderable);
        if !undo_requested && !redo_requested && !state_same(&before_state, &after_state) {
            self.undo_stack.push(EntityEditCommand {
                before: before_state,
                after: after_state,
            });
            if self.undo_stack.len() > 256 {
                self.undo_stack.remove(0);
            }
            self.redo_stack.clear();
        }
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
        style.spacing.item_spacing = egui::vec2(6.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 5.0);
        style.spacing.menu_margin = egui::Margin::symmetric(8, 6);
        style.visuals.window_fill = Color32::from_rgb(12, 14, 18);
        style.visuals.panel_fill = Color32::from_rgb(15, 18, 23);
        style.visuals.faint_bg_color = Color32::from_rgb(19, 22, 28);
        style.visuals.extreme_bg_color = Color32::from_rgb(7, 9, 13);
        style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(20, 24, 31);
        style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(27, 31, 39);
        style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(38, 45, 57);
        style.visuals.widgets.active.bg_fill = Color32::from_rgb(53, 63, 80);
        style.visuals.widgets.open.bg_fill = Color32::from_rgb(30, 35, 45);
        style.visuals.selection.bg_fill = Color32::from_rgb(64, 108, 176);
        style.visuals.window_stroke.color = Color32::from_rgb(38, 44, 54);
        style.visuals.widgets.noninteractive.corner_radius = 6.0.into();
        style.visuals.widgets.inactive.corner_radius = 6.0.into();
        style.visuals.widgets.hovered.corner_radius = 6.0.into();
        style.visuals.widgets.active.corner_radius = 6.0.into();
        style.visuals.menu_corner_radius = 8.0.into();
        style.visuals.window_corner_radius = 10.0.into();
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::new(22.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Body,
            egui::FontId::new(14.5, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Monospace,
            egui::FontId::new(13.0, egui::FontFamily::Monospace),
        );
        style.text_styles.insert(
            egui::TextStyle::Button,
            egui::FontId::new(13.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Small,
            egui::FontId::new(12.0, egui::FontFamily::Proportional),
        );
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
    gizmo_space: &mut GizmoSpace,
    gizmo_axis_lock: &mut Option<GizmoAxis>,
    terrain_mode: &mut bool,
    terrain_brush_mode: &mut components::TerrainBrushMode,
    terrain_brush_radius: &mut f32,
    snap_enabled: &mut bool,
    snap_translate: &mut f32,
    snap_rotate_deg: &mut f32,
    snap_scale: &mut f32,
    texture_thumbnail_cache: &mut HashMap<String, egui::TextureHandle>,
    mesh_thumbnail_color_cache: &mut HashMap<String, Color32>,
    scene_texture_id: Option<egui::TextureId>,
    asset_db: &AssetMetadataDb,
    icon_registry: &IconRegistry,
    dock_state: &mut DockState<EditorDockTab>,
    workspace_preset: &mut String,
    asset_search: &mut String,
    asset_kind_filter: &mut String,
    asset_sort_desc: &mut bool,
    icon_texture_cache: &HashMap<String, egui::TextureHandle>,
    console_filter: &mut String,
    show_material_editor: &mut bool,
    show_foliage_editor: &mut bool,
    show_icon_debug: &mut bool,
    show_perf_safety_check: &mut bool,
    show_scene_manager: &mut bool,
    scene_picker_choice: &mut String,
    scene_create_name: &mut String,
    scene_rename_name: &mut String,
    scene_duplicate_name: &mut String,
    undo_requested: &mut bool,
    redo_requested: &mut bool,
) {
    ctx.input(|i| {
        let command = i.modifiers.command || i.modifiers.ctrl;
        if command && i.key_pressed(egui::Key::Z) && !i.modifiers.shift {
            *undo_requested = true;
        }
        if (command && i.key_pressed(egui::Key::Y))
            || (command && i.modifiers.shift && i.key_pressed(egui::Key::Z))
        {
            *redo_requested = true;
        }
        if i.key_pressed(egui::Key::W) {
            *gizmo_mode = GizmoMode::Move;
            if !*terrain_mode { *gizmo_axis_lock = None; }
        } else if i.key_pressed(egui::Key::E) {
            *gizmo_mode = GizmoMode::Rotate;
            if !*terrain_mode { *gizmo_axis_lock = None; }
        } else if i.key_pressed(egui::Key::R) {
            *gizmo_mode = GizmoMode::Scale;
            if !*terrain_mode { *gizmo_axis_lock = None; }
        }
        if i.key_pressed(egui::Key::X) && !command && !*terrain_mode {
            *gizmo_axis_lock = match *gizmo_axis_lock {
                Some(GizmoAxis::X) => None,
                _ => Some(GizmoAxis::X),
            };
        }
        if i.key_pressed(egui::Key::Y) && !command && !*terrain_mode {
            *gizmo_axis_lock = match *gizmo_axis_lock {
                Some(GizmoAxis::Y) => None,
                _ => Some(GizmoAxis::Y),
            };
        }
        if i.key_pressed(egui::Key::Z) && !command && !*terrain_mode {
            *gizmo_axis_lock = match *gizmo_axis_lock {
                Some(GizmoAxis::Z) => None,
                _ => Some(GizmoAxis::Z),
            };
        }
        if i.key_pressed(egui::Key::T) {
            *terrain_mode = !*terrain_mode;
            if *terrain_mode { *gizmo_axis_lock = None; }
        }
        if *terrain_mode {
            if i.key_pressed(egui::Key::Num1) { *terrain_brush_mode = components::TerrainBrushMode::Raise; }
            if i.key_pressed(egui::Key::Num2) { *terrain_brush_mode = components::TerrainBrushMode::Lower; }
            if i.key_pressed(egui::Key::Num3) { *terrain_brush_mode = components::TerrainBrushMode::Smooth; }
            if i.key_pressed(egui::Key::Num4) { *terrain_brush_mode = components::TerrainBrushMode::Flatten; }
            if i.key_pressed(egui::Key::Num5) { *terrain_brush_mode = components::TerrainBrushMode::Paint; }
            if i.key_pressed(egui::Key::Num6) { *terrain_brush_mode = components::TerrainBrushMode::Foliage; }
            if i.key_pressed(egui::Key::OpenBracket) {
                *terrain_brush_radius = (*terrain_brush_radius - 0.5).max(0.5);
            }
            if i.key_pressed(egui::Key::CloseBracket) {
                *terrain_brush_radius = (*terrain_brush_radius + 0.5).min(50.0);
            }
        }
    });

    // If the Content Browser double-clicked a Lua script, open the Script
    // Editor tab and hand the file path over to it.
    let script_open_request: Option<String> = ctx
        .data_mut(|d| d.get_temp(egui::Id::new("script_editor_open")));
    if let Some(path) = script_open_request {
        ctx.data_mut(|d| d.remove_temp::<String>(egui::Id::new("script_editor_open")));
        match dock_state.find_tab_from(|t| *t == EditorDockTab::ScriptEditor) {
            Some(tab_path) => {
                let _ = dock_state.set_active_tab(tab_path);
                dock_state.set_focused_node_and_surface(tab_path.node_path());
            }
            None => {
                dock_state.push_to_first_leaf(EditorDockTab::ScriptEditor);
            }
        }
        let _ = path;
    }

    if !(args.settings.runtime.legacy_editor_ui || std::env::var("TRINITY_LEGACY_UI").is_ok()) {
        if args.available_scene_paths.iter().all(|p| p != scene_picker_choice) || scene_picker_choice.is_empty() {
            *scene_picker_choice = args.scene_path.clone();
        }
        let project_name = std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Untitled Project".to_string());
        let selected_label = args
            .selected_renderable
            .map(|e| format!("{:?}", e))
            .unwrap_or_else(|| "None".to_string());

        struct Viewer<'a, 'b> {
            args: &'a mut UiFrameArgs<'b>,
            texture_dragging: &'a mut Option<String>,
            texture_selected: &'a mut Option<String>,
            mesh_selected: &'a mut Option<String>,
            content_new_folder: &'a mut String,
            content_new_file: &'a mut String,
            gizmo_mode: &'a mut GizmoMode,
            gizmo_drag: &'a mut Option<GizmoDragState>,
            gizmo_space: GizmoSpace,
            gizmo_axis_lock: &'a mut Option<GizmoAxis>,
            terrain_mode: &'a mut bool,
            terrain_brush_mode: &'a mut components::TerrainBrushMode,
            terrain_brush_radius: &'a mut f32,
            snap_enabled: bool,
            snap_translate: f32,
            snap_rotate_deg: f32,
            snap_scale: f32,
            scene_texture_id: Option<egui::TextureId>,
            asset_db: &'a AssetMetadataDb,
            asset_search: &'a mut String,
            asset_kind_filter: &'a mut String,
            asset_sort_desc: &'a mut bool,
            texture_thumbnail_cache: &'a mut HashMap<String, egui::TextureHandle>,
            icon_texture_cache: &'a HashMap<String, egui::TextureHandle>,
            egui_ctx: &'a egui::Context,
            console_filter: &'a mut String,
            show_material_editor: &'a mut bool,
            show_foliage_editor: &'a mut bool,
        }
        impl<'a, 'b> TabViewer for Viewer<'a, 'b> {
            type Tab = EditorDockTab;

            fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
                match tab {
                    EditorDockTab::Viewport => "VIEWPORT".into(),
                    EditorDockTab::Outliner => "HIERARCHY".into(),
                    EditorDockTab::Details => "DETAILS".into(),
                    EditorDockTab::ContentBrowser => "CONTENT BROWSER".into(),
                    EditorDockTab::Profiler => "PROFILER".into(),
                    EditorDockTab::Console => "OUTPUT".into(),
                    EditorDockTab::ScriptEditor => "SCRIPT EDITOR".into(),
                    EditorDockTab::Levels => "LEVELS".into(),
                }
            }

            fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
                match tab {
                    EditorDockTab::Viewport => panels::render_viewport_panel(
                        ui,
                        self.args,
                        self.scene_texture_id,
                        self.gizmo_mode,
                        self.gizmo_drag,
                        self.gizmo_space,
                        *self.gizmo_axis_lock,
                        *self.terrain_mode,
                        *self.terrain_brush_mode,
                        *self.terrain_brush_radius,
                        self.snap_enabled,
                        self.snap_translate,
                        self.snap_rotate_deg,
                        self.snap_scale,
                    ),
                    EditorDockTab::Outliner => panels::render_outliner_panel(ui, self.args),
                    EditorDockTab::Details => panels::render_details_panel(ui, self.args),
                    EditorDockTab::ContentBrowser => panels::render_content_browser_panel(
                        ui,
                        self.asset_db,
                        self.asset_search,
                        self.asset_kind_filter,
                        self.asset_sort_desc,
                        self.texture_selected,
                        self.mesh_selected,
                        self.texture_dragging,
                        self.content_new_folder,
                        self.content_new_file,
                        self.texture_thumbnail_cache,
                        self.icon_texture_cache,
                        self.args.preferred_script_editor,
                        self.show_material_editor,
                        self.show_foliage_editor,
                        self.args.error_log,
                        self.egui_ctx,
                        self.args.bake_requested,
                    ),
                    EditorDockTab::Profiler => {
                        if let Some(text) = self.args.profiler.overlay_text() {
                            ui.label(text);
                        } else {
                            ui.label("Profiler data unavailable.");
                        }
                    }
                    EditorDockTab::Console => {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Output Log").strong());
                            ui.separator();
                            ui.label(RichText::new("Tagged runtime and editor messages").small().weak());
                        });
                        ui.horizontal(|ui| {
                            ui.label("Filter");
                            ui.add(
                                egui::TextEdit::singleline(self.console_filter)
                                    .desired_width(220.0)
                                    .hint_text("substring..."),
                            );
                            if ui.button("Clear").clicked() {
                                self.args.error_log.clear();
                            }
                            if ui.button("Copy").clicked() {
                                self.egui_ctx.copy_text(self.args.error_log.join("\n"));
                            }
                        });
                        ui.separator();
                        let filter = self.console_filter.to_ascii_lowercase();
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            let mut shown = 0usize;
                            for line in self.args.error_log.iter().rev() {
                                if !filter.is_empty() && !line.to_ascii_lowercase().contains(&filter) {
                                    continue;
                                }
                                shown += 1;
                                if shown > 400 {
                                    break;
                                }
                                egui::Frame::new()
                                    .fill(Color32::from_rgb(22, 22, 27))
                                    .corner_radius(3.0)
                                    .inner_margin(egui::Margin::same(4))
                                    .show(ui, |ui| {
                                        ui.label(console_line_rich(line));
                                    });
                                ui.add_space(1.0);
                            }
                            if shown == 0 {
                                ui.label(RichText::new("No messages.").italics().weak());
                            }
                        });
                    }
                    EditorDockTab::ScriptEditor => {
                        panels::render_script_editor_panel(ui, self.args, self.icon_texture_cache);
                    }
                    EditorDockTab::Levels => {
                        panels::render_levels_panel(ui, self.args);
                    }
                }
            }
        }

        egui::TopBottomPanel::top("editor_menu_bar_redesign")
            .exact_height(26.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("TRI").strong().color(Color32::from_rgb(236, 239, 243)));
                    ui.separator();
                    for menu in ["File", "Edit", "Window", "Tools", "Build", "Select", "Actor", "Help"] {
                        ui.menu_button(menu, |ui| {
                            if ui.button("Open Project Hub").clicked() {
                                *args.return_to_hub = true;
                                ui.close();
                            }
                            if ui.button("Save Layout").clicked() {
                                save_dock_layout(dock_state, workspace_preset);
                                ui.close();
                            }
                            if ui.button("Save Scene").clicked() {
                                let path = args.scene_path.clone();
                                match crate::scene::save_scene(&path, args.world) {
                                    Ok(()) => args.error_log.push(format!("[Scene] Saved {}", path)),
                                    Err(e) => args.error_log.push(format!("[Scene] Save failed: {}", e)),
                                }
                                ui.close();
                            }
                        });
                    }
                    ui.separator();
                    ui.label(RichText::new(project_name.clone()).small().color(Color32::from_rgb(150, 161, 178)));
                });
            });

        egui::TopBottomPanel::top("editor_toolbar_redesign")
            .exact_height(78.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    egui::Frame::new()
                        .fill(Color32::from_rgb(16, 19, 25))
                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(40, 48, 58)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new("Project").small().color(Color32::from_rgb(139, 150, 165)));
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(&project_name).strong());
                                    if ui.button("Hub").clicked() {
                                        *args.return_to_hub = true;
                                    }
                                    if ui.button("Save Layout").clicked() {
                                        save_dock_layout(dock_state, workspace_preset);
                                    }
                                });
                            });
                        });

                    egui::Frame::new()
                        .fill(Color32::from_rgb(16, 19, 25))
                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(40, 48, 58)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new("Simulation").small().color(Color32::from_rgb(139, 150, 165)));
                                ui.horizontal(|ui| {
                                    if ui.button(if *args.game_preview_mode { "Stop" } else { "Play" }).clicked() {
                                        *args.game_preview_mode = !*args.game_preview_mode;
                                    }
                                    if ui.button(if *args.sim_paused { "Resume" } else { "Pause" }).clicked() {
                                        *args.sim_paused = !*args.sim_paused;
                                    }
                                    if ui.button("Step").clicked() {
                                        *args.sim_step_once = true;
                                        *args.sim_paused = true;
                                    }
                                });
                            });
                        });

                    egui::Frame::new()
                        .fill(Color32::from_rgb(16, 19, 25))
                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(40, 48, 58)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new("Transform").small().color(Color32::from_rgb(139, 150, 165)));
                                ui.horizontal(|ui| {
                                    ui.selectable_value(gizmo_mode, GizmoMode::Move, "Move");
                                    ui.selectable_value(gizmo_mode, GizmoMode::Rotate, "Rotate");
                                    ui.selectable_value(gizmo_mode, GizmoMode::Scale, "Scale");
                                    ui.separator();
                                    ui.selectable_value(gizmo_space, GizmoSpace::World, "World");
                                    ui.selectable_value(gizmo_space, GizmoSpace::Local, "Local");
                                });
                            });
                        });

                    egui::Frame::new()
                        .fill(Color32::from_rgb(16, 19, 25))
                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(40, 48, 58)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new("Snap").small().color(Color32::from_rgb(139, 150, 165)));
                                ui.horizontal(|ui| {
                                    ui.checkbox(snap_enabled, "Enabled");
                                    ui.add(egui::DragValue::new(snap_translate).prefix("T ").speed(0.05).range(0.01..=5.0));
                                    ui.add(egui::DragValue::new(snap_rotate_deg).prefix("R ").speed(0.2).range(1.0..=45.0));
                                    ui.add(egui::DragValue::new(snap_scale).prefix("S ").speed(0.01).range(0.01..=1.0));
                                });
                            });
                        });

                    egui::Frame::new()
                        .fill(Color32::from_rgb(16, 19, 25))
                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(40, 48, 58)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new("Rendering").small().color(Color32::from_rgb(139, 150, 165)));
                                ui.horizontal(|ui| {
                                    preset_combo(ui, &mut args.settings.render.preset);
                                    ui.toggle_value(&mut args.renderer.features.bloom_enabled, "Bloom");
                                    ui.toggle_value(&mut args.renderer.features.ssao_enabled, "SSAO");
                                    ui.toggle_value(&mut args.renderer.features.volumetric_fog_enabled, "Fog");
                                    ui.toggle_value(&mut args.renderer.features.voxel_gi_enabled, "Voxel");
                                });
                            });
                        });

                    egui::Frame::new()
                        .fill(Color32::from_rgb(16, 19, 25))
                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(40, 48, 58)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new("Scene").small().color(Color32::from_rgb(139, 150, 165)));
                                ui.horizontal(|ui| {
                                    egui::ComboBox::from_id_salt("scene_picker_combo_redesign")
                                        .selected_text(scene_picker_choice.clone())
                                        .show_ui(ui, |ui| {
                                            for p in args.available_scene_paths {
                                                ui.selectable_value(scene_picker_choice, p.clone(), p);
                                            }
                                        });
                                    let same_scene = *scene_picker_choice == *args.scene_path;
                                    if ui.button("Load").clicked() {
                                        *args.requested_scene_switch = Some(scene_picker_choice.clone());
                                    }
                                    if ui
                                        .add_enabled(!same_scene, egui::Button::new("Make Current"))
                                        .clicked()
                                    {
                                        *args.requested_scene_switch = Some(scene_picker_choice.clone());
                                    }
                                    if ui.button("Manager").clicked() {
                                        *show_scene_manager = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(format!("Current {}", args.scene_path))
                                            .small()
                                            .color(Color32::from_rgb(170, 178, 189)),
                                    );
                                    if ui.button("Reload").clicked() {
                                        *args.requested_scene_switch = Some(args.scene_path.clone());
                                    }
                                });
                            });
                        });

                    egui::Frame::new()
                        .fill(Color32::from_rgb(16, 19, 25))
                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(40, 48, 58)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new("Windows").small().color(Color32::from_rgb(139, 150, 165)));
                                ui.horizontal_wrapped(|ui| {
                                    if ui.button("Project").clicked() {
                                        *show_project_launcher = true;
                                    }
                                    if ui.button("Materials").clicked() {
                                        *show_material_editor = true;
                                    }
                                    if ui.button("Foliage").clicked() {
                                        *show_foliage_editor = true;
                                    }
                                    if ui.button("Scene Mgr").clicked() {
                                        *show_scene_manager = true;
                                    }
                                    if ui.button("Perf").clicked() {
                                        *show_perf_safety_check = true;
                                    }
                                    if ui.button("Icons").clicked() {
                                        *show_icon_debug = true;
                                    }
                                });
                            });
                        });
                });
            });

        egui::SidePanel::left("mode_sidebar_redesign")
            .resizable(false)
            .exact_width(92.0)
            .show(ctx, |ui| {
                ui.add_space(10.0);
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("Modes").small().color(Color32::from_rgb(148, 158, 173)));
                    ui.add_space(8.0);
                    for mode in ["Place", "Select", "Landscape", "Foliage", "Paint", "Script", "FX"] {
                        let _ = ui.add_sized([72.0, 34.0], egui::Button::new(RichText::new(mode).strong()));
                    }
                });
            });

        egui::TopBottomPanel::bottom("status_bar_redesign")
            .exact_height(28.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Scene {}", args.scene_path)).small());
                    ui.separator();
                    ui.label(RichText::new(format!("Selected {}", selected_label)).small());
                    ui.separator();
                    ui.label(RichText::new(format!("Assets {}", asset_db.entries.len())).small());
                    ui.separator();
                    ui.label(RichText::new(format!("FPS Target {}", args.settings.runtime.max_fps)).small());
                    ui.separator();
                    ui.label(RichText::new(if *args.game_preview_mode { "Preview Running" } else { "Editing" }).small());
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Frame::new()
                .fill(Color32::from_rgb(12, 15, 20))
                .stroke(egui::Stroke::new(1.0, Color32::from_rgb(34, 40, 50)))
                .corner_radius(8.0)
                .inner_margin(egui::Margin::same(4))
                .show(ui, |ui| {
                    let mut viewer = Viewer {
                        args,
                        texture_dragging,
                        texture_selected,
                        mesh_selected,
                        content_new_folder,
                        content_new_file,
                        gizmo_mode,
                        gizmo_drag,
                        gizmo_space: *gizmo_space,
                        gizmo_axis_lock,
                        terrain_mode,
                        terrain_brush_mode,
                        terrain_brush_radius,
                        snap_enabled: *snap_enabled,
                        snap_translate: *snap_translate,
                        snap_rotate_deg: *snap_rotate_deg,
                        snap_scale: *snap_scale,
                        scene_texture_id,
                        asset_db,
                        asset_search,
                        asset_kind_filter,
                        asset_sort_desc,
                        texture_thumbnail_cache,
                        icon_texture_cache,
                        egui_ctx: ctx,
                        console_filter,
                        show_material_editor,
                        show_foliage_editor,
                    };
                    let mut dock_style = DockStyle::from_egui(ui.style().as_ref());
                    dock_style.tab_bar.fill_tab_bar = true;
                    DockArea::new(dock_state).style(dock_style).show_inside(ui, &mut viewer);
                });
        });

        if *show_material_editor {
            egui::Window::new("Material Editor")
                .default_pos([900.0, 110.0])
                .default_size([430.0, 420.0])
                .show(ctx, |ui| {
                    editor_tool_window_header(
                        ui,
                        "Material Editor",
                        "Instance-driven workflow mapped to your current render/material system.",
                    );
if let Some(entity) = args.selected_renderable.as_ref().copied() {
                        editor_tool_card(ui, "Selection", |ui| {
                            ui.label(RichText::new(format!("Selected {:?}", entity)).strong());
                        });
                        let mut pending_apply: Option<(String, Result<(), String>)> = None;
                        if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
                            editor_tool_card(ui, "Surface", |ui| {
                                ui.color_edit_button_rgb(&mut rend.color);
                                ui.add(egui::Slider::new(&mut rend.metallic, 0.0..=1.0).text("Metallic"));
                                ui.add(egui::Slider::new(&mut rend.roughness, 0.02..=1.0).text("Roughness"));
                                ui.add(egui::Slider::new(&mut rend.ao, 0.0..=1.0).text("AO"));
                            });
                            ui.add_space(8.0);
                            editor_tool_card(ui, "Material Instances", |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    for name in args.materials.instance_names() {
                                        if ui.button(&name).clicked() {
                                            pending_apply = Some((name.clone(), args.materials.apply_instance(&name, &mut rend)));
                                        }
                                    }
                                });
                            });
                        }
                        if let Some((name, res)) = pending_apply.take() {
                            if let Err(e) = res {
                                args.error_log.push(format!("[Material] {}", e));
                            } else if let Ok(extras) = args.materials.instance_extras(&name) {
                                let _ = args.world.insert(entity, (extras,));
                            }
                        }
                        ui.add_space(8.0);
                        editor_tool_card(ui, "Texture Slots", |ui| {
                            if let Ok(mt) = args.world.get::<&components::MaterialTexture>(entity) {
                                ui.monospace(format!("Albedo: {}", if mt.path.is_empty() { "<default>" } else { &mt.path }));
                                ui.monospace(format!("Normal: {}", if mt.normal_path.is_empty() { "<default flat>" } else { &mt.normal_path }));
                                ui.monospace(format!(
                                    "Metal/Rough: {}",
                                    if mt.metallic_roughness_path.is_empty() {
                                        "<default>"
                                    } else {
                                        &mt.metallic_roughness_path
                                    }
                                ));
                            } else {
                                ui.label("No texture overrides on selected entity.");
                            }
                        });
                    } else {
                        ui.label("Select a renderable entity to edit material data.");
                    }
                    if ui.button("Close").clicked() {
                        *show_material_editor = false;
                    }
                });
        }

        if *show_foliage_editor {
            egui::Window::new("Foliage Editor")
                .default_pos([900.0, 550.0])
                .default_size([430.0, 360.0])
                .show(ctx, |ui| {
                    editor_tool_window_header(
                        ui,
                        "Foliage",
                        "Brush-style tooling built on the foliage features your engine already has.",
                    );
                    let wx = args.terrain_cursor_x as f32 * args.terrain_world.cell_size - 32.0;
                    let wz = args.terrain_cursor_z as f32 * args.terrain_world.cell_size - 32.0;
                    editor_tool_card(ui, "Brush", |ui| {
                        ui.label(format!("Cursor world position: ({wx:.1}, {wz:.1})"));
                        let mut ring_radius = ui.data_mut(|d| d.get_temp::<f32>("foliage_ring_radius_window".into())).unwrap_or(4.0);
                        let mut ring_count = ui.data_mut(|d| d.get_temp::<u32>("foliage_ring_count_window".into())).unwrap_or(32);
                        let mut remove_radius = ui.data_mut(|d| d.get_temp::<f32>("foliage_remove_radius_window".into())).unwrap_or(4.5);
                        let mut tree_physics = ui.data_mut(|d| d.get_temp::<bool>("foliage_tree_physics_window".into())).unwrap_or(true);
                        ui.add(egui::Slider::new(&mut ring_radius, 1.0..=24.0).text("Brush radius"));
                        ui.add(egui::Slider::new(&mut ring_count, 4..=256).text("Instances"));
                        ui.add(egui::Slider::new(&mut remove_radius, 1.0..=30.0).text("Erase radius"));
                        ui.checkbox(&mut tree_physics, "Wind / physics");
                        ui.data_mut(|d| {
                            d.insert_temp("foliage_ring_radius_window".into(), ring_radius);
                            d.insert_temp("foliage_ring_count_window".into(), ring_count);
                            d.insert_temp("foliage_remove_radius_window".into(), remove_radius);
                            d.insert_temp("foliage_tree_physics_window".into(), tree_physics);
                        });
                        ui.horizontal_wrapped(|ui| {
                            if let Some(handle) = args.mesh_cache.get("meshes/cube.obj").copied() {
                                if ui.button("Paint Ring").clicked() {
                                    spawn_foliage_ring(args.world, handle, wx, wz, ring_radius, ring_count as usize, tree_physics);
                                }
                            }
                            if ui.button("Paint Patch").clicked() {
                                crate::editor::add_foliage_patch(args.world, args.meshes, args.mesh_cache);
                            }
                            if ui.button("Erase Near Cursor").clicked() {
                                let n = remove_nearby_foliage(args.world, wx, wz, remove_radius);
                                if n > 0 {
                                    args.error_log.push(format!("[Foliage] Removed {n} instances near cursor."));
                                }
                            }
                        });
                    });
                    editor_tool_card(ui, "Notes", |ui| {
                        ui.label("This window mirrors the dedicated foliage workspace in the references while staying inside Trinity's current foliage feature set.");
                    });
                    if ui.button("Close").clicked() {
                        *show_foliage_editor = false;
                    }
                });
        }

        if *show_icon_debug {
            const REQUIRED: &[&str] = &[
                "camera", "light", "sky", "sun", "player_start", "mesh", "prefab", "script",
                "material", "foliage", "file", "folder", "folder_open", "point_light", "fog", "volume",
            ];
            egui::Window::new("Icon Audit")
                .default_pos([600.0, 120.0])
                .default_size([420.0, 420.0])
                .show(ctx, |ui| {
                    editor_tool_window_header(ui, "Icon Audit", "Checks editor icon coverage against expected stems.");
                    for stem in REQUIRED {
                        let ok = icon_texture_cache.contains_key(*stem);
                        let color = if ok {
                            Color32::from_rgb(120, 210, 150)
                        } else {
                            Color32::from_rgb(235, 170, 120)
                        };
                        ui.colored_label(color, format!("{} {}", if ok { "OK" } else { "MISS" }, stem));
                    }
                    ui.separator();
                    if ui.button("Close").clicked() {
                        *show_icon_debug = false;
                    }
                });
        }

        if *show_perf_safety_check {
            let tier = args.settings.runtime.gpu_scalability_tier.to_ascii_lowercase();
            let mut cost_score = 0u32;
            if args.renderer.features.pcss_enabled {
                cost_score += 2;
            }
            if args.renderer.features.volumetric_fog_enabled {
                cost_score += 2;
            }
            if args.renderer.features.voxel_gi_enabled {
                cost_score += 3;
            }
            if args.renderer.features.ssao_enabled {
                cost_score += 1;
            }
            if args.renderer.features.bloom_enabled {
                cost_score += 1;
            }
            if args.renderer.features.shadow_resolution >= 4096 {
                cost_score += 2;
            } else if args.renderer.features.shadow_resolution >= 2048 {
                cost_score += 1;
            }
            if args.renderer.features.pcf_samples >= 16 {
                cost_score += 1;
            }
            let expected_tier = if cost_score >= 8 {
                "Desktop High-End"
            } else if cost_score >= 5 {
                "Mid Desktop / Strong Laptop"
            } else if cost_score >= 2 {
                "Balanced"
            } else {
                "Integrated / Entry"
            };
            let risky_on_low_tier = (tier == "auto" || tier == "low") && cost_score >= 5;
            let risky_on_balanced_tier = tier == "balanced" && cost_score >= 8;

            egui::Window::new("Performance Safety Check")
                .default_pos([760.0, 140.0])
                .default_size([460.0, 330.0])
                .show(ctx, |ui| {
                    editor_tool_window_header(ui, "Performance Safety Check", "Pre-flight estimate for the active rendering mix.");
                    editor_tool_card(ui, "Profile", |ui| {
                        ui.label(format!("GPU tier target: {}", args.settings.runtime.gpu_scalability_tier));
                        ui.label(format!("Feature cost score: {}", cost_score));
                        ui.label(format!("Expected hardware class: {}", expected_tier));
                    });
                    if risky_on_low_tier {
                        ui.colored_label(Color32::from_rgb(255, 180, 110), "Heavy for auto / low tier.");
                    } else if risky_on_balanced_tier {
                        ui.colored_label(Color32::from_rgb(255, 180, 110), "Heavy for balanced tier.");
                    } else {
                        ui.colored_label(Color32::from_rgb(120, 210, 150), "Looks safe for the selected target tier.");
                    }
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("Balanced").clicked() {
                            args.settings.render.preset = RenderPreset::Balanced;
                        }
                        if ui.button("Mobile").clicked() {
                            args.settings.render.preset = RenderPreset::Mobile;
                        }
                        if ui.button("Cinematic").clicked() {
                            args.settings.render.preset = RenderPreset::Cinematic;
                        }
                    });
                    if ui.button("Close").clicked() {
                        *show_perf_safety_check = false;
                    }
                });
        }

        if *show_scene_manager {
            egui::Window::new("Scene Manager")
                .default_pos([620.0, 120.0])
                .default_size([580.0, 430.0])
                .show(ctx, |ui| {
                    editor_tool_window_header(
                        ui,
                        "Scene Manager",
                        "Manage scene files and startup scene using your current engine scene system.",
                    );
                    editor_tool_card(ui, "Loaded Scene", |ui| {
                        ui.label(format!("Current loaded scene: {}", args.scene_path));
                        ui.separator();
                        ui.label(RichText::new("Available Scenes").small().strong());
                        egui::ScrollArea::vertical().max_height(90.0).show(ui, |ui| {
                            for p in args.available_scene_paths {
                                let current = p == args.scene_path.as_str();
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(current, p).clicked() {
                                        *scene_picker_choice = p.clone();
                                    }
                                    if ui.small_button("Load").clicked() {
                                        *scene_picker_choice = p.clone();
                                        *args.requested_scene_switch = Some(p.clone());
                                    }
                                });
                            }
                        });
                    });
                    ui.horizontal(|ui| {
                        ui.label("Startup");
                        egui::ComboBox::from_id_salt("startup_scene_combo_redesign")
                            .selected_text(args.settings.runtime.startup_scene_path.clone())
                            .show_ui(ui, |ui| {
                                for p in args.available_scene_paths {
                                    ui.selectable_value(&mut args.settings.runtime.startup_scene_path, p.clone(), p);
                                }
                            });
                        if ui.button("Save").clicked() {
                            match args.settings.save("engine_settings.toml") {
                                Ok(()) => args.error_log.push("[Scene] Saved startup scene to engine_settings.toml".to_string()),
                                Err(e) => args.error_log.push(format!("[Scene] Save failed: {}", e)),
                            }
                        }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Create");
                        ui.text_edit_singleline(scene_create_name);
                        if ui.button("Create Scene").clicked() {
                            let mut name = scene_create_name.trim().to_string();
                            if !name.ends_with(".scene") {
                                name.push_str(".scene");
                            }
                            let _ = fs::create_dir_all(crate::scene::SCENE_DIR);
                            let p = crate::scene::scene_path(&name);
                            if fs::write(&p, "").is_ok() {
                                *scene_picker_choice = p.clone();
                                *args.requested_scene_switch = Some(p);
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Duplicate");
                        ui.text_edit_singleline(scene_duplicate_name);
                        if ui.button("Duplicate Current").clicked() {
                            let src = args.scene_path.clone();
                            let mut dst_name = scene_duplicate_name.trim().to_string();
                            if !dst_name.ends_with(".scene") {
                                dst_name.push_str(".scene");
                            }
                            let dst = crate::scene::scene_path(&dst_name);
                            match fs::copy(&src, &dst) {
                                Ok(_) => args.error_log.push(format!("[Scene] Duplicated to {}", dst)),
                                Err(e) => args.error_log.push(format!("[Scene] Duplicate failed: {}", e)),
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Rename");
                        ui.text_edit_singleline(scene_rename_name);
                        if ui.button("Rename Current").clicked() {
                            let src = args.scene_path.clone();
                            let mut dst_name = scene_rename_name.trim().to_string();
                            if !dst_name.ends_with(".scene") {
                                dst_name.push_str(".scene");
                            }
                            let dst = crate::scene::scene_path(&dst_name);
                            match fs::rename(&src, &dst) {
                                Ok(_) => {
                                    *args.requested_scene_switch = Some(dst.clone());
                                    args.settings.runtime.startup_scene_path = dst;
                                }
                                Err(e) => args.error_log.push(format!("[Scene] Rename failed: {}", e)),
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Delete Current").clicked() {
                            let cur = args.scene_path.clone();
                            if cur.ends_with("main.scene") {
                                args.error_log.push("[Scene] Refused to delete main.scene from manager.".to_string());
                            } else {
                                match fs::remove_file(&cur) {
                                    Ok(()) => {
                                        args.error_log.push(format!("[Scene] Deleted {}", cur));
                                        *args.requested_scene_switch = Some(args.settings.runtime.startup_scene_path.clone());
                                    }
                                    Err(e) => args.error_log.push(format!("[Scene] Delete failed: {}", e)),
                                }
                            }
                        }
                        if ui.button("Close").clicked() {
                            *show_scene_manager = false;
                        }
                    });
                });
        }

        if *show_project_launcher {
            egui::Window::new("Project and Docs")
                .default_pos([360.0, 110.0])
                .default_size([700.0, 480.0])
                .show(ctx, |ui| {
                    editor_tool_window_header(ui, "Project and Docs", "Launcher-style utility window for project paths, editor integration, and engine docs.");
                    ui.columns(2, |columns| {
                        egui::Frame::new()
                            .fill(Color32::from_rgb(14, 17, 22))
                            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(33, 39, 49)))
                            .corner_radius(8.0)
                            .inner_margin(egui::Margin::same(10))
                            .show(&mut columns[0], |ui| {
                                ui.label(RichText::new("Project Tools").small().strong());
                                if ui.button("Open Current Project Folder").clicked() {
                                    let _ = open_external_editor("explorer \"{file}\"", ".");
                                }
                                if ui.button("Create New Project Folder In Documents").clicked() {
                                    if let Some(home) = std::env::var_os("USERPROFILE") {
                                        let p = std::path::PathBuf::from(home).join("Documents").join("TriengineProject");
                                        let _ = std::fs::create_dir_all(p.join("Content"));
                                        let _ = std::fs::create_dir_all(p.join(crate::scene::SCENE_DIR));
                                        let scene = p.join(crate::scene::SCENE_DIR).join("main.scene");
                                        if !scene.exists() {
                                            let _ = std::fs::write(&scene, "");
                                        }
                                        let _ = std::fs::write(
                                            p.join("engine_settings.toml"),
                                            std::fs::read_to_string("engine_settings.toml").unwrap_or_default(),
                                        );
                                        ui.ctx().copy_text(p.to_string_lossy().to_string());
                                    }
                                }
                                ui.separator();
                                ui.label(RichText::new("External Script Editor").small().strong());
                                ui.horizontal_wrapped(|ui| {
                                    if ui.button("VS Code").clicked() {
                                        *args.preferred_script_editor = "code -r \"{file}\"".to_string();
                                    }
                                    if ui.button("Notepad++").clicked() {
                                        *args.preferred_script_editor = "notepad++ \"{file}\"".to_string();
                                    }
                                    if ui.button("Rider").clicked() {
                                        *args.preferred_script_editor = "rider64 \"{file}\"".to_string();
                                    }
                                });
                                ui.text_edit_singleline(args.preferred_script_editor);
                                if ui.button("Test Open").clicked() {
                                    let test_file = format!("{}/player.lua", args.scripts_dir);
                                    if let Err(err) = open_external_editor(args.preferred_script_editor, &test_file) {
                                        args.error_log.push(format!("[EditorPicker] Test open failed: {}", err));
                                    }
                                }
                            });
                        egui::Frame::new()
                            .fill(Color32::from_rgb(14, 17, 22))
                            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(33, 39, 49)))
                            .corner_radius(8.0)
                            .inner_margin(egui::Margin::same(10))
                            .show(&mut columns[1], |ui| {
                                ui.label(RichText::new("Docs").small().strong());
                                ui.label("Open bundled guidance documents from the editor.");
                                if ui.button("Getting Started").clicked() {
                                    let _ = open_external_editor("cmd /C start \"\" \"{file}\"", "docs/GETTING_STARTED_GAME_CREATOR.md");
                                }
                                if ui.button("Materials and Textures").clicked() {
                                    let _ = open_external_editor("cmd /C start \"\" \"{file}\"", "docs/MATERIAL_TEXTURE_WORKFLOW.md");
                                }
                                if ui.button("Render and Lighting").clicked() {
                                    let _ = open_external_editor("cmd /C start \"\" \"{file}\"", "docs/RENDER_AND_LIGHTING_GUIDE.md");
                                }
                                ui.separator();
                                let startup_on = is_launch_on_startup_enabled();
                                ui.label(if startup_on { "Launch on startup: ON" } else { "Launch on startup: OFF" });
                                ui.horizontal(|ui| {
                                    if ui.button("Enable").clicked() {
                                        if let Err(err) = set_launch_on_startup(true) {
                                            args.error_log.push(format!("[Startup] {}", err));
                                        }
                                    }
                                    if ui.button("Disable").clicked() {
                                        if let Err(err) = set_launch_on_startup(false) {
                                            args.error_log.push(format!("[Startup] {}", err));
                                        }
                                    }
                                });
                            });
                    });
                    ui.separator();
                    if ui.button("Close").clicked() {
                        *show_project_launcher = false;
                    }
                });
        }

        egui::Window::new("World Settings")
            .default_pos([1090.0, 110.0])
            .default_size([270.0, 220.0])
            .resizable(false)
            .show(ctx, |ui| {
                editor_tool_window_header(ui, "World Settings", "Quick world context and scene state.");
                editor_tool_card(ui, "Context", |ui| {
                    ui.label(format!("Project: {}", project_name));
                    ui.label(format!("Scene: {}", args.scene_path));
                    ui.label(format!("Selected: {}", selected_label));
                    ui.label(format!("Icons indexed: {}", icon_registry.icons.len()));
                });
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Open Scene Manager").clicked() {
                        *show_scene_manager = true;
                    }
                    if ui.button("Project Hub").clicked() {
                        *args.return_to_hub = true;
                    }
                });
                editor_tool_card(ui, "Available Scenes", |ui| {
                    egui::ScrollArea::vertical().max_height(74.0).show(ui, |ui| {
                        for p in args.available_scene_paths.iter().take(8) {
                            let is_current = p == args.scene_path.as_str();
                            ui.horizontal(|ui| {
                                if ui.selectable_label(is_current, p).clicked() {
                                    *scene_picker_choice = p.clone();
                                }
                                if ui.small_button("Load").clicked() {
                                    *scene_picker_choice = p.clone();
                                    *args.requested_scene_switch = Some(p.clone());
                                }
                            });
                        }
                    });
                });
                editor_tool_card(ui, "Physics Runtime", |ui| {
                    ui.checkbox(&mut args.settings.runtime.physics_ccd_enabled, "CCD / substeps");
                    ui.label(RichText::new("Helps fast objects avoid tunneling through things. Heavier.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(&mut args.settings.runtime.physics_broadphase_enabled, "Broadphase");
                    ui.label(RichText::new("Reduces collision pair checks. Good to keep on.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(&mut args.settings.runtime.physics_friction_enabled, "Friction / sliding");
                    ui.label(RichText::new("Enables nicer wall and ground response.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(&mut args.settings.runtime.physics_sleeping_enabled, "Sleeping");
                    ui.label(RichText::new("Lets idle bodies stop updating.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(&mut args.settings.runtime.physics_collision_events_enabled, "Collision events");
                    ui.label(RichText::new("Enables enter, stay, and exit collision events for scripting.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(&mut args.settings.runtime.physics_position_correction_enabled, "Position correction");
                    ui.label(RichText::new("Pushes intersecting bodies apart after collisions.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(&mut args.settings.runtime.physics_smooth_foliage_motion, "Smooth foliage motion");
                    ui.label(RichText::new("Smooths foliage wind motion at extra runtime cost.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(&mut args.settings.runtime.physics_angular_dynamics_enabled, "Angular dynamics");
                    ui.label(RichText::new("Enables spinning bodies and torque behavior.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(
                        &mut args.settings.runtime.physics_advanced_constraints_enabled,
                        "Advanced constraints",
                    );
                    ui.label(RichText::new("Turns on the heavier joint and rope solving path.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(
                        &mut args.settings.runtime.physics_local_anchor_constraints_enabled,
                        "Local anchor joints",
                    );
                    ui.label(RichText::new("Uses local anchor offsets when solving constraints.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(
                        &mut args.settings.runtime.physics_3d_obb_contacts_enabled,
                        "3D OBB contacts",
                    );
                    ui.label(RichText::new("Enables heavier full 3D rotated-box contact testing.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(
                        &mut args.settings.runtime.physics_articulated_impulse_solver_enabled,
                        "Articulated impulse solver",
                    );
                    ui.label(RichText::new("Experimental articulated joint solver. Heavy and still incomplete.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(
                        &mut args.settings.runtime.physics_manifold_warm_start_enabled,
                        "Manifold warm start",
                    );
                    ui.label(RichText::new("Experimental cached contact warm starting for stability.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.checkbox(
                        &mut args.settings.runtime.physics_full_angular_3d_enabled,
                        "Full angular 3D",
                    );
                    ui.label(RichText::new("Experimental full 3-axis angular dynamics path.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.add(
                        egui::Slider::new(&mut args.settings.runtime.physics_max_substeps, 1..=12)
                            .text("Max substeps"),
                    );
                    ui.label(RichText::new("Higher values improve fast-object stability but use more CPU.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.add(
                        egui::Slider::new(&mut args.settings.runtime.physics_solver_iterations, 1..=12)
                            .text("Solver iterations"),
                    );
                    ui.label(RichText::new("More contact stability at higher CPU cost.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.add(
                        egui::Slider::new(&mut args.settings.runtime.physics_constraint_iterations, 1..=12)
                            .text("Constraint iterations"),
                    );
                    ui.label(RichText::new("More joint and rope stability at higher CPU cost.").small().color(Color32::from_rgb(126, 138, 154)));
                    ui.add(
                        egui::Slider::new(&mut args.settings.runtime.physics_broadphase_cell_size, 0.5..=8.0)
                            .text("Broadphase cell"),
                    );
                });
            });

        egui::Window::new("Widget Designer")
            .default_pos([108.0, 520.0])
            .default_size([380.0, 420.0])
            .collapsible(true)
            .show(ctx, |ui| {
                editor_tool_window_header(ui, "Widget Designer", "Author runtime HUD widgets driven by Lua values.");
                ui.horizontal(|ui| {
                    ui.label("Id");
                    ui.text_edit_singleline(widget_new_id);
                });
                ui.label(RichText::new("Type").small().strong());
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.selectable_value(widget_new_kind, UiWidgetKind::HealthBar, "Health");
                        ui.selectable_value(widget_new_kind, UiWidgetKind::Counter, "Counter");
                        ui.selectable_value(widget_new_kind, UiWidgetKind::Label, "Label");
                        ui.selectable_value(widget_new_kind, UiWidgetKind::Button, "Button");
                        ui.selectable_value(widget_new_kind, UiWidgetKind::Slider, "Slider");
                        ui.selectable_value(widget_new_kind, UiWidgetKind::Toggle, "Toggle");
                        ui.selectable_value(widget_new_kind, UiWidgetKind::Panel, "Panel");
                        ui.selectable_value(widget_new_kind, UiWidgetKind::ProgressRing, "Ring");
                        ui.selectable_value(widget_new_kind, UiWidgetKind::Meter, "Meter");
                    });
                });
                if ui.button("Add widget").clicked() && !widget_new_id.trim().is_empty() {
                    widget_specs.push(UiWidgetSpec {
                        id: widget_new_id.trim().to_string(),
                        kind: *widget_new_kind,
                        x: 30.0,
                        y: 90.0 + widget_specs.len() as f32 * 26.0,
                        w: 240.0,
                        h: match *widget_new_kind {
                            UiWidgetKind::Panel => 120.0,
                            UiWidgetKind::Meter => 28.0,
                            UiWidgetKind::ProgressRing => 64.0,
                            UiWidgetKind::Button => 32.0,
                            UiWidgetKind::Slider => 24.0,
                            UiWidgetKind::Toggle => 24.0,
                            _ => 24.0,
                        },
                        visible: true,
                        z_order: 0,
                        color: [1.0, 1.0, 1.0, 1.0],
                        bg_color: [0.0, 0.0, 0.0, 0.5],
                        font_size: 14.0,
                        anchor: UiAnchor::TopLeft,
                    });
                }
                ui.separator();
                ui.label(RichText::new("Placed Widgets").small().strong());
                let mut rm: Option<usize> = None;
                for (i, w) in widget_specs.iter_mut().enumerate() {
                    egui::Frame::new()
                        .fill(if w.visible { Color32::from_rgb(14, 17, 22) } else { Color32::from_rgb(8, 10, 14) })
                        .stroke(egui::Stroke::new(1.0, if w.visible { Color32::from_rgb(33, 39, 49) } else { Color32::from_rgb(20, 22, 28) }))
                        .corner_radius(6.0)
                        .inner_margin(egui::Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.monospace(&w.id);
                                ui.separator();
                                let vis_resp = ui.selectable_label(w.visible, if w.visible { "V" } else { "-" });
                                if vis_resp.clicked() { w.visible = !w.visible; }
                                ui.label(RichText::new(format!("{:?}", w.kind)).small().weak());
                                if ui.button("x").clicked() { rm = Some(i); }
                            });
                            ui.horizontal(|ui| {
                                ui.add(egui::DragValue::new(&mut w.x).prefix("x ").speed(1.0));
                                ui.add(egui::DragValue::new(&mut w.y).prefix("y ").speed(1.0));
                                ui.add(egui::DragValue::new(&mut w.w).prefix("w ").speed(1.0).range(20.0..=800.0));
                                ui.add(egui::DragValue::new(&mut w.h).prefix("h ").speed(1.0).range(10.0..=400.0));
                            });
                            ui.collapsing(RichText::new("Style").small(), |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Size:");
                                    ui.add(egui::DragValue::new(&mut w.font_size).range(8.0..=48.0).suffix("pt"));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Anchor:");
                                    ui.selectable_value(&mut w.anchor, UiAnchor::TopLeft, "TL");
                                    ui.selectable_value(&mut w.anchor, UiAnchor::TopCenter, "TC");
                                    ui.selectable_value(&mut w.anchor, UiAnchor::TopRight, "TR");
                                    ui.selectable_value(&mut w.anchor, UiAnchor::CenterLeft, "CL");
                                    ui.selectable_value(&mut w.anchor, UiAnchor::Center, "C");
                                    ui.selectable_value(&mut w.anchor, UiAnchor::CenterRight, "CR");
                                    ui.selectable_value(&mut w.anchor, UiAnchor::BottomLeft, "BL");
                                    ui.selectable_value(&mut w.anchor, UiAnchor::BottomCenter, "BC");
                                    ui.selectable_value(&mut w.anchor, UiAnchor::BottomRight, "BR");
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Z:");
                                    ui.add(egui::DragValue::new(&mut w.z_order).range(-10..=10));
                                });
                                // Colour editors â€” show RGB sliders for text and background.
                                ui.horizontal(|ui| {
                                    ui.label("Text:");
                                    ui.add(egui::DragValue::new(&mut w.color[0]).prefix("R ").speed(0.01).range(0.0..=1.0));
                                    ui.add(egui::DragValue::new(&mut w.color[1]).prefix("G ").speed(0.01).range(0.0..=1.0));
                                    ui.add(egui::DragValue::new(&mut w.color[2]).prefix("B ").speed(0.01).range(0.0..=1.0));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Bg :");
                                    ui.add(egui::DragValue::new(&mut w.bg_color[0]).prefix("R ").speed(0.01).range(0.0..=1.0));
                                    ui.add(egui::DragValue::new(&mut w.bg_color[1]).prefix("G ").speed(0.01).range(0.0..=1.0));
                                    ui.add(egui::DragValue::new(&mut w.bg_color[2]).prefix("B ").speed(0.01).range(0.0..=1.0));
                                    ui.add(egui::DragValue::new(&mut w.bg_color[3]).prefix("A ").speed(0.01).range(0.0..=1.0));
                                });
                            });
                            // Drag-and-drop handle â€” click & drag to reposition.
                            let resp = ui.interact(ui.max_rect(), egui::Id::new(format!("widget_drag_{}", w.id)), egui::Sense::drag());
                            if resp.dragged() {
                                let delta = resp.drag_delta();
                                w.x += delta.x;
                                w.y += delta.y;
                            }
                        });
                    ui.add_space(2.0);
                }
                if let Some(i) = rm {
                    widget_specs.remove(i);
                }
                ui.separator();
                ui.label(RichText::new("Lua API").small().strong());
                ui.label("set_ui_value(\"id\", 0.75)  â€” numeric value");
                ui.label("set_ui_text(\"id\", \"text\")  â€” text override");
                ui.label("set_ui_visible(\"id\", true)  â€” show/hide");
                ui.label("get_ui_value(\"id\")  â€” read back value");
            });

        for w in widget_specs.iter() {
            if !w.visible { continue; }
            let fg = egui::Color32::from_rgba_premultiplied(
                (w.color[0] * 255.0) as u8, (w.color[1] * 255.0) as u8,
                (w.color[2] * 255.0) as u8, (w.color[3] * 255.0) as u8,
            );
            let bg = egui::Color32::from_rgba_premultiplied(
                (w.bg_color[0] * 255.0) as u8, (w.bg_color[1] * 255.0) as u8,
                (w.bg_color[2] * 255.0) as u8, (w.bg_color[3] * 255.0) as u8,
            );
            let _text_style = egui::RichText::new("").size(w.font_size).color(fg);
            // Compute anchored position based on screen size and anchor.
            let anchor_pos = match w.anchor {
                UiAnchor::TopLeft      => egui::pos2(w.x, w.y),
                UiAnchor::TopCenter    => egui::pos2(ctx.screen_rect().width() * 0.5 - w.w * 0.5 + w.x, w.y),
                UiAnchor::TopRight     => egui::pos2(ctx.screen_rect().width() - w.w - w.x, w.y),
                UiAnchor::CenterLeft   => egui::pos2(w.x, ctx.screen_rect().height() * 0.5 - w.h * 0.5 + w.y),
                UiAnchor::Center       => egui::pos2(ctx.screen_rect().width() * 0.5 - w.w * 0.5 + w.x, ctx.screen_rect().height() * 0.5 - w.h * 0.5 + w.y),
                UiAnchor::CenterRight  => egui::pos2(ctx.screen_rect().width() - w.w - w.x, ctx.screen_rect().height() * 0.5 - w.h * 0.5 + w.y),
                UiAnchor::BottomLeft   => egui::pos2(w.x, ctx.screen_rect().height() - w.h - w.y),
                UiAnchor::BottomCenter => egui::pos2(ctx.screen_rect().width() * 0.5 - w.w * 0.5 + w.x, ctx.screen_rect().height() - w.h - w.y),
                UiAnchor::BottomRight  => egui::pos2(ctx.screen_rect().width() - w.w - w.x, ctx.screen_rect().height() - w.h - w.y),
            };
            egui::Area::new(format!("hud_widget_{}", w.id).into())
                .fixed_pos(anchor_pos)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| match w.kind {
                    UiWidgetKind::HealthBar => {
                        let v = args.scripts.ui_value(&w.id).clamp(0.0, 1.0);
                        let bar = egui::ProgressBar::new(v).desired_width(w.w).text(format!("{}: {:.0}%", w.id, v * 100.0));
                        ui.add(bar);
                    }
                    UiWidgetKind::Counter => {
                        let txt = args.scripts.ui_text(&w.id)
                            .unwrap_or_else(|| format!("{:.0}", args.scripts.ui_value(&w.id)));
                        ui.label(RichText::new(format!("{}: {}", w.id, txt)).size(w.font_size).color(fg).strong());
                    }
                    UiWidgetKind::Label => {
                        let txt = args.scripts.ui_text(&w.id).unwrap_or_else(|| w.id.clone());
                        ui.label(RichText::new(txt).size(w.font_size).color(fg));
                    }
                    UiWidgetKind::Button => {
                        let txt = args.scripts.ui_text(&w.id).unwrap_or_else(|| w.id.clone());
                        let resp = ui.add(egui::Button::new(RichText::new(txt).size(w.font_size).color(fg)).min_size(egui::vec2(w.w, w.h)));
                        if resp.clicked() {
                            // Fire Lua callback: on_ui_click("widget_id")
                            let _ = args.scripts.lua_create().globals().set("ui_click_event", w.id.clone());
                        }
                    }
UiWidgetKind::Slider => {
                            let mut val = args.scripts.ui_value(&w.id);
                            let label = format!("{}: {:.2}", w.id, val);
                            let resp = ui.add(egui::Slider::new(&mut val, 0.0..=1.0).text(label));
                            if resp.changed() {
                                args.scripts.set_ui_value(&w.id, val);
                            }
                        }
                    UiWidgetKind::Toggle => {
                        let mut val = args.scripts.ui_value(&w.id) > 0.5;
                        let resp = ui.checkbox(&mut val, format!("{}", w.id));
                        if resp.changed() {
                            args.scripts.set_ui_value(&w.id, if val { 1.0 } else { 0.0 });
                        }
                    }
                    UiWidgetKind::Panel => {
                        egui::Frame::new()
                            .fill(bg)
                            .corner_radius(8.0)
                            .inner_margin(egui::Margin::same(12))
                            .show(ui, |ui| {
                                ui.set_min_width(w.w - 24.0);
                                ui.set_min_height(w.h - 24.0);
                                let title = args.scripts.ui_text(&w.id).unwrap_or_else(|| w.id.clone());
                                ui.label(RichText::new(title).size(w.font_size).color(fg).strong());
                            });
                    }
                    UiWidgetKind::ProgressRing => {
                        let v = args.scripts.ui_value(&w.id).clamp(0.0, 1.0);
                        let radius = w.h * 0.5;
                        let (rect, _response) = ui.allocate_exact_size(egui::vec2(w.w, w.h), egui::Sense::hover());
                        let painter = ui.painter();
                        let center = rect.center();
                        // Background ring
                        painter.circle_stroke(center, radius, egui::Stroke::new(4.0, bg));
                        // Foreground arc
                        let start_angle = -std::f32::consts::FRAC_PI_2; // 12 o'clock
                        let sweep = v * std::f32::consts::TAU;
                        let _rect_ring = egui::Rect::from_center_size(center, egui::vec2(radius * 2.0, radius * 2.0));
                        let arc_points: Vec<egui::Pos2> = (0..=64)
                            .map(|i| {
                                let a = start_angle + sweep * (i as f32 / 64.0);
                                egui::pos2(center.x + a.cos() * radius, center.y + a.sin() * radius)
                            })
                            .collect();
                        painter.add(egui::Shape::line(arc_points, egui::Stroke::new(4.0, fg)));
                        painter.text(center, egui::Align2::CENTER_CENTER, format!("{:.0}%", v * 100.0), egui::FontId::proportional(w.font_size), fg);
                    }
                    UiWidgetKind::Meter => {
                        let segments = 10;
                        let v = args.scripts.ui_value(&w.id).clamp(0.0, 1.0);
                        let filled = (v * segments as f32).round() as u32;
                        let (rect, _response) = ui.allocate_exact_size(egui::vec2(w.w, w.h), egui::Sense::hover());
                        let painter = ui.painter();
                        let seg_w = (w.w - (segments as f32 - 1.0) * 2.0) / segments as f32;
                        for s in 0..segments {
                            let x = rect.left() + s as f32 * (seg_w + 2.0);
                            let seg_rect = egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(seg_w, w.h));
                            let fill = if s < filled { fg } else { bg };
                            painter.rect_filled(seg_rect, 2.0, fill);
                        }
                    }
                });
        }

        return;
    }

    egui::Panel::top("main_menu_bar").exact_height(28.0).show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("TRI").strong().color(Color32::from_rgb(215, 215, 220)));
            ui.separator();
            for menu in ["File", "Edit", "Window", "Tools", "Build", "Select", "Actor", "Help"] {
                ui.menu_button(menu, |ui| {
                    if ui.button("Action 1").clicked() {
                        ui.close();
                    }
                    if ui.button("Action 2").clicked() {
                        ui.close();
                    }
                });
            }
            ui.separator();
            ui.label(RichText::new("Start typing to search").small().weak());
        });
    });
    egui::Panel::top("top_toolbar").exact_height(34.0).show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Triengine Editor").strong().color(Color32::from_rgb(212, 212, 218)));
            ui.separator();
            if ui
                .button("Back to Hub")
                .on_hover_text("Return to project launcher without closing the app")
                .clicked()
            {
                *args.return_to_hub = true;
            }
            if ui.button("Undo").on_hover_text("Ctrl+Z").clicked() {
                *undo_requested = true;
            }
            if ui.button("Redo").on_hover_text("Ctrl+Y / Ctrl+Shift+Z").clicked() {
                *redo_requested = true;
            }
            if ui
                .button("Integrated GPU")
                .on_hover_text("Low-cost preset: smaller shadows, no probes/bloom/SSAO/voxel â€” good for Intel UHD and battery")
                .clicked()
            {
                args.renderer.features = RenderFeatures::low_end();
                args.settings.sync_render_from_renderer_features(&args.renderer.features);
                args.error_log.push(
                    "[Quality] Applied Integrated GPU profile â€” heavy effects off. Toggle them in the toolbar when on a faster PC."
                        .to_string(),
                );
            }
            ui.label("GPU Tier");
            egui::ComboBox::from_id_salt("gpu_scalability_tier")
                .selected_text(args.settings.runtime.gpu_scalability_tier.clone())
                .show_ui(ui, |ui| {
                    for n in ["auto", "low", "balanced", "high", "experimental"] {
                        ui.selectable_value(
                            &mut args.settings.runtime.gpu_scalability_tier,
                            n.to_string(),
                            n,
                        );
                    }
                });
            if ui.button("Apply GPU Tier").clicked() {
                args.renderer.features = RenderFeatures::from_tier_name(
                    &args.settings.runtime.gpu_scalability_tier,
                );
                args.settings
                    .sync_render_from_renderer_features(&args.renderer.features);
                args.error_log.push(format!(
                    "[Quality] Applied GPU tier '{}'.",
                    args.settings.runtime.gpu_scalability_tier
                ));
            }
            if ui.button("Icons Debug").clicked() {
                *show_icon_debug = true;
            }
            if ui.button("Performance Safety Check").clicked() {
                *show_perf_safety_check = true;
            }
            ui.separator();
            ui.label("Preset");
            preset_combo(ui, &mut args.settings.render.preset);
            ui.separator();
            ui.label("Platforms");
            let _ = ui.button("Windows");
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
            ui.selectable_value(gizmo_space, GizmoSpace::World, "World");
            ui.selectable_value(gizmo_space, GizmoSpace::Local, "Local");
            ui.checkbox(snap_enabled, "Snap");
            ui.add(egui::DragValue::new(snap_translate).prefix("T ").speed(0.05).range(0.01..=5.0));
            ui.add(egui::DragValue::new(snap_rotate_deg).prefix("R ").speed(0.2).range(1.0..=45.0));
            ui.add(egui::DragValue::new(snap_scale).prefix("S ").speed(0.01).range(0.01..=1.0));
            if ui.button("Project Launcher").clicked() {
                *show_project_launcher = true;
            }
            ui.separator();
            ui.label("Scene");
            egui::ComboBox::from_id_salt("scene_picker_combo")
                .selected_text(scene_picker_choice.clone())
                .show_ui(ui, |ui| {
                    for p in args.available_scene_paths {
                        ui.selectable_value(scene_picker_choice, p.clone(), p);
                    }
                });
            if ui.button("Load Scene").clicked() {
                *args.requested_scene_switch = Some(scene_picker_choice.clone());
            }
            if ui.button("Scene Manager").clicked() {
                *show_scene_manager = true;
            }
            if ui.button("Dock All").clicked() {
                *undock_hierarchy = false;
                *undock_inspector = false;
                *undock_asset_browser = false;
                *undock_viewport = false;
            }
            if ui.button("Save Layout").clicked() {
                save_dock_layout(dock_state, workspace_preset);
            }
            if ui.button("Load Layout").clicked() {
                if let Some((loaded_preset, state)) = load_saved_dock_layout() {
                    *workspace_preset = loaded_preset;
                    *dock_state = state;
                }
            }
            ui.separator();
            ui.label(format!("Assets: {}", asset_db.entries.len()));
            ui.label(format!("Icons: {}", icon_registry.icons.len()));
            ui.separator();
            ui.label("Workspace");
            if ui.selectable_label(*workspace_preset == "Default", "Default").clicked() {
                *workspace_preset = "Default".to_string();
                apply_workspace_preset(dock_state, workspace_preset);
            }
            if ui.selectable_label(*workspace_preset == "Level Design", "Level Design").clicked() {
                *workspace_preset = "Level Design".to_string();
                apply_workspace_preset(dock_state, workspace_preset);
            }
            if ui.selectable_label(*workspace_preset == "Scripting", "Scripting").clicked() {
                *workspace_preset = "Scripting".to_string();
                apply_workspace_preset(dock_state, workspace_preset);
            }
        });
    });
    egui::Panel::left("mode_sidebar")
        .resizable(false)
        .exact_width(74.0)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(8.0);
                ui.label(RichText::new("Mode").small().color(Color32::from_rgb(160, 168, 180)));
                ui.separator();
                for m in ["Select", "Terrain", "Foliage", "Paint", "Script", "FX"] {
                    let _ = ui.add_sized([58.0, 26.0], egui::Button::new(m));
                }
            });
        });
    egui::Panel::bottom("status_bar")
        .exact_height(24.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Content Drawer").small());
                ui.separator();
                ui.label(RichText::new("Output Log").small());
                ui.separator();
                ui.label(RichText::new(format!("Scene: {}", args.scene_path)).small());
                ui.separator();
                ui.label(RichText::new(format!("FPS target: {}", args.settings.runtime.max_fps)).small());
            });
        });

    struct Viewer<'a, 'b> {
        args: &'a mut UiFrameArgs<'b>,
        texture_dragging: &'a mut Option<String>,
        texture_selected: &'a mut Option<String>,
        mesh_selected: &'a mut Option<String>,
        content_new_folder: &'a mut String,
        content_new_file: &'a mut String,
        gizmo_mode: &'a mut GizmoMode,
        gizmo_drag: &'a mut Option<GizmoDragState>,
        gizmo_space: GizmoSpace,
        gizmo_axis_lock: &'a mut Option<GizmoAxis>,
        terrain_mode: &'a mut bool,
        terrain_brush_mode: &'a mut components::TerrainBrushMode,
        terrain_brush_radius: &'a mut f32,
        snap_enabled: bool,
        snap_translate: f32,
        snap_rotate_deg: f32,
        snap_scale: f32,
        scene_texture_id: Option<egui::TextureId>,
        asset_db: &'a AssetMetadataDb,
        asset_search: &'a mut String,
        asset_kind_filter: &'a mut String,
        asset_sort_desc: &'a mut bool,
        texture_thumbnail_cache: &'a mut HashMap<String, egui::TextureHandle>,
        icon_texture_cache: &'a HashMap<String, egui::TextureHandle>,
        egui_ctx: &'a egui::Context,
        console_filter: &'a mut String,
        show_material_editor: &'a mut bool,
        show_foliage_editor: &'a mut bool,
    }
    impl<'a, 'b> TabViewer for Viewer<'a, 'b> {
        type Tab = EditorDockTab;
        fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
            match tab {
                EditorDockTab::Viewport => "[vp] Viewport".into(),
                EditorDockTab::Outliner => "[wo] Outliner".into(),
                EditorDockTab::Details => "[dt] Details".into(),
                EditorDockTab::ContentBrowser => "[cb] Content Browser".into(),
                EditorDockTab::Profiler => "[pf] Profiler".into(),
                EditorDockTab::Console => "[log] Console".into(),
                EditorDockTab::ScriptEditor => "[sc] Script Editor".into(),
                EditorDockTab::Levels => "[lv] Levels".into(),
            }
        }
        fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
            match tab {
                EditorDockTab::Viewport => {
                    panels::render_viewport_panel(
                        ui,
                        self.args,
                        self.scene_texture_id,
                        self.gizmo_mode,
                        self.gizmo_drag,
                        self.gizmo_space,
                        *self.gizmo_axis_lock,
                        *self.terrain_mode,
                        *self.terrain_brush_mode,
                        *self.terrain_brush_radius,
                        self.snap_enabled,
                        self.snap_translate,
                        self.snap_rotate_deg,
                        self.snap_scale,
                    );
                }
                EditorDockTab::Outliner => {
                    panels::render_outliner_panel(ui, self.args);
                }
                EditorDockTab::Details => {
                    panels::render_details_panel(ui, self.args);
                }
                EditorDockTab::ContentBrowser => {
                    panels::render_content_browser_panel(
                        ui,
                        self.asset_db,
                        self.asset_search,
                        self.asset_kind_filter,
                        self.asset_sort_desc,
                        self.texture_selected,
                        self.mesh_selected,
                        self.texture_dragging,
                        self.content_new_folder,
                        self.content_new_file,
                        self.texture_thumbnail_cache,
                        self.icon_texture_cache,
                        self.args.preferred_script_editor,
                        self.show_material_editor,
                        self.show_foliage_editor,
                        self.args.error_log,
                        self.egui_ctx,
                        self.args.bake_requested,
                    );
                }
                EditorDockTab::Profiler => {
                    if let Some(text) = self.args.profiler.overlay_text() {
                        ui.label(text);
                    }
                }
                EditorDockTab::Console => {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Output Log").strong());
                        ui.separator();
                        ui.label(RichText::new("Tagged runtime/editor messages").small().weak());
                    });
                    ui.horizontal(|ui| {
                        ui.label("Filter");
                        ui.add(
                            egui::TextEdit::singleline(self.console_filter)
                                .desired_width(220.0)
                                .hint_text("substringâ€¦"),
                        );
                        if ui.button("Clear log").clicked() {
                            self.args.error_log.clear();
                        }
                        if ui.button("Copy all").clicked() {
                            let t = self.args.error_log.join("\n");
                            self.egui_ctx.copy_text(t);
                        }
                    });
                    ui.separator();
                    let filt = self.console_filter.to_ascii_lowercase();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let mut n = 0;
                        for line in self.args.error_log.iter().rev() {
                            if !filt.is_empty() && !line.to_ascii_lowercase().contains(&filt) {
                                continue;
                            }
                            n += 1;
                            if n > 400 {
                                break;
                            }
                            egui::Frame::new()
                                .fill(Color32::from_rgb(22, 22, 27))
                                .corner_radius(3.0)
                                .inner_margin(egui::Margin::same(4))
                                .show(ui, |ui| {
                                    ui.label(console_line_rich(line));
                                });
                            ui.add_space(1.0);
                        }
                        if self.args.error_log.is_empty() {
                            ui.label(RichText::new("No messages.").italics().weak());
                        }
                    });
                }
                EditorDockTab::ScriptEditor => {
                    panels::render_script_editor_panel(ui, self.args, self.icon_texture_cache);
                }
                EditorDockTab::Levels => {
                    panels::render_levels_panel(ui, self.args);
                }
            }
        }
    }

    let use_legacy_panels = args.settings.runtime.legacy_editor_ui
        || std::env::var("TRINITY_LEGACY_UI").is_ok();
    if !use_legacy_panels {
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut viewer = Viewer {
                args,
                texture_dragging,
                texture_selected,
                mesh_selected,
                content_new_folder,
                content_new_file,
                gizmo_mode,
                gizmo_drag,
                gizmo_space: *gizmo_space,
                gizmo_axis_lock,
                terrain_mode,
                terrain_brush_mode,
                terrain_brush_radius,
                snap_enabled: *snap_enabled,
                snap_translate: *snap_translate,
                snap_rotate_deg: *snap_rotate_deg,
                snap_scale: *snap_scale,
                scene_texture_id,
                asset_db,
                asset_search,
                asset_kind_filter,
                asset_sort_desc,
                texture_thumbnail_cache,
                icon_texture_cache,
                egui_ctx: ctx,
                console_filter,
                show_material_editor,
                show_foliage_editor,
            };
            DockArea::new(dock_state)
                .style(DockStyle::from_egui(ui.style().as_ref()))
                .show_inside(ui, &mut viewer);
        });
        egui::Window::new("HUD Widgets (Lua)")
            .default_pos([24.0, 520.0])
            .default_size([320.0, 220.0])
            .collapsible(true)
            .show(ctx, |ui| {
                ui.label("Runtime HUD driven by Lua: set_ui_value / set_ui_text.");
                ui.horizontal(|ui| {
                    ui.label("Id");
                    ui.text_edit_singleline(widget_new_id);
                });
                ui.horizontal(|ui| {
                    ui.selectable_value(widget_new_kind, UiWidgetKind::HealthBar, "Health");
                    ui.selectable_value(widget_new_kind, UiWidgetKind::Counter, "Counter");
                    ui.selectable_value(widget_new_kind, UiWidgetKind::Label, "Label");
                    ui.selectable_value(widget_new_kind, UiWidgetKind::Button, "Button");
                    ui.selectable_value(widget_new_kind, UiWidgetKind::Slider, "Slider");
                    ui.selectable_value(widget_new_kind, UiWidgetKind::Toggle, "Toggle");
                    ui.selectable_value(widget_new_kind, UiWidgetKind::Panel, "Panel");
                    ui.selectable_value(widget_new_kind, UiWidgetKind::ProgressRing, "Ring");
                    ui.selectable_value(widget_new_kind, UiWidgetKind::Meter, "Meter");
                });
                if ui.button("Add widget").clicked() && !widget_new_id.trim().is_empty() {
                    widget_specs.push(UiWidgetSpec {
                        id: widget_new_id.trim().to_string(),
                        kind: *widget_new_kind,
                        x: 30.0,
                        y: 90.0 + widget_specs.len() as f32 * 26.0,
                        w: 240.0,
                        h: match *widget_new_kind {
                            UiWidgetKind::Panel => 120.0,
                            UiWidgetKind::Meter => 28.0,
                            UiWidgetKind::ProgressRing => 64.0,
                            UiWidgetKind::Button => 32.0,
                            UiWidgetKind::Slider => 24.0,
                            UiWidgetKind::Toggle => 24.0,
                            _ => 24.0,
                        },
                        visible: true,
                        z_order: 0,
                        color: [1.0, 1.0, 1.0, 1.0],
                        bg_color: [0.0, 0.0, 0.0, 0.5],
                        font_size: 14.0,
                        anchor: UiAnchor::TopLeft,
                    });
                }
                ui.separator();
                let mut rm: Option<usize> = None;
                for (i, w) in widget_specs.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.monospace(&w.id);
                        let vis_resp = ui.selectable_label(w.visible, if w.visible { "V" } else { "-" });
                        if vis_resp.clicked() { w.visible = !w.visible; }
                        if ui.button("x").clicked() {
                            rm = Some(i);
                        }
                    });
                }
                if let Some(i) = rm {
                    widget_specs.remove(i);
                }
            });
        for w in widget_specs.iter() {
            if !w.visible { continue; }
            let fg = egui::Color32::from_rgba_premultiplied(
                (w.color[0] * 255.0) as u8, (w.color[1] * 255.0) as u8,
                (w.color[2] * 255.0) as u8, (w.color[3] * 255.0) as u8,
            );
            egui::Area::new(format!("hud_widget_{}", w.id).into())
                .fixed_pos([w.x, w.y])
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    match w.kind {
                        UiWidgetKind::HealthBar => {
                            let v = args.scripts.ui_value(&w.id).clamp(0.0, 1.0);
                            ui.add(egui::ProgressBar::new(v).desired_width(w.w).text(format!(
                                "{}: {:.0}%",
                                w.id,
                                v * 100.0
                            )));
                        }
                        UiWidgetKind::Counter => {
                            let txt = args.scripts.ui_text(&w.id).unwrap_or_else(|| {
                                format!("{:.0}", args.scripts.ui_value(&w.id))
                            });
                            ui.label(RichText::new(format!("{}: {}", w.id, txt)).size(w.font_size).color(fg).strong());
                        }
                        UiWidgetKind::Label => {
                            let txt = args.scripts.ui_text(&w.id).unwrap_or_else(|| w.id.clone());
                            ui.label(RichText::new(txt).size(w.font_size).color(fg));
                        }
                        UiWidgetKind::Button => {
                            let txt = args.scripts.ui_text(&w.id).unwrap_or_else(|| w.id.clone());
                            ui.add(egui::Button::new(RichText::new(txt).size(w.font_size).color(fg)).min_size(egui::vec2(w.w, w.h)));
                        }
                        UiWidgetKind::Slider => {
                            let mut val = args.scripts.ui_value(&w.id);
                            let label = format!("{}: {:.2}", w.id, val);
                            ui.add(egui::Slider::new(&mut val, 0.0..=1.0).text(label));
                        }
                        UiWidgetKind::Toggle => {
                            let mut val = args.scripts.ui_value(&w.id) > 0.5;
                            ui.checkbox(&mut val, format!("{}", w.id));
                        }
                        UiWidgetKind::Panel => {
                            let bg = egui::Color32::from_rgba_premultiplied(
                                (w.bg_color[0] * 255.0) as u8, (w.bg_color[1] * 255.0) as u8,
                                (w.bg_color[2] * 255.0) as u8, (w.bg_color[3] * 255.0) as u8,
                            );
                            egui::Frame::new().fill(bg).corner_radius(8.0).inner_margin(egui::Margin::same(12)).show(ui, |ui| {
                                let title = args.scripts.ui_text(&w.id).unwrap_or_else(|| w.id.clone());
                                ui.label(RichText::new(title).size(w.font_size).color(fg).strong());
                            });
                        }
                        UiWidgetKind::ProgressRing => {
                            let v = args.scripts.ui_value(&w.id).clamp(0.0, 1.0);
                            ui.label(RichText::new(format!("{}: {:.0}%", w.id, v * 100.0)).size(w.font_size).color(fg));
                        }
                        UiWidgetKind::Meter => {
                            ui.label(RichText::new(format!("{}: {:.0}%", w.id, args.scripts.ui_value(&w.id) * 100.0)).size(w.font_size).color(fg));
                        }
                    }
                });
        }
        return;
    }

    let mut draw_hierarchy_panel = |ui: &mut egui::Ui| {
        ui.heading("Hierarchy");
        ui.label("Professional editor icons (from assets/icons when present)");
        if ui.button("Delete Selected Entity").clicked() {
            if let Some(e) = *args.selected_renderable {
                if args.world.despawn(e).is_ok() {
                    *args.selected_renderable = None;
                    args.error_log.push("[Scene] Deleted selected entity.".to_string());
                }
            }
        }
        if ui.button("Create Prefab From Selected").clicked() {
            let prefab_dir = "Content/Prefabs";
            let _ = fs::create_dir_all(prefab_dir);
            if let Some(entity) = *args.selected_renderable {
                match save_selected_as_prefab(args, entity, prefab_dir) {
                    Ok(p) => args.error_log.push(format!("[Prefab] Saved {}", p)),
                    Err(e) => args.error_log.push(format!("[Prefab] Save failed: {}", e)),
                }
            } else {
                args.error_log
                    .push("[Prefab] Select an entity in Hierarchy first.".to_string());
            }
        }
        ui.separator();
        icon_row(ui, icon_texture_cache, "camera", "Main Camera");
        icon_row(ui, icon_texture_cache, "light", "Directional Light");
        icon_row(ui, icon_texture_cache, "sky", "Skylight / Sky Background");
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
            ui.horizontal(|ui| {
                ui.label("Sky Image (.hdr/.exr)");
                ui.text_edit_singleline(&mut args.settings.render.sky_hdr_path);
            });
            ui.horizontal_wrapped(|ui| {
                if ui.button("Apply Sky HDR").clicked() {
                    match args.renderer.apply_sky_environment(&args.settings.render.sky_hdr_path) {
                        Ok(()) => args.error_log.push("[Sky] Applied custom sky HDR.".to_string()),
                        Err(e) => args.error_log.push(format!("[Sky] Apply failed: {}", e)),
                    }
                }
                if ui.button("Use Default Sky").clicked() {
                    args.settings.render.sky_hdr_path.clear();
                    match args.renderer.apply_sky_environment("") {
                        Ok(()) => args.error_log.push("[Sky] Reverted to default sky.".to_string()),
                        Err(e) => args.error_log.push(format!("[Sky] Reset failed: {}", e)),
                    }
                }
            });
            ui.checkbox(&mut args.renderer.features.shadows_enabled, "Directional Shadows");
            ui.checkbox(&mut args.renderer.features.pcss_enabled, "Soft shadows (PCSS)");
            ui.add(egui::Slider::new(&mut args.renderer.features.bloom_strength, 0.0..=2.0).text("Bloom"));
            ui.add(egui::Slider::new(&mut args.renderer.features.ssao_strength, 0.0..=1.0).text("SSAO"));
            ui.add(egui::Slider::new(&mut args.renderer.features.fog_density, 0.0..=0.20).text("Fog density"));
            ui.separator();
            // â”€â”€ Tone Mapping + Colour Grading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            ui.horizontal(|ui| {
                ui.label("Tone Mapping");
                ui.checkbox(&mut args.renderer.features.tonemap_enabled, "");
            });
            if args.renderer.features.tonemap_enabled {
                ui.add(egui::Slider::new(&mut args.renderer.features.tonemap_exposure, -2.0..=2.0).text("Exposure"));
                ui.add(egui::Slider::new(&mut args.renderer.features.tonemap_temperature, -1.0..=1.0).text("Temperature"));
                ui.add(egui::Slider::new(&mut args.renderer.features.tonemap_saturation, -1.0..=1.0).text("Saturation"));
                ui.add(egui::Slider::new(&mut args.renderer.features.tonemap_contrast, -1.0..=1.0).text("Contrast"));
                ui.add(egui::Slider::new(&mut args.renderer.features.tonemap_vibrance, -1.0..=1.0).text("Vibrance"));
                ui.add(egui::Slider::new(&mut args.renderer.features.tonemap_grain, 0.0..=0.1).text("Film Grain"));
            }
            ui.separator();
            // â”€â”€ Wind â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            ui.label("Wind");
            ui.add(egui::Slider::new(&mut args.renderer.features.wind_strength, 0.0..=1.0).text("Strength"));
            ui.horizontal(|ui| {
                ui.label("Dir X:");
                ui.add(egui::Slider::new(&mut args.renderer.features.wind_dir[0], -1.0..=1.0).step_by(0.01));
                ui.label("Dir Z:");
                ui.add(egui::Slider::new(&mut args.renderer.features.wind_dir[2], -1.0..=1.0).step_by(0.01));
            });
            ui.separator();
            // â”€â”€ Screen-Space Reflections â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            ui.horizontal(|ui| {
                ui.label("Screen-Space Reflections");
                ui.checkbox(&mut args.renderer.features.ssr_enabled, "");
            });
            if args.renderer.features.ssr_enabled {
                ui.add(egui::Slider::new(&mut args.renderer.features.ssr_max_steps, 16..=128).text("Max Steps"));
                ui.add(egui::Slider::new(&mut args.renderer.features.ssr_max_distance, 5.0..=100.0).text("Max Distance"));
                ui.add(egui::Slider::new(&mut args.renderer.features.ssr_thickness, 0.01..=0.2).text("Thickness"));
                ui.add(egui::Slider::new(&mut args.renderer.features.ssr_intensity, 0.0..=2.0).text("Intensity"));
            }
            ui.separator();
            // â”€â”€ Water Rendering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            ui.horizontal(|ui| {
                ui.label("Water Surfaces");
                ui.checkbox(&mut args.renderer.features.water_enabled, "");
            });
            ui.separator();
            // â”€â”€ Lava Rendering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            ui.horizontal(|ui| {
                ui.label("Lava Surfaces");
                ui.checkbox(&mut args.renderer.features.lava_enabled, "");
            });
            ui.separator();
            // ── Rendering Features ────────────────────────────────────────────
            ui.collapsing("Rendering Features", |ui| {
                ui.label("Shadows");
                ui.checkbox(&mut args.renderer.features.shadows_enabled, "Directional Shadows");
                ui.checkbox(&mut args.renderer.features.pcf_enabled, "PCF Softening");
                ui.checkbox(&mut args.renderer.features.pcss_enabled, "Contact Shadows (PCSS)");
                ui.add(egui::Slider::new(&mut args.renderer.features.shadow_resolution, 512..=8192).logarithmic(true).text("Shadow Resolution"));
                ui.add(egui::Slider::new(&mut args.renderer.features.pcf_samples, 1..=32).text("PCF Samples"));
                ui.separator();
                ui.label("Global Illumination");
                ui.checkbox(&mut args.renderer.features.ibl_enabled, "Sky Light (IBL)");
                ui.checkbox(&mut args.renderer.features.probes_enabled, "Baked Light Probes");
                ui.checkbox(&mut args.renderer.features.voxel_gi_enabled, "Voxel GI (128³)");
                ui.add(egui::Slider::new(&mut args.renderer.features.voxel_gi_strength, 0.0..=1.0).text("Voxel GI Strength"));
                ui.separator();
                ui.label("Post-Processing");
                ui.checkbox(&mut args.renderer.features.bloom_enabled, "Bloom");
                ui.checkbox(&mut args.renderer.features.ssao_enabled, "SSAO");
                ui.checkbox(&mut args.renderer.features.volumetric_fog_enabled, "Volumetric Fog");
                ui.checkbox(&mut args.renderer.features.volumetric_enabled, "Volumetric Light Scatter");
                ui.checkbox(&mut args.renderer.features.tonemap_enabled, "Tone Mapping");
                ui.checkbox(&mut args.renderer.features.taa_enabled, "TAA");
                ui.checkbox(&mut args.renderer.features.motion_blur_enabled, "Motion Blur");
                ui.checkbox(&mut args.renderer.features.god_rays_enabled, "God Rays");
                ui.checkbox(&mut args.renderer.features.dof_enabled, "Depth of Field");
                ui.separator();
                ui.label("Screen-Space & Reflections");
                ui.checkbox(&mut args.renderer.features.ssr_enabled, "Screen-Space Reflections");
                ui.separator();
                ui.label("FX Surfaces");
                ui.checkbox(&mut args.renderer.features.water_enabled, "Water");
                ui.checkbox(&mut args.renderer.features.lava_enabled, "Lava");
                ui.checkbox(&mut args.renderer.features.fire_enabled, "Fire");
                ui.checkbox(&mut args.renderer.features.heat_distortion_enabled, "Heat Distortion");
                ui.checkbox(&mut args.renderer.features.underwater_enabled, "Underwater Post-FX");
                if args.renderer.features.underwater_enabled {
                    ui.add(egui::Slider::new(&mut args.renderer.features.underwater_fog_density, 0.0..=1.0).text("Fog Density"));
                    ui.add(egui::Slider::new(&mut args.renderer.features.underwater_caustics, 0.0..=2.0).text("Caustics"));
                    ui.add(egui::Slider::new(&mut args.renderer.features.underwater_god_rays, 0.0..=2.0).text("God Rays"));
                    ui.add(egui::Slider::new(&mut args.renderer.features.underwater_distortion, 0.0..=1.0).text("Distortion"));
                    ui.add(egui::Slider::new(&mut args.renderer.features.underwater_vignette, 0.0..=1.0).text("Vignette"));
                    ui.add(egui::Slider::new(&mut args.renderer.features.underwater_bloom, 0.0..=2.0).text("Bloom Boost"));
                }
                ui.separator();
                ui.label("Culling");
                ui.checkbox(&mut args.renderer.features.culling_enabled, "Culling");
                ui.checkbox(&mut args.renderer.features.frustum_culling_enabled, "Frustum Culling");
                ui.checkbox(&mut args.renderer.features.occlusion_culling_enabled, "Occlusion Culling");
                ui.add(egui::Slider::new(&mut args.renderer.features.culling_distance, 10.0..=500.0).text("Cull Distance"));
                ui.separator();
                ui.label("LOD Thresholds");
                ui.add(egui::Slider::new(&mut args.renderer.features.mesh_lod_threshold_1, 10.0..=400.0).text("LOD 0→1"));
                ui.add(egui::Slider::new(&mut args.renderer.features.mesh_lod_threshold_2, 10.0..=400.0).text("LOD 1→2"));
                ui.add(egui::Slider::new(&mut args.renderer.features.mesh_lod_threshold_3, 10.0..=400.0).text("LOD 2→3"));
                ui.add(egui::Slider::new(&mut args.renderer.features.mesh_lod_threshold_4, 10.0..=400.0).text("LOD 3→4"));
            });
            ui.separator();
            ui.label("Sun Direction (real-time day cycle)");
            ui.horizontal(|ui| {
                ui.label("Hour:");
                ui.add(egui::Slider::new(&mut args.time_of_day.hour, 0.0..=24.0).step_by(0.1));
            });
            ui.horizontal(|ui| {
                ui.label("Speed:");
                let speed_label = if args.time_of_day.paused {
                    "PAUSED"
                } else {
                    "running"
                };
                if ui.button(speed_label).clicked() {
                    args.time_of_day.paused = !args.time_of_day.paused;
                }
                ui.add(egui::Slider::new(&mut args.time_of_day.speed, 0.0..=2.0)
                    .text("game-hrs/sec"));
            });
            let daylight = args.time_of_day.daylight_factor();
            ui.label(format!("Daylight: {:.0}%", daylight * 100.0));
            ui.separator();
            ui.label("Night Sky");
            ui.horizontal(|ui| {
                ui.label("Stars");
                ui.checkbox(&mut args.sky.stars_enabled, "");
                let r = ui.add(egui::Slider::new(&mut args.sky.star_intensity, 0.0..=1.0).text("Brightness"));
                if r.changed() { args.sky.stars_auto = false; }
                let r2 = ui.add(egui::Slider::new(&mut args.sky.star_density, 0.0..=2.0).text("Density"));
                if r2.changed() { args.sky.stars_auto = false; }
                if ui.selectable_label(args.sky.stars_auto, "Auto").clicked() {
                    args.sky.stars_auto = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Moon");
                ui.checkbox(&mut args.sky.moon_enabled, "");
                let r = ui.add(egui::Slider::new(&mut args.sky.moon_intensity, 0.0..=2.0).text("Brightness"));
                if r.changed() { args.sky.moon_auto = false; }
                if ui.selectable_label(args.sky.moon_auto, "Auto").clicked() {
                    args.sky.moon_auto = true;
                }
            });
            ui.separator();
            ui.label("Weather");
            let conditions = [
                ("Clear", crate::environment::weather::WeatherCondition::Clear),
                ("Cloudy", crate::environment::weather::WeatherCondition::Cloudy),
                ("Overcast", crate::environment::weather::WeatherCondition::Overcast),
                ("Light Rain", crate::environment::weather::WeatherCondition::LightRain),
                ("Heavy Rain", crate::environment::weather::WeatherCondition::HeavyRain),
                ("Snow", crate::environment::weather::WeatherCondition::Snow),
                ("Fog", crate::environment::weather::WeatherCondition::Fog),
                ("Storm", crate::environment::weather::WeatherCondition::Storm),
            ];
            ui.horizontal(|ui| {
                for (label, condition) in &conditions {
                    if ui.selectable_label(args.weather.condition == *condition, *label).clicked() {
                        args.weather.condition = *condition;
                        args.weather.intensity = 0.8;
                    }
                }
            });
            ui.add(egui::Slider::new(&mut args.weather.intensity, 0.0..=1.0).text("Intensity"));
            ui.add(egui::Slider::new(&mut args.weather.cloud_coverage, 0.0..=1.0).text("Clouds"));
            ui.add(egui::Slider::new(&mut args.weather.wind_strength, 0.0..=20.0).text("Wind"));
            ui.separator();
            // â”€â”€ Audio Controls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            ui.label("Audio");
            if let Some(audio) = args.audio.as_mut() {
                ui.horizontal(|ui| {
                    ui.label("Master");
                    ui.add(egui::Slider::new(&mut audio.volume.master, 0.0..=1.0).text(""));
                });
                ui.horizontal(|ui| {
                    ui.label("Music  ");
                    ui.add(egui::Slider::new(&mut audio.volume.music, 0.0..=1.0).text(""));
                });
                ui.horizontal(|ui| {
                    ui.label("SFX    ");
                    ui.add(egui::Slider::new(&mut audio.volume.sfx, 0.0..=1.0).text(""));
                });
                ui.horizontal(|ui| {
                    ui.label("Ambient");
                    ui.add(egui::Slider::new(&mut audio.volume.ambient, 0.0..=1.0).text(""));
                });
                ui.label(format!("Active sounds: {}", audio.active_count()));
                ui.label(format!("Music playing: {}", audio.is_music_playing()));
                if ui.button("Stop All").clicked() {
                    audio.stop_all();
                }
            } else {
                ui.label("(Audio device not available)");
            }
            ui.separator();
            ui.label("Movable Point Light");
            if let Some(entity) = args.selected_renderable.as_ref().copied() {
                if let Ok(mut p) = args.world.get::<&mut components::PointLight>(entity) {
                    ui.add(egui::Slider::new(&mut p.color[0], 0.0..=2.0).text("R"));
                    ui.add(egui::Slider::new(&mut p.color[1], 0.0..=2.0).text("G"));
                    ui.add(egui::Slider::new(&mut p.color[2], 0.0..=2.0).text("B"));
                    ui.add(egui::Slider::new(&mut p.intensity, 0.0..=4.0).text("Intensity"));
                    ui.add(egui::Slider::new(&mut p.range, 0.5..=60.0).text("Range"));
                    ui.horizontal(|ui| {
                        ui.label("Type:");
                        ui.selectable_value(&mut p.light_type, 0.0, "Sun");
                        ui.selectable_value(&mut p.light_type, 1.0, "Point");
                        ui.selectable_value(&mut p.light_type, 2.0, "Spot");
                    });
                    if p.light_type == 2.0 {
                        ui.add(egui::Slider::new(&mut p.spot_angle, 5.0..=170.0).text("Cone Angle"));
                    }
                    ui.checkbox(&mut p.shadow_casting, "Shadow Casting");
                } else if ui.button("Add Point Light To Selected").clicked() {
                    let _ = args.world.insert(
                        entity,
                        (components::PointLight::default(),),
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
                        ..Default::default()
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
        ui.collapsing("Water Effects", |ui| {
            if let Some(entity) = args.selected_renderable.as_ref().copied() {
                if let Ok(mut wt) = args.world.get::<&mut components::WaterTrigger>(entity) {
                    ui.label("WaterTrigger");
                    ui.add(egui::Slider::new(&mut wt.splash_intensity, 0.0..=2.0).text("Splash Intensity"));
                    ui.checkbox(&mut wt.active, "Active");
                } else if ui.button("Add WaterTrigger").clicked() {
                    let _ = args.world.insert(entity, (components::WaterTrigger::default(),));
                }
                ui.separator();
                if let Ok(mut se) = args.world.get::<&mut components::SplashEffect>(entity) {
                    ui.label("SplashEffect");
                    ui.add(egui::Slider::new(&mut se.max_splashes, 1..=32).text("Max Splashes"));
                    ui.add(egui::Slider::new(&mut se.splash_duration, 0.1..=5.0).text("Duration (s)"));
                    ui.add(egui::Slider::new(&mut se.ripple_scale, 0.1..=5.0).text("Ripple Scale"));
                    ui.checkbox(&mut se.active, "Active");
                } else if ui.button("Add SplashEffect").clicked() {
                    let _ = args.world.insert(entity, (components::SplashEffect::default(),));
                }
            } else {
                ui.label("Select an entity to configure water effects.");
            }
        });
        ui.collapsing("Terrain Auto Material", |ui| {
            ui.label("Grass on flats, dirt transitions, rock on steep/high areas.");
            ui.add(
                egui::Slider::new(&mut args.terrain_world.material.slope_rock_start, 0.1..=1.6)
                    .text("Rock from slope"),
            );
            ui.add(
                egui::Slider::new(&mut args.terrain_world.material.height_rock_start, 0.0..=6.0)
                    .text("Rock from height"),
            );
            let world_x = args.terrain_cursor_x as f32 * args.terrain_world.cell_size - 32.0;
            let world_z = args.terrain_cursor_z as f32 * args.terrain_world.cell_size - 32.0;
            let preview = args.terrain_world.auto_surface_color_world(world_x, world_z);
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
                    if let Ok(extras) = args.materials.instance_extras("matte_black") {
                        let _ = args.world.insert(entity, (extras,));
                    }
                }
                if ui.button("Apply silver_brushed").clicked() {
                    if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
                        let _ = args.materials.apply_instance("silver_brushed", &mut rend);
                    }
                    if let Ok(extras) = args.materials.instance_extras("silver_brushed") {
                        let _ = args.world.insert(entity, (extras,));
                    }
                }
                if ui.button("Apply foliage_leaf").clicked() {
                    if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
                        let _ = args.materials.apply_instance("foliage_leaf", &mut rend);
                    }
                    if let Ok(extras) = args.materials.instance_extras("foliage_leaf") {
                        let _ = args.world.insert(entity, (extras,));
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
                    ui.label("Textures");
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let p = path.to_string_lossy().to_string();
                        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
                        if ["png", "jpg", "jpeg"].contains(&ext.as_str()) {
                            let selected = texture_selected.as_ref().map(|s| s == &p).unwrap_or(false);
                            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                            ui.vertical(|ui| {
                                if let Some(tex) = texture_thumbnail(ctx, texture_thumbnail_cache, &p) {
                                    let img = egui::Image::new((tex.id(), egui::vec2(96.0, 54.0))).sense(egui::Sense::click());
                                    if ui.add(img).clicked() {
                                        *texture_selected = Some(p.clone());
                                    }
                                } else if ui.selectable_label(selected, format!("TEX {}", name)).clicked() {
                                    *texture_selected = Some(p.clone());
                                }
                                if ui.selectable_label(selected, name).clicked() {
                                    *texture_selected = Some(p.clone());
                                }
                            });
                            }
                    }
                }
                if let Ok(entries) = fs::read_dir("Content/Meshes") {
                    ui.separator();
                    ui.label("Meshes");
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let p = path.to_string_lossy().to_string();
                        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
                        if ext == "obj" {
                            let selected = mesh_selected.as_ref().map(|s| s == &p).unwrap_or(false);
                            let col = mesh_thumb_color(mesh_thumbnail_color_cache, &p);
                            thumbnail_card(ui, "MESH", &path.file_name().unwrap_or_default().to_string_lossy(), Some(col));
                            if ui.selectable_label(selected, "Select mesh").clicked() {
                                *mesh_selected = Some(p);
                            };
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
                        let _ = spawn_mesh_entity(args, handle, [1.0, 1.0, 1.0], c, true);
                    }
                }
                if ui.button("Add foliage ring").clicked() {
                    if let Some(handle) = args.mesh_cache.get("meshes/cube.obj").copied() {
                        spawn_foliage_ring(
                            args.world,
                            handle,
                            args.terrain_cursor_x as f32 * args.terrain_world.cell_size - 32.0,
                            args.terrain_cursor_z as f32 * args.terrain_world.cell_size - 32.0,
                            4.0,
                            24,
                            true,
                        );
                    }
                }
                if ui.button("Remove nearby foliage").clicked() {
                    let _ = remove_nearby_foliage(
                        args.world,
                        args.terrain_cursor_x as f32 * args.terrain_world.cell_size - 32.0,
                        args.terrain_cursor_z as f32 * args.terrain_world.cell_size - 32.0,
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
        if let Some(texture_id) = scene_texture_id {
            ui.painter().image(
                texture_id,
                rect,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }
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
            draw_transform_gizmo(
                ui,
                args,
                entity,
                rect,
                response.interact_pointer_pos(),
                gizmo_mode,
                gizmo_drag,
                *gizmo_space,
                *gizmo_axis_lock,
                *snap_enabled,
                *snap_translate,
                *snap_rotate_deg,
                *snap_scale,
            );
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
                ui.selectable_value(widget_new_kind, UiWidgetKind::HealthBar, "Health");
                ui.selectable_value(widget_new_kind, UiWidgetKind::Counter, "Counter");
                ui.selectable_value(widget_new_kind, UiWidgetKind::Label, "Label");
                ui.selectable_value(widget_new_kind, UiWidgetKind::Button, "Button");
                ui.selectable_value(widget_new_kind, UiWidgetKind::Slider, "Slider");
                ui.selectable_value(widget_new_kind, UiWidgetKind::Toggle, "Toggle");
                ui.selectable_value(widget_new_kind, UiWidgetKind::Panel, "Panel");
                ui.selectable_value(widget_new_kind, UiWidgetKind::ProgressRing, "Ring");
                ui.selectable_value(widget_new_kind, UiWidgetKind::Meter, "Meter");
            });
            if ui.button("Add Widget").clicked() {
                if !widget_new_id.trim().is_empty() {
                    widget_specs.push(UiWidgetSpec {
                        id: widget_new_id.trim().to_string(),
                        kind: *widget_new_kind,
                        x: 30.0,
                        y: 90.0 + widget_specs.len() as f32 * 26.0,
                        w: 240.0,
                        h: match *widget_new_kind {
                            UiWidgetKind::Panel => 120.0,
                            UiWidgetKind::Meter => 28.0,
                            UiWidgetKind::ProgressRing => 64.0,
                            UiWidgetKind::Button => 32.0,
                            _ => 24.0,
                        },
                        visible: true,
                        z_order: 0,
                        color: [1.0, 1.0, 1.0, 1.0],
                        bg_color: [0.0, 0.0, 0.0, 0.5],
                        font_size: 14.0,
                        anchor: UiAnchor::TopLeft,
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
            ui.label("Lua API: set_ui_value, set_ui_text, set_ui_visible, get_ui_value");
        });

    for w in widget_specs.iter() {
        if !w.visible { continue; }
        let fg = egui::Color32::from_rgba_premultiplied(
            (w.color[0] * 255.0) as u8, (w.color[1] * 255.0) as u8,
            (w.color[2] * 255.0) as u8, (w.color[3] * 255.0) as u8,
        );
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
                        let txt = args.scripts.ui_text(&w.id).unwrap_or_else(|| format!("{:.0}", args.scripts.ui_value(&w.id)));
                        ui.label(RichText::new(format!("{}: {}", w.id, txt)).size(w.font_size).color(fg).strong());
                    }
                    UiWidgetKind::Label => {
                        let txt = args.scripts.ui_text(&w.id).unwrap_or_else(|| w.id.clone());
                        ui.label(RichText::new(txt).size(w.font_size).color(fg));
                    }
                    UiWidgetKind::Button => {
                        let txt = args.scripts.ui_text(&w.id).unwrap_or_else(|| w.id.clone());
                        ui.add(egui::Button::new(RichText::new(txt).size(w.font_size).color(fg)).min_size(egui::vec2(w.w, w.h)));
                    }
                    UiWidgetKind::Slider => {
                        let mut val = args.scripts.ui_value(&w.id);
                        let label = format!("{}: {:.2}", w.id, val);
                        ui.add(egui::Slider::new(&mut val, 0.0..=1.0).text(label));
                    }
                    UiWidgetKind::Toggle => {
                        let mut val = args.scripts.ui_value(&w.id) > 0.5;
                        ui.checkbox(&mut val, format!("{}", w.id));
                    }
                    UiWidgetKind::Panel => {
                        let bg = egui::Color32::from_rgba_premultiplied(
                            (w.bg_color[0] * 255.0) as u8, (w.bg_color[1] * 255.0) as u8,
                            (w.bg_color[2] * 255.0) as u8, (w.bg_color[3] * 255.0) as u8,
                        );
                        egui::Frame::new().fill(bg).corner_radius(8.0).inner_margin(egui::Margin::same(12)).show(ui, |ui| {
                            let title = args.scripts.ui_text(&w.id).unwrap_or_else(|| w.id.clone());
                            ui.label(RichText::new(title).size(w.font_size).color(fg).strong());
                        });
                    }
                    UiWidgetKind::ProgressRing => {
                        let v = args.scripts.ui_value(&w.id).clamp(0.0, 1.0);
                        ui.label(RichText::new(format!("{}: {:.0}%", w.id, v * 100.0)).size(w.font_size).color(fg));
                    }
                    UiWidgetKind::Meter => {
                        ui.label(RichText::new(format!("{}: {:.0}%", w.id, args.scripts.ui_value(&w.id) * 100.0)).size(w.font_size).color(fg));
                    }
                }
            });
    }

    if *show_material_editor {
        egui::Window::new("Material Editor")
            .default_pos([860.0, 120.0])
            .default_size([380.0, 320.0])
            .show(ctx, |ui| {
                ui.label("Material editor workspace (integrated panel).");
                ui.label("Use Details panel for live material instance edits.");
                ui.label("Double-click a .mat/.material asset in Content Browser to jump here.");
                if ui.button("Close").clicked() {
                    *show_material_editor = false;
                }
            });
    }

    if *show_foliage_editor {
        egui::Window::new("Foliage Editor")
            .default_pos([860.0, 460.0])
            .default_size([380.0, 300.0])
            .show(ctx, |ui| {
                ui.label("Foliage editor workspace (integrated panel).");
                ui.label("Use Details panel Foliage tools for brush/ring/remove controls.");
                ui.label("Double-click a .fol/.foliage asset in Content Browser to jump here.");
                if ui.button("Close").clicked() {
                    *show_foliage_editor = false;
                }
            });
    }

    if *show_icon_debug {
        const REQUIRED: &[&str] = &[
            "camera", "light", "sky", "sun", "player_start", "mesh", "prefab", "script",
            "material", "foliage", "file", "folder", "folder_open", "point_light", "fog", "volume",
        ];
        egui::Window::new("Icons Debug")
            .default_pos([480.0, 100.0])
            .default_size([460.0, 420.0])
            .show(ctx, |ui| {
                ui.label("Loaded icon stems from assets/icons");
                ui.label("Missing stems fall back to generic cards.");
                ui.separator();
                for stem in REQUIRED {
                    let ok = icon_texture_cache.contains_key(*stem);
                    let txt = if ok {
                        RichText::new(format!("OK   {}", stem)).color(Color32::from_rgb(120, 210, 150))
                    } else {
                        RichText::new(format!("MISS {}", stem)).color(Color32::from_rgb(235, 170, 120))
                    };
                    ui.label(txt);
                }
                ui.separator();
                if ui.button("Close").clicked() {
                    *show_icon_debug = false;
                }
            });
    }

    if *show_perf_safety_check {
        let tier = args.settings.runtime.gpu_scalability_tier.to_ascii_lowercase();
        let mut cost_score = 0u32;
        if args.renderer.features.pcss_enabled {
            cost_score += 2;
        }
        if args.renderer.features.volumetric_fog_enabled {
            cost_score += 2;
        }
        if args.renderer.features.voxel_gi_enabled {
            cost_score += 3;
        }
        if args.renderer.features.ssao_enabled {
            cost_score += 1;
        }
        if args.renderer.features.bloom_enabled {
            cost_score += 1;
        }
        if args.renderer.features.shadow_resolution >= 4096 {
            cost_score += 2;
        } else if args.renderer.features.shadow_resolution >= 2048 {
            cost_score += 1;
        }
        if args.renderer.features.pcf_samples >= 16 {
            cost_score += 1;
        }
        let expected_tier = if cost_score >= 8 {
            "Desktop High-End (RTX 3070+/RX 6800+)"
        } else if cost_score >= 5 {
            "Mid Desktop / Strong Laptop GPU (RTX 2060-3060 class)"
        } else if cost_score >= 2 {
            "Balanced Laptop/Desktop (GTX 1060/1650+)"
        } else {
            "Integrated/Entry GPU friendly"
        };
        let expected_fps = if cost_score >= 8 {
            "30-60 FPS at 1080p"
        } else if cost_score >= 5 {
            "45-90 FPS at 1080p"
        } else if cost_score >= 2 {
            "60+ FPS at 1080p"
        } else {
            "60+ FPS at 900p-1080p"
        };
        let risky_on_low_tier = (tier == "auto" || tier == "low") && cost_score >= 5;
        let risky_on_balanced_tier = tier == "balanced" && cost_score >= 8;

        egui::Window::new("Performance Safety Check")
            .default_pos([760.0, 120.0])
            .default_size([520.0, 360.0])
            .show(ctx, |ui| {
                ui.label("Pre-flight estimate before enabling heavy visual settings.");
                ui.separator();
                ui.label(format!("GPU tier target: {}", args.settings.runtime.gpu_scalability_tier));
                ui.label(format!("Feature cost score: {}", cost_score));
                ui.label(format!("Expected hardware tier: {}", expected_tier));
                ui.label(format!("Expected FPS tier: {}", expected_fps));
                ui.separator();
                if risky_on_low_tier {
                    ui.colored_label(
                        Color32::from_rgb(255, 180, 110),
                        "Warning: current feature mix is heavy for auto/low tier.",
                    );
                    ui.label("Suggested: disable Voxel GI + Volumetric Fog, keep 2048 shadows.");
                } else if risky_on_balanced_tier {
                    ui.colored_label(
                        Color32::from_rgb(255, 180, 110),
                        "Warning: cinematic mix may stutter on balanced tier hardware.",
                    );
                    ui.label("Suggested: lower shadow resolution or disable PCSS.");
                } else {
                    ui.colored_label(
                        Color32::from_rgb(120, 210, 150),
                        "Looks safe for the selected GPU tier target.",
                    );
                }
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Apply Balanced Preset").clicked() {
                        args.settings.render.preset = RenderPreset::Balanced;
                    }
                    if ui.button("Apply Mobile Preset").clicked() {
                        args.settings.render.preset = RenderPreset::Mobile;
                    }
                    if ui.button("Apply Cinematic Preset").clicked() {
                        args.settings.render.preset = RenderPreset::Cinematic;
                    }
                });
                if ui.button("Close").clicked() {
                    *show_perf_safety_check = false;
                }
            });
    }

    if *show_scene_manager {
        egui::Window::new("Scene Manager")
            .default_pos([620.0, 120.0])
            .default_size([560.0, 420.0])
            .show(ctx, |ui| {
                ui.label("Create, duplicate, rename, delete scenes and set startup scene.");
                ui.separator();
                ui.label(format!("Current loaded: {}", args.scene_path));
                ui.horizontal(|ui| {
                    ui.label("Startup scene");
                    egui::ComboBox::from_id_salt("startup_scene_combo")
                        .selected_text(args.settings.runtime.startup_scene_path.clone())
                        .show_ui(ui, |ui| {
                            for p in args.available_scene_paths {
                                ui.selectable_value(
                                    &mut args.settings.runtime.startup_scene_path,
                                    p.clone(),
                                    p,
                                );
                            }
                        });
                    if ui.button("Save Startup").clicked() {
                        match args.settings.save("engine_settings.toml") {
                            Ok(()) => args.error_log.push(
                                "[Scene] Saved startup scene to engine_settings.toml".to_string(),
                            ),
                            Err(e) => args.error_log.push(format!("[Scene] Save failed: {}", e)),
                        }
                    }
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Create");
                    ui.text_edit_singleline(scene_create_name);
                    if ui.button("Create Scene").clicked() {
                        let mut name = scene_create_name.trim().to_string();
                        if !name.ends_with(".scene") {
                            name.push_str(".scene");
                        }
                        let _ = fs::create_dir_all(crate::scene::SCENE_DIR);
                        let p = crate::scene::scene_path(&name);
                        if fs::write(&p, "").is_ok() {
                            *scene_picker_choice = p.clone();
                            *args.requested_scene_switch = Some(p);
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Duplicate as");
                    ui.text_edit_singleline(scene_duplicate_name);
                    if ui.button("Duplicate Current").clicked() {
                        let src = args.scene_path.clone();
                        let mut dst_name = scene_duplicate_name.trim().to_string();
                        if !dst_name.ends_with(".scene") {
                            dst_name.push_str(".scene");
                        }
                        let dst = crate::scene::scene_path(&dst_name);
                        match fs::copy(&src, &dst) {
                            Ok(_) => args.error_log.push(format!("[Scene] Duplicated to {}", dst)),
                            Err(e) => args.error_log.push(format!("[Scene] Duplicate failed: {}", e)),
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Rename current to");
                    ui.text_edit_singleline(scene_rename_name);
                    if ui.button("Rename Current").clicked() {
                        let src = args.scene_path.clone();
                        let mut dst_name = scene_rename_name.trim().to_string();
                        if !dst_name.ends_with(".scene") {
                            dst_name.push_str(".scene");
                        }
                        let dst = crate::scene::scene_path(&dst_name);
                        match fs::rename(&src, &dst) {
                            Ok(_) => {
                                *args.requested_scene_switch = Some(dst.clone());
                                args.settings.runtime.startup_scene_path = dst;
                            }
                            Err(e) => args.error_log.push(format!("[Scene] Rename failed: {}", e)),
                        }
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Delete Current Scene").clicked() {
                        let cur = args.scene_path.clone();
                        if cur.ends_with("main.scene") {
                            args.error_log.push(
                                "[Scene] Refused to delete main.scene from manager.".to_string(),
                            );
                        } else {
                            match fs::remove_file(&cur) {
                                Ok(()) => {
                                    args.error_log.push(format!("[Scene] Deleted {}", cur));
                                    *args.requested_scene_switch =
                                        Some(args.settings.runtime.startup_scene_path.clone());
                                }
                                Err(e) => args.error_log.push(format!("[Scene] Delete failed: {}", e)),
                            }
                        }
                    }
                    if ui.button("Close").clicked() {
                        *show_scene_manager = false;
                    }
                });
            });
    }

    if *show_project_launcher {
        egui::Window::new("Project Launcher")
            .default_pos([420.0, 120.0])
            .default_size([560.0, 360.0])
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
                        let _ = std::fs::create_dir_all(p.join(crate::scene::SCENE_DIR));
                        let scene = p.join(crate::scene::SCENE_DIR).join("main.scene");
                        if !scene.exists() {
                            let _ = std::fs::write(&scene, "");
                        }
                        let _ = std::fs::write(p.join("engine_settings.toml"), std::fs::read_to_string("engine_settings.toml").unwrap_or_default());
                        ui.ctx().copy_text(p.to_string_lossy().to_string());
                    }
                }
                if ui.button("Close Launcher").clicked() {
                    *show_project_launcher = false;
                }
                ui.separator();
                ui.label(RichText::new("External Script Editor").strong());
                ui.label("When you double-click a script in Content Browser, this command is used.");
                ui.label("Use {file} where the script path should be inserted.");
                ui.horizontal_wrapped(|ui| {
                    if ui.button("VS Code").clicked() {
                        *args.preferred_script_editor = "code -r \"{file}\"".to_string();
                    }
                    if ui.button("Notepad++").clicked() {
                        *args.preferred_script_editor = "notepad++ \"{file}\"".to_string();
                    }
                    if ui.button("Sublime").clicked() {
                        *args.preferred_script_editor = "subl \"{file}\"".to_string();
                    }
                    if ui.button("Rider").clicked() {
                        *args.preferred_script_editor = "rider64 \"{file}\"".to_string();
                    }
                    if ui.button("Visual Studio").clicked() {
                        *args.preferred_script_editor = "devenv \"{file}\"".to_string();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Command");
                    ui.text_edit_singleline(args.preferred_script_editor);
                    if ui.button("Test Open").clicked() {
                        let test_file = format!("{}/player.lua", args.scripts_dir);
                        if let Err(err) = open_external_editor(args.preferred_script_editor, &test_file) {
                            args.error_log.push(format!("[EditorPicker] Test open failed: {}", err));
                        }
                    }
                });
                ui.label("Examples:");
                ui.monospace("code -r \"{file}\"");
                ui.monospace("notepad \"{file}\"");
                ui.monospace("notepad++ \"{file}\"");
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
                                let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
                                let col = mesh_thumb_color(mesh_thumbnail_color_cache, &name);
                                thumbnail_card(ui, "PREFAB", &name, Some(col));
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
    let world_x = args.terrain_cursor_x as f32 * args.terrain_world.cell_size - 32.0;
    let world_z = args.terrain_cursor_z as f32 * args.terrain_world.cell_size - 32.0;
    args.terrain_world.auto_surface_color_world(world_x, world_z)
}

fn save_dock_layout(dock_state: &DockState<EditorDockTab>, workspace_preset: &str) {
    let dir = editor_persist::trinity_data_dir();
    let _ = fs::create_dir_all(&dir);
    let file = EditorDockLayoutFile {
        workspace_preset: workspace_preset.to_string(),
        dock_state: dock_state.clone(),
    };
    if let Ok(bytes) = serde_json::to_vec_pretty(&file) {
        let _ = fs::write(editor_persist::editor_dock_layout_path(), bytes);
    }
}

fn load_saved_dock_layout() -> Option<(String, DockState<EditorDockTab>)> {
    let bytes = fs::read(editor_persist::editor_dock_layout_path()).ok()?;
    let f: EditorDockLayoutFile = serde_json::from_slice(&bytes).ok()?;
    Some((f.workspace_preset, f.dock_state))
}

fn apply_workspace_preset(dock_state: &mut DockState<EditorDockTab>, preset: &str) {
    match preset {
        "Level Design" => {
            *dock_state = DockState::new(vec![EditorDockTab::Viewport]);
            let [main, _left] = dock_state
                .main_surface_mut()
                .split_left(NodeIndex::root(), 0.24, vec![EditorDockTab::Outliner]);
            dock_state
                .main_surface_mut()
                .split_right(main, 0.30, vec![EditorDockTab::Details, EditorDockTab::Profiler]);
            dock_state
                .main_surface_mut()
                .split_below(NodeIndex::root(), 0.72, vec![EditorDockTab::ContentBrowser, EditorDockTab::Levels]);
        }
        "Scripting" => {
            *dock_state = DockState::new(vec![EditorDockTab::ContentBrowser, EditorDockTab::Console]);
            dock_state
                .main_surface_mut()
                .split_right(NodeIndex::root(), 0.55, vec![EditorDockTab::Viewport]);
        }
        _ => {
            *dock_state = DockState::new(vec![EditorDockTab::Viewport]);
            let [main, left] = dock_state
                .main_surface_mut()
                .split_left(NodeIndex::root(), 0.20, vec![EditorDockTab::Outliner]);
            let [_m, _r] = dock_state
                .main_surface_mut()
                .split_right(main, 0.26, vec![EditorDockTab::Details]);
            dock_state
                .main_surface_mut()
                .split_below(left, 0.60, vec![EditorDockTab::ContentBrowser, EditorDockTab::Console, EditorDockTab::Profiler, EditorDockTab::Levels]);
        }
    }
}

fn refresh_icon_textures(
    cache: &mut HashMap<String, egui::TextureHandle>,
    registry: &IconRegistry,
    ctx: &egui::Context,
) {
    for (stem, path) in &registry.icons {
        if cache.contains_key(stem) {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !ext.eq_ignore_ascii_case("png") {
            continue;
        }
        let Ok(img) = image::open(path) else {
            continue;
        };
        let rgba = img.to_rgba8();
        // egui/wgpu panics if a texture exceeds the max texture side (2048 is
        // the portable maximum — the small-but-portable atlas limit used across
        // egui backends). Downscale oversized art (e.g. a 3000×2000 icon) so a
        // single bad PNG can never kill the editor.
        let max_side = 2048u32;
        let (pixels, w, h) = if rgba.width().max(rgba.height()) > max_side {
            let scale = max_side as f32 / rgba.width().max(rgba.height()) as f32;
            let nw = ((rgba.width() as f32 * scale).max(1.0)) as u32;
            let nh = ((rgba.height() as f32 * scale).max(1.0)) as u32;
            let resized = image::imageops::resize(&rgba, nw, nh, image::imageops::FilterType::Lanczos3);
            (resized.into_raw(), nw, nh)
        } else {
            let (w, h) = (rgba.width(), rgba.height());
            (rgba.into_raw(), w, h)
        };
        let size = [w as usize, h as usize];
        let color = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
        let h = ctx.load_texture(format!("icon_{stem}"), color, egui::TextureOptions::LINEAR);
        cache.insert(stem.clone(), h);
    }
}

fn ensure_splash_logo_texture(ctx: &egui::Context, slot: &mut Option<egui::TextureHandle>) {
    if slot.is_some() {
        return;
    }
    let Ok(img) = image::open(SPLASH_LOGO_PATH) else {
        return;
    };
    let rgba = img.to_rgba8();
    // Same portable 2048 cap as refresh_icon_textures; a splash art file larger
    // than that must never take the editor down with a texture panic.
    let max_side = 2048u32;
    let (pixels, w, h) = if rgba.width().max(rgba.height()) > max_side {
        let scale = max_side as f32 / rgba.width().max(rgba.height()) as f32;
        let nw = ((rgba.width() as f32 * scale).max(1.0)) as u32;
        let nh = ((rgba.height() as f32 * scale).max(1.0)) as u32;
        let resized = image::imageops::resize(&rgba, nw, nh, image::imageops::FilterType::Lanczos3);
        (resized.into_raw(), nw, nh)
    } else {
        let (w, h) = (rgba.width(), rgba.height());
        (rgba.into_raw(), w, h)
    };
    let size = [w as usize, h as usize];
    let color = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    let handle = ctx.load_texture("splash_logo", color, egui::TextureOptions::LINEAR);
    *slot = Some(handle);
}

fn apply_icon_alias_fallbacks(cache: &mut HashMap<String, egui::TextureHandle>) {
    // Alias stems let us keep UI icon names stable while art packs evolve.
    const ALIASES: &[(&str, &[&str])] = &[
        ("sun", &["light", "sky", "file", "script"]),
        ("point_light", &["light", "sun", "file", "script"]),
        ("fog", &["sky", "cloud", "file", "script"]),
        ("volume", &["fog", "sky", "file", "script"]),
        ("player_start", &["camera", "mesh", "file", "script"]),
    ];
    for (target, fallback_chain) in ALIASES {
        if cache.contains_key(*target) {
            continue;
        }
        for candidate in *fallback_chain {
            if let Some(tex) = cache.get(*candidate).cloned() {
                cache.insert((*target).to_string(), tex);
                break;
            }
        }
    }
}

fn icon_row(
    ui: &mut egui::Ui,
    icon_tex: &HashMap<String, egui::TextureHandle>,
    stem: &str,
    label: &str,
) {
    ui.horizontal(|ui| {
        if let Some(tex) = icon_tex.get(stem) {
            ui.add(egui::Image::new((tex.id(), egui::vec2(16.0, 16.0))));
        } else {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
            let p = ui.painter();
            p.rect_filled(rect, 3.0, Color32::from_rgb(28, 30, 36));
            p.rect_stroke(
                rect.shrink(2.0),
                2.0,
                egui::Stroke::new(1.0, Color32::from_rgb(120, 126, 138)),
                egui::StrokeKind::Middle,
            );
        }
        ui.label(label);
    });
}

fn console_line_rich(line: &str) -> RichText {
    let l = line.to_ascii_lowercase();
    let base = RichText::new(line).monospace();
    if l.contains("error") || l.contains("failed") || l.contains("panic") {
        base.color(Color32::from_rgb(255, 130, 130))
    } else if line.contains("[Hub]") || line.contains("[Editor]") {
        base.color(Color32::from_rgb(230, 195, 120))
    } else if line.contains("[Material]") || line.contains("[Lua]") || line.contains("[Script]") {
        base.color(Color32::from_rgb(150, 195, 255))
    } else if line.contains("[Foliage]") || line.contains("[Terrain]") {
        base.color(Color32::from_rgb(130, 220, 165))
    } else if line.contains("[Quality]") {
        base.color(Color32::from_rgb(200, 175, 255))
    } else if line.contains("[Content]") {
        base.color(Color32::from_rgb(200, 200, 140))
    } else {
        base.color(Color32::from_rgb(205, 208, 215))
    }
}

fn editor_tool_window_header(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(16, 19, 25))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(37, 44, 55)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.label(RichText::new(title).strong().size(18.0));
            ui.label(
                RichText::new(subtitle)
                    .small()
                    .color(Color32::from_rgb(148, 158, 173)),
            );
        });
    ui.add_space(8.0);
}

fn editor_tool_card<R>(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui) -> R) {
    egui::Frame::new()
        .fill(Color32::from_rgb(14, 17, 22))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(33, 39, 49)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.label(RichText::new(title).small().strong());
            ui.separator();
            body(ui);
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

pub(crate) fn pick_entity_in_viewport(
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

pub(crate) fn project_to_screen(camera: &dyn Camera, rect: egui::Rect, world: glam::Vec3) -> Option<egui::Pos2> {
    let clip = camera.view_projection_matrix() * glam::Vec4::new(world.x, world.y, world.z, 1.0);
    if clip.w <= 0.0001 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    let sx = rect.left() + ((ndc.x + 1.0) * 0.5) * rect.width();
    let sy = rect.top() + ((1.0 - (ndc.y + 1.0) * 0.5) * rect.height());
    Some(egui::pos2(sx, sy))
}

/// Unprojects a screen point onto the world plane passing through
/// `plane_point` with normal `plane_normal` (used for gizmo dragging).
pub(crate) fn screen_to_plane_world(
    camera: &dyn Camera,
    rect: egui::Rect,
    pointer: egui::Pos2,
    plane_point: glam::Vec3,
    plane_normal: glam::Vec3,
) -> Option<glam::Vec3> {
    let inv_vp = camera.view_projection_matrix().inverse();
    let ndc_x = ((pointer.x - rect.left()) / rect.width()) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((pointer.y - rect.top()) / rect.height()) * 2.0;
    let near = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let near = (near.truncate() / near.w.max(1e-6));
    let far = (far.truncate() / far.w.max(1e-6));
    let dir = (far - near).normalize();
    let denom = plane_normal.dot(dir);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = plane_normal.dot(plane_point - near) / denom;
    if t <= 0.0 {
        return None;
    }
    Some(near + dir * t)
}

fn dist_point_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let ab2 = ab.x * ab.x + ab.y * ab.y;
    if ab2 <= 1e-6 {
        return ((p.x - a.x).powi(2) + (p.y - a.y).powi(2)).sqrt();
    }
    let t = ((ap.x * ab.x + ap.y * ab.y) / ab2).clamp(0.0, 1.0);
    let c = a + ab * t;
    ((p.x - c.x).powi(2) + (p.y - c.y).powi(2)).sqrt()
}

fn nearest_axis_hit(
    pointer: egui::Pos2,
    origin: egui::Pos2,
    x_end: egui::Pos2,
    y_end: egui::Pos2,
    z_end: egui::Pos2,
    xy_center: egui::Pos2,
    yz_center: egui::Pos2,
    zx_center: egui::Pos2,
    mode: GizmoMode,
) -> Option<GizmoAxis> {
    let d = |a: egui::Pos2, b: egui::Pos2| ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
    if matches!(mode, GizmoMode::Rotate) {
        let r = d(pointer, origin);
        let rings = [(GizmoAxis::X, 42.0f32), (GizmoAxis::Y, 54.0f32), (GizmoAxis::Z, 66.0f32)];
        let mut best: Option<(GizmoAxis, f32)> = None;
        for (axis, rr) in rings {
            let err = (r - rr).abs();
            if err <= 8.0 {
                match best {
                    Some((_, be)) if err >= be => {}
                    _ => best = Some((axis, err)),
                }
            }
        }
        return best.map(|b| b.0);
    }

    if matches!(mode, GizmoMode::Scale) && d(pointer, origin) <= 10.0 {
        return Some(GizmoAxis::Uniform);
    }
    let plane_hits = [
        (GizmoAxis::XY, d(pointer, xy_center)),
        (GizmoAxis::YZ, d(pointer, yz_center)),
        (GizmoAxis::ZX, d(pointer, zx_center)),
    ];
    let mut best_plane: Option<(GizmoAxis, f32)> = None;
    for (ax, dist) in plane_hits {
        if dist <= 10.0 {
            if best_plane.map_or(true, |(_, best)| dist < best) {
                best_plane = Some((ax, dist));
            }
        }
    }
    if let Some((ax, _)) = best_plane {
        return Some(ax);
    }
    if d(pointer, origin) < 8.0 {
        return if matches!(mode, GizmoMode::Move) {
            Some(GizmoAxis::Uniform)
        } else {
            None
        };
    }
    let dx = dist_point_segment(pointer, origin, x_end);
    let dy = dist_point_segment(pointer, origin, y_end);
    let dz = dist_point_segment(pointer, origin, z_end);
    let mut best = (GizmoAxis::X, dx);
    if dy < best.1 {
        best = (GizmoAxis::Y, dy);
    }
    if dz < best.1 {
        best = (GizmoAxis::Z, dz);
    }
    if best.1 > 10.0 {
        None
    } else {
        Some(best.0)
    }
}

pub(crate) fn draw_transform_gizmo(
    ui: &mut egui::Ui,
    args: &mut UiFrameArgs<'_>,
    entity: hecs::Entity,
    rect: egui::Rect,
    pointer: Option<egui::Pos2>,
    mode: &mut GizmoMode,
    drag: &mut Option<GizmoDragState>,
    space: GizmoSpace,
    axis_lock: Option<GizmoAxis>,
    snap_enabled: bool,
    snap_translate: f32,
    snap_rotate_deg: f32,
    snap_scale: f32,
) {
    let (pos_x, pos_y, pos_z) = {
        let Ok(pos) = args.world.get::<&components::Position>(entity) else { return; };
        (pos.x, pos.y, pos.z)
    };
    let origin_world = glam::Vec3::new(pos_x, pos_y, pos_z);
    let local_rot = args.world.get::<&components::Rotation>(entity).ok().map(|r| *r);
    let (axis_x_world, axis_y_world, axis_z_world) = if matches!(space, GizmoSpace::Local) {
        if let Some(r) = local_rot {
            let q = glam::Quat::from_euler(glam::EulerRot::XYZ, r.pitch, r.yaw, r.roll);
            (q * glam::Vec3::X, q * glam::Vec3::Y, q * glam::Vec3::Z)
        } else {
            (glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z)
        }
    } else {
        (glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z)
    };
    let axis_len = 1.0;
    let Some(origin) = project_to_screen(args.camera, rect, origin_world) else { return; };
    let Some(x_end) = project_to_screen(args.camera, rect, origin_world + axis_x_world * axis_len) else { return; };
    let Some(y_end) = project_to_screen(args.camera, rect, origin_world + axis_y_world * axis_len) else { return; };
    let Some(z_end) = project_to_screen(args.camera, rect, origin_world + axis_z_world * axis_len) else { return; };
    let vx = x_end - origin;
    let vy = y_end - origin;
    let vz = z_end - origin;
    let xy_center = origin + vx * 0.20 + vy * 0.20;
    let yz_center = origin + vy * 0.20 + vz * 0.20;
    let zx_center = origin + vz * 0.20 + vx * 0.20;

    let p = ui.painter();
    let x_col = Color32::from_rgb(232, 84, 84);
    let y_col = Color32::from_rgb(96, 224, 112);
    let z_col = Color32::from_rgb(105, 165, 255);
    let hover_axis = if drag.is_none() {
        pointer.and_then(|ptr| {
            let hit = nearest_axis_hit(
                ptr,
                origin,
                x_end,
                y_end,
                z_end,
                xy_center,
                yz_center,
                zx_center,
                *mode,
            );
            if let Some(lock) = axis_lock {
                match lock {
                    GizmoAxis::X => hit.filter(|a| matches!(a, GizmoAxis::X)),
                    GizmoAxis::Y => hit.filter(|a| matches!(a, GizmoAxis::Y)),
                    GizmoAxis::Z => hit.filter(|a| matches!(a, GizmoAxis::Z)),
                    _ => hit,
                }
            } else {
                hit
            }
        })
    } else {
        None
    };
    let active_axis = drag.map(|d| d.axis);
    let stroke_for = |axis: GizmoAxis, base: Color32| {
        if Some(axis) == active_axis {
            egui::Stroke::new(4.2, base)
        } else if Some(axis) == hover_axis {
            egui::Stroke::new(3.5, base)
        } else {
            egui::Stroke::new(2.4, base)
        }
    };
    let glow = Color32::from_rgba_unmultiplied(250, 250, 255, 36);
    p.circle_filled(origin, 5.8, Color32::from_rgb(28, 30, 36));
    p.circle_stroke(origin, 6.6, egui::Stroke::new(1.2, glow));
    p.line_segment([origin, x_end], stroke_for(GizmoAxis::X, x_col));
    p.line_segment([origin, y_end], stroke_for(GizmoAxis::Y, y_col));
    p.line_segment([origin, z_end], stroke_for(GizmoAxis::Z, z_col));
    if !matches!(*mode, GizmoMode::Rotate) {
        let plane_col = |base: Color32| Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 55);
        p.circle_filled(xy_center, 5.0, plane_col(Color32::from_rgb(220, 210, 80)));
        p.circle_filled(yz_center, 5.0, plane_col(Color32::from_rgb(110, 210, 200)));
        p.circle_filled(zx_center, 5.0, plane_col(Color32::from_rgb(210, 120, 210)));
    }

    let draw_triangle_handle = |p: &egui::Painter, origin: egui::Pos2, end: egui::Pos2, col: Color32| {
        let axis = end - origin;
        let len = axis.length().max(1.0);
        let dir = axis / len;
        let perp = egui::vec2(-dir.y, dir.x);
        let tip = end + dir * 5.0;
        let base_c = end - dir * 8.0;
        let b1 = base_c + perp * 4.5;
        let b2 = base_c - perp * 4.5;
        p.add(egui::Shape::convex_polygon(
            vec![tip, b1, b2],
            col,
            egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(10, 10, 12, 180)),
        ));
    };
    let draw_cube_handle = |p: &egui::Painter, end: egui::Pos2, col: Color32| {
        let front = egui::Rect::from_center_size(end, egui::vec2(9.0, 9.0));
        let off = egui::vec2(2.4, -2.2);
        let back = front.translate(off);
        p.rect_filled(back, 1.2, Color32::from_rgba_unmultiplied(col.r() / 2, col.g() / 2, col.b() / 2, 220));
        p.rect_filled(front, 1.2, col);
        p.line_segment([front.left_top(), back.left_top()], egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(16, 16, 18, 180)));
        p.line_segment([front.right_top(), back.right_top()], egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(16, 16, 18, 180)));
        p.line_segment([front.right_bottom(), back.right_bottom()], egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(16, 16, 18, 180)));
        p.line_segment([back.left_top(), back.right_top()], egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(16, 16, 18, 180)));
        p.line_segment([back.right_top(), back.right_bottom()], egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(16, 16, 18, 180)));
    };

    match *mode {
        GizmoMode::Move => {
            draw_triangle_handle(p, origin, x_end, x_col);
            draw_triangle_handle(p, origin, y_end, y_col);
            draw_triangle_handle(p, origin, z_end, z_col);
        }
        GizmoMode::Scale => {
            draw_cube_handle(p, x_end, x_col);
            draw_cube_handle(p, y_end, y_col);
            draw_cube_handle(p, z_end, z_col);
            draw_cube_handle(p, origin, Color32::from_rgb(235, 235, 235));
        }
        GizmoMode::Rotate => {
            p.circle_stroke(origin, 42.0, stroke_for(GizmoAxis::X, x_col));
            p.circle_stroke(origin, 54.0, stroke_for(GizmoAxis::Y, y_col));
            p.circle_stroke(origin, 66.0, stroke_for(GizmoAxis::Z, z_col));
        }
    }
    let lock_label = |ax: GizmoAxis, col: Color32| {
        if Some(ax) == axis_lock { Color32::from_rgb(255, 255, 100) } else { col }
    };
    p.text(x_end + egui::vec2(5.0, -5.0), egui::Align2::LEFT_BOTTOM, "X", egui::FontId::proportional(11.0), lock_label(GizmoAxis::X, x_col));
    p.text(y_end + egui::vec2(5.0, -5.0), egui::Align2::LEFT_BOTTOM, "Y", egui::FontId::proportional(11.0), lock_label(GizmoAxis::Y, y_col));
    p.text(z_end + egui::vec2(5.0, -5.0), egui::Align2::LEFT_BOTTOM, "Z", egui::FontId::proportional(11.0), lock_label(GizmoAxis::Z, z_col));

    if let Some(ptr) = pointer {
        let primary_down = ui.ctx().input(|i| i.pointer.primary_down());
        if drag.is_none() && primary_down {
            if let Some(axis) = {
                let hit = nearest_axis_hit(
                    ptr,
                    origin,
                    x_end,
                    y_end,
                    z_end,
                    xy_center,
                    yz_center,
                    zx_center,
                    *mode,
                );
                if let Some(lock) = axis_lock {
                    match lock {
                        GizmoAxis::X => hit.filter(|a| matches!(a, GizmoAxis::X)),
                        GizmoAxis::Y => hit.filter(|a| matches!(a, GizmoAxis::Y)),
                        GizmoAxis::Z => hit.filter(|a| matches!(a, GizmoAxis::Z)),
                        _ => hit,
                    }
                } else {
                    hit
                }
            } {
                let rot = args.world.get::<&components::Rotation>(entity).ok();
                let rend = args.world.get::<&components::Renderable>(entity).ok();
                let (axis_screen, axis_screen_2) = match axis {
                    GizmoAxis::X => (x_end - origin, egui::Vec2::ZERO),
                    GizmoAxis::Y => (y_end - origin, egui::Vec2::ZERO),
                    GizmoAxis::Z => (z_end - origin, egui::Vec2::ZERO),
                    GizmoAxis::XY => (x_end - origin, y_end - origin),
                    GizmoAxis::YZ => (y_end - origin, z_end - origin),
                    GizmoAxis::ZX => (z_end - origin, x_end - origin),
                    GizmoAxis::Uniform => (egui::vec2(1.0, -1.0), egui::Vec2::ZERO),
                };
                let axis_dir_screen = axis_screen.normalized();
                let axis_dir_screen_2 = if axis_screen_2.length_sq() > 1e-4 {
                    axis_screen_2.normalized()
                } else {
                    egui::Vec2::ZERO
                };
                *drag = Some(GizmoDragState {
                    axis,
                    pointer_start: ptr,
                    pos_start: [pos_x, pos_y, pos_z],
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
                    axis_dir_screen,
                    axis_dir_screen_2,
                });
            }
        }
        if let Some(d) = *drag {
            if primary_down {
                let pointer_delta = ptr - d.pointer_start;
                let delta_px = pointer_delta.dot(d.axis_dir_screen);
                match *mode {
                    GizmoMode::Move => {
                        if let Ok(mut p) = args.world.get::<&mut components::Position>(entity) {
                            let mut amt = delta_px * 0.012;
                            let mut amt2 = pointer_delta.dot(d.axis_dir_screen_2) * 0.012;
                            if snap_enabled {
                                amt = (amt / snap_translate).round() * snap_translate;
                                amt2 = (amt2 / snap_translate).round() * snap_translate;
                            }
                            let axis_vec = |ax: GizmoAxis| match ax {
                                GizmoAxis::X => axis_x_world,
                                GizmoAxis::Y => axis_y_world,
                                GizmoAxis::Z => axis_z_world,
                                _ => glam::Vec3::ZERO,
                            };
                            let apply = match d.axis {
                                GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => axis_vec(d.axis) * amt,
                                GizmoAxis::XY => axis_vec(GizmoAxis::X) * amt + axis_vec(GizmoAxis::Y) * amt2,
                                GizmoAxis::YZ => axis_vec(GizmoAxis::Y) * amt + axis_vec(GizmoAxis::Z) * amt2,
                                GizmoAxis::ZX => axis_vec(GizmoAxis::Z) * amt + axis_vec(GizmoAxis::X) * amt2,
                                GizmoAxis::Uniform => (axis_x_world + axis_y_world + axis_z_world) * (amt * 0.333),
                            };
                            let base = glam::Vec3::new(d.pos_start[0], d.pos_start[1], d.pos_start[2]);
                            let result = base + apply;
                            p.x = result.x;
                            p.y = result.y;
                            p.z = result.z;
                        }
                    }
                    GizmoMode::Rotate => {
                        if let Ok(mut r) = args.world.get::<&mut components::Rotation>(entity) {
                            let mut amt = delta_px * 0.01;
                            if snap_enabled {
                                let snap_rad = snap_rotate_deg.to_radians();
                                amt = (amt / snap_rad).round() * snap_rad;
                            }
                            match d.axis {
                                GizmoAxis::X => r.pitch = d.rot_start[0] + amt,
                                GizmoAxis::Y => r.yaw = d.rot_start[1] + amt,
                                GizmoAxis::Z => r.roll = d.rot_start[2] + amt,
                                _ => {}
                            }
                            // Special behavior controls: rotation also drives global sun/fog direction knobs.
                            args.renderer.features.sun_azimuth_deg = r.yaw.to_degrees().rem_euclid(360.0);
                            args.renderer.features.sun_elevation_deg = r.pitch.to_degrees().clamp(-5.0, 89.0);
                            if args.renderer.features.volumetric_fog_enabled {
                                let face = glam::Vec3::new(r.yaw.cos(), r.pitch.sin(), r.yaw.sin()).normalize_or_zero();
                                args.renderer.features.fog_density =
                                    (0.015 + face.y.abs() * 0.07).clamp(0.0, 0.2);
                            }
                        }
                    }
                    GizmoMode::Scale => {
                        if let Ok(mut r) = args.world.get::<&mut components::Renderable>(entity) {
                            let mut amt = (delta_px * 0.01).max(-0.95);
                            let mut amt2 = (pointer_delta.dot(d.axis_dir_screen_2) * 0.01).max(-0.95);
                            if snap_enabled {
                                amt = (amt / snap_scale).round() * snap_scale;
                                amt2 = (amt2 / snap_scale).round() * snap_scale;
                            }
                            match d.axis {
                                GizmoAxis::X => r.scale[0] = (d.scale_start[0] + amt).max(0.05),
                                GizmoAxis::Y => r.scale[1] = (d.scale_start[1] + amt).max(0.05),
                                GizmoAxis::Z => r.scale[2] = (d.scale_start[2] + amt).max(0.05),
                                GizmoAxis::XY => {
                                    r.scale[0] = (d.scale_start[0] + amt).max(0.05);
                                    r.scale[1] = (d.scale_start[1] + amt2).max(0.05);
                                }
                                GizmoAxis::YZ => {
                                    r.scale[1] = (d.scale_start[1] + amt).max(0.05);
                                    r.scale[2] = (d.scale_start[2] + amt2).max(0.05);
                                }
                                GizmoAxis::ZX => {
                                    r.scale[2] = (d.scale_start[2] + amt).max(0.05);
                                    r.scale[0] = (d.scale_start[0] + amt2).max(0.05);
                                }
                                GizmoAxis::Uniform => {
                                    let u = (delta_px * 0.01).max(-0.95);
                                    let uu = if snap_enabled {
                                        (u / snap_scale).round() * snap_scale
                                    } else {
                                        u
                                    };
                                    r.scale[0] = (d.scale_start[0] + uu).max(0.05);
                                    r.scale[1] = (d.scale_start[1] + uu).max(0.05);
                                    r.scale[2] = (d.scale_start[2] + uu).max(0.05);
                                }
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

pub(crate) fn texture_thumbnail<'a>(
    ctx: &egui::Context,
    cache: &'a mut HashMap<String, egui::TextureHandle>,
    path: &str,
) -> Option<&'a egui::TextureHandle> {
    if !cache.contains_key(path) {
        let img = image::open(path).ok()?;
        let thumb = img.thumbnail(96, 54).to_rgba8();
        let size = [thumb.width() as usize, thumb.height() as usize];
        let pixels = thumb.into_raw();
        let color = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
        let handle = ctx.load_texture(format!("thumb_{}", path), color, egui::TextureOptions::LINEAR);
        cache.insert(path.to_string(), handle);
    }
    cache.get(path)
}

fn mesh_thumb_color(cache: &mut HashMap<String, Color32>, path: &str) -> Color32 {
    if let Some(col) = cache.get(path).copied() {
        return col;
    }
    let mut hash: u32 = 2166136261;
    for b in path.as_bytes() {
        hash ^= *b as u32;
        hash = hash.wrapping_mul(16777619);
    }
    let r = 60 + ((hash & 0xFF) as u8 / 3);
    let g = 60 + (((hash >> 8) & 0xFF) as u8 / 3);
    let b = 60 + (((hash >> 16) & 0xFF) as u8 / 3);
    let col = Color32::from_rgb(r, g, b);
    cache.insert(path.to_string(), col);
    col
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
    let x = args.terrain_cursor_x as f32 * args.terrain_world.cell_size - 32.0;
    let z = args.terrain_cursor_z as f32 * args.terrain_world.cell_size - 32.0;
    let y = args.terrain_world.height_at(x, z) + 0.5;
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
        let mut body = components::RigidBody::dynamic();
        body.friction = 0.6;
        let _ = args.world.insert(
            entity,
            (
                body,
                components::Collider {
                    half_w: scale[0].abs() * 0.5,
                    half_h: scale[1].abs() * 0.5,
                    half_d: scale[2].abs() * 0.5,
                    layer: 1,
                    mask: 1,
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
