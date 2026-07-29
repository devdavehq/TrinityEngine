// src/editor_ui/panels/anim_graph_editor.rs
// Visual animation graph editor — node-graph UI for designing animation state machines.
//
// ── UX Design ────────────────────────────────────────────────────────────────
// The editor has three areas:
//   1. Toolbar: Add State, Set Initial, Build Graph, Clear All
//   2. Canvas: Draggable state nodes with connection ports
//   3. Properties Panel: Edit selected state/transition + condition editor
//
// ── Interaction ──────────────────────────────────────────────────────────────
// • Drag nodes: Left-click + drag on node body
// • Create transition: Drag from output port (right edge) → input port (left edge)
// • Delete transition: Click transition line, then Delete key or X button
// • Select node: Left-click on node
// • Pan canvas: Middle-mouse drag
// • Zoom: Scroll wheel
// • Set initial: Right-click node → "Set as Initial"
// • Context menu: Right-click on canvas for quick actions

use crate::animation::anim_graph::{
    AnimGraph, AnimStateNode, AnimTransition, TransitionCondition,
};
use egui::{Color32, Pos2, Rect, RichText, Vec2};

// ── Editor State ─────────────────────────────────────────────────────────────

pub struct AnimGraphEditorState {
    /// All state nodes on the canvas.
    pub nodes: Vec<EditorAnimState>,
    /// All transitions (edges) between states.
    pub transitions: Vec<EditorAnimTransition>,
    /// Index of the currently selected state (by editor id, not graph index).
    pub selected_state: Option<usize>,
    /// Index of the currently selected transition.
    pub selected_transition: Option<usize>,
    /// Index of the initial state (editor id).
    pub initial_state: Option<usize>,
    /// Canvas pan offset (pixels).
    pub pan: [f32; 2],
    /// Canvas zoom level.
    pub zoom: f32,
    /// Is the editor visible?
    pub visible: bool,
    /// Next editor-unique ID.
    next_id: usize,
    /// Are we in "connect" mode? (dragging a transition)
    connect_from: Option<usize>,
    /// Graph name.
    pub graph_name: String,
    /// Default blend duration for new transitions.
    pub default_blend: f32,
}

impl Default for AnimGraphEditorState {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            transitions: Vec::new(),
            selected_state: None,
            selected_transition: None,
            initial_state: None,
            pan: [0.0, 0.0],
            zoom: 1.0,
            visible: true,
            next_id: 1,
            connect_from: None,
            graph_name: "MainGraph".to_string(),
            default_blend: 0.2,
        }
    }
}

#[derive(Clone)]
pub struct EditorAnimState {
    /// Unique editor ID (not the same as graph index).
    pub id: usize,
    /// State name (e.g., "Idle", "Walk", "Run").
    pub name: String,
    /// Index of the AnimationClip in AnimationClips.
    pub clip_index: usize,
    /// Canvas position (pixels, unscaled).
    pub position: [f32; 2],
    /// Color for this state.
    pub color: Color32,
    /// Whether this is the initial state.
    pub is_initial: bool,
}

#[derive(Clone)]
pub struct EditorAnimTransition {
    /// Unique editor ID.
    pub id: usize,
    /// Source state editor ID.
    pub from_state: usize,
    /// Target state editor ID.
    pub to_state: usize,
    /// Transition conditions (ALL must be true).
    pub conditions: Vec<EditorCondition>,
    /// Blend duration override (0 = use graph default).
    pub blend_duration: f32,
    /// Priority (higher = checked first).
    pub priority: i32,
}

#[derive(Clone)]
pub struct EditorCondition {
    pub condition_type: EditorConditionType,
    /// Whether this condition is enabled.
    pub enabled: bool,
}

#[derive(Clone)]
pub enum EditorConditionType {
    FloatGreaterThan { param: String, threshold: f32 },
    FloatLessThan { param: String, threshold: f32 },
    FloatInRange { param: String, min: f32, max: f32 },
    BoolEquals { param: String, expected: bool },
    EnumEquals { param: String, expected: u32 },
    StringEquals { param: String, expected: String },
    TimeInStateGreaterThan { seconds: f32 },
}

impl EditorConditionType {
    fn label(&self) -> String {
        match self {
            Self::FloatGreaterThan { param, threshold } => format!("{} > {:.2}", param, threshold),
            Self::FloatLessThan { param, threshold } => format!("{} < {:.2}", param, threshold),
            Self::FloatInRange { param, min, max } => format!("{} in [{:.2}, {:.2}]", param, min, max),
            Self::BoolEquals { param, expected } => format!("{} == {}", param, expected),
            Self::EnumEquals { param, expected } => format!("{} == {}", param, expected),
            Self::StringEquals { param, expected } => format!("{} == \"{}\"", param, expected),
            Self::TimeInStateGreaterThan { seconds } => format!("time > {:.1}s", seconds),
        }
    }

    fn all_types() -> Vec<(String, fn() -> Self)> {
        vec![
            ("Float >".into(), || Self::FloatGreaterThan { param: "speed".into(), threshold: 0.5 }),
            ("Float <".into(), || Self::FloatLessThan { param: "speed".into(), threshold: 0.5 }),
            ("Float in Range".into(), || Self::FloatInRange { param: "speed".into(), min: 0.0, max: 1.0 }),
            ("Bool Equals".into(), || Self::BoolEquals { param: "is_grounded".into(), expected: true }),
            ("Enum Equals".into(), || Self::EnumEquals { param: "ai_state".into(), expected: 0 }),
            ("String Equals".into(), || Self::StringEquals { param: "ai_state".into(), expected: "idle".into() }),
            ("Time in State >".into(), || Self::TimeInStateGreaterThan { seconds: 0.5 }),
        ]
    }
}

impl AnimGraphEditorState {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            transitions: Vec::new(),
            selected_state: None,
            selected_transition: None,
            initial_state: None,
            pan: [0.0, 0.0],
            zoom: 1.0,
            visible: true,
            next_id: 1,
            connect_from: None,
            graph_name: "Character".to_string(),
            default_blend: 0.25,
        }
    }

    /// Add a new state at the given canvas position.
    pub fn add_state(&mut self, name: &str, clip_index: usize, position: [f32; 2]) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let color = state_color_for_index(self.nodes.len());
        self.nodes.push(EditorAnimState {
            id, name: name.to_string(), clip_index, position, color, is_initial: false,
        });
        if self.initial_state.is_none() {
            self.initial_state = Some(id);
            if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id) {
                n.is_initial = true;
            }
        }
        id
    }

    /// Add a transition between two states.
    pub fn add_transition(&mut self, from_id: usize, to_id: usize) -> Option<usize> {
        if from_id == to_id { return None; }
        if self.transitions.iter().any(|t| t.from_state == from_id && t.to_state == to_id) {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.transitions.push(EditorAnimTransition {
            id,
            from_state: from_id,
            to_state: to_id,
            conditions: Vec::new(),
            blend_duration: 0.0,
            priority: 0,
        });
        Some(id)
    }

    /// Set which state is the initial state.
    pub fn set_initial(&mut self, id: usize) {
        for n in &mut self.nodes {
            n.is_initial = n.id == id;
        }
        self.initial_state = Some(id);
    }

    /// Remove a state and all its transitions.
    pub fn remove_state(&mut self, id: usize) {
        self.nodes.retain(|n| n.id != id);
        self.transitions.retain(|t| t.from_state != id && t.to_state != id);
        if self.selected_state == Some(id) { self.selected_state = None; }
        if self.initial_state == Some(id) {
            self.initial_state = self.nodes.first().map(|n| n.id);
            if let Some(new_init) = self.initial_state {
                if let Some(n) = self.nodes.iter_mut().find(|n| n.id == new_init) {
                    n.is_initial = true;
                }
            }
        }
    }

    /// Remove a transition.
    pub fn remove_transition(&mut self, id: usize) {
        self.transitions.retain(|t| t.id != id);
        if self.selected_transition == Some(id) { self.selected_transition = None; }
    }

    /// Build an AnimGraph from the editor state.
    pub fn build_anim_graph(&self) -> Option<AnimGraph> {
        if self.nodes.is_empty() { return None; }

        let mut graph = AnimGraph::new(&self.graph_name);
        graph.default_blend_duration = self.default_blend;

        // Map editor IDs → graph indices.
        let mut id_to_graph_idx = std::collections::HashMap::new();
        for node in &self.nodes {
            let idx = graph.add_state(AnimStateNode::new(&node.name, node.clip_index));
            id_to_graph_idx.insert(node.id, idx);
        }

        // Set initial state.
        if let Some(init_id) = self.initial_state {
            if let Some(&idx) = id_to_graph_idx.get(&init_id) {
                graph.set_initial_state(idx);
            }
        }

        // Add transitions.
        for trans in &self.transitions {
            if let (Some(&from_idx), Some(&to_idx)) = (
                id_to_graph_idx.get(&trans.from_state),
                id_to_graph_idx.get(&trans.to_state),
            ) {
                let mut anim_trans = AnimTransition::new(to_idx);
                anim_trans.blend_duration = trans.blend_duration;
                anim_trans.priority = trans.priority;

                for ec in &trans.conditions {
                    if ec.enabled {
                        anim_trans.conditions.push(ec.to_runtime_condition());
                    }
                }

                graph.add_transition(from_idx, anim_trans);
            }
        }

        Some(graph)
    }
}

impl EditorCondition {
    fn to_runtime_condition(&self) -> TransitionCondition {
        match &self.condition_type {
            EditorConditionType::FloatGreaterThan { param, threshold } => {
                TransitionCondition::FloatGreaterThan { param: param.clone(), threshold: *threshold }
            }
            EditorConditionType::FloatLessThan { param, threshold } => {
                TransitionCondition::FloatLessThan { param: param.clone(), threshold: *threshold }
            }
            EditorConditionType::FloatInRange { param, min, max } => {
                TransitionCondition::FloatInRange { param: param.clone(), min: *min, max: *max }
            }
            EditorConditionType::BoolEquals { param, expected } => {
                TransitionCondition::BoolEquals { param: param.clone(), expected: *expected }
            }
            EditorConditionType::EnumEquals { param, expected } => {
                TransitionCondition::EnumEquals { param: param.clone(), expected: *expected }
            }
            EditorConditionType::StringEquals { param, expected } => {
                TransitionCondition::StringEquals { param: param.clone(), expected: expected.clone() }
            }
            EditorConditionType::TimeInStateGreaterThan { seconds } => {
                TransitionCondition::TimeInStateGreaterThan { seconds: *seconds }
            }
        }
    }
}

// ── Colors ───────────────────────────────────────────────────────────────────

fn state_color_for_index(idx: usize) -> Color32 {
    const COLORS: &[Color32] = &[
        Color32::from_rgb(50, 130, 200),   // Blue (Idle)
        Color32::from_rgb(60, 180, 75),    // Green (Walk)
        Color32::from_rgb(220, 160, 40),   // Gold (Run)
        Color32::from_rgb(200, 60, 60),    // Red (Attack)
        Color32::from_rgb(160, 80, 200),   // Purple (Jump)
        Color32::from_rgb(60, 200, 200),   // Cyan (Fall)
        Color32::from_rgb(220, 120, 60),   // Orange (Land)
        Color32::from_rgb(100, 200, 120),  // Mint (Dash)
    ];
    COLORS[idx % COLORS.len()].clone()
}

const NODE_W: f32 = 180.0;
const NODE_H: f32 = 60.0;
const PORT_R: f32 = 6.0;

// ── Render ───────────────────────────────────────────────────────────────────

pub fn render_anim_graph_editor(ui: &mut egui::Ui, state: &mut AnimGraphEditorState) {
    // ── Header ──
    egui::Frame::new()
        .fill(Color32::from_rgb(16, 19, 25))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(37, 44, 55)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Animation Graph Editor").strong().color(Color32::from_rgb(228, 231, 236)));
                ui.separator();
                ui.label(RichText::new("States:").small().color(Color32::from_rgb(142, 152, 168)));
                ui.label(RichText::new(state.nodes.len().to_string()).small().strong().color(Color32::from_rgb(228, 231, 236)));
                ui.label(RichText::new("Transitions:").small().color(Color32::from_rgb(142, 152, 168)));
                ui.label(RichText::new(state.transitions.len().to_string()).small().strong().color(Color32::from_rgb(228, 231, 236)));
                if state.connect_from.is_some() {
                    ui.separator();
                    ui.label(RichText::new("🔗 CONNECTING... Click target state").color(Color32::from_rgb(255, 200, 60)));
                    if ui.button("Cancel").clicked() {
                        state.connect_from = None;
                    }
                }
            });
        });
    ui.add_space(4.0);

    // ── Toolbar ──
    ui.horizontal(|ui| {
        egui::Frame::new()
            .fill(Color32::from_rgb(14, 17, 22))
            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(32, 38, 48)))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::same(4))
            .show(ui, |ui| {
                // Add State
                egui::ComboBox::from_id_salt("anim_add_state")
                    .selected_text("+ Add State")
                    .show_ui(ui, |ui| {
                        for (name, clip_idx) in &[
                            ("Idle", 0), ("Walk", 1), ("Run", 2), ("Attack", 3),
                            ("Jump", 4), ("Fall", 5), ("Land", 6), ("Dash", 7),
                        ] {
                    if ui.selectable_label(false, *name).clicked() {
                                state.add_state(name, *clip_idx, [
                                    50.0 + state.nodes.len() as f32 * 220.0,
                                    100.0 + (state.nodes.len() % 3) as f32 * 100.0,
                                ]);
                            }
                        }
                    });

                // Add Custom State
                if ui.button("+ Custom").clicked() {
                    state.add_state("NewState", 0, [
                        50.0 + state.nodes.len() as f32 * 220.0,
                        100.0 + (state.nodes.len() % 3) as f32 * 100.0,
                    ]);
                }

                ui.separator();

                // Build Graph
                if ui.button("▶ Build Graph").clicked() {
                    if let Some(graph) = state.build_anim_graph() {
                        tracing::info!("[AnimGraph Editor] Built '{}' with {} states, {} transitions",
                            graph.name, graph.states.len(), graph.states.iter().map(|s| s.transitions.len()).sum::<usize>());
                    }
                }

                // Clear
                if ui.button("Clear All").clicked() {
                    state.nodes.clear();
                    state.transitions.clear();
                    state.selected_state = None;
                    state.selected_transition = None;
                    state.initial_state = None;
                    state.connect_from = None;
                }

                ui.separator();

                // Default blend
                ui.label(RichText::new("Blend:").small().color(Color32::from_rgb(142, 152, 168)));
                ui.add(egui::Slider::new(&mut state.default_blend, 0.05..=2.0).suffix("s").show_value(false));
            });
    });
    ui.add_space(4.0);

    // ── Canvas + Properties split ──
    let available = ui.available_size();
    let canvas_w = (available.x * 0.65).max(300.0);
    let (canvas_rect, _) = ui.allocate_exact_size(egui::vec2(canvas_w, available.y), egui::Sense::click_and_drag());
    let props_rect = Rect::from_min_size(
        canvas_rect.min + egui::vec2(canvas_w + 8.0, 0.0),
        egui::vec2((available.x - canvas_w - 8.0).max(200.0), available.y),
    );

    // ── Canvas ──
    let p = ui.painter();
    p.rect_filled(canvas_rect, 8.0, Color32::from_rgb(10, 12, 17));
    p.rect_stroke(canvas_rect, 8.0, egui::Stroke::new(1.0, Color32::from_rgb(38, 44, 54)), egui::StrokeKind::Middle);

    // Grid
    let grid_step = 40.0 * state.zoom;
    if grid_step > 8.0 {
        let origin = canvas_rect.min.to_vec2() + Vec2::new(state.pan[0], state.pan[1]);
        let start_x = (origin.x % grid_step + grid_step) % grid_step;
        let start_y = (origin.y % grid_step + grid_step) % grid_step;
        let mut x = canvas_rect.min.x + start_x;
        while x < canvas_rect.max.x {
            p.line_segment([egui::pos2(x, canvas_rect.min.y), egui::pos2(x, canvas_rect.max.y)],
                egui::Stroke::new(0.5, Color32::from_rgb(20, 24, 32)));
            x += grid_step;
        }
        let mut y = canvas_rect.min.y + start_y;
        while y < canvas_rect.max.y {
            p.line_segment([egui::pos2(canvas_rect.min.x, y), egui::pos2(canvas_rect.max.x, y)],
                egui::Stroke::new(0.5, Color32::from_rgb(20, 24, 32)));
            y += grid_step;
        }
    }

    let canvas_center = canvas_rect.center().to_vec2();

    // Pan via middle mouse
    if canvas_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO))) {
        if ui.input(|i| i.pointer.button_down(egui::PointerButton::Middle)) {
            let delta = ui.input(|i| i.pointer.delta());
            state.pan[0] += delta.x;
            state.pan[1] += delta.y;
        }
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            state.zoom = (state.zoom * (1.0 + scroll * 0.001)).clamp(0.2, 3.0);
        }
    }

    // ── Draw transitions ──
    for (_ti, trans) in state.transitions.iter().enumerate() {
        let from_node = state.nodes.iter().find(|n| n.id == trans.from_state);
        let to_node = state.nodes.iter().find(|n| n.id == trans.to_state);
        if let (Some(from), Some(to)) = (from_node, to_node) {
            let a = egui::pos2(
                canvas_center.x + from.position[0] * state.zoom + state.pan[0] + NODE_W * state.zoom,
                canvas_center.y + from.position[1] * state.zoom + state.pan[1] + NODE_H * state.zoom * 0.5,
            );
            let b = egui::pos2(
                canvas_center.x + to.position[0] * state.zoom + state.pan[0],
                canvas_center.y + to.position[1] * state.zoom + state.pan[1] + NODE_H * state.zoom * 0.5,
            );

            let is_selected = state.selected_transition == Some(trans.id);
            let color = if is_selected {
                Color32::from_rgb(255, 200, 60)
            } else if !trans.conditions.is_empty() {
                Color32::from_rgba_premultiplied(100, 180, 255, 200)
            } else {
                Color32::from_rgba_premultiplied(80, 120, 160, 150)
            };
            let stroke = egui::Stroke::new(if is_selected { 3.0 } else { 2.0 }, color);

            // Bezier curve
            let mid_x = (a.x + b.x) * 0.5;
            let ctrl1 = egui::pos2(mid_x, a.y);
            let ctrl2 = egui::pos2(mid_x, b.y);
            let bezier = egui::epaint::CubicBezierShape::from_points_stroke(
                [a, ctrl1, ctrl2, b], false, Color32::TRANSPARENT, stroke,
            );
            p.add(egui::Shape::CubicBezier(bezier));

            // Arrow head
            let arrow_size = 8.0 * state.zoom;
            let dir = (b - a).normalized();
            let perp = egui::vec2(-dir.y, dir.x);
            let arrow_tip = b - dir * 4.0;
            p.add(egui::Shape::convex_polygon(
                vec![
                    arrow_tip,
                    arrow_tip - dir * arrow_size + perp * arrow_size * 0.4,
                    arrow_tip - dir * arrow_size - perp * arrow_size * 0.4,
                ],
                color,
                egui::Stroke::NONE,
            ));

            // Condition label on the line
            if !trans.conditions.is_empty() {
                let label_pos = egui::pos2(mid_x, (a.y + b.y) * 0.5 - 10.0);
                let label_text = if trans.conditions.len() == 1 {
                    trans.conditions[0].condition_type.label()
                } else {
                    format!("{} conditions", trans.conditions.len())
                };
                p.text(label_pos, egui::Align2::CENTER_CENTER, &label_text,
                    egui::FontId::proportional(9.0), Color32::from_rgb(180, 200, 220));
            }

            // Click to select transition
            let click_rect = Rect::from_center_size(
                egui::pos2(mid_x, (a.y + b.y) * 0.5),
                egui::vec2((b.x - a.x).abs() * 0.5, 30.0),
            );
            if ui.input(|i| i.pointer.any_click())
                && click_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO)))
            {
                state.selected_transition = Some(trans.id);
                state.selected_state = None;
            }
        }
    }

    // ── Draw states ──
    let mut clicked_state: Option<usize> = None;
    let mut drag_state: Option<usize> = None;
    let mut port_click: Option<(usize, bool)> = None; // (state_id, is_output)

    for node in &state.nodes {
        let nx = canvas_center.x + node.position[0] * state.zoom + state.pan[0];
        let ny = canvas_center.y + node.position[1] * state.zoom + state.pan[1];
        let node_rect = Rect::from_min_size(egui::pos2(nx, ny), egui::vec2(NODE_W * state.zoom, NODE_H * state.zoom));

        let is_selected = state.selected_state == Some(node.id);
        let is_initial = node.is_initial;

        // Background
        let bg = if is_selected {
            node.color.gamma_multiply(0.95)
        } else {
            node.color.gamma_multiply(0.65)
        };
        p.rect_filled(node_rect, 8.0, bg);

        // Border
        let border_color = if is_initial {
            Color32::from_rgb(255, 215, 0) // Gold for initial state
        } else if is_selected {
            Color32::WHITE.gamma_multiply(0.9)
        } else {
            Color32::WHITE.gamma_multiply(0.3)
        };
        let border_width = if is_initial || is_selected { 2.5 } else { 1.0 };
        p.rect_stroke(node_rect, 8.0, egui::Stroke::new(border_width, border_color), egui::StrokeKind::Middle);

        // Initial state star
        if is_initial {
            let star_pos = egui::pos2(nx + 12.0, ny + 12.0);
            p.text(star_pos, egui::Align2::CENTER_CENTER, "★",
                egui::FontId::proportional(14.0), Color32::from_rgb(255, 215, 0));
        }

        // State name
        let text_y = if is_initial { ny + 8.0 } else { ny + 2.0 };
        p.text(
            egui::pos2(nx + NODE_W * state.zoom * 0.5, text_y + 8.0),
            egui::Align2::CENTER_CENTER,
            &node.name,
            egui::FontId::proportional(13.0),
            Color32::WHITE,
        );

        // Clip index
        p.text(
            egui::pos2(nx + NODE_W * state.zoom * 0.5, text_y + 28.0),
            egui::Align2::CENTER_CENTER,
            &format!("Clip: {}", node.clip_index),
            egui::FontId::proportional(9.0),
            Color32::from_rgb(180, 190, 200),
        );

        // ── Input port (left edge) ──
        let in_port = egui::pos2(nx, ny + NODE_H * state.zoom * 0.5);
        let in_rect = Rect::from_center_size(in_port, egui::vec2(PORT_R * 2.5, PORT_R * 2.5));
        let in_hover = in_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO)));
        p.circle_filled(in_port, PORT_R, if in_hover {
            Color32::from_rgb(120, 200, 255)
        } else {
            Color32::from_rgb(40, 50, 65)
        });
        p.circle_stroke(in_port, PORT_R, egui::Stroke::new(1.5, Color32::from_rgb(100, 160, 220)));

        if ui.input(|i| i.pointer.any_click()) && in_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO))) {
            port_click = Some((node.id, false));
        }

        // ── Output port (right edge) ──
        let out_port = egui::pos2(nx + NODE_W * state.zoom, ny + NODE_H * state.zoom * 0.5);
        let out_rect = Rect::from_center_size(out_port, egui::vec2(PORT_R * 2.5, PORT_R * 2.5));
        let out_hover = out_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO)));
        p.circle_filled(out_port, PORT_R, if out_hover {
            Color32::from_rgb(255, 180, 80)
        } else {
            Color32::from_rgb(40, 50, 65)
        });
        p.circle_stroke(out_port, PORT_R, egui::Stroke::new(1.5, Color32::from_rgb(220, 160, 60)));

        if ui.input(|i| i.pointer.any_click()) && out_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO))) {
            port_click = Some((node.id, true));
        }

        // ── Node body interaction ──
        let body_rect = Rect::from_min_size(
            egui::pos2(nx + PORT_R * 2.0, ny),
            egui::vec2(NODE_W * state.zoom - PORT_R * 4.0, NODE_H * state.zoom),
        );
        if ui.input(|i| i.pointer.any_click()) && body_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO))) {
            clicked_state = Some(node.id);
        }
        if ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary))
            && body_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO)))
        {
            drag_state = Some(node.id);
        }
    }

    // ── Handle drag ──
    if let Some(id) = drag_state {
        if ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary)) {
            let delta = ui.input(|i| i.pointer.delta());
            if let Some(node) = state.nodes.iter_mut().find(|n| n.id == id) {
                node.position[0] += delta.x / state.zoom;
                node.position[1] += delta.y / state.zoom;
            }
        }
    }

    // ── Handle click ──
    if let Some(id) = clicked_state {
        state.selected_state = Some(id);
        state.selected_transition = None;
    }

    // ── Handle port clicks (connection creation) ──
    if let Some((id, is_output)) = port_click {
        if is_output {
            // Start connecting from output port
            state.connect_from = Some(id);
        } else if let Some(from_id) = state.connect_from.take() {
            // Complete connection to input port
            state.add_transition(from_id, id);
        }
    }

    // ── Right-click context menu ──
    let hover_pos = ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO));
    if canvas_rect.contains(hover_pos) && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary)) {
        egui::Window::new("context_menu_anim")
            .title_bar(false)
            .resizable(false)
            .fixed_pos(hover_pos)
            .show(ui.ctx(), |ui| {
                ui.set_min_width(150.0);
                ui.label(RichText::new("Add State Here").strong());
                for (name, clip_idx) in &[("Idle", 0), ("Walk", 1), ("Run", 2), ("Attack", 3)] {
                    if ui.button(*name).clicked() {
                        let world_pos = [
                            (hover_pos.x - canvas_center.x - state.pan[0]) / state.zoom,
                            (hover_pos.y - canvas_center.y - state.pan[1]) / state.zoom,
                        ];
                        state.add_state(name, *clip_idx, world_pos);
                    }
                }
            });
    }

    // ── Delete key ──
    if ui.input(|i| i.key_pressed(egui::Key::Delete)) {
        if let Some(id) = state.selected_state.take() {
            state.remove_state(id);
        } else if let Some(id) = state.selected_transition.take() {
            state.remove_transition(id);
        }
    }

    // ── Properties Panel ──
    let p_ui = &mut *ui;
    p_ui.allocate_ui_at_rect(props_rect, |ui| {
        egui::Frame::new()
            .fill(Color32::from_rgb(14, 17, 22))
            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(32, 38, 48)))
            .corner_radius(8.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                if let Some(sel_id) = state.selected_state {
                    render_state_properties(ui, state, sel_id);
                } else if let Some(sel_id) = state.selected_transition {
                    render_transition_properties(ui, state, sel_id);
                } else {
                    ui.label(RichText::new("Properties").strong().small().color(Color32::from_rgb(228, 231, 236)));
                    ui.separator();
                    ui.label(RichText::new("Select a state or transition to edit.")
                        .small().color(Color32::from_rgb(142, 152, 168)));
                    ui.add_space(12.0);
                    ui.label(RichText::new("Quick Start").strong().small().color(Color32::from_rgb(180, 200, 220)));
                    ui.label(RichText::new("1. Add states from toolbar")
                        .small().color(Color32::from_rgb(142, 152, 168)));
                    ui.label(RichText::new("2. Drag from orange port → blue port to connect")
                        .small().color(Color32::from_rgb(142, 152, 168)));
                    ui.label(RichText::new("3. Select a transition to add conditions")
                        .small().color(Color32::from_rgb(142, 152, 168)));
                    ui.label(RichText::new("4. Right-click to add states at cursor")
                        .small().color(Color32::from_rgb(142, 152, 168)));
                    ui.label(RichText::new("5. Middle-mouse to pan, scroll to zoom")
                        .small().color(Color32::from_rgb(142, 152, 168)));
                }
            });
    });
}

// ── State Properties ─────────────────────────────────────────────────────────

fn render_state_properties(ui: &mut egui::Ui, state: &mut AnimGraphEditorState, sel_id: usize) {
    ui.label(RichText::new("State Properties").strong().small().color(Color32::from_rgb(228, 231, 236)));
    ui.separator();

    // Collect editable info first, then mutate to avoid double-borrow.
    let mut toggle_initial = false;
    if let Some(node) = state.nodes.iter_mut().find(|n| n.id == sel_id) {
        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.text_edit_singleline(&mut node.name);
        });

        ui.horizontal(|ui| {
            ui.label("Clip Index:");
            ui.add(egui::DragValue::new(&mut node.clip_index).speed(1).clamp_range(0..=20));
        });

        // Initial state toggle
        let was_initial = node.is_initial;
        ui.horizontal(|ui| {
            ui.label("Initial State:");
            ui.checkbox(&mut node.is_initial, "");
        });
        if node.is_initial && !was_initial {
            toggle_initial = true;
        } else if !node.is_initial && was_initial {
            node.is_initial = false;
            state.initial_state = None;
        }

        // Position display
        ui.label(RichText::new(format!("Pos: ({:.0}, {:.0})", node.position[0], node.position[1]))
            .small().color(Color32::from_rgb(100, 120, 140)));
    }
    if toggle_initial {
        state.set_initial(sel_id);
    }

    ui.separator();

    // Connected transitions
    ui.label(RichText::new("Transitions").strong().small().color(Color32::from_rgb(180, 200, 220)));
    let out_trans: Vec<_> = state.transitions.iter()
        .filter(|t| t.from_state == sel_id)
        .map(|t| (t.id, t.to_state, t.conditions.len()))
        .collect();
    let in_trans: Vec<_> = state.transitions.iter()
        .filter(|t| t.to_state == sel_id)
        .map(|t| (t.id, t.from_state, t.conditions.len()))
        .collect();

    for (tid, from_id, cond_count) in &in_trans {
        let name = state.nodes.iter().find(|n| n.id == *from_id).map(|n| n.name.as_str()).unwrap_or("?");
        let selected = state.selected_transition == Some(*tid);
        if ui.selectable_label(selected, RichText::new(format!("← {} ({} conditions)", name, cond_count)).small()).clicked() {
            state.selected_transition = Some(*tid);
            state.selected_state = None;
        }
    }
    for (tid, to_id, cond_count) in &out_trans {
        let name = state.nodes.iter().find(|n| n.id == *to_id).map(|n| n.name.as_str()).unwrap_or("?");
        let selected = state.selected_transition == Some(*tid);
        if ui.selectable_label(selected, RichText::new(format!("→ {} ({} conditions)", name, cond_count)).small()).clicked() {
            state.selected_transition = Some(*tid);
            state.selected_state = None;
        }
    }

    ui.separator();
    if ui.button(RichText::new("Delete State").color(Color32::from_rgb(220, 80, 80))).clicked() {
        state.remove_state(sel_id);
    }
}

// ── Transition Properties ────────────────────────────────────────────────────

fn render_transition_properties(ui: &mut egui::Ui, state: &mut AnimGraphEditorState, sel_id: usize) {
    ui.label(RichText::new("Transition Properties").strong().small().color(Color32::from_rgb(228, 231, 236)));
    ui.separator();

    let trans_idx = state.transitions.iter().position(|t| t.id == sel_id);
    if let Some(ti) = trans_idx {
        let from_name = state.transitions[ti].from_state;
        let to_name = state.transitions[ti].to_state;
        let from_label = state.nodes.iter().find(|n| n.id == from_name).map(|n| n.name.as_str()).unwrap_or("?");
        let to_label = state.nodes.iter().find(|n| n.id == to_name).map(|n| n.name.as_str()).unwrap_or("?");

        ui.label(RichText::new(format!("{} → {}", from_label, to_label))
            .color(Color32::from_rgb(180, 200, 220)));

        ui.add_space(4.0);

        // Blend duration
        ui.horizontal(|ui| {
            ui.label("Blend:");
            ui.add(egui::Slider::new(&mut state.transitions[ti].blend_duration, 0.0..=2.0).suffix("s"));
        });

        // Priority
        ui.horizontal(|ui| {
            ui.label("Priority:");
            ui.add(egui::DragValue::new(&mut state.transitions[ti].priority).speed(1).clamp_range(-100..=100));
        });

        ui.separator();

        // Conditions
        ui.label(RichText::new("Conditions").strong().small().color(Color32::from_rgb(180, 200, 220)));

        let mut remove_cond: Option<usize> = None;
        for (ci, cond) in state.transitions[ti].conditions.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.checkbox(&mut cond.enabled, "");
                ui.label(RichText::new(cond.condition_type.label()).small().color(
                    if cond.enabled { Color32::from_rgb(200, 220, 240) } else { Color32::from_rgb(100, 110, 120) }
                ));
                if ui.small_button("✕").clicked() {
                    remove_cond = Some(ci);
                }
            });

            // Inline editing based on condition type
            match &mut cond.condition_type {
                EditorConditionType::FloatGreaterThan { param, threshold } |
                EditorConditionType::FloatLessThan { param, threshold } => {
                    ui.horizontal(|ui| {
                        ui.label("  Param:");
                        ui.text_edit_singleline(param);
                        ui.label("Threshold:");
                        ui.add(egui::DragValue::new(threshold).speed(0.1).fixed_decimals(2));
                    });
                }
                EditorConditionType::FloatInRange { param, min, max } => {
                    ui.horizontal(|ui| {
                        ui.label("  Param:");
                        ui.text_edit_singleline(param);
                        ui.add(egui::DragValue::new(min).speed(0.1).fixed_decimals(2));
                        ui.label("..");
                        ui.add(egui::DragValue::new(max).speed(0.1).fixed_decimals(2));
                    });
                }
                EditorConditionType::BoolEquals { param, expected } => {
                    ui.horizontal(|ui| {
                        ui.label("  Param:");
                        ui.text_edit_singleline(param);
                        ui.checkbox(expected, "Expected");
                    });
                }
                EditorConditionType::EnumEquals { param, expected } => {
                    ui.horizontal(|ui| {
                        ui.label("  Param:");
                        ui.text_edit_singleline(param);
                        ui.add(egui::DragValue::new(expected).speed(1));
                    });
                }
                EditorConditionType::StringEquals { param, expected } => {
                    ui.horizontal(|ui| {
                        ui.label("  Param:");
                        ui.text_edit_singleline(param);
                        ui.text_edit_singleline(expected);
                    });
                }
                EditorConditionType::TimeInStateGreaterThan { seconds } => {
                    ui.horizontal(|ui| {
                        ui.label("  Seconds:");
                        ui.add(egui::DragValue::new(seconds).speed(0.1).fixed_decimals(1).clamp_range(0.0..=60.0));
                    });
                }
            }
        }

        if let Some(ci) = remove_cond {
            state.transitions[ti].conditions.remove(ci);
        }

        // Add condition
        let types = EditorConditionType::all_types();
        egui::ComboBox::from_id_salt(format!("add_cond_{}", sel_id))
            .selected_text("+ Add Condition")
            .show_ui(ui, |ui| {
                for (name, factory) in &types {
                    if ui.selectable_label(false, name.clone()).clicked() {
                        state.transitions[ti].conditions.push(EditorCondition {
                            condition_type: factory(),
                            enabled: true,
                        });
                    }
                }
            });

        ui.separator();
        if ui.button(RichText::new("Delete Transition").color(Color32::from_rgb(220, 80, 80))).clicked() {
            state.remove_transition(sel_id);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_add_state() {
        let mut state = AnimGraphEditorState::new();
        let id = state.add_state("Idle", 0, [0.0, 0.0]);
        assert_eq!(state.nodes.len(), 1);
        assert_eq!(state.nodes[0].name, "Idle");
        assert!(state.nodes[0].is_initial); // First state becomes initial
    }

    #[test]
    fn editor_add_transition() {
        let mut state = AnimGraphEditorState::new();
        let idle = state.add_state("Idle", 0, [0.0, 0.0]);
        let walk = state.add_state("Walk", 1, [200.0, 0.0]);
        let tid = state.add_transition(idle, walk);
        assert!(tid.is_some());
        assert_eq!(state.transitions.len(), 1);
    }

    #[test]
    fn editor_no_self_transition() {
        let mut state = AnimGraphEditorState::new();
        let idle = state.add_state("Idle", 0, [0.0, 0.0]);
        assert!(state.add_transition(idle, idle).is_none());
    }

    #[test]
    fn editor_no_duplicate_transition() {
        let mut state = AnimGraphEditorState::new();
        let idle = state.add_state("Idle", 0, [0.0, 0.0]);
        let walk = state.add_state("Walk", 1, [200.0, 0.0]);
        state.add_transition(idle, walk);
        assert!(state.add_transition(idle, walk).is_none());
    }

    #[test]
    fn editor_set_initial() {
        let mut state = AnimGraphEditorState::new();
        let idle = state.add_state("Idle", 0, [0.0, 0.0]);
        let walk = state.add_state("Walk", 1, [200.0, 0.0]);
        assert!(state.nodes.iter().find(|n| n.id == idle).unwrap().is_initial);
        state.set_initial(walk);
        assert!(state.nodes.iter().find(|n| n.id == walk).unwrap().is_initial);
        assert!(!state.nodes.iter().find(|n| n.id == idle).unwrap().is_initial);
    }

    #[test]
    fn editor_remove_state_cleans_transitions() {
        let mut state = AnimGraphEditorState::new();
        let idle = state.add_state("Idle", 0, [0.0, 0.0]);
        let walk = state.add_state("Walk", 1, [200.0, 0.0]);
        state.add_transition(idle, walk);
        state.add_transition(walk, idle);
        assert_eq!(state.transitions.len(), 2);
        state.remove_state(walk);
        assert_eq!(state.transitions.len(), 0);
    }

    #[test]
    fn editor_build_graph() {
        let mut state = AnimGraphEditorState::new();
        let idle = state.add_state("Idle", 0, [0.0, 0.0]);
        let walk = state.add_state("Walk", 1, [200.0, 0.0]);
        state.add_transition(idle, walk);

        let graph = state.build_anim_graph();
        assert!(graph.is_some());
        let g = graph.unwrap();
        assert_eq!(g.states.len(), 2);
        assert_eq!(g.states[0].name, "Idle");
        assert_eq!(g.states[1].name, "Walk");
        assert_eq!(g.states[0].transitions.len(), 1);
    }

    #[test]
    fn editor_build_graph_with_conditions() {
        let mut state = AnimGraphEditorState::new();
        let idle = state.add_state("Idle", 0, [0.0, 0.0]);
        let walk = state.add_state("Walk", 1, [200.0, 0.0]);
        let tid = state.add_transition(idle, walk).unwrap();

        // Add a condition
        let trans = state.transitions.iter_mut().find(|t| t.id == tid).unwrap();
        trans.conditions.push(EditorCondition {
            condition_type: EditorConditionType::FloatGreaterThan { param: "speed".into(), threshold: 0.5 },
            enabled: true,
        });

        let graph = state.build_anim_graph().unwrap();
        assert_eq!(graph.states[0].transitions[0].conditions.len(), 1);
    }

    #[test]
    fn editor_condition_label() {
        let cond = EditorConditionType::FloatGreaterThan { param: "speed".into(), threshold: 2.5 };
        assert_eq!(cond.label(), "speed > 2.50");
    }

    #[test]
    fn editor_condition_all_types() {
        let types = EditorConditionType::all_types();
        assert!(types.len() >= 7);
    }
}
