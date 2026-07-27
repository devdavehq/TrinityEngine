use crate::ai::behavior_tree::{
    BehaviorTree, BehaviorNode, Cooldown, Inverter, Log, MoveTo, Parallel, Patrol,
    Selector, Sequence, SetState, Wait,
};
use egui::{Color32, Pos2, Rect, RichText, Vec2};

#[derive(Clone)]
pub enum BtEditorNodeType {
    Sequence,
    Selector,
    Parallel { threshold: u32 },
    Inverter,
    Repeater { max_reps: i32 },
    Cooldown { duration: f32 },
    MoveTo { speed: f32, target_key: String },
    Patrol { speed: f32, waypoints_key: String },
    Wait { duration: f32 },
    SetState { state_name: String },
    Log { message: String },
    CustomAction { name: String },
}

impl BtEditorNodeType {
    fn label(&self) -> &str {
        match self {
            Self::Sequence => "Sequence",
            Self::Selector => "Selector",
            Self::Parallel { .. } => "Parallel",
            Self::Inverter => "Inverter",
            Self::Repeater { .. } => "Repeater",
            Self::Cooldown { .. } => "Cooldown",
            Self::MoveTo { .. } => "MoveTo",
            Self::Patrol { .. } => "Patrol",
            Self::Wait { .. } => "Wait",
            Self::SetState { .. } => "SetState",
            Self::Log { .. } => "Log",
            Self::CustomAction { .. } => "CustomAction",
        }
    }

    fn color(&self) -> Color32 {
        match self {
            Self::Sequence | Self::Selector | Self::Parallel { .. } => Color32::from_rgb(50, 120, 200),
            Self::Inverter | Self::Repeater { .. } | Self::Cooldown { .. } => Color32::from_rgb(200, 120, 50),
            _ => Color32::from_rgb(60, 180, 75),
        }
    }

    fn all() -> Vec<(String, BtEditorNodeType)> {
        vec![
            ("Sequence".into(), Self::Sequence),
            ("Selector".into(), Self::Selector),
            ("Parallel".into(), Self::Parallel { threshold: 1 }),
            ("Inverter".into(), Self::Inverter),
            ("Repeater".into(), Self::Repeater { max_reps: 3 }),
            ("Cooldown".into(), Self::Cooldown { duration: 2.0 }),
            ("MoveTo".into(), Self::MoveTo { speed: 5.0, target_key: "target_pos".into() }),
            ("Patrol".into(), Self::Patrol { speed: 3.0, waypoints_key: "patrol_points".into() }),
            ("Wait".into(), Self::Wait { duration: 1.0 }),
            ("SetState".into(), Self::SetState { state_name: "idle".into() }),
            ("Log".into(), Self::Log { message: "hello".into() }),
            ("CustomAction".into(), Self::CustomAction { name: "my_action".into() }),
        ]
    }
}

#[derive(Clone)]
pub struct BtEditorNode {
    pub id: usize,
    pub node_type: BtEditorNodeType,
    pub position: [f32; 2],
    pub label: String,
}

#[derive(Clone)]
pub struct BtEditorConnection {
    pub from_node: usize,
    pub to_node: usize,
}

pub struct BtEditorState {
    pub nodes: Vec<BtEditorNode>,
    pub connections: Vec<BtEditorConnection>,
    pub selected_node: Option<usize>,
    pub pan: [f32; 2],
    pub zoom: f32,
    pub visible: bool,
    next_id: usize,
}

impl BtEditorState {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            selected_node: None,
            pan: [0.0, 0.0],
            zoom: 1.0,
            visible: true,
            next_id: 1,
        }
    }

    pub fn add_node(&mut self, node_type: BtEditorNodeType, position: [f32; 2]) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let label = node_type.label().to_string();
        self.nodes.push(BtEditorNode { id, node_type, position, label });
        id
    }

    pub fn connect(&mut self, from: usize, to: usize) {
        if from != to && !self.connections.iter().any(|c| c.from_node == from && c.to_node == to) {
            self.connections.push(BtEditorConnection { from_node: from, to_node: to });
        }
    }

    pub fn remove_node(&mut self, id: usize) {
        self.nodes.retain(|n| n.id != id);
        self.connections.retain(|c| c.from_node != id && c.to_node != id);
        if self.selected_node == Some(id) {
            self.selected_node = None;
        }
    }

    fn find_children(&self, parent_id: usize) -> Vec<usize> {
        self.connections.iter()
            .filter(|c| c.from_node == parent_id)
            .map(|c| c.to_node)
            .collect()
    }

    fn build_node(&self, node_id: usize, built: &mut std::collections::HashSet<usize>) -> Option<Box<dyn BehaviorNode>> {
        if !built.insert(node_id) {
            return None;
        }
        let node = self.nodes.iter().find(|n| n.id == node_id)?;
        let children = self.find_children(node_id);
        let child_nodes: Vec<Box<dyn BehaviorNode>> = children.iter()
            .filter_map(|&cid| self.build_node(cid, built))
            .collect();

        Some(match &node.node_type {
            BtEditorNodeType::Sequence => Box::new(Sequence::new(&node.label, child_nodes)),
            BtEditorNodeType::Selector => Box::new(Selector::new(&node.label, child_nodes)),
            BtEditorNodeType::Parallel { threshold } => {
                let ft = child_nodes.len();
                Box::new(Parallel::new(&node.label, child_nodes, *threshold as usize, ft))
            }
            BtEditorNodeType::Inverter => {
                let child = child_nodes.into_iter().next().unwrap_or_else(|| Box::new(Wait::new("placeholder", 0.0)));
                Box::new(Inverter::new(&node.label, child))
            }
            BtEditorNodeType::Repeater { max_reps } => {
                let child = child_nodes.into_iter().next().unwrap_or_else(|| Box::new(Wait::new("placeholder", 0.0)));
                Box::new(crate::ai::behavior_tree::Repeater::new(&node.label, child, *max_reps.max(&0) as u32))
            }
            BtEditorNodeType::Cooldown { duration } => {
                let child = child_nodes.into_iter().next().unwrap_or_else(|| Box::new(Wait::new("placeholder", 0.0)));
                Box::new(Cooldown::new(&node.label, child, *duration))
            }
            BtEditorNodeType::MoveTo { speed, target_key } => {
                Box::new(MoveTo::new(&node.label, *speed, target_key))
            }
            BtEditorNodeType::Patrol { speed, waypoints_key } => {
                Box::new(Patrol::new(&node.label, *speed, waypoints_key))
            }
            BtEditorNodeType::Wait { duration } => {
                Box::new(Wait::new(&node.label, *duration))
            }
            BtEditorNodeType::SetState { state_name } => {
                Box::new(SetState::new(&node.label, state_name))
            }
            BtEditorNodeType::Log { message } => {
                Box::new(Log::new(&node.label, message))
            }
            BtEditorNodeType::CustomAction { name } => {
                let n = name.clone();
                Box::new(crate::ai::behavior_tree::CustomAction::new(&node.label, move |_ctx| {
                    tracing::info!("[BT:CustomAction] {}", n);
                    crate::ai::behavior_tree::Status::Success
                }))
            }
        })
    }

    pub fn build_tree(&self, name: &str) -> Option<BehaviorTree> {
        let root_id = self.nodes.iter().find(|n| {
            !self.connections.iter().any(|c| c.to_node == n.id)
        })?.id;
        let mut built = std::collections::HashSet::new();
        let root = self.build_node(root_id, &mut built)?;
        Some(BehaviorTree::new(name, root))
    }
}

const NODE_W: f32 = 160.0;
const NODE_H: f32 = 48.0;

pub fn render_bt_editor(ui: &mut egui::Ui, state: &mut BtEditorState) {
    egui::Frame::new()
        .fill(Color32::from_rgb(16, 19, 25))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(37, 44, 55)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.label(RichText::new("Behavior Tree Editor").strong().color(Color32::from_rgb(228, 231, 236)));
        });
    ui.add_space(6.0);

    // ── Toolbar ──
    ui.horizontal(|ui| {
        egui::Frame::new()
            .fill(Color32::from_rgb(14, 17, 22))
            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(32, 38, 48)))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::same(4))
            .show(ui, |ui| {
                let node_types = BtEditorNodeType::all();
                egui::ComboBox::from_id_salt("bt_add_node_combo")
                    .selected_text("+ Add Node")
                    .show_ui(ui, |ui| {
                        for (name, ntype) in &node_types {
                            let color = ntype.color();
                            if ui.selectable_label(false, RichText::new(name).color(color)).clicked() {
                                state.add_node(ntype.clone(), [0.0, 0.0]);
                            }
                        }
                    });
                ui.separator();
                if ui.button("Build Tree").clicked() {
                    if let Some(_tree) = state.build_tree("editor_built") {
                        tracing::info!("[BT Editor] Tree built successfully");
                    }
                }
                if ui.button("Clear All").clicked() {
                    state.nodes.clear();
                    state.connections.clear();
                    state.selected_node = None;
                }
            });
    });
    ui.add_space(4.0);

    // ── Canvas + Properties split ──
    let available = ui.available_size();
    let canvas_w = available.x * 0.7;
    let (canvas_rect, _) = ui.allocate_exact_size(egui::vec2(canvas_w.max(200.0), available.y), egui::Sense::click_and_drag());
    let props_rect = Rect::from_min_size(
        canvas_rect.min + egui::vec2(canvas_w.max(200.0) + 8.0, 0.0),
        egui::vec2(available.x - canvas_w.max(200.0) - 8.0, available.y),
    );

    // ── Canvas ──
    let p = ui.painter();
    p.rect_filled(canvas_rect, 8.0, Color32::from_rgb(10, 12, 17));
    p.rect_stroke(canvas_rect, 8.0, egui::Stroke::new(1.0, Color32::from_rgb(38, 44, 54)), egui::StrokeKind::Middle);

    let canvas_center = canvas_rect.center().to_vec2();

    // Pan via middle mouse drag
    if canvas_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO))) {
        if ui.input(|i| i.pointer.button_down(egui::PointerButton::Middle)) {
            let delta = ui.input(|i| i.pointer.delta());
            state.pan[0] += delta.x;
            state.pan[1] += delta.y;
        }
        // Zoom via scroll
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            state.zoom = (state.zoom * (1.0 + scroll * 0.001)).clamp(0.1, 5.0);
        }
    }

    // Draw connections
    for conn in &state.connections {
        if let (Some(from), Some(to)) = (
            state.nodes.iter().find(|n| n.id == conn.from_node),
            state.nodes.iter().find(|n| n.id == conn.to_node),
        ) {
            let a = egui::pos2(
                canvas_center.x + from.position[0] * state.zoom + state.pan[0] + NODE_W * state.zoom * 0.5,
                canvas_center.y + from.position[1] * state.zoom + state.pan[1] + NODE_H * state.zoom,
            );
            let b = egui::pos2(
                canvas_center.x + to.position[0] * state.zoom + state.pan[0],
                canvas_center.y + to.position[1] * state.zoom + state.pan[1],
            );
            let ctrl = egui::pos2(a.x, (a.y + b.y) * 0.5);
            p.add(egui::Shape::line(
                vec![a, ctrl, ctrl, b],
                egui::Stroke::new(2.0, Color32::from_rgba_premultiplied(100, 140, 200, 180)),
            ));
        }
    }

    // Draw nodes
    let mut clicked_node: Option<usize> = None;
    let mut drag_node: Option<usize> = None;
    for node in &state.nodes {
        let nx = canvas_center.x + node.position[0] * state.zoom + state.pan[0];
        let ny = canvas_center.y + node.position[1] * state.zoom + state.pan[1];
        let node_rect = Rect::from_min_size(egui::pos2(nx, ny), egui::vec2(NODE_W * state.zoom, NODE_H * state.zoom));

        let is_selected = state.selected_node == Some(node.id);
        let bg = if is_selected { node.node_type.color().gamma_multiply(0.9) } else { node.node_type.color().gamma_multiply(0.7) };
        p.rect_filled(node_rect, 6.0, bg);
        p.rect_stroke(node_rect, 6.0, egui::Stroke::new(if is_selected { 2.0 } else { 1.0 }, Color32::WHITE.gamma_multiply(if is_selected { 0.8 } else { 0.3 })), egui::StrokeKind::Middle);
        p.text(node_rect.center(), egui::Align2::CENTER_CENTER, &node.label, egui::FontId::proportional(11.0), Color32::WHITE);

        if ui.input(|i| i.pointer.any_click()) && node_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO))) {
            clicked_node = Some(node.id);
        }
        if ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary)) && node_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO))) {
            drag_node = Some(node.id);
        }
    }

    // Handle drag
    if let Some(id) = drag_node {
        if ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary)) {
            let delta = ui.input(|i| i.pointer.delta());
            if let Some(node) = state.nodes.iter_mut().find(|n| n.id == id) {
                node.position[0] += delta.x / state.zoom;
                node.position[1] += delta.y / state.zoom;
            }
        }
    }
    if let Some(id) = clicked_node {
        state.selected_node = Some(id);
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
                ui.label(RichText::new("Properties").strong().small().color(Color32::from_rgb(228, 231, 236)));
                ui.separator();

                let node_idx = state.selected_node.and_then(|idx| {
                    state.nodes.iter().position(|n| n.id == idx)
                });

                if let Some(ni) = node_idx {
                    let (inputs, outputs) = {
                        let node_id = state.nodes[ni].id;
                        let ins: Vec<usize> = state.connections.iter().filter(|c| c.to_node == node_id).map(|c| c.from_node).collect();
                        let outs: Vec<usize> = state.connections.iter().filter(|c| c.from_node == node_id).map(|c| c.to_node).collect();
                        (ins, outs)
                    };

                    let node = &mut state.nodes[ni];
                    ui.horizontal(|ui| {
                        ui.label("Label:");
                        ui.text_edit_singleline(&mut node.label);
                    });

                    let nt = &mut node.node_type;
                    if let BtEditorNodeType::Parallel { threshold } = nt {
                        ui.add(egui::Slider::new(threshold, 1..=32).text("Success threshold"));
                    } else if let BtEditorNodeType::Repeater { max_reps } = nt {
                        ui.add(egui::Slider::new(max_reps, 0..=100).text("Max reps (0=inf)"));
                    } else if let BtEditorNodeType::Cooldown { duration } = nt {
                        ui.add(egui::Slider::new(duration, 0.0..=60.0).text("Duration (s)"));
                    } else if let BtEditorNodeType::MoveTo { speed, target_key } = nt {
                        ui.add(egui::Slider::new(speed, 0.0..=50.0).text("Speed"));
                        ui.horizontal(|ui| { ui.label("Target key:"); ui.text_edit_singleline(target_key); });
                    } else if let BtEditorNodeType::Patrol { speed, waypoints_key } = nt {
                        ui.add(egui::Slider::new(speed, 0.0..=50.0).text("Speed"));
                        ui.horizontal(|ui| { ui.label("Waypoints key:"); ui.text_edit_singleline(waypoints_key); });
                    } else if let BtEditorNodeType::Wait { duration } = nt {
                        ui.add(egui::Slider::new(duration, 0.0..=60.0).text("Duration (s)"));
                    } else if let BtEditorNodeType::SetState { state_name } = nt {
                        ui.horizontal(|ui| { ui.label("State name:"); ui.text_edit_singleline(state_name); });
                    } else if let BtEditorNodeType::Log { message } = nt {
                        ui.horizontal(|ui| { ui.label("Message:"); ui.text_edit_singleline(message); });
                    } else if let BtEditorNodeType::CustomAction { name } = nt {
                        ui.horizontal(|ui| { ui.label("Action name:"); ui.text_edit_singleline(name); });
                    }

                    let remove_id = state.nodes[ni].id;
                    ui.separator();
                    if ui.button("Remove Node").clicked() {
                        state.remove_node(remove_id);
                    }

                    ui.separator();
                    ui.label("Connections:");
                    for from_id in &inputs {
                        if let Some(from) = state.nodes.iter().find(|n| n.id == *from_id) {
                            ui.label(format!("  ← {}", from.label));
                        }
                    }
                    for to_id in &outputs {
                        if let Some(to) = state.nodes.iter().find(|n| n.id == *to_id) {
                            ui.label(format!("  → {}", to.label));
                        }
                    }
                } else {
                    ui.colored_label(Color32::from_rgb(180, 150, 120), "Select a node to edit properties.");
                }
            });
    });
}
