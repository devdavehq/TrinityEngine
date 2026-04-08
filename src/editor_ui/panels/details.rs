use crate::components;
use crate::editor_ui::UiFrameArgs;
use crate::terrain::{remove_nearby_foliage, spawn_foliage_ring};
use egui::{Color32, RichText};

fn body_type_label(t: components::BodyType) -> &'static str {
    match t {
        components::BodyType::Static => "Static",
        components::BodyType::Dynamic => "Dynamic",
        components::BodyType::Kinematic => "Kinematic",
    }
}

fn entity_bits_label(entity: hecs::Entity) -> u64 {
    entity.to_bits().get()
}

fn pick_entity_bits(ui: &mut egui::Ui, label: &str, bits: &mut u64) {
    ui.add(
        egui::DragValue::new(bits)
            .prefix(format!("{label} "))
            .speed(1.0)
            .range(1..=u64::MAX),
    );
}

fn blueprint_divider(ui: &mut egui::Ui, stem: &str, title: &str) {
    let w = ui.available_width().max(120.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 24.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 6.0, Color32::from_rgb(17, 21, 28));
    ui.painter().rect_stroke(
        rect,
        6.0,
        egui::Stroke::new(1.0, Color32::from_rgb(35, 43, 54)),
        egui::StrokeKind::Middle,
    );
    ui.painter().text(
        rect.left_center() + egui::vec2(8.0, 0.0),
        egui::Align2::LEFT_CENTER,
        format!("{}  {}", stem.to_ascii_uppercase(), title),
        egui::FontId::proportional(12.5),
        Color32::from_rgb(214, 220, 230),
    );
    ui.add_space(3.0);
}

fn section_open(ui: &mut egui::Ui, key: &str, default_open: bool) -> bool {
    ui.data_mut(|d| d.get_temp::<bool>(format!("details_open_{key}").into()))
        .unwrap_or(default_open)
}

fn set_section_open(ui: &mut egui::Ui, key: &str, open: bool) {
    ui.data_mut(|d| d.insert_temp(format!("details_open_{key}").into(), open));
}

fn section_pinned(ui: &mut egui::Ui, key: &str) -> bool {
    ui.data_mut(|d| d.get_temp::<bool>(format!("details_pin_{key}").into()))
        .unwrap_or(false)
}

fn set_section_pinned(ui: &mut egui::Ui, key: &str, pinned: bool) {
    ui.data_mut(|d| d.insert_temp(format!("details_pin_{key}").into(), pinned));
}

fn section_shell<R>(
    ui: &mut egui::Ui,
    stem: &str,
    title: &str,
    key: &str,
    default_open: bool,
    body: impl FnOnce(&mut egui::Ui) -> R,
) {
    blueprint_divider(ui, stem, title);
    let mut open = section_open(ui, key, default_open);
    let mut pinned = section_pinned(ui, key);
    ui.horizontal(|ui| {
        if ui.small_button(if open { "▼" } else { "▶" }).clicked() {
            open = !open;
        }
        if ui.small_button(if pinned { "★" } else { "☆" }).clicked() {
            pinned = !pinned;
        }
        ui.label(RichText::new(title).small().strong());
    });
    set_section_open(ui, key, open);
    set_section_pinned(ui, key, pinned);
    if open || pinned {
        egui::Frame::new()
            .fill(Color32::from_rgb(14, 17, 22))
            .stroke(egui::Stroke::new(1.0, Color32::from_rgb(32, 38, 48)))
            .corner_radius(8.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                body(ui);
            });
        ui.add_space(4.0);
    }
}

/// Full inspector for docked Details tab (matches legacy inspector capabilities).
pub fn render_details_panel(ui: &mut egui::Ui, args: &mut UiFrameArgs<'_>) {
    egui::Frame::new()
        .fill(Color32::from_rgb(16, 19, 25))
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(37, 44, 55)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.label(
                RichText::new("Details")
                    .strong()
                    .color(Color32::from_rgb(228, 231, 236)),
            );
            ui.label(
                RichText::new("Inspector for transforms, rendering, physics, materials, foliage, and terrain")
                    .small()
                    .color(Color32::from_rgb(148, 158, 173)),
            );
        });
    ui.add_space(8.0);
    ui.horizontal_wrapped(|ui| {
        if ui.small_button("Expand All").clicked() {
            for k in ["render", "physics", "box_col", "obb_col", "hinge", "fixed", "spring", "rope", "material", "foliage", "terrain"] {
                set_section_open(ui, k, true);
            }
        }
        if ui.small_button("Collapse All").clicked() {
            for k in ["render", "physics", "box_col", "obb_col", "hinge", "fixed", "spring", "rope", "material", "foliage", "terrain"] {
                set_section_open(ui, k, false);
            }
        }
        ui.label(RichText::new("Pinned sections stay visible").small().weak());
    });
    ui.separator();

    if let Some(entity) = args.selected_renderable.as_ref().copied() {
        if let Ok(mut pos) = args.world.get::<&mut components::Position>(entity) {
            egui::Frame::new()
                .fill(Color32::from_rgb(14, 17, 22))
                .stroke(egui::Stroke::new(1.0, Color32::from_rgb(32, 38, 48)))
                .corner_radius(8.0)
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Selection").small().strong());
                        ui.separator();
                        ui.monospace(format!("{:?}", entity));
                    });
                    ui.separator();
                    ui.label(RichText::new("Transform").small().strong());
                    ui.columns(3, |columns| {
                        columns[0].label(RichText::new("X").small().color(Color32::from_rgb(198, 92, 92)));
                        columns[1].label(RichText::new("Y").small().color(Color32::from_rgb(104, 212, 120)));
                        columns[2].label(RichText::new("Z").small().color(Color32::from_rgb(106, 164, 240)));
                        columns[0].add(egui::DragValue::new(&mut pos.x).speed(0.05));
                        columns[1].add(egui::DragValue::new(&mut pos.y).speed(0.05));
                        columns[2].add(egui::DragValue::new(&mut pos.z).speed(0.05));
                    });
            });
        }
    } else {
        ui.colored_label(Color32::from_rgb(180, 150, 120), "Select an entity in the Outliner or viewport.");
    }

    section_shell(ui, "rnd", "Rendering", "render", true, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
            ui.label("Direct parameters (used by the forward PBR path).");
            ui.color_edit_button_rgb(&mut rend.color);
            ui.add(egui::Slider::new(&mut rend.metallic, 0.0..=1.0).text("Metallic"));
            ui.add(egui::Slider::new(&mut rend.roughness, 0.02..=1.0).text("Roughness"));
            ui.add(egui::Slider::new(&mut rend.ao, 0.0..=1.0).text("Ambient occlusion"));
            ui.separator();
            ui.label(
                RichText::new("This section drives the engine's current forward PBR material path.")
                    .small()
                    .color(Color32::from_rgb(141, 151, 165)),
            );
        } else {
            ui.label("Selected entity has no Renderable.");
        }
    });

    section_shell(ui, "phy", "Physics Body", "physics", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_rb = false;
        if let Ok(mut rb) = args.world.get::<&mut components::RigidBody>(entity) {
            egui::ComboBox::from_id_salt("body_type_combo")
                .selected_text(body_type_label(rb.body_type))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut rb.body_type, components::BodyType::Static, "Static");
                    ui.selectable_value(&mut rb.body_type, components::BodyType::Dynamic, "Dynamic");
                    ui.selectable_value(&mut rb.body_type, components::BodyType::Kinematic, "Kinematic");
                });
            ui.checkbox(&mut rb.use_gravity, "Use gravity");
            ui.checkbox(&mut rb.on_ground, "On ground (debug)");
            ui.checkbox(&mut rb.can_sleep, "Allow sleeping");
            ui.checkbox(&mut rb.sleeping, "Sleeping (debug)");
            ui.add(egui::Slider::new(&mut rb.velocity_x, -20.0..=20.0).text("Velocity X"));
            ui.add(egui::Slider::new(&mut rb.velocity_y, -30.0..=30.0).text("Velocity Y"));
            ui.add(egui::Slider::new(&mut rb.angular_velocity, -20.0..=20.0).text("Angular velocity"));
            ui.add(egui::Slider::new(&mut rb.mass, 0.1..=20.0).text("Mass"));
            ui.add(egui::Slider::new(&mut rb.inertia, 0.05..=40.0).text("Inertia"));
            ui.add(egui::Slider::new(&mut rb.friction, 0.0..=1.5).text("Friction"));
            ui.add(egui::Slider::new(&mut rb.restitution, 0.0..=1.0).text("Restitution"));
            ui.add(egui::Slider::new(&mut rb.linear_damping, 0.0..=2.0).text("Linear damping"));
            ui.add(egui::Slider::new(&mut rb.angular_damping, 0.0..=3.0).text("Angular damping"));
            ui.checkbox(&mut rb.lock_rotation, "Lock rotation");
            if ui.button("Reset velocity").clicked() {
                rb.velocity_x = 0.0;
                rb.velocity_y = 0.0;
                rb._velocity_z = 0.0;
                rb.angular_velocity = 0.0;
                rb.torque = 0.0;
            }
            if ui.button("Remove rigid body").clicked() {
                remove_rb = true;
            }
        } else {
            ui.label("No rigid body attached.");
            if ui.button("Add rigid body").clicked() {
                let body = components::RigidBody::dynamic();
                let _ = args.world.insert(
                    entity,
                    (body,),
                );
            }
        }
        if remove_rb {
            let _ = args.world.remove_one::<components::RigidBody>(entity);
        }
    });

    section_shell(ui, "jnt", "Hinge Joint", "hinge", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_joint = false;
        if let Ok(mut joint) = args.world.get::<&mut components::HingeJoint>(entity) {
            let mut bits = entity_bits_label(joint.connected);
            pick_entity_bits(ui, "Connected", &mut bits);
            if let Some(other) = hecs::Entity::from_bits(bits) {
                joint.connected = other;
            }
            ui.add(egui::Slider::new(&mut joint.rest_length, 0.0..=20.0).text("Rest length"));
            ui.add(egui::Slider::new(&mut joint.stiffness, 0.0..=1.0).text("Stiffness"));
            ui.add(egui::DragValue::new(&mut joint.anchor_a[0]).prefix("A X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_a[1]).prefix("A Y ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_a[2]).prefix("A Z ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_b[0]).prefix("B X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_b[1]).prefix("B Y ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_b[2]).prefix("B Z ").speed(0.05));
            if ui.button("Remove hinge joint").clicked() {
                remove_joint = true;
            }
        } else if ui.button("Add hinge joint").clicked() {
            let _ = args.world.insert(
                entity,
                (components::HingeJoint {
                    connected: entity,
                    rest_length: 0.0,
                    stiffness: 0.7,
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [0.0, 0.0, 0.0],
                },),
            );
        }
        if remove_joint {
            let _ = args.world.remove_one::<components::HingeJoint>(entity);
        }
    });

    section_shell(ui, "jnt", "Fixed Joint", "fixed", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_joint = false;
        if let Ok(mut joint) = args.world.get::<&mut components::FixedJoint>(entity) {
            let mut bits = entity_bits_label(joint.connected);
            pick_entity_bits(ui, "Connected", &mut bits);
            if let Some(other) = hecs::Entity::from_bits(bits) {
                joint.connected = other;
            }
            ui.add(egui::DragValue::new(&mut joint.offset_x).prefix("Offset X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.offset_y).prefix("Offset Y ").speed(0.05));
            ui.add(egui::Slider::new(&mut joint.stiffness, 0.0..=1.0).text("Stiffness"));
            ui.add(egui::DragValue::new(&mut joint.anchor_a[0]).prefix("A X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_a[1]).prefix("A Y ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_a[2]).prefix("A Z ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_b[0]).prefix("B X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_b[1]).prefix("B Y ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_b[2]).prefix("B Z ").speed(0.05));
            if ui.button("Remove fixed joint").clicked() {
                remove_joint = true;
            }
        } else if ui.button("Add fixed joint").clicked() {
            let _ = args.world.insert(
                entity,
                (components::FixedJoint {
                    connected: entity,
                    offset_x: 0.0,
                    offset_y: 0.0,
                    stiffness: 0.85,
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [0.0, 0.0, 0.0],
                },),
            );
        }
        if remove_joint {
            let _ = args.world.remove_one::<components::FixedJoint>(entity);
        }
    });

    section_shell(ui, "jnt", "Spring Joint", "spring", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_joint = false;
        if let Ok(mut joint) = args.world.get::<&mut components::SpringJoint>(entity) {
            let mut bits = entity_bits_label(joint.connected);
            pick_entity_bits(ui, "Connected", &mut bits);
            if let Some(other) = hecs::Entity::from_bits(bits) {
                joint.connected = other;
            }
            ui.add(egui::Slider::new(&mut joint.rest_length, 0.0..=20.0).text("Rest length"));
            ui.add(egui::Slider::new(&mut joint.stiffness, 0.0..=40.0).text("Stiffness"));
            ui.add(egui::Slider::new(&mut joint.damping, 0.0..=10.0).text("Damping"));
            ui.add(egui::DragValue::new(&mut joint.anchor_a[0]).prefix("A X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_a[1]).prefix("A Y ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_a[2]).prefix("A Z ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_b[0]).prefix("B X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_b[1]).prefix("B Y ").speed(0.05));
            ui.add(egui::DragValue::new(&mut joint.anchor_b[2]).prefix("B Z ").speed(0.05));
            if ui.button("Remove spring joint").clicked() {
                remove_joint = true;
            }
        } else if ui.button("Add spring joint").clicked() {
            let _ = args.world.insert(
                entity,
                (components::SpringJoint {
                    connected: entity,
                    rest_length: 1.0,
                    stiffness: 8.0,
                    damping: 1.2,
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [0.0, 0.0, 0.0],
                },),
            );
        }
        if remove_joint {
            let _ = args.world.remove_one::<components::SpringJoint>(entity);
        }
    });

    section_shell(ui, "jnt", "Rope Constraint", "rope", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_joint = false;
        if let Ok(mut rope) = args.world.get::<&mut components::RopeConstraint>(entity) {
            let mut bits = entity_bits_label(rope.connected);
            pick_entity_bits(ui, "Connected", &mut bits);
            if let Some(other) = hecs::Entity::from_bits(bits) {
                rope.connected = other;
            }
            ui.add(egui::Slider::new(&mut rope.max_length, 0.0..=30.0).text("Max length"));
            ui.add(egui::Slider::new(&mut rope.stiffness, 0.0..=1.0).text("Stiffness"));
            ui.add(egui::DragValue::new(&mut rope.anchor_a[0]).prefix("A X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut rope.anchor_a[1]).prefix("A Y ").speed(0.05));
            ui.add(egui::DragValue::new(&mut rope.anchor_a[2]).prefix("A Z ").speed(0.05));
            ui.add(egui::DragValue::new(&mut rope.anchor_b[0]).prefix("B X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut rope.anchor_b[1]).prefix("B Y ").speed(0.05));
            ui.add(egui::DragValue::new(&mut rope.anchor_b[2]).prefix("B Z ").speed(0.05));
            if ui.button("Remove rope constraint").clicked() {
                remove_joint = true;
            }
        } else if ui.button("Add rope constraint").clicked() {
            let _ = args.world.insert(
                entity,
                (components::RopeConstraint {
                    connected: entity,
                    max_length: 2.0,
                    stiffness: 0.9,
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [0.0, 0.0, 0.0],
                },),
            );
        }
        if remove_joint {
            let _ = args.world.remove_one::<components::RopeConstraint>(entity);
        }
    });

    section_shell(ui, "col", "Box Collision", "box_col", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_col = false;
        if let Ok(mut col) = args.world.get::<&mut components::Collider>(entity) {
            ui.label("AABB collider (physics uses axis-aligned boxes right now).");
            ui.add(egui::DragValue::new(&mut col.half_w).prefix("Half W ").speed(0.02).range(0.01..=256.0));
            ui.add(egui::DragValue::new(&mut col.half_h).prefix("Half H ").speed(0.02).range(0.01..=256.0));
            ui.add(egui::DragValue::new(&mut col.half_d).prefix("Half D ").speed(0.02).range(0.01..=256.0));
            ui.add(egui::DragValue::new(&mut col.layer).prefix("Layer ").speed(1.0).range(1..=u32::MAX));
            ui.add(egui::DragValue::new(&mut col.mask).prefix("Mask ").speed(1.0).range(1..=u32::MAX));
            if ui.button("Fit from render scale").clicked() {
                if let Ok(r) = args.world.get::<&components::Renderable>(entity) {
                    col.half_w = r.scale[0].abs() * 0.5;
                    col.half_h = r.scale[1].abs() * 0.5;
                    col.half_d = r.scale[2].abs() * 0.5;
                }
            }
            if ui.button("Remove collider").clicked() {
                remove_col = true;
            }
            ui.label(
                RichText::new("Collider rotation is editor-planned; current solver is AABB-only.")
                    .small()
                    .color(Color32::from_rgb(170, 160, 130)),
            );
        } else {
            ui.label("No collider attached.");
            if ui.button("Add box collider").clicked() {
                let mut half = [0.5, 0.5, 0.5];
                if let Ok(r) = args.world.get::<&components::Renderable>(entity) {
                    half = [r.scale[0].abs() * 0.5, r.scale[1].abs() * 0.5, r.scale[2].abs() * 0.5];
                }
                let _ = args.world.insert(
                    entity,
                    (components::Collider {
                        half_w: half[0],
                        half_h: half[1],
                        half_d: half[2],
                        layer: 1,
                        mask: 1,
                    },),
                );
            }
        }
        if remove_col {
            let _ = args.world.remove_one::<components::Collider>(entity);
        }
    });

    section_shell(ui, "obb", "OBB Collision (rotation-aware)", "obb_col", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_obb = false;
        if let Ok(mut obb) = args.world.get::<&mut components::OrientedBoxCollider>(entity) {
            ui.label("Rotated box collider. 3D depth is now exposed; full 3D SAT path is runtime-heavy.");
            ui.add(egui::DragValue::new(&mut obb.half_w).prefix("Half W ").speed(0.02).range(0.01..=256.0));
            ui.add(egui::DragValue::new(&mut obb.half_h).prefix("Half H ").speed(0.02).range(0.01..=256.0));
            ui.add(egui::DragValue::new(&mut obb.half_d).prefix("Half D ").speed(0.02).range(0.01..=256.0));
            ui.add(egui::DragValue::new(&mut obb.layer).prefix("Layer ").speed(1.0).range(1..=u32::MAX));
            ui.add(egui::DragValue::new(&mut obb.mask).prefix("Mask ").speed(1.0).range(1..=u32::MAX));
            let mut deg = obb.angle_rad.to_degrees();
            ui.add(egui::DragValue::new(&mut deg).prefix("Angle ").suffix(" deg").speed(0.5).range(-360.0..=360.0));
            obb.angle_rad = deg.to_radians();
            if ui.button("Match entity yaw").clicked() {
                if let Ok(r) = args.world.get::<&components::Rotation>(entity) {
                    obb.angle_rad = r.yaw;
                }
            }
            if ui.button("Remove OBB").clicked() {
                remove_obb = true;
            }
        } else {
            ui.label("No OBB collider attached.");
            if ui.button("Add OBB collider").clicked() {
                let mut half = [0.5, 0.5, 0.5];
                if let Ok(r) = args.world.get::<&components::Renderable>(entity) {
                    half = [r.scale[0].abs() * 0.5, r.scale[1].abs() * 0.5, r.scale[2].abs() * 0.5];
                }
                let mut angle = 0.0;
                if let Ok(rot) = args.world.get::<&components::Rotation>(entity) {
                    angle = rot.yaw;
                }
                let _ = args.world.insert(
                    entity,
                    (components::OrientedBoxCollider {
                        half_w: half[0],
                        half_h: half[1],
                        half_d: half[2],
                        angle_rad: angle,
                        layer: 1,
                        mask: 1,
                    },),
                );
            }
        }
        if remove_obb {
            let _ = args.world.remove_one::<components::OrientedBoxCollider>(entity);
        }
    });

    section_shell(ui, "mat", "Material Instances", "material", true, |ui| {
        ui.label("Instances multiply master defaults — same idea as UE master materials + material instances.");
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Select a mesh entity.");
            return;
        };
        if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
            ui.label(RichText::new("Quick instances").small().strong());
            ui.horizontal_wrapped(|ui| {
                for name in args.materials.instance_names() {
                    if ui.button(name).clicked() {
                        if let Err(e) = args.materials.apply_instance(name, &mut rend) {
                            args.error_log.push(format!("[Material] {}", e));
                        }
                    }
                }
            });
            ui.separator();
            ui.label(RichText::new("Masters").small().strong());
            for m in args.materials.master_names() {
                ui.label(RichText::new(format!("• {m}")).small());
            }
            if let Ok(mt) = args.world.get::<&components::MaterialTexture>(entity) {
                ui.separator();
                ui.label(RichText::new("Texture slots").small().strong());
                ui.monospace(format!(
                    "Albedo: {}",
                    if mt.path.is_empty() { "<default>" } else { &mt.path }
                ));
                ui.monospace(format!(
                    "Normal: {}",
                    if mt.normal_path.is_empty() {
                        "<default flat>"
                    } else {
                        &mt.normal_path
                    }
                ));
                ui.monospace(format!(
                    "Metal/Rough: {}",
                    if mt.metallic_roughness_path.is_empty() {
                        "<default>"
                    } else {
                        &mt.metallic_roughness_path
                    }
                ));
            }
        }
    });

    section_shell(ui, "fol", "Foliage Tools", "foliage", false, |ui| {
        ui.label("Spawn/remove foliage near the terrain cursor (same actions as Content Browser quick row).");
        let wx = args.terrain_cursor_x as f32 * args.terrain.cell_size - 32.0;
        let wz = args.terrain_cursor_z as f32 * args.terrain.cell_size - 32.0;
        ui.label(format!("Cursor world XZ: ({wx:.1}, {wz:.1})"));
        let mut ring_radius = ui
            .data_mut(|d| d.get_temp::<f32>("foliage_ring_radius".into()))
            .unwrap_or(4.0);
        let mut ring_count = ui
            .data_mut(|d| d.get_temp::<u32>("foliage_ring_count".into()))
            .unwrap_or(24);
        let mut remove_radius = ui
            .data_mut(|d| d.get_temp::<f32>("foliage_remove_radius".into()))
            .unwrap_or(4.5);
        let mut tree_physics = ui
            .data_mut(|d| d.get_temp::<bool>("foliage_tree_physics".into()))
            .unwrap_or(true);
        ui.add(egui::Slider::new(&mut ring_radius, 1.0..=24.0).text("Ring radius"));
        ui.add(egui::Slider::new(&mut ring_count, 4..=256).text("Ring instances"));
        ui.checkbox(&mut tree_physics, "Tree wind/physics");
        ui.add(egui::Slider::new(&mut remove_radius, 1.0..=30.0).text("Remove radius"));
        ui.data_mut(|d| {
            d.insert_temp("foliage_ring_radius".into(), ring_radius);
            d.insert_temp("foliage_ring_count".into(), ring_count);
            d.insert_temp("foliage_remove_radius".into(), remove_radius);
            d.insert_temp("foliage_tree_physics".into(), tree_physics);
        });
        if let Some(handle) = args.mesh_cache.get("meshes/cube.obj").copied() {
            if ui.button("Foliage ring (trees)").clicked() {
                spawn_foliage_ring(
                    args.world,
                    handle,
                    wx,
                    wz,
                    ring_radius,
                    ring_count as usize,
                    tree_physics,
                );
            }
        } else {
            ui.label("Load meshes/cube.obj first (add a cube to cache).");
        }
        if ui.button("Remove foliage near cursor").clicked() {
            let n = remove_nearby_foliage(args.world, wx, wz, remove_radius);
            if n > 0 {
                args.error_log
                    .push(format!("[Foliage] Removed {n} instances near cursor."));
            }
        }
        if ui.button("Paint foliage patch (64 instances)").clicked() {
            crate::editor::add_foliage_patch(args.world, args.meshes, args.mesh_cache);
        }
    });

    section_shell(ui, "ter", "Terrain Auto-Material", "terrain", false, |ui| {
        ui.label("Grass / dirt / rock blend by slope and height.");
        ui.add(
            egui::Slider::new(&mut args.terrain.material.slope_rock_start, 0.1..=1.6)
                .text("Rock from slope"),
        );
        ui.add(
            egui::Slider::new(&mut args.terrain.material.height_rock_start, 0.0..=6.0)
                .text("Rock from height"),
        );
    });
}
