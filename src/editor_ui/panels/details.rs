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
            for k in ["render", "physics", "box_col", "obb_col", "hinge", "fixed", "spring", "rope", "material", "foliage", "terrain", "terrain_brush", "rotation", "sphere_col", "capsule_col", "char_ctrl", "health", "fire_surf", "fire_src", "water_surf", "water_body", "smart_foliage", "lava_surf", "weather", "wind", "point_light", "material_extras", "water_trig", "splash", "script", "lighting"] {
                set_section_open(ui, k, true);
            }
        }
        if ui.small_button("Collapse All").clicked() {
            for k in ["render", "physics", "box_col", "obb_col", "hinge", "fixed", "spring", "rope", "material", "foliage", "terrain", "terrain_brush", "rotation", "sphere_col", "capsule_col", "char_ctrl", "health", "fire_surf", "fire_src", "water_surf", "water_body", "smart_foliage", "lava_surf", "weather", "wind", "point_light", "material_extras", "water_trig", "splash", "script", "lighting"] {
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
        let mut pending_apply: Option<(String, Result<(), String>)> = None;
        if let Ok(mut rend) = args.world.get::<&mut components::Renderable>(entity) {
            ui.label(RichText::new("Quick instances").small().strong());
            ui.horizontal_wrapped(|ui| {
                for name in args.materials.instance_names() {
                    if ui.button(&name).clicked() {
                        pending_apply = Some((name.clone(), args.materials.apply_instance(&name, &mut rend)));
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
        if let Some((name, res)) = pending_apply.take() {
            if let Err(e) = res {
                args.error_log.push(format!("[Material] {}", e));
            } else if let Ok(extras) = args.materials.instance_extras(&name) {
                let _ = args.world.insert(entity, (extras,));
            }
        }
    });

    section_shell(ui, "fol", "Foliage Tools", "foliage", false, |ui| {
        ui.label("Spawn/remove foliage near the terrain cursor (same actions as Content Browser quick row).");
        let wx = args.terrain_cursor_x as f32 * args.terrain_world.cell_size - 32.0;
        let wz = args.terrain_cursor_z as f32 * args.terrain_world.cell_size - 32.0;
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

    // ── Smart Foliage Asset Settings ────────────────────────────────
    section_shell(ui, "sfa", "Smart Foliage Asset", "smart_foliage", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_sfa = false;
        if let Ok(mut sfa) = args.world.get::<&mut components::SmartFoliageAsset>(entity) {
            ui.checkbox(&mut sfa.visible, "Visible");
            ui.checkbox(&mut sfa.locked, "Locked");
            ui.add(egui::Slider::new(&mut sfa.density_multiplier, 0.0..=5.0).text("Density multiplier"));
            ui.add(egui::Slider::new(&mut sfa.min_scale, 0.1..=3.0).text("Min scale"));
            ui.add(egui::Slider::new(&mut sfa.max_scale, 0.1..=3.0).text("Max scale"));
            ui.checkbox(&mut sfa.random_rotation, "Random rotation");
            ui.add(egui::Slider::new(&mut sfa.min_slope_deg, 0.0..=90.0).text("Min slope (deg)"));
            ui.add(egui::Slider::new(&mut sfa.max_slope_deg, 0.0..=90.0).text("Max slope (deg)"));
            ui.add(egui::Slider::new(&mut sfa.min_height, -10.0..=500.0).text("Min height"));
            ui.add(egui::Slider::new(&mut sfa.max_height, -10.0..=500.0).text("Max height"));
            egui::ComboBox::from_id_salt("foliage_paint_mode")
                .selected_text(format!("{:?}", sfa.paint_mode))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut sfa.paint_mode, components::FoliagePaintMode::Paint, "Paint");
                    ui.selectable_value(&mut sfa.paint_mode, components::FoliagePaintMode::Erase, "Erase");
                    ui.selectable_value(&mut sfa.paint_mode, components::FoliagePaintMode::Fill, "Fill");
                    ui.selectable_value(&mut sfa.paint_mode, components::FoliagePaintMode::Procedural, "Procedural");
                });
            if ui.button("Remove smart foliage asset").clicked() {
                remove_sfa = true;
            }
        } else if ui.button("Add smart foliage asset").clicked() {
            let _ = args.world.insert(entity, (components::SmartFoliageAsset::default(),));
        }
        if remove_sfa {
            let _ = args.world.remove_one::<components::SmartFoliageAsset>(entity);
        }
    });

    section_shell(ui, "ter", "Terrain Auto-Material", "terrain", false, |ui| {
        ui.label("Grass / dirt / rock blend by slope and height.");
        ui.add(
            egui::Slider::new(&mut args.terrain_world.material.slope_rock_start, 0.1..=1.6)
                .text("Rock from slope"),
        );
        ui.add(
            egui::Slider::new(&mut args.terrain_world.material.height_rock_start, 0.0..=6.0)
                .text("Rock from height"),
        );
    });

    // ── Terrain Editor Brush ─────────────────────────────────────────
    section_shell(ui, "tbr", "Terrain Editor", "terrain_brush", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Select an entity to enable terrain editing.");
            return;
        };
        // Add TerrainEditor component if missing.
        if args.world.get::<&components::TerrainEditor>(entity).is_err() {
            if ui.button("Enable Terrain Editor").clicked() {
                args.world.insert_one(entity, components::TerrainEditor::default()).ok();
            }
            ui.label("Attach a TerrainEditor to this entity for brush editing.");
            return;
        }
        // Edit TerrainEditor properties.
        if let Ok(mut te) = args.world.get::<&mut components::TerrainEditor>(entity) {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Active").small());
                ui.checkbox(&mut te.active, "");
                if te.active {
                    ui.colored_label(Color32::from_rgb(80, 200, 120), " Brush active");
                } else {
                    ui.colored_label(Color32::from_rgb(140, 100, 80), " Press T in viewport");
                }
            });
            ui.add_space(4.0);
            ui.label(RichText::new("Brush Mode").small().strong());
            let modes = [
                (components::TerrainBrushMode::Raise, "Raise [1]"),
                (components::TerrainBrushMode::Lower, "Lower [2]"),
                (components::TerrainBrushMode::Smooth, "Smooth [3]"),
                (components::TerrainBrushMode::Flatten, "Flatten [4]"),
                (components::TerrainBrushMode::Paint, "Paint [5]"),
                (components::TerrainBrushMode::Foliage, "Foliage [6]"),
            ];
            for (mode, label) in modes {
                let selected = te.brush_mode == mode;
                let btn = ui.selectable_label(selected, RichText::new(label).small());
                if btn.clicked() {
                    te.brush_mode = mode;
                }
            }
            ui.add_space(4.0);
            ui.add(
                egui::Slider::new(&mut te.brush_radius, 0.5..=50.0)
                    .text("Radius")
                    .prefix("R: "),
            );
            ui.add(
                egui::Slider::new(&mut te.brush_strength, 0.01..=2.0)
                    .text("Strength")
                    .prefix("S: "),
            );
            if te.brush_mode == components::TerrainBrushMode::Flatten {
                ui.add(
                    egui::Slider::new(&mut te.flatten_target, -50.0..=100.0)
                        .text("Target Height"),
                );
            }
            ui.checkbox(&mut te.show_cursor, "Show brush cursor");
            ui.add_space(4.0);
            ui.label(
                RichText::new("T = toggle mode · 1-6 = brush · [ / ] = radius")
                    .small()
                    .weak(),
            );
        }
    });

    // ── Rotation ─────────────────────────────────────────────────────
    section_shell(ui, "rot", "Rotation", "rotation", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_rot = false;
        if let Ok(mut rot) = args.world.get::<&mut components::Rotation>(entity) {
            let mut deg_pitch = rot.pitch.to_degrees();
            let mut deg_yaw = rot.yaw.to_degrees();
            let mut deg_roll = rot.roll.to_degrees();
            ui.add(egui::DragValue::new(&mut deg_pitch).prefix("Pitch ").suffix("°").speed(0.5).range(-360.0..=360.0));
            ui.add(egui::DragValue::new(&mut deg_yaw).prefix("Yaw ").suffix("°").speed(0.5).range(-360.0..=360.0));
            ui.add(egui::DragValue::new(&mut deg_roll).prefix("Roll ").suffix("°").speed(0.5).range(-360.0..=360.0));
            rot.pitch = deg_pitch.to_radians();
            rot.yaw = deg_yaw.to_radians();
            rot.roll = deg_roll.to_radians();
            if ui.button("Reset rotation").clicked() {
                rot.pitch = 0.0;
                rot.yaw = 0.0;
                rot.roll = 0.0;
            }
            if ui.button("Remove rotation").clicked() {
                remove_rot = true;
            }
        } else if ui.button("Add rotation").clicked() {
            let _ = args.world.insert(entity, (components::Rotation::default(),));
        }
        if remove_rot {
            let _ = args.world.remove_one::<components::Rotation>(entity);
        }
    });

    // ── Sphere Collider ─────────────────────────────────────────────
    section_shell(ui, "sph", "Sphere Collider", "sphere_col", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_col = false;
        if let Ok(mut col) = args.world.get::<&mut components::SphereCollider>(entity) {
            ui.add(egui::DragValue::new(&mut col.radius).prefix("Radius ").speed(0.02).range(0.01..=256.0));
            ui.add(egui::DragValue::new(&mut col.layer).prefix("Layer ").speed(1.0).range(1..=u32::MAX));
            ui.add(egui::DragValue::new(&mut col.mask).prefix("Mask ").speed(1.0).range(1..=u32::MAX));
            ui.checkbox(&mut col.is_trigger, "Is trigger");
            if ui.button("Remove sphere collider").clicked() {
                remove_col = true;
            }
        } else if ui.button("Add sphere collider").clicked() {
            let _ = args.world.insert(entity, (components::SphereCollider::default(),));
        }
        if remove_col {
            let _ = args.world.remove_one::<components::SphereCollider>(entity);
        }
    });

    // ── Capsule Collider ────────────────────────────────────────────
    section_shell(ui, "cap", "Capsule Collider", "capsule_col", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_col = false;
        if let Ok(mut col) = args.world.get::<&mut components::CapsuleCollider>(entity) {
            ui.add(egui::DragValue::new(&mut col.radius).prefix("Radius ").speed(0.02).range(0.01..=256.0));
            ui.add(egui::DragValue::new(&mut col.half_height).prefix("Half height ").speed(0.02).range(0.0..=256.0));
            ui.add(egui::DragValue::new(&mut col.layer).prefix("Layer ").speed(1.0).range(1..=u32::MAX));
            ui.add(egui::DragValue::new(&mut col.mask).prefix("Mask ").speed(1.0).range(1..=u32::MAX));
            ui.checkbox(&mut col.is_trigger, "Is trigger");
            if ui.button("Remove capsule collider").clicked() {
                remove_col = true;
            }
        } else if ui.button("Add capsule collider").clicked() {
            let _ = args.world.insert(entity, (components::CapsuleCollider::default(),));
        }
        if remove_col {
            let _ = args.world.remove_one::<components::CapsuleCollider>(entity);
        }
    });

    // ── Character Controller ────────────────────────────────────────
    section_shell(ui, "cc", "Character Controller", "char_ctrl", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_cc = false;
        if let Ok(mut cc) = args.world.get::<&mut components::CharacterController>(entity) {
            ui.add(egui::Slider::new(&mut cc.speed, 0.0..=30.0).text("Speed"));
            ui.add(egui::Slider::new(&mut cc.jump_force, 0.0..=30.0).text("Jump force"));
            ui.add(egui::Slider::new(&mut cc.ground_detect_dist, 0.01..=2.0).text("Ground detect dist"));
            ui.add(egui::Slider::new(&mut cc.max_slope_angle, 0.0..=std::f32::consts::FRAC_PI_2).text("Max slope angle (rad)"));
            ui.add(egui::Slider::new(&mut cc.step_height, 0.0..=1.0).text("Step height"));
            ui.add(egui::Slider::new(&mut cc.skin_width, 0.001..=0.1).text("Skin width"));
            ui.add(egui::Slider::new(&mut cc.gravity_scale, 0.0..=3.0).text("Gravity scale"));
            ui.checkbox(&mut cc.on_ground, "On ground (debug)");
            ui.checkbox(&mut cc.jump_pressed, "Jump pressed (debug)");
            if ui.button("Remove character controller").clicked() {
                remove_cc = true;
            }
        } else if ui.button("Add character controller").clicked() {
            let _ = args.world.insert(entity, (components::CharacterController::default(),));
        }
        if remove_cc {
            let _ = args.world.remove_one::<components::CharacterController>(entity);
        }
    });

    // ── Health ──────────────────────────────────────────────────────
    section_shell(ui, "hp", "Health", "health", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_hp = false;
        if let Ok(mut hp) = args.world.get::<&mut components::Health>(entity) {
            ui.add(egui::DragValue::new(&mut hp.current).prefix("Current ").speed(1).range(0..=i32::MAX));
            ui.add(egui::DragValue::new(&mut hp.max).prefix("Max ").speed(1).range(1..=i32::MAX));
            if hp.current > hp.max {
                hp.current = hp.max;
            }
            if ui.button("Reset to max").clicked() {
                hp.current = hp.max;
            }
            if ui.button("Remove health").clicked() {
                remove_hp = true;
            }
        } else if ui.button("Add health").clicked() {
            let _ = args.world.insert(entity, (components::Health::default(),));
        }
        if remove_hp {
            let _ = args.world.remove_one::<components::Health>(entity);
        }
    });

    // ── Fire Surface ────────────────────────────────────────────────
    section_shell(ui, "fss", "Fire Surface", "fire_surf", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_fs = false;
        if let Ok(mut fs) = args.world.get::<&mut components::FireSurface>(entity) {
            ui.color_edit_button_rgb(&mut fs.base_color);
            ui.label("Base color");
            ui.color_edit_button_rgb(&mut fs.tip_color);
            ui.label("Tip color");
            ui.add(egui::Slider::new(&mut fs.intensity, 0.0..=20.0).text("Intensity"));
            ui.add(egui::Slider::new(&mut fs.flame_speed, 0.0..=2.0).text("Flame speed"));
            ui.add(egui::Slider::new(&mut fs.noise_scale, 0.1..=10.0).text("Noise scale"));
            ui.add(egui::Slider::new(&mut fs.flicker_strength, 0.0..=1.0).text("Flicker strength"));
            ui.add(egui::Slider::new(&mut fs.flame_height, 0.1..=10.0).text("Flame height"));
            ui.add(egui::Slider::new(&mut fs.opacity, 0.0..=1.0).text("Opacity"));
            // ── Emissive light emission controls ─────────────────────────
            ui.separator();
            ui.label(RichText::new("Dynamic Light Emission").strong());
            ui.add(egui::Slider::new(&mut fs.emissive_light_strength, 0.0..=20.0).text("Light strength"));
            ui.add(egui::Slider::new(&mut fs.emissive_light_radius, 0.1..=50.0).text("Light radius"));
            ui.color_edit_button_rgb(&mut fs.emissive_light_color);
            ui.label("Light color");
            if ui.button("Remove fire surface").clicked() {
                remove_fs = true;
            }
        } else if ui.button("Add fire surface").clicked() {
            let _ = args.world.insert(entity, (components::FireSurface::default(),));
        }
        if remove_fs {
            let _ = args.world.remove_one::<components::FireSurface>(entity);
        }
    });

    // ── Fire Source ─────────────────────────────────────────────────
    section_shell(ui, "fsp", "Fire Source", "fire_src", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_fs = false;
        if let Ok(mut fs) = args.world.get::<&mut components::FireSource>(entity) {
            ui.add(egui::Slider::new(&mut fs.intensity, 0.0..=5.0).text("Intensity"));
            ui.add(egui::Slider::new(&mut fs.radius, 0.1..=20.0).text("Radius"));
            ui.add(egui::Slider::new(&mut fs.flame_height, 0.1..=10.0).text("Flame height"));
            ui.add(egui::Slider::new(&mut fs.smoke_amount, 0.0..=1.0).text("Smoke amount"));
            ui.add(egui::Slider::new(&mut fs.ember_amount, 0.0..=1.0).text("Ember amount"));
            ui.add(egui::Slider::new(&mut fs.wind_susceptibility, 0.0..=1.0).text("Wind susceptibility"));
            ui.checkbox(&mut fs.damaging, "Damaging");
            ui.add(egui::Slider::new(&mut fs.damage_per_second, 0.0..=100.0).text("Damage/sec"));
            if ui.button("Remove fire source").clicked() {
                remove_fs = true;
            }
        } else if ui.button("Add fire source").clicked() {
            let _ = args.world.insert(entity, (components::FireSource::default(),));
        }
        if remove_fs {
            let _ = args.world.remove_one::<components::FireSource>(entity);
        }
    });

    // ── Water Surface ───────────────────────────────────────────────
    section_shell(ui, "wfs", "Water Surface", "water_surf", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_ws = false;
        if let Ok(mut ws) = args.world.get::<&mut components::WaterSurface>(entity) {
            ui.add(egui::Slider::new(&mut ws.wave_height, 0.0..=5.0).text("Wave height"));
            ui.add(egui::Slider::new(&mut ws.wave_speed, 0.0..=3.0).text("Wave speed"));
            ui.color_edit_button_rgb(&mut ws.deep_color);
            ui.label("Deep color");
            ui.color_edit_button_rgb(&mut ws.shallow_color);
            ui.label("Shallow color");
            ui.add(egui::Slider::new(&mut ws.opacity, 0.0..=1.0).text("Opacity"));
            ui.add(egui::Slider::new(&mut ws.foam_intensity, 0.0..=1.0).text("Foam intensity"));
            ui.add(egui::Slider::new(&mut ws.specular_power, 1.0..=1024.0).text("Specular power"));
            if ui.button("Remove water surface").clicked() {
                remove_ws = true;
            }
        } else if ui.button("Add water surface").clicked() {
            let _ = args.world.insert(entity, (components::WaterSurface::default(),));
        }
        if remove_ws {
            let _ = args.world.remove_one::<components::WaterSurface>(entity);
        }
    });

    // ── Smart Water Body ───────────────────────────────────────────
    section_shell(ui, "wbd", "Water Body", "water_body", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_wb = false;
        if let Ok(mut wb) = args.world.get::<&mut components::WaterBody>(entity) {
            egui::ComboBox::from_id_salt("wb_type")
                .selected_text(wb.body_type.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut wb.body_type, components::WaterBodyType::Ocean, "Ocean");
                    ui.selectable_value(&mut wb.body_type, components::WaterBodyType::Lake, "Lake");
                    ui.selectable_value(&mut wb.body_type, components::WaterBodyType::River, "River");
                    ui.selectable_value(&mut wb.body_type, components::WaterBodyType::Pond, "Pond");
                    ui.selectable_value(&mut wb.body_type, components::WaterBodyType::Stream, "Stream");
                    ui.selectable_value(&mut wb.body_type, components::WaterBodyType::Waterfall, "Waterfall");
                    ui.selectable_value(&mut wb.body_type, components::WaterBodyType::Swamp, "Swamp");
                });
            ui.add(egui::Slider::new(&mut wb.size_x, 5.0..=500.0).text("Size X"));
            ui.add(egui::Slider::new(&mut wb.size_z, 5.0..=500.0).text("Size Z"));
            ui.add(egui::Slider::new(&mut wb.depth, 0.5..=100.0).text("Depth"));
            ui.add(egui::Slider::new(&mut wb.flow_speed, 0.0..=10.0).text("Flow speed"));
            ui.add(egui::Slider::new(&mut wb.turbulence, 0.0..=2.0).text("Turbulence"));
            ui.checkbox(&mut wb.auto_surface, "Auto surface");
            ui.checkbox(&mut wb.auto_physics, "Auto physics");
            ui.checkbox(&mut wb.auto_collision, "Auto collision");
            ui.checkbox(&mut wb.auto_reflections, "Auto reflections");
            ui.checkbox(&mut wb.auto_underwater, "Auto underwater");
            ui.add(egui::Slider::new(&mut wb.lod_distance, 50.0..=1000.0).text("LOD distance"));
            if ui.button("Remove water body").clicked() {
                remove_wb = true;
            }
        } else if ui.button("Add water body").clicked() {
            let _ = args.world.insert(entity, (components::WaterBody::default(),));
        }
        if remove_wb {
            let _ = args.world.remove_one::<components::WaterBody>(entity);
        }
    });

    // ── Lava Surface ────────────────────────────────────────────────
    section_shell(ui, "lvs", "Lava Surface", "lava_surf", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_ls = false;
        if let Ok(mut ls) = args.world.get::<&mut components::LavaSurface>(entity) {
            ui.color_edit_button_rgb(&mut ls.rock_color);
            ui.label("Rock color");
            ui.color_edit_button_rgb(&mut ls.emissive_color);
            ui.label("Emissive color");
            ui.add(egui::Slider::new(&mut ls.emissive_intensity, 0.0..=20.0).text("Emissive intensity"));
            ui.add(egui::Slider::new(&mut ls.flow_speed, 0.0..=1.0).text("Flow speed"));
            ui.add(egui::Slider::new(&mut ls.crack_scale, 0.1..=10.0).text("Crack scale"));
            ui.add(egui::Slider::new(&mut ls.crack_threshold, 0.0..=1.0).text("Crack threshold"));
            ui.add(egui::Slider::new(&mut ls.displacement_amp, 0.0..=0.5).text("Displacement amp"));
            ui.add(egui::Slider::new(&mut ls.opacity, 0.0..=1.0).text("Opacity"));
            // ── Emissive light emission controls ─────────────────────────
            ui.separator();
            ui.label(RichText::new("Dynamic Light Emission").strong());
            ui.add(egui::Slider::new(&mut ls.emissive_light_strength, 0.0..=20.0).text("Light strength"));
            ui.add(egui::Slider::new(&mut ls.emissive_light_radius, 0.1..=50.0).text("Light radius"));
            ui.color_edit_button_rgb(&mut ls.emissive_light_color);
            ui.label("Light color");
            if ui.button("Remove lava surface").clicked() {
                remove_ls = true;
            }
        } else if ui.button("Add lava surface").clicked() {
            let _ = args.world.insert(entity, (components::LavaSurface::default(),));
        }
        if remove_ls {
            let _ = args.world.remove_one::<components::LavaSurface>(entity);
        }
    });

    // ── Weather Zone ────────────────────────────────────────────────
    section_shell(ui, "wz", "Weather Zone", "weather", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_wz = false;
        if let Ok(mut wz) = args.world.get::<&mut components::WeatherZone>(entity) {
            egui::ComboBox::from_id_salt("weather_condition_combo")
                .selected_text(format!("{:?}", wz.condition))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut wz.condition, crate::environment::weather::WeatherCondition::Clear, "Clear");
                    ui.selectable_value(&mut wz.condition, crate::environment::weather::WeatherCondition::Cloudy, "Cloudy");
                    ui.selectable_value(&mut wz.condition, crate::environment::weather::WeatherCondition::Overcast, "Overcast");
                    ui.selectable_value(&mut wz.condition, crate::environment::weather::WeatherCondition::LightRain, "Light Rain");
                    ui.selectable_value(&mut wz.condition, crate::environment::weather::WeatherCondition::HeavyRain, "Heavy Rain");
                    ui.selectable_value(&mut wz.condition, crate::environment::weather::WeatherCondition::Snow, "Snow");
                    ui.selectable_value(&mut wz.condition, crate::environment::weather::WeatherCondition::Fog, "Fog");
                    ui.selectable_value(&mut wz.condition, crate::environment::weather::WeatherCondition::Storm, "Storm");
                });
            ui.add(egui::Slider::new(&mut wz.intensity, 0.0..=1.0).text("Intensity"));
            ui.add(egui::Slider::new(&mut wz.radius, 1.0..=500.0).text("Radius"));
            ui.add(egui::Slider::new(&mut wz.falloff, 0.0..=100.0).text("Falloff"));
            ui.checkbox(&mut wz.active, "Active");
            if ui.button("Remove weather zone").clicked() {
                remove_wz = true;
            }
        } else if ui.button("Add weather zone").clicked() {
            let _ = args.world.insert(entity, (components::WeatherZone::default(),));
        }
        if remove_wz {
            let _ = args.world.remove_one::<components::WeatherZone>(entity);
        }
    });

    // ── Wind Zone ───────────────────────────────────────────────────
    section_shell(ui, "wnd", "Wind Zone", "wind", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_wz = false;
        if let Ok(mut wz) = args.world.get::<&mut components::WindZone>(entity) {
            ui.label(RichText::new("Direction").small().strong());
            ui.add(egui::DragValue::new(&mut wz.direction[0]).prefix("X ").speed(0.05));
            ui.add(egui::DragValue::new(&mut wz.direction[1]).prefix("Y ").speed(0.05));
            ui.add(egui::DragValue::new(&mut wz.direction[2]).prefix("Z ").speed(0.05));
            ui.add(egui::Slider::new(&mut wz.strength, 0.0..=5.0).text("Strength"));
            ui.add(egui::Slider::new(&mut wz.radius, 1.0..=500.0).text("Radius"));
            ui.add(egui::Slider::new(&mut wz.falloff, 0.0..=100.0).text("Falloff"));
            ui.checkbox(&mut wz.active, "Active");
            if ui.button("Remove wind zone").clicked() {
                remove_wz = true;
            }
        } else if ui.button("Add wind zone").clicked() {
            let _ = args.world.insert(entity, (components::WindZone::default(),));
        }
        if remove_wz {
            let _ = args.world.remove_one::<components::WindZone>(entity);
        }
    });

    // ── Point Light ─────────────────────────────────────────────────
    section_shell(ui, "lit", "Point Light", "point_light", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_pl = false;
        if let Ok(mut pl) = args.world.get::<&mut components::PointLight>(entity) {
            ui.color_edit_button_rgb(&mut pl.color);
            ui.label("Color");
            ui.add(egui::Slider::new(&mut pl.intensity, 0.0..=50.0).text("Intensity"));
            ui.add(egui::Slider::new(&mut pl.range, 0.0..=200.0).text("Range"));
            egui::ComboBox::from_id_salt("light_type_combo")
                .selected_text(match pl.light_type as u32 {
                    0 => "Directional",
                    1 => "Point",
                    2 => "Spot",
                    _ => "Unknown",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut pl.light_type, 0.0, "Directional");
                    ui.selectable_value(&mut pl.light_type, 1.0, "Point");
                    ui.selectable_value(&mut pl.light_type, 2.0, "Spot");
                });
            ui.add(egui::Slider::new(&mut pl.spot_angle, 5.0..=170.0).text("Spot angle (°)"));
            ui.checkbox(&mut pl.shadow_casting, "Shadow casting");
            if ui.button("Remove point light").clicked() {
                remove_pl = true;
            }
        } else if ui.button("Add point light").clicked() {
            let _ = args.world.insert(entity, (components::PointLight::default(),));
        }
        if remove_pl {
            let _ = args.world.remove_one::<components::PointLight>(entity);
        }
    });

    // ── Material Extras ─────────────────────────────────────────────
    section_shell(ui, "mtx", "Material Extras", "material_extras", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_mx = false;
        if let Ok(mut mx) = args.world.get::<&mut components::MaterialExtras>(entity) {
            ui.add(egui::Slider::new(&mut mx.subsurface, 0.0..=1.0).text("Subsurface"));
            ui.add(egui::Slider::new(&mut mx.clearcoat, 0.0..=1.0).text("Clearcoat"));
            ui.add(egui::Slider::new(&mut mx.clearcoat_roughness, 0.0..=1.0).text("Clearcoat roughness"));
            ui.add(egui::Slider::new(&mut mx.emissive_strength, 0.0..=10.0).text("Emissive strength"));
            ui.separator();
            let mut checker_on = mx.checker > 0.5;
            if ui.checkbox(&mut checker_on, "Checkerboard (UE5-style debug grid)").changed() {
                mx.checker = if checker_on { 1.0 } else { 0.0 };
            }
            if checker_on {
                ui.add(egui::Slider::new(&mut mx.checker_scale, 0.05..=10.0).text("Checker tile size (m)"));
            }
            ui.label(
                RichText::new("Values are uploaded to the GPU as material_extras (binding 6).")
                    .small()
                    .color(Color32::from_rgb(141, 151, 165)),
            );
            if ui.button("Reset defaults").clicked() {
                mx.subsurface = 0.0;
                mx.clearcoat = 0.0;
                mx.clearcoat_roughness = 0.0;
                mx.emissive_strength = 0.0;
                mx.checker = 0.0;
                mx.checker_scale = 1.0;
            }
            if ui.button("Remove material extras").clicked() {
                remove_mx = true;
            }
        } else if ui.button("Add material extras").clicked() {
            let _ = args.world.insert(entity, (components::MaterialExtras::default(),));
        }
        if remove_mx {
            let _ = args.world.remove_one::<components::MaterialExtras>(entity);
        }
    });

    // ── Water Trigger ───────────────────────────────────────────────
    section_shell(ui, "wtr", "Water Trigger", "water_trig", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_wt = false;
        if let Ok(mut wt) = args.world.get::<&mut components::WaterTrigger>(entity) {
            ui.add(egui::Slider::new(&mut wt.splash_intensity, 0.0..=2.0).text("Splash intensity"));
            ui.checkbox(&mut wt.active, "Active");
            if ui.button("Remove water trigger").clicked() {
                remove_wt = true;
            }
        } else if ui.button("Add water trigger").clicked() {
            let _ = args.world.insert(entity, (components::WaterTrigger::default(),));
        }
        if remove_wt {
            let _ = args.world.remove_one::<components::WaterTrigger>(entity);
        }
    });

    // ── Splash Effect ───────────────────────────────────────────────
    section_shell(ui, "spl", "Splash Effect", "splash", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_se = false;
        if let Ok(mut se) = args.world.get::<&mut components::SplashEffect>(entity) {
            ui.add(egui::DragValue::new(&mut se.max_splashes).prefix("Max splashes ").speed(1).range(1..=u32::MAX));
            ui.add(egui::Slider::new(&mut se.splash_duration, 0.1..=5.0).text("Splash duration"));
            ui.add(egui::Slider::new(&mut se.ripple_scale, 0.1..=5.0).text("Ripple scale"));
            ui.checkbox(&mut se.active, "Active");
            if ui.button("Remove splash effect").clicked() {
                remove_se = true;
            }
        } else if ui.button("Add splash effect").clicked() {
            let _ = args.world.insert(entity, (components::SplashEffect::default(),));
        }
        if remove_se {
            let _ = args.world.remove_one::<components::SplashEffect>(entity);
        }
    });

    // ── Occluder ────────────────────────────────────────────────────
    // Marks this entity as a large static volume that hides geometry behind
    // it. Feeds the CPU occlusion culler so hidden meshes are never sent to
    // the GPU. Radius is the occluding volume's radius in world units.
    section_shell(ui, "occ", "Occluder", "occluder", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_occ = false;
        if let Ok(mut occ) = args.world.get::<&mut components::Occluder>(entity) {
            ui.add(egui::Slider::new(&mut occ.radius, 0.5..=200.0).text("Occlusion radius"));
            ui.label(
                RichText::new("Large bodies (buildings, terrain walls) hide meshes behind them, \
so far-away geometry inside the culled area is skipped.")
                    .small()
                    .color(Color32::from_rgb(141, 151, 165)),
            );
            if ui.button("Remove occluder").clicked() {
                remove_occ = true;
            }
        } else if ui.button("Add occluder").clicked() {
            let _ = args.world.insert(entity, (components::Occluder::default(),));
        }
        if remove_occ {
            let _ = args.world.remove_one::<components::Occluder>(entity);
        }
    });

    // ── Script ──────────────────────────────────────────────────────
    section_shell(ui, "scr", "Script", "script", false, |ui| {
        let Some(entity) = args.selected_renderable.as_ref().copied() else {
            ui.label("Nothing selected.");
            return;
        };
        let mut remove_scr = false;
        if let Ok(mut scr) = args.world.get::<&mut components::Script>(entity) {
            ui.label(RichText::new("Lua script path").small().strong());
            ui.monospace(&scr.path);
            let mut new_path = scr.path.clone();
            ui.add(egui::TextEdit::singleline(&mut new_path).hint_text("scripts/my_script.lua"));
            if ui.button("Apply path").clicked() {
                scr.path = new_path;
            }
            if ui.button("Remove script").clicked() {
                remove_scr = true;
            }
        } else if ui.button("Add script").clicked() {
            let _ = args.world.insert(entity, (components::Script::default(),));
        }
        if remove_scr {
            let _ = args.world.remove_one::<components::Script>(entity);
        }
    });

    // ── Lighting: baked light probes ──────────────────────────────────
    // These little light-bulbs capture the "bounced" light of the scene.
    // You place them, hit Bake Lighting, and the baked indirect light is
    // saved next to the scene and re-loaded every time the level opens.
    section_shell(ui, "LGT", "Light Probes (Baked GI)", "lighting", true, |ui| {
        ui.label(
            RichText::new("Bounced/indirect light. Place probes around rooms and \
under trees, then press Bake Lighting.")
                .small()
                .color(Color32::from_rgb(141, 151, 165)),
        );

        let mut hidden = ui
            .data_mut(|d| d.get_temp::<bool>("probes_hidden".into()))
            .unwrap_or(false);
        if ui.checkbox(&mut hidden, "Hide probes in viewport").changed() {
            ui.data_mut(|d| d.insert_temp("probes_hidden".into(), hidden));
            if hidden {
                // Also drop the selection so no halo lingers after hiding.
                ui.data_mut(|d| d.insert_temp("probe_selected_index".into(), Option::<usize>::None));
            }
        }
        ui.label(
            RichText::new("Hiding only stops them being drawn — they stay in the scene and still light it.")
                .small()
                .color(Color32::from_rgb(121, 131, 145)),
        );

        ui.horizontal(|ui| {
            if ui.button("Add at Camera").clicked() {
                let pos = args.camera.position();
                args.renderer.light_probes.add_probe(pos, 12.0);
                let idx = args.renderer.light_probes.probes.len() - 1;
                ui.data_mut(|d| d.insert_temp("probe_selected_index".into(), Some(idx)));
            }
            if ui.button("Add at Selection").clicked() {
                if let Some(e) = args.selected_renderable.as_ref().copied() {
                    if let Ok(pos) = args.world.get::<&components::Position>(e) {
                        args.renderer.light_probes
                            .add_probe(glam::Vec3::new(pos.x, pos.y, pos.z), 12.0);
                        let idx = args.renderer.light_probes.probes.len() - 1;
                        ui.data_mut(|d| d.insert_temp("probe_selected_index".into(), Some(idx)));
                    }
                }
            }
        });

        ui.horizontal(|ui| {
            if ui.button("Bake Lighting").clicked() {
                *args.bake_requested = true;
                args.error_log.push("[Lighting] Bake requested".to_string());
            }
            if ui.button("Clear All").clicked() {
                args.renderer.light_probes.probes.clear();
                args.renderer.light_probes.volumes.clear();
                ui.data_mut(|d| d.insert_temp("probe_selected_index".into(), Option::<usize>::None));
                ui.data_mut(|d| d.insert_temp("probe_group_set".into(), std::collections::HashSet::<usize>::new()));
                ui.data_mut(|d| d.insert_temp("probes_hidden".into(), false));
                args.error_log.push("[Lighting] Cleared all probes & volumes".to_string());
            }
        });

        // ── Probe volumes (HFW-style box placement) ─────────────────────
        ui.separator();
        ui.label(
            RichText::new("Probe Volumes")
                .small()
                .strong()
                .color(Color32::from_rgb(168, 176, 188)),
        );
        ui.label(
            RichText::new("Drop a box over a room — it fills itself with a probe grid.")
                .small()
                .color(Color32::from_rgb(121, 131, 145)),
        );
        ui.horizontal(|ui| {
            if ui.button("Add Volume at Camera").clicked() {
                let pos = args.camera.position();
                let n = args.renderer.light_probes.add_volume(pos, glam::Vec3::splat(24.0), [3, 2, 3]);
                args.error_log.push(format!("[Lighting] Added probe volume ({n} probes)"));
            }
            if ui.button("Repopulate All").clicked() {
                let vols = args.renderer.light_probes.volumes.len();
                for i in 0..vols {
                    args.renderer.light_probes.repopulate_volume(i);
                }
                args.error_log.push("[Lighting] Repopulated volumes".to_string());
            }
        });
        let mut remove_vol: Option<usize> = None;
        let mut repop_vol: Option<usize> = None;
        for v in 0..args.renderer.light_probes.volumes.len() {
            let mut vol = args.renderer.light_probes.volumes[v];
            egui::CollapsingHeader::new(format!("Volume {v}"))
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Center");
                        ui.add(egui::DragValue::new(&mut vol.center.x).speed(0.1));
                        ui.add(egui::DragValue::new(&mut vol.center.y).speed(0.1));
                        ui.add(egui::DragValue::new(&mut vol.center.z).speed(0.1));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Size");
                        ui.add(egui::DragValue::new(&mut vol.size.x).speed(0.1).range(1.0..=200.0));
                        ui.add(egui::DragValue::new(&mut vol.size.y).speed(0.1).range(1.0..=200.0));
                        ui.add(egui::DragValue::new(&mut vol.size.z).speed(0.1).range(1.0..=200.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Density");
                        for (a, axis) in ["X", "Y", "Z"].iter().enumerate() {
                            ui.label(*axis);
                            ui.add(egui::Slider::new(&mut vol.density[a], 1..=6));
                        }
                    });
                    if ui.button("Apply / Repopulate this volume").clicked() {
                        repop_vol = Some(v);
                    }
                    if ui.button("Remove volume").clicked() {
                        remove_vol = Some(v);
                    }
                });
            if v < args.renderer.light_probes.volumes.len() {
                args.renderer.light_probes.volumes[v] = vol;
            }
        }
        if let Some(v) = repop_vol {
            args.renderer.light_probes.repopulate_volume(v);
        }
        if let Some(v) = remove_vol {
            args.renderer.light_probes.remove_volume(v);
            args.error_log.push("[Lighting] Removed volume".to_string());
        }

        ui.separator();

        let probe_count = args.renderer.light_probes.probes.len();
        ui.label(
            RichText::new(format!("{} probe(s)", probe_count))
                .small()
                .strong()
                .color(Color32::from_rgb(168, 176, 188)),
        );

        let mut selected = ui
            .data_mut(|d| d.get_temp::<Option<usize>>("probe_selected_index".into()))
            .flatten();
        if selected.is_some() && selected.unwrap() >= probe_count {
            selected = None;
            ui.data_mut(|d| d.insert_temp("probe_selected_index".into(), Option::<usize>::None));
        }

        if probe_count > 0 {
            let mut group_set: std::collections::HashSet<usize> = ui
                .data_mut(|d| d.get_temp::<std::collections::HashSet<usize>>("probe_group_set".into()))
                .unwrap_or_default();
            egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                for (i, probe) in args.renderer.light_probes.probes.iter().enumerate() {
                    let mut in_set = group_set.contains(&i);
                    let label = format!(
                        "Probe {i}  ({:.1}, {:.1}, {:.1}){}",
                        probe.position.x,
                        probe.position.y,
                        probe.position.z,
                        if probe.group != 0 { format!("  [G{}]", probe.group) } else { String::new() }
                    );
                    ui.horizontal(|ui| {
                        if ui.checkbox(&mut in_set, "").changed() {
                            if in_set {
                                group_set.insert(i);
                            } else {
                                group_set.remove(&i);
                            }
                        }
                        if ui
                            .selectable_label(selected == Some(i), label)
                            .clicked()
                        {
                            selected = Some(i);
                            ui.data_mut(|d| d.insert_temp("probe_selected_index".into(), selected));
                        }
                    });
                }
            });
            ui.data_mut(|d| d.insert_temp("probe_group_set".into(), group_set.clone()));

            // ── Group controls ────────────────────────────────────────────
            let set: Vec<usize> = {
                let s: std::collections::HashSet<usize> = ui
                    .data(|d| d.get_temp::<std::collections::HashSet<usize>>("probe_group_set".into()))
                    .unwrap_or_default();
                let mut v: Vec<usize> = s.into_iter().collect();
                v.sort_unstable();
                v
            };
            ui.label(
                RichText::new(format!("{} probe(s) checked for grouping", set.len()))
                    .small()
                    .color(Color32::from_rgb(121, 131, 145)),
            );
            ui.horizontal(|ui| {
                if ui.button("Group checked").clicked() {
                    if !set.is_empty() {
                        let gid = args.renderer.light_probes.next_group_id();
                        args.renderer.light_probes.assign_group(&set, gid);
                        args.error_log.push(format!("[Lighting] Grouped {} probe(s) as group {gid}", set.len()));
                    }
                }
                if ui.button("Ungroup checked").clicked() {
                    args.renderer.light_probes.assign_group(&set, 0);
                    args.error_log.push("[Lighting] Ungrouped checked probes".to_string());
                }
                if ui.button("Clear checks").clicked() {
                    ui.data_mut(|d| d.insert_temp("probe_group_set".into(), std::collections::HashSet::<usize>::new()));
                }
            });
            ui.label(
                RichText::new("Drag any probe in a group (in the viewport or the list) and the whole group moves together.")
                    .small()
                    .color(Color32::from_rgb(121, 131, 145)),
            );
        }

        if let Some(idx) = selected {
            if idx < args.renderer.light_probes.probes.len() {
                let mut remove = false;
                let gid = args.renderer.light_probes.probes[idx].group;
                {
                    let start_pos = args.renderer.light_probes.probes[idx].position;
                    let mut edit_pos = start_pos;
                    {
                        let probe = &mut args.renderer.light_probes.probes[idx];
                        ui.horizontal(|ui| {
                            ui.label("Radius");
                            ui.add(egui::Slider::new(&mut probe.radius, 2.0..=80.0));
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.label("Position");
                        ui.add(
                            egui::DragValue::new(&mut edit_pos.x).speed(0.1).prefix("x "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut edit_pos.y).speed(0.1).prefix("y "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut edit_pos.z).speed(0.1).prefix("z "),
                        );
                    });
                    let delta = edit_pos - start_pos;
                    if delta.length_squared() > 0.0001 {
                        if gid != 0 {
                            args.renderer.light_probes.move_probe_or_group(idx, delta);
                        } else {
                            args.renderer.light_probes.probes[idx].position = edit_pos;
                        }
                    }
                    if gid != 0 {
                        ui.label(
                            RichText::new(format!("In group G{gid} — dragging moves the whole group."))
                                .small()
                                .color(Color32::from_rgb(255, 214, 102)),
                        );
                    }
                }
                ui.label(
                    RichText::new("A bigger radius makes the probe spread its light further; \
overlap two probes to blend between them.")
                        .small()
                        .color(Color32::from_rgb(121, 131, 145)),
                );
                if ui.button("Remove Selected").clicked() {
                    remove = true;
                }
                if remove {
                    args.renderer.light_probes.probes.remove(idx);
                    ui.data_mut(|d| d.insert_temp("probe_selected_index".into(), Option::<usize>::None));
                }
            }
        } else {
            ui.label(RichText::new("Click a probe in the list or viewport to edit it.").small().weak());
        }
    });
}
