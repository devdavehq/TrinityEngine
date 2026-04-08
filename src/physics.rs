use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use crate::components::{
    BodyType, Collider, CollisionPair, CollisionPhase, FixedJoint, FoliageWind, HingeJoint,
    OrientedBoxCollider, Position, RigidBody, RopeConstraint, Rotation, SpringJoint,
};
use crate::jobs::JobSystem;
use crate::settings::RuntimeSettings;
use hecs::{Entity, World};

const GRAVITY: f32 = 9.8;
const TERMINAL_VELOCITY: f32 = 20.0;
const SLEEP_LINEAR_THRESHOLD: f32 = 0.02;
const SLEEP_TIME_THRESHOLD: f32 = 0.45;
const POSITION_SLOP: f32 = 0.005;
const POSITION_PERCENT: f32 = 0.78;

#[derive(Clone, Copy)]
enum ShapeKind {
    Box3D,
    Obb2D,
}

#[derive(Clone, Copy)]
struct BodyShape {
    center: glam::Vec3,
    half: glam::Vec3,
    rotation: glam::Quat,
    angle: f32,
    layer: u32,
    mask: u32,
    kind: ShapeKind,
}

#[derive(Clone, Copy)]
struct BodyInfo {
    entity: Entity,
    shape: BodyShape,
    body_type: BodyType,
    inv_mass: f32,
    inv_inertia: f32,
    velocity: glam::Vec3,
    angular_velocity: f32,
    friction: f32,
    restitution: f32,
    lock_rotation: bool,
    sleeping: bool,
}

#[derive(Clone, Copy)]
struct Contact {
    a: Entity,
    b: Entity,
    normal: glam::Vec3,
    penetration: f32,
}

fn collision_state() -> &'static Mutex<HashSet<(u64, u64)>> {
    static STATE: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(HashSet::new()))
}

fn pair_key(a: Entity, b: Entity) -> (u64, u64) {
    let ab = a.to_bits().get();
    let bb = b.to_bits().get();
    if ab < bb { (ab, bb) } else { (bb, ab) }
}

fn inv_mass(body_type: BodyType, mass: f32) -> f32 {
    match body_type {
        BodyType::Static => 0.0,
        BodyType::Dynamic => {
            if mass <= 1e-6 { 1.0 } else { 1.0 / mass }
        }
        BodyType::Kinematic => 0.0,
    }
}

fn inv_inertia(body_type: BodyType, inertia: f32, lock_rotation: bool) -> f32 {
    if lock_rotation {
        return 0.0;
    }
    match body_type {
        BodyType::Static | BodyType::Kinematic => 0.0,
        BodyType::Dynamic => {
            if inertia <= 1e-6 { 1.0 } else { 1.0 / inertia }
        }
    }
}

fn box_axes(angle: f32) -> [glam::Vec2; 2] {
    let (s, c) = angle.sin_cos();
    [glam::Vec2::new(c, s), glam::Vec2::new(-s, c)]
}

fn obb_axes_3d(shape: &BodyShape) -> [glam::Vec3; 3] {
    [
        (shape.rotation * glam::Vec3::X).normalize_or_zero(),
        (shape.rotation * glam::Vec3::Y).normalize_or_zero(),
        (shape.rotation * glam::Vec3::Z).normalize_or_zero(),
    ]
}

fn box_vertices_3d(shape: &BodyShape) -> [glam::Vec3; 8] {
    let axes = obb_axes_3d(shape);
    let ex = axes[0] * shape.half.x;
    let ey = axes[1] * shape.half.y;
    let ez = axes[2] * shape.half.z;
    [
        shape.center - ex - ey - ez,
        shape.center + ex - ey - ez,
        shape.center + ex + ey - ez,
        shape.center - ex + ey - ez,
        shape.center - ex - ey + ez,
        shape.center + ex - ey + ez,
        shape.center + ex + ey + ez,
        shape.center - ex + ey + ez,
    ]
}

fn project_on_axis_3d(points: &[glam::Vec3; 8], axis: glam::Vec3) -> (f32, f32) {
    let mut min = points[0].dot(axis);
    let mut max = min;
    for p in points.iter().skip(1) {
        let d = p.dot(axis);
        min = min.min(d);
        max = max.max(d);
    }
    (min, max)
}

fn sat_contact_3d(a: &BodyShape, b: &BodyShape) -> Option<(glam::Vec3, f32)> {
    let va = box_vertices_3d(a);
    let vb = box_vertices_3d(b);
    let aa = obb_axes_3d(a);
    let ab = obb_axes_3d(b);
    let mut axes = Vec::with_capacity(15);
    axes.extend_from_slice(&aa);
    axes.extend_from_slice(&ab);
    for ax in aa {
        for bx in ab {
            let cross = ax.cross(bx);
            if cross.length_squared() > 1e-6 {
                axes.push(cross.normalize());
            }
        }
    }
    let mut min_overlap = f32::MAX;
    let mut best_axis = glam::Vec3::X;
    for axis in axes {
        if axis.length_squared() < 1e-6 {
            continue;
        }
        let (amin, amax) = project_on_axis_3d(&va, axis);
        let (bmin, bmax) = project_on_axis_3d(&vb, axis);
        let overlap = (amax.min(bmax)) - (amin.max(bmin));
        if overlap <= 0.0 {
            return None;
        }
        if overlap < min_overlap {
            min_overlap = overlap;
            best_axis = axis;
        }
    }
    let dir = (b.center - a.center).normalize_or_zero();
    let normal = if best_axis.dot(dir) < 0.0 { -best_axis } else { best_axis };
    Some((normal, min_overlap))
}

fn box_vertices_2d(shape: &BodyShape) -> [glam::Vec2; 4] {
    let center = shape.center.truncate();
    let axes = box_axes(shape.angle);
    let ux = axes[0] * shape.half.x;
    let uy = axes[1] * shape.half.y;
    [
        center - ux - uy,
        center + ux - uy,
        center + ux + uy,
        center - ux + uy,
    ]
}

fn project_on_axis(points: &[glam::Vec2; 4], axis: glam::Vec2) -> (f32, f32) {
    let mut min = points[0].dot(axis);
    let mut max = min;
    for p in points.iter().skip(1) {
        let d = p.dot(axis);
        min = min.min(d);
        max = max.max(d);
    }
    (min, max)
}

fn sat_contact_2d(a: &BodyShape, b: &BodyShape) -> Option<(glam::Vec3, f32)> {
    let z_overlap = (a.center.z + a.half.z).min(b.center.z + b.half.z)
        - (a.center.z - a.half.z).max(b.center.z - b.half.z);
    if z_overlap <= 0.0 {
        return None;
    }
    let va = box_vertices_2d(a);
    let vb = box_vertices_2d(b);
    let aa = box_axes(a.angle);
    let ab = box_axes(b.angle);
    let axes = [aa[0], aa[1], ab[0], ab[1]];
    let mut min_overlap = z_overlap;
    let mut best_axis = glam::Vec3::Z;
    for axis_raw in axes {
        let axis = axis_raw.normalize_or_zero();
        if axis.length_squared() < 1e-6 {
            continue;
        }
        let (amin, amax) = project_on_axis(&va, axis);
        let (bmin, bmax) = project_on_axis(&vb, axis);
        let overlap = (amax.min(bmax)) - (amin.max(bmin));
        if overlap <= 0.0 {
            return None;
        }
        if overlap < min_overlap {
            min_overlap = overlap;
            best_axis = glam::vec3(axis.x, axis.y, 0.0);
        }
    }
    let dir = (b.center - a.center).normalize_or_zero();
    let normal = if best_axis.dot(dir) < 0.0 { -best_axis } else { best_axis };
    Some((normal, min_overlap))
}

fn aabb_bounds(shape: &BodyShape) -> (glam::Vec3, glam::Vec3) {
    match shape.kind {
        ShapeKind::Box3D => (shape.center - shape.half, shape.center + shape.half),
        ShapeKind::Obb2D => {
            let verts = box_vertices_3d(shape);
            let mut min = verts[0];
            let mut max = verts[0];
            for v in verts.iter().skip(1) {
                min.x = min.x.min(v.x);
                min.y = min.y.min(v.y);
                min.z = min.z.min(v.z);
                max.x = max.x.max(v.x);
                max.y = max.y.max(v.y);
                max.z = max.z.max(v.z);
            }
            (min, max)
        }
    }
}

fn layer_allows(a: &BodyShape, b: &BodyShape) -> bool {
    (a.mask & b.layer) != 0 && (b.mask & a.layer) != 0
}

fn aabb_contact_3d(a: &BodyShape, b: &BodyShape) -> Option<(glam::Vec3, f32)> {
    let delta = b.center - a.center;
    let overlap_x = a.half.x + b.half.x - delta.x.abs();
    let overlap_y = a.half.y + b.half.y - delta.y.abs();
    let overlap_z = a.half.z + b.half.z - delta.z.abs();
    if overlap_x <= 0.0 || overlap_y <= 0.0 || overlap_z <= 0.0 {
        return None;
    }
    if overlap_x <= overlap_y && overlap_x <= overlap_z {
        Some((glam::vec3(delta.x.signum().max(1.0).copysign(delta.x), 0.0, 0.0).normalize_or_zero(), overlap_x))
    } else if overlap_y <= overlap_z {
        Some((glam::vec3(0.0, delta.y.signum().max(1.0).copysign(delta.y), 0.0).normalize_or_zero(), overlap_y))
    } else {
        Some((glam::vec3(0.0, 0.0, delta.z.signum().max(1.0).copysign(delta.z)).normalize_or_zero(), overlap_z))
    }
}

fn contact_between(a: &BodyShape, b: &BodyShape, runtime: &RuntimeSettings) -> Option<(glam::Vec3, f32)> {
    match (a.kind, b.kind) {
        (ShapeKind::Box3D, ShapeKind::Box3D) => aabb_contact_3d(a, b),
        _ => {
            if runtime.physics_3d_obb_contacts_enabled {
                sat_contact_3d(a, b).or_else(|| sat_contact_2d(a, b))
            } else {
                sat_contact_2d(a, b)
            }
        }
    }
}

fn collect_body_infos(world: &World) -> Vec<BodyInfo> {
    let mut bodies = Vec::new();
    for (e, pos, obb) in world.query::<(Entity, &Position, &OrientedBoxCollider)>().iter() {
        let rb = world.get::<&RigidBody>(e).ok();
        let rot = world.get::<&Rotation>(e).ok().map(|r| *r);
        let body_type = rb.as_ref().map(|b| b.body_type).unwrap_or(BodyType::Static);
        bodies.push(BodyInfo {
            entity: e,
            shape: BodyShape {
                center: glam::vec3(pos.x, pos.y, pos.z),
                half: glam::vec3(obb.half_w.max(0.01), obb.half_h.max(0.01), obb.half_d.max(0.01)),
                rotation: rot
                    .map(|r| glam::Quat::from_euler(glam::EulerRot::XYZ, r.pitch, r.yaw, r.roll))
                    .unwrap_or_else(|| glam::Quat::from_rotation_y(obb.angle_rad)),
                angle: obb.angle_rad,
                layer: obb.layer,
                mask: obb.mask,
                kind: ShapeKind::Obb2D,
            },
            body_type,
            inv_mass: rb.as_ref().map(|b| inv_mass(b.body_type, b.mass)).unwrap_or(0.0),
            inv_inertia: rb
                .as_ref()
                .map(|b| inv_inertia(b.body_type, b.inertia, b.lock_rotation))
                .unwrap_or(0.0),
            velocity: rb
                .as_ref()
                .map(|b| glam::vec3(b.velocity_x, b.velocity_y, b._velocity_z))
                .unwrap_or(glam::Vec3::ZERO),
            angular_velocity: rb.as_ref().map(|b| b.angular_velocity).unwrap_or(0.0),
            friction: rb.as_ref().map(|b| b.friction).unwrap_or(0.5),
            restitution: rb.as_ref().map(|b| b.restitution).unwrap_or(0.0),
            lock_rotation: rb.as_ref().map(|b| b.lock_rotation).unwrap_or(true),
            sleeping: rb.as_ref().map(|b| b.sleeping).unwrap_or(false),
        });
    }
    for (e, pos, col) in world.query::<(Entity, &Position, &Collider)>().iter() {
        if world.get::<&OrientedBoxCollider>(e).is_ok() {
            continue;
        }
        let rb = world.get::<&RigidBody>(e).ok();
        let body_type = rb.as_ref().map(|b| b.body_type).unwrap_or(BodyType::Static);
        bodies.push(BodyInfo {
            entity: e,
            shape: BodyShape {
                center: glam::vec3(pos.x, pos.y, pos.z),
                half: glam::vec3(col.half_w.max(0.01), col.half_h.max(0.01), col.half_d.max(0.01)),
                rotation: glam::Quat::IDENTITY,
                angle: 0.0,
                layer: col.layer,
                mask: col.mask,
                kind: ShapeKind::Box3D,
            },
            body_type,
            inv_mass: rb.as_ref().map(|b| inv_mass(b.body_type, b.mass)).unwrap_or(0.0),
            inv_inertia: rb
                .as_ref()
                .map(|b| inv_inertia(b.body_type, b.inertia, b.lock_rotation))
                .unwrap_or(0.0),
            velocity: rb
                .as_ref()
                .map(|b| glam::vec3(b.velocity_x, b.velocity_y, b._velocity_z))
                .unwrap_or(glam::Vec3::ZERO),
            angular_velocity: rb.as_ref().map(|b| b.angular_velocity).unwrap_or(0.0),
            friction: rb.as_ref().map(|b| b.friction).unwrap_or(0.5),
            restitution: rb.as_ref().map(|b| b.restitution).unwrap_or(0.0),
            lock_rotation: rb.as_ref().map(|b| b.lock_rotation).unwrap_or(true),
            sleeping: rb.as_ref().map(|b| b.sleeping).unwrap_or(false),
        });
    }
    bodies
}

fn broadphase_pairs(bodies: &[BodyInfo], settings: &RuntimeSettings) -> Vec<(usize, usize)> {
    if !settings.physics_broadphase_enabled {
        let mut pairs = Vec::new();
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                pairs.push((i, j));
            }
        }
        return pairs;
    }
    let cell = settings.physics_broadphase_cell_size.max(0.25);
    let mut buckets: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (idx, body) in bodies.iter().enumerate() {
        let vel_pad = body.velocity.abs() * 0.016;
        let (mut min, mut max) = aabb_bounds(&body.shape);
        min -= vel_pad;
        max += vel_pad;
        let min_x = (min.x / cell).floor() as i32;
        let min_y = (min.y / cell).floor() as i32;
        let min_z = (min.z / cell).floor() as i32;
        let max_x = (max.x / cell).floor() as i32;
        let max_y = (max.y / cell).floor() as i32;
        let max_z = (max.z / cell).floor() as i32;
        for gx in min_x..=max_x {
            for gy in min_y..=max_y {
                for gz in min_z..=max_z {
                    buckets.entry((gx, gy, gz)).or_default().push(idx);
                }
            }
        }
    }
    let mut unique = HashSet::new();
    let mut pairs = Vec::new();
    for indices in buckets.values() {
        for a in 0..indices.len() {
            for b in (a + 1)..indices.len() {
                let ia = indices[a];
                let ib = indices[b];
                let key = if ia < ib { (ia, ib) } else { (ib, ia) };
                if unique.insert(key) {
                    pairs.push(key);
                }
            }
        }
    }
    pairs
}

fn sync_body_info_from_world(world: &World, infos: &mut [BodyInfo]) {
    for info in infos {
        if let Ok(pos) = world.get::<&Position>(info.entity) {
            info.shape.center = glam::vec3(pos.x, pos.y, pos.z);
        }
        if let Ok(rot) = world.get::<&Rotation>(info.entity) {
            info.shape.rotation = glam::Quat::from_euler(glam::EulerRot::XYZ, rot.pitch, rot.yaw, rot.roll);
        } else {
            info.shape.rotation = glam::Quat::IDENTITY;
        }
        if let Ok(rb) = world.get::<&RigidBody>(info.entity) {
            info.velocity = glam::vec3(rb.velocity_x, rb.velocity_y, rb._velocity_z);
            info.body_type = rb.body_type;
            info.inv_mass = inv_mass(rb.body_type, rb.mass);
            info.inv_inertia = inv_inertia(rb.body_type, rb.inertia, rb.lock_rotation);
            info.friction = rb.friction;
            info.restitution = rb.restitution;
            info.angular_velocity = rb.angular_velocity;
            info.lock_rotation = rb.lock_rotation;
            info.sleeping = rb.sleeping;
        } else {
            info.body_type = BodyType::Static;
            info.inv_mass = 0.0;
            info.inv_inertia = 0.0;
            info.velocity = glam::Vec3::ZERO;
            info.angular_velocity = 0.0;
            info.lock_rotation = true;
            info.sleeping = false;
        }
        if let Ok(obb) = world.get::<&OrientedBoxCollider>(info.entity) {
            info.shape.angle = obb.angle_rad;
            info.shape.layer = obb.layer;
            info.shape.mask = obb.mask;
            info.shape.kind = ShapeKind::Obb2D;
            info.shape.half.x = obb.half_w;
            info.shape.half.y = obb.half_h;
            info.shape.half.z = obb.half_d;
        } else if let Ok(col) = world.get::<&Collider>(info.entity) {
            info.shape.angle = 0.0;
            info.shape.layer = col.layer;
            info.shape.mask = col.mask;
            info.shape.kind = ShapeKind::Box3D;
            info.shape.half = glam::vec3(col.half_w, col.half_h, col.half_d);
        }
    }
}

fn positional_correction(world: &mut World, contact: &Contact, infos: &[BodyInfo], settings: &RuntimeSettings) {
    if !settings.physics_position_correction_enabled {
        return;
    }
    let Some(a) = infos.iter().find(|i| i.entity == contact.a) else { return; };
    let Some(b) = infos.iter().find(|i| i.entity == contact.b) else { return; };
    let inv_sum = a.inv_mass + b.inv_mass;
    if inv_sum <= 1e-6 {
        return;
    }
    let correction_mag = ((contact.penetration - POSITION_SLOP).max(0.0) / inv_sum) * POSITION_PERCENT;
    let correction = contact.normal * correction_mag;
    if a.inv_mass > 0.0 {
        if let Ok(mut pos) = world.get::<&mut Position>(contact.a) {
            pos.x -= correction.x * a.inv_mass;
            pos.y -= correction.y * a.inv_mass;
            pos.z -= correction.z * a.inv_mass;
        }
    }
    if b.inv_mass > 0.0 {
        if let Ok(mut pos) = world.get::<&mut Position>(contact.b) {
            pos.x += correction.x * b.inv_mass;
            pos.y += correction.y * b.inv_mass;
            pos.z += correction.z * b.inv_mass;
        }
    }
}

fn resolve_velocity_contact(world: &mut World, contact: &Contact, infos: &[BodyInfo], settings: &RuntimeSettings) {
    let Some(a) = infos.iter().find(|i| i.entity == contact.a) else { return; };
    let Some(b) = infos.iter().find(|i| i.entity == contact.b) else { return; };
    let inv_sum = a.inv_mass + b.inv_mass;
    if inv_sum <= 1e-6 {
        return;
    }
    let rv = b.velocity - a.velocity;
    let vel_along_normal = rv.dot(contact.normal);
    if vel_along_normal > 0.0 {
        positional_correction(world, contact, infos, settings);
        return;
    }
    let restitution = a.restitution.min(b.restitution);
    let impulse_scalar = -(1.0 + restitution) * vel_along_normal / inv_sum;
    let impulse = contact.normal * impulse_scalar;
    let center_delta = b.shape.center - a.shape.center;
    let spin_impulse = glam::vec2(center_delta.x, center_delta.y).perp_dot(glam::vec2(impulse.x, impulse.y)) * 0.12;

    if a.inv_mass > 0.0 {
        if let Ok(mut rb) = world.get::<&mut RigidBody>(contact.a) {
            rb.velocity_x -= impulse.x * a.inv_mass;
            rb.velocity_y -= impulse.y * a.inv_mass;
            rb._velocity_z -= impulse.z * a.inv_mass;
            rb.sleeping = false;
            rb.sleep_timer = 0.0;
            if contact.normal.y > 0.55 {
                rb.on_ground = true;
            }
            if settings.physics_angular_dynamics_enabled && !rb.lock_rotation && a.inv_inertia > 0.0 {
                rb.angular_velocity -= spin_impulse * a.inv_inertia;
            }
        }
    }
    if b.inv_mass > 0.0 {
        if let Ok(mut rb) = world.get::<&mut RigidBody>(contact.b) {
            rb.velocity_x += impulse.x * b.inv_mass;
            rb.velocity_y += impulse.y * b.inv_mass;
            rb._velocity_z += impulse.z * b.inv_mass;
            rb.sleeping = false;
            rb.sleep_timer = 0.0;
            if contact.normal.y < -0.55 {
                rb.on_ground = true;
            }
            if settings.physics_angular_dynamics_enabled && !rb.lock_rotation && b.inv_inertia > 0.0 {
                rb.angular_velocity += spin_impulse * b.inv_inertia;
            }
        }
    }

    if settings.physics_friction_enabled {
        let rv2 = {
            let av = world
                .get::<&RigidBody>(contact.a)
                .ok()
                .map(|b| glam::vec3(b.velocity_x, b.velocity_y, b._velocity_z))
                .unwrap_or(glam::Vec3::ZERO);
            let bv = world
                .get::<&RigidBody>(contact.b)
                .ok()
                .map(|b| glam::vec3(b.velocity_x, b.velocity_y, b._velocity_z))
                .unwrap_or(glam::Vec3::ZERO);
            bv - av
        };
        let tangent = (rv2 - contact.normal * rv2.dot(contact.normal)).normalize_or_zero();
        if tangent.length_squared() > 1e-6 {
            let jt = -rv2.dot(tangent) / inv_sum;
            let mu = (a.friction * b.friction).sqrt();
            let friction_scalar = jt.clamp(-impulse_scalar * mu, impulse_scalar * mu);
            let friction_impulse = tangent * friction_scalar;
            if a.inv_mass > 0.0 {
                if let Ok(mut rb) = world.get::<&mut RigidBody>(contact.a) {
                    rb.velocity_x -= friction_impulse.x * a.inv_mass;
                    rb.velocity_y -= friction_impulse.y * a.inv_mass;
                    rb._velocity_z -= friction_impulse.z * a.inv_mass;
                }
            }
            if b.inv_mass > 0.0 {
                if let Ok(mut rb) = world.get::<&mut RigidBody>(contact.b) {
                    rb.velocity_x += friction_impulse.x * b.inv_mass;
                    rb.velocity_y += friction_impulse.y * b.inv_mass;
                    rb._velocity_z += friction_impulse.z * b.inv_mass;
                }
            }
        }
    }
    positional_correction(world, contact, infos, settings);
}

fn apply_positional_delta(world: &mut World, entity: Entity, delta: glam::Vec3) {
    if let Ok(mut pos) = world.get::<&mut Position>(entity) {
        pos.x += delta.x;
        pos.y += delta.y;
        pos.z += delta.z;
    }
}

fn apply_velocity_delta(world: &mut World, entity: Entity, delta: glam::Vec3) {
    if let Ok(mut rb) = world.get::<&mut RigidBody>(entity) {
        rb.velocity_x += delta.x;
        rb.velocity_y += delta.y;
        rb._velocity_z += delta.z;
        rb.sleeping = false;
        rb.sleep_timer = 0.0;
    }
}

fn rotate_local_anchor(anchor: [f32; 3], rot: Option<Rotation>) -> glam::Vec3 {
    if let Some(rot) = rot {
        glam::Quat::from_euler(glam::EulerRot::XYZ, rot.pitch, rot.yaw, rot.roll)
            * glam::vec3(anchor[0], anchor[1], anchor[2])
    } else {
        glam::vec3(anchor[0], anchor[1], anchor[2])
    }
}

fn anchored_positions(
    world: &World,
    info_a: &BodyInfo,
    info_b: &BodyInfo,
    anchor_a: [f32; 3],
    anchor_b: [f32; 3],
    use_local_anchors: bool,
) -> (glam::Vec3, glam::Vec3) {
    if use_local_anchors {
        let rot_a = world.get::<&Rotation>(info_a.entity).ok().map(|r| *r);
        let rot_b = world.get::<&Rotation>(info_b.entity).ok().map(|r| *r);
        (
            info_a.shape.center + rotate_local_anchor(anchor_a, rot_a),
            info_b.shape.center + rotate_local_anchor(anchor_b, rot_b),
        )
    } else {
        (info_a.shape.center, info_b.shape.center)
    }
}

fn apply_joint_distance(
    world: &mut World,
    a: Entity,
    b: Entity,
    target_distance: f32,
    stiffness: f32,
    infos: &[BodyInfo],
    anchor_a: [f32; 3],
    anchor_b: [f32; 3],
    use_local_anchors: bool,
) {
    let Some(ia) = infos.iter().find(|i| i.entity == a) else { return; };
    let Some(ib) = infos.iter().find(|i| i.entity == b) else { return; };
    let (anchored_a, anchored_b) = anchored_positions(world, ia, ib, anchor_a, anchor_b, use_local_anchors);
    let delta = anchored_b - anchored_a;
    let dist = delta.length();
    if dist <= 1e-5 {
        return;
    }
    let dir = delta / dist;
    let error = dist - target_distance;
    if error.abs() <= 1e-4 {
        return;
    }
    let inv_sum = ia.inv_mass + ib.inv_mass;
    if inv_sum <= 1e-6 {
        return;
    }
    let corr_mag = error * stiffness.clamp(0.0, 1.0) / inv_sum;
    let corr = dir * corr_mag;
    if ia.inv_mass > 0.0 {
        apply_positional_delta(world, a, corr * ia.inv_mass);
    }
    if ib.inv_mass > 0.0 {
        apply_positional_delta(world, b, -corr * ib.inv_mass);
    }
}

fn solve_constraints(world: &mut World, infos: &[BodyInfo], dt: f32, runtime: &RuntimeSettings) {
    if !runtime.physics_advanced_constraints_enabled {
        return;
    }
    let hinges: Vec<(Entity, HingeJoint)> = world.query::<(Entity, &HingeJoint)>().iter().map(|(e, j)| (e, *j)).collect();
    for (entity, joint) in hinges {
        apply_joint_distance(
            world,
            entity,
            joint.connected,
            joint.rest_length.max(0.0),
            joint.stiffness,
            infos,
            joint.anchor_a,
            joint.anchor_b,
            runtime.physics_local_anchor_constraints_enabled,
        );
    }

    let fixeds: Vec<(Entity, FixedJoint)> = world.query::<(Entity, &FixedJoint)>().iter().map(|(e, j)| (e, *j)).collect();
    for (entity, joint) in fixeds {
        let Some(info_a) = infos.iter().find(|i| i.entity == entity) else { continue; };
        let Some(info_b) = infos.iter().find(|i| i.entity == joint.connected) else { continue; };
        let (anchored_a, anchored_b) = anchored_positions(
            world,
            info_a,
            info_b,
            joint.anchor_a,
            joint.anchor_b,
            runtime.physics_local_anchor_constraints_enabled,
        );
        let desired_b = anchored_a + glam::vec3(joint.offset_x, joint.offset_y, 0.0);
        let delta = desired_b - anchored_b;
        if info_b.inv_mass > 0.0 {
            apply_positional_delta(world, joint.connected, delta * joint.stiffness.clamp(0.0, 1.0));
        }
    }

    let springs: Vec<(Entity, SpringJoint)> = world.query::<(Entity, &SpringJoint)>().iter().map(|(e, j)| (e, *j)).collect();
    for (entity, joint) in springs {
        let Some(ia) = infos.iter().find(|i| i.entity == entity) else { continue; };
        let Some(ib) = infos.iter().find(|i| i.entity == joint.connected) else { continue; };
        let (anchored_a, anchored_b) = anchored_positions(
            world,
            ia,
            ib,
            joint.anchor_a,
            joint.anchor_b,
            runtime.physics_local_anchor_constraints_enabled,
        );
        let delta = anchored_b - anchored_a;
        let dist = delta.length();
        if dist <= 1e-5 {
            continue;
        }
        let dir = delta / dist;
        let displacement = dist - joint.rest_length.max(0.0);
        let rel_vel = (ib.velocity - ia.velocity).dot(dir);
        let force_mag = displacement * joint.stiffness.max(0.0) - rel_vel * joint.damping.max(0.0);
        let impulse = dir * (force_mag * dt);
        if ia.inv_mass > 0.0 {
            apply_velocity_delta(world, entity, impulse * ia.inv_mass);
        }
        if ib.inv_mass > 0.0 {
            apply_velocity_delta(world, joint.connected, -impulse * ib.inv_mass);
        }
    }

    let ropes: Vec<(Entity, RopeConstraint)> = world.query::<(Entity, &RopeConstraint)>().iter().map(|(e, j)| (e, *j)).collect();
    for (entity, rope) in ropes {
        let Some(ia) = infos.iter().find(|i| i.entity == entity) else { continue; };
        let Some(ib) = infos.iter().find(|i| i.entity == rope.connected) else { continue; };
        let (anchored_a, anchored_b) = anchored_positions(
            world,
            ia,
            ib,
            rope.anchor_a,
            rope.anchor_b,
            runtime.physics_local_anchor_constraints_enabled,
        );
        let dist = (anchored_b - anchored_a).length();
        if dist <= rope.max_length.max(0.0) || dist <= 1e-5 {
            continue;
        }
        apply_joint_distance(
            world,
            entity,
            rope.connected,
            rope.max_length.max(0.0),
            rope.stiffness,
            infos,
            rope.anchor_a,
            rope.anchor_b,
            runtime.physics_local_anchor_constraints_enabled,
        );
    }
}

fn collision_events(contacts: &[Contact], settings: &RuntimeSettings) -> Vec<CollisionPair> {
    if !settings.physics_collision_events_enabled {
        return contacts
            .iter()
            .map(|c| CollisionPair {
                entity_a: c.a,
                entity_b: c.b,
                normal: [c.normal.x, c.normal.y, c.normal.z],
                penetration: c.penetration,
                phase: CollisionPhase::Ongoing,
            })
            .collect();
    }
    let mut guard = collision_state().lock().expect("collision state mutex poisoned");
    let previous = guard.clone();
    let mut current = HashSet::new();
    let mut events = Vec::new();
    for c in contacts {
        let key = pair_key(c.a, c.b);
        current.insert(key);
        events.push(CollisionPair {
            entity_a: c.a,
            entity_b: c.b,
            normal: [c.normal.x, c.normal.y, c.normal.z],
            penetration: c.penetration,
            phase: if previous.contains(&key) { CollisionPhase::Ongoing } else { CollisionPhase::Started },
        });
    }
    for ended in previous.difference(&current) {
        events.push(CollisionPair {
            entity_a: Entity::from_bits(ended.0).expect("valid entity bits"),
            entity_b: Entity::from_bits(ended.1).expect("valid entity bits"),
            normal: [0.0, 0.0, 0.0],
            penetration: 0.0,
            phase: CollisionPhase::Ended,
        });
    }
    *guard = current;
    events
}

pub fn physics_system(
    world: &mut World,
    dt: f32,
    time_s: f32,
    _jobs: &JobSystem,
    runtime: &RuntimeSettings,
) -> Vec<CollisionPair> {
    for body in world.query_mut::<&mut RigidBody>() {
        match body.body_type {
            BodyType::Static => {
                body.velocity_x = 0.0;
                body.velocity_y = 0.0;
                body._velocity_z = 0.0;
                body.angular_velocity = 0.0;
                body.torque = 0.0;
                body.on_ground = true;
                body.sleeping = true;
            }
            BodyType::Dynamic | BodyType::Kinematic => {
                if runtime.physics_sleeping_enabled && body.can_sleep && body.sleeping {
                    continue;
                }
                if body.body_type == BodyType::Dynamic && body.use_gravity && !body.on_ground {
                    body.velocity_y -= GRAVITY * dt;
                    body.velocity_y = body.velocity_y.max(-TERMINAL_VELOCITY);
                }
                let damping = (1.0 - body.linear_damping * dt).clamp(0.0, 1.0);
                body.velocity_x *= damping;
                body.velocity_y *= damping;
                body._velocity_z *= damping;
                if runtime.physics_angular_dynamics_enabled && body.body_type == BodyType::Dynamic && !body.lock_rotation {
                    let ang_accel = if body.inertia > 1e-6 { body.torque / body.inertia } else { 0.0 };
                    body.angular_velocity += ang_accel * dt;
                    let angular_damping = (1.0 - body.angular_damping * dt).clamp(0.0, 1.0);
                    body.angular_velocity *= angular_damping;
                } else {
                    body.angular_velocity = 0.0;
                }
                body.torque = 0.0;
            }
        }
    }

    for (pos, body) in world.query_mut::<(&mut Position, &RigidBody)>() {
        if matches!(body.body_type, BodyType::Static) || body.sleeping {
            continue;
        }
        pos.x += body.velocity_x * dt;
        pos.y += body.velocity_y * dt;
        pos.z += body._velocity_z * dt;
    }

    let angular_steps: Vec<(Entity, f32)> = world
        .query::<(Entity, &RigidBody)>()
        .iter()
        .filter_map(|(e, body)| {
            if matches!(body.body_type, BodyType::Static) || body.sleeping || body.lock_rotation {
                None
            } else {
                Some((e, body.angular_velocity * dt))
            }
        })
        .collect();
    for (entity, delta_yaw) in angular_steps {
        if let Ok(mut rot) = world.get::<&mut Rotation>(entity) {
            rot.yaw += delta_yaw;
        }
        if let Ok(mut obb) = world.get::<&mut OrientedBoxCollider>(entity) {
            obb.angle_rad += delta_yaw;
        }
    }

    if runtime.foliage_wind_enabled && runtime.physics_smooth_foliage_motion {
        for (pos, wind) in world.query_mut::<(&mut Position, &FoliageWind)>() {
            let sway = (time_s * wind.frequency + wind.base_x * 0.11 + wind.base_z * 0.07).sin();
            let sway2 = (time_s * wind.frequency * 0.7 + wind.base_z * 0.09).cos();
            let target_x = wind.base_x + sway * wind.amplitude;
            let target_z = wind.base_z + sway2 * wind.amplitude * 0.35;
            let blend = (dt * 9.0).clamp(0.0, 1.0);
            pos.x += (target_x - pos.x) * blend;
            pos.z += (target_z - pos.z) * blend;
        }
    } else if runtime.foliage_wind_enabled {
        for (pos, wind) in world.query_mut::<(&mut Position, &FoliageWind)>() {
            let sway = (time_s * wind.frequency + wind.base_x * 0.11 + wind.base_z * 0.07).sin();
            let sway2 = (time_s * wind.frequency * 0.7 + wind.base_z * 0.09).cos();
            pos.x = wind.base_x + sway * wind.amplitude;
            pos.z = wind.base_z + sway2 * wind.amplitude * 0.35;
        }
    }

    let mut infos = collect_body_infos(world);
    let pairs = broadphase_pairs(&infos, runtime);
    let mut contacts = Vec::new();
    for _ in 0..runtime.physics_solver_iterations.max(1) {
        sync_body_info_from_world(world, &mut infos);
        contacts.clear();
        for (ia, ib) in &pairs {
            let a = infos[*ia];
            let b = infos[*ib];
            if a.sleeping && b.sleeping {
                continue;
            }
            if a.inv_mass <= 1e-6 && b.inv_mass <= 1e-6 {
                continue;
            }
            if !layer_allows(&a.shape, &b.shape) {
                continue;
            }
            let Some((normal, penetration)) = contact_between(&a.shape, &b.shape, runtime) else { continue; };
            let contact = Contact { a: a.entity, b: b.entity, normal, penetration };
            resolve_velocity_contact(world, &contact, &infos, runtime);
            contacts.push(contact);
        }
        for _ in 0..runtime.physics_constraint_iterations.max(1) {
            sync_body_info_from_world(world, &mut infos);
            solve_constraints(world, &infos, dt, runtime);
        }
    }

    for body in world.query_mut::<&mut RigidBody>() {
        if !runtime.physics_sleeping_enabled || !body.can_sleep || body.body_type != BodyType::Dynamic {
            continue;
        }
        let speed_sq = body.velocity_x * body.velocity_x
            + body.velocity_y * body.velocity_y
            + body._velocity_z * body._velocity_z;
        let ang_speed = body.angular_velocity.abs();
        if speed_sq <= SLEEP_LINEAR_THRESHOLD * SLEEP_LINEAR_THRESHOLD && ang_speed <= 0.02 {
            body.sleep_timer += dt;
            if body.sleep_timer >= SLEEP_TIME_THRESHOLD {
                body.sleeping = true;
                body.velocity_x = 0.0;
                body.velocity_y = 0.0;
                body._velocity_z = 0.0;
                body.angular_velocity = 0.0;
            }
        } else {
            body.sleep_timer = 0.0;
            body.sleeping = false;
        }
    }
    collision_events(&contacts, runtime)
}
