use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use crate::components::{
    BodyType, CapsuleCollider, CharacterController, Collider, CollisionPair, CollisionPhase,
    FixedJoint, FoliageWind, HingeJoint, OrientedBoxCollider, Position, Ragdoll, RagdollBone,
    RigidBody, RopeConstraint, Rotation, SphereCollider, SpringJoint,
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShapeKind {
    Box3D,
    Obb2D,
    Sphere,
    Capsule,
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
    /// Radius for Sphere or Capsule shapes.
    radius: f32,
    /// Half-height of the cylinder part for Capsule shapes (along Y axis).
    capsule_half_height: f32,
    /// If true, this collider generates trigger events but no velocity response.
    is_trigger: bool,
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

// ── Sphere vs Sphere ─────────────────────────────────────────────────────────
// Two spheres overlap when distance between centers < sum of radii.
// The contact normal is the direction between centers; penetration = sum_radii - distance.
fn sphere_vs_sphere(a: &BodyShape, b: &BodyShape) -> Option<(glam::Vec3, f32)> {
    let delta = b.center - a.center;
    let dist_sq = delta.length_squared();
    let sum_r = a.radius + b.radius;
    if dist_sq >= sum_r * sum_r {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 { delta / dist } else { glam::Vec3::Y };
    let penetration = sum_r - dist;
    Some((normal, penetration))
}

// ── Sphere vs AABB ───────────────────────────────────────────────────────────
// Clamp sphere center to AABB surface; distance from clamped point to center
// is the overlap amount if less than radius.
fn sphere_vs_aabb(sphere: &BodyShape, aabb: &BodyShape) -> Option<(glam::Vec3, f32)> {
    let closest = glam::vec3(
        sphere.center.x.clamp(aabb.center.x - aabb.half.x, aabb.center.x + aabb.half.x),
        sphere.center.y.clamp(aabb.center.y - aabb.half.y, aabb.center.y + aabb.half.y),
        sphere.center.z.clamp(aabb.center.z - aabb.half.z, aabb.center.z + aabb.half.z),
    );
    let delta = sphere.center - closest;
    let dist_sq = delta.length_squared();
    if dist_sq >= sphere.radius * sphere.radius {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 { delta / dist } else { glam::Vec3::Y };
    let penetration = sphere.radius - dist;
    Some((normal, penetration))
}

// ── Sphere vs OBB ────────────────────────────────────────────────────────────
// Transform sphere center into OBB local space, clamp to box, then treat as sphere vs AABB.
fn sphere_vs_obb(sphere: &BodyShape, obb: &BodyShape) -> Option<(glam::Vec3, f32)> {
    let axes = obb_axes_3d(obb);
    let local = sphere.center - obb.center;
    let mut closest_local = glam::Vec3::ZERO;
    for (i, axis) in axes.iter().enumerate() {
        let half = match i {
            0 => obb.half.x,
            1 => obb.half.y,
            _ => obb.half.z,
        };
        let proj = local.dot(*axis).clamp(-half, half);
        closest_local += *axis * proj;
    }
    let closest_world = obb.center + closest_local;
    let delta = sphere.center - closest_world;
    let dist_sq = delta.length_squared();
    if dist_sq >= sphere.radius * sphere.radius {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 { delta / dist } else { glam::Vec3::Y };
    let penetration = sphere.radius - dist;
    Some((normal, penetration))
}

// ── Capsule vs Capsule ───────────────────────────────────────────────────────
// A capsule is a swept sphere along a line segment (center +/- half_height * Y).
// Find closest points on two line segments, then test as sphere vs sphere.
fn capsule_vs_capsule(a: &BodyShape, b: &BodyShape) -> Option<(glam::Vec3, f32)> {
    let a_top = a.center + glam::Vec3::Y * a.capsule_half_height;
    let a_bot = a.center - glam::Vec3::Y * a.capsule_half_height;
    let b_top = b.center + glam::Vec3::Y * b.capsule_half_height;
    let b_bot = b.center - glam::Vec3::Y * b.capsule_half_height;
    let (closest_a, closest_b) = closest_points_on_segments(a_bot, a_top, b_bot, b_top);
    let delta = closest_b - closest_a;
    let dist_sq = delta.length_squared();
    let sum_r = a.radius + b.radius;
    if dist_sq >= sum_r * sum_r {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 { delta / dist } else { glam::Vec3::Y };
    let penetration = sum_r - dist;
    Some((normal, penetration))
}

// ── Capsule vs Sphere ────────────────────────────────────────────────────────
// Clamp sphere center to capsule segment, then test as sphere vs sphere.
fn capsule_vs_sphere(capsule: &BodyShape, sphere: &BodyShape) -> Option<(glam::Vec3, f32)> {
    let top = capsule.center + glam::Vec3::Y * capsule.capsule_half_height;
    let bot = capsule.center - glam::Vec3::Y * capsule.capsule_half_height;
    let closest = closest_point_on_segment(sphere.center, bot, top);
    let delta = sphere.center - closest;
    let dist_sq = delta.length_squared();
    let sum_r = capsule.radius + sphere.radius;
    if dist_sq >= sum_r * sum_r {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 { delta / dist } else { glam::Vec3::Y };
    let penetration = sum_r - dist;
    Some((normal, penetration))
}

// ── Capsule vs AABB ──────────────────────────────────────────────────────────
// Test each cap-sphere against the AABB, then test the midline cylinder.
fn capsule_vs_aabb(capsule: &BodyShape, aabb: &BodyShape) -> Option<(glam::Vec3, f32)> {
    let top = capsule.center + glam::Vec3::Y * capsule.capsule_half_height;
    let bot = capsule.center - glam::Vec3::Y * capsule.capsule_half_height;
    let mut min_pen = f32::MAX;
    let mut best_normal = glam::Vec3::Y;
    for sphere_center in [bot, top] {
        let closest = glam::vec3(
            sphere_center.x.clamp(aabb.center.x - aabb.half.x, aabb.center.x + aabb.half.x),
            sphere_center.y.clamp(aabb.center.y - aabb.half.y, aabb.center.y + aabb.half.y),
            sphere_center.z.clamp(aabb.center.z - aabb.half.z, aabb.center.z + aabb.half.z),
        );
        let delta = sphere_center - closest;
        let dist_sq = delta.length_squared();
        if dist_sq >= capsule.radius * capsule.radius {
            continue;
        }
        let dist = dist_sq.sqrt();
        let pen = capsule.radius - dist;
        if pen < min_pen {
            min_pen = pen;
            best_normal = if dist > 1e-6 { delta / dist } else { glam::Vec3::Y };
        }
    }
    let center_clamped = glam::vec3(
        capsule.center.x.clamp(aabb.center.x - aabb.half.x, aabb.center.x + aabb.half.x),
        capsule.center.y.clamp(aabb.center.y - aabb.half.y, aabb.center.y + aabb.half.y),
        capsule.center.z.clamp(aabb.center.z - aabb.half.z, aabb.center.z + aabb.half.z),
    );
    let delta_mid = capsule.center - center_clamped;
    let dist_mid_sq = delta_mid.length_squared();
    if dist_mid_sq < capsule.radius * capsule.radius {
        let dist_mid = dist_mid_sq.sqrt();
        let pen_mid = capsule.radius - dist_mid;
        if pen_mid < min_pen {
            min_pen = pen_mid;
            best_normal = if dist_mid > 1e-6 { delta_mid / dist_mid } else { glam::Vec3::Y };
        }
    }
    if min_pen < f32::MAX {
        Some((best_normal, min_pen))
    } else {
        None
    }
}

// ── Capsule vs OBB ───────────────────────────────────────────────────────────
// Transform capsule into OBB local space, test as capsule vs AABB there, then
// transform the contact normal back to world space.
fn capsule_vs_obb(capsule: &BodyShape, obb: &BodyShape) -> Option<(glam::Vec3, f32)> {
    let inv_rot = obb.rotation.inverse();
    let local_center = inv_rot * (capsule.center - obb.center);
    let local_top = local_center + glam::Vec3::Y * capsule.capsule_half_height;
    let local_bot = local_center - glam::Vec3::Y * capsule.capsule_half_height;
    let aabb_half = obb.half;
    let mut min_pen = f32::MAX;
    let mut best_normal_local = glam::Vec3::Y;
    for sc in [local_bot, local_top] {
        let closest = glam::vec3(
            sc.x.clamp(-aabb_half.x, aabb_half.x),
            sc.y.clamp(-aabb_half.y, aabb_half.y),
            sc.z.clamp(-aabb_half.z, aabb_half.z),
        );
        let delta = sc - closest;
        let dist_sq = delta.length_squared();
        if dist_sq >= capsule.radius * capsule.radius {
            continue;
        }
        let dist = dist_sq.sqrt();
        let pen = capsule.radius - dist;
        if pen < min_pen {
            min_pen = pen;
            best_normal_local = if dist > 1e-6 { delta / dist } else { glam::Vec3::Y };
        }
    }
    let center_clamped = glam::vec3(
        local_center.x.clamp(-aabb_half.x, aabb_half.x),
        local_center.y.clamp(-aabb_half.y, aabb_half.y),
        local_center.z.clamp(-aabb_half.z, aabb_half.z),
    );
    let delta_mid = local_center - center_clamped;
    let dist_mid_sq = delta_mid.length_squared();
    if dist_mid_sq < capsule.radius * capsule.radius {
        let dist_mid = dist_mid_sq.sqrt();
        let pen_mid = capsule.radius - dist_mid;
        if pen_mid < min_pen {
            min_pen = pen_mid;
            best_normal_local = if dist_mid > 1e-6 { delta_mid / dist_mid } else { glam::Vec3::Y };
        }
    }
    if min_pen < f32::MAX {
        let world_normal = obb.rotation * best_normal_local;
        Some((world_normal.normalize_or_zero(), min_pen))
    } else {
        None
    }
}

// ── Segment-segment closest points ───────────────────────────────────────────
// Returns the closest point on each of two line segments.
fn closest_points_on_segments(
    a0: glam::Vec3, a1: glam::Vec3,
    b0: glam::Vec3, b1: glam::Vec3,
) -> (glam::Vec3, glam::Vec3) {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let mut s = 0.0f32;
    let mut t = 0.0f32;
    if a <= 1e-6 && e <= 1e-6 {
        s = 0.0;
        t = 0.0;
    } else if a <= 1e-6 {
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= 1e-6 {
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b_val = d1.dot(d2);
            let denom = a * e - b_val * b_val;
            if denom.abs() > 1e-6 {
                s = ((b_val * f - c * e) / denom).clamp(0.0, 1.0);
            }
            t = (b_val * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b_val - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    (a0 + d1 * s, b0 + d2 * t)
}

// Closest point on a segment to a given point.
fn closest_point_on_segment(point: glam::Vec3, seg_a: glam::Vec3, seg_b: glam::Vec3) -> glam::Vec3 {
    let d = seg_b - seg_a;
    let len_sq = d.length_squared();
    if len_sq < 1e-6 {
        return seg_a;
    }
    let t = ((point - seg_a).dot(d) / len_sq).clamp(0.0, 1.0);
    seg_a + d * t
}

// ── CCD: Swept sphere vs AABB ────────────────────────────────────────────────
// Move a sphere along a velocity vector and return the time of impact (0..1)
// against a stationary AABB. None if no hit within the timestep.
fn swept_sphere_aabb(
    sphere_center: glam::Vec3,
    sphere_radius: f32,
    velocity: glam::Vec3,
    aabb_min: glam::Vec3,
    aabb_max: glam::Vec3,
) -> Option<f32> {
    let inv_dir = glam::vec3(
        if velocity.x.abs() > 1e-8 { 1.0 / velocity.x } else { f32::INFINITY * velocity.x.signum() },
        if velocity.y.abs() > 1e-8 { 1.0 / velocity.y } else { f32::INFINITY * velocity.y.signum() },
        if velocity.z.abs() > 1e-8 { 1.0 / velocity.z } else { f32::INFINITY * velocity.z.signum() },
    );
    let expanded_min = aabb_min - glam::Vec3::splat(sphere_radius);
    let expanded_max = aabb_max + glam::Vec3::splat(sphere_radius);
    let t1 = (expanded_min - sphere_center) * inv_dir;
    let t2 = (expanded_max - sphere_center) * inv_dir;
    let t_near = t1.max_element().max(0.0);
    let t_far = t2.min_element();
    if t_near <= t_far && t_far >= 0.0 && t_near <= 1.0 {
        Some(t_near.min(1.0))
    } else {
        None
    }
}

// ── CCD: Swept sphere vs Sphere ──────────────────────────────────────────────
// Move a sphere along velocity against a stationary sphere.
fn swept_sphere_sphere(
    center_a: glam::Vec3,
    radius_a: f32,
    velocity: glam::Vec3,
    center_b: glam::Vec3,
    radius_b: f32,
) -> Option<f32> {
    let oc = center_a - center_b;
    let sum_r = radius_a + radius_b;
    let a = velocity.dot(velocity);
    let b = 2.0 * oc.dot(velocity);
    let c = oc.dot(oc) - sum_r * sum_r;
    if a < 1e-8 {
        if c < 0.0 {
            return Some(0.0);
        }
        return None;
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    let t = if t1 >= 0.0 { t1 } else if t2 >= 0.0 { t2 } else { return None; };
    if t <= 1.0 { Some(t) } else { None }
}

fn aabb_bounds(shape: &BodyShape) -> (glam::Vec3, glam::Vec3) {
    match shape.kind {
        ShapeKind::Box3D | ShapeKind::Sphere | ShapeKind::Capsule => {
            (shape.center - shape.half, shape.center + shape.half)
        }
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
        (ShapeKind::Sphere, ShapeKind::Sphere) => sphere_vs_sphere(a, b),
        (ShapeKind::Sphere, ShapeKind::Box3D) => sphere_vs_aabb(a, b),
        (ShapeKind::Box3D, ShapeKind::Sphere) => sphere_vs_aabb(b, a).map(|(n, p)| (-n, p)),
        (ShapeKind::Sphere, ShapeKind::Obb2D) => sphere_vs_obb(a, b),
        (ShapeKind::Obb2D, ShapeKind::Sphere) => sphere_vs_obb(b, a).map(|(n, p)| (-n, p)),
        (ShapeKind::Capsule, ShapeKind::Capsule) => capsule_vs_capsule(a, b),
        (ShapeKind::Capsule, ShapeKind::Sphere) => capsule_vs_sphere(a, b),
        (ShapeKind::Sphere, ShapeKind::Capsule) => capsule_vs_sphere(b, a).map(|(n, p)| (-n, p)),
        (ShapeKind::Capsule, ShapeKind::Box3D) => capsule_vs_aabb(a, b),
        (ShapeKind::Box3D, ShapeKind::Capsule) => capsule_vs_aabb(b, a).map(|(n, p)| (-n, p)),
        (ShapeKind::Capsule, ShapeKind::Obb2D) => capsule_vs_obb(a, b),
        (ShapeKind::Obb2D, ShapeKind::Capsule) => capsule_vs_obb(b, a).map(|(n, p)| (-n, p)),
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
    // Track entities already processed so we don't double-count.
    let mut seen: HashSet<Entity> = HashSet::new();
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
                radius: 0.0,
                capsule_half_height: 0.0,
                is_trigger: false,
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
        seen.insert(e);
    }
    for (e, pos, col) in world.query::<(Entity, &Position, &Collider)>().iter() {
        if seen.contains(&e) {
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
                radius: 0.0,
                capsule_half_height: 0.0,
                is_trigger: false,
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
        seen.insert(e);
    }
    // SphereColliders — sphere primitive.
    for (e, pos, sph) in world.query::<(Entity, &Position, &SphereCollider)>().iter() {
        if seen.contains(&e) {
            continue;
        }
        let rb = world.get::<&RigidBody>(e).ok();
        let body_type = rb.as_ref().map(|b| b.body_type).unwrap_or(BodyType::Static);
        bodies.push(BodyInfo {
            entity: e,
            shape: BodyShape {
                center: glam::vec3(pos.x, pos.y, pos.z),
                half: glam::vec3(sph.radius, sph.radius, sph.radius),
                rotation: glam::Quat::IDENTITY,
                angle: 0.0,
                layer: sph.layer,
                mask: sph.mask,
                kind: ShapeKind::Sphere,
                radius: sph.radius.max(0.01),
                capsule_half_height: 0.0,
                is_trigger: sph.is_trigger,
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
        seen.insert(e);
    }
    // CapsuleColliders — cylinder with hemispherical caps along Y axis.
    for (e, pos, cap) in world.query::<(Entity, &Position, &CapsuleCollider)>().iter() {
        if seen.contains(&e) {
            continue;
        }
        let rb = world.get::<&RigidBody>(e).ok();
        let body_type = rb.as_ref().map(|b| b.body_type).unwrap_or(BodyType::Static);
        let total_hh = cap.half_height + cap.radius;
        bodies.push(BodyInfo {
            entity: e,
            shape: BodyShape {
                center: glam::vec3(pos.x, pos.y, pos.z),
                half: glam::vec3(cap.radius, total_hh, cap.radius),
                rotation: glam::Quat::IDENTITY,
                angle: 0.0,
                layer: cap.layer,
                mask: cap.mask,
                kind: ShapeKind::Capsule,
                radius: cap.radius.max(0.01),
                capsule_half_height: cap.half_height.max(0.0),
                is_trigger: cap.is_trigger,
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
        seen.insert(e);
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
            info.shape.is_trigger = false;
        } else if let Ok(sph) = world.get::<&SphereCollider>(info.entity) {
            info.shape.layer = sph.layer;
            info.shape.mask = sph.mask;
            info.shape.kind = ShapeKind::Sphere;
            info.shape.radius = sph.radius;
            info.shape.half = glam::Vec3::splat(sph.radius);
            info.shape.is_trigger = sph.is_trigger;
        } else if let Ok(cap) = world.get::<&CapsuleCollider>(info.entity) {
            info.shape.layer = cap.layer;
            info.shape.mask = cap.mask;
            info.shape.kind = ShapeKind::Capsule;
            info.shape.radius = cap.radius;
            info.shape.capsule_half_height = cap.half_height;
            let total_hh = cap.half_height + cap.radius;
            info.shape.half = glam::vec3(cap.radius, total_hh, cap.radius);
            info.shape.is_trigger = cap.is_trigger;
        } else if let Ok(col) = world.get::<&Collider>(info.entity) {
            info.shape.angle = 0.0;
            info.shape.layer = col.layer;
            info.shape.mask = col.mask;
            info.shape.kind = ShapeKind::Box3D;
            info.shape.half = glam::vec3(col.half_w, col.half_h, col.half_d);
            info.shape.is_trigger = false;
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

// ── Character Controller ─────────────────────────────────────────────────────
// Processes CharacterController components: applies gravity (scaled), handles jump,
// moves the entity based on input velocity, does ground detection via downward sphere
// cast, slope limiting, step climbing, and depenetration via capsule/sphere shape.
pub fn character_controller_system(world: &mut World, dt: f32) {
    // Collect entities with CharacterController + RigidBody + Position so we can
    // borrow the world mutably per-entity without conflicting borrows.
    let controllers: Vec<(Entity, f32, f32, f32, bool, f32)> = world
        .query::<(Entity, &mut CharacterController, &RigidBody, &Position)>()
        .iter()
        .map(|(e, cc, rb, pos)| {
            // Extract what we need — we can't hold borrows across the mutable loop below.
            let input_vx = rb.velocity_x;
            let input_vz = rb._velocity_z;
            let jump = cc.jump_pressed;
            let speed = cc.speed;
            (e, input_vx, input_vz, speed, jump, pos.y)
        })
        .collect();

    for (e, input_vx, input_vz, speed, _jump, _pos_y) in &controllers {
        let e = *e;
        // Read controller params (immutable borrow, short-lived).
        let (max_slope, _step_h, _skin_w, jump_force, ground_dist, gravity_scale, was_jump) = {
            if let Ok(cc) = world.get::<&CharacterController>(e) {
                (cc.max_slope_angle, cc.step_height, cc.skin_width, cc.jump_force,
                 cc.ground_detect_dist, cc.gravity_scale, cc.jump_pressed)
            } else {
                continue;
            }
        };

        // Ground detection: cast downward from entity center.
        let ground_hit = {
            let pos = world.get::<&Position>(e).ok();
            let sph = world.get::<&SphereCollider>(e).ok();
            let cap = world.get::<&CapsuleCollider>(e).ok();
            let col = world.get::<&Collider>(e).ok();
            let center = pos.map(|p| glam::vec3(p.x, p.y, p.z)).unwrap_or(glam::Vec3::ZERO);
            let shape_radius = sph.map(|s| s.radius)
                .or_else(|| cap.map(|c| c.radius))
                .unwrap_or_else(|| col.map(|c| c.half_w.max(c.half_d)).unwrap_or(0.3));
            // Simple ground check: sphere cast downward.
            cast_downward_for_ground(world, e, center, shape_radius, ground_dist, max_slope)
        };

        // Update ground state and apply gravity + jump.
        if let Ok(mut cc) = world.get::<&mut CharacterController>(e) {
            cc.on_ground = ground_hit;
            if was_jump && ground_hit {
                if let Ok(mut rb) = world.get::<&mut RigidBody>(e) {
                    rb.velocity_y = jump_force;
                    rb.on_ground = false;
                    rb.sleeping = false;
                    rb.sleep_timer = 0.0;
                }
            }
        }

        // Apply gravity (scaled).
        if let Ok(mut rb) = world.get::<&mut RigidBody>(e) {
            if rb.use_gravity && !ground_hit {
                rb.velocity_y -= GRAVITY * gravity_scale * dt;
                rb.velocity_y = rb.velocity_y.max(-TERMINAL_VELOCITY);
            } else if ground_hit && rb.velocity_y < 0.0 {
                rb.velocity_y = 0.0;
            }
            // Horizontal movement from input.
            rb.velocity_x = input_vx * speed;
            rb._velocity_z = input_vz * speed;
            rb.sleeping = false;
            rb.sleep_timer = 0.0;
        }
    }
}

/// Cast a sphere downward from `center` to detect ground within `max_dist`.
/// Returns true if ground is found with slope angle < `max_slope`.
fn cast_downward_for_ground(
    world: &World,
    self_entity: Entity,
    center: glam::Vec3,
    shape_radius: f32,
    max_dist: f32,
    _max_slope: f32,
) -> bool {
    let cast_dir = glam::Vec3::NEG_Y;
    let cast_len = max_dist + shape_radius;
    for (e, pos, obb) in world.query::<(Entity, &Position, &OrientedBoxCollider)>().iter() {
        if e == self_entity { continue; }
        let other_center = glam::vec3(pos.x, pos.y, pos.z);
        let other_half = glam::vec3(obb.half_w, obb.half_h, obb.half_d);
        let aabb_min = other_center - other_half;
        let aabb_max = other_center + other_half;
        if let Some(_t) = swept_sphere_aabb(center, shape_radius, cast_dir * cast_len, aabb_min, aabb_max) {
            return true;
        }
    }
    for (e, pos, col) in world.query::<(Entity, &Position, &Collider)>().iter() {
        if e == self_entity { continue; }
        let other_center = glam::vec3(pos.x, pos.y, pos.z);
        let other_half = glam::vec3(col.half_w, col.half_h, col.half_d);
        let aabb_min = other_center - other_half;
        let aabb_max = other_center + other_half;
        if let Some(_t) = swept_sphere_aabb(center, shape_radius, cast_dir * cast_len, aabb_min, aabb_max) {
            return true;
        }
    }
    // Check sphere colliders as ground.
    for (e, pos, sph) in world.query::<(Entity, &Position, &SphereCollider)>().iter() {
        if e == self_entity { continue; }
        let other_center = glam::vec3(pos.x, pos.y, pos.z);
        if let Some(_t) = swept_sphere_sphere(center, shape_radius, cast_dir * cast_len, other_center, sph.radius) {
            return true;
        }
    }
    false
}

// ── Ragdoll System ───────────────────────────────────────────────────────────
// Drives ragdoll bones with ball-socket joint constraints.
// Each bone is pulled toward its parent with distance + cone limits.
pub fn ragdoll_system(world: &mut World, _dt: f32) {
    let ragdolls: Vec<(Entity, Vec<RagdollBone>)> = world
        .query::<(Entity, &Ragdoll)>()
        .iter()
        .map(|(e, rag)| (e, rag.bones.clone()))
        .collect();

    for (_root, bones) in &ragdolls {
        // Solve each bone's joint constraint against its parent.
        for (_idx, bone) in bones.iter().enumerate() {
            if bone.parent_index < 0 {
                continue; // Root bone — no parent constraint.
            }
            let parent_idx = bone.parent_index as usize;
            if parent_idx >= bones.len() {
                continue;
            }
            let parent = &bones[parent_idx];

            // Get world positions of both bones.
            let parent_pos = world.get::<&Position>(parent.entity).ok()
                .map(|p| glam::vec3(p.x, p.y, p.z))
                .unwrap_or(glam::Vec3::ZERO);
            let bone_pos = world.get::<&Position>(bone.entity).ok()
                .map(|p| glam::vec3(p.x, p.y, p.z))
                .unwrap_or(glam::Vec3::ZERO);

            // Desired position: parent center + rotated local offset.
            let parent_rot = world.get::<&Rotation>(parent.entity).ok()
                .map(|r| glam::Quat::from_euler(glam::EulerRot::XYZ, r.pitch, r.yaw, r.roll))
                .unwrap_or(glam::Quat::IDENTITY);
            let desired_offset = parent_rot * glam::vec3(bone.local_offset[0], bone.local_offset[1], bone.local_offset[2]);
            let desired_pos = parent_pos + desired_offset;

            // Compute correction vector.
            let delta = desired_pos - bone_pos;
            let dist = delta.length();
            if dist < 1e-4 {
                continue;
            }

            // Swing limit: if the bone drifts too far from its parent's reach,
            // dampen the correction. The swing_limit field controls the maximum
            // cone angle; we use it to scale the max reach distance.
            let base_reach = (bone.local_offset[0] * bone.local_offset[0]
                + bone.local_offset[1] * bone.local_offset[1]
                + bone.local_offset[2] * bone.local_offset[2]).sqrt();
            let max_reach = base_reach * (1.0 + bone.swing_limit.clamp(0.0, 1.57));
            let correction = if dist > max_reach {
                let dir = delta / dist;
                dir * (dist - max_reach) * 0.8 // Spring back with damping
            } else {
                delta * 0.5 // Soft follow
            };

            // Apply correction with per-bone damping.
            let damping = bone.damping.clamp(0.01, 1.0);
            let final_correction = correction * damping;

            if let Ok(mut pos) = world.get::<&mut Position>(bone.entity) {
                pos.x += final_correction.x;
                pos.y += final_correction.y;
                pos.z += final_correction.z;
            }
        }
    }
}

/// Check for water trigger collisions. Returns list of splash events.
pub fn water_trigger_system(world: &mut hecs::World) -> Vec<crate::core::events::WaterSplashEvent> {
    use crate::components::{Position, SphereCollider, RigidBody, WaterTrigger};

    let mut events = Vec::new();

    // Collect water entities first (immutable borrow).
    let water_entities: Vec<(hecs::Entity, Position, SphereCollider, WaterTrigger)> = world
        .query::<(hecs::Entity, &Position, &SphereCollider, &WaterTrigger)>()
        .iter()
        .filter(|(_, _, _, wt)| wt.active)
        .map(|(e, p, c, wt)| (e, *p, *c, *wt))
        .collect();

    // Check dynamic entities against water triggers.
    let dynamic_entities: Vec<(hecs::Entity, Position, SphereCollider, RigidBody)> = world
        .query::<(hecs::Entity, &Position, &SphereCollider, &RigidBody)>()
        .iter()
        .map(|(e, p, c, rb)| (e, *p, *c, *rb))
        .collect();

    for (dyn_entity, dyn_pos, dyn_col, dyn_rb) in &dynamic_entities {
        // Skip static bodies.
        if dyn_rb.sleeping { continue; }

        for (water_entity, water_pos, water_col, water_wt) in &water_entities {
            // Don't self-collide.
            if dyn_entity == water_entity { continue; }

            // Sphere-sphere overlap test.
            let dx = dyn_pos.x - water_pos.x;
            let dy = dyn_pos.y - water_pos.y;
            let dz = dyn_pos.z - water_pos.z;
            let dist_sq = dx*dx + dy*dy + dz*dz;
            let combined_radius = dyn_col.radius + water_col.radius;

            if dist_sq < combined_radius * combined_radius {
                // Calculate impact velocity (downward speed).
                let impact_velocity = -dyn_rb.velocity_y;

                // Only splash if entity is moving downward (into water).
                if impact_velocity > 0.5 {
                    let water_bits = water_entity.to_bits().get();
                    let dyn_bits = dyn_entity.to_bits().get();

                    tracing::debug!(
                        "[Water] Entity {:?} entered water {:?} at velocity {:.1}",
                        dyn_entity, water_entity, impact_velocity
                    );

                    events.push(crate::core::events::WaterSplashEvent {
                        entity_bits: dyn_bits,
                        water_entity_bits: water_bits,
                        impact_velocity,
                        splash_intensity: water_wt.splash_intensity,
                    });
                }
            }
        }
    }

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
            // Trigger colliders generate events but skip velocity/position resolution.
            let is_trigger_pair = a.shape.is_trigger || b.shape.is_trigger;
            if !is_trigger_pair {
                resolve_velocity_contact(world, &contact, &infos, runtime);
            }
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
