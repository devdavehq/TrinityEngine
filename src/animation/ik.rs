// src/animation/ik.rs
// ──────────────────────────────────────────────────────────────────────────────
// Two-Bone IK (Inverse Kinematics)
//
// WHY IT EXISTS:
//   Pre-baked animation can't react to the world. "The character raises his arm"
//   is a fixed clip. IK lets a hand *reach* a doorknob, a foot land on an uneven
//   step, or a character grab a weapon that physically moved. That reactivity is
//   what separates rigid bots from living creatures.
//
// HOW IT WORKS
//   Given three bone joints (root → mid → end) in a pose, two-bone IK solves the
//   shoulder/elbow angles so the end effector lands on a target, using the law
//   of cosines (the classic "reach" solver). The rest of the skeleton is left
//   untouched, so the solve is additive — it can layer on top of any clip,
//   including blends from the animation graph.
//
// WHERE IT PLUGS IN
//   Add an IkChain component to an entity that has a BlendedPose. Each frame the
//   ik_system re-solves the chain endpoints toward each target and rewrites the
//   pose's joint matrices. The renderer picks up the updated matrices, so no GPU
//   changes are needed.
// ──────────────────────────────────────────────────────────────────────────────

use crate::animation::blending::BlendedPose;
use crate::components::{Position, Renderable, Rotation};
use glam::{Mat4, Quat, Vec3};
use hecs::World;

/// Pure two-bone IK solve.
///
/// * `a` — root joint position (skeleton-local space)
/// * `b` — mid joint position
/// * `c` — end-effector position
/// * `target` — where the end effector should land (same space as a/b/c)
/// * `stretch` — 0..=1, how aggressively to extend to full reach
///
/// Returns two correction rotations:
///  - `.0`: the root delta; rotate everything below the root about `a`.
///  - `.1`: the mid delta; rotate everything below the mid about the solved
///    mid position.
#[inline]
pub fn solve_two_bone(a: Vec3, b: Vec3, c: Vec3, target: Vec3, stretch: f32) -> (Quat, Quat) {
    let l1 = (b - a).length().max(1e-5);
    let l2 = (c - b).length().max(1e-5);

    // Degenerate chain — nothing to solve.
    if l1 < 1e-4 || l2 < 1e-4 {
        return (Quat::IDENTITY, Quat::IDENTITY);
    }

    // Clamp the target into the reachable sphere (with a touch of slack).
    let max_reach = (l1 + l2) * (0.94 + 0.12 * stretch.clamp(0.0, 1.0));
    let to_target = target - a;
    let dist = to_target.length();
    let clamped_target = if dist > max_reach {
        a + to_target / dist * max_reach
    } else {
        target
    };

    let to_target = clamped_target - a;
    let dist = to_target.length().max(1e-5);
    let dir_t = to_target / dist;

    // Angle at `a` between (a→b) and (a→target) using the law of cosines,
    // where the opposite side is leg l2.
    let cos_ang_a = ((l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist)).clamp(-1.0, 1.0);
    let ang_a = cos_ang_a.acos();

    let dir_ab = (b - a) / l1;

    // Plane normal of the bend — keeps the elbow bent out of the straight line.
    let axis = dir_ab.cross(dir_t).normalize_or(dir_ab);

    // Keep the elbow on the same side of the plane as it currently is so we
    // don't pop the joint through the plane from frame to frame.
    let side_sign = {
        let bend = dir_ab.cross(dir_t);
        if bend.length_squared() < 1e-8 {
            1.0
        } else {
            let current = (c - b).cross(b - a).normalize_or(Vec3::Y);
            if current.dot(bend.normalize()) < 0.0 { -1.0 } else { 1.0 }
        }
    };
    let theta = ang_a * side_sign;

    // Solved mid position: rotate the target direction away from the straight
    // line by `theta` so the law-of-cosines triangle is satisfied.
    let dir_mid = Quat::from_axis_angle(axis, theta) * dir_t;
    let mid2 = a + dir_mid * l1;

    // Root delta: shortest arc taking the current (b−a) onto the solved mid.
    let rot_root = quat_from_dir(dir_ab, dir_mid);

    // Mid delta: shortest arc taking (c−b) onto (target−mid2).
    let rot_mid = quat_from_dir(
        (c - b).normalize_or(dir_t),
        (clamped_target - mid2).normalize_or(dir_mid),
    );

    (rot_root, rot_mid)
}

/// Rotation that takes one unit direction onto another (shortest arc).
fn quat_from_dir(from: Vec3, to: Vec3) -> Quat {
    let from = from.normalize_or(Vec3::Z);
    let to = to.normalize_or(Vec3::Z);
    let d = from.dot(to).clamp(-1.0, 1.0);
    if d > 0.999999 {
        Quat::IDENTITY
    } else if d < -0.999999 {
        // Antiparallel: rotate 180° about any perpendicular.
        let axis = from.cross(Vec3::Z).normalize_or(Vec3::Y);
        Quat::from_axis_angle(axis, std::f32::consts::PI)
    } else {
        let axis = from.cross(to).normalize();
        Quat::from_axis_angle(axis, d.acos())
    }
}

/// A per-character IkChain component.
///
/// Store on an entity alongside a `BlendedPose`. Indexes refer to the pose's
/// joint matrix list (order = skeleton bone order).
#[derive(Clone, Debug)]
pub struct IkChain {
    /// Root joint index in the BlendedPose.joint_matrices.
    pub root_joint: usize,
    /// Mid (elbow/knee) joint index.
    pub mid_joint: usize,
    /// End-effector (hand/foot) joint index.
    pub end_joint: usize,
    /// World-space position the end effector is pulled toward.
    pub target: [f32; 3],
    /// 0..=1 — how aggressively to stretch to full reach.
    pub reach: f32,
    /// Blend weight 0..=1 — how much of the solve is applied (prevents popping).
    pub weight: f32,
    /// When true the target is interpreted as skeleton-local, skipping the
    /// entity transform inversion.
    pub target_local: bool,
}

impl IkChain {
    /// End-effector chain, indexing into the pose's joint matrices.
    pub fn new(root: usize, mid: usize, end: usize) -> Self {
        Self {
            root_joint: root,
            mid_joint: mid,
            end_joint: end,
            target: [0.0; 3],
            reach: 0.5,
            weight: 1.0,
            target_local: false,
        }
    }
}

/// Per-frame system: solve every IkChain against its target and rewrite the
/// joint matrices in place. Runs after animation_blending_system/anim_graph_system.
pub fn ik_system(world: &mut World) {
    let chains: Vec<(hecs::Entity, Vec3, f32, f32, bool)> = world
        .query::<(hecs::Entity, &IkChain)>()
        .iter()
        .map(|(e, chain): (hecs::Entity, &IkChain)| {
            (
                e,
                Vec3::from_array(chain.target),
                chain.reach,
                chain.weight,
                chain.target_local,
            )
        })
        .collect();

    for (entity, target, reach, weight, local) in chains {
        apply_chain(world, entity, target, reach, weight, local);
    }
}

/// Apply IK to one entity.
///
/// `target` is world-space unless `local` is true (then skeleton-space).
/// The pose's joint matrices are expected in skeleton-local space, so a world
/// target is first pulled back into that space using the entity's transform.
/// We then rotate the mid/end matrices about their parent pivots and write the
/// result back into `BlendedPose`. Returns true on success.
pub fn apply_chain(
    world: &mut hecs::World,
    entity: hecs::Entity,
    target: Vec3,
    reach: f32,
    weight: f32,
    local_target: bool,
) -> bool {
    let (root, mid, end, target_local) = {
        let Ok(chain) = world.get::<&IkChain>(entity) else { return false };
        (
            chain.root_joint,
            chain.mid_joint,
            chain.end_joint,
            chain.target_local,
        )
    };

    // Read the pose's joint matrices.
    let (mats, model) = {
        let Ok(pose) = world.get::<&BlendedPose>(entity) else { return false };
        if end >= pose.joint_matrices.len() || mid >= pose.joint_matrices.len() || root >= pose.joint_matrices.len() {
            return false;
        }
        (pose.joint_matrices.clone(), pose_model(world, entity, local_target || target_local))
    };

    let a = mats[root].w_axis.truncate();
    let b = mats[mid].w_axis.truncate();
    let c = mats[end].w_axis.truncate();

    // Undo the entity placement so the target is in skeleton-local space.
    let target_local = if model == Mat4::IDENTITY {
        target
    } else {
        (model.inverse() * target.extend(1.0)).truncate()
    };

    let (rot_root, rot_mid) = solve_two_bone(a, b, c, target_local, reach);
    let w = weight.clamp(0.0, 1.0);

    if w <= 0.0 {
        return false;
    }

    // Solved mid/end positions (used as pivots).
    let b2 = a + rot_root * (b - a);

    // Blend the deltas so a low weight only partially moves the chain.
    let qr = Quat::slerp(Quat::IDENTITY, rot_root, w);
    let qm = Quat::slerp(Quat::IDENTITY, rot_mid, w);

    let Ok(mut pose_mut) = world.get::<&mut BlendedPose>(entity) else {
        return false;
    };
    let mats = &mut pose_mut.joint_matrices;

    // Root rotation about a propagates to mid & end.
    mats[mid] = rotate_about_center(mats[mid], a, qr);
    mats[end] = rotate_about_center(mats[end], a, qr);
    // Mid bend about the solved mid pivot propagates to the end effector.
    mats[mid] = rotate_about_center(mats[mid], b2, qm);
    mats[end] = rotate_about_center(mats[end], b2, qm);

    true
}

/// Entity's model transform (matches the renderer's skinned model build).
fn pose_model(world: &hecs::World, entity: hecs::Entity, local: bool) -> Mat4 {
    if local {
        return Mat4::IDENTITY;
    }
    let (Ok(pos), Ok(rot), Ok(renderable)) = (
        world.get::<&Position>(entity),
        world.get::<&Rotation>(entity),
        world.get::<&Renderable>(entity),
    ) else {
        return Mat4::IDENTITY;
    };
    let t = Mat4::from_translation(Vec3::new(pos.x, pos.y, pos.z));
    let ry = Mat4::from_rotation_y(rot.yaw);
    let rp = Mat4::from_rotation_x(rot.pitch);
    let rr = Mat4::from_rotation_z(rot.roll);
    let s = Mat4::from_scale(Vec3::new(
        renderable.scale[0],
        renderable.scale[1],
        renderable.scale[2],
    ));
    t * ry * rp * rr * s
}

/// Rotate a 4x4 matrix about `pivot` by `q`, keeping the translation intact
/// as a world-relative rotate-about-point.
fn rotate_about_center(m: Mat4, pivot: Vec3, q: Quat) -> Mat4 {
    let rot = Mat4::from_quat(q);
    let t_neg = Mat4::from_translation(-pivot);
    let t_pos = Mat4::from_translation(pivot);
    t_pos * rot * t_neg * m
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn dist(a: Vec3, b: Vec3) -> f32 {
        (a - b).length()
    }

    #[test]
    fn reaches_clamped_target() {
        let a = Vec3::ZERO;
        let b = Vec3::new(0.0, 0.0, 1.0);
        let c = Vec3::new(0.3, 0.2, 2.0);
        let target = Vec3::new(0.0, 0.1, 1.8);
        let (rot_root, rot_mid) = solve_two_bone(a, b, c, target, 0.5);
        let b2 = a + rot_root * (b - a);
        let c2 = b2 + rot_mid * (c - b);
        // Fulcrum lengths are preserved and the end lands on the (clamped) target.
        assert!((dist(b2, a) - dist(b, a)).abs() < 1e-3, "root leg preserved");
        assert!((dist(c2, b2) - dist(c, b)).abs() < 1e-3, "mid leg preserved");
        assert!(
            dist(c2, target) < 0.05,
            "end effector lands on target (got {})",
            dist(c2, target)
        );
    }

    #[test]
    fn unreachable_target_clamps_to_reach() {
        let a = Vec3::ZERO;
        let b = Vec3::new(0.0, 0.0, 1.0);
        let c = Vec3::new(0.0, 0.0, 2.0);
        let far = Vec3::new(100.0, 0.0, 0.0);
        let reach = (1.0 + 1.0) * (0.94 + 0.12 * 0.5);
        let (rot_root, rot_mid) = solve_two_bone(a, b, c, far, 0.5);
        let b2 = a + rot_root * (b - a);
        let c2 = b2 + rot_mid * (c - b);
        assert!(dist(a, b2) <= reach + 1e-3, "mid stays within reach sphere");
        assert!(dist(a, c2) <= reach + 1e-3, "end effector stays within reach");
    }

    #[test]
    fn zero_weight_is_identity() {
        let a = Vec3::ZERO;
        let b = Vec3::new(0.0, 0.0, 1.0);
        let c = Vec3::new(0.3, 0.2, 2.0);
        let (rr, rm) = solve_two_bone(a, b, c, Vec3::new(1.0, 0.0, 1.0), 0.5);
        assert_eq!(Quat::slerp(Quat::IDENTITY, rr, 0.0), Quat::IDENTITY);
        assert_eq!(Quat::slerp(Quat::IDENTITY, rm, 0.0), Quat::IDENTITY);
    }
}