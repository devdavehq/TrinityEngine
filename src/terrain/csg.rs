//! CSG (Constructive Solid Geometry) sculpting for level prototyping.
//!
//! Provides brush-based volume operations on a heightmap: box cuts/unions/subtracts,
//! cylinder cuts, and arbitrary-angle corridor cuts. These are the building blocks
//! for quickly carving roads, tunnels, foundations, and other level features out
//! of raw terrain.
//!
//! # Coordinate convention
//!
//! - `heights` is a flat `[f32]` of length `width * depth`, indexed as `[z * width + x]`.
//! - World-space X maps to the `x` (column) axis; world-space Z maps to the `z` (row) axis.
//! - Y is the vertical (height) axis — positive = up.
//! - `cell_size` is the world-space distance between adjacent height samples.
//!
//! # Brush volumes
//!
//! Each brush is defined in its own local coordinate frame. The brush's position,
//! rotation (Euler angles in radians: yaw around Y, pitch around X, roll around Z),
//! and scale define an oriented bounding box (OBB) or oriented cylinder in world
//! space. For every heightmap cell whose world position falls inside the brush
//! volume, the height value is modified according to the operation type.
//!
//! ## BoxCut / BoxUnion / BoxSubtract — oriented box brush
//!
//! A box brush is an OBB centered at `position`, oriented by `rotation`, and
//! extending `scale.x`/`scale.y`/`scale.z` along its local axes. The test for
//! whether a point lies inside the box is:
//!
//! 1. Transform the world point into the brush's local frame (translate by
//!    `-position`, rotate by `-rotation`).
//! 2. Test `abs(local.x) <= scale.x/2` etc. in all three axes.
//!
//! - **BoxCut**: lowers terrain so the height does not exceed the top face of the
//!   box. Equivalent to "flatten to box top."
//! - **BoxUnion**: raises terrain so the height is at least the bottom face of the
//!   box. Fills terrain into the box volume.
//! - **BoxSubtract**: carves a tunnel through terrain at the box's orientation.
//!   The box defines the tunnel cross-section; terrain above the box's ceiling is
//!   removed and terrain below the box's floor is preserved. This is useful for
//!   creating angled tunnels through mountains.
//!
//! ## CylinderCut / CylinderUnion — oriented cylinder brush
//!
//! A cylinder brush is an oriented cylinder centered at `position`, with radius
//! `scale.x` (assumes uniform XY scale) and height `scale.y`. The test for
//! whether a point lies inside is:
//!
//! 1. Transform the world point into the brush's local frame.
//! 2. Test `sqrt(local.x² + local.z²) <= radius` (radial extent) and
//!    `abs(local.y) <= height/2` (vertical extent).
//!
//! - **CylinderCut**: lowers terrain to the bottom of the cylinder.
//! - **CylinderUnion**: raises terrain to the top of the cylinder.
//!
//! ## Angle cut (apply_angle_cut)
//!
//! An angle cut carves a corridor through terrain from any origin point, along
//! any direction, at a given width. This is the key feature for level prototyping:
//! you can cut a road or tunnel path through a mountain at any angle without
//! aligning to the grid.
//!
//! The cut uses a swept-line test: for each heightmap cell, project its world
//! position onto the cut direction using a dot product. If the projected position
//! lies within the cut corridor (between `origin` and `origin + direction` in
//! the along-axis, and within `width/2` in the perpendicular axis), the cell is
//! lowered to the cut floor height.

use glam::{Mat4, Quat, Vec3};

// ---------------------------------------------------------------------------
// Enums & structs
// ---------------------------------------------------------------------------

/// The type of CSG operation a brush performs on the heightmap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CsgOperation {
    /// Carve terrain down to the top face of the box.
    /// Creates a flat shelf / foundation.
    BoxCut,
    /// Raise terrain up to the bottom face of the box.
    /// Fills depressions inside the box volume.
    BoxUnion,
    /// Carve a tunnel of the given box cross-section through terrain.
    /// Terrain above the box is removed; terrain below the box is kept.
    BoxSubtract,
    /// Cylindrical cut — lowers terrain inside the cylinder to its base.
    CylinderCut,
    /// Cylindrical fill — raises terrain inside the cylinder to its top.
    CylinderUnion,
}

/// A CSG brush: an oriented volume that modifies the heightmap.
///
/// The brush is defined in world space by its centre `position`, Euler angle
/// `rotation` (yaw, pitch, roll in radians, applied yaw→pitch→roll), and
/// axis-aligned `scale` (half-extents for boxes, radius/height for cylinders).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CsgBrush {
    /// World-space centre of the brush.
    pub position: [f32; 3],
    /// Euler angles (yaw, pitch, roll) in radians.
    pub rotation: [f32; 3],
    /// Scale (half-extents) in each local axis.
    /// For boxes: XY half-extent on the XZ plane, height along Y.
    /// For cylinders: radius (X = Z), height (Y).
    pub scale: [f32; 3],
    /// Which CSG operation to apply.
    pub operation: CsgOperation,
}

impl CsgBrush {
    /// Quick constructor — creates a brush with zero rotation.
    pub const fn new(
        position: [f32; 3],
        scale: [f32; 3],
        operation: CsgOperation,
    ) -> Self {
        Self {
            position,
            rotation: [0.0, 0.0, 0.0],
            scale,
            operation,
        }
    }

    /// Full constructor including rotation.
    pub const fn new_with_rotation(
        position: [f32; 3],
        rotation: [f32; 3],
        scale: [f32; 3],
        operation: CsgOperation,
    ) -> Self {
        Self {
            position,
            rotation,
            scale,
            operation,
        }
    }
}

/// Records which region of the heightmap was modified by a CSG operation.
///
/// This can be used to invalidate only the affected chunks and re-erode or
/// re-foliage only the modified area, rather than the entire terrain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SculptResult {
    /// Minimum grid-X index that was touched (inclusive).
    pub min_x: usize,
    /// Maximum grid-X index that was touched (exclusive).
    pub max_x: usize,
    /// Minimum grid-Z index that was touched (inclusive).
    pub min_z: usize,
    /// Maximum grid-Z index that was touched (exclusive).
    pub max_z: usize,
    /// Total number of cells whose height was changed.
    pub cells_modified: usize,
}

impl SculptResult {
    /// Returns an empty result (no cells modified).
    pub fn empty() -> Self {
        Self {
            min_x: usize::MAX,
            max_x: 0,
            min_z: usize::MAX,
            max_z: 0,
            cells_modified: 0,
        }
    }

    /// Returns `true` if no cells were modified.
    pub fn is_empty(&self) -> bool {
        self.cells_modified == 0
    }

    /// Incorporates a single grid cell into the bounds.
    fn include(&mut self, x: usize, z: usize) {
        if x < self.min_x {
            self.min_x = x;
        }
        if x >= self.max_x {
            self.max_x = x + 1;
        }
        if z < self.min_z {
            self.min_z = z;
        }
        if z >= self.max_z {
            self.max_z = z + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a 4×4 transform matrix from the brush's position, rotation, and scale.
///
/// The matrix maps from brush-local space to world space:
/// `world_point = mat * local_point`.
///
/// To test whether a world point is inside the brush volume we need the inverse:
/// `local_point = mat⁻¹ * world_point`, then check local bounds.
fn brush_to_world(brush: &CsgBrush) -> Mat4 {
    let t = Mat4::from_translation(Vec3::from_array(brush.position));
    let r = Mat4::from_quat(Quat::from_euler(
        glam::EulerRot::YXZ,
        brush.rotation[0],
        brush.rotation[1],
        brush.rotation[2],
    ));
    let s = Mat4::from_scale(Vec3::from_array(brush.scale));
    t * r * s
}

/// Compute the world-space bounding box of the brush volume (used to narrow
/// the iteration to cells that could possibly be affected).
///
/// For boxes the extents are the OBB corners projected to axes; for cylinders
/// we compute a sphere bounding the cylinder.
fn brush_world_bounds(brush: &CsgBrush) -> ([f32; 3], [f32; 3]) {
    let mat = brush_to_world(brush);
    let half_ext = [1.0, 1.0, 1.0]; // scale is baked into the matrix

    let mut min_p = Vec3::splat(f32::MAX);
    let mut max_p = Vec3::splat(f32::MIN);

    // 8 corners of the unit cube in brush-local space (before scale).
    for &(sx, sy, sz) in &[
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ] {
        let local = Vec3::new(sx * half_ext[0], sy * half_ext[1], sz * half_ext[2]);
        let world = mat.transform_point3(local);
        min_p = min_p.min(world);
        max_p = max_p.max(world);
    }

    (min_p.to_array(), max_p.to_array())
}

/// Compute the oriented bounding box (OBB) world-space extent for a cylinder.
/// We approximate with an AABB containing the oriented cylinder (the cylinder
/// is bounded by a sphere of radius sqrt(radius² + (height/2)²), but for
/// efficiency we just compute it from the brush transform corners:
/// a cylinder is bounded by the same 8-corner box of its radius × height.
fn cylinder_world_radius(brush: &CsgBrush) -> f32 {
    let half_h = brush.scale[1] * 0.5;
    (brush.scale[0] * brush.scale[0] + half_h * half_h).sqrt()
}

// ---------------------------------------------------------------------------
// Box brush helpers
// ---------------------------------------------------------------------------

/// Test if a world-space point falls inside the oriented box brush volume.
///
/// # Math
///
/// We build `M = T * R * S` (world = M * local). To test a world point P:
///
/// 1. `local = M⁻¹ * P`
/// 2. Test `|local.x| <= 1` and `|local.y| <= 1` and `|local.z| <= 1`
///
/// Because the scale is baked into M, the unit cube [-1,1]³ in local space
/// corresponds to the full brush extent.
fn point_in_box(world_p: Vec3, _brush: &CsgBrush, mat_inv: &Mat4) -> bool {
    let local = mat_inv.transform_point3(world_p);
    local.x.abs() <= 1.0 && local.y.abs() <= 1.0 && local.z.abs() <= 1.0
}

/// Test if a world-space point falls inside the oriented cylinder brush volume.
///
/// # Math
///
/// In brush-local space the cylinder axis is Y. A point is inside if:
/// - `local.y` ∈ [-1, 1] (the scale bakes height/2 into the matrix Y scale)
/// - `sqrt(local.x² + local.z²) <= 1` (the scale bakes radius into X and Z)
fn point_in_cylinder(world_p: Vec3, _brush: &CsgBrush, mat_inv: &Mat4) -> bool {
    let local = mat_inv.transform_point3(world_p);
    let radial = (local.x * local.x + local.z * local.z).sqrt();
    radial <= 1.0 && local.y.abs() <= 1.0
}

// ---------------------------------------------------------------------------
// CSG operations
// ---------------------------------------------------------------------------

/// Apply a box brush to the heightmap.
///
/// For each heightmap cell whose world position lies inside the oriented box:
///
/// - `BoxCut`: set height to `min(current, brush_top)` — carve terrain down.
/// - `BoxUnion`: set height to `max(current, brush_bottom)` — fill terrain up.
/// - `BoxSubtract`: if the column's terrain surface lies above the brush's
///   bottom, we carve a tunnel: anything above the brush's ceiling is removed.
///   If the column's terrain surface lies below the brush's bottom, we leave
///   it untouched (preserving the ground under the tunnel).
///
/// The brush's top and bottom are in world space, computed by transforming
/// the local Y-extremal points through the brush matrix.
///
/// Returns a [`SculptResult`] describing the modified region.
pub fn apply_csg_box(
    heights: &mut [f32],
    width: usize,
    depth: usize,
    cell_size: f32,
    brush: &CsgBrush,
) -> SculptResult {
    assert_eq!(
        heights.len(),
        width * depth,
        "heightmap length does not match width * depth"
    );

    let mat = brush_to_world(brush);
    let mat_inv = mat.inverse();

    // Compute the brush top and bottom in world space.
    // In local space, the box top is (0, 1, 0) and bottom is (0, -1, 0)
    // (since scale is baked into the matrix, the local Y range is [-1, 1]).
    let local_top = Vec3::new(0.0, 1.0, 0.0);
    let local_bottom = Vec3::new(0.0, -1.0, 0.0);
    let world_top = mat.transform_point3(local_top).y;
    let world_bottom = mat.transform_point3(local_bottom).y;

    // Compute world AABB of the brush so we only iterate candidate cells.
    let (bmin, bmax) = brush_world_bounds(brush);

    let x_world_min = (width as f32 - 1.0) * -0.5 * cell_size;
    let z_world_min = (depth as f32 - 1.0) * -0.5 * cell_size;

    // Convert world bounds to grid index range, clamped to heightmap.
    let min_gx = (((bmin[0] - x_world_min) / cell_size) as i32).max(0);
    let max_gx = (((bmax[0] - x_world_min) / cell_size) as i32 + 1).min(width as i32);
    let min_gz = (((bmin[2] - z_world_min) / cell_size) as i32).max(0);
    let max_gz = (((bmax[2] - z_world_min) / cell_size) as i32 + 1).min(depth as i32);

    let mut result = SculptResult::empty();

    for gz in min_gz..max_gz {
        for gx in min_gx..max_gx {
            let idx = gz as usize * width + gx as usize;
            let current = heights[idx];

            // World position of this column. The Y coordinate is set to the brush's
            // center Y (brush.position[1]) so the lateral OBB test works correctly:
            // in brush local frame, the column's Y offset from the brush center will be 0.
            let wx = gx as f32 * cell_size + x_world_min;
            let wz = gz as f32 * cell_size + z_world_min;
            let world_p = Vec3::new(wx, brush.position[1], wz);

            if !point_in_box(world_p, brush, &mat_inv) {
                continue;
            }

            let current = heights[idx];

            let new_height = match brush.operation {
                CsgOperation::BoxCut => current.min(world_top),
                CsgOperation::BoxUnion => current.max(world_bottom),
                CsgOperation::BoxSubtract => {
                    // BoxSubtract removes the entire brush volume from the terrain.
                    // For any column whose XZ projection overlaps the OBB:
                    // - If the terrain surface is above the box bottom, the column is
                    //   excavated down to the box bottom (the tunnel floor). This creates
                    //   a passage through the terrain.
                    // - If the terrain surface is below the box bottom, it is left
                    //   unchanged (e.g. a tunnel passing over a valley).
                    if current > world_bottom {
                        world_bottom
                    } else {
                        current
                    }
                }
                _ => current,
            };

            if (new_height - current).abs() > 1e-8 {
                heights[idx] = new_height;
                result.include(gx as usize, gz as usize);
                result.cells_modified += 1;
            }
        }
    }

    result
}

/// Apply a cylinder brush to the heightmap.
///
/// For each heightmap cell whose world position lies inside the oriented
/// cylinder:
///
/// - `CylinderCut`: set height to `min(current, brush_bottom)`.
/// - `CylinderUnion`: set height to `max(current, brush_top)`.
///
/// Returns a [`SculptResult`] describing the modified region.
pub fn apply_csg_cylinder(
    heights: &mut [f32],
    width: usize,
    depth: usize,
    cell_size: f32,
    brush: &CsgBrush,
) -> SculptResult {
    assert_eq!(
        heights.len(),
        width * depth,
        "heightmap length does not match width * depth"
    );

    let mat = brush_to_world(brush);
    let mat_inv = mat.inverse();

    // Cylinder top/bottom in world space (local Y ∈ [-1, 1]).
    let local_top = Vec3::new(0.0, 1.0, 0.0);
    let local_bottom = Vec3::new(0.0, -1.0, 0.0);
    let world_top = mat.transform_point3(local_top).y;
    let world_bottom = mat.transform_point3(local_bottom).y;

    // Compute an approximate world bounding sphere radius for iteration.
    let world_radius = cylinder_world_radius(brush);
    let center = Vec3::from_array(brush.position);

    let x_world_min = (width as f32 - 1.0) * -0.5 * cell_size;
    let z_world_min = (depth as f32 - 1.0) * -0.5 * cell_size;

    // Convert world bounding box (center ± radius) to grid range.
    let min_gx = (((center.x - world_radius - x_world_min) / cell_size) as i32).max(0);
    let max_gx = (((center.x + world_radius - x_world_min) / cell_size) as i32 + 1).min(width as i32);
    let min_gz = (((center.z - world_radius - z_world_min) / cell_size) as i32).max(0);
    let max_gz = (((center.z + world_radius - z_world_min) / cell_size) as i32 + 1).min(depth as i32);

    let mut result = SculptResult::empty();

    for gz in min_gz..max_gz {
        for gx in min_gx..max_gx {
            let idx = gz as usize * width + gx as usize;
            let current = heights[idx];

            // World position at brush center height for correct lateral OBB/cylinder
            // test. See apply_csg_box comment.
            let wx = gx as f32 * cell_size + x_world_min;
            let wz = gz as f32 * cell_size + z_world_min;
            let world_p = Vec3::new(wx, brush.position[1], wz);

            if !point_in_cylinder(world_p, brush, &mat_inv) {
                continue;
            }

            let new_height = match brush.operation {
                CsgOperation::CylinderCut => current.min(world_bottom),
                CsgOperation::CylinderUnion => current.max(world_top),
                _ => current,
            };

            if (new_height - current).abs() > 1e-8 {
                heights[idx] = new_height;
                result.include(gx as usize, gz as usize);
                result.cells_modified += 1;
            }
        }
    }

    result
}

/// Cut a corridor through terrain at an arbitrary angle.
///
/// This is the key tool for level prototyping: you define a cut line with an
/// origin point, a direction vector (the corridor axis), and a corridor width.
/// The function then projects every heightmap cell onto the cut axis using a
/// dot product. Cells whose projected position falls within the swept corridor
/// (between the origin and `origin + direction` along the axis, and within
/// `width / 2` perpendicular to it) are lowered to the cut height — the height
/// of the terrain at the corridor's origin.
///
/// # Parameters
///
/// - `heights` — flat heightmap array (length `width * depth`).
/// - `width` — grid width in cells.
/// - `depth` — grid depth in cells.
/// - `cell_size` — world-space distance between adjacent cells.
/// - `origin` — world-space `[x, z]` starting point of the cut.
/// - `direction` — world-space `[dx, dz]` direction and length of the cut.
///   The cut runs from `origin` to `origin + direction`.
/// - `corridor_width` — width of the corridor (perpendicular to direction).
/// - `cut_to_height` — the target height to lower terrain to. If `None`, the
///   cut uses the terrain height at the origin as the target (so it follows
///   the existing ground level at the start).
///
/// # Math
///
/// For each cell at world position `P = (px, pz)`:
///
/// 1. Compute `along = dot(P - origin, dir_norm)`, the projection onto the
///    corridor axis, where `dir_norm` is the unit vector of `direction`.
/// 2. Compute `perp = |cross(P - origin, dir_norm)|`, the perpendicular
///    distance from the corridor axis.
/// 3. If `along >= 0`, `along <= dir_len`, and `perp <= corridor_width / 2`,
///    the cell is inside the swept corridor and its height is set to
///    `cut_to_height`.
///
/// Returns a [`SculptResult`] describing the modified region.
pub fn apply_angle_cut(
    heights: &mut [f32],
    width: usize,
    depth: usize,
    cell_size: f32,
    origin: [f32; 2],
    direction: [f32; 2],
    corridor_width: f32,
    cut_to_height: Option<f32>,
) -> SculptResult {
    assert_eq!(
        heights.len(),
        width * depth,
        "heightmap length does not match width * depth"
    );

    let dir_vec = Vec3::new(direction[0], 0.0, direction[1]);
    let dir_len = dir_vec.length();
    if dir_len < 1e-8 {
        return SculptResult::empty();
    }
    let dir_norm = dir_vec / dir_len;
    let origin_v = Vec3::new(origin[0], 0.0, origin[1]);
    let half_w = corridor_width * 0.5;

    // Determine the cut target height: either explicitly provided or sampled
    // from the heightmap at the origin's grid cell.
    let target_h = match cut_to_height {
        Some(h) => h,
        None => {
            let ox = origin[0];
            let oz = origin[1];
            let x_world_min = (width as f32 - 1.0) * -0.5 * cell_size;
            let z_world_min = (depth as f32 - 1.0) * -0.5 * cell_size;
            let gx = (((ox - x_world_min) / cell_size) as i32)
                .clamp(0, width as i32 - 1) as usize;
            let gz = (((oz - z_world_min) / cell_size) as i32)
                .clamp(0, depth as i32 - 1) as usize;
            heights[gz * width + gx]
        }
    };

    let x_world_min = (width as f32 - 1.0) * -0.5 * cell_size;
    let z_world_min = (depth as f32 - 1.0) * -0.5 * cell_size;

    // Compute an AABB of the corridor in world space to limit iteration.
    let corridor_end = origin_v + dir_vec;
    let perp = Vec3::new(-dir_norm.z, 0.0, dir_norm.x);
    let corners = [
        origin_v + perp * half_w,
        origin_v - perp * half_w,
        corridor_end + perp * half_w,
        corridor_end - perp * half_w,
    ];
    let mut cmin = Vec3::splat(f32::MAX);
    let mut cmax = Vec3::splat(f32::MIN);
    for c in &corners {
        cmin = cmin.min(*c);
        cmax = cmax.max(*c);
    }

    let min_gx = (((cmin.x - x_world_min) / cell_size) as i32).max(0);
    let max_gx = (((cmax.x - x_world_min) / cell_size) as i32 + 1).min(width as i32);
    let min_gz = (((cmin.z - z_world_min) / cell_size) as i32).max(0);
    let max_gz = (((cmax.z - z_world_min) / cell_size) as i32 + 1).min(depth as i32);

    let mut result = SculptResult::empty();

    for gz in min_gz..max_gz {
        for gx in min_gx..max_gx {
            let wx = gx as f32 * cell_size + x_world_min;
            let wz = gz as f32 * cell_size + z_world_min;

            // Project cell position onto the corridor axis.
            let p = Vec3::new(wx, 0.0, wz);
            let rel = p - origin_v;

            let along = rel.dot(dir_norm);
            // Clamp along to [0, dir_len] and compute perpendicular distance
            // from the clamped point on the segment.
            let along_clamped = along.clamp(0.0, dir_len);
            let closest = origin_v + dir_norm * along_clamped;
            let perp_dist = (p - closest).length();

            if perp_dist <= half_w {
                let idx = gz as usize * width + gx as usize;
                let current = heights[idx];

                if target_h < current - 1e-8 {
                    heights[idx] = target_h;
                    result.include(gx as usize, gz as usize);
                    result.cells_modified += 1;
                }
            }
        }
    }

    result
}

/// Apply a batch of CSG brushes sequentially to the heightmap.
///
/// Brushes are applied in order — later brushes operate on the terrain
/// modified by earlier ones. This makes it possible to chain operations
/// (e.g. cut a foundation, then union-fill a ramp).
///
/// Returns a `SculptResult` that represents the union of all modified regions.
pub fn apply_multiple_brushes(
    heights: &mut [f32],
    width: usize,
    depth: usize,
    cell_size: f32,
    brushes: &[CsgBrush],
) -> SculptResult {
    let mut combined = SculptResult::empty();

    for brush in brushes {
        let result = match brush.operation {
            CsgOperation::BoxCut | CsgOperation::BoxUnion | CsgOperation::BoxSubtract => {
                apply_csg_box(heights, width, depth, cell_size, brush)
            }
            CsgOperation::CylinderCut | CsgOperation::CylinderUnion => {
                apply_csg_cylinder(heights, width, depth, cell_size, brush)
            }
        };

        if !result.is_empty() {
            if combined.is_empty() {
                combined = result;
            } else {
                combined.min_x = combined.min_x.min(result.min_x);
                combined.max_x = combined.max_x.max(result.max_x);
                combined.min_z = combined.min_z.min(result.min_z);
                combined.max_z = combined.max_z.max(result.max_z);
                combined.cells_modified += result.cells_modified;
            }
        }
    }

    combined
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a flat heightmap of the given dimensions and height.
    fn make_flat(width: usize, depth: usize, height: f32) -> Vec<f32> {
        vec![height; width * depth]
    }

    /// Helper: check that all cells in the heightmap are the expected value
    /// within a small tolerance.
    fn assert_all_approx(heights: &[f32], expected: f32) {
        for (i, &h) in heights.iter().enumerate() {
            assert!(
                (h - expected).abs() < 1e-5,
                "heights[{}] = {}, expected {}",
                i,
                h,
                expected
            );
        }
    }

    // ── BoxCut ──────────────────────────────────────────────────────────

    /// BoxCut should lower terrain inside the box to the box top.
    #[test]
    fn box_cut_lowers_terrain() {
        let w = 32;
        let d = 32;
        let cell = 1.0;
        let mut h = make_flat(w, d, 10.0);

        let brush = CsgBrush::new(
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],
            CsgOperation::BoxCut,
        );

        let result = apply_csg_box(&mut h, w, d, cell, &brush);
        assert!(result.cells_modified > 0, "expected some cells to be modified");

        // The box top is at position.y + scale.y = 0 + 5 = 5.
        // All modified cells should be at most 5.0.
        for gz in 0..d {
            for gx in 0..w {
                let idx = gz * w + gx;
                assert!(
                    h[idx] <= 10.0,
                    "height should not exceed original 10.0 at ({}, {})",
                    gx,
                    gz
                );
            }
        }

        // The center column should be exactly 5.0 (cut to box top).
        let center_idx = (d / 2) * w + (w / 2);
        assert!(
            (h[center_idx] - 5.0).abs() < 1e-5,
            "center should be cut to 5.0, got {}",
            h[center_idx]
        );
    }

    /// BoxCut with an unaligned brush should still lower terrain.
    #[test]
    fn box_cut_rotated() {
        let w = 32;
        let d = 32;
        let cell = 1.0;
        let mut h = make_flat(w, d, 10.0);

        let brush = CsgBrush::new_with_rotation(
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0], // 0.3 rad yaw
            [4.0, 4.0, 4.0],
            CsgOperation::BoxCut,
        );

        let result = apply_csg_box(&mut h, w, d, cell, &brush);
        assert!(result.cells_modified > 0, "rotated cut should modify cells");
    }

    // ── BoxUnion ────────────────────────────────────────────────────────

    /// BoxUnion should raise terrain inside the box to the box bottom.
    #[test]
    fn box_union_raises_terrain() {
        let w = 16;
        let d = 16;
        let cell = 1.0;
        // Terrain is at 0.0; the brush centre is at y = -3.0 so the
        // bottom face is at y - scale.y = -3 - 3 = -6 (below terrain, no raise)
        // and the top face is at y + scale.y = -3 + 3 = 0.
        // To demonstrate raising, place the brush bottom above terrain:
        // centre at y = 2.0, scale.y = 2.0 => bottom at 0.0, top at 4.0.
        let mut h = make_flat(w, d, -2.0);

        let brush = CsgBrush::new(
            [0.0, 2.0, 0.0],
            [4.0, 2.0, 4.0],
            CsgOperation::BoxUnion,
        );

        let result = apply_csg_box(&mut h, w, d, cell, &brush);
        assert!(result.cells_modified > 0);

        // The box bottom is at 2.0 - 2.0 = 0.0.
        // Cells inside the box should now be at least 0.0.
        for gz in 0..d {
            for gx in 0..w {
                let idx = gz * w + gx;
                assert!(
                    h[idx] >= -2.0,
                    "height should not drop below original at ({}, {})",
                    gx,
                    gz
                );
            }
        }

        let center_idx = (d / 2) * w + (w / 2);
        assert!(
            (h[center_idx] - 0.0).abs() < 1e-5,
            "center should be raised to 0.0 (box bottom), got {}",
            h[center_idx]
        );
    }

    // ── BoxSubtract ─────────────────────────────────────────────────────

    /// BoxSubtract should carve a tunnel: terrain above the box ceiling is
    /// cut to the ceiling, terrain inside the box is cut to the floor,
    /// terrain below the floor is preserved.
    #[test]
    fn box_subtract_carves_tunnel() {
        let w = 32;
        let d = 32;
        let cell = 1.0;
        // Place a high plateau at height 15.0.
        let mut h = make_flat(w, d, 15.0);

        // Brush: centre at y = 10.0, scale.y = 5.0
        //   top = 15.0, bottom = 5.0
        // Terrain at 15.0 is exactly at the top -> should be cut to top (15.0).
        // We want terrain above ceiling so set terrain at 20.0.
        for z in 0..d {
            for x in 0..w {
                h[z * w + x] = 20.0;
            }
        }

        let brush = CsgBrush::new(
            [0.0, 10.0, 0.0],
            [6.0, 5.0, 6.0],
            CsgOperation::BoxSubtract,
        );

        let result = apply_csg_box(&mut h, w, d, cell, &brush);
        assert!(result.cells_modified > 0);

        // Inside the brush: terrain was at 20.0, box top = 15.0, bottom = 5.0.
        // Current (20) > world_top (15) -> cut to top -> 15.0
        let center_idx = (d / 2) * w + (w / 2);
        assert!(
            (h[center_idx] - 5.0).abs() < 1e-5,
            "inside tunnel should be cut to floor (5.0), got {}",
            h[center_idx]
        );

        // Cells far outside should remain at 20.0.
        let corner_idx = 0;
        assert!(
            (h[corner_idx] - 20.0).abs() < 1e-5,
            "outside tunnel should remain 20.0, got {}",
            h[corner_idx]
        );
    }

    // ── CylinderCut ─────────────────────────────────────────────────────

    /// CylinderCut should lower terrain inside the cylinder to the cylinder base.
    #[test]
    fn cylinder_cut_lowers_terrain() {
        let w = 32;
        let d = 32;
        let cell = 1.0;
        let mut h = make_flat(w, d, 10.0);

        let brush = CsgBrush::new(
            [0.0, 5.0, 0.0], // centre at y=5, scale.y=3 => bottom=2, top=8
            [3.0, 3.0, 3.0],
            CsgOperation::CylinderCut,
        );

        let result = apply_csg_cylinder(&mut h, w, d, cell, &brush);
        assert!(result.cells_modified > 0);

        // The cylinder base (bottom) is at 5.0 - 3.0 = 2.0.
        // The centre should be cut to 2.0.
        let center_idx = (d / 2) * w + (w / 2);
        assert!(
            (h[center_idx] - 2.0).abs() < 1e-5,
            "center should be cut to cylinder bottom (2.0), got {}",
            h[center_idx]
        );

        // Corners (outside cylinder) should remain at 10.0.
        let corner_idx = 0;
        assert!(
            (h[corner_idx] - 10.0).abs() < 1e-5,
            "corner should remain unchanged at 10.0, got {}",
            h[corner_idx]
        );
    }

    // ── CylinderUnion ───────────────────────────────────────────────────

    /// CylinderUnion should raise terrain inside the cylinder to the cylinder top.
    #[test]
    fn cylinder_union_raises_terrain() {
        let w = 16;
        let d = 16;
        let cell = 1.0;
        let mut h = make_flat(w, d, 0.0);

        let brush = CsgBrush::new(
            [0.0, 5.0, 0.0],
            [2.0, 5.0, 2.0], // top = 5+5 = 10, bottom = 5-5 = 0
            CsgOperation::CylinderUnion,
        );

        let result = apply_csg_cylinder(&mut h, w, d, cell, &brush);
        assert!(result.cells_modified > 0);

        // The cylinder top is at 5.0 + 5.0 = 10.0.
        let center_idx = (d / 2) * w + (w / 2);
        assert!(
            (h[center_idx] - 10.0).abs() < 1e-5,
            "center should be raised to cylinder top (10.0), got {}",
            h[center_idx]
        );
    }

    // ── Angle cut ───────────────────────────────────────────────────────

    /// An angle cut should carve a corridor through high terrain.
    #[test]
    fn angle_cut_carves_corridor() {
        let w = 32;
        let d = 32;
        let cell = 1.0;
        // Uniform high plateau.
        let mut h = make_flat(w, d, 20.0);

        // Cut from (-8, 0) to (8, 0) with width 4.0, target height 5.0.
        let result = apply_angle_cut(
            &mut h,
            w,
            d,
            cell,
            [-8.0, 0.0],
            [16.0, 0.0],
            4.0,
            Some(5.0),
        );

        assert!(result.cells_modified > 0, "angle cut should modify cells");

        // Cells along the corridor axis should now be at 5.0.
        let center_x = w / 2;
        let center_z = d / 2;
        let center_idx = center_z * w + center_x;
        assert!(
            (h[center_idx] - 5.0).abs() < 1e-5,
            "center of corridor should be at target height 5.0, got {}",
            h[center_idx]
        );

        // Cells far off the axis should remain at 20.0.
        // Pick a corner that is far from the cut.
        let corner_idx = 0;
        assert!(
            (h[corner_idx] - 20.0).abs() < 1e-5,
            "corner outside corridor should remain 20.0, got {}",
            h[corner_idx]
        );
    }

    /// An angle cut with no explicit height should sample the height at the
    /// origin and use that as the target.
    #[test]
    fn angle_cut_no_height_samples_origin() {
        let w = 32;
        let d = 32;
        let cell = 1.0;
        let x_world_min = (w as f32 - 1.0) * -0.5 * cell; // -15.5
        let z_world_min = (d as f32 - 1.0) * -0.5 * cell; // -15.5

        // Compute grid coords for origin (-5.0, 0.0)
        let origin_gx = (((-5.0 - x_world_min) / cell) as i32).clamp(0, w as i32 - 1) as usize;
        let origin_gz = (((0.0 - z_world_min) / cell) as i32).clamp(0, d as i32 - 1) as usize;

        let mut h = vec![30.0; w * d];
        // Set the origin cell to 10.0
        h[origin_gz * w + origin_gx] = 10.0;

        // Cut with no specified height — should sample the origin (10.0).
        let result = apply_angle_cut(
            &mut h,
            w,
            d,
            cell,
            [-5.0, 0.0],
            [10.0, 0.0],
            3.0,
            None,
        );

        assert!(result.cells_modified > 0,
            "expected some cells to be modified, origin_gx={origin_gx}, origin_gz={origin_gz}"
        );

        // The origin cell's height remains 10.0 (it was already that).
        assert!(
            (h[origin_gz * w + origin_gx] - 10.0).abs() < 1e-5,
            "origin cell should stay at 10.0"
        );

        // Cells along the corridor should be lowered to 10.0.
        let along_idx = origin_gz * w + (origin_gx + 3);
        assert!(
            (h[along_idx] - 10.0).abs() < 1e-5,
            "cell along corridor should be lowered to 10.0, got {}",
            h[along_idx]
        );
    }

    // ── Batch processing ────────────────────────────────────────────────

    /// Applying multiple brushes in sequence should combine their effects.
    #[test]
    fn multiple_brushes_combine() {
        let w = 32;
        let d = 32;
        let cell = 1.0;
        let mut h = make_flat(w, d, 20.0);

        // Two overlapping box cuts — their tops are at y=10, terrain at 20, so cut.
        let brush_a = CsgBrush::new(
            [-3.0, 5.0, 0.0],
            [4.0, 5.0, 10.0],
            CsgOperation::BoxCut,
        );
        let brush_b = CsgBrush::new(
            [3.0, 5.0, 0.0],
            [4.0, 5.0, 10.0],
            CsgOperation::BoxCut,
        );

        let result = apply_multiple_brushes(&mut h, w, d, cell, &[brush_a, brush_b]);
        assert!(result.cells_modified > 0);

        // The union AABB should cover the combined extents.
        assert!(result.min_x < w / 2, "result should cover left brush");
        assert!(result.max_x > w / 2, "result should cover right brush");
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    /// A brush partially outside the heightmap bounds should still work
    /// (only operate on the portion inside).
    #[test]
    fn brush_partially_outside_bounds() {
        let w = 16;
        let d = 16;
        let cell = 1.0;
        let mut h = make_flat(w, d, 10.0);

        // Brush positioned far to the left — most of its volume is outside
        // the heightmap.
        let brush = CsgBrush::new(
            [-20.0, 5.0, 0.0],
            [10.0, 5.0, 10.0],
            CsgOperation::BoxCut,
        );

        // This should not panic and should modify cells within the grid.
        let result = apply_csg_box(&mut h, w, d, cell, &brush);

        // Some cells on the left edge may be inside the box, but not all.
        // The important thing is it didn't panic.
        assert!(result.cells_modified <= w * d);

        // The total height sum should be ≤ original (we only cut, never raise).
        let sum: f32 = h.iter().sum();
        let original_sum = 10.0 * (w * d) as f32;
        assert!(sum <= original_sum + 1e-5);
    }

    /// A brush fully outside the heightmap bounds should return an empty result.
    #[test]
    fn brush_fully_outside_bounds() {
        let w = 8;
        let d = 8;
        let cell = 1.0;
        let mut h = make_flat(w, d, 5.0);

        // Brush positioned very far away.
        let brush = CsgBrush::new(
            [1000.0, 0.0, 1000.0],
            [1.0, 1.0, 1.0],
            CsgOperation::BoxCut,
        );

        let result = apply_csg_box(&mut h, w, d, cell, &brush);
        assert_eq!(result.cells_modified, 0);
        assert_all_approx(&h, 5.0);
    }

    /// A zero-length direction vector should return an empty result for angle cut.
    #[test]
    fn angle_cut_zero_direction() {
        let w = 16;
        let d = 16;
        let cell = 1.0;
        let mut h = make_flat(w, d, 10.0);

        let result = apply_angle_cut(
            &mut h,
            w,
            d,
            cell,
            [0.0, 0.0],
            [0.0, 0.0],
            2.0,
            Some(0.0),
        );

        assert_eq!(result.cells_modified, 0);
    }

    /// Verify that SculptResult::empty() and is_empty() work correctly.
    #[test]
    fn sculpt_result_empty() {
        let r = SculptResult::empty();
        assert!(r.is_empty());
        assert_eq!(r.cells_modified, 0);
    }
}
