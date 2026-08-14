// src/render/occlusion.rs
// ──────────────────────────────────────────────────────────────────────────────
// Software occlusion culling.
//
// WHY IT EXISTS:
//   Frustum culling only skips things *outside* the view. Occlusion culling
//   skips things *behind* the things in front of them — the reason a street
//   full of buildings should cost as much as a park. This module implements a
//   coarse, fully-CPU hierarchical-z style occlusion test so we can reject
//   hidden meshes before the GPU ever sees them.
//
// HOW IT WORKS
//   Each frame the engine submits "occluders" (large static meshes marked with
//   the Occluder component). We project each occluder's bounding sphere into a
//   low-resolution NxN screen-space grid, keeping the nearest occluder depth per
//   cell. Later, any drawable whose projected bounding sphere falls entirely in
//   cells already filled by a nearer occluder is rejected.
//
//   The grid is deliberately coarse (e.g. 48x48) so a handful of occluders is
//   enough to hide a whole street. It's conservative: we never cull a sphere
//   that even partially peeks out of the covered cells, so no popping.
// ──────────────────────────────────────────────────────────────────────────────

use glam::{Mat4, Vec3};

/// Default grid resolution per axis.
pub const GRID_SIZE: usize = 48;
/// Conservative depth margin (world units) so thin occluders don't cause
/// false positives from depth rounding.
const DEPTH_MARGIN: f32 = 0.05;

/// A software depth grid used for CPU occlusion queries.
#[derive(Clone, Debug)]
pub struct OcclusionCuller {
    /// Nearest occluder depth per cell (camera distance), f32::INFINITY = empty.
    depth: Vec<f32>,
    /// Grid resolution (GRID_SIZE x GRID_SIZE).
    grid: usize,
}

impl Default for OcclusionCuller {
    fn default() -> Self {
        Self::new(GRID_SIZE)
    }
}

impl OcclusionCuller {
    pub fn new(grid: usize) -> Self {
        Self {
            depth: vec![f32::INFINITY; grid * grid],
            grid: grid.max(8),
        }
    }

    /// Clear the grid — call once at the start of each frame before submitting
    /// occluders and testing drawables.
    pub fn begin_frame(&mut self) {
        for d in &mut self.depth {
            *d = f32::INFINITY;
        }
    }

    /// Project a point into NDC via the view-projection matrix.
    /// Returns `None` if it lands outside the viewport or behind the camera.
    fn project(vp: Mat4, p: Vec3) -> Option<(f32, f32, f32)> {
        let clip = vp * p.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
            return None;
        }
        Some((ndc.x, ndc.y, ndc.z))
    }

    /// Like `project` but also returns homogenous w, used to reject occluders
    /// behind the camera without conflating them with "off-screen to the side".
    fn project_w(vp: Mat4, p: Vec3) -> Option<(f32, f32, f32, f32)> {
        let clip = vp * p.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        Some((clip.x, clip.y, clip.z, clip.w))
    }

    /// Submit a bounding sphere as an occluder. Writes its depth into every
    /// grid cell the sphere overlaps, keeping the *nearest* depth per cell.
    ///
    /// `cam_pos` is the camera position used to compute view-space depth.
    pub fn submit_occluder(&mut self, vp: Mat4, center: Vec3, radius: f32, cam_pos: Vec3) {
        // Occluders fully behind the camera (or behind the near plane) can't
        // hide anything we'd draw; skip them. Projecting a clipped sphere
        // would otherwise smear an AABB across the whole view and could
        // spuriously cover the screen.
        if Self::project_w(vp, center).is_none() {
            return;
        }

        // Rasterize the occluder's bounding sphere via a coarse AABB in NDC.
        let corners = [
            center + Vec3::new(-radius, -radius, -radius),
            center + Vec3::new(radius, -radius, -radius),
            center + Vec3::new(-radius, radius, -radius),
            center + Vec3::new(radius, radius, -radius),
            center + Vec3::new(-radius, -radius, radius),
            center + Vec3::new(radius, -radius, radius),
            center + Vec3::new(-radius, radius, radius),
            center + Vec3::new(radius, radius, radius),
        ];
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut any = false;
        for c in corners {
            if let Some((nx, ny, _)) = Self::project(vp, c) {
                min_x = min_x.min(nx);
                max_x = max_x.max(nx);
                min_y = min_y.min(ny);
                max_y = max_y.max(ny);
                any = true;
            }
        }
        if !any {
            return;
        }

        // Also fill the sphere's center cell even if some corners are clipped.
        if let Some((nx, ny, _)) = Self::project(vp, center) {
            min_x = min_x.min(nx);
            max_x = max_x.max(nx);
            min_y = min_y.min(ny);
            max_y = max_y.max(ny);
        }

        // Depth = distance from camera to the sphere's front face.
        let depth = (center - cam_pos).length() - radius;

        let g = self.grid as f32;
        let cx0 = (((min_x + 1.0) * 0.5) * g).floor().clamp(0.0, g - 1.0) as usize;
        let cx1 = (((max_x + 1.0) * 0.5) * g).ceil().clamp(0.0, g - 1.0) as usize;
        let cy0 = (((min_y + 1.0) * 0.5) * g).floor().clamp(0.0, g - 1.0) as usize;
        let cy1 = (((max_y + 1.0) * 0.5) * g).ceil().clamp(0.0, g - 1.0) as usize;

        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                let idx = cy * self.grid + cx;
                if depth < self.depth[idx] {
                    self.depth[idx] = depth;
                }
            }
        }
    }

    /// Conservative occlusion test for a drawable bounding sphere.
    ///
    /// Returns true (cull) only if every cell covered by the sphere's
    /// *projected extent* is already filled by an occluder nearer than the
    /// sphere's front face. If any covered cell is empty or farther, we keep
    /// it — this is what avoids visual popping. Checking the full projected
    /// extent (instead of just the centre's neighbourhood) means large
    /// drawables that peek out of the covered region survive the cull.
    pub fn is_occluded(&self, vp: Mat4, center: Vec3, radius: f32, cam_pos: Vec3) -> bool {
        // Project the sphere's bounding octahedron extent into NDC.
        let oc = [
            center + Vec3::new(-radius, -radius, -radius),
            center + Vec3::new(radius, -radius, -radius),
            center + Vec3::new(-radius, radius, -radius),
            center + Vec3::new(radius, radius, -radius),
            center + Vec3::new(-radius, -radius, radius),
            center + Vec3::new(radius, -radius, radius),
            center + Vec3::new(-radius, radius, radius),
            center + Vec3::new(radius, radius, radius),
            center, // centre cell keeps small spheres from being skipped
        ];
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut any = false;
        for c in oc {
            if let Some((nx, ny, _)) = Self::project(vp, c) {
                min_x = min_x.min(nx);
                max_x = max_x.max(nx);
                min_y = min_y.min(ny);
                max_y = max_y.max(ny);
                any = true;
            }
        }
        // Nothing on screen: frustum culling handles it, don't cull here.
        if !any {
            return false;
        }

        // Depth of the front face of this sphere.
        let depth = (center - cam_pos).length() - radius;

        let g = self.grid as f32;
        let cx0 = (((min_x + 1.0) * 0.5) * g).floor().clamp(0.0, g - 1.0) as usize;
        let cx1 = (((max_x + 1.0) * 0.5) * g).ceil().clamp(0.0, g - 1.0) as usize;
        let cy0 = (((min_y + 1.0) * 0.5) * g).floor().clamp(0.0, g - 1.0) as usize;
        let cy1 = (((max_y + 1.0) * 0.5) * g).ceil().clamp(0.0, g - 1.0) as usize;

        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                let cell_depth = self.depth[cy * self.grid + cx];
                // Cell empty, or the occluder stored is farther than our front
                // face → this sphere pokes out somewhere → keep it.
                if cell_depth >= depth - DEPTH_MARGIN {
                    return false;
                }
            }
        }
        true
    }

    /// True when the grid has any occluders (used to skip work when empty).
    pub fn has_occluders(&self) -> bool {
        self.depth.iter().any(|d| *d < f32::INFINITY)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    fn camera() -> (Mat4, Vec3) {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        (proj * view, Vec3::ZERO)
    }

    #[test]
    fn empty_grid_never_occludes() {
        let mut cull = OcclusionCuller::new(32);
        cull.begin_frame();
        let (vp, cam) = camera();
        assert!(!cull.is_occluded(vp, Vec3::new(0.0, 0.0, -5.0), 1.0, cam));
        assert!(!cull.has_occluders());
    }

    #[test]
    fn occluder_hides_sphere_behind_it() {
        let mut cull = OcclusionCuller::new(32);
        cull.begin_frame();
        let (vp, cam) = camera();
        // A big wall 4 units in front (front face at distance ~1).
        cull.submit_occluder(vp, Vec3::new(0.0, 0.0, -4.0), 3.0, cam);
        assert!(cull.has_occluders());
        // A small sphere fully behind the wall should be culled.
        assert!(cull.is_occluded(vp, Vec3::new(0.0, 0.0, -8.0), 0.5, cam));
        // A sphere clearly in front of the wall's front face must NOT be culled.
        assert!(!cull.is_occluded(vp, Vec3::new(0.0, 0.0, -0.5), 0.25, cam));
    }

    #[test]
    fn sphere_peeking_outside_occluder_survives() {
        let mut cull = OcclusionCuller::new(32);
        cull.begin_frame();
        let (vp, cam) = camera();
        // Small occluder off to the left.
        cull.submit_occluder(vp, Vec3::new(-2.0, 0.0, -4.0), 0.5, cam);
        // Sphere at the right, same depth — not covered by the occluder.
        assert!(!cull.is_occluded(vp, Vec3::new(2.0, 0.0, -4.0), 0.5, cam));
    }
}