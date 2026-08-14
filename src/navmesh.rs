// src/navmesh.rs
// ──────────────────────────────────────────────────────────────────────────────
// Additive polygon navmesh generated from the terrain heightfield.
//
// WHY (vs the plain NavGrid in navigation.rs):
//   NavGrid runs A* over a binary walkable cell grid — fine for flat 2D games,
//   but it ignores real 3D surface height and produces axis-aligned "stairstep"
//   paths. A triangle navmesh is the standard 3D solution: walkable terrain is
//   triangulated, A* runs over triangle adjacency, and paths are smoothed.
//
// This module is ADDITIVE on purpose: it never replaces NavGrid (which the
// behavior-tree MoveTo node still uses). It exposes a self-contained query:
//   navmesh.find_path(from, to) -> Vec<[f32;3]>  (world-space waypoints)
// plus the mesh itself for debug rendering. Games that want triangle-level 3D
// pathing call this; legacy BT keeps using NavGrid.
//
// Triangulation: each walkable grid cell becomes two triangles (the natural
// heightfield tessellation). Cell corners inherit the real terrain height, so
// the mesh sits on the actual surface — not on a flat grid. Cell coordinates
// match NavGrid exactly (cell == 1 world unit, world origin at grid centre), so
// both systems agree on reachable space and world positions are interchangeable.
// ──────────────────────────────────────────────────────────────────────────────

use crate::navigation::NavGrid;
use crate::terrain::TerrainWorld;
use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap};

/// One triangle of the navmesh, in world space.
#[derive(Clone, Copy, Debug)]
pub struct NavTriangle {
    pub verts: [[f32; 3]; 3],
    pub center: [f32; 3],
}

/// Spatial-hash lookup over triangle centres.
///
/// WHY: `nearest_triangle` was a linear scan over every triangle on every
/// `find_path` call (plus every `is_walkable` check).  On a 48×48 grid that is
/// ~4600 triangles per query; hot agents would O(n) on every frame.  A uniform
/// grid buckets triangles by centre so the lookup only inspects the local
/// cell (and rings of neighbours if the starting cell is empty) — near O(1)
/// for a single query instead of O(triangle_count).
#[derive(Clone, Debug)]
struct SpatialHash {
    /// Grid cell size in world units. NavMesh cells are 1 world unit, so a
    /// small multiple keeps buckets dense without pathological collisions.
    cell_size: f32,
    /// Cell coords (ix, iz) → triangle indices whose centre falls in the cell.
    buckets: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialHash {
    fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            buckets: HashMap::new(),
        }
    }

    #[inline]
    fn key(&self, x: f32, z: f32) -> (i32, i32) {
        ((x / self.cell_size).floor() as i32, (z / self.cell_size).floor() as i32)
    }

    fn insert(&mut self, tri_idx: usize, center: [f32; 3]) {
        self.buckets
            .entry(self.key(center[0], center[2]))
            .or_default()
            .push(tri_idx);
    }

    /// Find the triangle whose centre is nearest to `p` in the XZ plane.
    /// Searches the exact cell first, then expands in square rings until a
    /// non-empty bucket is found (bounded by the mesh extent).
    fn nearest(&self, p: [f32; 3], triangles: &[NavTriangle]) -> Option<usize> {
        let (kx, kz) = self.key(p[0], p[2]);
        // Bounded by half the grid dimension in cells (safety ceiling).
        let max_ring = (triangles.len() as f32).sqrt() as i32 + 1;
        for ring in 0..=max_ring {
            let mut best: Option<usize> = None;
            let mut best_d = f32::INFINITY;
            for dx in -ring..=ring {
                for dz in -ring..=ring {
                    if ring > 0 && dx.abs() != ring && dz.abs() != ring {
                        continue;
                    }
                    if let Some(bucket) = self.buckets.get(&(kx + dx, kz + dz)) {
                        for &i in bucket {
                            let c = triangles[i].center;
                            let d = (p[0] - c[0]).powi(2) + (p[2] - c[2]).powi(2);
                            if d < best_d {
                                best_d = d;
                                best = Some(i);
                            }
                        }
                    }
                }
            }
            if best.is_some() {
                return best;
            }
        }
        None
    }
}

/// A polygon navmesh triangulated from a walkable heightfield.
#[derive(Clone, Debug)]
pub struct NavMesh {
    width: usize,
    depth: usize,
    triangles: Vec<NavTriangle>,
    adjacency: Vec<Vec<usize>>,
    /// Uniform-grid index over triangle centres for near-O(1) point lookup.
    spatial: SpatialHash,
    /// Per-agent cache of the triangle each agent last stood on.  Lets
    /// `find_path` reuse the previous result when an agent has barely moved
    /// instead of re-running the point lookup from scratch.
    agent_triangles: RefCell<HashMap<u64, usize>>,
}

/// World-space position of a grid cell corner (0..=width). NavGrid cells are
/// 1 world unit and the grid is centred on the origin, so this matches
/// BT `MoveTo::world_to_grid` exactly.
#[inline]
fn corner_world(ci: usize, dim: usize) -> f32 {
    ci as f32 - dim as f32 * 0.5
}

impl NavMesh {
    /// Empty mesh — use `from_terrain` after a terrain is created.  Provided so
    /// engine state can allocate a NavMesh field before the terrain exists.
    pub fn empty() -> Self {
        Self {
            width: 0,
            depth: 0,
            triangles: Vec::new(),
            adjacency: Vec::new(),
            spatial: SpatialHash::new(2.0),
            agent_triangles: RefCell::new(HashMap::new()),
        }
    }
    /// Build a triangle navmesh from the terrain world.  Reachability reuses
    /// the NavGrid's slope/height mask so the two systems agree; corner heights
    /// come from TerrainWorld so the surface follows real ground.
    pub fn from_terrain(grid: &NavGrid, terrain: &TerrainWorld) -> Self {
        let width = grid.width;
        let depth = grid.depth;
        let mut triangles: Vec<NavTriangle> = Vec::new();

        for z in 0..depth {
            for x in 0..width {
                if !grid.walkable[z * width + x] {
                    continue;
                }
                // Corner world positions (x, z at grid resolution).
                let x0 = corner_world(x, width);
                let x1 = corner_world(x + 1, width);
                let z0 = corner_world(z, depth);
                let z1 = corner_world(z + 1, depth);

                // Real surface heights from the terrain world.
                let y00 = terrain.height_at(x0, z0);
                let y10 = terrain.height_at(x1, z0);
                let y01 = terrain.height_at(x0, z1);
                let y11 = terrain.height_at(x1, z1);

                // Triangle 0: (x0,z0) (x1,z0) (x0,z1)  →  upper-left.
                triangles.push(NavTriangle {
                    verts: [[x0, y00, z0], [x1, y10, z0], [x0, y01, z1]],
                    center: [(x0 + x1) * 0.5, (y00 + y10 + y01) / 3.0, (z0 + z1) * 0.5],
                });
                // Triangle 1: (x1,z0) (x1,z1) (x0,z1)  →  lower-right.
                triangles.push(NavTriangle {
                    verts: [[x1, y10, z0], [x1, y11, z1], [x0, y01, z1]],
                    center: [(x0 + x1) * 0.5, (y10 + y11 + y01) / 3.0, (z0 + z1) * 0.5],
                });
            }
        }

        let mut mesh = NavMesh {
            width,
            depth,
            triangles,
            adjacency: Vec::new(),
            spatial: SpatialHash::new(2.0),
            agent_triangles: RefCell::new(HashMap::new()),
        };
        mesh.build_adjacency();
        mesh.build_spatial();
        mesh
    }

    /// Index every triangle centre into the uniform grid.
    fn build_spatial(&mut self) {
        let mut spatial = SpatialHash::new(2.0);
        for (i, tri) in self.triangles.iter().enumerate() {
            spatial.insert(i, tri.center);
        }
        self.spatial = spatial;
    }

    /// Link triangles that share a geometric edge.  Shared edges inside the
    /// same walkable cell plus across adjacent walkable cells.
    fn build_adjacency(&mut self) {
        let n = self.triangles.len();
        let mut adjacency = vec![Vec::new(); n];
        let mut edge_map: HashMap<(i64, i64, i64, i64, i64, i64), usize> = HashMap::new();
        let snap = |p: [f32; 3]| {
            (((p[0] * 1e3) as i64), ((p[1] * 1e3) as i64), ((p[2] * 1e3) as i64))
        };
        for (ti, tri) in self.triangles.iter().enumerate() {
            let edge_keys = [
                (snap(tri.verts[0]), snap(tri.verts[1])),
                (snap(tri.verts[1]), snap(tri.verts[2])),
                (snap(tri.verts[2]), snap(tri.verts[0])),
            ];
            for (a, b) in edge_keys {
                // Canonical undirected key.
                let key = if a <= b {
                    (a.0, a.1, a.2, b.0, b.1, b.2)
                } else {
                    (b.0, b.1, b.2, a.0, a.1, a.2)
                };
                if let Some(&other) = edge_map.get(&key) {
                    if !adjacency[ti].contains(&other) { adjacency[ti].push(other); }
                    if !adjacency[other].contains(&ti) { adjacency[other].push(ti); }
                } else {
                    edge_map.insert(key, ti);
                }
            }
        }
        self.adjacency = adjacency;
    }

    /// A* over the triangle adjacency graph from `from` to `to` (world space).
    /// Returns a smoothed polyline of world-space waypoints.
    pub fn find_path(&self, from: [f32; 3], to: [f32; 3]) -> Option<Vec<[f32; 3]>> {
        let start = self.nearest_triangle(from)?;
        let goal = self.nearest_triangle(to)?;
        if start == goal {
            return Some(vec![from, to]);
        }
        self.find_path_between(start, goal, from, to)
    }

    /// Agent-cached variant.  The caller passes a stable agent id; the mesh
    /// remembers the triangle that agent last stood on, so a nearly-stationary
    /// agent skips the point lookup entirely.  Falls back to `find_path` when
    /// the cache is cold.
    pub fn find_path_for_agent(
        &self,
        agent: u64,
        from: [f32; 3],
        to: [f32; 3],
    ) -> Option<Vec<[f32; 3]>> {
        let start = {
            let cache = self.agent_triangles.borrow();
            match cache.get(&agent) {
                Some(&idx) if idx < self.triangles.len() => idx,
                _ => self.nearest_triangle(from)?,
            }
        };
        let goal = self.nearest_triangle(to)?;
        if start == goal {
            self.agent_triangles.borrow_mut().insert(agent, goal);
            return Some(vec![from, to]);
        }
        let path = self.find_path_between(start, goal, from, to);
        if path.is_some() {
            self.agent_triangles.borrow_mut().insert(agent, goal);
        }
        path
    }

    fn find_path_between(
        &self,
        start: usize,
        goal: usize,
        from: [f32; 3],
        to: [f32; 3],
    ) -> Option<Vec<[f32; 3]>> {
        let mut frontier: BinaryHeap<AStarNode> = BinaryHeap::new();
        let mut cost: HashMap<usize, f32> = HashMap::new();
        let mut came: HashMap<usize, usize> = HashMap::new();
        frontier.push(AStarNode { tri: start, priority: 0.0 });
        cost.insert(start, 0.0);

        while let Some(AStarNode { tri, .. }) = frontier.pop() {
            if tri == goal {
                break;
            }
            let base = *cost.get(&tri).unwrap_or(&f32::INFINITY);
            for &next in &self.adjacency[tri] {
                let step = self.tri_distance(tri, next);
                let new_cost = base + step;
                let old = *cost.get(&next).unwrap_or(&f32::INFINITY);
                if new_cost < old {
                    cost.insert(next, new_cost);
                    came.insert(next, tri);
                    let h = self.tri_distance(next, goal);
                    frontier.push(AStarNode { tri: next, priority: -(new_cost + h) });
                }
            }
        }

        if !cost.contains_key(&goal) {
            return None;
        }

        let mut tri_path = vec![goal];
        let mut cur = goal;
        while let Some(&prev) = came.get(&cur) {
            tri_path.push(prev);
            if prev == start {
                break;
            }
            cur = prev;
        }
        tri_path.reverse();

        let mut pts: Vec<[f32; 3]> = Vec::with_capacity(tri_path.len() + 2);
        pts.push(from);
        for ti in &tri_path {
            pts.push(self.triangles[*ti].center);
        }
        pts.push(to);
        Some(self.funnel(&pts, &tri_path, from, to))
    }

    /// String-pulling (funnel) pass over the A* corridor.
    ///
    /// WHY: triangle-centre waypoints produce a zig-zag "stairstep" path that
    /// follows the triangulation instead of the natural line of travel.  The
    /// funnel algorithm walks the corridor portals (shared edges between
    /// consecutive triangles) and keeps the left/right funnel edges tight, so
    /// the output hugs corners only when the corridor actually forces a turn.
    fn funnel(
        &self,
        pts: &[[f32; 3]],
        tri_path: &[usize],
        from: [f32; 3],
        to: [f32; 3],
    ) -> Vec<[f32; 3]> {
        // Collect the shared-edge portals along the corridor: triangle i → i+1
        // share an edge; the funnel must pass through it.
        let mut portals: Vec<[[f32; 3]; 2]> = Vec::with_capacity(tri_path.len().saturating_sub(1));
        for w in tri_path.windows(2) {
            let p = w[0];
            let q = w[1];
            if let Some((l, r)) = self.shared_edge(p, q) {
                portals.push([l, r]);
            }
        }
        if portals.is_empty() {
            return self.smooth(pts);
        }

        let mut result: Vec<[f32; 3]> = vec![from];
        let mut apex = from;
        let mut left = from;
        let mut right = from;

        for portal in &portals {
            let portal_left = portal[0];
            let portal_right = portal[1];

            // Right turn past the current left edge → corner at `left`.
            if tri_cross(&apex, &right, &portal_right) >= 0.0 {
                if tri_cross(&apex, &left, &portal_right) < 0.0 {
                    right = portal_right;
                } else {
                    result.push(left);
                    apex = left;
                    right = apex;
                }
            }
            // Left turn past the current right edge → corner at `right`.
            if tri_cross(&apex, &left, &portal_left) <= 0.0 {
                if tri_cross(&apex, &right, &portal_left) > 0.0 {
                    left = portal_left;
                } else {
                    result.push(right);
                    apex = right;
                    left = apex;
                }
            }
        }

        let last = *result.last().unwrap_or(&from);
        let d = ((to[0] - last[0]).powi(2) + (to[1] - last[1]).powi(2) + (to[2] - last[2]).powi(2)).sqrt();
        if d > 0.01 {
            result.push(to);
        }
        if result.len() == 1 {
            result.push(to);
        }
        result
    }

    /// Shared geometric edge between two adjacent triangles (as a portal).
    /// Returns (left, right) vertices of the shared edge.
    fn shared_edge(&self, a: usize, b: usize) -> Option<([f32; 3], [f32; 3])> {
        if a == b {
            return None;
        }
        let tri_a = &self.triangles[a];
        let tri_b = &self.triangles[b];
        for i in 0..3 {
            for j in 0..3 {
                let ai = tri_a.verts[i];
                let aj = tri_a.verts[(i + 1) % 3];
                let bi = tri_b.verts[j];
                let bj = tri_b.verts[(j + 1) % 3];
                if vert_eq(ai, bi) && vert_eq(aj, bj) {
                    return Some((ai, aj));
                }
                if vert_eq(ai, bj) && vert_eq(aj, bi) {
                    return Some((ai, aj));
                }
            }
        }
        None
    }

    /// Polish: drop near-collinear/redundant interior points (keeps corners).
    fn smooth(&self, pts: &[[f32; 3]]) -> Vec<[f32; 3]> {
        if pts.len() <= 2 {
            return pts.to_vec();
        }
        let mut out = vec![pts[0]];
        let mut last = pts[0];
        for &p in pts {
            let d = ((p[0] - last[0]).powi(2) + (p[1] - last[1]).powi(2) + (p[2] - last[2]).powi(2)).sqrt();
            if d > 0.25 {
                out.push(p);
                last = p;
            }
        }
        let tail = pts[pts.len() - 1];
        if out.last().map_or(true, |l| *l != tail) {
            out.push(tail);
        }
        out
    }

    fn tri_distance(&self, a: usize, b: usize) -> f32 {
        let pa = self.triangles[a].center;
        let pb = self.triangles[b].center;
        ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt()
    }

    /// Triangle whose centre is nearest in the XZ plane to `p`.
    /// Uses the spatial hash for near-O(1) lookups instead of a full scan.
    fn nearest_triangle(&self, p: [f32; 3]) -> Option<usize> {
        self.spatial.nearest(p, &self.triangles)
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// World-space extent (grid dimension, matching NavGrid).
    pub fn extents(&self) -> (f32, f32) {
        (self.width as f32 * 0.5, self.depth as f32 * 0.5)
    }

    /// Whether the given world position is inside walkable terrain.
    pub fn is_walkable_at(&self, p: [f32; 3]) -> bool {
        self.nearest_triangle(p).is_some()
    }
}

/// Signed "orientation" test (b - a) × (c - a) in the XZ plane — the cross
/// product of two 2D vectors, used by the funnel algorithm to detect whether a
/// portal edge tightens the funnel or forces a corner.
fn tri_cross(a: &[f32; 3], b: &[f32; 3], c: &[f32; 3]) -> f32 {
    let abx = b[0] - a[0];
    let abz = b[2] - a[2];
    let acx = c[0] - a[0];
    let acz = c[2] - a[2];
    abx * acz - abz * acx
}

/// Exact vertex equality within the quantised navmesh coordinates.
fn vert_eq(a: [f32; 3], b: [f32; 3]) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
}

#[derive(Clone, Copy, PartialEq)]
struct AStarNode {
    tri: usize,
    priority: f32,
}

impl Eq for AStarNode {}
impl PartialOrd for AStarNode {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for AStarNode {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.priority.partial_cmp(&o.priority).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_triangles_from_flat_terrain() {
        let tw = TerrainWorld::new(48, 48, 16, 1.0);
        let grid = NavGrid { width: 48, depth: 48, walkable: vec![true; 48 * 48], max_slope: 0.8, contour_edges: Vec::new(), region_count: 0 };
        let mesh = NavMesh::from_terrain(&grid, &tw);
        assert!(mesh.triangle_count() > 0, "expected some triangles");
    }

    #[test]
    fn find_path_across_open_mesh() {
        let tw = TerrainWorld::new(48, 48, 16, 1.0);
        let grid = NavGrid { width: 48, depth: 48, walkable: vec![true; 48 * 48], max_slope: 0.8, contour_edges: Vec::new(), region_count: 0 };
        let mesh = NavMesh::from_terrain(&grid, &tw);
        let path = mesh.find_path([-20.0, 0.0, -20.0], [20.0, 0.0, 20.0]);
        assert!(path.is_some(), "expected a path across open terrain");
        let p = path.unwrap();
        assert!(p.len() >= 2, "path should have at least start+end");
    }
}