use crate::terrain::TerrainGrid;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone)]
pub struct NavGrid {
    pub width: usize,
    pub depth: usize,
    pub walkable: Vec<bool>,
    pub max_slope: f32,
    pub contour_edges: Vec<((usize, usize), (usize, usize))>,
    pub region_count: usize,
}

impl NavGrid {
    pub fn from_terrain(terrain: &TerrainGrid, max_slope: f32) -> Self {
        let mut nav = Self {
            width: terrain.width,
            depth: terrain.depth,
            walkable: vec![true; terrain.width * terrain.depth],
            max_slope,
            contour_edges: Vec::new(),
            region_count: 0,
        };
        nav.rebuild(terrain);
        nav
    }

    pub fn rebuild(&mut self, terrain: &TerrainGrid) {
        self.width = terrain.width;
        self.depth = terrain.depth;
        self.walkable.resize(self.width * self.depth, true);
        for z in 0..self.depth {
            for x in 0..self.width {
                let wx = x as f32 * terrain.cell_size - (self.width as f32 * 0.5);
                let wz = z as f32 * terrain.cell_size - (self.depth as f32 * 0.5);
                let slope = terrain.sample_slope_world(wx, wz);
                self.walkable[z * self.width + x] = slope <= self.max_slope;
            }
        }
        self.contour_edges = self.extract_contours();
        self.region_count = self.count_regions();
    }

    pub fn walkable_count(&self) -> usize {
        self.walkable.iter().filter(|x| **x).count()
    }

    pub fn find_path(&self, start: (usize, usize), goal: (usize, usize)) -> Option<Vec<(usize, usize)>> {
        if !self.in_bounds(start.0, start.1) || !self.in_bounds(goal.0, goal.1) {
            return None;
        }
        if !self.is_walkable(start.0, start.1) || !self.is_walkable(goal.0, goal.1) {
            return None;
        }
        let mut frontier = BinaryHeap::new();
        frontier.push(Node { pos: start, priority: 0.0 });
        let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut cost_so_far: HashMap<(usize, usize), f32> = HashMap::new();
        cost_so_far.insert(start, 0.0);

        while let Some(Node { pos, .. }) = frontier.pop() {
            if pos == goal {
                let mut out = vec![goal];
                let mut cur = goal;
                while let Some(prev) = came_from.get(&cur).copied() {
                    out.push(prev);
                    if prev == start {
                        break;
                    }
                    cur = prev;
                }
                out.reverse();
                return Some(out);
            }

            for next in neighbors(pos, self.width, self.depth) {
                if !self.is_walkable(next.0, next.1) {
                    continue;
                }
                let base = *cost_so_far.get(&pos).unwrap_or(&0.0);
                let new_cost = base + 1.0;
                let old = *cost_so_far.get(&next).unwrap_or(&f32::INFINITY);
                if new_cost < old {
                    cost_so_far.insert(next, new_cost);
                    let h = ((goal.0 as i32 - next.0 as i32).abs() + (goal.1 as i32 - next.1 as i32).abs()) as f32;
                    frontier.push(Node {
                        pos: next,
                        priority: -(new_cost + h),
                    });
                    came_from.insert(next, pos);
                }
            }
        }
        None
    }

    pub fn smooth_path(&self, path: &[(usize, usize)]) -> Vec<(usize, usize)> {
        if path.len() <= 2 {
            return path.to_vec();
        }
        let mut out = vec![path[0]];
        let mut i = 0usize;
        while i < path.len() - 1 {
            let mut best = i + 1;
            for j in ((i + 1)..path.len()).rev() {
                if self.line_walkable(path[i], path[j]) {
                    best = j;
                    break;
                }
            }
            out.push(path[best]);
            i = best;
        }
        out
    }

    fn in_bounds(&self, x: usize, z: usize) -> bool {
        x < self.width && z < self.depth
    }

    fn is_walkable(&self, x: usize, z: usize) -> bool {
        self.walkable[z * self.width + x]
    }

    fn extract_contours(&self) -> Vec<((usize, usize), (usize, usize))> {
        let mut edges = Vec::new();
        for z in 0..self.depth {
            for x in 0..self.width {
                if !self.is_walkable(x, z) {
                    continue;
                }
                let n = [
                    (x.wrapping_sub(1), z, x == 0),
                    (x + 1, z, x + 1 >= self.width),
                    (x, z.wrapping_sub(1), z == 0),
                    (x, z + 1, z + 1 >= self.depth),
                ];
                if n[0].2 || !self.is_walkable(n[0].0, n[0].1) {
                    edges.push(((x, z), (x, z + 1)));
                }
                if n[1].2 || !self.is_walkable(n[1].0, n[1].1) {
                    edges.push(((x + 1, z), (x + 1, z + 1)));
                }
                if n[2].2 || !self.is_walkable(n[2].0, n[2].1) {
                    edges.push(((x, z), (x + 1, z)));
                }
                if n[3].2 || !self.is_walkable(n[3].0, n[3].1) {
                    edges.push(((x, z + 1), (x + 1, z + 1)));
                }
            }
        }
        edges
    }

    fn count_regions(&self) -> usize {
        let mut seen = vec![false; self.width * self.depth];
        let mut regions = 0usize;
        for z in 0..self.depth {
            for x in 0..self.width {
                let idx = z * self.width + x;
                if seen[idx] || !self.is_walkable(x, z) {
                    continue;
                }
                regions += 1;
                let mut stack = vec![(x, z)];
                while let Some((cx, cz)) = stack.pop() {
                    let i = cz * self.width + cx;
                    if seen[i] || !self.is_walkable(cx, cz) {
                        continue;
                    }
                    seen[i] = true;
                    for nb in neighbors((cx, cz), self.width, self.depth) {
                        let ni = nb.1 * self.width + nb.0;
                        if !seen[ni] && self.is_walkable(nb.0, nb.1) {
                            stack.push(nb);
                        }
                    }
                }
            }
        }
        regions
    }

    fn line_walkable(&self, a: (usize, usize), b: (usize, usize)) -> bool {
        let mut x0 = a.0 as i32;
        let mut y0 = a.1 as i32;
        let x1 = b.0 as i32;
        let y1 = b.1 as i32;
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            if x0 < 0 || y0 < 0 {
                return false;
            }
            let ux = x0 as usize;
            let uz = y0 as usize;
            if ux >= self.width || uz >= self.depth || !self.is_walkable(ux, uz) {
                return false;
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
        true
    }
}

#[derive(Clone, Copy)]
struct Node {
    pos: (usize, usize),
    priority: f32,
}

impl Eq for Node {}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.partial_cmp(&other.priority).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn neighbors(pos: (usize, usize), w: usize, d: usize) -> [(usize, usize); 4] {
    let (x, z) = pos;
    [
        (x.saturating_sub(1), z),
        ((x + 1).min(w.saturating_sub(1)), z),
        (x, z.saturating_sub(1)),
        (x, (z + 1).min(d.saturating_sub(1))),
    ]
}
