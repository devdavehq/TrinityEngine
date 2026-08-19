//! Terrain chunk streaming and LOD system.
//!
//! The terrain is divided into a grid of [`TerrainChunk`]s, each storing a
//! heightmap at a particular level of detail. A [`ChunkGrid`] owns all chunks
//! and provides world-space sampling, sculpting brushes, and LOD management.

use std::collections::HashMap;

/// A single terrain chunk storing a square heightmap.
///
/// Coordinates are in **chunk space** — integer indices into the
/// [`ChunkGrid`]'s hashmap — not world-space positions.
#[derive(Debug, Clone)]
pub struct TerrainChunk {
    /// Chunk-space X index (column).
    pub offset_x: i32,
    /// Chunk-space Z index (row).
    pub offset_z: i32,
    /// Number of cells per side (heightmap resolution for this LOD level).
    pub size: usize,
    /// Level of detail: 0 = highest, 1 = half resolution, 2 = quarter.
    pub lod_level: u8,
    /// Height values in row-major order (`z * size + x`).
    pub heights: Vec<f32>,
    /// Whether the chunk's mesh needs to be rebuilt.
    pub dirty: bool,
}

/// A grid of [`TerrainChunk`]s forming the complete terrain surface.
#[derive(Debug, Clone)]
pub struct ChunkGrid {
    /// Number of cells per chunk side at LOD 0.
    pub chunk_size: usize,
    /// All loaded chunks keyed by `(chunk_x, chunk_z)`.
    pub chunks: HashMap<(i32, i32), TerrainChunk>,
    /// Total terrain width in cells.
    pub total_width: usize,
    /// Total terrain depth in cells.
    pub total_depth: usize,
    /// World-space size of a single cell.
    pub cell_size: f32,
}

/// Creates a new [`ChunkGrid`] populated with chunks at LOD 0.
///
/// The grid is divided into `(total_width / chunk_size) × (total_depth / chunk_size)`
/// chunks, each initialized with a flat zero heightmap.
pub fn new_grid(
    total_width: usize,
    total_depth: usize,
    chunk_size: usize,
    cell_size: f32,
) -> ChunkGrid {
    assert!(chunk_size > 0, "chunk_size must be > 0");
    assert!(total_width % chunk_size == 0, "total_width must be divisible by chunk_size");
    assert!(total_depth % chunk_size == 0, "total_depth must be divisible by chunk_size");

    let chunks_x = total_width / chunk_size;
    let chunks_z = total_depth / chunk_size;
    let mut chunks = HashMap::with_capacity(chunks_x * chunks_z);

    for cz in 0..chunks_z as i32 {
        for cx in 0..chunks_x as i32 {
            chunks.insert((cx, cz), blank_chunk(cx, cz, chunk_size));
        }
    }

    ChunkGrid {
        chunk_size,
        chunks,
        total_width,
        total_depth,
        cell_size,
    }
}

/// Creates a [`ChunkGrid`] with the same bounds as `new_grid` but with **no
/// chunks populated** — the grid knows its full extent (for bounds-checking
/// and streaming math) but starts empty, so nothing is resident in memory
/// until `terrain::streaming` loads chunks in around the player.
///
/// Use `new_grid` for terrains small enough to just keep entirely resident
/// (the common case); use this for open-world regions large enough that
/// holding every chunk in memory forever stops being reasonable.
pub fn new_grid_streamed(
    total_width: usize,
    total_depth: usize,
    chunk_size: usize,
    cell_size: f32,
) -> ChunkGrid {
    assert!(chunk_size > 0, "chunk_size must be > 0");
    assert!(total_width % chunk_size == 0, "total_width must be divisible by chunk_size");
    assert!(total_depth % chunk_size == 0, "total_depth must be divisible by chunk_size");
    ChunkGrid {
        chunk_size,
        chunks: HashMap::new(),
        total_width,
        total_depth,
        cell_size,
    }
}

/// A freshly-generated chunk at LOD 0 with a flat zero heightmap — the same
/// default `new_grid` populates every slot with, factored out so streaming
/// can create the same thing on demand for a chunk with no saved edits yet.
pub fn blank_chunk(chunk_x: i32, chunk_z: i32, chunk_size: usize) -> TerrainChunk {
    TerrainChunk {
        offset_x: chunk_x,
        offset_z: chunk_z,
        size: chunk_size,
        lod_level: 0,
        heights: vec![0.0; chunk_size * chunk_size],
        dirty: false,
    }
}

/// Returns the world-space size of one cell at the given LOD level.
fn cell_world_size(grid: &ChunkGrid, lod: u8) -> f32 {
    grid.cell_size * (1 << lod) as f32
}

/// Converts a world-space position to chunk coordinates and a local offset
/// within that chunk.
///
/// Returns `(chunk_x, chunk_z, local_x, local_z)` where the local offsets
/// are floating-point positions within the chunk's cell grid.
pub fn world_to_chunk(grid: &ChunkGrid, world_x: f32, world_z: f32) -> (i32, i32, f32, f32) {
    let cs = grid.chunk_size as f32;
    let chunk_world = cs * grid.cell_size;
    let cx = (world_x / chunk_world).floor() as i32;
    let cz = (world_z / chunk_world).floor() as i32;
    let local_x = (world_x / grid.cell_size) - (cx as f32) * cs;
    let local_z = (world_z / grid.cell_size) - (cz as f32) * cs;
    (cx, cz, local_x, local_z)
}

/// Converts a world-space position to chunk coordinates and local offset,
/// accounting for the chunk's LOD level.
fn world_to_chunk_lod(grid: &ChunkGrid, world_x: f32, world_z: f32, lod: u8) -> (i32, i32, f32, f32) {
    let cw = cell_world_size(grid, lod);
    let cs = grid.chunk_size as f32;
    let chunk_world = cs * cw;
    let cx = (world_x / chunk_world).floor() as i32;
    let cz = (world_z / chunk_world).floor() as i32;
    let local_x = (world_x / cw) - (cx as f32) * cs;
    let local_z = (world_z / cw) - (cz as f32) * cs;
    (cx, cz, local_x, local_z)
}

/// Looks up a single height value at integer cell coordinates within a chunk.
/// Returns 0.0 if the cell falls outside the loaded grid.
fn get_cell_height(grid: &ChunkGrid, cell_x: isize, cell_z: isize) -> f32 {
    if cell_x < 0 || cell_z < 0 {
        return 0.0;
    }
    let cs = grid.chunk_size as isize;
    let cx = cell_x / cs;
    let cz = cell_z / cs;
    let lx = cell_x - cx * cs;
    let lz = cell_z - cz * cs;
    match grid.chunks.get(&(cx as i32, cz as i32)) {
        Some(chunk) => chunk.heights[lz as usize * chunk.size + lx as usize],
        None => 0.0,
    }
}

/// Samples the terrain height at a world-space position using bilinear
/// interpolation across chunk boundaries.
///
/// Converts the world position to cell coordinates, then bilinearly
/// interpolates using the four surrounding cell values (each potentially
/// stored in a different chunk). Returns `0.0` for positions outside the
/// loaded terrain.
pub fn sample_height(grid: &ChunkGrid, world_x: f32, world_z: f32) -> f32 {
    let inv_cs = 1.0 / grid.cell_size;
    let cell_x_f = world_x * inv_cs;
    let cell_z_f = world_z * inv_cs;

    let x0 = cell_x_f.floor() as isize;
    let z0 = cell_z_f.floor() as isize;
    let x1 = x0 + 1;
    let z1 = z0 + 1;
    let fx = cell_x_f - cell_x_f.floor();
    let fz = cell_z_f - cell_z_f.floor();

    let h00 = get_cell_height(grid, x0, z0);
    let h10 = get_cell_height(grid, x1, z0);
    let h01 = get_cell_height(grid, x0, z1);
    let h11 = get_cell_height(grid, x1, z1);

    let h0 = h00 + (h10 - h00) * fx;
    let h1 = h01 + (h11 - h01) * fx;
    h0 + (h1 - h0) * fz
}

/// Computes the surface normal at a world-space position.
///
/// Uses central finite differences with a step of one cell. The returned
/// vector is unit-length with Y pointing up.
pub fn sample_normal(grid: &ChunkGrid, world_x: f32, world_z: f32) -> [f32; 3] {
    let step = grid.cell_size;
    let h_l = sample_height(grid, world_x - step, world_z);
    let h_r = sample_height(grid, world_x + step, world_z);
    let h_d = sample_height(grid, world_x, world_z - step);
    let h_u = sample_height(grid, world_x, world_z + step);

    let dx = h_r - h_l;
    let dz = h_u - h_d;
    let mut nx = -dx;
    let ny = 2.0 * step;
    let mut nz = -dz;
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 0.0 {
        nx /= len;
        let ny_n = ny / len;
        nz /= len;
        [nx, ny_n, nz]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Returns the slope magnitude at a world-space position.
///
/// A value of `0.0` means perfectly flat; values above `1.0` indicate steep
/// terrain. The result is the horizontal gradient magnitude normalised by
/// cell size.
pub fn sample_slope(grid: &ChunkGrid, world_x: f32, world_z: f32) -> f32 {
    let step = grid.cell_size;
    let h_l = sample_height(grid, world_x - step, world_z);
    let h_r = sample_height(grid, world_x + step, world_z);
    let h_d = sample_height(grid, world_x, world_z - step);
    let h_u = sample_height(grid, world_x, world_z + step);

    let dhdx = (h_r - h_l) / (2.0 * step);
    let dhdz = (h_u - h_d) / (2.0 * step);
    (dhdx * dhdx + dhdz * dhdz).sqrt()
}

/// Sets the height at a single world-space position and marks the
/// containing chunk dirty.
///
/// Positions landing on chunk boundaries affect the chunk whose local
/// coordinate rounds down.
pub fn set_height(grid: &mut ChunkGrid, world_x: f32, world_z: f32, height: f32) {
    let (cx, cz, lx, lz) = world_to_chunk(grid, world_x, world_z);
    if let Some(chunk) = grid.chunks.get_mut(&(cx, cz)) {
        let x = lx.round() as isize;
        let z = lz.round() as isize;
        if x >= 0 && x < chunk.size as isize && z >= 0 && z < chunk.size as isize {
            chunk.heights[z as usize * chunk.size + x as usize] = height;
            chunk.dirty = true;
        }
    }
}

/// Applies a sculpting brush that raises terrain with quadratic falloff.
///
/// The `amount` is applied at the centre and falls off to zero at the edge
/// of `radius`. All affected chunks are marked dirty.
pub fn raise_brush(grid: &mut ChunkGrid, world_x: f32, world_z: f32, radius: f32, amount: f32) {
    let radius_cells = (radius / grid.cell_size).ceil() as i32;
    for dz in -radius_cells..=radius_cells {
        for dx in -radius_cells..=radius_cells {
            let wx = world_x + dx as f32 * grid.cell_size;
            let wz = world_z + dz as f32 * grid.cell_size;
            let dist_sq = (dx as f32 * grid.cell_size).powi(2) + (dz as f32 * grid.cell_size).powi(2);
            if dist_sq > radius * radius {
                continue;
            }
            let t = dist_sq / (radius * radius);
            let falloff = 1.0 - t;
            let delta = amount * falloff;

            let (cx, cz, lx, lz) = world_to_chunk(grid, wx, wz);
            if let Some(chunk) = grid.chunks.get_mut(&(cx, cz)) {
                let x = lx.round() as isize;
                let z = lz.round() as isize;
                if x >= 0 && x < chunk.size as isize && z >= 0 && z < chunk.size as isize {
                    let idx = z as usize * chunk.size + x as usize;
                    chunk.heights[idx] += delta;
                    chunk.dirty = true;
                }
            }
        }
    }
}

/// Applies a flatten brush that blends the terrain toward `target_height`.
///
/// `blend` controls how strongly the target is applied per sample (0 = no
/// change, 1 = snap to target). Affected chunks are marked dirty.
pub fn flatten_brush(
    grid: &mut ChunkGrid,
    world_x: f32,
    world_z: f32,
    radius: f32,
    target_height: f32,
    blend: f32,
) {
    let radius_cells = (radius / grid.cell_size).ceil() as i32;

    for dz in -radius_cells..=radius_cells {
        for dx in -radius_cells..=radius_cells {
            let wx = world_x + dx as f32 * grid.cell_size;
            let wz = world_z + dz as f32 * grid.cell_size;
            let dist = ((dx as f32 * grid.cell_size).powi(2) + (dz as f32 * grid.cell_size).powi(2)).sqrt();
            if dist > radius {
                continue;
            }
            let edge_falloff = 1.0 - (dist / radius);
            let effective_blend = blend * edge_falloff;

            let (cx, cz, lx, lz) = world_to_chunk(grid, wx, wz);
            if let Some(chunk) = grid.chunks.get_mut(&(cx, cz)) {
                let x = lx.round() as isize;
                let z = lz.round() as isize;
                if x >= 0 && x < chunk.size as isize && z >= 0 && z < chunk.size as isize {
                    let idx = z as usize * chunk.size + x as usize;
                    chunk.heights[idx] += (target_height - chunk.heights[idx]) * effective_blend;
                    chunk.dirty = true;
                }
            }
        }
    }
}

/// Applies a smooth brush that averages each cell with its neighbours.
///
/// `strength` controls the blend between the original and the averaged value
/// (0 = no change, 1 = fully averaged). Affected chunks are marked dirty.
pub fn smooth_brush(grid: &mut ChunkGrid, world_x: f32, world_z: f32, radius: f32, strength: f32) {
    let radius_cells = (radius / grid.cell_size).ceil() as i32;

    // Collect target positions and their smoothed values first to avoid
    // read-after-write bias within the same brush stroke.
    let mut targets: Vec<(i32, i32, isize, isize, f32)> = Vec::new();

    for dz in -radius_cells..=radius_cells {
        for dx in -radius_cells..=radius_cells {
            let wx = world_x + dx as f32 * grid.cell_size;
            let wz = world_z + dz as f32 * grid.cell_size;
            let dist = ((dx as f32 * grid.cell_size).powi(2) + (dz as f32 * grid.cell_size).powi(2)).sqrt();
            if dist > radius {
                continue;
            }

            let (cx, cz, lx, lz) = world_to_chunk(grid, wx, wz);
            if let Some(chunk) = grid.chunks.get(&(cx, cz)) {
                let x = lx.round() as isize;
                let z = lz.round() as isize;
                if x >= 0 && x < chunk.size as isize && z >= 0 && z < chunk.size as isize {
                    // Average with 4-connected neighbours (read from original data).
                    let size = chunk.size;
                    let mut sum = chunk.heights[z as usize * size + x as usize];
                    let mut count = 1.0f32;

                    if x > 0 {
                        sum += chunk.heights[z as usize * size + (x - 1) as usize];
                        count += 1.0;
                    }
                    if x < size as isize - 1 {
                        sum += chunk.heights[z as usize * size + (x + 1) as usize];
                        count += 1.0;
                    }
                    if z > 0 {
                        sum += chunk.heights[(z - 1) as usize * size + x as usize];
                        count += 1.0;
                    }
                    if z < size as isize - 1 {
                        sum += chunk.heights[(z + 1) as usize * size + x as usize];
                        count += 1.0;
                    }

                    let avg = sum / count;
                    targets.push((cx, cz, x, z, avg));
                }
            }
        }
    }

    for (cx, cz, x, z, avg) in targets {
        if let Some(chunk) = grid.chunks.get_mut(&(cx, cz)) {
            let idx = z as usize * chunk.size + x as usize;
            chunk.heights[idx] += (avg - chunk.heights[idx]) * strength;
            chunk.dirty = true;
        }
    }
}

/// Marks all chunks within `radius` of a world-space position as dirty.
pub fn mark_dirty(grid: &mut ChunkGrid, world_x: f32, world_z: f32, radius: f32) {
    let radius_cells = (radius / grid.cell_size).ceil() as i32;

    for dz in -radius_cells..=radius_cells {
        for dx in -radius_cells..=radius_cells {
            let wx = world_x + dx as f32 * grid.cell_size;
            let wz = world_z + dz as f32 * grid.cell_size;
            let dist = ((dx as f32 * grid.cell_size).powi(2) + (dz as f32 * grid.cell_size).powi(2)).sqrt();
            if dist > radius {
                continue;
            }
            let (cx, cz, _, _) = world_to_chunk(grid, wx, wz);
            if let Some(chunk) = grid.chunks.get_mut(&(cx, cz)) {
                chunk.dirty = true;
            }
        }
    }
}

/// Sets the LOD level of every chunk based on its distance from the camera.
///
/// `lod_distances` is a slice where index 0 is the distance threshold to
/// switch from LOD 0 to LOD 1, index 1 is the threshold for LOD 2, and so
/// on. Chunks closer than `lod_distances[0]` stay at LOD 0.
pub fn update_lods(grid: &mut ChunkGrid, camera_x: f32, camera_z: f32, lod_distances: &[f32]) {
    let chunk_world = grid.chunk_size as f32 * grid.cell_size;

    for (_key, chunk) in grid.chunks.iter_mut() {
        let chunk_cx = chunk.offset_x;
        let chunk_cz = chunk.offset_z;

        // Centre of this chunk in world space.
        let cx = (chunk_cx as f32 + 0.5) * chunk_world;
        let cz = (chunk_cz as f32 + 0.5) * chunk_world;
        let dx = camera_x - cx;
        let dz = camera_z - cz;
        let dist = (dx * dx + dz * dz).sqrt();

        let mut new_lod: u8 = 0;
        for (i, &threshold) in lod_distances.iter().enumerate() {
            if dist >= threshold {
                new_lod = (i + 1) as u8;
            }
        }

        if new_lod != chunk.lod_level {
            chunk.lod_level = new_lod;
            chunk.dirty = true;
        }
    }
}

/// Returns the chunk coordinates of every chunk that is marked dirty.
pub fn dirty_chunks(grid: &ChunkGrid) -> Vec<(i32, i32)> {
    grid.chunks
        .iter()
        .filter(|(_, c)| c.dirty)
        .map(|(&(cx, cz), _)| (cx, cz))
        .collect()
}

/// Clears the dirty flag on the specified chunk.
pub fn clear_dirty(grid: &mut ChunkGrid, chunk_x: i32, chunk_z: i32) {
    if let Some(chunk) = grid.chunks.get_mut(&(chunk_x, chunk_z)) {
        chunk.dirty = false;
    }
}

/// Returns a reference to the raw height data of the specified chunk.
///
/// Useful for mesh generation. Returns `None` if the chunk is not loaded.
pub fn chunk_heights(grid: &ChunkGrid, chunk_x: i32, chunk_z: i32) -> Option<&[f32]> {
    grid.chunks
        .get(&(chunk_x, chunk_z))
        .map(|c| c.heights.as_slice())
}

/// Flattens all chunks back into a single contiguous heightmap (row-major,
/// size `total_depth × total_width`).
///
/// This is provided for legacy compatibility with code that expects a
/// monolithic heightmap.
pub fn total_heights(grid: &ChunkGrid) -> Vec<f32> {
    let mut out = vec![0.0; grid.total_width * grid.total_depth];
    let cs = grid.chunk_size;
    let chunks_x = grid.total_width / cs;
    let chunks_z = grid.total_depth / cs;

    for cz_i in 0..chunks_z {
        for cx_i in 0..chunks_x {
            if let Some(chunk) = grid.chunks.get(&(cx_i as i32, cz_i as i32)) {
                for z in 0..cs {
                    for x in 0..cs {
                        let world_x = cx_i * cs + x;
                        let world_z = cz_i * cs + z;
                        out[world_z * grid.total_width + world_x] =
                            chunk.heights[z * cs + x];
                    }
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_grid() -> ChunkGrid {
        new_grid(64, 64, 16, 1.0)
    }

    #[test]
    fn height_sampling_across_chunks() {
        let grid = &mut make_test_grid();

        // Set a height in chunk (0, 0) near its right edge.
        let wx = 15.0;
        let wz = 5.0;
        set_height(grid, wx, wz, 10.0);

        // Sample at exactly the set point.
        let h = sample_height(grid, wx, wz);
        assert!((h - 10.0).abs() < 1e-5, "expected ~10.0, got {h}");

        // Sample one cell to the right, crossing into chunk (1, 0).
        // Chunk (1, 0) is still flat (height 0), so interpolation should
        // give roughly 5.0 at the boundary.
        let h_mid = sample_height(grid, 15.5, wz);
        assert!(h_mid > 0.0 && h_mid < 10.0, "expected blended height, got {h_mid}");

        // Fully into chunk (1, 0) which is flat.
        let h_next = sample_height(grid, 17.0, wz);
        assert!((h_next).abs() < 1e-5, "expected ~0.0 in adjacent chunk, got {h_next}");
    }

    #[test]
    fn lod_updates_by_distance() {
        let grid = &mut make_test_grid();

        // Camera at origin with large thresholds — all 4x4 chunks (centers up
        // to ~79 units away) should stay at LOD 0.
        update_lods(grid, 0.0, 0.0, &[100.0, 200.0]);
        for chunk in grid.chunks.values() {
            assert_eq!(chunk.lod_level, 0);
        }

        // Camera at (40, 40) = centre of the 64x64 grid.
        // Chunk (2,2) is at distance 0 → LOD 0.
        // Chunks (1,1)/(3,3) at ~22.6 → LOD 1.
        // Chunks (0,0)/(3,0)/(0,3) at ~36-45 → LOD 2.
        update_lods(grid, 40.0, 40.0, &[15.0, 30.0]);
        let mut found_lod_1 = false;
        let mut found_lod_2 = false;
        for chunk in grid.chunks.values() {
            if chunk.lod_level == 1 {
                found_lod_1 = true;
            }
            if chunk.lod_level == 2 {
                found_lod_2 = true;
            }
        }
        assert!(found_lod_1, "expected at least one LOD-1 chunk");
        assert!(found_lod_2, "expected at least one LOD-2 chunk");
    }

    #[test]
    fn brush_operations() {
        let mut grid = make_test_grid();
        let cx = 32.0;
        let cz = 32.0;

        // raise_brush
        raise_brush(&mut grid, cx, cz, 5.0, 4.0);
        let h_center = sample_height(&mut grid, cx, cz);
        assert!(h_center > 0.0, "raise brush should increase height, got {h_center}");

        // flatten_brush — should bring it back down toward 0.
        flatten_brush(&mut grid, cx, cz, 5.0, 0.0, 1.0);
        let h_flat = sample_height(&mut grid, cx, cz);
        assert!(h_flat.abs() < 0.5, "flatten brush should bring height near target, got {h_flat}");

        // smooth_brush — should not crash and should not produce NaN.
        smooth_brush(&mut grid, cx, cz, 5.0, 0.5);
        let h_smooth = sample_height(&mut grid, cx, cz);
        assert!(h_smooth.is_finite(), "smooth brush produced non-finite value");
    }
}
