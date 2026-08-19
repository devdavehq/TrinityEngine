// src/terrain/streaming.rs
// ──────────────────────────────────────────────────────────────────────────────
// Distance-based terrain chunk streaming.
//
// `chunk::ChunkGrid` normally holds every chunk in memory for the whole life
// of the terrain (see `chunk::new_grid`) — fine for a bounded region, but not
// for an open world large enough that "every chunk, forever" stops being a
// reasonable memory budget. This module is the opt-in layer for that case:
// build the grid with `chunk::new_grid_streamed` (same bounds, zero chunks
// resident), then call `check_chunk_streaming` + `apply_streaming` each tick
// to load chunks in around the player and evict ones left behind.
//
// This mirrors `levels/streaming.rs`'s hysteresis pattern on purpose: a load
// radius and a strictly larger unload radius create a dead zone so a chunk
// sitting near the boundary doesn't flicker in and out every frame as the
// player walks back and forth across it. Checks are throttled by
// `check_interval`, same as level streaming.
//
// Edits matter, so unloading a chunk isn't just dropping it — it's saved to
// disk first (raw f32 heights) and reloaded from there later if present,
// falling back to a fresh flat chunk only the first time a chunk is ever
// visited. Placed props/foliage within a chunk are a separate concern (they
// live as ECS entities, not in the heightmap) and aren't handled here.
// ──────────────────────────────────────────────────────────────────────────────

use super::chunk::{blank_chunk, ChunkGrid, TerrainChunk};

/// Hysteresis configuration for chunk streaming, in **chunk units** (not
/// world units) so it stays correct regardless of `chunk_size`/`cell_size`.
pub struct TerrainStreamingConfig {
    /// Chunks within this radius of the player's chunk should be resident.
    pub load_radius_chunks: f32,
    /// Chunks beyond this radius should be evicted. Must be strictly larger
    /// than `load_radius_chunks` — that gap is the hysteresis band.
    pub unload_radius_chunks: f32,
    /// Check interval in seconds (checks are throttled, not per-frame).
    pub check_interval: f32,
    /// Internal timer — resets each time a check runs.
    pub timer: f32,
}

impl TerrainStreamingConfig {
    pub fn new(load_radius_chunks: f32, unload_radius_chunks: f32) -> Self {
        let load_radius_chunks = load_radius_chunks.max(0.0);
        Self {
            load_radius_chunks,
            unload_radius_chunks: unload_radius_chunks.max(load_radius_chunks + 0.001),
            check_interval: 0.5,
            timer: 0.0,
        }
    }
}

impl Default for TerrainStreamingConfig {
    fn default() -> Self {
        Self::new(4.0, 6.0)
    }
}

/// Result of a streaming check — chunk coordinates that need loading/unloading.
pub struct ChunkStreamingResult {
    pub chunks_to_load: Vec<(i32, i32)>,
    pub chunks_to_unload: Vec<(i32, i32)>,
}

/// Check which chunks should load/unload based on the player's world-space
/// position. Returns `None` if it's not time to check yet (throttled by
/// `check_interval`) or nothing changed.
pub fn check_chunk_streaming(
    grid: &ChunkGrid,
    player_x: f32,
    player_z: f32,
    dt: f32,
    config: &mut TerrainStreamingConfig,
) -> Option<ChunkStreamingResult> {
    config.timer += dt;
    if config.timer < config.check_interval {
        return None;
    }
    config.timer = 0.0;

    if grid.chunk_size == 0 {
        return None;
    }
    let chunk_world = grid.chunk_size as f32 * grid.cell_size;
    let chunks_x = (grid.total_width / grid.chunk_size) as i32;
    let chunks_z = (grid.total_depth / grid.chunk_size) as i32;
    let pcx = (player_x / chunk_world).floor();
    let pcz = (player_z / chunk_world).floor();

    let mut to_unload = Vec::new();
    for &(cx, cz) in grid.chunks.keys() {
        let dist = chunk_distance(pcx, pcz, cx, cz);
        if dist > config.unload_radius_chunks {
            to_unload.push((cx, cz));
        }
    }

    let mut to_load = Vec::new();
    let r = config.load_radius_chunks.ceil() as i32;
    let (pcx_i, pcz_i) = (pcx as i32, pcz as i32);
    for cz in (pcz_i - r)..=(pcz_i + r) {
        if cz < 0 || cz >= chunks_z {
            continue;
        }
        for cx in (pcx_i - r)..=(pcx_i + r) {
            if cx < 0 || cx >= chunks_x {
                continue;
            }
            if grid.chunks.contains_key(&(cx, cz)) {
                continue;
            }
            if chunk_distance(pcx, pcz, cx, cz) <= config.load_radius_chunks {
                to_load.push((cx, cz));
            }
        }
    }

    if to_load.is_empty() && to_unload.is_empty() {
        None
    } else {
        Some(ChunkStreamingResult { chunks_to_load: to_load, chunks_to_unload: to_unload })
    }
}

fn chunk_distance(pcx: f32, pcz: f32, cx: i32, cz: i32) -> f32 {
    // Distance from the player's fractional chunk position to the nearest
    // point of chunk (cx, cz)'s unit footprint [cx, cx+1) x [cz, cz+1) —
    // standard point-to-AABB distance. A chunk the player is standing inside
    // is distance 0, which is what makes the load radius feel right at
    // small values (radius 1 still loads the ring around the player's chunk).
    let closest_x = pcx.clamp(cx as f32, cx as f32 + 1.0);
    let closest_z = pcz.clamp(cz as f32, cz as f32 + 1.0);
    let dx = pcx - closest_x;
    let dz = pcz - closest_z;
    (dx * dx + dz * dz).sqrt()
}

/// Apply a streaming result: save-and-evict unloaded chunks, load-or-create
/// loaded chunks. `save_dir` is a VFS directory path (e.g.
/// `"Content/terrain_chunks"`) chunk edits persist under between sessions.
pub fn apply_streaming(grid: &mut ChunkGrid, result: &ChunkStreamingResult, save_dir: &str) {
    for &(cx, cz) in &result.chunks_to_unload {
        if let Some(chunk) = grid.chunks.remove(&(cx, cz)) {
            if let Err(e) = save_chunk(save_dir, &chunk) {
                tracing::warn!("[TerrainStreaming] Failed to save chunk ({}, {}): {}", cx, cz, e);
            }
        }
    }
    for &(cx, cz) in &result.chunks_to_load {
        let chunk = load_chunk(save_dir, cx, cz, grid.chunk_size)
            .unwrap_or_else(|| blank_chunk(cx, cz, grid.chunk_size));
        grid.chunks.insert((cx, cz), chunk);
    }
}

fn chunk_path(save_dir: &str, cx: i32, cz: i32) -> String {
    format!("{}/chunk_{}_{}.bin", save_dir.trim_end_matches('/'), cx, cz)
}

/// Persist a chunk's heightmap as raw little-endian f32 bytes.
pub fn save_chunk(save_dir: &str, chunk: &TerrainChunk) -> Result<(), String> {
    let path = chunk_path(save_dir, chunk.offset_x, chunk.offset_z);
    let bytes: &[u8] = bytemuck::cast_slice(&chunk.heights);
    crate::vfs::write(&path, bytes).map_err(|e| e.to_string())
}

/// Load a previously-saved chunk, or `None` if it was never saved (a brand
/// new, never-edited region — the caller should fall back to `blank_chunk`).
pub fn load_chunk(save_dir: &str, cx: i32, cz: i32, chunk_size: usize) -> Option<TerrainChunk> {
    let path = chunk_path(save_dir, cx, cz);
    if !crate::vfs::exists(&path) {
        return None;
    }
    let bytes = crate::vfs::read(&path).ok()?;
    let expected_len = chunk_size * chunk_size * std::mem::size_of::<f32>();
    if bytes.len() != expected_len {
        tracing::warn!(
            "[TerrainStreaming] Chunk ({}, {}) save has {} bytes, expected {} for chunk_size {}; discarding.",
            cx, cz, bytes.len(), expected_len, chunk_size
        );
        return None;
    }
    let heights: &[f32] = bytemuck::cast_slice(&bytes);
    Some(TerrainChunk {
        offset_x: cx,
        offset_z: cz,
        size: chunk_size,
        lod_level: 0,
        heights: heights.to_vec(),
        dirty: true, // freshly loaded -> mesh needs (re)building.
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::chunk;

    fn empty_grid() -> ChunkGrid {
        // 8x8 chunks of size 16, cell_size 1.0 -> each chunk is 16 world units.
        chunk::new_grid_streamed(128, 128, 16, 1.0)
    }

    /// Same convention as `save_slots.rs`'s `tmp_slots`: a per-process,
    /// per-test OS temp directory so parallel test runs never collide and
    /// nothing gets written into the actual project tree.
    fn tmp_dir(tag: &str) -> String {
        let dir = std::env::temp_dir().join(format!("trinity_terrain_streaming_{}_{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);
        dir.to_string_lossy().to_string()
    }

    #[test]
    fn load_trigger_finds_nearby_unloaded_chunks() {
        let grid = empty_grid();
        let mut config = TerrainStreamingConfig::new(1.0, 2.0);
        config.check_interval = 0.0;

        // Player standing at world origin -> chunk (0,0) should want loading.
        let result = check_chunk_streaming(&grid, 0.0, 0.0, 1.0, &mut config).unwrap();
        assert!(result.chunks_to_load.contains(&(0, 0)));
        assert!(result.chunks_to_unload.is_empty());
    }

    #[test]
    fn unload_trigger_evicts_far_chunks() {
        let mut grid = empty_grid();
        grid.chunks.insert((0, 0), chunk::blank_chunk(0, 0, grid.chunk_size));
        // Far corner of the grid: chunk (7,7) center is ~120 units away ->
        // ~7.5 chunk-units, well past a small unload radius.
        let mut config = TerrainStreamingConfig::new(1.0, 2.0);
        config.check_interval = 0.0;

        let result = check_chunk_streaming(&grid, 112.0, 112.0, 1.0, &mut config).unwrap();
        assert!(result.chunks_to_unload.contains(&(0, 0)));
    }

    #[test]
    fn hysteresis_band_holds_a_chunk_between_thresholds() {
        let mut grid = empty_grid();
        grid.chunks.insert((3, 0), chunk::blank_chunk(3, 0, grid.chunk_size));
        // load_radius=1 (won't re-trigger load), unload_radius=5 (won't evict
        // either) -> a chunk sitting in the gap does nothing, same as the
        // level-streaming hysteresis test.
        let mut config = TerrainStreamingConfig::new(1.0, 5.0);
        config.check_interval = 0.0;

        let result = check_chunk_streaming(&grid, 0.0, 0.0, 1.0, &mut config);
        if let Some(r) = result {
            assert!(!r.chunks_to_unload.contains(&(3, 0)));
        }
    }

    #[test]
    fn throttled_check_returns_none_before_interval_elapses() {
        let grid = empty_grid();
        let mut config = TerrainStreamingConfig::new(1.0, 2.0);
        config.check_interval = 1.0;
        assert!(check_chunk_streaming(&grid, 0.0, 0.0, 0.1, &mut config).is_none());
    }

    #[test]
    fn save_and_load_chunk_roundtrip_preserves_edits() {
        let dir = tmp_dir("roundtrip");
        let mut c = chunk::blank_chunk(2, 3, 4);
        for (i, h) in c.heights.iter_mut().enumerate() {
            *h = i as f32 * 0.5;
        }
        save_chunk(&dir, &c).unwrap();

        let loaded = load_chunk(&dir, 2, 3, 4).expect("chunk should have been saved");
        assert_eq!(loaded.heights, c.heights);
        assert_eq!(loaded.offset_x, 2);
        assert_eq!(loaded.offset_z, 3);
    }

    #[test]
    fn load_missing_chunk_returns_none() {
        let dir = tmp_dir("never_saved");
        assert!(load_chunk(&dir, 9, 9, 16).is_none());
    }

    #[test]
    fn apply_streaming_unloads_then_reloads_same_edits() {
        let dir = tmp_dir("apply_roundtrip");
        let mut grid = empty_grid();
        let mut edited = chunk::blank_chunk(1, 1, grid.chunk_size);
        edited.heights[0] = 42.0;
        grid.chunks.insert((1, 1), edited);

        apply_streaming(&mut grid, &ChunkStreamingResult {
            chunks_to_load: vec![],
            chunks_to_unload: vec![(1, 1)],
        }, &dir);
        assert!(!grid.chunks.contains_key(&(1, 1)));

        apply_streaming(&mut grid, &ChunkStreamingResult {
            chunks_to_load: vec![(1, 1)],
            chunks_to_unload: vec![],
        }, &dir);
        assert_eq!(grid.chunks.get(&(1, 1)).unwrap().heights[0], 42.0);
    }

    #[test]
    fn apply_streaming_creates_blank_chunk_when_never_saved() {
        let dir = tmp_dir("never_saved_blank");
        let mut grid = empty_grid();
        apply_streaming(&mut grid, &ChunkStreamingResult {
            chunks_to_load: vec![(5, 5)],
            chunks_to_unload: vec![],
        }, &dir);
        let c = grid.chunks.get(&(5, 5)).unwrap();
        assert!(c.heights.iter().all(|&h| h == 0.0));
    }
}
