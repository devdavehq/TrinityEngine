//! Hydraulic and thermal erosion simulation for terrain heightmaps.
//!
//! This module provides physically-inspired erosion algorithms that sculpt
//! raw procedural terrain into more realistic landforms. Two erosion passes
//! are available:
//!
//! * **Hydraulic erosion** simulates rainfall, surface water flow, sediment
//!   transport, and deposition. Water accumulates on the terrain, flows
//!   downhill to the lowest neighbouring cell, picks up sediment where it
//!   moves quickly, and deposits sediment where it slows. This carves
//!   river valleys, smooths slopes, and creates alluvial fans at the base
//!   of mountains.
//!
//! * **Thermal erosion** (also called mass-wasting) redistributes material
//!   from steep slopes to shallower ones, mimicking soil creep and
//!   freeze-thaw weathering. It caps slopes at a configurable angle of
//!   repose and produces natural-looking scree slopes and rounded ridges.
//!
//! # Algorithm Overview
//!
//! ## Hydraulic Erosion
//!
//! Each iteration performs the following steps on every cell:
//!
//! 1. **Rain** — A small amount of water (`rain_rate`) is added to every
//!    cell, simulating uniform precipitation.
//!
//! 2. **Flow** — Water flows from each cell to its lowest neighbour that
//!    sits below the combined height + water level. The flow amount is
//!    proportional to the height difference, bounded by the available
//!    water. This is a simplified pipe-model that produces convincing
//!    river networks without solving the full shallow-water equations.
//!
//! 3. **Erode** — Where the combined water + height is significantly
//!    lower than the source (i.e. water is accelerating), terrain material
//!    is removed and added to the cell's sediment payload. The erosion
//!    rate scales with `hydraulic_strength` and the height differential.
//!
//! 4. **Deposit** — Where the water velocity is low (the destination is
//!    close in height to the source), sediment is dropped back onto the
//!    terrain. This prevents infinite sediment transport and creates
//!    depositional landforms.
//!
//! 5. **Evaporate** — A fraction (`evaporation_rate`) of water is removed,
//!    ensuring that water does not accumulate indefinitely.
//!
//! ## Thermal Erosion
//!
//! Each iteration performs the following on every cell:
//!
//! 1. For each cell, compute the steepest downward slope to any of its
//!    four cardinal neighbours. Slope is measured as the height difference
//!    divided by `cell_size`.
//!
//! 2. If the slope exceeds `thermal_angle_of_repose`, material is moved
//!    from the higher cell to the lower cell. The transfer amount is
//!    proportional to `(slope - angle_of_repose) * thermal_strength`.
//!
//! 3. A temporary buffer accumulates all transfers simultaneously, then
//!    is applied after the full pass. This avoids order-dependent
//!    artefacts where cells processed earlier bias the results.
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::terrain::erosion::{ErosionConfig, erode};
//!
//! let config = ErosionConfig::default();
//! erode(&mut heightmap, width, depth, cell_size, &config);
//! ```

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the erosion simulation.
///
/// All numeric fields have sensible defaults that produce visually
/// reasonable results on a standard terrain grid. See the individual
/// field docs for guidelines on tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErosionConfig {
    /// Number of hydraulic erosion iterations to run per `erode` call.
    /// More iterations produce deeper valleys and more pronounced river
    /// networks, but increase computation time linearly.
    /// **Default:** 50
    pub hydraulic_iterations: u32,

    /// Number of thermal erosion iterations to run per `erode` call.
    /// More iterations allow thermal processes to round steeper slopes.
    /// **Default:** 30
    pub thermal_iterations: u32,

    /// Controls how aggressively hydraulic erosion removes terrain
    /// material per flow step. Higher values carve deeper channels more
    /// quickly but can produce unnatural-looking gouges.
    /// **Default:** 0.01
    pub hydraulic_strength: f32,

    /// Controls how aggressively thermal erosion moves material per step.
    /// Higher values flatten slopes faster.
    /// **Default:** 0.02
    pub thermal_strength: f32,

    /// Maximum stable slope angle in radians. Slopes steeper than this
    /// will have material eroded downward. The default of ~0.7 radians
    /// is approximately 40 degrees, which is typical for loose soil and
    /// gravel.
    /// **Default:** 0.7
    pub thermal_angle_of_repose: f32,

    /// Amount of water height added to every cell each hydraulic
    /// iteration, simulating uniform rainfall. Higher values produce
    /// wetter terrain with more erosion.
    /// **Default:** 0.001
    pub rain_rate: f32,

    /// Fraction of water lost to evaporation each hydraulic step.
    /// A value of 0.01 means 1% of water evaporates per iteration.
    /// **Default:** 0.01
    pub evaporation_rate: f32,

    /// Rate at which sediment is deposited back onto the terrain when
    /// water velocity is low. Higher values cause faster sediment drop-
    /// off, building up alluvial fans and filling basins.
    /// **Default:** 0.015
    pub deposition_rate: f32,

    /// Maximum amount of sediment a water cell can carry. When sediment
    /// exceeds this capacity the excess is deposited. Higher values allow
    /// water to transport more material before depositing.
    /// **Default:** 0.1
    pub sediment_capacity: f32,
}

impl Default for ErosionConfig {
    fn default() -> Self {
        Self {
            hydraulic_iterations: 50,
            thermal_iterations: 30,
            hydraulic_strength: 0.01,
            thermal_strength: 0.02,
            thermal_angle_of_repose: 0.7,
            rain_rate: 0.001,
            evaporation_rate: 0.01,
            deposition_rate: 0.015,
            sediment_capacity: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Erosion state
// ---------------------------------------------------------------------------

/// Mutable state carried across hydraulic erosion iterations.
///
/// Maintains per-cell water depth and sediment load so that material
/// picked up in one iteration can be deposited in a later one.
#[derive(Debug, Clone)]
pub struct ErosionState {
    /// Water depth at each cell (length = width * depth).
    pub water: Vec<f32>,
    /// Suspended sediment at each cell (length = width * depth).
    pub sediment: Vec<f32>,
}

impl ErosionState {
    /// Creates a new zeroed erosion state for a grid of the given
    /// dimensions.
    pub fn new_state(width: usize, depth: usize) -> Self {
        let len = width * depth;
        Self {
            water: vec![0.0; len],
            sediment: vec![0.0; len],
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the height at grid position `(x, z)`.
///
/// Coordinates are clamped to the grid bounds, so callers never need to
/// worry about out-of-bounds access. This is intentional: edge cells
/// are treated as having the height of the nearest valid cell, which
/// prevents artificial cliffs at terrain borders.
pub fn height_at(heights: &[f32], width: usize, x: usize, z: usize) -> f32 {
    let x = x.min(width - 1);
    let z = z.min((heights.len() / width).saturating_sub(1));
    heights[z * width + x]
}

/// Sets the height at grid position `(x, z)`.
///
/// Does nothing if `(x, z)` is outside the grid.
pub fn set_height(heights: &mut [f32], width: usize, x: usize, z: usize, val: f32) {
    let depth = heights.len() / width;
    if x < width && z < depth {
        heights[z * width + x] = val;
    }
}

// ---------------------------------------------------------------------------
// Hydraulic erosion
// ---------------------------------------------------------------------------

/// Runs hydraulic erosion on a heightmap.
///
/// This function simulates rainfall, surface water flow, sediment
/// transport, and deposition over the given number of iterations. It
/// modifies the heightmap in place and also updates the erosion state
/// so that sediment and water persist between calls (useful for
/// incremental editing workflows).
///
/// # Algorithm detail
///
/// For each iteration:
///
/// 1. **Rainfall** — Every cell gains `rain_rate` water.
///
/// 2. **Flow** — For every cell the combined height + water level is
///    compared against the four cardinal neighbours. Water flows to the
///    lowest neighbour proportionally to the height difference. The total
///    outflow from a cell is clamped to its available water so we never
///    drain more water than exists.
///
/// 3. **Erode** — At each destination cell, if the combined height + water
///    at the source is significantly higher than at the destination (water
///    is accelerating), terrain material is eroded proportional to
///    `hydraulic_strength * height_diff`. Eroded material is added to the
///    destination cell's sediment.
///
/// 4. **Deposit** — Where the water velocity is low (source and
///    destination are close in height), sediment is deposited back onto
///    the terrain at `deposition_rate`, capped by the available sediment.
///
/// 5. **Evaporate** — Each cell loses `evaporation_rate` fraction of its
///    water.
///
/// All per-cell transfers within a single iteration are accumulated into
/// temporary buffers and applied atomically at the end of the iteration,
/// preventing scan-order bias.
pub fn simulate_hydraulic(
    heights: &mut [f32],
    state: &mut ErosionState,
    width: usize,
    depth: usize,
    _cell_size: f32,
    config: &ErosionConfig,
    iterations: u32,
) {
    let len = width * depth;
    assert_eq!(
        heights.len(),
        len,
        "heightmap length ({}) does not match width*depth ({})",
        heights.len(),
        len
    );
    assert_eq!(
        state.water.len(),
        len,
        "state.water length does not match width*depth"
    );
    assert_eq!(
        state.sediment.len(),
        len,
        "state.sediment length does not match width*depth"
    );

    let mut delta_water = vec![0.0f32; len];
    let mut delta_height = vec![0.0f32; len];
    let mut delta_sediment = vec![0.0f32; len];

    // Cardinal neighbour offsets: +x, -x, +z, -z
    let neighbour_offsets: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    for _ in 0..iterations {
        delta_water.iter_mut().for_each(|v| *v = 0.0);
        delta_height.iter_mut().for_each(|v| *v = 0.0);
        delta_sediment.iter_mut().for_each(|v| *v = 0.0);

        // 1. Rain — add water to every cell
        for i in 0..len {
            delta_water[i] += config.rain_rate;
        }

        // 2. Flow + Erode + Deposit — per-cell, accumulative
        for z in 0..depth {
            for x in 0..width {
                let i = z * width + x;

                let combined_here = heights[i] + state.water[i];

                // Find the lowest neighbour (below us in combined height)
                let mut lowest_combined = combined_here;
                let mut lowest_idx: Option<usize> = None;

                for &(dx, dz) in &neighbour_offsets {
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;
                    if nx < 0 || nx >= width as i32 || nz < 0 || nz >= depth as i32 {
                        continue;
                    }
                    let ni = nz as usize * width + nx as usize;
                    let combined_neighbour = heights[ni] + state.water[ni];
                    if combined_neighbour < lowest_combined {
                        lowest_combined = combined_neighbour;
                        lowest_idx = Some(ni);
                    }
                }

                if let Some(ni) = lowest_idx {
                    let height_diff = combined_here - lowest_combined;
                    // Flow proportional to height difference, clamped to available water
                    let flow = (height_diff * config.hydraulic_strength * 10.0)
                        .min(state.water[i])
                        .max(0.0);

                    if flow > 1e-8 {
                        // Move water from source to destination
                        delta_water[i] -= flow;
                        delta_water[ni] += flow;

                        // Transport proportional share of sediment with the flow
                        let sediment_ratio = if state.water[i] > 1e-8 {
                            flow / state.water[i]
                        } else {
                            0.0
                        };
                        let sediment_transfer = state.sediment[i] * sediment_ratio;
                        delta_sediment[i] -= sediment_transfer;
                        delta_sediment[ni] += sediment_transfer;

                        // Erode: where water accelerates (significant height drop),
                        // remove terrain material and add it to the destination's
                        // sediment payload
                        if height_diff > 0.05 {
                            let erode_amount =
                                config.hydraulic_strength * height_diff * flow.min(1.0);
                            let erode_amount = erode_amount.min(heights[i]);
                            delta_height[i] -= erode_amount;
                            delta_sediment[ni] += erode_amount;
                        }
                    }

                    // Deposit: where water velocity is low (source and dest close
                    // in height), drop sediment back onto the terrain
                    if height_diff < 0.05 {
                        let deposit = (state.sediment[i] * config.deposition_rate)
                            .min(state.sediment[i]);
                        if deposit > 1e-8 {
                            delta_height[i] += deposit;
                            delta_sediment[i] -= deposit;
                        }
                    }
                }
            }
        }

        // 3. Apply deltas
        for i in 0..len {
            state.water[i] += delta_water[i];
            if state.water[i] < 0.0 {
                state.water[i] = 0.0;
            }

            state.sediment[i] += delta_sediment[i];
            if state.sediment[i] < 0.0 {
                state.sediment[i] = 0.0;
            }

            heights[i] += delta_height[i];
            if heights[i] < 0.0 {
                heights[i] = 0.0;
            }
        }

        // Deposit excess sediment above capacity
        for i in 0..len {
            if state.sediment[i] > config.sediment_capacity {
                let excess = state.sediment[i] - config.sediment_capacity;
                state.sediment[i] = config.sediment_capacity;
                heights[i] += excess;
            }
        }

        // 4. Evaporate
        for i in 0..len {
            state.water[i] *= 1.0 - config.evaporation_rate;
            if state.water[i] < 1e-8 {
                state.water[i] = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Thermal erosion
// ---------------------------------------------------------------------------

/// Runs thermal (mass-wasting) erosion on a heightmap.
///
/// Thermal erosion simulates the gradual downhill movement of soil and
/// rock driven by gravity — processes such as soil creep, freeze-thaw
/// weathering, and small landslides. It is often used in combination
/// with hydraulic erosion to produce more natural-looking terrain: while
/// hydraulic erosion carves channels, thermal erosion rounds off the
/// sharp ridges and steep slopes that hydraulic erosion can leave
/// behind.
///
/// # Algorithm detail
///
/// For each iteration:
///
/// 1. For every cell, compute the slope to each of the four cardinal
///    neighbours. Slope is `max(0, height_diff / cell_size)` where
///    `height_diff` is the difference between this cell and the lower
///    neighbour.
///
/// 2. Find the steepest downward slope. If it exceeds
///    `thermal_angle_of_repose`, compute a transfer amount:
///
///    ```text
///    transfer = (max_slope - angle_of_repose) * thermal_strength
///    ```
///
///    The transfer is clamped so that the source cell is not lowered
///    below the destination cell.
///
/// 3. All transfers are accumulated in a temporary buffer. After the
///    full grid has been scanned, the buffer is applied to the
///    heightmap. This two-pass approach avoids order-dependent artefacts
///    where cells processed earlier bias subsequent calculations.
pub fn simulate_thermal(
    heights: &mut [f32],
    width: usize,
    depth: usize,
    cell_size: f32,
    config: &ErosionConfig,
    iterations: u32,
) {
    let len = width * depth;
    assert_eq!(
        heights.len(),
        len,
        "heightmap length ({}) does not match width*depth ({})",
        heights.len(),
        len
    );

    let mut delta = vec![0.0f32; len];
    let neighbour_offsets: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    for _ in 0..iterations {
        delta.iter_mut().for_each(|v| *v = 0.0);

        for z in 0..depth {
            for x in 0..width {
                let i = z * width + x;
                let h_here = heights[i];

                let mut max_slope = 0.0f32;
                let mut slope_neighbour: Option<usize> = None;

                for &(dx, dz) in &neighbour_offsets {
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;
                    if nx < 0 || nx >= width as i32 || nz < 0 || nz >= depth as i32 {
                        continue;
                    }
                    let ni = nz as usize * width + nx as usize;
                    let diff = h_here - heights[ni];
                    if diff > 0.0 {
                        let slope = diff / cell_size;
                        if slope > max_slope {
                            max_slope = slope;
                            slope_neighbour = Some(ni);
                        }
                    }
                }

                if let Some(ni) = slope_neighbour {
                    if max_slope > config.thermal_angle_of_repose {
                        let diff = max_slope - config.thermal_angle_of_repose;
                        let transfer = (diff * config.thermal_strength * cell_size)
                            .min((h_here - heights[ni]) * 0.5);
                        if transfer > 1e-8 {
                            delta[i] -= transfer;
                            delta[ni] += transfer;
                        }
                    }
                }
            }
        }

        // Apply accumulated deltas
        for i in 0..len {
            heights[i] += delta[i];
            if heights[i] < 0.0 {
                heights[i] = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Runs a complete erosion pass (hydraulic then thermal) on a heightmap.
///
/// This is the simplest way to apply erosion. It creates a fresh
/// [`ErosionState`], runs hydraulic erosion for the configured number of
/// iterations, then runs thermal erosion for its configured number of
/// iterations.
///
/// For incremental or region-based erosion (e.g. after a terrain editing
/// brush stroke), use [`erode_region`] instead, which allows you to
/// supply your own persistent [`ErosionState`] and restrict the affected
/// area.
pub fn erode(
    heights: &mut [f32],
    width: usize,
    depth: usize,
    cell_size: f32,
    config: &ErosionConfig,
) {
    let mut state = ErosionState::new_state(width, depth);
    simulate_hydraulic(
        heights,
        &mut state,
        width,
        depth,
        cell_size,
        config,
        config.hydraulic_iterations,
    );
    simulate_thermal(
        heights,
        width,
        depth,
        cell_size,
        config,
        config.thermal_iterations,
    );
}

/// Erodes only a rectangular sub-region of the heightmap.
///
/// This is useful when the terrain is being edited interactively — for
/// example, when a sculpting brush modifies a patch of terrain and only
/// that patch needs re-erosion. The region is defined by an inclusive
/// bounding box `[min_x, max_x) x [min_z, max_z)`.
///
/// The supplied [`ErosionState`] persists between calls, so water and
/// sediment carry over from previous edits. This produces smoother
/// results than starting from a zeroed state each time.
///
/// Hydraulic erosion is applied within the region (with a small border
/// so that flow at the edges is handled correctly), followed by thermal
/// erosion across the full heightmap (thermal erosion is cheap and
/// benefits from global context).
pub fn erode_region(
    heights: &mut [f32],
    state: &mut ErosionState,
    width: usize,
    depth: usize,
    cell_size: f32,
    config: &ErosionConfig,
    min_x: usize,
    max_x: usize,
    min_z: usize,
    max_z: usize,
) {
    // Clamp region to grid bounds
    let min_x = min_x.min(width - 1);
    let max_x = max_x.min(width);
    let min_z = min_z.min(depth - 1);
    let max_z = max_z.min(depth);

    if min_x >= max_x || min_z >= max_z {
        return;
    }

    // For hydraulic erosion we simulate a padded region so that flow at
    // the boundary has neighbouring context.
    let pad = 2usize;
    let h_min_x = min_x.saturating_sub(pad);
    let h_max_x = (max_x + pad).min(width);
    let h_min_z = min_z.saturating_sub(pad);
    let h_max_z = (max_z + pad).min(depth);

    // Create a temporary sub-heightmap and state for the padded region
    let r_width = h_max_x - h_min_x;
    let r_depth = h_max_z - h_min_z;
    let r_len = r_width * r_depth;

    let mut sub_heights = vec![0.0f32; r_len];
    let mut sub_state = ErosionState::new_state(r_width, r_depth);

    // Copy out the sub-region
    for rz in 0..r_depth {
        for rx in 0..r_width {
            let src_x = h_min_x + rx;
            let src_z = h_min_z + rz;
            let src_i = src_z * width + src_x;
            let dst_i = rz * r_width + rx;
            sub_heights[dst_i] = heights[src_i];
            sub_state.water[dst_i] = state.water[src_i];
            sub_state.sediment[dst_i] = state.sediment[src_i];
        }
    }

    // Run hydraulic on the sub-region
    simulate_hydraulic(
        &mut sub_heights,
        &mut sub_state,
        r_width,
        r_depth,
        cell_size,
        config,
        config.hydraulic_iterations,
    );

    // Copy results back for the core region (not the padding)
    for rz in 0..r_depth {
        for rx in 0..r_width {
            let src_x = h_min_x + rx;
            let src_z = h_min_z + rz;
            if src_x >= min_x && src_x < max_x && src_z >= min_z && src_z < max_z {
                let src_i = src_z * width + src_x;
                let dst_i = rz * r_width + rx;
                heights[src_i] = sub_heights[dst_i];
                state.water[src_i] = sub_state.water[dst_i];
                state.sediment[src_i] = sub_state.sediment[dst_i];
            }
        }
    }

    // Thermal erosion is applied to the full heightmap because it is
    // cheap and benefits from seeing the full slope context. We only
    // run it for the padded region to keep the cost reasonable while
    // still capturing cross-boundary slope effects.
    let t_min_x = min_x.saturating_sub(pad);
    let t_max_x = (max_x + pad).min(width);
    let t_min_z = min_z.saturating_sub(pad);
    let t_max_z = (max_z + pad).min(depth);

    // For thermal we simulate on a sub-slice; since thermal is local
    // (cardinal neighbours only), we can run on a sub-grid with boundary
    // clamping. We run on the full heightmap to keep it simple.
    let mut thermal_delta = vec![0.0f32; width * depth];
    let neighbour_offsets: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    for _ in 0..config.thermal_iterations {
        thermal_delta.iter_mut().for_each(|v| *v = 0.0);

        for z in t_min_z..t_max_z {
            for x in t_min_x..t_max_x {
                let i = z * width + x;
                let h_here = heights[i];

                let mut max_slope = 0.0f32;
                let mut slope_neighbour: Option<usize> = None;

                for &(dx, dz) in &neighbour_offsets {
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;
                    if nx < 0 || nx >= width as i32 || nz < 0 || nz >= depth as i32 {
                        continue;
                    }
                    let ni = nz as usize * width + nx as usize;
                    let diff = h_here - heights[ni];
                    if diff > 0.0 {
                        let slope = diff / cell_size;
                        if slope > max_slope {
                            max_slope = slope;
                            slope_neighbour = Some(ni);
                        }
                    }
                }

                if let Some(ni) = slope_neighbour {
                    if max_slope > config.thermal_angle_of_repose {
                        let diff = max_slope - config.thermal_angle_of_repose;
                        let transfer = (diff * config.thermal_strength * cell_size)
                            .min((h_here - heights[ni]) * 0.5);
                        if transfer > 1e-8 {
                            thermal_delta[i] -= transfer;
                            thermal_delta[ni] += transfer;
                        }
                    }
                }
            }
        }

        for z in t_min_z..t_max_z {
            for x in t_min_x..t_max_x {
                let i = z * width + x;
                heights[i] += thermal_delta[i];
                if heights[i] < 0.0 {
                    heights[i] = 0.0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a small flat heightmap and verifies that hydraulic erosion
    /// lowers terrain near the rain source. We use a small grid with
    /// slightly uneven terrain and confirm that after hydraulic erosion
    /// the total height has decreased (material has been eroded away).
    #[test]
    fn hydraulic_erodes_terrain() {
        let width = 16;
        let depth = 16;
        let mut heights = vec![5.0f32; width * depth];
        // Create a slight dip in the center to guide water flow
        for z in 6..10 {
            for x in 6..10 {
                heights[z * width + x] = 4.8;
            }
        }

        let initial_total: f32 = heights.iter().sum();

        let config = ErosionConfig {
            hydraulic_iterations: 100,
            thermal_iterations: 0,
            hydraulic_strength: 0.02,
            rain_rate: 0.005,
            evaporation_rate: 0.02,
            ..Default::default()
        };

        let mut state = ErosionState::new_state(width, depth);
        simulate_hydraulic(
            &mut heights,
            &mut state,
            width,
            depth,
            1.0,
            &config,
            config.hydraulic_iterations,
        );

        let final_total: f32 = heights.iter().sum();
        // Hydraulic erosion should have removed some terrain overall,
        // or at minimum redistributed height toward the dip
        assert!(
            final_total < initial_total,
            "expected hydraulic erosion to lower terrain: initial={initial_total}, final={final_total}"
        );

        // The center cells should be lower than the surrounding cells
        // after erosion concentrates flow there
        let mut center_sum = 0.0f32;
        for z in 6..10 {
            for x in 6..10 {
                center_sum += heights[z * width + x];
            }
        }
        let center_avg = center_sum / 16.0;
        let mut edge_sum = 0.0f32;
        for z in 0..4 {
            for x in 0..4 {
                edge_sum += heights[z * width + x];
            }
        }
        let edge_avg = edge_sum / 16.0;
        assert!(
            center_avg < edge_avg,
            "center should be lower than edges after hydraulic erosion: center={center_avg}, edge={edge_avg}"
        );
    }

    /// Creates terrain with a steep slope and verifies that thermal
    /// erosion reduces the slope toward the angle of repose.
    #[test]
    fn thermal_reduces_slope() {
        let width = 16;
        let depth = 16;
        let mut heights = vec![0.0f32; width * depth];

        // Create a steep step: left half high, right half low
        for z in 0..depth {
            for x in 0..width / 2 {
                heights[z * width + x] = 10.0;
            }
            for x in width / 2..width {
                heights[z * width + x] = 0.0;
            }
        }

        let config = ErosionConfig {
            hydraulic_iterations: 0,
            thermal_iterations: 200,
            thermal_strength: 0.05,
            thermal_angle_of_repose: 0.7,
            ..Default::default()
        };

        simulate_thermal(&mut heights, width, depth, 1.0, &config, 200);

        // Check the slope across the boundary (adjacent cells at x=7 and x=8)
        let boundary_high = heights[7];
        let boundary_low = heights[8];
        let final_boundary_slope = (boundary_high - boundary_low).abs();
        let initial_boundary_slope = 10.0f32; // was 10.0 - 0.0 = 10.0 before erosion

        assert!(
            final_boundary_slope < initial_boundary_slope,
            "thermal erosion should reduce boundary slope: initial={initial_boundary_slope}, final={final_boundary_slope}"
        );

        // The high cell at the boundary (x=7) should have lost some height
        assert!(
            heights[7] < 10.0,
            "high boundary cell should be eroded: height={}",
            heights[7]
        );
        // The low cell at the boundary (x=8) should have gained some height
        assert!(
            heights[8] > 0.0,
            "low boundary cell should have gained material: height={}",
            heights[8]
        );
    }

    /// Verifies that `erode_region` only affects the specified area and
    /// leaves the rest of the heightmap untouched.
    #[test]
    fn erode_region_isolates_changes() {
        let width = 32;
        let depth = 32;
        let mut heights = vec![5.0f32; width * depth];

        // Make a slightly uneven region in the center
        for z in 12..20 {
            for x in 12..20 {
                heights[z * width + x] = 8.0;
            }
        }

        let heights_before = heights.clone();

        let config = ErosionConfig {
            hydraulic_iterations: 20,
            thermal_iterations: 10,
            ..Default::default()
        };

        let mut state = ErosionState::new_state(width, depth);
        let min_x = 8;
        let max_x = 24;
        let min_z = 8;
        let max_z = 24;

        erode_region(
            &mut heights,
            &mut state,
            width,
            depth,
            1.0,
            &config,
            min_x,
            max_x,
            min_z,
            max_z,
        );

        // Cells outside the padded region should be unchanged.
        // The padding is 2, so cells outside [min_x-2, max_x+2) should
        // be identical. We check cells well outside (corners).
        let corners = [
            (0, 0),
            (0, depth - 1),
            (width - 1, 0),
            (width - 1, depth - 1),
            (1, 1),
            (width - 2, 1),
            (1, depth - 2),
            (width - 2, depth - 2),
        ];

        for &(x, z) in &corners {
            let i = z * width + x;
            assert!(
                (heights[i] - heights_before[i]).abs() < 1e-6,
                "cell ({x},{z}) should be unchanged: before={}, after={}",
                heights_before[i],
                heights[i]
            );
        }

        // At least some cells inside the region should have changed
        let mut changed = false;
        for z in min_z..max_z {
            for x in min_x..max_x {
                let i = z * width + x;
                if (heights[i] - heights_before[i]).abs() > 1e-6 {
                    changed = true;
                    break;
                }
            }
            if changed {
                break;
            }
        }
        assert!(
            changed,
            "at least some cells inside the region should have changed"
        );
    }
}
