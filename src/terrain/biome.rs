//! Biome rules engine for terrain generation and texture splatting.
//!
//! This module evaluates terrain properties (height, slope, moisture) against
//! configurable rules to determine which biome layers to apply and how to
//! blend them. It supports up to 4 simultaneous layers per terrain texel for
//! GPU texture splatting workflows.
//!
//! # Overview
//!
//! 1. Define [`BiomeRule`]s that match ranges of height, slope, and moisture.
//! 2. Attach a [`BiomeLayer`] (albedo, roughness, textures, tiling) to each rule.
//! 3. Group rules into a [`BiomeConfig`] and evaluate them via [`evaluate_biome`].
//! 4. Optionally feed a [`MoistureMap`] into [`TerrainBiomeSystem`] to drive
//!    moisture from height and proximity to water.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// A single rule that maps terrain properties to a biome layer.
///
/// Rules define inclusive ranges for height, slope, and moisture. A terrain
/// point matches a rule when **all** of its properties fall within the
/// specified ranges simultaneously.
///
/// When multiple rules match a single point, [`priority`] is used as a
/// tiebreaker — higher priority wins. If priorities are equal the rule that
/// appears later in the config list wins (last-match semantics).
///
/// # Fields
///
/// * `name`       — Human-readable identifier (e.g. "Grassland").
/// * `min_height` / `max_height` — Height range in world units (inclusive).
/// * `min_slope`  / `max_slope`  — Slope range in radians, `[0, PI/2]`.
/// * `min_moisture` / `max_moisture` — Moisture range `[0.0, 1.0]`.
/// * `priority`   — Higher values take precedence when rules overlap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomeRule {
    pub name: String,
    pub min_height: f32,
    pub max_height: f32,
    pub min_slope: f32,
    pub max_slope: f32,
    pub min_moisture: f32,
    pub max_moisture: f32,
    pub priority: u32,
}

impl BiomeRule {
    /// Returns `true` when the given terrain properties satisfy every range
    /// constraint of this rule.
    pub fn matches(&self, height: f32, slope: f32, moisture: f32) -> bool {
        height >= self.min_height
            && height <= self.max_height
            && slope >= self.min_slope
            && slope <= self.max_slope
            && moisture >= self.min_moisture
            && moisture <= self.max_moisture
    }
}

/// Describes the visual/material layer that a biome rule maps to.
///
/// These properties are designed to feed directly into PBR material workflows:
/// * `albedo_color`    — Base colour as `[r, g, b]` in linear space.
/// * `roughness`       — Surface roughness `[0.0, 1.0]`.
/// * `metallic`        — Metallic value `[0.0, 1.0]`.
/// * `normal_strength` — Normal-map intensity multiplier.
/// * `texture_path`    — File-system path to the diffuse/albedo texture.
/// * `tiling`          — How many times the texture repeats across one
///                       terrain tile (texture repeat factor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomeLayer {
    pub name: String,
    pub albedo_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub normal_strength: f32,
    pub texture_path: String,
    pub tiling: u32,
}

/// Top-level configuration that bundles biome rules with a fallback layer.
///
/// When **no** rule matches a terrain point the `default_layer` is used with
/// a blend weight of `1.0`.
///
/// # Example (TOML)
///
/// ```toml
/// [[rules]]
/// name = "Grassland"
/// min_height = 0.0
/// max_height = 50.0
/// min_slope = 0.0
/// max_slope = 0.35
/// min_moisture = 0.4
/// max_moisture = 1.0
/// priority = 1
///
/// [default_layer]
/// name = "Default"
/// albedo_color = [0.5, 0.5, 0.5]
/// roughness = 0.8
/// metallic = 0.0
/// normal_strength = 1.0
/// texture_path = "textures/default.png"
/// tiling = 1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomeConfig {
    /// Ordered list of biome rules. Later entries break priority ties.
    pub rules: Vec<BiomeRule>,
    /// Fallback layer used when no rule matches.
    pub default_layer: BiomeLayer,
}

/// Result returned by [`evaluate_biome`].
///
/// Contains the index of the dominant layer and up to four blend weights
/// for texture splatting on the GPU. Weights are normalised so they sum to
/// `1.0`. Unused slots are set to `0.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct BiomeResult {
    /// Index of the highest-weight layer inside the config's rule list.
    /// When no rule matched this points to the default layer (index `usize::MAX`).
    pub layer_index: usize,
    /// Blend weights for up to 4 layers. Weights are in `[0.0, 1.0]` and
    /// sum to `1.0`.
    pub blend_weights: [f32; 4],
}

/// Evaluates every rule in `config` against the supplied terrain properties
/// and returns the top four matching layers as a [`BiomeResult`].
///
/// # Algorithm
///
/// 1. Collect all rules whose ranges contain `(height, slope, moisture)`.
/// 2. Sort matches by `priority` descending, then by list position descending
///    (later rules win ties).
/// 3. Take at most four matches. If fewer than four match, the default layer
///    fills the remaining slots.
/// 4. Blend weights are distributed by normalising each rule's priority
///    contribution — higher priority rules receive proportionally more weight.
///
/// # Arguments
///
/// * `height`   — Terrain height in world units.
/// * `slope`    — Terrain surface slope in radians `[0, PI/2]`.
/// * `moisture` — Moisture value `[0.0, 1.0]`.
/// * `config`   — The active [`BiomeConfig`].
///
/// # Returns
///
/// A [`BiomeResult`] with the dominant layer index and four blend weights.
pub fn evaluate_biome(
    height: f32,
    slope: f32,
    moisture: f32,
    config: &BiomeConfig,
) -> BiomeResult {
    let mut matches: Vec<(usize, &BiomeRule)> = config
        .rules
        .iter()
        .enumerate()
        .filter(|(_, rule)| rule.matches(height, slope, moisture))
        .collect();

    // Sort: highest priority first, then later-in-list first for tie-breaking.
    matches.sort_by(|a, b| b.1.priority.cmp(&a.1.priority).then(b.0.cmp(&a.0)));

    let matches = matches;

    if matches.is_empty() {
        return BiomeResult {
            layer_index: usize::MAX,
            blend_weights: [1.0, 0.0, 0.0, 0.0],
        };
    }

    let count = matches.len().min(4);
    let total_priority: f32 = matches[..count]
        .iter()
        .map(|(_, r)| r.priority.max(1) as f32)
        .sum();

    let mut weights = [0.0f32; 4];
    for i in 0..count {
        weights[i] = matches[i].1.priority.max(1) as f32 / total_priority;
    }

    // Normalise to guard against floating-point drift.
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }

    BiomeResult {
        layer_index: matches[0].0,
        blend_weights: weights,
    }
}

/// A 2-D grid of moisture values that can be sampled at arbitrary world
/// coordinates via bilinear interpolation.
///
/// Moisture values are in `[0.0, 1.0]` where `0.0` is bone-dry and `1.0`
/// is fully saturated.
///
/// # Fields
///
/// * `values` — Row-major moisture data (`depth` rows × `width` columns).
/// * `width`  — Number of samples along the X axis.
/// * `depth`  — Number of samples along the Z axis.
/// * `origin_x` / `origin_z` — World-space position of sample `[0][0]`.
/// * `cell_size` — World-space distance between adjacent samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoistureMap {
    pub values: Vec<f32>,
    pub width: usize,
    pub depth: usize,
    pub origin_x: f32,
    pub origin_z: f32,
    pub cell_size: f32,
}

impl MoistureMap {
    /// Creates a new zeroed moisture map.
    pub fn new(width: usize, depth: usize, cell_size: f32) -> Self {
        Self {
            values: vec![0.0; width * depth],
            width,
            depth,
            origin_x: 0.0,
            origin_z: 0.0,
            cell_size,
        }
    }

    /// Returns a reference to the value at `(col, row)` or `None` if out of
    /// bounds.
    pub fn get(&self, col: usize, row: usize) -> Option<f32> {
        if col < self.width && row < self.depth {
            Some(self.values[row * self.width + col])
        } else {
            None
        }
    }

    /// Sets the value at `(col, row)`, clamping to `[0.0, 1.0]`.
    pub fn set(&mut self, col: usize, row: usize, value: f32) {
        if col < self.width && row < self.depth {
            self.values[row * self.width + col] = value.clamp(0.0, 1.0);
        }
    }

    /// Samples the moisture map at arbitrary world coordinates using bilinear
    /// interpolation.
    ///
    /// Coordinates outside the map are clamped to the nearest edge.
    pub fn sample_world(&self, world_x: f32, world_z: f32) -> f32 {
        let fx = (world_x - self.origin_x) / self.cell_size;
        let fz = (world_z - self.origin_z) / self.cell_size;

        let x0 = fx.floor().max(0.0) as usize;
        let z0 = fz.floor().max(0.0) as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let z1 = (z0 + 1).min(self.depth - 1);

        let tx = (fx - fx.floor()).clamp(0.0, 1.0);
        let tz = (fz - fz.floor()).clamp(0.0, 1.0);

        let v00 = self.values[z0 * self.width + x0];
        let v10 = self.values[z0 * self.width + x1];
        let v01 = self.values[z1 * self.width + x0];
        let v11 = self.values[z1 * self.width + x1];

        let top = v00 * (1.0 - tx) + v10 * tx;
        let bot = v01 * (1.0 - tx) + v11 * tx;
        (top * (1.0 - tz) + bot * tz).clamp(0.0, 1.0)
    }

    /// Returns the total number of samples (`width * depth`).
    pub fn len(&self) -> usize {
        self.width * self.depth
    }

    /// Returns `true` when the map contains no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// High-level system that ties biome configuration, moisture data, and
/// height-field information together.
///
/// Use [`TerrainBiomeSystem::update_moisture`] to (re)generate the moisture
/// map from a height-field, then call [`evaluate_biome`] for each terrain
/// texel to obtain splatting weights.
///
/// # Moisture Generation
///
/// Moisture is derived from two heuristics:
///
/// 1. **Height falloff** — lower terrain is wetter. The normalised height is
///    inverted and scaled by a configurable factor.
/// 2. **Water proximity** — cells below a configurable water level receive a
///    boost that decays with distance (flood-fill inspired approximation).
///
/// Both contributions are clamped to `[0.0, 1.0]` and blended additively.
#[derive(Debug, Clone)]
pub struct TerrainBiomeSystem {
    pub config: BiomeConfig,
    pub moisture: MoistureMap,
    /// Water level in world units. Terrain at or below this height is
    /// considered "near water".
    pub water_level: f32,
    /// How strongly low height contributes to moisture `[0.0, 1.0]`.
    pub height_moisture_factor: f32,
    /// Maximum distance (in samples) over which water proximity adds moisture.
    pub water_influence_radius: f32,
}

impl TerrainBiomeSystem {
    /// Constructs a new system with the given config and an empty moisture map.
    pub fn new(config: BiomeConfig, moisture: MoistureMap) -> Self {
        Self {
            config,
            moisture,
            water_level: 1.0,
            height_moisture_factor: 0.6,
            water_influence_radius: 8.0,
        }
    }

    /// Populates the moisture map from a height-field.
    ///
    /// * `heights` — Row-major height values, same dimensions as the moisture
    ///   map (`depth` rows × `width` columns).
    ///
    /// The algorithm:
    ///
    /// 1. Find the global min/max height for normalisation.
    /// 2. For each cell compute `height_moisture = 1 - normalised_height`.
    /// 3. For each cell at or below `water_level` add a proximity falloff that
    ///    decays linearly over `water_influence_radius` cells.
    /// 4. Clamp the final value to `[0.0, 1.0]`.
    pub fn update_moisture(&mut self, heights: &[f32]) {
        assert_eq!(
            heights.len(),
            self.moisture.len(),
            "height array length must match moisture map dimensions"
        );

        let min_h = heights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_h = heights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_h - min_h).max(1e-6);

        let w = self.moisture.width;
        let d = self.moisture.depth;
        let wl = self.water_level;
        let h_factor = self.height_moisture_factor;
        let radius = self.water_influence_radius;
        let _radius_i = radius.ceil() as isize;

        // Collect water-cell positions for the proximity pass.
        let water_cells: Vec<(isize, isize)> = (0..d)
            .flat_map(|z| (0..w).map(move |x| (x as isize, z as isize)))
            .filter(|&(x, z)| heights[z as usize * w + x as usize] <= wl)
            .collect();

        for z in 0..d {
            for x in 0..w {
                let h = heights[z * w + x];
                let normalised = ((h - min_h) / range).clamp(0.0, 1.0);
                let mut moisture = (1.0 - normalised) * h_factor;

                // Water proximity boost.
                if !water_cells.is_empty() {
                    let mut closest_dist = f32::MAX;
                    for &(wx, wz) in &water_cells {
                        let dx = x as isize - wx;
                        let dz = z as isize - wz;
                        let dist = ((dx * dx + dz * dz) as f32).sqrt();
                        if dist < closest_dist {
                            closest_dist = dist;
                        }
                    }
                    let proximity = (1.0 - closest_dist / radius).max(0.0);
                    moisture += proximity * (1.0 - h_factor);
                }

                self.moisture.set(x, z, moisture.clamp(0.0, 1.0));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Default biome presets
// ---------------------------------------------------------------------------

/// Returns a [`BiomeConfig`] pre-populated with six sensible default biomes:
///
/// | Biome    | Height      | Slope         | Moisture  | Priority |
/// |----------|-------------|---------------|-----------|----------|
/// | Snow     | 200 – 9999  | 0 – π/2       | 0.0 – 1.0 | 5        |
/// | Mountain | 120 – 200   | 0.35 – π/2    | 0.0 – 1.0 | 4        |
/// | Cliff    | 0 – 9999    | 1.1 – π/2     | 0.0 – 1.0 | 6        |
/// | Forest   | 30 – 120    | 0 – 0.7       | 0.3 – 1.0 | 3        |
/// | Grassland| 0 – 50      | 0 – 0.35      | 0.4 – 1.0 | 2        |
/// | Beach    | 0 – 5       | 0 – 0.15      | 0.0 – 1.0 | 1        |
///
/// The default fallback layer is a neutral grey dirt material.
pub fn default_biome_config() -> BiomeConfig {
    BiomeConfig {
        rules: vec![
            BiomeRule {
                name: "Beach".into(),
                min_height: 0.0,
                max_height: 5.0,
                min_slope: 0.0,
                max_slope: 0.15,
                min_moisture: 0.0,
                max_moisture: 1.0,
                priority: 1,
            },
            BiomeRule {
                name: "Grassland".into(),
                min_height: 0.0,
                max_height: 50.0,
                min_slope: 0.0,
                max_slope: 0.35,
                min_moisture: 0.4,
                max_moisture: 1.0,
                priority: 2,
            },
            BiomeRule {
                name: "Forest".into(),
                min_height: 30.0,
                max_height: 120.0,
                min_slope: 0.0,
                max_slope: 0.7,
                min_moisture: 0.3,
                max_moisture: 1.0,
                priority: 3,
            },
            BiomeRule {
                name: "Mountain".into(),
                min_height: 120.0,
                max_height: 200.0,
                min_slope: 0.35,
                max_slope: PI / 2.0,
                min_moisture: 0.0,
                max_moisture: 1.0,
                priority: 4,
            },
            BiomeRule {
                name: "Snow".into(),
                min_height: 200.0,
                max_height: 9999.0,
                min_slope: 0.0,
                max_slope: PI / 2.0,
                min_moisture: 0.0,
                max_moisture: 1.0,
                priority: 5,
            },
            BiomeRule {
                name: "Cliff".into(),
                min_height: 0.0,
                max_height: 9999.0,
                min_slope: 1.1,
                max_slope: PI / 2.0,
                min_moisture: 0.0,
                max_moisture: 1.0,
                priority: 6,
            },
        ],
        default_layer: BiomeLayer {
            name: "DefaultDirt".into(),
            albedo_color: [0.45, 0.35, 0.25],
            roughness: 0.9,
            metallic: 0.0,
            normal_strength: 1.0,
            texture_path: "textures/default_dirt.png".into(),
            tiling: 4,
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the default config evaluates to a sensible dominant layer
    /// for several representative terrain configurations.
    #[test]
    fn test_default_evaluation() {
        let config = default_biome_config();

        // Low, flat, wet → Grassland (priority 2, beats Beach priority 1).
        let r = evaluate_biome(20.0, 0.1, 0.7, &config);
        assert_eq!(config.rules[r.layer_index].name, "Grassland");

        // High & steep → Mountain (priority 4).
        let r = evaluate_biome(160.0, 0.8, 0.5, &config);
        assert_eq!(config.rules[r.layer_index].name, "Mountain");

        // Very high → Snow (priority 5).
        let r = evaluate_biome(300.0, 0.1, 0.5, &config);
        assert_eq!(config.rules[r.layer_index].name, "Snow");

        // Very steep at any height → Cliff (priority 6, highest).
        let r = evaluate_biome(80.0, 1.3, 0.5, &config);
        assert_eq!(config.rules[r.layer_index].name, "Cliff");

        // Near water level, flat, low moisture → Beach (priority 1).
        // moisture=0.2 is below Grassland's min_moisture (0.4), so only Beach matches.
        let r = evaluate_biome(2.0, 0.05, 0.2, &config);
        assert_eq!(config.rules[r.layer_index].name, "Beach");

        // Mid height, moderate slope, wet → Forest (priority 3).
        let r = evaluate_biome(75.0, 0.5, 0.6, &config);
        assert_eq!(config.rules[r.layer_index].name, "Forest");

        // Blend weights should sum to 1.0 (within float tolerance).
        let r = evaluate_biome(20.0, 0.1, 0.7, &config);
        let sum: f32 = r.blend_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights should sum to 1.0, got {sum}");

        // Completely out-of-range properties → default layer.
        let r = evaluate_biome(-10.0, -1.0, -1.0, &config);
        assert_eq!(r.layer_index, usize::MAX);
    }

    /// Verifies moisture generation from a simple height-field.
    #[test]
    fn test_moisture_generation() {
        let moisture_map = MoistureMap::new(4, 4, 10.0);
        let config = default_biome_config();
        let mut system = TerrainBiomeSystem::new(config, moisture_map);
        system.water_level = 5.0;
        system.height_moisture_factor = 0.6;
        system.water_influence_radius = 2.0;

        // Flat plane at height 0 — should be fully moist.
        let heights = [0.0f32; 16];
        system.update_moisture(&heights);
        for v in &system.moisture.values {
            assert!(*v > 0.9, "low flat terrain should be very wet, got {v}");
        }

        // Tall uniform terrain — should be drier than the flat plane.
        let heights = [200.0f32; 16];
        system.update_moisture(&heights);
        for v in &system.moisture.values {
            // All heights equal → normalised = 0 → moisture = (1-0)*0.6 = 0.6.
            // No water cells, so no proximity boost.
            assert!(
                *v < 0.7,
                "high uniform terrain should be drier than flat, got {v}"
            );
        }

        // Mixed: one low cell (water) and one high cell nearby.
        let mut heights = [100.0f32; 16];
        heights[0] = 0.0; // water cell at (0, 0)
        system.update_moisture(&heights);

        // Cell (0,0) itself — low, so high moisture.
        assert!(
            system.moisture.get(0, 0).unwrap() > 0.5,
            "water cell should be wet"
        );

        // Cell (3,3) — far from water, high → should be drier.
        let far = system.moisture.get(3, 3).unwrap();
        assert!(
            far < 0.3,
            "distant high cell should be relatively dry, got {far}"
        );
    }

    /// Ensures that higher-priority rules win when multiple rules match the
    /// same terrain point.
    #[test]
    fn test_biome_priority() {
        let config = default_biome_config();

        // Height=40, slope=0.1, moisture=0.6
        // Matches: Beach (p1), Grassland (p2), Forest (p3), Cliff would
        // need slope >= 1.1 so no.  Winner should be Forest.
        let r = evaluate_biome(40.0, 0.1, 0.6, &config);
        assert_eq!(
            config.rules[r.layer_index].name, "Forest",
            "Forest (p3) should beat Grassland (p2) and Beach (p1)"
        );
        // Blend weights: Forest should have the largest share.
        assert!(
            r.blend_weights[0] > r.blend_weights[1],
            "top layer should have highest weight"
        );

        // Modify config to verify custom priority overrides.
        let mut custom = config.clone();
        // Give Beach the highest priority. Use height=2 so Beach's range [0,5] matches.
        custom.rules[0].priority = 100;
        let r = evaluate_biome(2.0, 0.1, 0.6, &custom);
        assert_eq!(
            custom.rules[r.layer_index].name, "Beach",
            "Beach should win with artificially high priority"
        );
    }
}
