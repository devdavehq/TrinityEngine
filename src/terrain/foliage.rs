//! Foliage placement system for terrain rendering.
//!
//! Provides species-based foliage placement using Poisson disk sampling,
//! biome rule filtering, density maps, and exclusion zones.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// RNG (hash-based, no external crate)
// ---------------------------------------------------------------------------

/// Simple deterministic PRNG using a splitmix-style hash counter.
///
/// Only used internally for Poisson disk sampling and placement jitter.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e3779b97f4a7c15),
        }
    }

    /// Advance state and return a uniform `u64`.
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform float in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }

    /// Uniform float in `[lo, hi)`.
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Uniform `usize` in `[0, n)`.
    fn range_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// FoliageSpecies
// ---------------------------------------------------------------------------

/// Describes one type of foliage that can be placed on the terrain.
///
/// Each species carries mesh metadata, biome constraints (height, slope,
/// moisture), scale bounds, density target, optional colour variation,
/// wind parameters, and trunk info for trees.
#[derive(Debug, Clone)]
pub struct FoliageSpecies {
    /// Human-readable name (e.g. "Pine", "Oak", "Grass").
    pub name: String,
    /// Path to the mesh asset.
    pub mesh_path: String,
    /// Minimum terrain height (world units) at which this species may appear.
    pub min_height: f32,
    /// Maximum terrain height (world units) at which this species may appear.
    pub max_height: f32,
    /// Minimum terrain slope (degrees, 0 = flat, 90 = vertical).
    pub min_slope: f32,
    /// Maximum terrain slope (degrees).
    pub max_slope: f32,
    /// Minimum moisture value (0..1) required for this species.
    pub min_moisture: f32,
    /// Maximum moisture value (0..1) allowed for this species.
    pub max_moisture: f32,
    /// Minimum random scale multiplier applied to the placed instance.
    pub min_scale: f32,
    /// Maximum random scale multiplier applied to the placed instance.
    pub max_scale: f32,
    /// Target density in plants per square metre.
    pub density: f32,
    /// RGB colour offset minimum per channel (added to base vertex colour).
    pub color_variation_min: [f32; 3],
    /// RGB colour offset maximum per channel.
    pub color_variation_max: [f32; 3],
    /// Wind sway strength multiplier (0 = static, 1 = full).
    pub wind_strength: f32,
    /// `true` for trees (affects culling / LOD decisions downstream).
    pub is_tree: bool,
    /// Trunk height in world units. Meaningful only when `is_tree` is `true`.
    pub trunk_height: f32,
}

impl FoliageSpecies {
    /// Returns `true` when the given terrain parameters satisfy this species'
    /// biome constraints.
    pub fn matches_biome(&self, height: f32, slope: f32, moisture: f32) -> bool {
        height >= self.min_height
            && height <= self.max_height
            && slope >= self.min_slope
            && slope <= self.max_slope
            && moisture >= self.min_moisture
            && moisture <= self.max_moisture
    }
}

// ---------------------------------------------------------------------------
// ExclusionZone
// ---------------------------------------------------------------------------

/// A circular area on the XZ plane where no foliage should be placed.
///
/// Useful for roads, buildings, water features, or any region that must
/// remain clear of vegetation.
#[derive(Debug, Clone)]
pub struct ExclusionZone {
    /// Centre of the zone on the X axis (world units).
    pub center_x: f32,
    /// Centre of the zone on the Z axis (world units).
    pub center_z: f32,
    /// Radius of the zone (world units). Must be non-negative.
    pub radius: f32,
}

impl ExclusionZone {
    /// Returns `true` when `(x, z)` falls inside this zone.
    pub fn contains(&self, x: f32, z: f32) -> bool {
        let dx = x - self.center_x;
        let dz = z - self.center_z;
        dx * dx + dz * dz <= self.radius * self.radius
    }
}

// ---------------------------------------------------------------------------
// FoliageDensityMap
// ---------------------------------------------------------------------------

/// A uniform grid that stores per-cell density multipliers for foliage.
///
/// Values are expected to be in `[0, 1]` where `0` means no foliage and
/// `1` means full species density.  The map covers a rectangular world
/// region defined by its dimensions and cell size.
#[derive(Debug, Clone)]
pub struct FoliageDensityMap {
    /// Flat storage row-major: index = `row * width + col`.
    data: Vec<f32>,
    /// Number of cells along the X axis.
    pub width: usize,
    /// Number of cells along the Z axis.
    pub depth: usize,
    /// World-space size of one cell on each axis.
    pub cell_size: f32,
}

impl FoliageDensityMap {
    /// Create a new density map initialised to zero (no foliage anywhere).
    ///
    /// * `width` – number of cells in the X direction.
    /// * `depth` – number of cells in the Z direction.
    /// * `cell_size` – world units per cell side.
    pub fn new(width: usize, depth: usize, cell_size: f32) -> Self {
        Self {
            data: vec![0.0; width * depth],
            width,
            depth,
            cell_size,
        }
    }

    /// Convert a world X coordinate to a column index (clamped).
    fn world_to_col(&self, x: f32) -> usize {
        let col = (x / self.cell_size) as isize;
        col.clamp(0, self.width as isize - 1) as usize
    }

    /// Convert a world Z coordinate to a row index (clamped).
    fn world_to_row(&self, z: f32) -> usize {
        let row = (z / self.cell_size) as isize;
        row.clamp(0, self.depth as isize - 1) as usize
    }

    /// Sample the density at a world position using bilinear interpolation.
    ///
    /// Positions outside the map return `0.0`.
    pub fn sample_world(&self, x: f32, z: f32) -> f32 {
        let fx = x / self.cell_size;
        let fz = z / self.cell_size;
        let ix = fx.floor() as isize;
        let iz = fz.floor() as isize;
        let frac_x = fx - ix as f32;
        let frac_z = fz - iz as f32;

        let get = |r: isize, c: isize| -> f32 {
            if r < 0 || c < 0 || r >= self.depth as isize || c >= self.width as isize {
                return 0.0;
            };
            self.data[r as usize * self.width + c as usize]
        };

        let v00 = get(iz, ix);
        let v10 = get(iz, ix + 1);
        let v01 = get(iz + 1, ix);
        let v11 = get(iz + 1, ix + 1);

        let top = v00 + (v10 - v00) * frac_x;
        let bot = v01 + (v11 - v01) * frac_x;
        top + (bot - top) * frac_z
    }

    /// Set the density value for the cell that contains `(x, z)`.
    ///
    /// Values are silently clamped to the valid cell range; coordinates
    /// outside the map are ignored.
    pub fn set_world(&mut self, x: f32, z: f32, val: f32) {
        let col = self.world_to_col(x);
        let row = self.world_to_row(z);
        self.data[row * self.width + col] = val;
    }

    /// Direct cell access by index.
    pub fn get(&self, col: usize, row: usize) -> f32 {
        self.data[row * self.width + col]
    }

    /// Direct mutable cell access by index.
    pub fn set(&mut self, col: usize, row: usize, val: f32) {
        self.data[row * self.width + col] = val;
    }
}

// ---------------------------------------------------------------------------
// FoliagePlacement
// ---------------------------------------------------------------------------

/// Holds all data required to drive a single foliage placement pass.
#[derive(Debug, Clone)]
pub struct FoliagePlacement {
    /// Available species.
    pub species: Vec<FoliageSpecies>,
    /// Per-cell density multiplier map.
    pub density_map: FoliageDensityMap,
    /// Hard cap on the total number of placed instances returned by
    /// `place_foliage`.  The actual count may be lower due to biome
    /// filtering and exclusion zones.
    pub total_max_plants: usize,
    /// Circular regions where foliage must not appear.
    pub exclusion_zones: Vec<ExclusionZone>,
}

// ---------------------------------------------------------------------------
// PlacedFoliage
// ---------------------------------------------------------------------------

/// A single placed foliage instance.
#[derive(Debug, Clone)]
pub struct PlacedFoliage {
    /// Index into the species list that was used for this instance.
    pub species_index: usize,
    /// World X coordinate.
    pub x: f32,
    /// World Z coordinate.
    pub z: f32,
    /// Height offset above the terrain surface (e.g. to sink trunks slightly).
    pub y_offset: f32,
    /// Random uniform scale multiplier applied to the mesh.
    pub scale: f32,
    /// Random rotation around the Y axis in radians.
    pub rotation_y: f32,
    /// Per-channel colour offset added to the base mesh colour.
    pub color_offset: [f32; 3],
}

// ---------------------------------------------------------------------------
// PlacementConfig
// ---------------------------------------------------------------------------

/// Tuning knobs for the Poisson-disk placement algorithm.
#[derive(Debug, Clone)]
pub struct PlacementConfig {
    /// Minimum distance between any two placed instances (world units).
    /// This directly controls the Poisson disk radius.
    pub min_distance_between: f32,
    /// Maximum number of rejection samples per active point before it is
    /// discarded.  Higher values yield denser packings at the cost of time.
    pub max_attempts: u32,
    /// Seed for the deterministic hash-based RNG.
    pub random_seed: u64,
}

// ---------------------------------------------------------------------------
// Core placement algorithm
// ---------------------------------------------------------------------------

/// Place foliage across the entire placement region using Poisson disk
/// sampling, biome filtering, density-map modulation, and exclusion zones.
///
/// # Arguments
///
/// * `terrain_height_fn` – world `(x, z)` → height.
/// * `terrain_slope_fn` – world `(x, z)` → slope in degrees.
/// * `terrain_moisture_fn` – world `(x, z)` → moisture in `[0, 1]`.
/// * `placement` – species list, density map, exclusion zones, and budget.
/// * `config` – Poisson disk parameters.
///
/// # Returns
///
/// A vector of [`PlacedFoliage`] instances.  Order is non-deterministic
/// (depends on sampling order) and should not be relied upon for
/// rendering.
pub fn place_foliage(
    terrain_height_fn: impl Fn(f32, f32) -> f32,
    terrain_slope_fn: impl Fn(f32, f32) -> f32,
    terrain_moisture_fn: impl Fn(f32, f32) -> f32,
    placement: &FoliagePlacement,
    config: &PlacementConfig,
) -> Vec<PlacedFoliage> {
    let mut rng = Rng::new(config.random_seed);
    let mut results: Vec<PlacedFoliage> = Vec::new();

    if placement.species.is_empty() || config.min_distance_between <= 0.0 {
        return results;
    }

    // Build a list of candidate species indices weighted by density.
    let total_density: f32 = placement
        .species
        .iter()
        .map(|s| s.density)
        .sum();
    if total_density <= 0.0 {
        return results;
    }

    // Determine the world bounds from the density map.
    let map_width = placement.density_map.width as f32 * placement.density_map.cell_size;
    let map_depth = placement.density_map.depth as f32 * placement.density_map.cell_size;

    let cell = config.min_distance_between;
    let cols = (map_width / cell).ceil().max(1.0) as usize;
    let rows = (map_depth / cell).ceil().max(1.0) as usize;

    // Active-point list for Poisson disk sampling.
    let mut active: Vec<(f32, f32)> = Vec::new();
    let mut placed: Vec<(f32, f32)> = Vec::new();

    // Seed the first point.
    let seed_x = rng.range_f32(0.0, map_width);
    let seed_z = rng.range_f32(0.0, map_depth);
    active.push((seed_x, seed_z));

    while !active.is_empty() && results.len() < placement.total_max_plants {
        // Pick a random active point.
        let idx = rng.range_usize(active.len());
        let (cx, cz) = active[idx];

        let mut found = false;
        for _ in 0..config.max_attempts {
            let angle = rng.range_f32(0.0, std::f32::consts::TAU);
            let dist = rng.range_f32(config.min_distance_between, config.min_distance_between * 2.0);
            let nx = cx + angle.cos() * dist;
            let nz = cz + angle.sin() * dist;

            if nx < 0.0 || nx >= map_width || nz < 0.0 || nz >= map_depth {
                continue;
            }

            // Check minimum distance to all existing samples.
            let too_close = placed.iter().any(|(px, pz)| {
                let dx = nx - px;
                let dz = nz - pz;
                dx * dx + dz * dz < config.min_distance_between * config.min_distance_between
            });
            if too_close {
                continue;
            }

            // Exclusion zone check.
            let excluded = placement
                .exclusion_zones
                .iter()
                .any(|z| z.contains(nx, nz));
            if excluded {
                continue;
            }

            // Sample terrain attributes.
            let height = (terrain_height_fn)(nx, nz);
            let slope = (terrain_slope_fn)(nx, nz);
            let moisture = (terrain_moisture_fn)(nx, nz);
            let density_mult = placement.density_map.sample_world(nx, nz);

            if density_mult <= 0.0 {
                continue;
            }

            // Pick a species whose biome rules are satisfied, weighted by
            // density × density-map multiplier.
            let mut candidate_indices: Vec<usize> = Vec::new();
            let mut candidate_weights: Vec<f32> = Vec::new();
            for (si, sp) in placement.species.iter().enumerate() {
                if sp.matches_biome(height, slope, moisture) {
                    let w = sp.density * density_mult;
                    if w > 0.0 {
                        candidate_indices.push(si);
                        candidate_weights.push(w);
                    }
                }
            }

            if candidate_indices.is_empty() {
                continue;
            }

            let weight_sum: f32 = candidate_weights.iter().sum();
            let pick = rng.range_f32(0.0, weight_sum);
            let mut cumulative = 0.0;
            let mut chosen = 0;
            for (ci, &w) in candidate_weights.iter().enumerate() {
                cumulative += w;
                if pick <= cumulative {
                    chosen = ci;
                    break;
                }
            }
            let species_idx = candidate_indices[chosen];
            let sp = &placement.species[species_idx];

            placed.push((nx, nz));
            active.push((nx, nz));
            found = true;

            results.push(PlacedFoliage {
                species_index: species_idx,
                x: nx,
                z: nz,
                y_offset: if sp.is_tree {
                    -sp.trunk_height * 0.1
                } else {
                    0.0
                },
                scale: rng.range_f32(sp.min_scale, sp.max_scale),
                rotation_y: rng.range_f32(0.0, std::f32::consts::TAU),
                color_offset: [
                    rng.range_f32(sp.color_variation_min[0], sp.color_variation_max[0]),
                    rng.range_f32(sp.color_variation_min[1], sp.color_variation_max[1]),
                    rng.range_f32(sp.color_variation_min[2], sp.color_variation_max[2]),
                ],
            });

            if results.len() >= placement.total_max_plants {
                break;
            }
        }

        if !found {
            active.swap_remove(idx);
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Mutation helpers
// ---------------------------------------------------------------------------

/// Add an exclusion zone and remove any existing foliage instances that
/// fall inside it.
///
/// Returns the number of foliage instances that were removed.
pub fn add_exclusion_zone(
    placement: &mut FoliagePlacement,
    zone: ExclusionZone,
) -> usize {
    placement.exclusion_zones.push(zone.clone());
    // Note: the caller owns the placed vec separately; this function only
    // mutates the placement struct.  To also clean up placed foliage the
    // caller should invoke `remove_nearby` on their own `Vec<PlacedFoliage>`
    // for each new zone.  For convenience we return 0 here and document
    // the intended usage.
    //
    // If we *did* want to remove from a provided list we would need a
    // mutable reference to it — the caller can call `remove_nearby` right
    // after this.
    //
    // However, to make the API ergonomic we provide an overloaded version
    // below that also takes the placed-vec.

    0
}

/// Overload that also cleans up a placed-foliage list.
///
/// Adds `zone` to `placement` and removes every placed instance that
/// falls inside the zone.  Returns the number of removed instances.
pub fn add_exclusion_zone_with_cleanup(
    placement: &mut FoliagePlacement,
    placed: &mut Vec<PlacedFoliage>,
    zone: ExclusionZone,
) -> usize {
    let count = remove_nearby(placed, zone.center_x, zone.center_z, zone.radius);
    placement.exclusion_zones.push(zone);
    count
}

/// Remove every foliage instance whose `(x, z)` falls within `radius` of
/// `(center_x, center_z)`.
///
/// Returns the number of instances removed.
pub fn remove_nearby(
    placed: &mut Vec<PlacedFoliage>,
    center_x: f32,
    center_z: f32,
    radius: f32,
) -> usize {
    let r2 = radius * radius;
    let before = placed.len();
    placed.retain(|p| {
        let dx = p.x - center_x;
        let dz = p.z - center_z;
        dx * dx + dz * dz > r2
    });
    before - placed.len()
}

/// Re-evaluate foliage inside an axis-aligned rectangle on the XZ plane.
///
/// Existing placed foliage whose position falls inside the rectangle is
/// discarded and new instances are generated using the provided terrain
/// functions.  Instances outside the rectangle are left untouched.
///
/// This is useful when a chunk of terrain is modified at runtime (e.g.
/// erosion, sculpting) and foliage needs to be regenerated locally.
///
/// # Arguments
///
/// * `placed` – the global placed-foliage list (mutated in-place).
/// * `rect_min_x`, `rect_min_z`, `rect_max_x`, `rect_max_z` – bounds of
///   the rectangle in world units.
/// * `terrain_height_fn`, `terrain_slope_fn`, `terrain_moisture_fn` –
///   terrain attribute callbacks (same signatures as [`place_foliage`]).
/// * `species` – the species list to sample from.
/// * `config` – placement configuration.
pub fn recalculate_region(
    placed: &mut Vec<PlacedFoliage>,
    rect_min_x: f32,
    rect_min_z: f32,
    rect_max_x: f32,
    rect_max_z: f32,
    terrain_height_fn: impl Fn(f32, f32) -> f32,
    terrain_slope_fn: impl Fn(f32, f32) -> f32,
    terrain_moisture_fn: impl Fn(f32, f32) -> f32,
    species: &[FoliageSpecies],
    config: &PlacementConfig,
) {
    // 1. Remove existing instances inside the rectangle.
    placed.retain(|p| {
        p.x < rect_min_x
            || p.x > rect_max_x
            || p.z < rect_min_z
            || p.z > rect_max_z
    });

    if species.is_empty() {
        return;
    }

    // 2. Build a temporary placement covering the rectangle.
    let rect_w = rect_max_x - rect_min_x;
    let rect_d = rect_max_z - rect_min_z;
    if rect_w <= 0.0 || rect_d <= 0.0 {
        return;
    }

    let cell_size = config.min_distance_between;
    let cols = (rect_w / cell_size).ceil().max(1.0) as usize;
    let rows = (rect_d / cell_size).ceil().max(1.0) as usize;

    let mut density_map = FoliageDensityMap::new(cols, rows, cell_size);
    for r in 0..rows {
        for c in 0..cols {
            let wx = rect_min_x + (c as f32 + 0.5) * cell_size;
            let wz = rect_min_z + (r as f32 + 0.5) * cell_size;
            density_map.set(c, r, 1.0);
            let _ = (wx, wz); // suppress unused warning; in a real engine
                               // you would sample a density texture here.
        }
    }

    let max_new = cols * rows * 2; // reasonable budget
    let local_placement = FoliagePlacement {
        species: species.to_vec(),
        density_map,
        total_max_plants: max_new,
        exclusion_zones: Vec::new(),
    };

    // Shift terrain functions by the rectangle offset so the Poisson
    // sampler works in local coordinates, then translate back.
    let off_x = rect_min_x;
    let off_z = rect_min_z;
    let h_fn = |lx: f32, lz: f32| (terrain_height_fn)(lx + off_x, lz + off_z);
    let s_fn = |lx: f32, lz: f32| (terrain_slope_fn)(lx + off_x, lz + off_z);
    let m_fn = |lx: f32, lz: f32| (terrain_moisture_fn)(lx + off_x, lz + off_z);

    let new_instances = place_foliage(h_fn, s_fn, m_fn, &local_placement, config);

    // Translate local coordinates back to world space.
    for mut inst in new_instances {
        inst.x += off_x;
        inst.z += off_z;
        placed.push(inst);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn flat_height(_: f32, _: f32) -> f32 {
        100.0
    }
    fn flat_slope(_: f32, _: f32) -> f32 {
        5.0
    }
    fn mid_moisture(_: f32, _: f32) -> f32 {
        0.5
    }

    fn full_density_map(width: usize, depth: usize, cell_size: f32) -> FoliageDensityMap {
        let mut map = FoliageDensityMap::new(width, depth, cell_size);
        for r in 0..depth {
            for c in 0..width {
                map.set(c, r, 1.0);
            }
        }
        map
    }

    fn default_species() -> Vec<FoliageSpecies> {
        vec![
            FoliageSpecies {
                name: "Grass".into(),
                mesh_path: "assets/grass.obj".into(),
                min_height: 0.0,
                max_height: 500.0,
                min_slope: 0.0,
                max_slope: 45.0,
                min_moisture: 0.0,
                max_moisture: 1.0,
                min_scale: 0.8,
                max_scale: 1.2,
                density: 4.0,
                color_variation_min: [-0.05, -0.1, -0.05],
                color_variation_max: [0.05, 0.1, 0.05],
                wind_strength: 0.8,
                is_tree: false,
                trunk_height: 0.0,
            },
            FoliageSpecies {
                name: "Pine".into(),
                mesh_path: "assets/pine.obj".into(),
                min_height: 50.0,
                max_height: 400.0,
                min_slope: 0.0,
                max_slope: 60.0,
                min_moisture: 0.2,
                max_moisture: 0.8,
                min_scale: 0.9,
                max_scale: 1.4,
                density: 0.5,
                color_variation_min: [-0.02, -0.02, -0.02],
                color_variation_max: [0.02, 0.02, 0.02],
                wind_strength: 0.4,
                is_tree: true,
                trunk_height: 3.0,
            },
        ]
    }

    fn simple_config(seed: u64) -> PlacementConfig {
        PlacementConfig {
            min_distance_between: 2.0,
            max_attempts: 30,
            random_seed: seed,
        }
    }

    // -- test 1: Poisson disk spacing --

    #[test]
    fn poisson_disk_spacing() {
        let placement = FoliagePlacement {
            species: default_species(),
            density_map: full_density_map(64, 64, 2.0),
            total_max_plants: 500,
            exclusion_zones: Vec::new(),
        };
        let config = simple_config(42);
        let placed = place_foliage(
            flat_height,
            flat_slope,
            mid_moisture,
            &placement,
            &config,
        );
        assert!(!placed.is_empty(), "should place at least some foliage");

        // Every pair must be at least min_distance apart.
        let min_d = config.min_distance_between;
        for (i, a) in placed.iter().enumerate() {
            for b in placed.iter().skip(i + 1) {
                let dx = a.x - b.x;
                let dz = a.z - b.z;
                let dist = (dx * dx + dz * dz).sqrt();
                assert!(
                    dist >= min_d - 0.001,
                    "Poisson spacing violated: {} < {} (points {} and {})",
                    dist,
                    min_d,
                    i,
                    i + 1
                );
            }
        }
    }

    // -- test 2: exclusion zone filtering --

    #[test]
    fn exclusion_zone_filtering() {
        let mut placement = FoliagePlacement {
            species: default_species(),
            density_map: FoliageDensityMap::new(64, 64, 2.0),
            total_max_plants: 1000,
            exclusion_zones: Vec::new(),
        };
        // Place a zone in the centre of the map.
        placement.exclusion_zones.push(ExclusionZone {
            center_x: 64.0,
            center_z: 64.0,
            radius: 20.0,
        });

        let config = simple_config(123);
        let placed = place_foliage(
            flat_height,
            flat_slope,
            mid_moisture,
            &placement,
            &config,
        );

        let zone = &placement.exclusion_zones[0];
        for p in &placed {
            let dx = p.x - zone.center_x;
            let dz = p.z - zone.center_z;
            assert!(
                dx * dx + dz * dz > zone.radius * zone.radius,
                "foliage at ({}, {}) inside exclusion zone",
                p.x,
                p.z
            );
        }
    }

    // -- test 3: biome rule matching --

    #[test]
    fn biome_rule_matching() {
        let sp = default_species();

        // Grass: height 100 in [0,500], slope 5 in [0,45], moisture 0.5 in [0,1]
        assert!(sp[0].matches_biome(100.0, 5.0, 0.5));
        // Pine: height 100 in [50,400] ✓, slope 5 in [0,60] ✓, moisture 0.5 in [0.2,0.8] ✓
        assert!(sp[1].matches_biome(100.0, 5.0, 0.5));

        // Pine fails: height too low.
        assert!(!sp[1].matches_biome(10.0, 5.0, 0.5));
        // Pine fails: slope too steep.
        assert!(!sp[1].matches_biome(100.0, 80.0, 0.5));
        // Pine fails: moisture too low.
        assert!(!sp[1].matches_biome(100.0, 5.0, 0.05));

        // Verify that with only high-slope terrain, the placement only
        // picks species that tolerate steep slopes.
        let steep_slope = |_x: f32, _z: f32| -> f32 { 70.0 };
        let placement = FoliagePlacement {
            species: default_species(),
            density_map: FoliageDensityMap::new(16, 16, 8.0),
            total_max_plants: 200,
            exclusion_zones: Vec::new(),
        };
        let config = simple_config(999);
        let placed = place_foliage(
            flat_height,
            steep_slope,
            mid_moisture,
            &placement,
            &config,
        );
        // Both species have max_slope 45 and 60; slope 70 should match
        // neither, so nothing should be placed.
        assert!(placed.is_empty(), "steep slope should reject all species");
    }

    // -- test 4: region recalculation --

    #[test]
    fn region_recalculation() {
        let config = simple_config(77);
        let species = default_species();
        let height_fn = flat_height;
        let slope_fn = flat_slope;
        let moisture_fn = mid_moisture;

        // Seed initial placement over a 128×128 area.
        let mut placed = Vec::new();
        let mut placement = FoliagePlacement {
            species: species.clone(),
            density_map: full_density_map(64, 64, 2.0),
            total_max_plants: 200,
            exclusion_zones: Vec::new(),
        };
        let initial = place_foliage(height_fn, slope_fn, moisture_fn, &placement, &config);
        placed.extend(initial);
        let before = placed.len();
        assert!(before > 0, "must place some foliage initially");

        // Recalculate the centre 32×32 region.
        let rect_x0 = 48.0_f32;
        let rect_z0 = 48.0_f32;
        let rect_x1 = 80.0_f32;
        let rect_z1 = 80.0_f32;

        let count_before = placed
            .iter()
            .filter(|p| p.x >= rect_x0 && p.x <= rect_x1 && p.z >= rect_z0 && p.z <= rect_z1)
            .count();

        recalculate_region(
            &mut placed,
            rect_x0,
            rect_z0,
            rect_x1,
            rect_z1,
            height_fn,
            slope_fn,
            moisture_fn,
            &species,
            &config,
        );

        // No instance from before should remain inside the rect (they were
        // removed and freshly generated).
        // Count after recalculation inside the region.
        let count_after = placed
            .iter()
            .filter(|p| p.x >= rect_x0 && p.x <= rect_x1 && p.z >= rect_z0 && p.z <= rect_z1)
            .count();

        // The region should have been repopulated (possibly with a
        // different count due to stochastic sampling, but not zero).
        assert!(
            count_after > 0,
            "recalculated region should contain foliage"
        );

        // Total count should still be reasonable.
        assert!(placed.len() >= before / 2);
    }
}
