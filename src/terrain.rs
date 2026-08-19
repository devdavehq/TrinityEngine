// src/terrain.rs
// ──────────────────────────────────────────────────────────────────────────────
// Terrain system — heightmap-based terrain with chunked LOD, biome rules,
// spline-based roads/paths, foliage placement, and erosion simulation.
//
// MODULES:
//   chunk    — terrain chunking, LOD, brushes, height sampling
//   biome    — data-driven biome rules with texture splatting layers
//   foliage  — Poisson-disk foliage placement with species and density maps
//   spline   — Catmull-Rom splines for roads/paths with terrain flattening
//   erosion  — hydraulic and thermal erosion simulation
//
// The TerrainWorld struct ties everything together: chunks + biomes + foliage
// + splines. It provides a single API for the editor and gameplay code.
// ──────────────────────────────────────────────────────────────────────────────

pub mod biome;
pub mod chunk;
pub mod csg;
pub mod erosion;
pub mod foliage;
pub mod spline;
pub mod streaming;

use crate::assets::{Handle, Mesh};
use crate::components::{Collider, FoliageWind, Position, Renderable, RigidBody};
use hecs::World;

use self::biome::{BiomeResult, MoistureMap, TerrainBiomeSystem, default_biome_config, evaluate_biome};
use self::chunk::{ChunkGrid, new_grid, sample_height, sample_normal, sample_slope};
use self::erosion::{ErosionConfig, ErosionState};
use self::foliage::{
    FoliageDensityMap, FoliagePlacement, PlacementConfig, PlacedFoliage, place_foliage,
};
use self::spline::{TerrainSpline, SplineSettings};
use serde::{Deserialize, Serialize};

// ── Legacy TerrainGrid (kept for backward compatibility) ─────────────────────

pub struct TerrainGrid {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,
    heights: Vec<f32>,
    pub material: TerrainMaterialProfile,
}

pub struct TerrainMaterialProfile {
    pub grass_color: [f32; 3],
    pub dirt_color: [f32; 3],
    pub rock_color: [f32; 3],
    pub slope_rock_start: f32,
    pub height_rock_start: f32,
}

impl TerrainGrid {
    pub fn new(width: usize, depth: usize, cell_size: f32) -> Self {
        Self {
            width,
            depth,
            cell_size,
            heights: vec![0.0; width * depth],
            material: TerrainMaterialProfile::default(),
        }
    }

    pub fn raise_brush(&mut self, cx: usize, cz: usize, radius: usize, amount: f32) {
        let r2 = (radius * radius) as i32;
        for z in 0..self.depth {
            for x in 0..self.width {
                let dx = x as i32 - cx as i32;
                let dz = z as i32 - cz as i32;
                let d2 = dx * dx + dz * dz;
                if d2 <= r2 {
                    let falloff = 1.0 - (d2 as f32 / r2.max(1) as f32);
                    let idx = z * self.width + x;
                    self.heights[idx] += amount * falloff;
                }
            }
        }
    }

    pub fn lower_brush(&mut self, cx: usize, cz: usize, radius: usize, amount: f32) {
        self.raise_brush(cx, cz, radius, -amount);
    }

    pub fn sample_height(&self, x: usize, z: usize) -> f32 {
        self.heights[z * self.width + x]
    }

    pub fn sample_height_world(&self, world_x: f32, world_z: f32) -> f32 {
        let gx = ((world_x / self.cell_size) + (self.width as f32 * 0.5))
            .clamp(0.0, (self.width.saturating_sub(1)) as f32) as usize;
        let gz = ((world_z / self.cell_size) + (self.depth as f32 * 0.5))
            .clamp(0.0, (self.depth.saturating_sub(1)) as f32) as usize;
        self.sample_height(gx, gz)
    }

    pub fn sample_slope_world(&self, world_x: f32, world_z: f32) -> f32 {
        let gx = ((world_x / self.cell_size) + (self.width as f32 * 0.5))
            .clamp(1.0, (self.width.saturating_sub(2)) as f32) as usize;
        let gz = ((world_z / self.cell_size) + (self.depth as f32 * 0.5))
            .clamp(1.0, (self.depth.saturating_sub(2)) as f32) as usize;

        let h_l = self.sample_height(gx - 1, gz);
        let h_r = self.sample_height(gx + 1, gz);
        let h_d = self.sample_height(gx, gz - 1);
        let h_u = self.sample_height(gx, gz + 1);
        let dx = (h_r - h_l) / (2.0 * self.cell_size.max(0.001));
        let dz = (h_u - h_d) / (2.0 * self.cell_size.max(0.001));
        (dx * dx + dz * dz).sqrt()
    }

    pub fn auto_surface_color_world(&self, world_x: f32, world_z: f32) -> [f32; 3] {
        let h = self.sample_height_world(world_x, world_z);
        let slope = self.sample_slope_world(world_x, world_z);
        self.material.blend_color(h, slope)
    }
}

impl TerrainMaterialProfile {
    fn blend_color(&self, height: f32, slope: f32) -> [f32; 3] {
        let rock_from_slope = ((slope - self.slope_rock_start) * 2.0).clamp(0.0, 1.0);
        let rock_from_height = ((height - self.height_rock_start) * 0.8).clamp(0.0, 1.0);
        let rock_w = rock_from_slope.max(rock_from_height);
        let dirt_w = (1.0 - rock_w) * (0.45 + slope.clamp(0.0, 1.0) * 0.35);
        let grass_w = (1.0 - rock_w - dirt_w).clamp(0.0, 1.0);
        let sum = (grass_w + dirt_w + rock_w).max(0.0001);
        let gw = grass_w / sum;
        let dw = dirt_w / sum;
        let rw = rock_w / sum;
        [
            self.grass_color[0] * gw + self.dirt_color[0] * dw + self.rock_color[0] * rw,
            self.grass_color[1] * gw + self.dirt_color[1] * dw + self.rock_color[1] * rw,
            self.grass_color[2] * gw + self.dirt_color[2] * dw + self.rock_color[2] * rw,
        ]
    }
}

impl Default for TerrainMaterialProfile {
    fn default() -> Self {
        Self {
            grass_color: [0.25, 0.52, 0.23],
            dirt_color: [0.40, 0.31, 0.22],
            rock_color: [0.46, 0.46, 0.48],
            slope_rock_start: 0.45,
            height_rock_start: 2.2,
        }
    }
}

// ── Foliage Spawning (legacy, kept for backward compatibility) ───────────────

pub fn spawn_foliage_ring(
    world: &mut World,
    mesh_handle: Handle<Mesh>,
    center_x: f32,
    center_z: f32,
    radius: f32,
    count: usize,
    with_tree_physics: bool,
) {
    for i in 0..count {
        let t = i as f32 / count.max(1) as f32;
        let angle = t * std::f32::consts::TAU;
        let x = center_x + angle.cos() * radius;
        let z = center_z + angle.sin() * radius;
        let scale = 0.25 + (i as f32 % 7.0) * 0.02;
        let e = world.spawn((
            Position { x, y: 0.0, z },
            Renderable {
                mesh: mesh_handle,
                color: [0.18, 0.46, 0.20],
                metallic: 0.0,
                roughness: 0.92,
                ao: 1.0,
                scale: [scale, scale * 3.0, scale],
            },
        ));

        if with_tree_physics {
            let mut foliage_body = RigidBody::kinematic();
            foliage_body.friction = 0.85;
            let _ = world.insert(
                e,
                (
                    foliage_body,
                    Collider {
                        half_w: scale * 0.5,
                        half_h: scale * 1.5,
                        half_d: scale * 0.5,
                        layer: 1,
                        mask: 1,
                    },
                    FoliageWind {
                        base_x: x,
                        base_z: z,
                        amplitude: 0.08 + scale * 0.2,
                        frequency: 1.2 + (i % 5) as f32 * 0.25,
                    },
                ),
            );
        }
    }
}

pub fn remove_nearby_foliage(world: &mut World, center_x: f32, center_z: f32, radius: f32) -> usize {
    let r2 = radius * radius;
    let to_remove: Vec<hecs::Entity> = world
        .query::<(hecs::Entity, &Position, &Renderable)>()
        .iter()
        .filter_map(|(e, pos, _)| {
            let dx = pos.x - center_x;
            let dz = pos.z - center_z;
            if dx * dx + dz * dz <= r2 {
                Some(e)
            } else {
                None
            }
        })
        .collect();
    let count = to_remove.len();
    for e in to_remove {
        let _ = world.despawn(e);
    }
    count
}

// ── TerrainWorld: integrated terrain system ──────────────────────────────────

/// GPU terrain layer — one splatting texture with PBR properties.
/// Up to 4 layers blend per texel via the biome system.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TerrainLayer {
    /// Display name (e.g. "Grass", "Dirt", "Rock", "Snow").
    pub name: String,
    /// Path to albedo texture.
    pub albedo_texture: String,
    /// Path to normal map texture.
    pub normal_texture: String,
    /// Path to roughness/metallic/AO (glTF convention: B=metallic, G=roughness, R=AO).
    pub material_texture: String,
    /// Tiling factor (how many times the texture repeats per world-space unit).
    pub tiling: f32,
    /// Per-layer roughness multiplier (0..2).
    pub roughness_scale: f32,
    /// Per-layer metallic multiplier (0..1).
    pub metallic_scale: f32,
    /// Per-layer normal strength (0 = flat, 1 = full, 2 = exaggerated).
    pub normal_strength: f32,
}

impl Default for TerrainLayer {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            albedo_texture: String::new(),
            normal_texture: String::new(),
            material_texture: String::new(),
            tiling: 10.0,
            roughness_scale: 1.0,
            metallic_scale: 0.0,
            normal_strength: 1.0,
        }
    }
}

/// Settings for spline-based terrain flattening.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TerrainFlattenSettings {
    /// How far from the spline center to flatten (world units).
    pub flatten_radius: f32,
    /// Blend factor (0..1) for flattening strength.
    pub flatten_strength: f32,
    /// Distance over which terrain blends back to natural slope.
    pub slope_blend_distance: f32,
}

impl Default for TerrainFlattenSettings {
    fn default() -> Self {
        Self {
            flatten_radius: 8.0,
            flatten_strength: 0.85,
            slope_blend_distance: 12.0,
        }
    }
}

/// Persistent state for incremental erosion (avoids re-allocating water/sediment buffers).
pub struct TerrainErosionState {
    pub state: ErosionState,
    pub config: ErosionConfig,
}

impl TerrainErosionState {
    pub fn new(width: usize, depth: usize) -> Self {
        Self {
            state: ErosionState::new_state(width, depth),
            config: ErosionConfig::default(),
        }
    }
}

/// The main integrated terrain world — ties together chunks, biomes, foliage,
/// splines, and erosion into a single coherent system.
pub struct TerrainWorld {
    /// Chunked heightmap grid with LOD support.
    pub grid: ChunkGrid,

    /// Biome evaluation system (rules + moisture).
    pub biome_system: TerrainBiomeSystem,

    /// All splines (roads, paths, rivers).
    pub splines: Vec<TerrainSpline>,

    /// Spline flattening settings.
    pub flatten_settings: TerrainFlattenSettings,

    /// Foliage placement config and results.
    pub foliage: FoliagePlacement,
    pub placed_foliage: Vec<PlacedFoliage>,
    pub foliage_config: PlacementConfig,

    /// PBR terrain layers (up to 4 for splatting).
    pub layers: Vec<TerrainLayer>,

    /// Erosion state for incremental simulation.
    pub erosion: TerrainErosionState,

    /// Auto-surface material blend settings.
    pub material: TerrainMaterialProfile,

    /// Grid dimensions for erosion state allocation.
    width: usize,
    depth: usize,
    pub cell_size: f32,
}

impl TerrainWorld {
    /// Create a new terrain world with default biome rules.
    pub fn new(width: usize, depth: usize, chunk_size: usize, cell_size: f32) -> Self {
        let grid = new_grid(width, depth, chunk_size, cell_size);
        let moisture_map = MoistureMap::new(width, depth, cell_size);
        let biome_system = TerrainBiomeSystem::new(default_biome_config(), moisture_map);
        let erosion = TerrainErosionState::new(width, depth);

        Self {
            grid,
            biome_system,
            splines: Vec::new(),
            flatten_settings: TerrainFlattenSettings::default(),
            foliage: FoliagePlacement {
                species: Vec::new(),
                density_map: FoliageDensityMap::new(width, depth, cell_size),
                total_max_plants: 0,
                exclusion_zones: Vec::new(),
            },
            placed_foliage: Vec::new(),
            foliage_config: PlacementConfig {
                min_distance_between: 2.0,
                max_attempts: 30,
                random_seed: 0,
            },
            material: TerrainMaterialProfile::default(),
            layers: Self::default_layers(),
            erosion,
            width,
            depth,
            cell_size,
        }
    }

    /// Default PBR terrain layers: Grass, Dirt, Rock, Snow.
    fn default_layers() -> Vec<TerrainLayer> {
        vec![
            TerrainLayer {
                name: "Grass".to_string(),
                tiling: 12.0,
                roughness_scale: 0.9,
                normal_strength: 1.0,
                ..Default::default()
            },
            TerrainLayer {
                name: "Dirt".to_string(),
                tiling: 10.0,
                roughness_scale: 1.0,
                metallic_scale: 0.0,
                normal_strength: 0.8,
                ..Default::default()
            },
            TerrainLayer {
                name: "Rock".to_string(),
                tiling: 8.0,
                roughness_scale: 0.7,
                normal_strength: 1.2,
                ..Default::default()
            },
            TerrainLayer {
                name: "Snow".to_string(),
                tiling: 15.0,
                roughness_scale: 0.3,
                metallic_scale: 0.05,
                normal_strength: 0.5,
                ..Default::default()
            },
        ]
    }

    /// Sample height at a world position.
    pub fn height_at(&self, world_x: f32, world_z: f32) -> f32 {
        sample_height(&self.grid, world_x, world_z)
    }

    /// Sample surface normal at a world position.
    pub fn normal_at(&self, world_x: f32, world_z: f32) -> [f32; 3] {
        sample_normal(&self.grid, world_x, world_z)
    }

    /// Sample slope at a world position.
    pub fn slope_at(&self, world_x: f32, world_z: f32) -> f32 {
        sample_slope(&self.grid, world_x, world_z)
    }

    /// Compute auto-surface color using height/slope material blending.
    pub fn auto_surface_color_world(&self, world_x: f32, world_z: f32) -> [f32; 3] {
        let h = self.height_at(world_x, world_z);
        let slope = self.slope_at(world_x, world_z);
        self.material.blend_color(h, slope)
    }

    /// Evaluate biome at a world position — returns splatting weights for up to 4 layers.
    pub fn biome_at(&self, world_x: f32, world_z: f32) -> BiomeResult {
        let h = self.height_at(world_x, world_z);
        let s = self.slope_at(world_x, world_z);
        let m = self.biome_system.moisture.sample_world(world_x, world_z);
        evaluate_biome(h, s, m, &self.biome_system.config)
    }

    // ── Brush Operations ────────────────────────────────────────────────

    /// Raise terrain at a world position with quadratic falloff.
    pub fn raise(&mut self, world_x: f32, world_z: f32, radius: f32, amount: f32) {
        chunk::raise_brush(&mut self.grid, world_x, world_z, radius, amount);
    }

    /// Lower terrain at a world position.
    pub fn lower(&mut self, world_x: f32, world_z: f32, radius: f32, amount: f32) {
        self.raise(world_x, world_z, radius, -amount);
    }

    /// Smooth terrain at a world position.
    pub fn smooth(&mut self, world_x: f32, world_z: f32, radius: f32, strength: f32) {
        chunk::smooth_brush(&mut self.grid, world_x, world_z, radius, strength);
    }

    /// Flatten terrain toward a target height.
    pub fn flatten(&mut self, world_x: f32, world_z: f32, radius: f32, target: f32, blend: f32) {
        chunk::flatten_brush(&mut self.grid, world_x, world_z, radius, target, blend);
    }

    // ── Spline Operations ───────────────────────────────────────────────

    /// Add a spline (road/path) to the terrain.
    pub fn add_spline(&mut self, spline: TerrainSpline) {
        self.splines.push(spline);
    }

    /// Flatten terrain along all splines.
    pub fn apply_spline_flattening(&mut self) {
        let splines = std::mem::take(&mut self.splines);
        let heights = chunk::total_heights(&self.grid);
        let mut h = heights;

        for spline in &splines {
            spline::flatten_terrain(
                &mut h,
                self.width,
                self.depth,
                self.cell_size,
                spline,
                &SplineSettings {
                    subdivision_steps: 8,
                    flatten_radius: self.flatten_settings.flatten_radius,
                    flatten_strength: self.flatten_settings.flatten_strength,
                    slope_blend_distance: self.flatten_settings.slope_blend_distance,
                },
            );
        }

        // Write heights back to grid.
        for z in 0..self.depth {
            for x in 0..self.width {
                let world_x = x as f32 * self.cell_size;
                let world_z = z as f32 * self.cell_size;
                let val = h[z * self.width + x];
                chunk::set_height(&mut self.grid, world_x, world_z, val);
            }
        }

        self.splines = splines;
    }

    // ── Foliage Operations ──────────────────────────────────────────────

    /// Place all foliage using Poisson disk sampling and biome rules.
    pub fn generate_foliage(&mut self) {
        let grid = &self.grid;
        let moisture = &self.biome_system.moisture;
        let h_fn = |wx: f32, wz: f32| sample_height(grid, wx, wz);
        let s_fn = |wx: f32, wz: f32| sample_slope(grid, wx, wz);
        let m_fn = |wx: f32, wz: f32| moisture.sample_world(wx, wz);

        self.placed_foliage = place_foliage(h_fn, s_fn, m_fn, &self.foliage, &self.foliage_config);
    }

    /// Remove all foliage within a radius (e.g., when placing a building).
    pub fn remove_foliage_in(&mut self, center_x: f32, center_z: f32, radius: f32) -> usize {
        foliage::remove_nearby(&mut self.placed_foliage, center_x, center_z, radius)
    }

    /// Re-evaluate foliage in a rectangular region after terrain edits.
    pub fn refresh_foliage_region(&mut self, min_x: f32, min_z: f32, max_x: f32, max_z: f32) {
        let grid = &self.grid;
        let moisture = &self.biome_system.moisture;
        let h_fn = |wx: f32, wz: f32| sample_height(grid, wx, wz);
        let s_fn = |wx: f32, wz: f32| sample_slope(grid, wx, wz);
        let m_fn = |wx: f32, wz: f32| moisture.sample_world(wx, wz);

        foliage::recalculate_region(
            &mut self.placed_foliage,
            min_x,
            min_z,
            max_x,
            max_z,
            h_fn,
            s_fn,
            m_fn,
            &self.foliage.species,
            &self.foliage_config,
        );
    }

    // ── Erosion ─────────────────────────────────────────────────────────

    /// Run full erosion (hydraulic + thermal) across the entire terrain.
    pub fn erode_all(&mut self) {
        let heights = chunk::total_heights(&self.grid);
        let mut h = heights;

        erosion::erode(
            &mut h,
            self.width,
            self.depth,
            self.cell_size,
            &self.erosion.config,
        );

        self.write_heights_back(&h);
    }

    /// Run erosion on a rectangular region only (for incremental updates after editing).
    pub fn erode_region(&mut self, min_x: f32, min_z: f32, max_x: f32, max_z: f32) {
        let cx = ((min_x / self.cell_size) + self.width as f32 * 0.5).max(0.0) as usize;
        let cz = ((min_z / self.cell_size) + self.depth as f32 * 0.5).max(0.0) as usize;
        let cx2 = ((max_x / self.cell_size) + self.width as f32 * 0.5).min(self.width as f32 - 1.0) as usize;
        let cz2 = ((max_z / self.cell_size) + self.depth as f32 * 0.5).min(self.depth as f32 - 1.0) as usize;

        let heights = chunk::total_heights(&self.grid);
        let mut h = heights;

        erosion::erode_region(
            &mut h,
            &mut self.erosion.state,
            self.width,
            self.depth,
            self.cell_size,
            &self.erosion.config,
            cx.min(cx2),
            cx.max(cx2),
            cz.min(cz2),
            cz.max(cz2),
        );

        self.write_heights_back(&h);
    }

    /// Helper: write a flat heightmap back to the chunk grid.
    fn write_heights_back(&mut self, heights: &[f32]) {
        for z in 0..self.depth {
            for x in 0..self.width {
                let world_x = x as f32 * self.cell_size;
                let world_z = z as f32 * self.cell_size;
                chunk::set_height(&mut self.grid, world_x, world_z, heights[z * self.width + x]);
            }
        }
    }

    // ── LOD ─────────────────────────────────────────────────────────────

    /// Update chunk LOD levels based on camera position.
    pub fn update_lod(&mut self, camera_x: f32, camera_z: f32, lod_distances: &[f32]) {
        chunk::update_lods(&mut self.grid, camera_x, camera_z, lod_distances);
    }

    /// Get list of chunks that need mesh rebuild.
    pub fn dirty_chunks(&self) -> Vec<(i32, i32)> {
        chunk::dirty_chunks(&self.grid)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_world_height_sampling() {
        let tw = TerrainWorld::new(64, 64, 16, 1.0);
        // Flat terrain should return 0.0 at center.
        let h = tw.height_at(0.0, 0.0);
        assert!((h - 0.0).abs() < 0.001);
    }

    #[test]
    fn terrain_world_brush_and_biome() {
        let mut tw = TerrainWorld::new(64, 64, 16, 1.0);
        // Raise terrain at center.
        tw.raise(0.0, 0.0, 10.0, 5.0);
        let h = tw.height_at(0.0, 0.0);
        assert!(h > 0.0, "Height should be raised, got {}", h);

        // Biome evaluation should still work and return a valid result.
        let biome = tw.biome_at(0.0, 0.0);
        let sum: f32 = biome.blend_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "blend weights should sum to 1.0, got {sum}");
    }

    #[test]
    fn terrain_world_spline_flattening() {
        let mut tw = TerrainWorld::new(64, 64, 16, 1.0);
        // Raise entire terrain.
        tw.raise(0.0, 0.0, 60.0, 3.0);

        // Add a straight spline with elevation offset so flattening actually lowers terrain.
        let mut spline = TerrainSpline::new("Road");
        spline.points.push(spline::SplinePoint::full(0.0, -30.0, 4.0, 5.0, 0));
        spline.points.push(spline::SplinePoint::full(0.0, 30.0, 4.0, 5.0, 0));
        tw.add_spline(spline);

        tw.apply_spline_flattening();

        // Height along spline path should be lower than surroundings.
        let h_center = tw.height_at(0.0, 0.0);
        let h_off = tw.height_at(20.0, 0.0);
        assert!(
            h_center < h_off,
            "Center ({}) should be flatter than off-spline ({})",
            h_center,
            h_off
        );
    }

    #[test]
    fn terrain_world_default_layers() {
        let tw = TerrainWorld::new(32, 32, 8, 1.0);
        assert_eq!(tw.layers.len(), 4);
        assert_eq!(tw.layers[0].name, "Grass");
        assert_eq!(tw.layers[1].name, "Dirt");
        assert_eq!(tw.layers[2].name, "Rock");
        assert_eq!(tw.layers[3].name, "Snow");
    }

    #[test]
    fn terrain_world_smooth_brush() {
        let mut tw = TerrainWorld::new(32, 32, 8, 1.0);
        // Create a spike.
        tw.raise(0.0, 0.0, 3.0, 10.0);
        let h_before = tw.height_at(0.0, 0.0);
        // Smooth it.
        tw.smooth(0.0, 0.0, 5.0, 0.5);
        let h_after = tw.height_at(0.0, 0.0);
        assert!(
            h_after < h_before,
            "Smoothing should reduce spike: before={}, after={}",
            h_before,
            h_after
        );
    }
}
