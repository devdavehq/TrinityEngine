use crate::assets::{Handle, Mesh};
use crate::components::{Collider, FoliageWind, Position, Renderable, RigidBody};
use hecs::World;

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
            let _ = world.insert(
                e,
                (
                    RigidBody {
                        velocity_x: 0.0,
                        velocity_y: 0.0,
                        _velocity_z: 0.0,
                        on_ground: true,
                        use_gravity: false,
                    },
                    Collider {
                        half_w: scale * 0.5,
                        half_h: scale * 1.5,
                        half_d: scale * 0.5,
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

