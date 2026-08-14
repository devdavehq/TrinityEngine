// src/destruction.rs
// ──────────────────────────────────────────────────────────────────────────────
// Fractured Mesh / Destruction System
//
// WHY IT EXISTS:
//   A "destroyed" box that still looks like one solid box is fake. Games feel
//   cheap when props blink out. Real destruction drops shards: a crate bursts
//   into planks, a wall crumbles into rubble, a statue topples into blocks.
//   Each shard is its own small physics body, so it tumbles, bounces, and
//   comes to rest exactly like debris in the real world.
//
// HOW IT WORKS:
//   1. Mark an entity with the `Destructible` component.
//   2. When that entity's Health drops to 0, the destruction system subdivides
//      its mesh into shards and spawns each shard as its own dynamic body.
//   3. Shards carry an outward blast velocity, live for a limited lifetime,
//      then despawn. The original mesh entity is removed.
//
// Shards inherit the parent entity's material, so a red crate shatters into
// red chunks. The split is deterministic (no RNG), keeping the system easy to
// test and debug.
// ──────────────────────────────────────────────────────────────────────────────

use hecs::{Entity, World};

use crate::assets::mesh::Vertex;
use crate::assets::Mesh;
use crate::components::{Collider, Health, Position, Renderable, RigidBody};

/// Marks an entity as destructible: when its Health hits zero it shatters.
#[derive(Clone, Copy)]
pub struct Destructible {
    /// How many shards the mesh splits into. clamped to 2..=48.
    pub shard_count: usize,
    /// Mass of each shard body (used for simple physics).
    pub shard_mass: f32,
    /// Outward blast velocity applied to each shard (units per second).
    pub blast_velocity: f32,
    /// Seconds a shard lives before it is despawned.
    pub shard_lifetime: f32,
}

impl Default for Destructible {
    fn default() -> Self {
        Self {
            shard_count: 8,
            shard_mass: 0.3,
            blast_velocity: 4.0,
            shard_lifetime: 6.0,
        }
    }
}

/// Marker on shard entities so the system knows to age them out.
#[derive(Clone, Copy)]
pub struct ShardMarker {
    /// Remaining lifetime in seconds.
    pub time_left: f32,
}

// ── Pure fracture ───────────────────────────────────────────────────────────
//
// Splits `vertices` (a triangle soup: every 3 vertices = one triangle) into
// `shards` groups by slicing the bounding box along its longest axis and
// bucketing each triangle by position. Shard offsets are built so each
// sub-mesh is centered on its own centroid — the entity transform places it.
//
// Returns `shards` vertex lists (in order along the longest axis).

pub fn split_mesh(vertices: &[Vertex], shards: usize) -> Vec<Vec<Vertex>> {
    let shards = shards.clamp(2, 48);
    if vertices.is_empty() {
        return Vec::new();
    }

    // Bounding box of the whole mesh.
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in vertices {
        for (i, p) in v.position.iter().enumerate() {
            min[i] = min[i].min(*p);
            max[i] = max[i].max(*p);
        }
    }
    let span = [
        max[0] - min[0],
        max[1] - min[1],
        max[2] - min[2],
    ];
    // Longest axis is the slicing axis.
    let axis = if span[0] >= span[1] && span[0] >= span[2] {
        0
    } else if span[1] >= span[2] {
        1
    } else {
        2
    };

    let mut buckets: Vec<Vec<Vertex>> = vec![Vec::new(); shards];
    for tri in vertices.chunks_exact(3) {
        // Centroid of this triangle.
        let c = [
            (tri[0].position[0] + tri[1].position[0] + tri[2].position[0]) / 3.0,
            (tri[0].position[1] + tri[1].position[1] + tri[2].position[1]) / 3.0,
            (tri[0].position[2] + tri[1].position[2] + tri[2].position[2]) / 3.0,
        ];
        let axis_span = span[axis].max(1e-6);
        let t = ((c[axis] - min[axis]) / axis_span).clamp(0.0, 0.9999);
        let idx = (t * shards as f32) as usize;
        let idx = idx.clamp(0, shards - 1);
        buckets[idx].extend_from_slice(tri);
    }

    // Re-center each shard on its own centroid so the shard entity's
    // transform determines world placement.
    for bucket in &mut buckets {
        if bucket.is_empty() {
            continue;
        }
        let mut cmin = [f32::INFINITY; 3];
        let mut cmax = [f32::NEG_INFINITY; 3];
        for v in bucket.iter() {
            for (i, p) in v.position.iter().enumerate() {
                cmin[i] = cmin[i].min(*p);
                cmax[i] = cmax[i].max(*p);
            }
        }
        let center = [
            (cmin[0] + cmax[0]) * 0.5,
            (cmin[1] + cmax[1]) * 0.5,
            (cmin[2] + cmax[2]) * 0.5,
        ];
        for v in bucket.iter_mut() {
            v.position[0] -= center[0];
            v.position[1] -= center[1];
            v.position[2] -= center[2];
        }
    }

    buckets
}

/// Compute the extent (half size) of a shard mesh, used to size its collider.
pub fn shard_half_extents(vertices: &[Vertex]) -> [f32; 3] {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in vertices {
        for (i, p) in v.position.iter().enumerate() {
            min[i] = min[i].min(*p);
            max[i] = max[i].max(*p);
        }
    }
    [
        (max[0] - min[0]) * 0.5,
        (max[1] - min[1]) * 0.5,
        (max[2] - min[2]) * 0.5,
    ]
}

// ── Fracture function ────────────────────────────────────────────────────────
//
// Destroys `entity`, spawning `destructible` shard entities. Requires the
// mesh store so each shard gets its own unique mesh handle.

pub fn fracture(
    world: &mut World,
    meshes: &mut crate::assets::AssetStore<Mesh>,
    entity: Entity,
) -> usize {
    let Some(params_ref) = world.get::<&Destructible>(entity).ok() else {
        return 0;
    };
    let params = *params_ref;
    drop(params_ref); // release the immutable borrow so we can spawn below

    // Gather source mesh + transform + material up-front so we don't hold a
    // borrow while spawning.
    let Some((mesh_handle, color, roughness, metallic, ao)) = world
        .get::<&Renderable>(entity)
        .ok()
        .map(|r| (r.mesh, r.color, r.roughness, r.metallic, r.ao))
    else {
        return 0;
    };
    let Some(pos) = world.get::<&Position>(entity).ok() else {
        return 0;
    };
    let center = [pos.x, pos.y, pos.z];
    drop(pos); // release the immutable borrow before mutating the world

    let Some(src) = meshes.get(&mesh_handle) else {
        return 0;
    };
    let shard_meshes = split_mesh(&src.vertices, params.shard_count);
    let _ = src; // release borrow on meshes before we add new ones below

    // Remove the original while we still have a clean handle on its data.
    let _ = world.despawn(entity);

    let mut spawned = 0;
    for shard_vertices in shard_meshes {
        if shard_vertices.is_empty() {
            continue;
        }
        let half = shard_half_extents(&shard_vertices);
        let s = half.map(|h| h.max(0.01));

        let shard_mesh = Mesh { vertices: shard_vertices };
        let handle = meshes.add(shard_mesh);

        let shard = world.spawn((
            Position {
                x: center[0],
                y: center[1],
                z: center[2],
            },
            crate::components::Rotation::default(),
            Renderable {
                mesh: handle,
                color,
                metallic,
                roughness,
                ao,
                scale: [s[0] * 2.0, s[1] * 2.0, s[2] * 2.0],
            },
            Collider {
                half_w: s[0],
                half_h: s[1],
                half_d: s[2],
                layer: 1,
                mask: 1,
            },
        ));

        // Dynamic body + outward impulse. Direction is deterministic: fan out
        // shards slightly by index so they separate instead of piling up.
        let mut body = RigidBody::dynamic();
        body.mass = params.shard_mass;
        let fan = ((spawned as f32 / params.shard_count.max(1) as f32) * std::f32::consts::TAU)
            .sin() as f32;
        body.velocity_x = params.blast_velocity * (0.4 + fan * 0.6);
        body.velocity_y = params.blast_velocity * 0.6;
        body._velocity_z = params.blast_velocity * (0.6 - fan * 0.4);
        let _ = world.insert(
            shard,
            (
                body,
                ShardMarker {
                    time_left: params.shard_lifetime,
                },
            ),
        );
        spawned += 1;
    }
    spawned
}

/// The system entry point wires into main.rs:
///   destruction_system(world, meshes, dt)
///
/// Each simulation frame it:
///   1. Fractures any entity that has a `Destructible` component and whose
///      `Health` is <= 0 (replaced by shards, then removed).
///   2. Ages existing shards and despawns them once their lifetime runs out,
///      so debris never accumulates forever.
///
/// Call it after physics and scripted damage have run, before rendering.
pub fn destruction_system(
    world: &mut World,
    meshes: &mut crate::assets::AssetStore<Mesh>,
    dt: f32,
) -> usize {
    // Step 1 — fracture anything whose health is gone.
    let doomed: Vec<hecs::Entity> = world
        .query::<(hecs::Entity, &Destructible, &Health)>()
        .iter()
        .filter(|(_, _, h)| h.current <= 0)
        .map(|(e, _, _)| e)
        .collect();

    let mut shards_spawned = 0;
    for entity in doomed {
        shards_spawned += fracture(world, meshes, entity);
    }

    // Step 2 — age shards and collect expired ones.
    let mut expired: Vec<hecs::Entity> = Vec::new();
    for (e, marker) in world.query::<(hecs::Entity, &ShardMarker)>().iter() {
        let remaining = marker.time_left - dt;
        if let Ok(mut m) = world.get::<&mut ShardMarker>(e) {
            m.time_left = remaining;
        }
        if remaining <= 0.0 {
            expired.push(e);
        }
    }
    for e in expired {
        let _ = world.despawn(e);
    }

    shards_spawned
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_mesh() -> Vec<Vertex> {
        // A unit box (8 corners, triangulated; simplified 12-triangle box).
        let s = 1.0;
        let corners = [
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        ];
        let faces: [[usize; 4]; 6] = [
            [0, 1, 2, 3], // far
            [4, 5, 6, 7], // near? whatever, it's a closed box
            [0, 1, 5, 4], // bottom
            [2, 3, 7, 6], // top
            [1, 2, 6, 5], // right
            [0, 3, 7, 4], // left
        ];
        let mut verts = Vec::new();
        for f in faces {
            let quad = [corners[f[0]], corners[f[1]], corners[f[2]], corners[f[3]]];
            // n = first edge cross second edge as rough normal
            let e1 = [
                quad[1][0] - quad[0][0],
                quad[1][1] - quad[0][1],
                quad[1][2] - quad[0][2],
            ];
            let e2 = [
                quad[3][0] - quad[0][0],
                quad[3][1] - quad[0][1],
                quad[3][2] - quad[0][2],
            ];
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            for (a, b, c) in [(0,1,2),(0,2,3)] {
                verts.push(Vertex::new(quad[a], n, [1.0,1.0,1.0]));
                verts.push(Vertex::new(quad[b], n, [1.0,1.0,1.0]));
                verts.push(Vertex::new(quad[c], n, [1.0,1.0,1.0]));
            }
        }
        verts
    }

    #[test]
    fn split_mesh_produces_expected_shard_count() {
        let mesh = box_mesh();
        let shards = split_mesh(&mesh, 4);
        assert_eq!(shards.len(), 4);
    }

    #[test]
    fn split_mesh_rejoins_to_original_vertex_count() {
        let mesh = box_mesh();
        let shards = split_mesh(&mesh, 5);
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, mesh.len());
    }

    #[test]
    fn split_mesh_empty_input_returns_empty() {
        let shards = split_mesh(&[], 4);
        assert!(shards.is_empty());
    }

    #[test]
    fn split_mesh_handles_single_triangle() {
        let mesh = vec![
            Vertex::new([0.0,0.0,0.0], [0.0,1.0,0.0], [1.0,1.0,1.0]),
            Vertex::new([1.0,0.0,0.0], [0.0,1.0,0.0], [1.0,1.0,1.0]),
            Vertex::new([0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,1.0,1.0]),
        ];
        let shards = split_mesh(&mesh, 4);
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, 3);
    }
}