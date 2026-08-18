//! CPU light probe baker — the "indirect light" backbone.
//!
//! Horizon Forbidden West's look comes mostly from baked, precomputed light:
//! irradiance volumes + a voxelized albedo cache computed offline. This module
//! provides the first, lightweight half of that: an offline SH light probe
//! baker.
//!
//! How it works:
//! - The scene's world-space triangles are gathered from the ECS `World` +
//!   `AssetStore<Mesh>` (mesh data is kept CPU-side forever, so no extraction
//!   round-trip is needed).
//! - A median-split BVH is built over those triangles for fast ray casts.
//! - For each probe, `samples` directions are generated uniformly over the
//!   sphere (golden-spiral). A ray is cast per direction; the radiance
//!   arriving along that direction is:
//!     * if it hits a surface — that surface's direct contribution from the
//!       sun + point/spot lights (each shadow-cast via a second ray) plus a
//!       cheap one-bounce albedo*sky term;
//!     * if it misses — the sky colour (IBL average).
//! - The radiance field is projected into L2 spherical harmonics, matching the
//!   layout of `IrradianceSH` (`[coeff; 9]` RGB) so the existing
//!   `evaluate_sh()` / `interpolate()` in `light_probes.rs` work unchanged.
//!
//! The result is saved as JSON beside the scene and loaded on level load, so
//! the "bake" is a one-time offline step in the editor — the running game
//! just samples precomputed coefficients.

use std::path::Path;

use glam::{Mat4, Vec3};
use hecs::World;

use crate::assets::mesh::{Mesh, Vertex};
use crate::assets::store::AssetStore;
use crate::components::{PointLight, Position, Renderable, Rotation};
use crate::renderer::light_probes::{IrradianceSH, LightProbeGrid, ProbeVolume};

// ── Small deterministic PRNG (no external rand dependency) ──────────────────

struct Prng(u64);

impl Prng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    /// Uniform float in [0,1).
    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64) as f32
    }

    /// Uniform unit vec3 over the sphere.
    fn sphere(&mut self) -> Vec3 {
        let u = self.next_f32();
        let v = self.next_f32();
        let theta = 2.0 * std::f32::consts::PI * u;
        let z = 2.0 * v - 1.0;
        let r = (1.0 - z * z).sqrt();
        Vec3::new(r * theta.cos(), r * theta.sin(), z)
    }
}

// ── Scene triangle collection ───────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Tri {
    a: Vec3,
    b: Vec3,
    c: Vec3,
}

/// A finite light source (sun = directional with large fixed range).
struct SceneLight {
    pos: Vec3,
    dir: Vec3,
    color: Vec3,
    intensity: f32,
    range: f32,
    is_directional: bool,
}

pub(crate) struct BakeScene {
    tris: Vec<Tri>,
    albedo: Vec<Vec3>,
    lights: Vec<SceneLight>,
    sky: Vec3,
}

// Fast ray-vs-triangle (Möller–Trumbore).
fn ray_tri(origin: Vec3, dir: Vec3, t: &Tri) -> Option<f32> {
    let e1 = t.b - t.a;
    let e2 = t.c - t.a;
    let p = dir.cross(e1);
    let det = e2.dot(p);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / det;
    let s = origin - t.a;
    let u = s.dot(p) * inv;
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = s.cross(e2);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t_hit = e1.dot(q) * inv;
    if t_hit <= 0.0 {
        return None;
    }
    Some(t_hit)
}

// ── Median-split BVH ────────────────────────────────────────────────────────

enum BvhNode {
    Leaf { start: usize, end: usize },
    Branch { left: Box<BvhNode>, right: Box<BvhNode>, bounds: (Vec3, Vec3) },
}

fn aabb_tris(tris: &[Tri], indices: &[usize], start: usize, end: usize) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for &i in &indices[start..end] {
        for p in [tris[i].a, tris[i].b, tris[i].c] {
            min = min.min(p);
            max = max.max(p);
        }
    }
    (min, max)
}

fn build_bvh(tris: &[Tri], indices: &mut Vec<usize>, start: usize, end: usize, depth: usize) -> BvhNode {
    if end - start <= 8 || depth > 32 {
        return BvhNode::Leaf { start, end };
    }
    let (min, max) = aabb_tris(tris, indices, start, end);
    let extents = max - min;
    let axis = if extents.x >= extents.y && extents.x >= extents.z {
        0
    } else if extents.y >= extents.z {
        1
    } else {
        2
    };
    let mid = (min + max) / 2.0;
    let mut i = start;
    let mut j = end;
    while i < j {
        let ci = indices[i];
        let c = (tris[ci].a + tris[ci].b + tris[ci].c) / 3.0;
        if c[axis] <= mid[axis] {
            i += 1;
        } else {
            j -= 1;
            indices.swap(i, j);
        }
    }
    if i == start || i == end {
        i = (start + end) / 2;
    }
    BvhNode::Branch {
        left: Box::new(build_bvh(tris, indices, start, i, depth + 1)),
        right: Box::new(build_bvh(tris, indices, i, end, depth + 1)),
        bounds: (min, max),
    }
}

fn raycast(node: &BvhNode, tris: &[Tri], albedo: &[Vec3], indices: &[usize], origin: Vec3, dir: Vec3) -> Option<(f32, Vec3)> {
    match node {
        BvhNode::Leaf { start, end } => {
            let mut best: Option<(f32, Vec3)> = None;
            for &i in &indices[*start..*end] {
                if let Some(dist) = ray_tri(origin, dir, &tris[i]) {
                    if best.map_or(true, |(b, _)| dist < b) {
                        best = Some((dist, albedo[i]));
                    }
                }
            }
            best
        }
        BvhNode::Branch { left, right, bounds } => {
            let (min, max) = *bounds;
            // AABB slab test.
            let mut t_min = 0.0_f32;
            let mut t_max = f32::MAX;
            for axis in 0..3 {
                if dir[axis].abs() < 1e-8 {
                    if origin[axis] < min[axis] || origin[axis] > max[axis] {
                        return None;
                    }
                } else {
                    let inv = 1.0 / dir[axis];
                    let mut t1 = (min[axis] - origin[axis]) * inv;
                    let mut t2 = (max[axis] - origin[axis]) * inv;
                    if t1 > t2 {
                        std::mem::swap(&mut t1, &mut t2);
                    }
                    t_min = t_min.max(t1);
                    t_max = t_max.min(t2);
                    if t_min > t_max {
                        return None;
                    }
                }
            }
            let left_hit = raycast(left, tris, albedo, indices, origin, dir);
            let right_hit = raycast(right, tris, albedo, indices, origin, dir);
            match (left_hit, right_hit) {
                (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            }
        }
    }
}

// ── SH basis (same convention as light_probes.rs) ───────────────────────────

fn sh_basis(dir: Vec3) -> [f32; 9] {
    let (x, y, z) = (dir.x, dir.y, dir.z);
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

// ── Bake driver ─────────────────────────────────────────────────────────────

pub struct BakeSettings {
    /// Rays per probe. Higher = smoother but slower (256–1024 is typical).
    pub samples: usize,
    /// Sun direction (pointing from sun → scene, matching light_dir uniform).
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
    pub sun_intensity: f32,
    /// Average sky/IBL colour used when a probe ray misses everything.
    pub sky_color: Vec3,
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            samples: 512,
            sun_dir: Vec3::new(-0.4, -0.7, -0.2).normalize(),
            sun_color: Vec3::new(1.0, 0.95, 0.85),
            sun_intensity: 3.0,
            sky_color: Vec3::new(0.35, 0.5, 0.8),
        }
    }
}

/// Gather the world + meshes into a CPU bake-ready triangle soup.
pub fn collect_scene(world: &World, meshes: &AssetStore<Mesh>) -> BakeScene {
    let mut tris: Vec<Tri> = Vec::new();
    let mut albedo_flat: Vec<Vec3> = Vec::new();

    for (pos, renderable, rot) in world
        .query::<(&Position, &Renderable, Option<&Rotation>)>()
        .iter()
    {
        let Some(mesh) = meshes.get(&renderable.mesh) else { continue };
        let rot = rot.copied().unwrap_or(Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 });
        let model = Mat4::from_translation(Vec3::new(pos.x, pos.y, pos.z))
            * Mat4::from_rotation_y(rot.yaw)
            * Mat4::from_rotation_x(rot.pitch)
            * Mat4::from_rotation_z(rot.roll)
            * Mat4::from_scale(Vec3::from(renderable.scale));

        let base_albedo = Vec3::from(renderable.color).max(Vec3::splat(0.01));
        transform_mesh(&mesh.vertices, &model, base_albedo, &mut tris, &mut albedo_flat);
    }

    // Lights: sun handled by caller settings; gather PointLight entities.
    let mut lights: Vec<SceneLight> = Vec::new();
    for (pos, pl, rot) in world.query::<(&Position, &PointLight, Option<&Rotation>)>().iter() {
        let color = Vec3::from(pl.color);
        let lt = pl.light_type as u32;
        let dir = match rot {
            Some(r) => {
                let fwd = Mat4::from_rotation_y(r.yaw) * Mat4::from_rotation_x(r.pitch);
                let d = fwd.transform_vector3(Vec3::new(0.0, 0.0, -1.0)).normalize();
                d
            }
            None => Vec3::new(0.0, 0.0, -1.0),
        };
        if lt == 0 {
            // Extra directional light.
            lights.push(SceneLight {
                pos: Vec3::splat(0.0),
                dir,
                color: color * pl.intensity,
                intensity: 1.0,
                range: 0.0,
                is_directional: true,
            });
        } else {
            lights.push(SceneLight {
                pos: Vec3::new(pos.x, pos.y, pos.z),
                dir,
                color,
                intensity: pl.intensity.max(0.001),
                range: pl.range.max(0.01),
                is_directional: false,
            });
        }
    }

    BakeScene {
        tris,
        albedo: albedo_flat,
        lights,
        sky: BakeSettings::default().sky_color,
    }
}

fn transform_mesh(
    verts: &[Vertex],
    model: &Mat4,
    base_albedo: Vec3,
    tris: &mut Vec<Tri>,
    albedo: &mut Vec<Vec3>,
) {
    // Triangle soup: every 3 consecutive vertices form one triangle.
    for chunk in verts.chunks_exact(3) {
        let a = model.transform_point3(Vec3::from(chunk[0].position));
        let b = model.transform_point3(Vec3::from(chunk[1].position));
        let c = model.transform_point3(Vec3::from(chunk[2].position));
        tris.push(Tri { a, b, c });
        albedo.push(base_albedo);
    }
}

/// Scene-space AABB over the triangle soup — used to auto-place probes.
fn scene_bounds(scene: &BakeScene) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for t in &scene.tris {
        min = min.min(t.a).min(t.b).min(t.c);
        max = max.max(t.a).max(t.b).max(t.c);
    }
    if min.x > max.x {
        (Vec3::splat(-16.0), Vec3::splat(16.0))
    } else {
        (min, max)
    }
}

/// Bake SH irradiance for every probe in `grid`.
pub fn bake_probe_grid(
    grid: &mut LightProbeGrid,
    scene: &BakeScene,
    settings: &BakeSettings,
) -> Result<u64, String> {
    if scene.tris.is_empty() {
        return Err("bake aborted: no meshes in the scene".into());
    }

    // Auto-place a probe grid if the user hasn't placed probes yet. The
    // shader supports up to 32 probes, so 3×2×3 = 18 comfortably fits.
    if grid.probes.is_empty() {
        let (min, max) = scene_bounds(scene);
        let span = (max - min).max(Vec3::splat(12.0));
        let cells = [3usize, 2usize, 3usize];
        for x in 0..cells[0] {
            for y in 0..cells[1] {
                for z in 0..cells[2] {
                    let t = Vec3::new(
                        (x as f32 + 0.5) / cells[0] as f32,
                        (y as f32 + 0.5) / cells[1] as f32,
                        (z as f32 + 0.5) / cells[2] as f32,
                    );
                    let pos = min + span * t;
                    // ~1.5× the largest cell edge so volumes overlap smoothly.
                    let radius = span.max_element() * 1.5 / 2.0;
                    grid.add_probe(pos, radius);
                }
            }
        }
    }

    // Build BVH.
    let mut indices: Vec<usize> = (0..scene.tris.len()).collect();
    let root = build_bvh(&scene.tris, &mut indices, 0, scene.tris.len(), 0);

    // Accumulate sun into the light list as a directional.
    let mut rng = Prng::new(0x5EED_BBEE);
    let _ = rng.sphere(); // warm up

    for probe in grid.probes.iter_mut() {
        let origin = probe.position;
        let mut coeffs = [[0.0f32; 3]; 9];
        let inv_n = 1.0 / settings.samples as f32;

        // Sky-hemisphere occlusion: how much of the upper hemisphere's
        // geometry blocks the sky at this probe. Only rays above the horizon
        // count, so an open field reads ~0 and a room/underhang reads ~1.
        let mut up_rays = 0usize;
        let mut up_blocked = 0usize;

        for _ in 0..settings.samples {
            let dir = rng.sphere();
            if dir.y > 0.0 {
                up_rays += 1;
            }
            // Skip rays grazing the probe — they contribute ~nothing.
            let hit = raycast(&root, &scene.tris, &scene.albedo, &indices, origin, dir);
            if dir.y > 0.0 && hit.is_some() {
                up_blocked += 1;
            }
            let radiance: Vec3 = match hit {
                Some((_, alb)) => {
                    // Surface bounce: ambient sky albedo + direct lights.
                    let mut l = alb * scene.sky * 0.35;
                    // Sun directional.
                    let ndl = (-settings.sun_dir).dot(dir).max(0.0);
                    if ndl > 0.0
                        && raycast(
                            &root,
                            &scene.tris,
                            &scene.albedo,
                            &indices,
                            origin,
                            -settings.sun_dir,
                        )
                        .is_none()
                    {
                        l += alb * settings.sun_color * settings.sun_intensity * ndl;
                    }
                    // Point/spot lights.
                    for light in &scene.lights {
                        if light.is_directional {
                            continue;
                        }
                        let to_l = light.pos - origin;
                        let dist = to_l.length();
                        if dist > light.range {
                            continue;
                        }
                        let falloff = (1.0 - dist / light.range).max(0.0);
                        let ldir = to_l / dist.max(1e-5);
                        let ndl2 = dir.dot(ldir).max(0.0);
                        if ndl2 <= 0.0 {
                            continue;
                        }
                        if raycast(&root, &scene.tris, &scene.albedo, &indices, origin, ldir).is_none() {
                            l += alb * light.color * light.intensity * falloff * falloff * ndl2;
                        }
                    }
                    l
                }
                None => scene.sky,
            };

            let basis = sh_basis(dir);
            for (k, b) in basis.iter().enumerate() {
                coeffs[k][0] += radiance.x * *b * inv_n;
                coeffs[k][1] += radiance.y * *b * inv_n;
                coeffs[k][2] += radiance.z * *b * inv_n;
            }
        }

        // SH projection: c_lm = (4π/N) Σ L Y_lm. Balanced weights.
        let scale = 4.0 * std::f32::consts::PI * inv_n;
        let mut sh = IrradianceSH::default();
        for (k, c) in coeffs.iter_mut().enumerate() {
            sh.coeffs[k][0] = c[0] * scale;
            sh.coeffs[k][1] = c[1] * scale;
            sh.coeffs[k][2] = c[2] * scale;
        }
        probe.irradiance = sh;
        probe.sky_occlusion = if up_rays > 0 {
            up_blocked as f32 / up_rays as f32
        } else {
            1.0
        };
    }

    // Deterministic result: number of boxes folded (debug value).
    let total: u64 = scene.tris.len() as u64;
    let _ = total;
    Ok(0)
}

/// Serialise the baked probe grid as JSON beside the scene.
pub fn save_probes(path: &Path, grid: &LightProbeGrid) -> Result<(), String> {
    let mut out = String::from("{\"probes\":[");
    for (i, p) in grid.probes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"x\":{},\"y\":{},\"z\":{},\"r\":{},\"w\":{},\"g\":{},\"o\":{},",
            p.position.x, p.position.y, p.position.z, p.radius, p.weight, p.group, p.sky_occlusion
        ));
        out.push_str("\"c\":[");
        for (k, c) in p.irradiance.coeffs.iter().enumerate() {
            if k > 0 {
                out.push(',');
            }
            out.push_str(&format!("[{},{},{}]", c[0], c[1], c[2]));
        }
        out.push_str("]}");
    }
    out.push_str("],\"volumes\":[");
    for (i, v) in grid.volumes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"cx\":{},\"cy\":{},\"cz\":{},\"sx\":{},\"sy\":{},\"sz\":{},\"dx\":{},\"dy\":{},\"dz\":{}}}",
            v.center.x, v.center.y, v.center.z,
            v.size.x, v.size.y, v.size.z,
            v.density[0], v.density[1], v.density[2]
        ));
    }
    out.push_str("]}");
    std::fs::write(path, out).map_err(|e| e.to_string())
}

/// Load a baked probe grid from JSON.
pub fn load_probes(path: &Path, grid: &mut LightProbeGrid) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let parsed: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("bad probe json: {}", e))?;
    let arr = parsed
        .get("probes")
        .and_then(|v| v.as_array())
        .ok_or("no probes array")?;
    grid.probes.clear();
    grid.volumes.clear();
    for item in arr {
        let num = |k: &str| item.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let pos = Vec3::new(num("x"), num("y"), num("z"));
        let radius = num("r");
        let weight = num("w");
        let group = item.get("g").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        grid.add_probe(pos, radius);
        let last = grid.probes.last_mut().unwrap();
        last.weight = weight;
        last.group = group;
        last.sky_occlusion = item.get("o").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        if let Some(c) = item.get("c").and_then(|v| v.as_array()) {
            for (k, coeff) in c.iter().enumerate().take(9) {
                if let Some(vals) = coeff.as_array() {
                    for ch in 0..3 {
                        if let Some(v) = vals.get(ch).and_then(|x| x.as_f64()) {
                            last.irradiance.coeffs[k][ch] = v as f32;
                        }
                    }
                }
            }
        }
    }
    // Rebuild volumes (kept as live data so boxes stay editable).
    if let Some(vs) = parsed.get("volumes").and_then(|v| v.as_array()) {
        for item in vs {
            let num = |k: &str| item.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let den = |k: &str| item.get(k).and_then(|v| v.as_u64()).unwrap_or(3) as u32;
            grid.volumes.push(ProbeVolume {
                center:  Vec3::new(num("cx"), num("cy"), num("cz")),
                size:    Vec3::new(num("sx"), num("sy"), num("sz")),
                density: [den("dx"), den("dy"), den("dz")],
            });
        }
    }
    Ok(())
}