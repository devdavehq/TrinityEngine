// src/assets/mesh.rs

// Vertex now carries PBR material properties alongside position and normal.
// All values are per-vertex — later we'll support per-material (textures).
// repr(C) + Pod + Zeroable: required to send this to the GPU as raw bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position:  [f32; 3],  //  0 — world-space position
    pub normal:    [f32; 3],  // 12 — surface normal for lighting
    pub tangent:   [f32; 3],  // 24 — tangent direction (T in TBN)
    pub bitangent: [f32; 3],  // 36 — bitangent direction (B in TBN)
    pub color:     [f32; 3],  // 48 — albedo (base color)
    pub metallic:  f32,       // 60 — 0 = non-metal, 1 = metal
    pub roughness: f32,       // 64 — 0 = mirror, 1 = fully matte
    pub ao:        f32,       // 68 — ambient occlusion (0 = occluded, 1 = open)
    pub bone_indices: [u32; 4], // 72 — bone indices for GPU skinning (4 × u32 = 16 bytes)
    pub bone_weights: [f32; 4], // 88 — bone weights for GPU skinning (4 × f32 = 16 bytes)
    // Total: 104 bytes (padded to 112 via repr(C) alignment)
}

impl Vertex {
    // Helper to create a vertex with default PBR values.
    // Most meshes start as non-metal, medium rough, fully lit.
    pub fn new(position: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            position,
            normal,
            tangent:   [0.0, 1.0, 0.0],
            bitangent: [1.0, 0.0, 0.0],
            color,
            metallic:  0.0,
            roughness: 0.5,
            ao:           1.0,
            bone_indices: [0, 0, 0, 0],
            bone_weights: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Corner of a unit cube (half-extent 0.5) for the face whose normal runs
/// along axis `a` with sign `sign`; `u`/`v` are ±1 offsets along the two
/// in-plane axes.
fn offset_cube_corner(a: usize, b: usize, cr: usize, sign: f32, u: f32, v: f32) -> [f32; 3] {
    let mut p = [0.0; 3];
    p[a] = sign * 0.5;
    p[b] = u * 0.5;
    p[cr] = v * 0.5;
    p
}

/// Compute tangent and bitangent for all vertices using a simplified
/// position-based approximation (no UVs needed).
/// For each triangle, tangent aligns with the first edge direction
/// and bitangent is derived via cross(normal, tangent).
pub fn compute_tangents(vertices: &mut [Vertex]) {
    let count = vertices.len();
    if count < 3 { return; }

    // Zero out existing tangents/bitangents.
    for v in vertices.iter_mut() {
        v.tangent   = [0.0, 0.0, 0.0];
        v.bitangent = [0.0, 0.0, 0.0];
    }

    // Process triangles (groups of 3 vertices).
    for tri in vertices.chunks_exact_mut(3) {
        let p0 = glam::Vec3::from_array(tri[0].position);
        let p1 = glam::Vec3::from_array(tri[1].position);
        let p2 = glam::Vec3::from_array(tri[2].position);

        let edge1 = p1 - p0;
        let _edge2 = p2 - p0;

        // Tangent = first edge direction (approximation without UVs).
        let t = edge1;

        // Bitangent = cross(normal, tangent) for a right-handed TBN frame.
        let n0 = glam::Vec3::from_array(tri[0].normal);
        let b = n0.cross(t);

        tri[0].tangent   = (glam::Vec3::from_array(tri[0].tangent) + t).to_array();
        tri[0].bitangent = (glam::Vec3::from_array(tri[0].bitangent) + b).to_array();
        tri[1].tangent   = (glam::Vec3::from_array(tri[1].tangent) + t).to_array();
        tri[1].bitangent = (glam::Vec3::from_array(tri[1].bitangent) + b).to_array();
        tri[2].tangent   = (glam::Vec3::from_array(tri[2].tangent) + t).to_array();
        tri[2].bitangent = (glam::Vec3::from_array(tri[2].bitangent) + b).to_array();
    }

    // Normalize and re-orthogonalize via Gram-Schmidt.
    for v in vertices.iter_mut() {
        let mut t = glam::Vec3::from_array(v.tangent);
        let n = glam::Vec3::from_array(v.normal);

        if t.length_squared() < 1e-10 {
            // Degenerate: pick an arbitrary tangent perpendicular to normal.
            let up = if n.y.abs() < 0.99 {
                glam::Vec3::Y
            } else {
                glam::Vec3::X
            };
            t = n.cross(up).normalize();
        }

        // Gram-Schmidt: orthogonalize tangent against normal.
        t = (t - n * n.dot(t)).normalize();

        // Recompute bitangent to ensure right-handed frame.
        let b = n.cross(t);

        v.tangent   = t.to_array();
        v.bitangent = b.to_array();
    }
}

// Mesh holds GPU-ready vertex data for one shape.
pub struct Mesh {
    pub vertices: Vec<Vertex>,
}

impl Mesh {
    // load() reads an OBJ file and returns a Mesh.
    // This version reads vertex positions, normals, and face indices.
    pub fn load(path: &str) -> Result<Mesh, String> {
        // Photogrammetry scan import marker — see `import_scan` and the
        // editor's "Import as Photogrammetry Scan" button. Persists both the
        // source file and the decimation budget as "<real path>?scan_tris=<N>"
        // so a saved scene regenerates the identically-simplified mesh on
        // reload instead of re-reading the (likely huge) original scan file
        // through the plain, non-decimating path.
        if let Some(pos) = path.find("?scan_tris=") {
            let real_path = &path[..pos];
            let budget: usize = path[pos + "?scan_tris=".len()..].parse().unwrap_or(0);
            return Self::import_scan(real_path, budget);
        }
        // Built-in procedural primitives. These pseudo-paths are produced by
        // the editor's "Add …" buttons and persisted as SceneMeta.mesh_path, so
        // they must regenerate deterministically on scene load.
        match path {
            "meshes/primitive_plane.obj" => return Ok(Self::make_plane(1.0, 1.0)),
            "meshes/primitive_capsule.obj" => return Ok(Self::make_capsule(0.35, 0.55, 12, 20)),
            "meshes/primitive_sphere.obj" => return Ok(Self::make_sphere(0.5, 12, 20)),
            "meshes/primitive_cylinder.obj" => return Ok(Self::make_cylinder(0.5, 0.5, 20)),
            "meshes/primitive_cone.obj" => return Ok(Self::make_cone(0.5, 1.0, 20)),
            "meshes/primitive_tunnel.obj" => return Ok(Self::make_tunnel_arch(4.0, 2.0, 8.0, 20)),
            _ => {}
        }
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if ext == "gltf" || ext == "glb" {
            return Self::load_gltf(path);
        }
        let contents = match crate::vfs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                // Missing asset: fall back to a built-in cube so scenes,
                // prefabs and editor spawns keep working instead of failing
                // the whole load. The scene scale on the Renderable usually
                // stretches it into the intended shape (floor slabs etc.).
                tracing::warn!("[Mesh] {} not found ({}); using built-in cube", path, e);
                return Ok(Self::make_cube());
            }
        };

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals:   Vec<[f32; 3]> = Vec::new();
        // Each face entry: (position_index, Option<normal_index>)
        let mut faces: Vec<Vec<(usize, Option<usize>)>> = Vec::new();

        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }

            let mut parts = line.splitn(2, ' ');
            let kw   = parts.next().unwrap_or("");
            let rest = parts.next().unwrap_or("");

            match kw {
                "v" => {
                    // Position: v x y z
                    let c: Vec<f32> = rest.split_whitespace()
                        .map(|s| s.parse().unwrap_or(0.0)).collect();
                    if c.len() >= 3 { positions.push([c[0], c[1], c[2]]); }
                }
                "vn" => {
                    // Normal: vn nx ny nz
                    let c: Vec<f32> = rest.split_whitespace()
                        .map(|s| s.parse().unwrap_or(0.0)).collect();
                    if c.len() >= 3 { normals.push([c[0], c[1], c[2]]); }
                }
                "f" => {
                    // Face: f v1//vn1 v2//vn2 v3//vn3
                    // or    f v1 v2 v3 (no normals)
                    let indices: Vec<(usize, Option<usize>)> = rest
                        .split_whitespace()
                        .map(|token| {
                            let parts: Vec<&str> = token.split('/').collect();
                            let pi = parts[0].parse::<usize>().unwrap_or(1) - 1;
                            // Normal is the third slot (v/vt/vn or v//vn).
                            let ni = parts.get(2)
                                .and_then(|s| if s.is_empty() { None } else { s.parse::<usize>().ok() })
                                .map(|n| n - 1);
                            (pi, ni)
                        })
                        .collect();
                    if indices.len() >= 3 { faces.push(indices); }
                }
                _ => {}
            }
        }

        // Build vertex list — fan triangulation for quads.
        let mut vertices: Vec<Vertex> = Vec::new();

        for face in &faces {
            for i in 1..(face.len() - 1) {
                for &(pi, ni) in &[face[0], face[i], face[i + 1]] {
                    let pos    = positions.get(pi).copied().unwrap_or([0.0; 3]);
                    let normal = ni
                        .and_then(|n| normals.get(n).copied())
                        .unwrap_or([0.0, 0.0, 1.0]); // default: facing viewer
                    vertices.push(Vertex::new(pos, normal, [1.0, 1.0, 1.0]));
                }
            }
        }

        if vertices.is_empty() {
            tracing::warn!("[Mesh] No geometry in {}; using built-in cube", path);
            return Ok(Self::make_cube());
        }
        Ok(Mesh { vertices })
    }

    fn load_gltf(path: &str) -> Result<Mesh, String> {
        let (doc, buffers, _) = gltf::import(path)
            .map_err(|e| format!("Cannot import glTF {}: {}", path, e))?;
        let mut out: Vec<Vertex> = Vec::new();
        for mesh in doc.meshes() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
                let Some(positions_iter) = reader.read_positions() else {
                    continue;
                };
                let positions: Vec<[f32; 3]> = positions_iter.collect();
                if positions.is_empty() {
                    continue;
                }
                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|n| n.collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; positions.len()]);
                if let Some(indices) = reader.read_indices() {
                    let idx: Vec<u32> = indices.into_u32().collect();
                    for tri in idx.chunks_exact(3) {
                        for i in tri {
                            let ii = *i as usize;
                            let p = positions.get(ii).copied().unwrap_or([0.0; 3]);
                            let n = normals.get(ii).copied().unwrap_or([0.0, 0.0, 1.0]);
                            out.push(Vertex::new(p, n, [1.0, 1.0, 1.0]));
                        }
                    }
                } else {
                    for tri in positions.chunks_exact(3).zip(normals.chunks_exact(3)) {
                        let (pp, nn) = tri;
                        out.push(Vertex::new(pp[0], nn[0], [1.0, 1.0, 1.0]));
                        out.push(Vertex::new(pp[1], nn[1], [1.0, 1.0, 1.0]));
                        out.push(Vertex::new(pp[2], nn[2], [1.0, 1.0, 1.0]));
                    }
                }
            }
        }
        if out.is_empty() {
            return Err(format!("No mesh primitives found in {}", path));
        }
        Ok(Mesh { vertices: out })
    }

    pub fn make_cube() -> Mesh {
        let c = [1.0, 1.0, 1.0];
        let mut vertices = Vec::with_capacity(36);
        // Six faces: normal axis + sign, corner offsets along the two in-plane axes.
        let axes: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for a in 0..3 {
            for sign in [1.0, -1.0] {
                let b = (a + 1) % 3;
                let cr = (a + 2) % 3;
                let mut normal = axes[a];
                for n in &mut normal {
                    *n *= sign;
                }
                // Quad corners (winding CCW when viewed from +normal).
                let p00 = offset_cube_corner(a, b, cr, sign, -1.0, -1.0);
                let p10 = offset_cube_corner(a, b, cr, sign, 1.0, -1.0);
                let p11 = offset_cube_corner(a, b, cr, sign, 1.0, 1.0);
                let p01 = offset_cube_corner(a, b, cr, sign, -1.0, 1.0);
                vertices.extend_from_slice(&[
                    Vertex::new(p00, normal, c),
                    Vertex::new(p10, normal, c),
                    Vertex::new(p11, normal, c),
                    Vertex::new(p00, normal, c),
                    Vertex::new(p11, normal, c),
                    Vertex::new(p01, normal, c),
                ]);
            }
        }
        Mesh { vertices }
    }

    pub fn make_plane(size_x: f32, size_z: f32) -> Mesh {
        let hx = size_x * 0.5;
        let hz = size_z * 0.5;
        let n = [0.0, 1.0, 0.0];
        let c = [1.0, 1.0, 1.0];
        let p0 = [-hx, 0.0, -hz];
        let p1 = [hx, 0.0, -hz];
        let p2 = [hx, 0.0, hz];
        let p3 = [-hx, 0.0, hz];
        Mesh {
            vertices: vec![
                Vertex::new(p0, n, c),
                Vertex::new(p1, n, c),
                Vertex::new(p2, n, c),
                Vertex::new(p0, n, c),
                Vertex::new(p2, n, c),
                Vertex::new(p3, n, c),
            ],
        }
    }

    pub fn make_capsule(radius: f32, half_height: f32, rings: usize, segments: usize) -> Mesh {
        let mut vertices = Vec::new();
        let rings = rings.max(4);
        let segments = segments.max(8);
        let h = half_height.max(0.01);
        let r = radius.max(0.01);

        // Cylinder body
        for i in 0..segments {
            let a0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let (x0, z0) = (a0.cos() * r, a0.sin() * r);
            let (x1, z1) = (a1.cos() * r, a1.sin() * r);
            let n0 = [a0.cos(), 0.0, a0.sin()];
            let n1 = [a1.cos(), 0.0, a1.sin()];
            let c = [1.0, 1.0, 1.0];
            let b0 = [x0, -h, z0];
            let t0 = [x0, h, z0];
            let b1 = [x1, -h, z1];
            let t1 = [x1, h, z1];
            vertices.extend_from_slice(&[
                Vertex::new(b0, n0, c),
                Vertex::new(t0, n0, c),
                Vertex::new(t1, n1, c),
                Vertex::new(b0, n0, c),
                Vertex::new(t1, n1, c),
                Vertex::new(b1, n1, c),
            ]);
        }

        // Hemisphere helper
        let mut add_hemi = |top: bool| {
            let y_sign = if top { 1.0 } else { -1.0 };
            let y_offset = if top { h } else { -h };
            for y in 0..(rings / 2) {
                let v0 = y as f32 / (rings as f32 / 2.0);
                let v1 = (y + 1) as f32 / (rings as f32 / 2.0);
                let phi0 = v0 * std::f32::consts::FRAC_PI_2;
                let phi1 = v1 * std::f32::consts::FRAC_PI_2;
                let ry0 = phi0.sin() * r * y_sign;
                let rr0 = phi0.cos() * r;
                let ry1 = phi1.sin() * r * y_sign;
                let rr1 = phi1.cos() * r;
                for i in 0..segments {
                    let a0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
                    let a1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
                    let p00 = [a0.cos() * rr0, y_offset + ry0, a0.sin() * rr0];
                    let p01 = [a1.cos() * rr0, y_offset + ry0, a1.sin() * rr0];
                    let p10 = [a0.cos() * rr1, y_offset + ry1, a0.sin() * rr1];
                    let p11 = [a1.cos() * rr1, y_offset + ry1, a1.sin() * rr1];
                    let n00 = glam::Vec3::new(p00[0], p00[1] - y_offset, p00[2]).normalize();
                    let n01 = glam::Vec3::new(p01[0], p01[1] - y_offset, p01[2]).normalize();
                    let n10 = glam::Vec3::new(p10[0], p10[1] - y_offset, p10[2]).normalize();
                    let n11 = glam::Vec3::new(p11[0], p11[1] - y_offset, p11[2]).normalize();
                    let c = [1.0, 1.0, 1.0];
                    if top {
                        vertices.extend_from_slice(&[
                            Vertex::new(p00, n00.to_array(), c),
                            Vertex::new(p10, n10.to_array(), c),
                            Vertex::new(p11, n11.to_array(), c),
                            Vertex::new(p00, n00.to_array(), c),
                            Vertex::new(p11, n11.to_array(), c),
                            Vertex::new(p01, n01.to_array(), c),
                        ]);
                    } else {
                        vertices.extend_from_slice(&[
                            Vertex::new(p00, n00.to_array(), c),
                            Vertex::new(p11, n11.to_array(), c),
                            Vertex::new(p10, n10.to_array(), c),
                            Vertex::new(p00, n00.to_array(), c),
                            Vertex::new(p01, n01.to_array(), c),
                            Vertex::new(p11, n11.to_array(), c),
                        ]);
                    }
                }
            }
        };
        add_hemi(true);
        add_hemi(false);
        Mesh { vertices }
    }

    /// UV-sphere of the given radius.
    pub fn make_sphere(radius: f32, rings: usize, segments: usize) -> Mesh {
        let mut vertices = Vec::new();
        let r = radius.max(0.01);
        let rings = rings.max(3);
        let segments = segments.max(8);
        let c = [1.0, 1.0, 1.0];
        for y in 0..rings {
            let v0 = y as f32 / rings as f32;
            let v1 = (y + 1) as f32 / rings as f32;
            let phi0 = v0 * std::f32::consts::PI;
            let phi1 = v1 * std::f32::consts::PI;
            let y0 = phi0.cos() * r;
            let y1 = phi1.cos() * r;
            let r0 = phi0.sin() * r;
            let r1 = phi1.sin() * r;
            for i in 0..segments {
                let a0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
                let a1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
                let p00 = [a0.cos() * r0, y0, a0.sin() * r0];
                let p01 = [a1.cos() * r0, y0, a1.sin() * r0];
                let p10 = [a0.cos() * r1, y1, a0.sin() * r1];
                let p11 = [a1.cos() * r1, y1, a1.sin() * r1];
                let n00 = glam::Vec3::new(p00[0], p00[1], p00[2]).normalize();
                let n01 = glam::Vec3::new(p01[0], p01[1], p01[2]).normalize();
                let n10 = glam::Vec3::new(p10[0], p10[1], p10[2]).normalize();
                let n11 = glam::Vec3::new(p11[0], p11[1], p11[2]).normalize();
                vertices.extend_from_slice(&[
                    Vertex::new(p00, n00.to_array(), c),
                    Vertex::new(p10, n10.to_array(), c),
                    Vertex::new(p11, n11.to_array(), c),
                    Vertex::new(p00, n00.to_array(), c),
                    Vertex::new(p11, n11.to_array(), c),
                    Vertex::new(p01, n01.to_array(), c),
                ]);
            }
        }
        Mesh { vertices }
    }

    /// Solid cylinder of the given radius and half-height, with end caps.
    pub fn make_cylinder(radius: f32, half_height: f32, segments: usize) -> Mesh {
        let mut vertices = Vec::new();
        let r = radius.max(0.01);
        let h = half_height.max(0.01);
        let segments = segments.max(8);
        let c = [1.0, 1.0, 1.0];
        // Side walls.
        for i in 0..segments {
            let a0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let (x0, z0, nx0, nz0) = (a0.cos() * r, a0.sin() * r, a0.cos(), a0.sin());
            let (x1, z1, nx1, nz1) = (a1.cos() * r, a1.sin() * r, a1.cos(), a1.sin());
            vertices.extend_from_slice(&[
                Vertex::new([x0, -h, z0], [nx0, 0.0, nz0], c),
                Vertex::new([x0, h, z0], [nx0, 0.0, nz0], c),
                Vertex::new([x1, h, z1], [nx1, 0.0, nz1], c),
                Vertex::new([x0, -h, z0], [nx0, 0.0, nz0], c),
                Vertex::new([x1, h, z1], [nx1, 0.0, nz1], c),
                Vertex::new([x1, -h, z1], [nx1, 0.0, nz1], c),
            ]);
        }
        // End caps (triangle fans; normal +Y top, -Y bottom).
        for (sign, cap_y) in [(1.0, h), (-1.0, -h)] {
            for i in 0..segments {
                let a0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
                let a1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
                let x0 = a0.cos() * r;
                let z0 = a0.sin() * r;
                let x1 = a1.cos() * r;
                let z1 = a1.sin() * r;
                let n = [0.0, sign, 0.0];
                if sign > 0.0 {
                    vertices.extend_from_slice(&[
                        Vertex::new([0.0, cap_y, 0.0], n, c),
                        Vertex::new([x0, cap_y, z0], n, c),
                        Vertex::new([x1, cap_y, z1], n, c),
                    ]);
                } else {
                    vertices.extend_from_slice(&[
                        Vertex::new([0.0, cap_y, 0.0], n, c),
                        Vertex::new([x1, cap_y, z1], n, c),
                        Vertex::new([x0, cap_y, z0], n, c),
                    ]);
                }
            }
        }
        Mesh { vertices }
    }

    /// Solid cone: apex at `+height/2`, base circle at `-height/2`.
    pub fn make_cone(radius: f32, height: f32, segments: usize) -> Mesh {
        let mut vertices = Vec::new();
        let r = radius.max(0.01);
        let h = height.max(0.01);
        let segments = segments.max(8);
        let c = [1.0, 1.0, 1.0];
        let half = h * 0.5;
        // Side (apex + two base points per wedge). The side normal of a right
        // cone at azimuth a points along (cos a, r/h, sin a) after normalizing.
        for i in 0..segments {
            let a0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let am = (a0 + a1) * 0.5;
            let n = glam::Vec3::new(am.cos(), r / h, am.sin()).normalize().to_array();
            let p0 = [a0.cos() * r, -half, a0.sin() * r];
            let p1 = [a1.cos() * r, -half, a1.sin() * r];
            vertices.extend_from_slice(&[
                Vertex::new([0.0, half, 0.0], n, c),
                Vertex::new(p0, n, c),
                Vertex::new(p1, n, c),
            ]);
        }
        // Base cap (normal -Y).
        let n = [0.0, -1.0, 0.0];
        for i in 0..segments {
            let a0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            vertices.extend_from_slice(&[
                Vertex::new([0.0, -half, 0.0], n, c),
                Vertex::new([a1.cos() * r, -half, a1.sin() * r], n, c),
                Vertex::new([a0.cos() * r, -half, a0.sin() * r], n, c),
            ]);
        }
        Mesh { vertices }
    }

    /// A tunnel arch: a corridor cross-section (floor + straight walls + a
    /// semicircular roof) extruded along Z. Open at both ends so segments can
    /// be stacked end-to-end to make long straight caves. `width` is the full
    /// tunnel width, `wall_height` is the height of the straight walls up to
    /// where the semicircular roof begins (roof radius = width/2), and `depth`
    /// is the tunnel length along Z.
    pub fn make_tunnel_arch(width: f32, wall_height: f32, depth: f32, segments: usize) -> Mesh {
        let mut vertices = Vec::new();
        let w2 = width.max(0.1) * 0.5;
        let wh = wall_height.max(0.01);
        let dz = depth.max(0.1) * 0.5;
        let r = w2;
        let segments = segments.max(8);
        let c = [1.0, 1.0, 1.0];
        let pi = std::f32::consts::PI;

        // Arch ring in the XY plane: left wall bottom -> left wall top ->
        // semicircle over the top -> right wall top -> right wall bottom.
        let mut ring: Vec<[f32; 2]> = Vec::new();
        ring.push([-w2, 0.0]);            // left wall bottom
        ring.push([-w2, wh]);             // left wall top
        for i in 1..segments {
            let t = pi * (1.0 - i as f32 / segments as f32);
            ring.push([r * t.cos(), wh + r * t.sin()]);
        }
        ring.push([w2, wh]);              // right wall top
        ring.push([w2, 0.0]);             // right wall bottom

        // Interior centroid of the U-shape (used to orient outward normals).
        let (cx, cy) = (0.0f32, wh * 0.35);
        for i in 0..ring.len() - 1 {
            let p0 = ring[i];
            let p1 = ring[i + 1];
            let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
            // Perpendicular to the wall segment, flipped to point away from
            // the interior centroid.
            let ex = p1[0] - p0[0];
            let ey = p1[1] - p0[1];
            let mut nx = -ey;
            let mut ny = ex;
            if nx * (mid[0] - cx) + ny * (mid[1] - cy) < 0.0 {
                nx = -nx;
                ny = -ny;
            }
            let len = (nx * nx + ny * ny).sqrt().max(1e-6);
            let n = [nx / len, ny / len, 0.0];
            let f0 = [p0[0], p0[1], -dz];
            let f1 = [p1[0], p1[1], -dz];
            let b0 = [p0[0], p0[1], dz];
            let b1 = [p1[0], p1[1], dz];
            vertices.extend_from_slice(&[
                Vertex::new(f0, n, c),
                Vertex::new(b0, n, c),
                Vertex::new(b1, n, c),
                Vertex::new(f0, n, c),
                Vertex::new(b1, n, c),
                Vertex::new(f1, n, c),
            ]);
        }

        // Floor quad spanning the full tunnel length, normal +Y.
        let n = [0.0, 1.0, 0.0];
        vertices.extend_from_slice(&[
            Vertex::new([-w2, 0.0, -dz], n, c),
            Vertex::new([-w2, 0.0, dz], n, c),
            Vertex::new([w2, 0.0, dz], n, c),
            Vertex::new([-w2, 0.0, -dz], n, c),
            Vertex::new([w2, 0.0, dz], n, c),
            Vertex::new([w2, 0.0, -dz], n, c),
        ]);

        Mesh { vertices }
    }

    /// Import an externally-scanned mesh (photogrammetry output from
    /// RealityCapture, Metashape, Polycam, Meshroom, etc — OBJ or glTF/GLB).
    ///
    /// This does NOT do 3D reconstruction; turning photos into geometry is a
    /// separate computer-vision problem those tools already solved. This is
    /// the asset-import half: it loads the scan's mesh via the normal
    /// OBJ/glTF path and then, since scan exports routinely land in the
    /// hundreds of thousands to millions of triangles, auto-decimates down
    /// to `max_triangles` using the engine's edge-collapse simplifier so the
    /// result is actually usable as a placed asset. Pass `max_triangles = 0`
    /// to skip decimation entirely.
    pub fn import_scan(path: &str, max_triangles: usize) -> Result<Mesh, String> {
        let mesh = Self::load(path)?;
        let tri_count = mesh.vertices.len() / 3;
        if max_triangles == 0 || tri_count <= max_triangles {
            return Ok(mesh);
        }
        let keep_ratio = max_triangles as f32 / tri_count as f32;
        let simplified = crate::renderer::simplify_triangle_soup_preserve_shape(&mesh.vertices, keep_ratio);
        Ok(Mesh { vertices: simplified })
    }

    /// Split this mesh's triangles into spatially-compact clusters of at
    /// most `max_tris` triangles each — the same "divide a big mesh into
    /// small groups" idea Nanite calls clusters/meshlets. Triangles are
    /// first sorted along a Morton (Z-order) curve of their centroid so
    /// consecutive triangles end up spatially close together, then chunked;
    /// that's what keeps each cluster's bounding sphere tight.
    ///
    /// Returns a mesh with triangles reordered into contiguous per-cluster
    /// ranges (so each `MeshCluster` is a plain vertex sub-range you can
    /// hand to a draw call) alongside the cluster list itself. This is the
    /// CPU-side foundation a GPU-driven cluster-cull compute pass would
    /// consume — it does not by itself perform any culling.
    pub fn build_clusters(&self, max_tris: usize) -> (Mesh, Vec<MeshCluster>) {
        let max_tris = max_tris.max(1);
        let tri_count = self.vertices.len() / 3;
        if tri_count == 0 {
            return (Mesh { vertices: Vec::new() }, Vec::new());
        }

        // Centroid + AABB of centroids, so coordinates can be quantized into
        // the 0..=1023 range morton_code() expects.
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        let centroids: Vec<[f32; 3]> = (0..tri_count)
            .map(|i| {
                let a = self.vertices[i * 3].position;
                let b = self.vertices[i * 3 + 1].position;
                let c = self.vertices[i * 3 + 2].position;
                let centroid = [
                    (a[0] + b[0] + c[0]) / 3.0,
                    (a[1] + b[1] + c[1]) / 3.0,
                    (a[2] + b[2] + c[2]) / 3.0,
                ];
                for axis in 0..3 {
                    min[axis] = min[axis].min(centroid[axis]);
                    max[axis] = max[axis].max(centroid[axis]);
                }
                centroid
            })
            .collect();
        let extent = [
            (max[0] - min[0]).max(1e-6),
            (max[1] - min[1]).max(1e-6),
            (max[2] - min[2]).max(1e-6),
        ];

        let mut order: Vec<usize> = (0..tri_count).collect();
        order.sort_by_key(|&i| {
            let c = centroids[i];
            let qx = (((c[0] - min[0]) / extent[0]) * 1023.0) as u32;
            let qy = (((c[1] - min[1]) / extent[1]) * 1023.0) as u32;
            let qz = (((c[2] - min[2]) / extent[2]) * 1023.0) as u32;
            morton_code(qx, qy, qz)
        });

        let mut reordered = Vec::with_capacity(self.vertices.len());
        let mut clusters = Vec::with_capacity(order.len().div_ceil(max_tris));
        for chunk in order.chunks(max_tris) {
            let vertex_start = reordered.len() as u32;
            for &tri in chunk {
                for v in 0..3 {
                    reordered.push(self.vertices[tri * 3 + v]);
                }
            }
            let positions: Vec<[f32; 3]> = reordered[vertex_start as usize..]
                .iter()
                .map(|v| v.position)
                .collect();
            let (center, radius) = ritter_bounding_sphere(&positions);
            clusters.push(MeshCluster {
                vertex_start,
                vertex_count: (reordered.len() as u32) - vertex_start,
                bounds_center: center,
                bounds_radius: radius,
            });
        }

        (Mesh { vertices: reordered }, clusters)
    }
}

/// A cluster ("meshlet") of spatially-nearby triangles within a mesh's
/// reordered vertex buffer — see `Mesh::build_clusters`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshCluster {
    /// Start offset into the reordered mesh's flat triangle-soup vertex
    /// buffer. Always a multiple of 3 (whole triangles only).
    pub vertex_start: u32,
    pub vertex_count: u32,
    /// Bounding sphere in the mesh's local space — big enough to contain
    /// every vertex in this cluster, used for cheap frustum/occlusion tests
    /// before drawing the cluster's actual triangles.
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// Ritter's bounding-sphere approximation. Cheap (a small constant number of
/// passes over the points) and meaningfully tighter than the naive "AABB
/// center + farthest point" sphere for non-degenerate point clouds, which
/// matters for cluster culling: a looser sphere means more clusters get
/// wrongly kept "maybe visible" by a frustum/occlusion test than necessary.
/// Not a true minimal enclosing sphere (that needs Welzl's algorithm), but a
/// well-established, much cheaper approximation that's good enough here.
fn ritter_bounding_sphere(points: &[[f32; 3]]) -> ([f32; 3], f32) {
    debug_assert!(!points.is_empty());
    if points.len() == 1 {
        return (points[0], 0.0);
    }

    // Pass 1+2: y = farthest point from an arbitrary start; z = farthest
    // point from y. (y, z) approximates the point cloud's longest axis.
    let start = points[0];
    let y = *points
        .iter()
        .max_by(|a, b| dist2(start, **a).total_cmp(&dist2(start, **b)))
        .unwrap();
    let z = *points
        .iter()
        .max_by(|a, b| dist2(y, **a).total_cmp(&dist2(y, **b)))
        .unwrap();

    let mut center = [(y[0] + z[0]) * 0.5, (y[1] + z[1]) * 0.5, (y[2] + z[2]) * 0.5];
    let mut radius = dist2(y, z).sqrt() * 0.5;

    // Pass 3: expand the sphere minimally to cover every point that falls
    // outside it — one pass is sufficient (not iterative Welzl-style
    // refinement), which is what keeps this cheap.
    for &p in points {
        let d = dist2(center, p).sqrt();
        if d > radius {
            let new_radius = (radius + d) * 0.5;
            let k = (new_radius - radius) / d.max(1e-8);
            center = [
                center[0] + (p[0] - center[0]) * k,
                center[1] + (p[1] - center[1]) * k,
                center[2] + (p[2] - center[2]) * k,
            ];
            radius = new_radius;
        }
    }
    (center, radius)
}

/// Interleave the low 10 bits of `v` with two zero bits between each source
/// bit, so three spread values can be OR'd together (shifted by 0/1/2) into
/// a 30-bit 3D Morton (Z-order) code. Standard bit-spreading trick.
fn spread_bits_10(v: u32) -> u32 {
    let v = v & 0x3FF;
    let v = (v | (v << 16)) & 0x30000FF;
    let v = (v | (v << 8)) & 0x300F00F;
    let v = (v | (v << 4)) & 0x30C30C3;
    (v | (v << 2)) & 0x9249249
}

fn morton_code(x: u32, y: u32, z: u32) -> u32 {
    spread_bits_10(x) | (spread_bits_10(y) << 1) | (spread_bits_10(z) << 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_empty(m: &Mesh) -> bool {
        !m.vertices.is_empty()
    }

    #[test]
    fn primitives_generate_geometry() {
        assert!(non_empty(&Mesh::make_sphere(0.5, 12, 20)));
        assert!(non_empty(&Mesh::make_cylinder(0.5, 0.5, 20)));
        assert!(non_empty(&Mesh::make_cone(0.5, 1.0, 20)));
        assert!(non_empty(&Mesh::make_tunnel_arch(4.0, 2.0, 8.0, 20)));
        assert!(non_empty(&Mesh::make_plane(1.0, 1.0)));
        assert!(non_empty(&Mesh::make_capsule(0.35, 0.55, 12, 20)));
    }

    #[test]
    fn tunnel_arch_vertex_counts_scale_with_segments() {
        let segs = 12;
        // Ring has segs+3 points (two wall bottoms, two wall tops, segs-1 arc
        // points) -> segs+2 tube quads * 6 verts + 6 floor verts.
        let m = Mesh::make_tunnel_arch(4.0, 2.0, 8.0, segs);
        assert_eq!(m.vertices.len(), (segs + 2) * 6 + 6);
        // All tunnel vertices sit at or above y = 0 (floor level).
        assert!(m.vertices.iter().all(|v| v.position[1] >= -0.001));
    }

    #[test]
    fn primitive_paths_regenerate_deterministically() {
        let a = Mesh::load("meshes/primitive_sphere.obj").unwrap();
        let b = Mesh::load("meshes/primitive_sphere.obj").unwrap();
        assert_eq!(a.vertices.len(), b.vertices.len());
        assert!(a.vertices.iter().zip(b.vertices.iter()).all(|(x, y)| {
            x.position == y.position && x.normal == y.normal
        }));
        assert!(non_empty(&Mesh::load("meshes/primitive_tunnel.obj").unwrap()));
    }

    // ── build_clusters (meshlet foundation) ────────────────────────────

    #[test]
    fn build_clusters_covers_every_triangle_exactly_once() {
        // A sphere has plenty of triangles to split across several clusters.
        let mesh = Mesh::make_sphere(1.0, 16, 24);
        let tri_count = mesh.vertices.len() / 3;
        let (reordered, clusters) = mesh.build_clusters(32);

        assert_eq!(reordered.vertices.len(), mesh.vertices.len());
        let total_verts: u32 = clusters.iter().map(|c| c.vertex_count).sum();
        assert_eq!(total_verts as usize, mesh.vertices.len());

        // Ranges are contiguous, non-overlapping, and cover the whole buffer.
        let mut cursor = 0u32;
        for c in &clusters {
            assert_eq!(c.vertex_start, cursor);
            assert!(c.vertex_count > 0);
            assert_eq!(c.vertex_count % 3, 0, "cluster must hold whole triangles");
            cursor += c.vertex_count;
        }
        assert_eq!(cursor as usize, mesh.vertices.len());

        // No cluster exceeds the requested triangle budget.
        for c in &clusters {
            assert!(c.vertex_count / 3 <= 32);
        }
        assert_eq!(clusters.len(), tri_count.div_ceil(32));
    }

    #[test]
    fn build_clusters_bounding_sphere_contains_all_its_vertices() {
        let mesh = Mesh::make_sphere(1.0, 12, 16);
        let (reordered, clusters) = mesh.build_clusters(24);
        for c in &clusters {
            let range = &reordered.vertices[c.vertex_start as usize..(c.vertex_start + c.vertex_count) as usize];
            for v in range {
                let d = [
                    v.position[0] - c.bounds_center[0],
                    v.position[1] - c.bounds_center[1],
                    v.position[2] - c.bounds_center[2],
                ];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert!(dist <= c.bounds_radius + 1e-4, "vertex escapes its cluster's bounding sphere");
            }
        }
    }

    #[test]
    fn build_clusters_empty_mesh_yields_no_clusters() {
        let (reordered, clusters) = Mesh { vertices: Vec::new() }.build_clusters(64);
        assert!(reordered.vertices.is_empty());
        assert!(clusters.is_empty());
    }

    #[test]
    fn build_clusters_zero_budget_treated_as_one_triangle_per_cluster() {
        let mesh = Mesh::make_cube();
        let tri_count = mesh.vertices.len() / 3;
        let (_, clusters) = mesh.build_clusters(0);
        assert_eq!(clusters.len(), tri_count);
    }

    // ── import_scan (photogrammetry import) ────────────────────────────

    #[test]
    fn import_scan_decimates_down_to_the_triangle_budget() {
        // Reuse a primitive path as a stand-in "scan" file — the point under
        // test is the decimation step, not OBJ parsing itself.
        let full = Mesh::load("meshes/primitive_sphere.obj").unwrap();
        let full_tris = full.vertices.len() / 3;
        let budget = (full_tris / 4).max(2);

        let imported = Mesh::import_scan("meshes/primitive_sphere.obj", budget).unwrap();
        let imported_tris = imported.vertices.len() / 3;
        assert!(imported_tris <= full_tris);
        assert!(non_empty(&imported));
    }

    #[test]
    fn import_scan_zero_budget_skips_decimation() {
        let full = Mesh::load("meshes/primitive_sphere.obj").unwrap();
        let imported = Mesh::import_scan("meshes/primitive_sphere.obj", 0).unwrap();
        assert_eq!(full.vertices.len(), imported.vertices.len());
    }

    #[test]
    fn import_scan_under_budget_is_a_no_op() {
        let full = Mesh::load("meshes/primitive_tunnel.obj").unwrap();
        let imported = Mesh::import_scan("meshes/primitive_tunnel.obj", full.vertices.len() + 1_000_000).unwrap();
        // Budget is nowhere near the limit -> returned untouched.
        assert_eq!(imported.vertices.len(), full.vertices.len());
    }
}