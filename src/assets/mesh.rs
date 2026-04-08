// src/assets/mesh.rs

// Vertex now carries PBR material properties alongside position and normal.
// All values are per-vertex — later we'll support per-material (textures).
// repr(C) + Pod + Zeroable: required to send this to the GPU as raw bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position:  [f32; 3],  // world-space position
    pub normal:    [f32; 3],  // surface normal for lighting
    pub color:     [f32; 3],  // albedo (base color)
    pub metallic:  f32,       // 0 = non-metal, 1 = metal
    pub roughness: f32,       // 0 = mirror, 1 = fully matte
    pub ao:        f32,       // ambient occlusion (0 = occluded, 1 = open)
    // Padding: GPU requires structs to align to 16-byte boundaries.
    // Our struct so far: 3+3+3+1+1+1 = 12 f32s = 48 bytes.
    // Next multiple of 16 = 48. We're fine — but add one f32 pad anyway
    // in case the driver is strict. (Some are.)
    _pad: f32,
}

impl Vertex {
    // Helper to create a vertex with default PBR values.
    // Most meshes start as non-metal, medium rough, fully lit.
    pub fn new(position: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            position,
            normal,
            color,
            metallic:  0.0,
            roughness: 0.5,
            ao:        1.0,
            _pad:      0.0,
        }
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
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if ext == "gltf" || ext == "glb" {
            return Self::load_gltf(path);
        }
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("Cannot read {}: {}", path, e))?;

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
            return Err(format!("No geometry in {}", path));
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
}