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
}