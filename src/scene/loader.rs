// src/scene/loader.rs
// Reads .scene files and returns a list of entity descriptions.
// The engine spawns ECS entities from these descriptions.

// EntityDesc holds everything needed to spawn one entity.
// All fields have defaults — you only write what you need in the scene file.
#[derive(Debug, Clone)]
pub struct EntityDesc {
    pub name:      String,
    pub mesh:      String,          // path to .obj file
    pub position:  [f32; 3],
    pub scale:     [f32; 3],        // non-uniform scale: x, y, z
    pub color:     [f32; 3],
    pub metallic:  f32,
    pub roughness: f32,
    pub ao:        f32,
    pub script:    Option<String>,  // optional Lua script path
}

impl Default for EntityDesc {
    fn default() -> Self {
        Self {
            name:      "entity".to_string(),
            mesh:      "meshes/cube.obj".to_string(),
            position:  [0.0, 0.0, 0.0],
            scale:     [1.0, 1.0, 1.0],   // default: no scaling
            color:     [1.0, 1.0, 1.0],   // default: white
            metallic:  0.0,
            roughness: 0.5,
            ao:        1.0,
            script:    None,
        }
    }
}

// parse_scene() reads a .scene file and returns all entity descriptions.
// Returns an error string if the file can't be read.
pub fn parse_scene(path: &str) -> Result<Vec<EntityDesc>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read scene {}: {}", path, e))?;

    let mut entities: Vec<EntityDesc> = Vec::new();
    // current holds the entity being built right now.
    // When we hit a new [entity] line, we push current and start fresh.
    let mut current: Option<EntityDesc> = None;

    for line in contents.lines() {
        let line = line.trim();

        // Skip blank lines and comments.
        if line.is_empty() || line.starts_with('#') { continue; }

        if line == "[entity]" {
            // Starting a new entity block.
            // If we were building one, save it.
            if let Some(desc) = current.take() {
                entities.push(desc);
            }
            // Start fresh with defaults.
            current = Some(EntityDesc::default());
            continue;
        }

        // Parse "key = value" lines.
        // splitn(2, '=') splits on the first '=' only.
        let mut parts = line.splitn(2, '=');
        let key   = parts.next().unwrap_or("").trim();
        let value = parts.next().unwrap_or("").trim();

        // If we're not inside an [entity] block, skip.
        let desc = match current.as_mut() {
            Some(d) => d,
            None    => continue,
        };

        // Parse the value based on the key.
        // parse_floats() is a helper defined below.
        match key {
            "name"      => desc.name      = value.to_string(),
            "mesh"      => desc.mesh      = value.to_string(),
            "script"    => desc.script    = Some(value.to_string()),
            "metallic"  => desc.metallic  = value.parse().unwrap_or(0.0),
            "roughness" => desc.roughness = value.parse().unwrap_or(0.5),
            "ao"        => desc.ao        = value.parse().unwrap_or(1.0),

            "position" => {
                // "0.0  1.0  0.0" → [0.0, 1.0, 0.0]
                let v = parse_floats(value);
                if v.len() >= 3 { desc.position = [v[0], v[1], v[2]]; }
            }
            "scale" => {
                let v = parse_floats(value);
                if v.len() >= 3 { desc.scale = [v[0], v[1], v[2]]; }
                // If only one value given, use it uniformly.
                else if v.len() == 1 { desc.scale = [v[0], v[0], v[0]]; }
            }
            "color" => {
                let v = parse_floats(value);
                if v.len() >= 3 { desc.color = [v[0], v[1], v[2]]; }
            }
            _ => {
                // Unknown key — silently ignore.
                // This lets you add comments like "# todo = fix this" without crashing.
            }
        }
    }

    // Don't forget the last entity — push it if we were building one.
    if let Some(desc) = current {
        entities.push(desc);
    }

    Ok(entities)
}

// parse_floats() splits a whitespace-separated string into f32 values.
// "0.5  1.0  -2.0" → [0.5, 1.0, -2.0]
fn parse_floats(s: &str) -> Vec<f32> {
    s.split_whitespace()
        .map(|tok| tok.parse::<f32>().unwrap_or(0.0))
        .collect()
}