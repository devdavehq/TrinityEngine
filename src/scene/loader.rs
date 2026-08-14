// src/scene/loader.rs
// Reads .scene files and returns a list of entity descriptions.
// The engine spawns ECS entities from these descriptions.
//
// ── Data-Driven Scene Format ──────────────────────────────────────────────
// The .scene format is a simple INI-like layout where each [entity] block
// defines one entity with its components. This is the primary data-driven
// mechanism for level designers — no code changes needed to place objects.
//
// Supported fields (all optional except [entity] header):
//   name      = string           Entity name (for editor display)
//   mesh      = path             Mesh file (.obj, .gltf, .glb)
//   position  = x y z            World position (3 floats, space-separated)
//   rotation  = pitch yaw roll   Euler rotation in degrees
//   scale     = x y z  OR  s     Uniform or non-uniform scale
//   material  = name             Material instance name from MaterialLibrary
//   rigidbody = mass             Rigid body with given mass (0 = static)
//   light     = type r g b intensity range   Point light
//   script    = path             Lua script to attach
//   color     = r g b            Override base color (if no material)
//   metallic  = f                Override metallic (if no material)
//   roughness = f                Override roughness (if no material)
//   ao        = f                Override ambient occlusion
//   health    = current max      Health component (tracks damage/destruction)
//   alive     = 0|1              Set 0 to skip spawning (used by save/load)
// ─────────────────────────────────────────────────────────────────────────

// EntityDesc holds everything needed to spawn one entity.
// All fields have defaults — you only write what you need in the scene file.
#[derive(Debug, Clone)]
pub struct EntityDesc {
    pub name:      String,
    pub mesh:      String,          // path to mesh file
    pub position:  [f32; 3],
    pub rotation:  [f32; 3],        // Euler angles in degrees (pitch, yaw, roll)
    pub scale:     [f32; 3],        // non-uniform scale: x, y, z
    pub color:     [f32; 3],
    pub metallic:  f32,
    pub roughness: f32,
    pub ao:        f32,
    pub material:  Option<String>,  // material instance name (overrides color/metallic/roughness)
    pub rigidbody: Option<f32>,     // mass (0 = static, >0 = dynamic)
    pub light:     Option<LightDesc>, // optional point light
    pub script:    Option<String>,  // optional Lua script path
    /// Optional prefab file path. When set, default values come from the prefab
    /// and scene fields override only what's specified.
    pub prefab:    Option<String>,
    /// Optional health (current, max). When set the entity spawns a Health
    /// component so damage/destruction and save/load can track it.
    pub health:    Option<(i32, i32)>,
    /// Whether the entity should be spawned at all. `alive = 0` is used by the
    /// save/load path to keep a killed NPC/collectible from respawning.
    pub alive:     bool,
}

#[derive(Debug, Clone)]
pub struct LightDesc {
    pub light_type: String,         // "point" (future: "spot", "directional")
    pub color:      [f32; 3],
    pub intensity:  f32,
    pub range:      f32,
    pub spot_angle: f32,            // spot cone angle in degrees (spot lights only)
}

impl Default for LightDesc {
    fn default() -> Self {
        Self {
            light_type: "point".to_string(),
            color:      [1.0, 1.0, 1.0],
            intensity:  1.0,
            range:      10.0,
            spot_angle: 45.0,
        }
    }
}

impl Default for EntityDesc {
    fn default() -> Self {
        Self {
            name:      "entity".to_string(),
            mesh:      "meshes/cube.obj".to_string(),
            position:  [0.0, 0.0, 0.0],
            rotation:  [0.0, 0.0, 0.0],
            scale:     [1.0, 1.0, 1.0],   // default: no scaling
            color:     [1.0, 1.0, 1.0],   // default: white
            metallic:  0.0,
            roughness: 0.5,
            ao:        1.0,
            material:  None,
            rigidbody: None,
            light:     None,
            script:    None,
            prefab:    None,
            health:    None,
            alive:     true,
        }
    }
}

// parse_scene() reads a .scene file and returns all entity descriptions.
// Returns an error string if the file can't be read.
pub fn parse_scene(path: &str) -> Result<Vec<EntityDesc>, String> {
    let contents = crate::vfs::read_to_string(path)
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
            "rotation" => {
                // "45.0 90.0 0.0" → [45.0, 90.0, 0.0] (degrees)
                let v = parse_floats(value);
                if v.len() >= 3 { desc.rotation = [v[0], v[1], v[2]]; }
                else if v.len() == 1 { desc.rotation = [0.0, v[0], 0.0]; } // single value = yaw only
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
            "material" => {
                desc.material = Some(value.to_string());
            }
            "prefab" => {
                desc.prefab = Some(value.to_string());
            }
            "rigidbody" => {
                // "0" = static (mass=0), "1.5" = dynamic with mass 1.5
                let mass: f32 = value.parse().unwrap_or(1.0);
                desc.rigidbody = Some(mass);
            }
            "health" => {
                // "current max" → (current, max). Single value = current == max.
                let v = parse_floats(value);
                if v.len() >= 2 {
                    desc.health = Some((v[0] as i32, v[1] as i32));
                } else if v.len() == 1 {
                    let h = v[0] as i32;
                    desc.health = Some((h, h));
                }
            }
            "alive" => {
                // "alive = 1" (default); "alive = 0" → don't spawn.
                desc.alive = value != "0";
            }
            "light" => {
                // "point 1.0 0.9 0.8 2.0 15.0"
                // "spot 1.0 0.9 0.8 2.0 15.0 30.0"  (7th token = cone angle, degrees)
                // type r g b intensity range [spot_angle]
                let tokens: Vec<&str> = value.split_whitespace().collect();
                if tokens.len() >= 6 {
                    desc.light = Some(LightDesc {
                        light_type: tokens[0].to_string(),
                        color: [
                            tokens[1].parse().unwrap_or(1.0),
                            tokens[2].parse().unwrap_or(1.0),
                            tokens[3].parse().unwrap_or(1.0),
                        ],
                        intensity: tokens[4].parse().unwrap_or(1.0),
                        range:     tokens[5].parse().unwrap_or(10.0),
                        spot_angle: tokens.get(6)
                            .and_then(|t| t.parse::<f32>().ok())
                            .map(|a| a.clamp(5.0, 170.0))
                            .unwrap_or(45.0),
                    });
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(contents: &str) -> Vec<EntityDesc> {
        let name = std::thread::current()
            .name()
            .unwrap_or("unknown")
            .replace("::", "_");
        let path = std::env::temp_dir().join(format!("trinity_{}_{}.scene", std::process::id(), name));
        std::fs::write(&path, contents).expect("write temp scene");
        let result = parse_scene(&path.to_string_lossy());
        let _ = std::fs::remove_file(&path);
        result.unwrap_or_else(|e| panic!("parse failed: {}", e))
    }

    #[test]
    fn parses_spot_light_cone_angle() {
        let entities = parse_str(
            "# scene\n\
             [entity]\n\
             name = spotlight\n\
             light = spot 1.0 0.9 0.8 2.0 15.0 30.0\n",
        );
        assert_eq!(entities.len(), 1);
        let light = entities[0].light.as_ref().expect("light parsed");
        assert_eq!(light.light_type, "spot");
        assert_eq!(light.spot_angle, 30.0);
    }

    #[test]
    fn defaults_cone_angle_when_missing() {
        let entities = parse_str(
            "# scene\n\
             [entity]\n\
             name = pointlight\n\
             light = point 1.0 1.0 1.0 1.0 10.0\n",
        );
        let light = entities[0].light.as_ref().expect("light parsed");
        assert_eq!(light.light_type, "point");
        assert_eq!(light.spot_angle, 45.0);
    }

    #[test]
    fn clamps_cone_angle_to_valid_range() {
        let entities = parse_str(
            "# scene\n\
             [entity]\n\
             name = wide\n\
             light = spot 1.0 1.0 1.0 1.0 10.0 999.0\n",
        );
        let light = entities[0].light.as_ref().expect("light parsed");
        assert_eq!(light.spot_angle, 170.0);
    }
}
