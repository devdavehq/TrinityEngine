use std::collections::HashMap;

use crate::components::Renderable;

#[derive(Clone, Copy)]
pub struct MasterMaterial {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
}

#[derive(Clone, Copy)]
pub struct MaterialInstance {
    pub master: &'static str,
    pub color_tint: [f32; 3],
    pub metallic_mul: f32,
    pub roughness_mul: f32,
    pub ao_mul: f32,
}

pub struct MaterialLibrary {
    masters: HashMap<&'static str, MasterMaterial>,
    instances: HashMap<&'static str, MaterialInstance>,
}

impl MaterialLibrary {
    pub fn new_defaults() -> Self {
        let mut masters = HashMap::new();
        masters.insert(
            "master_surface",
            MasterMaterial {
                base_color: [1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.7,
                ao: 1.0,
            },
        );
        masters.insert(
            "master_metal",
            MasterMaterial {
                base_color: [0.82, 0.82, 0.82],
                metallic: 1.0,
                roughness: 0.3,
                ao: 1.0,
            },
        );
        masters.insert(
            "master_foliage",
            MasterMaterial {
                base_color: [0.18, 0.42, 0.20],
                metallic: 0.0,
                roughness: 0.9,
                ao: 1.0,
            },
        );

        let mut instances = HashMap::new();
        instances.insert(
            "matte_black",
            MaterialInstance {
                master: "master_surface",
                color_tint: [0.12, 0.12, 0.12],
                metallic_mul: 0.0,
                roughness_mul: 1.1,
                ao_mul: 1.0,
            },
        );
        instances.insert(
            "silver_brushed",
            MaterialInstance {
                master: "master_metal",
                color_tint: [0.92, 0.92, 0.94],
                metallic_mul: 1.0,
                roughness_mul: 1.2,
                ao_mul: 1.0,
            },
        );
        instances.insert(
            "foliage_leaf",
            MaterialInstance {
                master: "master_foliage",
                color_tint: [0.20, 0.55, 0.22],
                metallic_mul: 0.0,
                roughness_mul: 1.0,
                ao_mul: 1.0,
            },
        );

        Self { masters, instances }
    }

    /// Names of material instances (for editor buttons).
    pub fn instance_names(&self) -> Vec<&'static str> {
        let mut v: Vec<_> = self.instances.keys().copied().collect();
        v.sort();
        v
    }

    /// Registered master material ids.
    pub fn master_names(&self) -> Vec<&'static str> {
        let mut v: Vec<_> = self.masters.keys().copied().collect();
        v.sort();
        v
    }

    pub fn print_help() {
        println!("[Materials] Master + Instance workflow (no node graph):");
        println!("  1 = apply 'matte_black' to selected entity");
        println!("  2 = apply 'silver_brushed' to selected entity");
        println!("  3 = apply 'foliage_leaf' to selected entity");
        println!("  N / M = select previous/next renderable entity");
    }

    pub fn apply_instance(&self, name: &str, renderable: &mut Renderable) -> Result<(), String> {
        let inst = self
            .instances
            .get(name)
            .ok_or_else(|| format!("Material instance '{}' not found", name))?;
        let master = self
            .masters
            .get(inst.master)
            .ok_or_else(|| format!("Master material '{}' not found", inst.master))?;

        renderable.color = [
            (master.base_color[0] * inst.color_tint[0]).clamp(0.0, 1.0),
            (master.base_color[1] * inst.color_tint[1]).clamp(0.0, 1.0),
            (master.base_color[2] * inst.color_tint[2]).clamp(0.0, 1.0),
        ];
        renderable.metallic = (master.metallic * inst.metallic_mul).clamp(0.0, 1.0);
        renderable.roughness = (master.roughness * inst.roughness_mul).clamp(0.02, 1.0);
        renderable.ao = (master.ao * inst.ao_mul).clamp(0.0, 1.0);
        Ok(())
    }
}
