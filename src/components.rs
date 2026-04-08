// src/components.rs — all ECS component types in one file.
// Small enough that splitting further would just create noise.

use crate::assets::{Handle, Mesh};

// Position in 3D world space.
#[derive(Clone, Copy)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// Rotation in radians (pitch, yaw, roll).
#[derive(Clone, Copy)]
pub struct Rotation {
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
}

// Velocity in world units per second.
#[allow(dead_code)]
pub struct Velocity {
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
}

// RigidBody — physics simulation state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyType {
    Static,
    Dynamic,
    Kinematic,
}

#[derive(Clone, Copy)]
pub struct RigidBody {
    pub body_type:   BodyType,
    pub velocity_x:  f32,
    pub velocity_y:  f32,
    pub _velocity_z: f32,
    pub angular_velocity: f32,
    pub angular_damping: f32,
    pub torque: f32,
    pub on_ground:   bool,
    pub use_gravity: bool,
    pub mass:        f32,
    pub inertia:     f32,
    pub restitution: f32,
    pub friction:    f32,
    pub linear_damping: f32,
    pub lock_rotation: bool,
    pub can_sleep:   bool,
    pub sleeping:    bool,
    pub sleep_timer: f32,
}

impl RigidBody {
    pub fn dynamic() -> Self {
        Self {
            body_type: BodyType::Dynamic,
            velocity_x: 0.0,
            velocity_y: 0.0,
            _velocity_z: 0.0,
            angular_velocity: 0.0,
            angular_damping: 0.16,
            torque: 0.0,
            on_ground: false,
            use_gravity: true,
            mass: 1.0,
            inertia: 1.0,
            restitution: 0.0,
            friction: 0.55,
            linear_damping: 0.08,
            lock_rotation: false,
            can_sleep: true,
            sleeping: false,
            sleep_timer: 0.0,
        }
    }

    pub fn kinematic() -> Self {
        let mut body = Self::dynamic();
        body.body_type = BodyType::Kinematic;
        body.use_gravity = false;
        body.on_ground = true;
        body.lock_rotation = true;
        body.can_sleep = false;
        body
    }

    #[allow(dead_code)]
    pub fn static_body() -> Self {
        let mut body = Self::dynamic();
        body.body_type = BodyType::Static;
        body.use_gravity = false;
        body.on_ground = true;
        body.mass = 0.0;
        body.inertia = 0.0;
        body.lock_rotation = true;
        body.can_sleep = false;
        body
    }
}

#[derive(Clone, Copy)]
pub struct HingeJoint {
    pub connected: hecs::Entity,
    pub rest_length: f32,
    pub stiffness: f32,
    pub anchor_a: [f32; 3],
    pub anchor_b: [f32; 3],
}

#[derive(Clone, Copy)]
pub struct FixedJoint {
    pub connected: hecs::Entity,
    pub offset_x: f32,
    pub offset_y: f32,
    pub stiffness: f32,
    pub anchor_a: [f32; 3],
    pub anchor_b: [f32; 3],
}

#[derive(Clone, Copy)]
pub struct SpringJoint {
    pub connected: hecs::Entity,
    pub rest_length: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub anchor_a: [f32; 3],
    pub anchor_b: [f32; 3],
}

#[derive(Clone, Copy)]
pub struct RopeConstraint {
    pub connected: hecs::Entity,
    pub max_length: f32,
    pub stiffness: f32,
    pub anchor_a: [f32; 3],
    pub anchor_b: [f32; 3],
}

// Collider — axis-aligned bounding box.
#[derive(Clone, Copy)]
pub struct Collider {
    pub half_w: f32,
    pub half_h: f32,
    #[allow(dead_code)]
    pub half_d: f32,  // depth for 3D
    pub layer: u32,
    pub mask: u32,
}

// OrientedBoxCollider — 2D rotation-aware physics box (X/Y plane).
// Angle is in radians. Used by SAT overlap tests in physics.
#[derive(Clone, Copy)]
pub struct OrientedBoxCollider {
    pub half_w: f32,
    pub half_h: f32,
    pub half_d: f32,
    pub angle_rad: f32,
    pub layer: u32,
    pub mask: u32,
}

// FoliageWind stores base pose and wind sway parameters for vegetation.
pub struct FoliageWind {
    pub base_x: f32,
    pub base_z: f32,
    pub amplitude: f32,
    pub frequency: f32,
}

// Script — attaches a Lua script to an entity.
pub struct Script {
    pub path: String,
}

// MaterialTexture keeps the content texture path assigned by editor/tools.
#[derive(Clone)]
pub struct MaterialTexture {
    pub path: String,
    pub normal_path: String,
    pub metallic_roughness_path: String,
}

// PlayerStart marks where player-controlled entities spawn in Game Preview.
pub struct PlayerStart;

// PointLight is a simple movable local light source.
#[derive(Clone, Copy)]
pub struct PointLight {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

// Renderable — everything the renderer needs to draw an entity.
// Now carries PBR material properties directly.
#[derive(Clone, Copy)]
pub struct Renderable {
    pub mesh:      Handle<Mesh>,

    // PBR material values.
    pub color:     [f32; 3],  // albedo / base color
    pub metallic:  f32,       // 0 = plastic, 1 = metal
    pub roughness: f32,       // 0 = mirror,  1 = matte
    pub ao:        f32,       // ambient occlusion 0..1

    // Non-uniform scale so we can make flat floors, thin walls, etc.
    // [1.0, 0.2, 10.0] = normal width, very flat, very deep
    pub scale:     [f32; 3],
}


// Health — tracks hit points for any entity that can take damage.
pub struct Health {
    pub current: i32,
    pub max:     i32,
}

#[allow(dead_code)]
impl Health {
    pub fn new(max: i32) -> Self {
        Self { current: max, max }
    }

    // is_dead() — convenience check used by both Rust and the Lua binding.
    pub fn is_dead(&self) -> bool {
        self.current <= 0
    }
}



// CollisionPair — generated by physics when two entities overlap.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CollisionPair {
    pub entity_a: hecs::Entity,
    pub entity_b: hecs::Entity,
    pub normal: [f32; 3],
    pub penetration: f32,
    pub phase: CollisionPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionPhase {
    Started,
    Ongoing,
    Ended,
}
