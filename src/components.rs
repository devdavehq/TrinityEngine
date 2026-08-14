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

impl Default for Rotation {
    fn default() -> Self {
        Self { pitch: 0.0, yaw: 0.0, roll: 0.0 }
    }
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

// SphereCollider — sphere primitive for physics.
// Supports trigger mode (generates events without velocity response).
#[derive(Clone, Copy)]
pub struct SphereCollider {
    pub radius: f32,
    pub layer: u32,
    pub mask: u32,
    pub is_trigger: bool,
}

impl SphereCollider {
    pub fn new(radius: f32) -> Self {
        Self { radius, layer: 1, mask: u32::MAX, is_trigger: false }
    }
}

impl Default for SphereCollider {
    fn default() -> Self {
        Self::new(0.5)
    }
}

// CapsuleCollider — cylinder with hemispherical caps along a local axis.
// half_height is the distance from center to the start of the cap (not total half-height).
// total height = 2 * (half_height + radius).
#[derive(Clone, Copy)]
pub struct CapsuleCollider {
    pub radius: f32,
    pub half_height: f32,
    pub layer: u32,
    pub mask: u32,
    pub is_trigger: bool,
}

impl CapsuleCollider {
    pub fn new(radius: f32, half_height: f32) -> Self {
        Self { radius, half_height, layer: 1, mask: u32::MAX, is_trigger: false }
    }
}

impl Default for CapsuleCollider {
    fn default() -> Self {
        Self::new(0.3, 0.5)
    }
}

// PhysicsMaterial — defines surface properties for collision response.
// Separate from RigidBody so multiple bodies can share the same material.
// This is the foundation for a full physics material system like Chaos/PhysX.
#[derive(Clone, Copy)]
pub struct PhysicsMaterial {
    /// Static friction coefficient (0 = ice, >1 = rubber).
    pub static_friction: f32,
    /// Dynamic friction coefficient (usually slightly less than static).
    pub dynamic_friction: f32,
    /// Restitution (bounciness): 0 = inelastic, 1 = perfectly elastic.
    pub restitution: f32,
    /// How to combine friction with the other material in a collision.
    pub friction_combine: CombineMode,
    /// How to combine restitution with the other material in a collision.
    pub restitution_combine: CombineMode,
    /// Density in kg/m^3 (affects mass when computed from volume).
    pub density: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CombineMode {
    /// Average: (a + b) / 2
    Average,
    /// Minimum: min(a, b)
    Minimum,
    /// Maximum: max(a, b)
    Maximum,
    /// Multiply: a * b
    Multiply,
}

impl PhysicsMaterial {
    /// Default physical material (generic solid object).
    pub fn solid() -> Self {
        Self {
            static_friction: 0.6,
            dynamic_friction: 0.5,
            restitution: 0.1,
            friction_combine: CombineMode::Average,
            restitution_combine: CombineMode::Average,
            density: 1000.0,
        }
    }

    /// Ice: very slippery, low friction.
    pub fn ice() -> Self {
        Self {
            static_friction: 0.05,
            dynamic_friction: 0.03,
            restitution: 0.2,
            friction_combine: CombineMode::Minimum,
            restitution_combine: CombineMode::Average,
            density: 900.0,
        }
    }

    /// Rubber: high friction, moderate bounce.
    pub fn rubber() -> Self {
        Self {
            static_friction: 1.2,
            dynamic_friction: 1.0,
            restitution: 0.6,
            friction_combine: CombineMode::Average,
            restitution_combine: CombineMode::Maximum,
            density: 1200.0,
        }
    }

    /// Metal: moderate friction, low restitution.
    pub fn metal() -> Self {
        Self {
            static_friction: 0.5,
            dynamic_friction: 0.4,
            restitution: 0.05,
            friction_combine: CombineMode::Average,
            restitution_combine: CombineMode::Average,
            density: 7800.0,
        }
    }

    /// Combined friction value from two materials.
    pub fn combine_friction(a: &Self, b: &Self) -> f32 {
        match (a.friction_combine, b.friction_combine) {
            (CombineMode::Average, _) | (_, CombineMode::Average) => {
                (a.static_friction + b.static_friction) * 0.5
            }
            (CombineMode::Minimum, _) | (_, CombineMode::Minimum) => {
                a.static_friction.min(b.static_friction)
            }
            (CombineMode::Maximum, _) | (_, CombineMode::Maximum) => {
                a.static_friction.max(b.static_friction)
            }
            (CombineMode::Multiply, CombineMode::Multiply) => {
                a.static_friction * b.static_friction
            }
        }
    }

    /// Combined restitution from two materials.
    pub fn combine_restitution(a: &Self, b: &Self) -> f32 {
        match (a.restitution_combine, b.restitution_combine) {
            (CombineMode::Average, _) | (_, CombineMode::Average) => {
                (a.restitution + b.restitution) * 0.5
            }
            (CombineMode::Minimum, _) | (_, CombineMode::Minimum) => {
                a.restitution.min(b.restitution)
            }
            (CombineMode::Maximum, _) | (_, CombineMode::Maximum) => {
                a.restitution.max(b.restitution)
            }
            (CombineMode::Multiply, CombineMode::Multiply) => {
                a.restitution * b.restitution
            }
        }
    }
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self::solid()
    }
}

// CharacterController — first/third-person movement with slope, step, wall handling.
#[derive(Clone, Copy)]
pub struct CharacterController {
    /// Maximum walkable slope angle in radians (default ~50°).
    pub max_slope_angle: f32,
    /// Maximum step height the character can climb in one frame.
    pub step_height: f32,
    /// Skin width for depenetration (small value prevents jitter).
    pub skin_width: f32,
    /// Movement speed in world units/second.
    pub speed: f32,
    /// Initial upward velocity when jumping.
    pub jump_force: f32,
    /// How far down to cast for ground detection.
    pub ground_detect_dist: f32,
    /// Whether the character is currently on the ground.
    pub on_ground: bool,
    /// Gravity multiplier (1.0 = normal, 0.0 = no gravity like kinematic).
    pub gravity_scale: f32,
    /// Whether jump is held this frame.
    pub jump_pressed: bool,
}

impl Default for CharacterController {
    fn default() -> Self {
        Self {
            max_slope_angle: 0.8727, // ~50 degrees in radians
            step_height: 0.35,
            skin_width: 0.02,
            speed: 6.0,
            jump_force: 8.0,
            ground_detect_dist: 0.15,
            on_ground: false,
            gravity_scale: 1.0,
            jump_pressed: false,
        }
    }
}

// Ragdoll — marks an entity as the root of a ragdoll.
// The physics system will drive all bones with ball-socket constraints.
#[derive(Clone)]
pub struct Ragdoll {
    /// The bones that make up this ragdoll, in order from root to extremities.
    pub bones: Vec<RagdollBone>,
}

/// A single bone in a ragdoll chain.
#[derive(Clone, Copy)]
pub struct RagdollBone {
    /// The entity that represents this bone's rigid body.
    pub entity: hecs::Entity,
    /// Index of the parent bone in Ragdoll.bones (-1 = root).
    pub parent_index: i32,
    /// Offset from parent bone center to this bone's center (in parent local space).
    pub local_offset: [f32; 3],
    /// Maximum angle the joint can swing (radians, cone limit).
    pub swing_limit: f32,
    /// Damping applied to the joint spring.
    pub damping: f32,
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

impl Default for Script {
    fn default() -> Self {
        Self { path: String::new() }
    }
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

// Decal — a projector volume that paints albedo onto the G-buffer after the
// geometry pass. The box (transform scale) is where the paint appears; pixels
// of the stored depth that pass through the box get the decal colour blended
// in. Great for bullet holes, warning stripes, road markings, dirt.
#[derive(Clone, Copy)]
pub struct Decal {
    /// Decal colour / tint (RGB).
    pub color: [f32; 3],
    /// Blend opacity (0 = invisible, 1 = full).
    pub opacity: f32,
    /// Rotation in degrees around the decal's facing axis (roll on the surface).
    pub roll_deg: f32,
    /// Projector box size — the volume of the surface that receives the paint.
    pub size: [f32; 3],
}

impl Default for Decal {
    fn default() -> Self {
        Self {
            color: [0.9, 0.2, 0.2],
            opacity: 0.85,
            roll_deg: 0.0,
            size: [2.0, 2.0, 0.6],
        }
    }
}

// PointLight — a local light source (point, spot, or directional).
// Multiple lights are supported via the multi-light uniform buffer (up to 16).
#[derive(Clone, Copy)]
pub struct PointLight {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    /// 0 = directional (sun), 1 = point (omnidirectional), 2 = spot (cone).
    pub light_type: f32,
    /// Spot cone angle in degrees (only used when light_type == 2).
    pub spot_angle: f32,
    /// Whether this light casts shadows.
    pub shadow_casting: bool,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: [1.0, 0.95, 0.85],
            intensity: 1.5,
            range: 12.0,
            light_type: 1.0,     // point
            spot_angle: 45.0,    // 45° cone
            shadow_casting: false,
        }
    }
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

// MaterialExtras — per-entity material shading overrides.
// Maps to GpuMaterialExtras in the renderer (binding 6).
#[derive(Clone, Copy)]
pub struct MaterialExtras {
    /// Subsurface scattering amount (0 = off, 1 = full SSS).
    pub subsurface: f32,
    /// Clearcoat layer strength (0 = off, 1 = full clearcoat).
    pub clearcoat: f32,
    /// Clearcoat roughness (0 = mirror, 1 = rough clearcoat).
    pub clearcoat_roughness: f32,
    /// Emissive intensity multiplier (0 = none, 10 = very bright).
    pub emissive_strength: f32,
}

impl Default for MaterialExtras {
    fn default() -> Self {
        Self {
            subsurface: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            emissive_strength: 0.0,
        }
    }
}

// TerrainBlend — distance-based alpha blending so placed objects smoothly
// merge into the terrain.  The bottom vertices of the mesh fade out based
// on blend_distance (world units of fade zone) and blend_offset (model-space Y shift).
#[derive(Clone, Copy)]
pub struct TerrainBlend {
    /// World-space height of the fade zone at the object's base (0 = disabled).
    pub blend_distance: f32,
    /// Model-space Y offset for the fade origin (shifts the fade zone up/down).
    pub blend_offset: f32,
}

impl Default for TerrainBlend {
    fn default() -> Self {
        Self { blend_distance: 0.0, blend_offset: 0.0 }
    }
}

// Health — tracks hit points for any entity that can take damage.
pub struct Health {
    pub current: i32,
    pub max:     i32,
}

impl Default for Health {
    fn default() -> Self {
        Self { current: 100, max: 100 }
    }
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

// SceneMeta — stores the original entity name and mesh path for scene save.
// Attached to every entity spawned from a .scene file so we can round-trip.
pub struct SceneMeta {
    pub name: String,
    pub mesh_path: String,
}

// NetId — marks an entity as shared over the network. The host replicates every
// NetId entity in its snapshot; clients apply updates to local copies with the
// same id. Simply adding this component to both sides makes an entity
// multiplayer-aware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NetId {
    pub id: u32,
}

// Occluder — marks an entity as a large static volume that should occlude
// geometry behind it. Used by the software occlusion culler to reject hidden
// meshes. Entities with this component are still rendered normally; they just
// also feed the occlusion grid.
#[derive(Clone, Copy)]
pub struct Occluder {
    /// Radius of the occluding volume in world units. Larger = hides more.
    pub radius: f32,
}

impl Default for Occluder {
    fn default() -> Self {
        Self { radius: 5.0 }
    }
}

// WaterSurface — marks an entity as a water body.
// The water renderer draws these with a special shader that handles:
//   - Gerstner wave vertex displacement
//   - Fresnel-based reflection/refraction blending
//   - Depth-based colour absorption
//   - Foam at wave peaks
//   - Transparency and refraction of the scene below
#[derive(Clone, Copy)]
pub struct WaterSurface {
    /// Wave height (metres). 0 = flat calm, 2+ = stormy.
    pub wave_height: f32,
    /// Wave speed multiplier.
    pub wave_speed: f32,
    /// Deep water colour (RGB 0-1).
    pub deep_color: [f32; 3],
    /// Shallow water colour (RGB 0-1).
    pub shallow_color: [f32; 3],
    /// Surface opacity (0 = fully transparent, 1 = fully opaque).
    pub opacity: f32,
    /// Foam intensity at wave crests (0 = none).
    pub foam_intensity: f32,
    /// Specular highlight power (higher = tighter highlight).
    pub specular_power: f32,
}

impl Default for WaterSurface {
    fn default() -> Self {
        Self {
            wave_height: 0.3,
            wave_speed: 1.0,
            deep_color: [0.01, 0.06, 0.15],
            shallow_color: [0.05, 0.25, 0.35],
            opacity: 0.85,
            foam_intensity: 0.15,
            specular_power: 256.0,
        }
    }
}

// LavaSurface — marks an entity as a lava/magma body.
// The lava renderer draws these with a special shader that handles:
//   - Animated emissive flow patterns (scrolling noise)
//   - Molten crack patterns with bright glow
//   - Dark rocky base with hot cracks
//   - Heat distortion via vertex displacement
//   - Glow bloom contribution
#[derive(Clone, Copy)]
pub struct LavaSurface {
    /// Base colour of cooled rock (RGB 0-1).
    pub rock_color: [f32; 3],
    /// Emissive colour of molten cracks (RGB 0-1).
    pub emissive_color: [f32; 3],
    /// Emissive intensity multiplier (higher = brighter glow, drives bloom).
    pub emissive_intensity: f32,
    /// Flow speed of the crack pattern (UV scroll speed).
    pub flow_speed: f32,
    /// Scale of the crack pattern (smaller = larger cracks).
    pub crack_scale: f32,
    /// Brightness threshold for crack visibility (0-1).
    pub crack_threshold: f32,
    /// Vertex displacement amplitude for heat shimmer.
    pub displacement_amp: f32,
    /// Overall opacity (0 = invisible, 1 = fully opaque).
    pub opacity: f32,

    // ── Dynamic light emission fields ──────────────────────────────────────
    // These control a point light that is automatically spawned at the entity's
    // position each frame, giving lava surfaces RDR2-quality dynamic lighting
    // that illuminates nearby geometry in real-time.

    /// Intensity of the dynamic point light emitted by this lava surface.
    /// Higher values cast brighter light onto surrounding surfaces.
    pub emissive_light_strength: f32,
    /// Radius (range) of the dynamic point light in world units.
    /// Controls how far the light reaches from the lava surface.
    pub emissive_light_radius: f32,
    /// RGB colour of the dynamic point light (0-1 per channel).
    /// Defaults to a deep molten orange to match the emissive crack colour.
    pub emissive_light_color: [f32; 3],
}

impl Default for LavaSurface {
    fn default() -> Self {
        Self {
            rock_color:       [0.08, 0.02, 0.01],
            emissive_color:   [1.0, 0.3, 0.02],
            emissive_intensity: 3.0,
            flow_speed:       0.15,
            crack_scale:      3.0,
            crack_threshold:  0.45,
            displacement_amp: 0.08,
            opacity:          1.0,
            // Dynamic light defaults — deep orange glow, moderate range.
            emissive_light_strength: 1.5,
            emissive_light_radius:   12.0,
            emissive_light_color:    [1.0, 0.3, 0.02],
        }
    }
}

// FireSurface — marks an entity as a fire rendering surface.
// The fire renderer draws these with a special shader that handles:
//   - Animated procedural flame shape (scrolling FBM noise)
//   - Height-based colour gradient (white-hot → orange → red tips)
//   - Semi-transparent with additive blending
//   - Flickering wind displacement
//   - Emissive output that drives bloom
#[derive(Clone, Copy)]
pub struct FireSurface {
    /// Base flame colour (RGB 0-1). Bright = white-hot, orange = typical fire.
    pub base_color: [f32; 3],
    /// Tip colour (RGB 0-1). Darker red = dying flame tips.
    pub tip_color: [f32; 3],
    /// Emissive intensity multiplier (higher = brighter glow, drives bloom).
    pub intensity: f32,
    /// Flame animation speed (UV scroll rate).
    pub flame_speed: f32,
    /// Noise scale (smaller = larger flame features).
    pub noise_scale: f32,
    /// How much the flame flickers sideways.
    pub flicker_strength: f32,
    /// Height of the flame in world units.
    pub flame_height: f32,
    /// Overall opacity (0 = invisible, 1 = fully visible).
    pub opacity: f32,

    // ── Dynamic light emission fields ──────────────────────────────────────
    // These control a point light that is automatically spawned at the entity's
    // position each frame, giving fire surfaces RDR2-quality dynamic lighting
    // that casts flickering orange light onto nearby geometry.

    /// Intensity of the dynamic point light emitted by this fire surface.
    /// This value is modulated by flicker_strength each frame for realistic
    /// fire-light dancing on nearby surfaces.
    pub emissive_light_strength: f32,
    /// Radius (range) of the dynamic point light in world units.
    /// Controls how far the fire light reaches from the flame.
    pub emissive_light_radius: f32,
    /// RGB colour of the dynamic point light (0-1 per channel).
    /// Defaults to warm orange to match typical fire illumination.
    pub emissive_light_color: [f32; 3],
}

impl Default for FireSurface {
    fn default() -> Self {
        Self {
            base_color:       [1.0, 0.7, 0.15],
            tip_color:        [0.8, 0.15, 0.02],
            intensity:        4.0,
            flame_speed:      0.3,
            noise_scale:      2.5,
            flicker_strength: 0.15,
            flame_height:     2.0,
            opacity:          0.9,
            // Dynamic light defaults — warm fire glow, moderate range.
            emissive_light_strength: 2.0,
            emissive_light_radius:   8.0,
            emissive_light_color:    [1.0, 0.6, 0.1],
        }
    }
}

// WaterTrigger — marks an entity as a water surface that detects entry.
// When another entity's collider enters this volume, a WaterSplashEvent fires.
#[derive(Clone, Copy)]
pub struct WaterTrigger {
    /// Splash intensity multiplier (0-1).
    pub splash_intensity: f32,
    /// Whether this trigger is currently active.
    pub active: bool,
}

impl Default for WaterTrigger {
    fn default() -> Self {
        Self {
            splash_intensity: 1.0,
            active: true,
        }
    }
}

// FireSource — marks an entity as a fire emitter.
// The particle system spawns fire, smoke, and ember particles from this entity.
// A PointLight component on the same entity provides dynamic firelight.
#[derive(Clone, Copy)]
pub struct FireSource {
    /// Fire intensity (0 = dying embers, 1 = roaring blaze).
    pub intensity: f32,
    /// Radius of the fire effect in world units.
    pub radius: f32,
    /// Height of the flame column.
    pub flame_height: f32,
    /// How much smoke is produced (0 = clean flame, 1 = heavy smoke).
    pub smoke_amount: f32,
    /// How many embers float upward (0 = none, 1 = many).
    pub ember_amount: f32,
    /// Wind susceptibility (0 = campfire不受风, 1 = fully wind-driven).
    pub wind_susceptibility: f32,
    /// Whether the fire damages entities that enter it.
    pub damaging: bool,
    /// Damage per second when inside the fire radius.
    pub damage_per_second: f32,
}

impl Default for FireSource {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            radius: 1.0,
            flame_height: 1.5,
            smoke_amount: 0.3,
            ember_amount: 0.15,
            wind_susceptibility: 0.4,
            damaging: true,
            damage_per_second: 10.0,
        }
    }
}

// WeatherZone — a spherical region where weather differs from the global default.
// Entities inside the zone receive the zone's weather; entities outside get global.
// Zones blend at their edges for smooth transitions.
#[derive(Clone, Copy)]
pub struct WeatherZone {
    /// Weather condition within this zone.
    pub condition: crate::environment::weather::WeatherCondition,
    /// Weather intensity within this zone (0-1).
    pub intensity: f32,
    /// Radius of the zone in world units.
    pub radius: f32,
    /// Falloff distance at the edge for smooth blending (0 = hard edge).
    pub falloff: f32,
    /// Whether this zone is active.
    pub active: bool,
}

impl Default for WeatherZone {
    fn default() -> Self {
        Self {
            condition: crate::environment::weather::WeatherCondition::Clear,
            intensity: 0.5,
            radius: 50.0,
            falloff: 10.0,
            active: true,
        }
    }
}

// WindZone — a spherical region with localized wind direction and strength.
// Entities inside receive this wind instead of the global wind.
// Affects water (wave direction), foliage (sway), and particles (drift).
#[derive(Clone, Copy)]
pub struct WindZone {
    /// Wind direction (will be normalized internally).
    pub direction: [f32; 3],
    /// Wind strength in m/s (0 = calm, 1 = strong).
    pub strength: f32,
    /// Radius of influence in world units.
    pub radius: f32,
    /// Falloff distance at the edge for smooth blending.
    pub falloff: f32,
    /// Whether this zone is active.
    pub active: bool,
}

impl Default for WindZone {
    fn default() -> Self {
        Self {
            direction: [1.0, 0.0, 0.0],
            strength: 0.5,
            radius: 30.0,
            falloff: 10.0,
            active: true,
        }
    }
}

// SplashEffect — marks an entity as having splash effects when entities enter its water.
#[derive(Clone, Copy)]
pub struct SplashEffect {
    /// Maximum number of concurrent splash particle systems.
    pub max_splashes: u32,
    /// Duration of each splash in seconds.
    pub splash_duration: f32,
    /// Ripple ring scale multiplier.
    pub ripple_scale: f32,
    /// Whether this effect is active.
    pub active: bool,
}

impl Default for SplashEffect {
    fn default() -> Self {
        Self {
            max_splashes: 8,
            splash_duration: 1.0,
            ripple_scale: 1.0,
            active: true,
        }
    }
}

// ── Entity Hierarchy ──────────────────────────────────────────────────────
// Parent/Children components for scene graph hierarchy.

/// Marks this entity as a child of another entity.
/// The parent must exist in the world.
#[derive(Clone, Copy, Debug)]
pub struct Parent {
    /// The parent entity.
    pub entity: hecs::Entity,
}

/// Marks this entity as a parent with children.
/// Children list is stored here for fast iteration.
#[derive(Clone, Debug)]
pub struct Children {
    /// Ordered list of child entities.
    pub entities: Vec<hecs::Entity>,
}

impl Children {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }
}

impl Default for Children {
    fn default() -> Self {
        Self::new()
    }
}

// ── Smart Water System ─────────────────────────────────────────────────────
// Water body types enable auto-generated water surfaces, physics volumes,
// collision, reflections, and underwater effects per body type.

/// Water body type classification. Each type auto-configures wave params,
/// physics, rendering, and interaction behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WaterBodyType {
    Ocean,
    Lake,
    River,
    Pond,
    Stream,
    Waterfall,
    Swamp,
}

impl WaterBodyType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ocean => "Ocean",
            Self::Lake => "Lake",
            Self::River => "River",
            Self::Pond => "Pond",
            Self::Stream => "Stream",
            Self::Waterfall => "Waterfall",
            Self::Swamp => "Swamp",
        }
    }
}

impl Default for WaterBodyType {
    fn default() -> Self { Self::Lake }
}

/// Smart Water Body component — replaces plain WaterSurface for placed water.
/// Auto-generates surface mesh, material, physics volume, swimming volume,
/// collision, reflections, LOD, streaming, and underwater rendering.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct WaterBody {
    pub body_type: WaterBodyType,
    pub size_x: f32,
    pub size_z: f32,
    pub depth: f32,
    pub flow_direction: [f32; 3],
    pub flow_speed: f32,
    pub turbulence: f32,
    pub auto_surface: bool,
    pub auto_physics: bool,
    pub auto_collision: bool,
    pub auto_reflections: bool,
    pub auto_underwater: bool,
    pub lod_distance: f32,
}

impl Default for WaterBody {
    fn default() -> Self {
        Self {
            body_type: WaterBodyType::Lake,
            size_x: 50.0,
            size_z: 50.0,
            depth: 10.0,
            flow_direction: [1.0, 0.0, 0.0],
            flow_speed: 1.0,
            turbulence: 0.3,
            auto_surface: true,
            auto_physics: true,
            auto_collision: true,
            auto_reflections: true,
            auto_underwater: true,
            lod_distance: 200.0,
        }
    }
}

/// Underwater rendering settings applied when camera is below the waterline.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct UnderwaterSettings {
    pub tint: [f32; 3],
    pub fog_density: f32,
    pub caustics_intensity: f32,
    pub god_rays_intensity: f32,
    pub bloom_strength: f32,
    pub distortion_strength: f32,
    pub swimming_enabled: bool,
    pub buoyancy_force: f32,
}

impl Default for UnderwaterSettings {
    fn default() -> Self {
        Self {
            tint: [0.01, 0.08, 0.12],
            fog_density: 0.04,
            caustics_intensity: 0.6,
            god_rays_intensity: 0.3,
            bloom_strength: 0.15,
            distortion_strength: 0.003,
            swimming_enabled: true,
            buoyancy_force: 8.0,
        }
    }
}

/// Marks an entity as a "folder" or "group" node in the hierarchy.
/// Folder nodes don't have renderables — they're purely organizational.

/// Terrain brush mode — controls which operation the terrain brush performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TerrainBrushMode {
    Raise,
    Lower,
    Smooth,
    Flatten,
    Paint,
    Foliage,
}

impl TerrainBrushMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Raise   => "Raise",
            Self::Lower   => "Lower",
            Self::Smooth  => "Smooth",
            Self::Flatten => "Flatten",
            Self::Paint   => "Paint",
            Self::Foliage => "Foliage",
        }
    }
    pub fn key_hint(self) -> &'static str {
        match self {
            Self::Raise   => "1",
            Self::Lower   => "2",
            Self::Smooth  => "3",
            Self::Flatten => "4",
            Self::Paint   => "5",
            Self::Foliage => "6",
        }
    }
}

impl Default for TerrainBrushMode {
    fn default() -> Self { Self::Raise }
}

/// Terrain editor component — attach to any entity to enable terrain brush editing
/// when that entity is selected. The brush applies operations to the global TerrainWorld
/// at the cursor position projected from the viewport.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct TerrainEditor {
    pub active: bool,
    pub brush_mode: TerrainBrushMode,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub flatten_target: f32,
    pub show_cursor: bool,
}

impl Default for TerrainEditor {
    fn default() -> Self {
        Self {
            active: false,
            brush_mode: TerrainBrushMode::Raise,
            brush_radius: 5.0,
            brush_strength: 0.5,
            flatten_target: 0.0,
            show_cursor: true,
        }
    }
}

/// Per-asset foliage settings for the Smart Foliage system.
/// Controls visibility, locking, density, scale, rotation, slope/height limits
/// for individual foliage species placed by the foliage painter.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SmartFoliageAsset {
    pub visible: bool,
    pub locked: bool,
    pub density_multiplier: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub random_rotation: bool,
    pub min_slope_deg: f32,
    pub max_slope_deg: f32,
    pub min_height: f32,
    pub max_height: f32,
    pub paint_mode: FoliagePaintMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FoliagePaintMode {
    Paint,
    Erase,
    Fill,
    Procedural,
}

impl Default for SmartFoliageAsset {
    fn default() -> Self {
        Self {
            visible: true,
            locked: false,
            density_multiplier: 1.0,
            min_scale: 0.8,
            max_scale: 1.2,
            random_rotation: true,
            min_slope_deg: 0.0,
            max_slope_deg: 60.0,
            min_height: -10.0,
            max_height: 500.0,
            paint_mode: FoliagePaintMode::Paint,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct GroupNode {
    /// Display name for this group.
    pub name: [u8; 32],
    /// Whether this group is expanded in the editor outliner.
    pub expanded: bool,
}

impl GroupNode {
    pub fn new(name: &str) -> Self {
        let mut bytes = [0u8; 32];
        let name_bytes = name.as_bytes();
        let len = name_bytes.len().min(32);
        bytes[..len].copy_from_slice(&name_bytes[..len]);
        Self {
            name: bytes,
            expanded: true,
        }
    }

    pub fn name_str(&self) -> String {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(32);
        String::from_utf8_lossy(&self.name[..end]).to_string()
    }
}
