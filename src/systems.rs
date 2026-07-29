use crate::camera::Camera2D;
use crate::components::{Position, Velocity};
use hecs::World;

use crate::components::Script;
use crate::input::InputState;
use crate::scripting::ScriptEngine;
use crate::audio::AudioSystem;
use crate::ai::AiRegistry;
use crate::navigation::NavGrid;
use crate::terrain::TerrainWorld;
use crate::core::systems::SystemScheduler;

/// EngineSystems wraps the SystemScheduler and provides a central registration
/// point for engine systems that are currently called inline in main.rs.
///
/// This is a FOUNDATION for migrating the monolithic GameApp::frame() loop
/// into a data-driven, composable pipeline. Systems are registered here but
/// NOT yet removed from main.rs — the migration is incremental.
pub struct EngineSystems {
    pub scheduler: SystemScheduler,
}

impl EngineSystems {
    pub fn new() -> Self {
        let scheduler = SystemScheduler::new();
        // Register the systems that are currently called inline in main.rs.
        // DO NOT remove them from main.rs yet — just register them for future use.
        //
        // TODO: Implement SystemMut for each system and register them here:
        //   scheduler.register(Box::new(AnimationSystem));
        //   scheduler.register(Box::new(AnimationBlendingSystem));
        //   scheduler.register(Box::new(FloodSystem));
        //   scheduler.register(Box::new(AiSystem));
        //   scheduler.resolve().expect("System dependency cycle detected");
        Self { scheduler }
    }
}

impl Default for EngineSystems {
    fn default() -> Self {
        Self::new()
    }
}

// scripting_system() runs the Lua update() for every entity with a Script.
//
// Why pass ScriptEngine by reference?
// ScriptEngine owns the Lua runtime. We borrow it to call scripts.
// We don't want the system to own it — main.rs should own it.
pub fn scripting_system(
    world: &mut World,
    scripts: &mut ScriptEngine,
    input: &InputState,
    camera_pos: [f32; 3],
    camera_target: [f32; 3],
    dt: f32,
    mut audio: Option<&mut AudioSystem>,
    nav_grid: &NavGrid,
    ai_registry: &mut AiRegistry,
    terrain_world: &mut TerrainWorld,
    screen_w: f32,
    screen_h: f32,
    camera_fov: f32,
) {
    // We can't query and mutate world at the same time with hecs,
    // so we collect the (entity, path) pairs first, then run scripts.
    // Why: run_update() needs &mut World, but the query already borrows it.
    // Collecting into a Vec ends the borrow before we call run_update().
    let script_entities: Vec<(hecs::Entity, String)> = world
        .query::<(hecs::Entity, &Script)>()
        .iter()
        .map(|(entity, script)| (entity, script.path.clone()))
        .collect();

    // Now run each script — world is free to borrow mutably.
    // Provide NavGrid and AiRegistry pointers for bt/nav Lua APIs.
    scripts.set_external_refs(nav_grid, ai_registry, terrain_world);
    for (entity, path) in script_entities {
        // Load and run the script for this entity.
        // Why load here? For simplicity — later we'll cache loaded scripts.
        // If the script fails, we print the error and continue.
        // A scripting error should never crash the engine.
        if let Err(e) = scripts.run_update(
            world,
            input,
            camera_pos,
            camera_target,
            entity,
            &path,
            dt,
            audio.as_deref_mut(),
            screen_w,
            screen_h,
            camera_fov,
        ) {
            tracing::error!("[Scripting] Error in {}: {}", path, e);
        }
    }
}

#[allow(dead_code)]
pub fn movement_system(world: &mut World) {
    for (pos, vel) in world.query_mut::<(&mut Position, &Velocity)>() {
        pos.x += vel.dx;
        pos.y += vel.dy;
    }
}

// camera_follow_system makes the camera track an entity smoothly.
//
// Why "follow" instead of "snap"?
// Snapping (setting camera position = entity position every frame) feels rigid.
// Smooth following (moving the camera a fraction toward the target each frame)
// feels natural — the camera has a little "lag" that feels good in games.
//
// Parameters:
//   camera        — mutable ref to the camera we want to move
//   world         — ECS world to query the target entity's position
//   target        — which entity to follow (the player's entity ID)
//   follow_speed  — how quickly the camera catches up, 0.0–1.0
//                   0.1 = slow/floaty, 0.3 = snappy, 1.0 = instant snap
#[allow(dead_code)]
pub fn camera_follow_system(
    camera: &mut Camera2D,
    world: &World,
    target: hecs::Entity,
    follow_speed: f32,
) {
    // Try to get the Position component of the target entity.
    // get::<&Position>(target) returns Result — Ok if found, Err if not.
    // "if let Ok(pos)" means: only run the body if we successfully got it.
    // If the entity was destroyed (died, despawned), this safely does nothing.
    if let Ok(pos) = world.get::<&Position>(target) {
        // Lerp = linear interpolation.
        // lerp(a, b, t) moves from a toward b by fraction t.
        // t = follow_speed: 0.1 moves 10% of the remaining distance each frame.
        // This creates smooth exponential easing — fast at first, slowing as it arrives.
        //
        // Why glam::Vec3? Camera position is a Vec3 for future 3D support.
        // pos.x and pos.y come from the entity's 2D Position component.
        let target_pos = glam::Vec3::new(pos.x, pos.y, 0.0);

        // Move camera position toward target position.
        // Vec3::lerp(self, other, t) is a method on glam's Vec3.
        camera.position = camera.position.lerp(target_pos, follow_speed);
    }
}
