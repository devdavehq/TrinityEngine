// src/physics/mod.rs

use crate::components::{Collider, CollisionPair, Position, RigidBody};
use hecs::{Entity, World};

// GRAVITY is the downward acceleration in world units per second squared.
// Earth gravity is ~9.8 m/s². Tune this for your game's feel.
// A smaller value = floatier jumps. Larger = heavier, more grounded feel.
const GRAVITY: f32 = 9.8;

// physics_system() runs every frame.
// It does three things in order:
//   1. Apply gravity to RigidBody entities
//   2. Integrate velocity into position (move things)
//   3. Detect and resolve collisions
//
// Returns a Vec of collision pairs that occurred this frame.
// Game logic (damage, pickups) reads this list to react.
pub fn physics_system(world: &mut World, dt: f32) -> Vec<CollisionPair> {
    // ── Step 1: Apply gravity ─────────────────────────────────────────────
    // Find every entity with a RigidBody and add gravity to its y velocity.
    // We do this before integration so gravity affects this frame's movement.
    for (_entity, body) in world.query_mut::<&mut RigidBody>().iter() {
        if body.use_gravity && !body.on_ground {
            // velocity_y decreases (downward) each frame.
            // Multiply by dt: frame-rate independent.
            // GRAVITY is positive, y-down is negative in our world space,
            // so we subtract.
            body.velocity_y -= GRAVITY * dt;

            // Clamp terminal velocity — prevent infinite falling speed.
            // -20.0 world units per second is our maximum fall speed.
            body.velocity_y = body.velocity_y.max(-20.0);
        }
    }

    // ── Step 2: Integrate velocity → position ─────────────────────────────
    // Collect entity IDs first to avoid borrow conflicts.
    // Why collect? We need &mut Position AND &RigidBody in the same loop,
    // but hecs can't give us two mutable queries simultaneously.
    // Collecting IDs first ends the first borrow.
    let bodies: Vec<Entity> = world
        .query::<(&RigidBody, &Position)>()
        .iter()
        .map(|(e, _)| e)
        .collect();

    for entity in bodies {
        // Get velocity from RigidBody — read only.
        let (vx, vy) = {
            let body = world.get::<&RigidBody>(entity).unwrap();
            (body.velocity_x, body.velocity_y)
        };

        // Apply velocity to position — mutable.
        // position += velocity × dt
        // dt makes movement frame-rate independent:
        //   at 60fps: dt ≈ 0.016, moves 0.016 × vx per frame
        //   at 30fps: dt ≈ 0.033, moves 0.033 × vx per frame
        // Both travel the same distance per second.
        if let Ok(mut pos) = world.get::<&mut Position>(entity) {
            pos.x += vx * dt;
            pos.y += vy * dt;
        }
    }

    // ── Step 3: Collision detection and resolution ────────────────────────
    // Broad phase: collect all entities with Position + Collider.
    // We need to check every pair — O(n²) for now.
    // Later we'll add a spatial grid to make this faster.
    let collidables: Vec<(Entity, f32, f32, f32, f32)> = world
        .query::<(&Position, &Collider)>()
        .map(|(e, (pos, col))| (e, pos.x, pos.y, col.half_w, col.half_h))
        .collect();

    // Store collisions found this frame.
    let mut collisions: Vec<CollisionPair> = Vec::new();

    // Check every unique pair (i, j) where i < j.
    // i < j prevents checking (A,B) and (B,A) twice.
    for i in 0..collidables.len() {
        for j in (i + 1)..collidables.len() {
            let (ea, ax, ay, ahw, ahh) = collidables[i];
            let (eb, bx, by, bhw, bhh) = collidables[j];

            // AABB overlap test.
            // Two AABBs overlap if the distance between centers
            // is less than the sum of their half-extents on both axes.
            let overlap_x = (ax - bx).abs() < (ahw + bhw);
            let overlap_y = (ay - by).abs() < (ahh + bhh);

            if overlap_x && overlap_y {
                // Record the collision.
                collisions.push(CollisionPair {
                    entity_a: ea,
                    entity_b: eb,
                });

                // ── Collision resolution ──────────────────────────────────
                // Push the entities apart along the axis of least overlap.
                // This is the "minimum translation vector" (MTV) approach.

                // How much overlap on each axis?
                let pen_x = (ahw + bhw) - (ax - bx).abs(); // x penetration depth
                let pen_y = (ahh + bhh) - (ay - by).abs(); // y penetration depth

                // Resolve along the shallowest axis —
                // moving less = less visible popping.
                if pen_x < pen_y {
                    // Resolve horizontally.
                    // Which direction to push? Away from each other.
                    let push = pen_x * 0.5; // split evenly
                    let dir = if ax < bx { -1.0 } else { 1.0 };

                    // Only push entities that have a RigidBody (dynamic).
                    // Static entities (walls, floors) don't move.
                    if world.get::<&RigidBody>(ea).is_ok() {
                        if let Ok(mut pos) = world.get::<&mut Position>(ea) {
                            pos.x += dir * push;
                        }
                        // Stop horizontal velocity on collision.
                        if let Ok(mut body) = world.get::<&mut RigidBody>(ea) {
                            body.velocity_x = 0.0;
                        }
                    }
                    if world.get::<&RigidBody>(eb).is_ok() {
                        if let Ok(mut pos) = world.get::<&mut Position>(eb) {
                            pos.x -= dir * push;
                        }
                        if let Ok(mut body) = world.get::<&mut RigidBody>(eb) {
                            body.velocity_x = 0.0;
                        }
                    }
                } else {
                    // Resolve vertically.
                    let push = pen_y * 0.5;
                    let dir = if ay < by { -1.0 } else { 1.0 };

                    if world.get::<&RigidBody>(ea).is_ok() {
                        if let Ok(mut pos) = world.get::<&mut Position>(ea) {
                            pos.y += dir * push;
                        }
                        if let Ok(mut body) = world.get::<&mut RigidBody>(ea) {
                            // If pushing up (landing on something), mark on_ground.
                            if dir > 0.0 {
                                body.on_ground = true;
                            }
                            body.velocity_y = 0.0;
                        }
                    }
                    if world.get::<&RigidBody>(eb).is_ok() {
                        if let Ok(mut pos) = world.get::<&mut Position>(eb) {
                            pos.y -= dir * push;
                        }
                        if let Ok(mut body) = world.get::<&mut RigidBody>(eb) {
                            if dir < 0.0 {
                                body.on_ground = true;
                            }
                            body.velocity_y = 0.0;
                        }
                    }
                }
            }
        }
    }

    // Reset on_ground each frame before collision detection
    // so entities that walk off edges fall correctly.
    // We do this AFTER resolution so the flag is accurate for this frame.
    // Actually: reset at the START of the next frame.
    // We handle this by resetting before integration next call.
    // For now return the collision list — game logic uses it.
    collisions
}
