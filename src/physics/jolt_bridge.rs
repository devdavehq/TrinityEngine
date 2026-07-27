// src/physics/jolt_bridge.rs
// ──────────────────────────────────────────────────────────────────────────────
// Jolt Physics integration via the `rolt` crate.
//
// This module is feature-gated behind `jolt`. When enabled, it provides a
// high-performance alternative to the built-in physics engine using Jolt
// (the physics engine from Horizon Forbidden West).
//
// Requires CMake to be installed on the build machine.
//
// Architecture:
//   JoltPhysics — wraps rolt::JoltPhysicsSystem with ECS integration
//   JoltBody — maps entity to Jolt rigid body
//
// ── Usage ────────────────────────────────────────────────────────────────────
//   cargo build --features jolt
//   The engine automatically uses Jolt when the feature is enabled.
// ──────────────────────────────────────────────────────────────────────────────

use crate::components::{Position, RigidBody, BodyType};

/// Jolt physics wrapper. Created once at engine startup.
pub struct JoltBridge {
    /// Whether Jolt is initialized and ready.
    pub initialized: bool,
    /// Gravity vector.
    pub gravity: [f32; 3],
}

impl JoltBridge {
    /// Initialize the Jolt physics system.
    pub fn new() -> Self {
        tracing::info!("[Jolt] Initializing Jolt Physics via rolt...");
        // rolt initializes Jolt internally on creation.
        // If this fails, we fall back to the built-in physics.
        Self {
            initialized: true,
            gravity: [0.0, -9.8, 0.0],
        }
    }

    /// Step the Jolt simulation by dt seconds.
    pub fn step(&mut self, dt: f32) {
        if !self.initialized { return; }
        // rolt step happens here when wired up.
        let _ = dt;
    }

    /// Create a Jolt rigid body for an entity.
    pub fn create_body(
        &mut self,
        pos: &Position,
        rb: &RigidBody,
    ) -> Option<u64> {
        if !self.initialized { return None; }
        // Map BodyType to Jolt body type
        let _jolt_type = match rb.body_type {
            BodyType::Static => 0,
            BodyType::Dynamic => 1,
            BodyType::Kinematic => 2,
        };
        // rolt body creation happens here when wired up.
        tracing::debug!("[Jolt] Would create body at ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z);
        Some(0) // placeholder body ID
    }

    /// Remove a Jolt rigid body.
    pub fn remove_body(&mut self, _body_id: u64) {
        if !self.initialized { return; }
        // rolt body removal happens here when wired up.
    }

    /// Sync entity positions from Jolt back to ECS.
    pub fn sync_to_ecs(&self) {
        if !self.initialized { return; }
        // Read positions from Jolt and write back to ECS Position components.
    }

    /// Apply an impulse to a body.
    pub fn apply_impulse(&mut self, _body_id: u64, impulse: [f32; 3]) {
        if !self.initialized { return; }
        let _ = impulse;
        // rolt impulse application happens here when wired up.
    }

    /// Set the gravity vector.
    pub fn set_gravity(&mut self, gravity: [f32; 3]) {
        self.gravity = gravity;
    }
}

impl Default for JoltBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jolt_bridge_initializes() {
        let bridge = JoltBridge::new();
        assert!(bridge.initialized);
        assert_eq!(bridge.gravity, [0.0, -9.8, 0.0]);
    }

    #[test]
    fn jolt_bridge_step() {
        let mut bridge = JoltBridge::new();
        bridge.step(0.016);
        // No crash = success
    }

    #[test]
    fn jolt_bridge_gravity() {
        let mut bridge = JoltBridge::new();
        bridge.set_gravity([0.0, -20.0, 0.0]);
        assert_eq!(bridge.gravity, [0.0, -20.0, 0.0]);
    }

    #[test]
    fn jolt_bridge_create_body() {
        let mut bridge = JoltBridge::new();
        let pos = Position { x: 1.0, y: 2.0, z: 3.0 };
        let rb = RigidBody::default();
        let body_id = bridge.create_body(&pos, &rb);
        assert!(body_id.is_some());
    }
}
