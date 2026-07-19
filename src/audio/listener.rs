// src/audio/listener.rs
// 3D audio listener — represents the "ears" of the player.
//
// ── How 3D audio works ───────────────────────────────────────────────────────
// 1. Each sound source has a world-space position (via entity Position component).
// 2. The listener has a position and orientation (from the Camera).
// 3. For each playing sound, the audio system computes:
//    - Distance from listener to source
//    - Direction from listener to source (for stereo panning)
//    - Doppler shift (optional, for fast-moving objects)
// 4. Volume attenuation is applied based on distance (inverse square law or
//    linear falloff, configurable per source).
// 5. Stereo panning is applied based on the horizontal angle.
//
// ── Data flow ────────────────────────────────────────────────────────────────
// Camera position/orientation → AudioListener → AudioSystem.update_3d()
// Entity position + SoundComponent → AudioSource → AudioSystem.update_3d()

/// Represents the listener's position and orientation in world space.
/// Typically attached to the camera.
#[derive(Clone, Debug)]
pub struct AudioListener {
    /// World-space position of the listener.
    pub position: [f32; 3],
    /// Forward direction (normalized).
    pub forward: [f32; 3],
    /// Up direction (normalized).
    pub up: [f32; 3],
}

impl Default for AudioListener {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            forward: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
        }
    }
}

impl AudioListener {
    /// Create from a camera's position and view direction.
    pub fn from_camera(position: glam::Vec3, forward: glam::Vec3, up: glam::Vec3) -> Self {
        Self {
            position: position.to_array(),
            forward: forward.normalize().to_array(),
            up: up.normalize().to_array(),
        }
    }

    /// Compute the distance between the listener and a source position.
    pub fn distance_to(&self, source_pos: [f32; 3]) -> f32 {
        let dx = source_pos[0] - self.position[0];
        let dy = source_pos[1] - self.position[1];
        let dz = source_pos[2] - self.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Compute stereo pan from a source position.
    /// Returns -1.0 (full left) to +1.0 (full right).
    pub fn stereo_pan(&self, source_pos: [f32; 3]) -> f32 {
        // Vector from listener to source.
        let dx = source_pos[0] - self.position[0];
        let dz = source_pos[2] - self.position[2];

        // Listener's right vector = forward × up.
        let fx = self.forward[0];
        let fz = self.forward[2];
        let ux = self.up[0];
        let uy = self.up[1];
        let uz = self.up[2];
        let rx = fz * uy - self.forward[1] * uz; // cross(forward, up).x — simplified
        let rz = fx * uy - self.forward[1] * ux; // cross(forward, up).z — simplified

        // Project source direction onto listener's right axis.
        let right_dot = dx * rx + dz * rz;
        let len = (rx * rx + rz * rz).sqrt().max(0.001);
        (right_dot / len).clamp(-1.0, 1.0)
    }

    /// Compute volume attenuation based on distance.
    /// Uses inverse-square law with a minimum distance clamp.
    pub fn distance_attenuation(distance: f32, min_distance: f32, max_distance: f32) -> f32 {
        if distance <= min_distance {
            return 1.0;
        }
        if distance >= max_distance {
            return 0.0;
        }
        // Smooth rolloff between min and max distance.
        let t = (distance - min_distance) / (max_distance - min_distance);
        1.0 - t * t // Quadratic falloff.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_calculation() {
        let listener = AudioListener {
            position: [0.0, 0.0, 0.0],
            forward: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
        };
        let dist = listener.distance_to([3.0, 4.0, 0.0]);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn attenuation_at_min() {
        let att = AudioListener::distance_attenuation(1.0, 1.0, 50.0);
        assert!((att - 1.0).abs() < 0.001);
    }

    #[test]
    fn attenuation_at_max() {
        let att = AudioListener::distance_attenuation(50.0, 1.0, 50.0);
        assert!((att - 0.0).abs() < 0.001);
    }

    #[test]
    fn attenuation_midpoint() {
        let att = AudioListener::distance_attenuation(25.5, 1.0, 50.0);
        assert!(att > 0.0 && att < 1.0);
    }

    #[test]
    fn stereo_pan_center() {
        let listener = AudioListener::default();
        // Source directly in front should have ~0 pan.
        let pan = listener.stereo_pan([0.0, 0.0, -5.0]);
        assert!(pan.abs() < 0.1);
    }
}
