// src/particles.rs
// GPU-accelerated particle system for weather effects (rain, snow, fog).
//
// Architecture:
//   ParticleSystem → manages emitters + global wind
//   ParticleEmitter → spawns and updates individual particles (CPU-side)
//   GpuParticle → per-instance data uploaded to GPU each frame
//   ParticleRenderer → instanced quad rendering (see renderer/particle.rs)
//
// WHY CPU UPDATE + GPU RENDER?
//   - CPU update: simple, debuggable, correct physics, no compute shader needed
//   - GPU render: thousands of particles drawn in one instanced draw call
//   - Future: move update to compute shader if CPU becomes a bottleneck (>50K particles)
//
// DATA FLOW (per frame):
//   1. ParticleSystem::update(dt, wind_dir, wind_strength, camera_pos)
//      → Each emitter spawns/kills/physics-updates particles
//   2. ParticleSystem::gpu_instances() → Vec<GpuParticle>
//      → CPU collects live particles into a flat array
//   3. Renderer uploads Vec<GpuParticle> to instance buffer
//   4. Renderer draws particle_quad mesh N instances

use glam::{Vec3, Vec4};
use std::collections::HashMap;

// ── Single particle (CPU-side) ───────────────────────────────────────────────
// Each emitter maintains a pool of these. Not sent to GPU directly.
#[derive(Clone, Debug)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub age: f32,
    pub lifetime: f32,
    pub size: f32,
    /// RGBA colour + alpha. Alpha fades as the particle ages.
    pub color: Vec4,
}

// ── GPU instance data (one per live particle) ────────────────────────────────
// repr(C) + Pod + Zeroable: uploaded to GPU via bytemuck.
// Matches the InstanceIn struct in particle.wgsl.
// 48 bytes per particle: 12 (pos) + 4 (size) + 16 (color) + 16 (velocity for streaking).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 3], // 12 bytes — world position
    pub size: f32,          //  4 bytes — point/quad size in world units
    pub color: [f32; 4],    // 16 bytes — RGBA (alpha = fade)
    pub velocity: [f32; 3], // 12 bytes — for motion-blur streaking in shader
    pub _pad: f32,          //  4 bytes — pad to 16-byte alignment
}

impl GpuParticle {
    pub const STRIDE: usize = 48; // bytes
}

// ── Particle Emitter ─────────────────────────────────────────────────────────
// Generic emitter that spawns particles at a rate, with configurable physics.
// Specific weather types (rain, snow) set different spawn parameters.
#[derive(Clone, Debug)]
pub struct ParticleEmitter {
    /// Spawn area center (world space). Rain emitters follow the camera.
    pub position: Vec3,
    /// Half-extents of the spawn box. Particles spawn randomly within this box.
    /// For rain: large XZ (covers the screen), small Y (just above camera).
    pub spawn_extents: Vec3,
    /// Base velocity given to new particles (before randomness).
    pub initial_velocity: Vec3,
    /// Random velocity offset added to each particle. Controls spread.
    pub velocity_spread: Vec3,
    /// Particles per second.
    pub spawn_rate: f32,
    /// Lifetime range (min, max) in seconds.
    pub lifetime_min: f32,
    pub lifetime_max: f32,
    /// Size range (min, max) in world units.
    pub size_min: f32,
    pub size_max: f32,
    /// Base colour (alpha channel used as initial opacity).
    pub color: Vec4,
    /// Acceleration applied each frame (gravity, wind, etc).
    pub acceleration: Vec3,
    /// Maximum live particles. Prevents unbounded memory growth.
    pub max_particles: usize,
    /// Whether this emitter is currently active.
    pub active: bool,
    /// Internal particle pool.
    particles: Vec<Particle>,
    /// Fractional particle accumulator for sub-frame spawning.
    spawn_accumulator: f32,
}

impl ParticleEmitter {
    pub fn new(max_particles: usize) -> Self {
        Self {
            position: Vec3::ZERO,
            spawn_extents: Vec3::new(20.0, 1.0, 20.0),
            initial_velocity: Vec3::new(0.0, -8.0, 0.0),
            velocity_spread: Vec3::new(0.5, 0.5, 0.5),
            spawn_rate: 200.0,
            lifetime_min: 0.8,
            lifetime_max: 1.5,
            size_min: 0.02,
            size_max: 0.04,
            color: Vec4::new(0.8, 0.85, 0.95, 0.6),
            acceleration: Vec3::new(0.0, -9.8, 0.0),
            max_particles,
            active: true,
            particles: Vec::with_capacity(max_particles),
            spawn_accumulator: 0.0,
        }
    }

    /// Spawn a single particle with randomized parameters within configured ranges.
    fn spawn_one(&mut self, rng_seed: &mut f32) {
        if self.particles.len() >= self.max_particles {
            return;
        }
        // Simple hash-based pseudo-random (fast, good enough for particles).
        *rng_seed = (*rng_seed * 16807.0 + 1.0) % 2147483647.0;
        let r1 = (*rng_seed / 2147483647.0) * 2.0 - 1.0; // [-1, 1]
        *rng_seed = (*rng_seed * 16807.0 + 1.0) % 2147483647.0;
        let r2 = (*rng_seed / 2147483647.0) * 2.0 - 1.0;
        *rng_seed = (*rng_seed * 16807.0 + 1.0) % 2147483647.0;
        let r3 = (*rng_seed / 2147483647.0) * 2.0 - 1.0;
        *rng_seed = (*rng_seed * 16807.0 + 1.0) % 2147483647.0;
        let r4 = (*rng_seed / 2147483647.0) * 2.0 - 1.0;
        *rng_seed = (*rng_seed * 16807.0 + 1.0) % 2147483647.0;
        let r5 = (*rng_seed / 2147483647.0); // [0, 1]

        let pos = self.position + Vec3::new(
            r1 * self.spawn_extents.x,
            r2 * self.spawn_extents.y,
            r3 * self.spawn_extents.z,
        );
        let vel = self.initial_velocity + Vec3::new(
            r4 * self.velocity_spread.x,
            (r1 * 0.5) * self.velocity_spread.y,
            r2 * self.velocity_spread.z,
        );
        let t = r5;
        let lifetime = self.lifetime_min + (self.lifetime_max - self.lifetime_min) * t;
        let size = self.size_min + (self.size_max - self.size_min) * t;

        self.particles.push(Particle {
            position: pos,
            velocity: vel,
            age: 0.0,
            lifetime,
            size,
            color: self.color,
        });
    }

    /// Advance all particles by dt seconds. Apply physics, kill dead, spawn new.
    /// Turbulence adds noise-based perturbation for realistic wind gusting.
    pub fn update(&mut self, dt: f32, wind_dir: Vec3, wind_strength: f32, rng_seed: &mut f32, time: f32) {
        if !self.active {
            return;
        }

        // ── Spawn ──────────────────────────────────────────────────────────
        let new_count = self.spawn_rate * dt;
        self.spawn_accumulator += new_count;
        while self.spawn_accumulator >= 1.0 {
            self.spawn_one(rng_seed);
            self.spawn_accumulator -= 1.0;
        }

        // ── Physics update with turbulence ─────────────────────────────────
        let wind_force = wind_dir * wind_strength * 2.0;
        for p in &mut self.particles {
            p.age += dt;

            // Turbulence: Perlin-like noise gusting based on particle position + time.
            // Uses sin/cos hash for cheap turbulence without a noise texture.
            let turb_freq = 0.8;
            let turb_strength = wind_strength * 0.15;
            let turb = Vec3::new(
                (p.position.x * turb_freq + time * 1.3).sin() * (p.position.z * turb_freq * 0.7 + time * 0.9).cos(),
                (p.position.y * turb_freq * 0.5 + time * 0.7).sin() * 0.3,
                (p.position.z * turb_freq + time * 1.1).cos() * (p.position.x * turb_freq * 0.6 + time * 1.2).sin(),
            ) * turb_strength;

            // Apply acceleration (gravity + wind + turbulence).
            p.velocity += (self.acceleration + wind_force + turb) * dt;
            p.position += p.velocity * dt;
        }

        // ── Kill dead particles ────────────────────────────────────────────
        self.particles.retain(|p| p.age < p.lifetime);
    }

    /// Collect live particles into GPU-ready instance data.
    /// Called each frame before the particle draw call.
    pub fn gpu_instances(&self) -> Vec<GpuParticle> {
        self.particles.iter().map(|p| {
            // Fade alpha as the particle ages.
            let life_ratio = (p.age / p.lifetime).clamp(0.0, 1.0);
            let fade = 1.0 - life_ratio; // linear fade
            // Size grows slightly as raindrop falls (perspective illusion).
            let size = p.size * (1.0 + life_ratio * 0.3);
            let mut color = p.color;
            color.w *= fade;

            GpuParticle {
                position: p.position.to_array(),
                size,
                color: color.to_array(),
                velocity: p.velocity.to_array(),
                _pad: 0.0,
            }
        }).collect()
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Move the emitter's spawn area (e.g. follow camera).
    pub fn set_position(&mut self, pos: Vec3) {
        self.position = pos;
    }
}

// ── Particle System ──────────────────────────────────────────────────────────
// Manages multiple emitters and global particle settings.
// Called once per frame from the main loop.
pub struct ParticleSystem {
    /// Named emitters. Weather system creates/removes these dynamically.
    pub emitters: Vec<ParticleEmitter>,
    /// Global wind applied to all particles (from WeatherState).
    pub wind_dir: Vec3,
    pub wind_strength: f32,
    /// RNG seed — simple state for pseudo-random spawning.
    rng_seed: f32,
    /// Per-entity fire emitters: entity_bits -> [fire_idx, smoke_idx, ember_idx].
    fire_sources: HashMap<u64, [usize; 3]>,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            emitters: Vec::new(),
            wind_dir: Vec3::new(1.0, 0.0, 0.3),
            wind_strength: 0.1,
            rng_seed: 42.0,
            fire_sources: HashMap::new(),
        }
    }

    /// Create and return the index of a new emitter.
    pub fn add_emitter(&mut self, emitter: ParticleEmitter) -> usize {
        self.emitters.push(emitter);
        self.emitters.len() - 1
    }

    /// Update all emitters. Call once per frame.
    pub fn update(&mut self, dt: f32, camera_pos: Vec3, time: f32) {
        for emitter in &mut self.emitters {
            // Center spawn area on camera XZ, offset Y upward.
            let spawn_center = Vec3::new(
                camera_pos.x,
                camera_pos.y + emitter.spawn_extents.y + 5.0,
                camera_pos.z,
            );
            emitter.set_position(spawn_center);
            emitter.update(dt, self.wind_dir, self.wind_strength, &mut self.rng_seed, time);
        }
    }

    /// Collect all GPU particle instances from all emitters.
    pub fn gpu_instances(&self) -> Vec<GpuParticle> {
        let mut all = Vec::new();
        for emitter in &self.emitters {
            let instances = emitter.gpu_instances();
            all.extend_from_slice(&instances);
        }
        all
    }

    /// Total live particle count across all emitters.
    pub fn total_particles(&self) -> usize {
        self.emitters.iter().map(|e| e.particle_count()).sum()
    }

    /// Update wind parameters from WeatherState.
    pub fn set_wind(&mut self, dir: Vec3, strength: f32) {
        self.wind_dir = dir;
        self.wind_strength = strength;
    }
}

// ── Factory: create weather-appropriate emitters ─────────────────────────────
impl ParticleSystem {
    /// Create a rain emitter (heavy drops, fast, blue-white).
    pub fn create_rain_emitter() -> ParticleEmitter {
        let mut e = ParticleEmitter::new(8000);
        e.spawn_rate = 400.0;
        e.initial_velocity = Vec3::new(0.0, -12.0, 0.0);
        e.velocity_spread = Vec3::new(1.0, 1.0, 1.0);
        e.spawn_extents = Vec3::new(25.0, 1.0, 25.0);
        e.lifetime_min = 0.5;
        e.lifetime_max = 1.2;
        e.size_min = 0.015;
        e.size_max = 0.035;
        e.color = Vec4::new(0.75, 0.82, 0.95, 0.5);
        e.acceleration = Vec3::new(0.0, -4.0, 0.0); // slight extra downward pull
        e.active = false; // activated when weather is rainy
        e
    }

    /// Create a snow emitter (gentle flakes, slow, white).
    pub fn create_snow_emitter() -> ParticleEmitter {
        let mut e = ParticleEmitter::new(5000);
        e.spawn_rate = 150.0;
        e.initial_velocity = Vec3::new(0.0, -1.5, 0.0);
        e.velocity_spread = Vec3::new(1.5, 0.5, 1.5);
        e.spawn_extents = Vec3::new(30.0, 1.0, 30.0);
        e.lifetime_min = 3.0;
        e.lifetime_max = 6.0;
        e.size_min = 0.008;
        e.size_max = 0.025;
        e.color = Vec4::new(0.95, 0.95, 1.0, 0.7);
        e.acceleration = Vec3::new(0.0, -0.5, 0.0); // snow floats
        e.active = false; // activated when weather is snowy
        e
    }

    /// Create a mist/fog emitter (low, slow, very transparent).
    pub fn create_mist_emitter() -> ParticleEmitter {
        let mut e = ParticleEmitter::new(500);
        e.spawn_rate = 20.0;
        e.initial_velocity = Vec3::new(0.0, 0.1, 0.0);
        e.velocity_spread = Vec3::new(0.3, 0.2, 0.3);
        e.spawn_extents = Vec3::new(20.0, 2.0, 20.0);
        e.lifetime_min = 4.0;
        e.lifetime_max = 8.0;
        e.size_min = 2.0;
        e.size_max = 5.0;
        e.color = Vec4::new(0.85, 0.85, 0.88, 0.08);
        e.acceleration = Vec3::ZERO;
        e.active = false;
        e
    }

    /// Create a splatter emitter for rain impact particles (tiny, short-lived).
    pub fn create_rain_splatter_emitter() -> ParticleEmitter {
        let mut e = ParticleEmitter::new(2000);
        e.spawn_rate = 0.0; // driven externally
        e.initial_velocity = Vec3::new(0.0, 2.0, 0.0);
        e.velocity_spread = Vec3::new(3.0, 2.0, 3.0);
        e.spawn_extents = Vec3::new(0.1, 0.1, 0.1);
        e.lifetime_min = 0.1;
        e.lifetime_max = 0.3;
        e.size_min = 0.005;
        e.size_max = 0.01;
        e.color = Vec4::new(0.8, 0.85, 0.95, 0.3);
        e.acceleration = Vec3::new(0.0, -15.0, 0.0);
        e.active = false;
        e
    }

    /// Set up the standard weather emitter set (rain + snow + mist).
    /// Returns emitter indices: [rain=0, snow=1, mist=2, splatter=3].
    pub fn setup_weather_emitters() -> (Self, [usize; 4]) {
        let mut system = ParticleSystem::new();
        let rain_idx = system.add_emitter(Self::create_rain_emitter());
        let snow_idx = system.add_emitter(Self::create_snow_emitter());
        let mist_idx = system.add_emitter(Self::create_mist_emitter());
        let splatter_idx = system.add_emitter(Self::create_rain_splatter_emitter());
        (system, [rain_idx, snow_idx, mist_idx, splatter_idx])
    }

    /// Create a fire emitter (bright orange-yellow, upward, flickering).
    pub fn create_fire_emitter() -> ParticleEmitter {
        let mut e = ParticleEmitter::new(1500);
        e.spawn_rate = 120.0;
        e.initial_velocity = Vec3::new(0.0, 4.0, 0.0);
        e.velocity_spread = Vec3::new(1.0, 2.0, 1.0);
        e.spawn_extents = Vec3::new(0.3, 0.1, 0.3);
        e.lifetime_min = 0.3;
        e.lifetime_max = 0.8;
        e.size_min = 0.04;
        e.size_max = 0.12;
        // Fire gradient: white-hot base -> orange -> red tips
        e.color = Vec4::new(1.0, 0.6, 0.1, 0.9);
        e.acceleration = Vec3::new(0.0, -2.0, 0.0); // slows as it rises
        e.active = true;
        e
    }

    /// Create a smoke emitter (dark gray, slow, large, rising).
    pub fn create_smoke_emitter() -> ParticleEmitter {
        let mut e = ParticleEmitter::new(800);
        e.spawn_rate = 30.0;
        e.initial_velocity = Vec3::new(0.0, 2.0, 0.0);
        e.velocity_spread = Vec3::new(0.5, 1.0, 0.5);
        e.spawn_extents = Vec3::new(0.4, 0.1, 0.4);
        e.lifetime_min = 2.0;
        e.lifetime_max = 5.0;
        e.size_min = 0.15;
        e.size_max = 0.5;
        e.color = Vec4::new(0.25, 0.25, 0.28, 0.25);
        e.acceleration = Vec3::new(0.0, 0.5, 0.0); // rises slowly
        e.active = true;
        e
    }

    /// Create an ember particle emitter (tiny bright dots floating upward).
    pub fn create_ember_emitter() -> ParticleEmitter {
        let mut e = ParticleEmitter::new(600);
        e.spawn_rate = 25.0;
        e.initial_velocity = Vec3::new(0.0, 5.0, 0.0);
        e.velocity_spread = Vec3::new(2.0, 3.0, 2.0);
        e.spawn_extents = Vec3::new(0.5, 0.2, 0.5);
        e.lifetime_min = 1.0;
        e.lifetime_max = 3.0;
        e.size_min = 0.003;
        e.size_max = 0.008;
        e.color = Vec4::new(1.0, 0.5, 0.05, 1.0);
        e.acceleration = Vec3::new(0.0, 1.0, 0.0);
        e.active = true;
        e
    }

    /// Enable/disable emitters based on current weather conditions.
    pub fn apply_weather(
        &mut self,
        indices: [usize; 4],
        condition: crate::environment::weather::WeatherCondition,
        intensity: f32,
        wind_dir: Vec3,
        wind_strength: f32,
    ) {
        use crate::environment::weather::WeatherCondition;

        self.set_wind(wind_dir, wind_strength);

        let [rain_idx, snow_idx, mist_idx, splatter_idx] = indices;

        // ── Rain ────────────────────────────────────────────────────────────
        let is_rainy = matches!(condition,
            WeatherCondition::LightRain | WeatherCondition::HeavyRain | WeatherCondition::Storm);
        if let Some(e) = self.emitters.get_mut(rain_idx) {
            e.active = is_rainy;
            if is_rainy {
                // Scale spawn rate and speed by intensity.
                let t = intensity.clamp(0.0, 1.0);
                e.spawn_rate = 150.0 + 500.0 * t;
                e.initial_velocity.y = -8.0 - 8.0 * t;
                e.size_min = 0.01 + 0.01 * t;
                e.size_max = 0.02 + 0.03 * t;
            }
        }
        if let Some(e) = self.emitters.get_mut(splatter_idx) {
            e.active = is_rainy;
        }

        // ── Snow ────────────────────────────────────────────────────────────
        let is_snowy = condition == WeatherCondition::Snow;
        if let Some(e) = self.emitters.get_mut(snow_idx) {
            e.active = is_snowy;
            if is_snowy {
                let t = intensity.clamp(0.0, 1.0);
                e.spawn_rate = 50.0 + 200.0 * t;
                e.size_min = 0.005 + 0.01 * t;
                e.size_max = 0.015 + 0.02 * t;
            }
        }

        // ── Mist ────────────────────────────────────────────────────────────
        let is_foggy = condition == WeatherCondition::Fog
            || condition == WeatherCondition::Overcast;
        if let Some(e) = self.emitters.get_mut(mist_idx) {
            e.active = is_foggy;
            if is_foggy {
                e.spawn_rate = 10.0 + 30.0 * intensity;
            }
        }
    }

    /// Iterator over tracked fire source entity keys.
    pub fn fire_source_keys(&self) -> impl Iterator<Item = &u64> {
        self.fire_sources.keys()
    }

    /// Add fire/smoke/ember emitters for an entity with a FireSource component.
    /// `entity_bits` is the hecs entity's `.to_bits()`, `pos` is world position,
    /// `intensity` scales the emitter rates.
    pub fn add_fire_source(&mut self, entity_bits: u64, pos: Vec3, intensity: f32) {
        if self.fire_sources.contains_key(&entity_bits) {
            // Already tracked — just update position.
            if let Some(&[fi, si, ei]) = self.fire_sources.get(&entity_bits) {
                if let Some(e) = self.emitters.get_mut(fi) {
                    e.position = pos;
                    e.spawn_rate = 120.0 * intensity;
                    e.active = intensity > 0.01;
                }
                if let Some(e) = self.emitters.get_mut(si) {
                    e.position = pos + Vec3::new(0.0, 0.5, 0.0);
                    e.spawn_rate = 30.0 * intensity;
                    e.active = intensity > 0.01;
                }
                if let Some(e) = self.emitters.get_mut(ei) {
                    e.position = pos;
                    e.spawn_rate = 25.0 * intensity;
                    e.active = intensity > 0.01;
                }
            }
            return;
        }
        let mut fire = Self::create_fire_emitter();
        fire.position = pos;
        fire.spawn_rate *= intensity;
        fire.active = intensity > 0.01;
        let fi = self.add_emitter(fire);

        let mut smoke = Self::create_smoke_emitter();
        smoke.position = pos + Vec3::new(0.0, 0.5, 0.0);
        smoke.spawn_rate *= intensity;
        smoke.active = intensity > 0.01;
        let si = self.add_emitter(smoke);

        let mut ember = Self::create_ember_emitter();
        ember.position = pos;
        ember.spawn_rate *= intensity;
        ember.active = intensity > 0.01;
        let ei = self.add_emitter(ember);

        self.fire_sources.insert(entity_bits, [fi, si, ei]);
    }

    /// Remove fire emitters for an entity that no longer has a FireSource.
    pub fn remove_fire_source(&mut self, entity_bits: u64) {
        if let Some([fi, si, ei]) = self.fire_sources.remove(&entity_bits) {
            // Mark emitters inactive (can't easily remove from Vec without shifting indices).
            if let Some(e) = self.emitters.get_mut(fi) { e.active = false; }
            if let Some(e) = self.emitters.get_mut(si) { e.active = false; }
            if let Some(e) = self.emitters.get_mut(ei) { e.active = false; }
        }
    }
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_particle_stride() {
        assert_eq!(GpuParticle::STRIDE, std::mem::size_of::<GpuParticle>());
        // 48 bytes = 3 (pos) + 1 (size) + 4 (color) + 3 (vel) + 1 (pad) = 12 f32s
        assert_eq!(GpuParticle::STRIDE, 48);
    }

    #[test]
    fn emitter_spawns_particles() {
        let mut e = ParticleEmitter::new(100);
        e.spawn_rate = 1000.0; // fast spawn
        e.active = true;
        let mut seed = 42.0;
        e.update(0.1, Vec3::X, 0.0, &mut seed, 0.0); // 0.1s at 1000/s = 100 particles
        assert!(e.particle_count() > 0);
        assert!(e.particle_count() <= 100);
    }

    #[test]
    fn emitter_respects_max() {
        let mut e = ParticleEmitter::new(10);
        e.spawn_rate = 10000.0;
        e.lifetime_min = 10.0; // long enough that none die during the test
        e.lifetime_max = 10.0;
        e.active = true;
        let mut seed = 42.0;
        e.update(1.0, Vec3::X, 0.0, &mut seed, 0.0);
        assert!(e.particle_count() <= 10, "Expected ≤10, got {}", e.particle_count());
    }

    #[test]
    fn particles_die_after_lifetime() {
        let mut e = ParticleEmitter::new(100);
        e.spawn_rate = 1000.0;
        e.lifetime_min = 0.1;
        e.lifetime_max = 0.1;
        e.active = true;
        let mut seed = 42.0;
        e.update(0.01, Vec3::X, 0.0, &mut seed, 0.0); // spawn some
        assert!(e.particle_count() > 0);
        e.update(0.5, Vec3::X, 0.0, &mut seed, 0.5); // wait long enough
        assert_eq!(e.particle_count(), 0);
    }

    #[test]
    fn wind_affects_particles() {
        let mut e = ParticleEmitter::new(100);
        e.spawn_rate = 100.0;
        e.lifetime_min = 2.0;
        e.lifetime_max = 2.0;
        e.active = true;
        let mut seed = 42.0;
        e.update(0.5, Vec3::X, 10.0, &mut seed, 0.0); // strong wind in X
        for p in &e.particles {
            // Wind should push particles in X direction.
            assert!(p.velocity.x > 0.0 || p.position.x > e.position.x);
        }
    }

    #[test]
    fn gpu_instances_match_particle_count() {
        let mut e = ParticleEmitter::new(100);
        e.spawn_rate = 500.0;
        e.active = true;
        let mut seed = 42.0;
        e.update(0.1, Vec3::X, 0.0, &mut seed, 0.0);
        let instances = e.gpu_instances();
        assert_eq!(instances.len(), e.particle_count());
    }

    #[test]
    fn system_total_particles() {
        let (mut system, indices) = ParticleSystem::setup_weather_emitters();
        system.apply_weather(
            indices,
            crate::environment::weather::WeatherCondition::HeavyRain,
            0.8,
            Vec3::X,
            0.5,
        );
        let mut seed = 42.0;
        system.update(0.1, Vec3::ZERO, 0.0);
        // Should have some rain particles.
        assert!(system.total_particles() > 0);
    }

    #[test]
    fn inactive_emitter_spawns_nothing() {
        let mut e = ParticleEmitter::new(100);
        e.spawn_rate = 1000.0;
        e.active = false;
        let mut seed = 42.0;
        e.update(1.0, Vec3::X, 0.0, &mut seed, 0.0);
        assert_eq!(e.particle_count(), 0);
    }

    #[test]
    fn fade_reduces_alpha() {
        let mut e = ParticleEmitter::new(1);
        e.spawn_rate = 1000.0;
        e.lifetime_min = 1.0;
        e.lifetime_max = 1.0;
        e.color = Vec4::new(1.0, 1.0, 1.0, 1.0);
        e.active = true;
        let mut seed = 42.0;
        e.update(0.01, Vec3::ZERO, 0.0, &mut seed, 0.0); // spawn
        e.update(0.9, Vec3::ZERO, 0.0, &mut seed, 0.9); // age to 0.91
        let instances = e.gpu_instances();
        if let Some(inst) = instances.first() {
            // Alpha should be significantly reduced (life_ratio ~0.91 → fade ~0.09).
            assert!(inst.color[3] < 0.2);
        }
    }

    #[test]
    fn weather_apply_toggles_emitters() {
        let (mut system, indices) = ParticleSystem::setup_weather_emitters();
        // Start with rain.
        system.apply_weather(
            indices,
            crate::environment::weather::WeatherCondition::HeavyRain,
            0.8,
            Vec3::X,
            0.5,
        );
        assert!(system.emitters[indices[0]].active); // rain
        assert!(!system.emitters[indices[1]].active); // snow off

        // Switch to snow.
        system.apply_weather(
            indices,
            crate::environment::weather::WeatherCondition::Snow,
            0.5,
            Vec3::X,
            0.3,
        );
        assert!(!system.emitters[indices[0]].active); // rain off
        assert!(system.emitters[indices[1]].active);  // snow on
    }
}
