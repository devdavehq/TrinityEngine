// src/environment/splash.rs
// ── Splash Visual System ─────────────────────────────────────────────────
// Consumes WaterSplashEvent and produces visual splash effects.

/// Active splash visual — tracks an in-progress splash animation.
pub struct ActiveSplash {
    pub position: [f32; 3],
    pub start_time: f32,
    pub duration: f32,
    pub intensity: f32,
    pub scale: f32,
}

/// Splash with normalized progress (0.0 → 1.0) — returned by `update()`.
pub struct ActiveSplashProgress {
    pub position: [f32; 3],
    pub start_time: f32,
    pub duration: f32,
    pub intensity: f32,
    pub scale: f32,
    pub progress: f32,
}

/// Manages splash visual effects triggered by WaterSplashEvent.
pub struct SplashManager {
    /// Currently active splash visuals.
    pub active_splashes: Vec<ActiveSplash>,
    /// Maximum total active splashes (prevents performance issues).
    pub max_total: usize,
}

impl SplashManager {
    pub fn new() -> Self {
        Self {
            active_splashes: Vec::new(),
            max_total: 32,
        }
    }

    /// Called when a WaterSplashEvent occurs. Creates a new splash visual.
    pub fn on_splash(
        &mut self,
        position: [f32; 3],
        impact_velocity: f32,
        splash_intensity: f32,
        current_time: f32,
    ) {
        if self.active_splashes.len() >= self.max_total {
            self.active_splashes.remove(0);
        }

        self.active_splashes.push(ActiveSplash {
            position,
            start_time: current_time,
            duration: 0.5 + impact_velocity * 0.1,
            intensity: splash_intensity * (0.5 + impact_velocity * 0.1).min(1.0),
            scale: 0.3 + impact_velocity * 0.2,
        });
    }

    /// Update all active splashes. Remove expired ones.
    /// Returns the list of active splashes with their normalized progress (0.0 to 1.0).
    pub fn update(&mut self, current_time: f32) -> Vec<ActiveSplashProgress> {
        let mut results = Vec::new();
        let mut expired = Vec::new();

        for (i, splash) in self.active_splashes.iter().enumerate() {
            let elapsed = current_time - splash.start_time;
            if elapsed < splash.duration {
                let progress = elapsed / splash.duration;
                results.push(ActiveSplashProgress {
                    position: splash.position,
                    start_time: splash.start_time,
                    duration: splash.duration,
                    intensity: splash.intensity,
                    scale: splash.scale,
                    progress,
                });
            } else {
                expired.push(i);
            }
        }

        for i in expired.into_iter().rev() {
            self.active_splashes.remove(i);
        }

        results
    }

    /// Get the number of active splashes.
    pub fn active_count(&self) -> usize {
        self.active_splashes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splash_manager_limits_total() {
        let mut mgr = SplashManager::new();
        mgr.max_total = 3;
        for i in 0..5 {
            mgr.on_splash([0.0, 0.0, i as f32], 2.0, 1.0, 0.0);
        }
        assert_eq!(mgr.active_splashes.len(), 3);
    }

    #[test]
    fn splash_expires() {
        let mut mgr = SplashManager::new();
        mgr.on_splash([0.0, 0.0, 0.0], 2.0, 1.0, 0.0);
        assert_eq!(mgr.active_count(), 1);
        let results = mgr.update(10.0);
        assert!(results.is_empty());
        assert_eq!(mgr.active_count(), 0);
    }
}
