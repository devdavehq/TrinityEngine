// src/levels/loading_screen.rs
// ──────────────────────────────────────────────────────────────────────────────
// Loading screen shown during level load/unload transitions.
//
// Provides visual feedback to the player while levels are streaming in or
// out. The loading screen has a fade-in/fade-out alpha transition and a
// progress indicator. Game code calls show() to display it, update_progress()
// as loading advances, and hide() when loading completes. The update() method
// is called each frame to animate the fade transitions.
// ──────────────────────────────────────────────────────────────────────────────

/// Loading screen shown during level load/unload transitions.
pub struct LoadingScreen {
    /// Whether the loading screen is currently visible.
    pub visible: bool,
    /// Loading progress from 0.0 to 1.0.
    pub progress: f32,
    /// Message displayed to the player (e.g., "Loading Dungeon...").
    pub message: String,
    /// Current fade alpha (0.0 = fully transparent, 1.0 = fully opaque).
    pub fade_alpha: f32,
    /// Target alpha for the current fade transition.
    target_alpha: f32,
    /// Speed of fade transitions (alpha units per second).
    fade_speed: f32,
    /// Delay in seconds before fade-out begins after hide() is called.
    fade_out_delay: f32,
    /// Timer counting down before fade-out starts.
    fade_out_timer: f32,
    /// Whether we're waiting to start fading out.
    pending_hide: bool,
}

impl LoadingScreen {
    /// Create a new loading screen (initially hidden).
    pub fn new() -> Self {
        Self {
            visible: false,
            progress: 0.0,
            message: String::new(),
            fade_alpha: 0.0,
            target_alpha: 0.0,
            fade_speed: 2.0,
            fade_out_delay: 0.0,
            fade_out_timer: 0.0,
            pending_hide: false,
        }
    }

    /// Show the loading screen with a message. Begins fading in immediately.
    pub fn show(&mut self, message: &str) {
        self.message = message.to_string();
        self.visible = true;
        self.progress = 0.0;
        self.target_alpha = 1.0;
        self.pending_hide = false;
        self.fade_out_timer = 0.0;
    }

    /// Update loading progress (0.0 to 1.0).
    pub fn update_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 1.0);
    }

    /// Begin hiding the loading screen. The fade-out starts after the delay.
    pub fn hide(&mut self) {
        if self.fade_out_delay > 0.0 {
            self.pending_hide = true;
            self.fade_out_timer = self.fade_out_delay;
        } else {
            self.target_alpha = 0.0;
        }
    }

    /// Set the delay (in seconds) before fade-out begins after hide() is called.
    pub fn set_fade_out_delay(&mut self, delay: f32) {
        self.fade_out_delay = delay;
    }

    /// Set the speed of fade transitions (alpha units per second).
    pub fn set_fade_speed(&mut self, speed: f32) {
        self.fade_speed = speed;
    }

    /// Update the loading screen each frame. `dt` is delta time in seconds.
    pub fn update(&mut self, dt: f32) {
        // Handle pending fade-out delay.
        if self.pending_hide {
            self.fade_out_timer -= dt;
            if self.fade_out_timer <= 0.0 {
                self.pending_hide = false;
                self.target_alpha = 0.0;
            }
        }

        // Animate fade_alpha toward target_alpha.
        if self.fade_alpha < self.target_alpha {
            self.fade_alpha = (self.fade_alpha + self.fade_speed * dt).min(self.target_alpha);
        } else if self.fade_alpha > self.target_alpha {
            self.fade_alpha = (self.fade_alpha - self.fade_speed * dt).max(self.target_alpha);
        }

        // Mark as not visible once fully faded out.
        if self.fade_alpha <= 0.0 && self.target_alpha <= 0.0 {
            self.visible = false;
        }
    }
}

impl Default for LoadingScreen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_show_hide_cycle() {
        let mut ls = LoadingScreen::new();
        assert!(!ls.visible);
        assert_eq!(ls.fade_alpha, 0.0);

        ls.show("Loading...");
        assert!(ls.visible);
        assert_eq!(ls.message, "Loading...");
        assert_eq!(ls.progress, 0.0);

        // Simulate fade in.
        for _ in 0..25 {
            ls.update(0.02); // 25 * 0.02 = 0.5s at speed 2.0 -> alpha = 1.0
        }
        assert!((ls.fade_alpha - 1.0).abs() < 0.01);

        ls.update_progress(0.5);
        assert_eq!(ls.progress, 0.5);

        ls.hide();
        // Still visible until alpha reaches 0.
        assert!(ls.visible);

        for _ in 0..25 {
            ls.update(0.02);
        }
        assert!((ls.fade_alpha).abs() < 0.01);
        assert!(!ls.visible);
    }

    #[test]
    fn test_progress_clamping() {
        let mut ls = LoadingScreen::new();
        ls.update_progress(1.5);
        assert_eq!(ls.progress, 1.0);
        ls.update_progress(-0.5);
        assert_eq!(ls.progress, 0.0);
    }

    #[test]
    fn test_fade_out_delay() {
        let mut ls = LoadingScreen::new();
        ls.set_fade_out_delay(0.1);
        ls.show("Loading...");

        // Fade in fully.
        for _ in 0..50 {
            ls.update(0.02);
        }
        assert!((ls.fade_alpha - 1.0).abs() < 0.01);

        ls.hide();
        // Still fully opaque during delay.
        ls.update(0.02);
        assert!((ls.fade_alpha - 1.0).abs() < 0.01);

        // After delay, alpha starts dropping.
        for _ in 0..25 {
            ls.update(0.02);
        }
        assert!(ls.fade_alpha < 0.99);
    }
}
