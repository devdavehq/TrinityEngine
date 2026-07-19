// src/scene/transition.rs
// ──────────────────────────────────────────────────────────────────────────────
// Scene Transition System — fade-to-black visual effect when switching scenes.
//
// WHY:
//   Hard-cutting between scenes is jarring. A smooth fade-to-black (or fade-
//   through-black) gives the player visual feedback that a transition is
//   happening and masks any loading hitch.
//
// STATE MACHINE:
//
//   None ──start_transition()──► FadingOut ──timer expire──► BlackScreen
//                                                         │
//     (renderer loads scene here)                         │
//   None ◄──timer expire── FadingIn ◄─────────────────────┘
//
// INTEGRATION:
//   1. When a scene switch is requested, call start_transition("scene/path.scene").
//   2. Each frame, call update(dt). It returns Some(path) when the screen is
//      fully black — that's when main.rs should actually load the new scene.
//   3. Call fade_opacity() to get the current screen overlay opacity (0..1).
//      Render a black quad at this opacity on top of the frame.
//
// The transition is non-blocking: rendering continues throughout, just with
// a black overlay that fades in and out.
// ──────────────────────────────────────────────────────────────────────────────

/// State of a scene transition effect.
pub struct SceneTransition {
    /// Current transition state (none / fading out / black / fading in).
    pub state: TransitionState,
    /// Duration of the fade-out phase in seconds.
    pub fade_out_duration: f32,
    /// Duration of the fade-in phase in seconds.
    pub fade_in_duration: f32,
    /// Timer tracking progress through the current phase (seconds).
    pub timer: f32,
    /// The scene path to load during the black phase.
    /// Stored here so update() can return it when the screen goes fully black.
    pub pending_scene: Option<String>,
    /// Current screen overlay opacity (0 = transparent, 1 = fully black).
    /// The renderer uses this to draw a fullscreen black quad.
    pub opacity: f32,
}

/// The four states of the transition state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionState {
    /// No transition in progress — scene is running normally.
    None,
    /// Fading to black. The old scene is still rendering underneath.
    FadingOut,
    /// Fully black — the scene swap happens at this moment.
    /// Transitions to FadingIn immediately (same frame).
    BlackScreen,
    /// Fading back in. The new scene is now rendering underneath.
    FadingIn,
}

impl SceneTransition {
    /// Create a new transition controller with default durations.
    pub fn new() -> Self {
        Self {
            state: TransitionState::None,
            fade_out_duration: 0.5,
            fade_in_duration: 0.5,
            timer: 0.0,
            pending_scene: None,
            opacity: 0.0,
        }
    }

    /// Start a transition to a new scene.
    ///
    /// This sets the state to FadingOut and stores the target path.
    /// The actual scene load happens later when update() returns Some(path)
    /// (at the moment the screen is fully black).
    pub fn start_transition(&mut self, target_scene: &str) {
        tracing::info!(
            "[Transition] Starting fade-to-black for '{}'",
            target_scene
        );
        self.state = TransitionState::FadingOut;
        self.timer = 0.0;
        self.pending_scene = Some(target_scene.to_string());
    }

    /// Update the transition each frame.
    ///
    /// Returns `Some(scene_path)` exactly once — when the screen is fully
    /// black and it's time to load the new scene. The caller should:
    ///   1. Clear the world
    ///   2. Load the new scene
    ///   3. Set `pending_scene` to None (happens automatically via `take()`)
    ///
    /// Returns `None` on all other frames.
    pub fn update(&mut self, dt: f32) -> Option<String> {
        match self.state {
            // ── No transition: nothing to do ─────────────────────────────
            TransitionState::None => None,

            // ── Fading out: opacity goes 0 → 1 ──────────────────────────
            TransitionState::FadingOut => {
                self.timer += dt;
                // Linear interpolation from 0 to 1 over fade_out_duration.
                self.opacity = (self.timer / self.fade_out_duration).min(1.0);

                if self.timer >= self.fade_out_duration {
                    // Screen is fully black — time to load the new scene.
                    self.state = TransitionState::BlackScreen;
                    self.timer = 0.0;
                    self.opacity = 1.0; // Ensure exactly 1.0, not 0.999...
                    // Return the pending scene path so main.rs can load it.
                    self.pending_scene.take()
                } else {
                    None
                }
            }

            // ── Black screen: scene load happens here ────────────────────
            // Immediately transition to FadingIn. The actual scene load is
            // triggered by the Some(path) return in FadingOut. By the time
            // we reach BlackScreen, the scene has already been loaded.
            TransitionState::BlackScreen => {
                tracing::info!("[Transition] Black screen reached — fading in");
                self.state = TransitionState::FadingIn;
                self.timer = 0.0;
                None
            }

            // ── Fading in: opacity goes 1 → 0 ───────────────────────────
            TransitionState::FadingIn => {
                self.timer += dt;
                // Linear interpolation from 1 to 0 over fade_in_duration.
                self.opacity = 1.0 - (self.timer / self.fade_in_duration).min(1.0);

                if self.timer >= self.fade_in_duration {
                    // Transition complete — back to normal.
                    self.state = TransitionState::None;
                    self.opacity = 0.0;
                    tracing::info!("[Transition] Fade-in complete");
                }
                None
            }
        }
    }

    /// Get the current fade opacity for rendering (0..1).
    ///
    /// Render a fullscreen quad with `color = vec4(0, 0, 0, opacity)` on top
    /// of the frame to create the fade effect.
    pub fn fade_opacity(&self) -> f32 {
        self.opacity
    }

    /// Is a transition currently active (fading out, black, or fading in)?
    pub fn is_active(&self) -> bool {
        self.state != TransitionState::None
    }

    /// Set custom fade durations (in seconds).
    pub fn set_durations(&mut self, fade_out: f32, fade_in: f32) {
        self.fade_out_duration = fade_out.max(0.01);
        self.fade_in_duration = fade_in.max(0.01);
    }

    /// Cancel an in-progress transition (snaps back to no-transition state).
    /// Useful if the user presses Escape during a transition.
    pub fn cancel(&mut self) {
        self.state = TransitionState::None;
        self.timer = 0.0;
        self.opacity = 0.0;
        self.pending_scene = None;
    }
}
