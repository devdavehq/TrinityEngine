// src/cinematics.rs
// ──────────────────────────────────────────────────────────────────────────────
// Lightweight cutscene / cinematic director.
//
// A cutscene is a sequence of camera "shots".  Each shot has a start position,
// a target (look-at) point, and a duration.  The director advances a timeline
// clock and, for the current shot, interpolates the camera between the previous
// shot's endpoint and this shot's endpoint using smooth (smoothstep) easing.
//
// The director is a data-only, Lua-independent core (unit-testable).  The Lua
// `cinematic.*` bridge in scripting.rs drives it per-frame: it advances time,
// feeds the interpolated (position, target) to the engine camera via the
// existing pending-camera slot, and fires per-shot "start" and a final "end"
// callback at the right timestamps.
//
// Model:
//   start() clears any current cutscene and sets current_time = 0.
//   add_shot(start, target, duration) appends a shot to the timeline.
//   play() begins running; pause()/resume() control stepping.
//   skip() jumps current_time to the total duration (ends the cut).
//   step(dt) advances current_time and reports which shot is now active.
// ──────────────────────────────────────────────────────────────────────────────

/// Scalar easing applied while interpolating inside a shot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ease {
    /// Constant velocity between keyframes.
    Linear,
    /// Smoothstep: ease-in-out, no velocity discontinuity at keyframes.
    SmoothStep,
}

/// One camera shot within a cutscene.
#[derive(Clone, Debug)]
pub struct Shot {
    /// Camera position at the start of this shot.
    pub start: [f32; 3],
    /// Camera look-at / target point.
    pub target: [f32; 3],
    /// Seconds this shot lasts before the next begins.
    pub duration: f32,
    /// Interpolation curve.
    pub ease: Ease,
    /// Optional authoring-friendly name ("wide_shot", "closeup").
    pub name: String,
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// A cutscene timeline assembled from sequential `Shot`s.
#[derive(Clone, Debug)]
pub struct Cutscene {
    shots: Vec<Shot>,
    /// Elapsed play time in seconds.
    current_time: f32,
    /// Whether the timeline is advancing.
    playing: bool,
    /// Whether to loop back to the start on reaching the end.
    pub looping: bool,
    /// Set true once current_time has reached the end.
    finished: bool,
    /// Index of the last shot we reported as active (for edge detection).
    last_active_shot: usize,
    /// Optional authoring name for the whole cutscene.
    name: Option<String>,
    /// Whether this cutscene takes over the engine camera.  Set false for
    /// cutscenes that play while the player keeps gameplay camera control.
    drives_camera: bool,
}

impl Cutscene {
    pub fn new() -> Self {
        Self {
            shots: Vec::new(),
            current_time: 0.0,
            playing: false,
            looping: false,
            finished: false,
            last_active_shot: 0,
            name: None,
            drives_camera: true,
        }
    }

    /// Total timeline length in seconds.
    pub fn duration(&self) -> f32 {
        self.shots.iter().map(|s| s.duration).sum()
    }

    /// Append a shot. Returns its index.
    pub fn add_shot(&mut self, start: [f32; 3], target: [f32; 3], duration: f32, name: Option<&str>) -> usize {
        let idx = self.shots.len();
        self.shots.push(Shot {
            start,
            target,
            duration: duration.max(0.001),
            ease: Ease::SmoothStep,
            name: name.unwrap_or("").to_string(),
        });
        idx
    }

    /// Replace the easing for a shot from a curve name.
    /// Supported: "linear" | "smooth" | "smoothstep" (defaults to smooth).
    pub fn set_ease(&mut self, index: usize, mode: &str) -> bool {
        let ease = match mode {
            "linear" | "Linear" => Ease::Linear,
            _ => Ease::SmoothStep,
        };
        if let Some(s) = self.shots.get_mut(index) {
            s.ease = ease;
            true
        } else {
            false
        }
    }

    /// Reset and optionally rename the cutscene (keeps shots).
    pub fn start(&mut self, name: Option<String>) {
        self.name = name;
        self.reset();
    }

    /// Number of shots in the timeline.
    pub fn len(&self) -> usize {
        self.shots.len()
    }

    /// Whether this cutscene owns the engine camera while playing.
    pub fn drives_camera(&self) -> bool {
        self.drives_camera
    }

    /// Turn camera ownership for this cutscene on/off.
    pub fn set_drives_camera(&mut self, drives: bool) {
        self.drives_camera = drives;
    }

    /// Whether the timeline has no shots.
    pub fn is_empty(&self) -> bool {
        self.shots.is_empty()
    }

    /// Reset the timeline (keep shots, clear time and play state).
    pub fn reset(&mut self) {
        self.current_time = 0.0;
        self.playing = false;
        self.finished = false;
        self.last_active_shot = 0;
    }

    /// Clear all shots (start of a brand-new authoring session).
    pub fn clear(&mut self) {
        self.shots.clear();
        self.reset();
    }

    pub fn play(&mut self) {
        if self.finished && !self.looping {
            self.reset();
        }
        self.playing = true;
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn resume(&mut self) {
        self.playing = true;
    }

    pub fn is_playing(&self) -> bool {
        self.playing && !self.finished
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Alias used by the Lua bridge (`cinematic.time`).
    pub fn time(&self) -> f32 {
        self.current_time
    }

    /// Hard-stop playback and rewind to the start (not playing).
    pub fn stop(&mut self) {
        self.reset();
    }

    pub fn shot_count(&self) -> usize {
        self.shots.len()
    }

    /// Jump straight to the end of the timeline.
    pub fn skip(&mut self) {
        let d = self.duration();
        self.current_time = d;
        self.finished = true;
    }

    /// Advance time by `dt`. Returns the index of the now-active shot.
    pub fn step(&mut self, dt: f32) -> usize {
        if !self.playing || self.finished {
            return self.shot_index_at(self.current_time);
        }
        let d = self.duration();
        if d <= 0.0 {
            self.finished = true;
            return 0;
        }
        self.current_time = (self.current_time + dt).min(d);
        if self.current_time >= d {
            if self.looping {
                self.current_time = 0.0;
            } else {
                self.finished = true;
            }
        }
        self.last_active_shot = self.shot_index_at(self.current_time);
        self.last_active_shot
    }

    /// Index of the shot occupied by time `t`.
    fn shot_index_at(&self, t: f32) -> usize {
        let mut acc = 0.0;
        for (i, s) in self.shots.iter().enumerate() {
            if t < acc + s.duration {
                return i;
            }
            acc += s.duration;
        }
        self.shots.len().saturating_sub(1)
    }

    /// The shot currently occupied by the timeline pointer.
    pub fn current_shot_index(&self) -> usize {
        self.last_active_shot
    }

    /// Camera (position, target) interpolated at the current play time.
    /// Returns None when there are no shots.
    pub fn current_camera(&self) -> Option<([f32; 3], [f32; 3])> {
        if self.shots.is_empty() {
            return None;
        }
        // Peaved time within the active shot.
        let t = self.current_time;
        let idx = self.shot_index_at(t);
        let shot = &self.shots[idx];

        // Local time inside this shot.
        let start_off = self.shots[..idx].iter().map(|s| s.duration).sum::<f32>();
        let local = (t - start_off).clamp(0.0, shot.duration);

        // Previous shot's *end* position acts as this shot's start keyframe, so
        // the camera glides continuously across shot boundaries.
        let prev_start = if idx > 0 { self.shots[idx - 1].start } else { shot.start };
        let u = if shot.duration > 0.001 {
            local / shot.duration
        } else {
            1.0
        };
        let eased = match shot.ease {
            Ease::Linear => u,
            Ease::SmoothStep => smoothstep(u),
        };

        // Position glides from previous shot's start to this shot's start;
        // the target is eased between the two shots' look-at points.
        let pos = lerp3(prev_start, shot.start, eased);
        let tgt = {
            let prev_target = if idx > 0 { self.shots[idx - 1].target } else { shot.target };
            lerp3(prev_target, shot.target, eased)
        };
        Some((pos, tgt))
    }

    pub fn shot(&self, index: usize) -> Option<&Shot> {
        self.shots.get(index)
    }
}

impl Default for Cutscene {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let c = Cutscene::new();
        assert_eq!(c.duration(), 0.0);
        assert_eq!(c.shot_count(), 0);
        assert!(c.current_camera().is_none());
    }

    #[test]
    fn duration_sums_shots() {
        let mut c = Cutscene::new();
        c.add_shot([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, Some("a"));
        c.add_shot([1.0, 0.0, 0.0], [1.0, 0.0, -1.0], 2.0, Some("b"));
        assert_eq!(c.duration(), 3.0);
    }

    #[test]
    fn step_advances_time_and_reports_shot() {
        let mut c = Cutscene::new();
        c.add_shot([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, Some("a"));
        c.add_shot([1.0, 0.0, 0.0], [1.0, 0.0, -1.0], 2.0, Some("b"));
        c.play();

        assert_eq!(c.step(0.5), 0); // still shot 0
        assert_eq!(c.step(0.7), 1); // crossed into shot 1 (total 1.2s)
        assert!(!c.is_finished());
        c.step(10.0); // past the end
        assert!(c.is_finished());
        assert!(!c.is_playing());
    }

    #[test]
    fn camera_interpolates_between_keyframes() {
        let mut c = Cutscene::new();
        c.add_shot([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, Some("a"));
        c.add_shot([10.0, 0.0, 0.0], [10.0, 0.0, -1.0], 1.0, Some("b"));
        c.play();

        c.step(0.0);
        let (pos0, _) = c.current_camera().unwrap();
        assert_eq!(pos0, [0.0, 0.0, 0.0]);

        c.step(1.5); // halfway through shot 1 (total time 1.5s)
        let (pos_mid, _) = c.current_camera().unwrap();
        // Glides from shot 0's start toward shot 1's start → somewhere in the middle.
        assert!(pos_mid[0] > 1.0 && pos_mid[0] < 9.0, "mid was {}", pos_mid[0]);
    }

    #[test]
    fn skip_jumps_to_end() {
        let mut c = Cutscene::new();
        c.add_shot([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, Some("a"));
        c.add_shot([1.0, 0.0, 0.0], [1.0, 0.0, -1.0], 2.0, Some("b"));
        c.play();
        c.skip();
        assert!(c.is_finished());
        assert!((c.current_time() - c.duration()).abs() < 1e-4);
    }

    #[test]
    fn camera_ownership_can_be_turned_off() {
        let mut c = Cutscene::new();
        assert!(c.drives_camera());
        c.set_drives_camera(false);
        assert!(!c.drives_camera());
        assert!(c.duration() == 0.0);
    }

    #[test]
    fn looping_wraps_time() {
        let mut c = Cutscene::new();
        c.add_shot([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, Some("a"));
        c.looping = true;
        c.play();
        c.step(0.5);
        c.step(0.5); // total 1.0 → wraps
        assert!(!c.is_finished());
        assert!(c.current_time() < 1.0);
    }
}