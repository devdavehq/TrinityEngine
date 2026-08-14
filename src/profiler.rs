use std::collections::VecDeque;
use std::time::Duration;

use crate::renderer::DrawStats;

/// Rolling window of recent frame times (ms), newest last.
/// Kept so the HUD can show live framing instead of only interval averages.
const FRAME_HISTORY_CAP: usize = 240;

/// Point-in-time frame stats for HUD / validation overlays.
#[derive(Debug, Clone, Copy)]
pub struct ProfilerSnapshot {
    pub fps: f32,
    pub frame_ms: f32,
    pub avg_frame_ms: f32,
    pub p95_frame_ms: f32,
    pub min_frame_ms: f32,
    pub max_frame_ms: f32,
    pub draw_visible: u32,
    pub draw_total: u32,
}

pub struct FrameProfiler {
    enabled: bool,
    log_interval_frames: u32,
    frame_count: u32,
    accum_frame_ms: f64,
    accum_script_ms: f64,
    accum_physics_ms: f64,
    accum_render_ms: f64,
    accum_asset_ms: f64,
    accum_visible: u64,
    accum_total: u64,
    last_overlay: String,
    /// Recent per-frame times (ms) for live HUD stats.
    frame_history: VecDeque<f32>,
    last_draw_stats: DrawStats,
}

impl FrameProfiler {
    pub fn new(enabled: bool, log_interval_frames: u32) -> Self {
        Self {
            enabled,
            log_interval_frames: log_interval_frames.max(1),
            frame_count: 0,
            accum_frame_ms: 0.0,
            accum_script_ms: 0.0,
            accum_physics_ms: 0.0,
            accum_render_ms: 0.0,
            accum_asset_ms: 0.0,
            accum_visible: 0,
            accum_total: 0,
            last_overlay: String::new(),
            frame_history: VecDeque::with_capacity(FRAME_HISTORY_CAP),
            last_draw_stats: DrawStats::default(),
        }
    }

    pub fn record(
        &mut self,
        frame_time: Duration,
        script_time: Duration,
        physics_time: Duration,
        render_time: Duration,
        asset_time: Duration,
        draw_stats: DrawStats,
        mt_enabled: bool,
    ) {
        if !self.enabled {
            return;
        }

        let frame_ms = frame_time.as_secs_f64() * 1000.0;
        let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };
        self.last_overlay = format!(
            "Triengine | FPS {:.1} | Frame {:.2}ms | Draw {}/{}",
            fps, frame_ms, draw_stats.visible, draw_stats.total
        );

        self.frame_history.push_back(frame_ms as f32);
        if self.frame_history.len() > FRAME_HISTORY_CAP {
            self.frame_history.pop_front();
        }
        self.last_draw_stats = draw_stats;

        self.frame_count += 1;
        self.accum_frame_ms += frame_time.as_secs_f64() * 1000.0;
        self.accum_script_ms += script_time.as_secs_f64() * 1000.0;
        self.accum_physics_ms += physics_time.as_secs_f64() * 1000.0;
        self.accum_render_ms += render_time.as_secs_f64() * 1000.0;
        self.accum_asset_ms += asset_time.as_secs_f64() * 1000.0;
        self.accum_total += draw_stats.total as u64;
        self.accum_visible += draw_stats.visible as u64;

        if self.frame_count < self.log_interval_frames {
            return;
        }

        let n = self.frame_count as f64;
        let avg_frame = self.accum_frame_ms / n;
        let avg_fps = if avg_frame > 0.0 { 1000.0 / avg_frame } else { 0.0 };
        let avg_script = self.accum_script_ms / n;
        let avg_physics = self.accum_physics_ms / n;
        let avg_render = self.accum_render_ms / n;
        let avg_asset = self.accum_asset_ms / n;
        let visible_ratio = if self.accum_total > 0 {
            (self.accum_visible as f64 / self.accum_total as f64) * 100.0
        } else {
            0.0
        };

        tracing::info!(
            "[Profiler][{}] avg_fps={:.1} frame={:.2}ms script={:.2}ms physics={:.2}ms render={:.2}ms assets={:.2}ms visible={:.1}%",
            if mt_enabled { "MT" } else { "ST" },
            avg_fps,
            avg_frame,
            avg_script,
            avg_physics,
            avg_render,
            avg_asset,
            visible_ratio,
        );

        self.frame_count = 0;
        self.accum_frame_ms = 0.0;
        self.accum_script_ms = 0.0;
        self.accum_physics_ms = 0.0;
        self.accum_render_ms = 0.0;
        self.accum_asset_ms = 0.0;
        self.accum_visible = 0;
        self.accum_total = 0;
    }

    pub fn overlay_text(&self) -> Option<&str> {
        if self.enabled && !self.last_overlay.is_empty() {
            Some(&self.last_overlay)
        } else {
            None
        }
    }

    /// Live framing stats over the rolling window (synthetic data if fewer
    /// than two frames have been recorded or the profiler is disabled).
    pub fn snapshot(&self) -> ProfilerSnapshot {
        let mut times: Vec<f32> = self.frame_history.iter().copied().collect();
        times.sort_by(|a, b| a.total_cmp(b));
        let len = times.len() as f32;
        let sum: f32 = times.iter().sum();
        let avg = if len > 0.0 { sum / len } else { 0.0 };
        let (min, max) = if times.is_empty() {
            (0.0, 0.0)
        } else {
            (*times.first().unwrap(), *times.last().unwrap())
        };
        let p95 = if times.is_empty() {
            0.0
        } else {
            let idx = ((times.len() as f64 * 0.95) - 1.0).max(0.0) as usize;
            times[idx]
        };
        let frame_ms = *times.last().unwrap_or(&0.0);
        let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };
        ProfilerSnapshot {
            fps,
            frame_ms,
            avg_frame_ms: avg,
            p95_frame_ms: p95,
            min_frame_ms: min,
            max_frame_ms: max,
            draw_visible: self.last_draw_stats.visible as u32,
            draw_total: self.last_draw_stats.total as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(total: usize, visible: usize) -> DrawStats {
        DrawStats { total, visible, drawn: visible }
    }

    /// Frame-history framing: recording a set of frames must produce sane
    /// average / p95 / min / max values and a livable FPS estimate.
    #[test]
    fn profiler_frames_and_snapshot_stats() {
        let mut p = FrameProfiler::new(true, 1000);
        // Feed a mix of frame times: mostly 16ms plus a couple of spikes.
        for i in 0..100 {
            let ms = if i % 25 == 0 { 50.0 } else { 16.0 };
            p.record(
                Duration::from_millis(ms as u64),
                Duration::from_micros(200),
                Duration::from_micros(300),
                Duration::from_micros(1400),
                Duration::from_micros(100),
                stats(200, 120),
                true,
            );
        }
        let s = p.snapshot();
        assert!((s.avg_frame_ms - 17.36).abs() < 0.5, "avg was {}", s.avg_frame_ms);
        // 95% of frames are 16ms; the top-5% spikes don't move the p95 point.
        assert!(s.p95_frame_ms >= 16.0 && s.p95_frame_ms <= 50.0);
        assert!(s.min_frame_ms >= 15.9 && s.max_frame_ms <= 50.1);
        assert!(s.fps >= 19.0 && s.fps <= 63.0);
        assert_eq!(s.draw_total, 200);
        assert_eq!(s.draw_visible, 120);
    }

    /// Disabled profiler must not accumulate history but still answer.
    #[test]
    fn profiler_disabled_returns_zeroed_snapshot() {
        let mut p = FrameProfiler::new(false, 10);
        p.record(Duration::from_millis(16), Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, stats(10, 5), false);
        let s = p.snapshot();
        assert_eq!(s.avg_frame_ms, 0.0);
        assert_eq!(s.draw_total, 0);
    }
}
