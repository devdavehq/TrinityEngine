use std::time::Duration;

use crate::renderer::DrawStats;

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
}
