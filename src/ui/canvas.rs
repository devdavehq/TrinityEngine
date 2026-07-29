// src/ui/canvas.rs
// ──────────────────────────────────────────────────────────────────────────────
// Interactive canvas for the visual UI designer.
//
// The canvas renders the current UI design in the editor viewport, handling:
//   - Widget rendering (preview of what the player will see)
//   - Drag-drop repositioning of widgets
//   - Selection highlighting
//   - Grid overlay (snap-to-grid)
//   - Multi-select via box selection
//   - Zoom and pan
//   - Resize handles on selected widget
// ──────────────────────────────────────────────────────────────────────────────

use super::*;

/// Canvas interaction state.
#[derive(Clone, Debug, Default)]
pub struct CanvasState {
    pub dragging: Option<DragState>,
    pub resizing: Option<ResizeState>,
    pub selecting: bool,
    pub selection_box: Option<[f32; 4]>,
    pub pan_offset: [f32; 2],
    pub zoom: f32,
}

#[derive(Clone, Debug)]
pub struct DragState {
    pub widget_index: usize,
    pub start_x: f32,
    pub start_y: f32,
    pub widget_start_x: f32,
    pub widget_start_y: f32,
}

#[derive(Clone, Debug)]
pub struct ResizeState {
    pub widget_index: usize,
    pub handle: ResizeHandle,
    pub start_x: f32,
    pub start_y: f32,
    pub widget_start_w: f32,
    pub widget_start_h: f32,
    pub widget_start_x: f32,
    pub widget_start_y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeHandle {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Top,
    Bottom,
    Left,
    Right,
}

const HANDLE_SIZE: f32 = 8.0;

/// Render the UI design canvas.
/// Returns interaction events (widget selection, drag, resize, etc.)
pub fn render_canvas(
    ui: &mut egui::Ui,
    design: &mut UiDesign,
    canvas_state: &mut CanvasState,
    screen_w: f32,
    screen_h: f32,
) -> CanvasEvent {
    let mut event = CanvasEvent::None;
    let avail = ui.available_size();
    let (response, painter) = ui.allocate_painter(egui::vec2(avail.x, avail.y), egui::Sense::click_and_drag());
    let rect = response.rect;

    // ── Background ───────────────────────────────────────────────────────────
    painter.rect_filled(rect, 0.0, egui::Color32::from_rgba_premultiplied(30, 30, 35, 255));

    // ── Grid overlay ─────────────────────────────────────────────────────────
    if design.show_grid {
        let grid_color = egui::Color32::from_rgba_premultiplied(60, 60, 70, 255);
        let grid_step = design.grid_size * canvas_state.zoom;
        if grid_step >= 4.0 {
            let mut x = (canvas_state.pan_offset[0] % grid_step) as f32;
            while x < rect.width() {
                painter.line_segment(
                    [egui::pos2(rect.left() + x, rect.top()), egui::pos2(rect.left() + x, rect.bottom())],
                    egui::Stroke::new(1.0, grid_color),
                );
                x += grid_step;
            }
            let mut y = (canvas_state.pan_offset[1] % grid_step) as f32;
            while y < rect.height() {
                painter.line_segment(
                    [egui::pos2(rect.left(), rect.top() + y), egui::pos2(rect.right(), rect.top() + y)],
                    egui::Stroke::new(1.0, grid_color),
                );
                y += grid_step;
            }
        }
    }

    // ── Screen bounds indicator ──────────────────────────────────────────────
    let screen_rect = egui::Rect::from_min_size(
        egui::pos2(rect.left() + canvas_state.pan_offset[0], rect.top() + canvas_state.pan_offset[1]),
        egui::vec2(screen_w * canvas_state.zoom, screen_h * canvas_state.zoom),
    );
    painter.rect_stroke(screen_rect, 0.0, egui::Stroke::new(2.0, egui::Color32::from_rgba_premultiplied(80, 80, 100, 180)), egui::StrokeKind::Middle);

    // ── Render each widget ───────────────────────────────────────────────────
    for (i, w) in design.widgets.iter().enumerate() {
        if !w.visible && design.selected != Some(i) {
            continue;
        }

        let [sx, sy] = w.screen_pos(screen_w, screen_h);
        let widget_rect = egui::Rect::from_min_size(
            egui::pos2(
                rect.left() + sx * canvas_state.zoom + canvas_state.pan_offset[0],
                rect.top() + sy * canvas_state.zoom + canvas_state.pan_offset[1],
            ),
            egui::vec2(w.w * canvas_state.zoom, w.h * canvas_state.zoom),
        );

        // ── Widget shadow ────────────────────────────────────────────────
        if w.style.shadow_offset[0] != 0.0 || w.style.shadow_offset[1] != 0.0 {
            let shadow_rect = widget_rect.translate(egui::vec2(
                w.style.shadow_offset[0] * canvas_state.zoom,
                w.style.shadow_offset[1] * canvas_state.zoom,
            ));
            let sc = w.style.shadow_color;
            painter.rect_filled(shadow_rect, w.style.corner_radius * canvas_state.zoom,
                egui::Color32::from_rgba_premultiplied(
                    (sc[0] * 255.0) as u8, (sc[1] * 255.0) as u8,
                    (sc[2] * 255.0) as u8, (sc[3] * 255.0) as u8,
                ));
        }

        // ── Widget background ────────────────────────────────────────────
        let bc = w.style.bg_color;
        let bg = egui::Color32::from_rgba_premultiplied(
            (bc[0] * 255.0) as u8, (bc[1] * 255.0) as u8,
            (bc[2] * 255.0) as u8, (bc[3] * 255.0) as u8,
        );
        painter.rect_filled(widget_rect, w.style.corner_radius * canvas_state.zoom, bg);

        // ── Widget border ────────────────────────────────────────────────
        if w.style.border_width > 0.0 {
            let brc = w.style.border_color;
            let border = egui::Color32::from_rgba_premultiplied(
                (brc[0] * 255.0) as u8, (brc[1] * 255.0) as u8,
                (brc[2] * 255.0) as u8, (brc[3] * 255.0) as u8,
            );
            painter.rect_stroke(widget_rect, w.style.corner_radius * canvas_state.zoom,
                egui::Stroke::new(w.style.border_width * canvas_state.zoom, border), egui::StrokeKind::Middle);
        }

        // ── Glow effect ──────────────────────────────────────────────────
        if w.style.glow_enabled {
            let gc = w.style.glow_color;
            let glow_color = egui::Color32::from_rgba_premultiplied(
                (gc[0] * 255.0) as u8, (gc[1] * 255.0) as u8,
                (gc[2] * 255.0) as u8, (gc[3] * 255.0) as u8,
            );
            let glow_rect = widget_rect.expand(w.style.glow_radius * canvas_state.zoom);
            painter.rect_stroke(glow_rect, w.style.corner_radius * canvas_state.zoom + w.style.glow_radius * canvas_state.zoom,
                egui::Stroke::new(w.style.glow_radius * canvas_state.zoom * 0.5, glow_color), egui::StrokeKind::Middle);
        }

        // ── Widget content preview ───────────────────────────────────────
        let tc = w.style.text_color;
        let text_color = egui::Color32::from_rgba_premultiplied(
            (tc[0] * 255.0) as u8, (tc[1] * 255.0) as u8,
            (tc[2] * 255.0) as u8, (tc[3] * 255.0) as u8,
        );
        let font_size = (w.style.font_size * canvas_state.zoom).max(8.0);

        match w.kind {
            UiWidgetKind::HealthBar | UiWidgetKind::ManaBar | UiWidgetKind::StaminaBar => {
                let fill_pct = (w.value / w.max_value).clamp(0.0, 1.0);
                let fill_rect = egui::Rect::from_min_size(
                    widget_rect.min,
                    egui::vec2(widget_rect.width() * fill_pct, widget_rect.height()),
                );
                let fc = w.style.bar_fill_color;
                let fill_color = egui::Color32::from_rgba_premultiplied(
                    (fc[0] * 255.0) as u8, (fc[1] * 255.0) as u8,
                    (fc[2] * 255.0) as u8, (fc[3] * 255.0) as u8,
                );
                painter.rect_filled(fill_rect, w.style.bar_corner_radius * canvas_state.zoom, fill_color);

                // Value text
                let label = if w.text.is_empty() { format!("{:.0}", w.value) } else { w.text.clone() };
                painter.text(widget_rect.center(), egui::Align2::CENTER_CENTER, &label,
                    egui::FontId::proportional(font_size), text_color);
            }
            UiWidgetKind::ProgressRing => {
                let center = widget_rect.center();
                let radius = widget_rect.width() * 0.4;
                painter.circle_stroke(center, radius, egui::Stroke::new(3.0 * canvas_state.zoom, text_color));
                let fill_angle = std::f32::consts::TAU * (w.value / w.max_value).clamp(0.0, 1.0);
                let points: Vec<egui::Pos2> = (0..=24).map(|i| {
                    let a = -std::f32::consts::FRAC_PI_2 + fill_angle * (i as f32 / 24.0);
                    egui::pos2(center.x + a.cos() * radius, center.y + a.sin() * radius)
                }).collect();
                painter.add(egui::Shape::line(points, egui::Stroke::new(3.0 * canvas_state.zoom, text_color)));
                let label = if w.text.is_empty() { format!("{:.0}%", w.value / w.max_value * 100.0) } else { w.text.clone() };
                painter.text(center, egui::Align2::CENTER_CENTER, &label,
                    egui::FontId::proportional(font_size * 0.9), text_color);
            }
            UiWidgetKind::Meter => {
                let segments = 10;
                let fill_segments = ((w.value / w.max_value).clamp(0.0, 1.0) * segments as f32) as usize;
                let seg_w = (widget_rect.width() - 2.0 * canvas_state.zoom) / segments as f32;
                for s in 0..segments {
                    let seg_rect = egui::Rect::from_min_size(
                        egui::pos2(widget_rect.left() + 1.0 + seg_w * s as f32, widget_rect.top() + 1.0),
                        egui::vec2(seg_w - 1.0 * canvas_state.zoom, widget_rect.height() - 2.0),
                    );
                    let color = if s < fill_segments { text_color } else { bg };
                    painter.rect_filled(seg_rect, 1.0, color);
                }
            }
            UiWidgetKind::Toggle => {
                let toggle_color = if w.value > 0.5 { text_color } else { bg };
                painter.rect_filled(widget_rect, widget_rect.height() * 0.5, toggle_color);
                let knob_x = if w.value > 0.5 { widget_rect.right() - widget_rect.height() * 0.5 } else { widget_rect.left() };
                painter.circle_filled(
                    egui::pos2(knob_x + widget_rect.height() * 0.25, widget_rect.center().y),
                    widget_rect.height() * 0.35, egui::Color32::WHITE,
                );
            }
            _ => {
                let label = if w.text.is_empty() { w.kind.name().to_string() } else { w.text.clone() };
                painter.text(widget_rect.center(), egui::Align2::CENTER_CENTER, &label,
                    egui::FontId::proportional(font_size), text_color);
            }
        }

        // ── Selection highlight ──────────────────────────────────────────
        if design.selected == Some(i) {
            painter.rect_stroke(widget_rect, 0.0, egui::Stroke::new(2.0, egui::Color32::from_rgb(60, 140, 255)), egui::StrokeKind::Middle);

            // Resize handles
            let handle_rects = get_handle_rects(widget_rect, HANDLE_SIZE * canvas_state.zoom);
            for (_, hr) in handle_rects {
                painter.rect_filled(hr, 1.0, egui::Color32::from_rgb(60, 140, 255));
            }
        } else if !w.visible {
            // Ghost outline for hidden widgets
            painter.rect_stroke(widget_rect, 0.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(100, 100, 100, 80)), egui::StrokeKind::Middle);
        }
    }

    // ── Handle interaction ───────────────────────────────────────────────
    if let Some(cursor) = response.interact_pointer_pos() {

        // Check resize handles first (only for selected widget)
        if let Some(idx) = design.selected {
            let w = &design.widgets[idx];
            let [sx, sy] = w.screen_pos(screen_w, screen_h);
            let widget_rect = egui::Rect::from_min_size(
                egui::pos2(
                    rect.left() + sx * canvas_state.zoom + canvas_state.pan_offset[0],
                    rect.top() + sy * canvas_state.zoom + canvas_state.pan_offset[1],
                ),
                egui::vec2(w.w * canvas_state.zoom, w.h * canvas_state.zoom),
            );
            let handle_rects = get_handle_rects(widget_rect, HANDLE_SIZE * canvas_state.zoom);
            for (handle, hr) in handle_rects {
                if hr.contains(cursor) && response.drag_started() {
                    canvas_state.resizing = Some(ResizeState {
                        widget_index: idx,
                        handle,
                        start_x: cursor.x,
                        start_y: cursor.y,
                        widget_start_w: w.w,
                        widget_start_h: w.h,
                        widget_start_x: w.x,
                        widget_start_y: w.y,
                    });
                    return CanvasEvent::ResizeStarted(idx);
                }
            }
        }

        // Check widget hit testing for drag
        if let Some(idx) = design.selected {
            if response.drag_started() {
                let w = &design.widgets[idx];
                let [sx, sy] = w.screen_pos(screen_w, screen_h);
                let widget_rect = egui::Rect::from_min_size(
                    egui::pos2(
                        rect.left() + sx * canvas_state.zoom + canvas_state.pan_offset[0],
                        rect.top() + sy * canvas_state.zoom + canvas_state.pan_offset[1],
                    ),
                    egui::vec2(w.w * canvas_state.zoom, w.h * canvas_state.zoom),
                );
                if widget_rect.contains(cursor) {
                    canvas_state.dragging = Some(DragState {
                        widget_index: idx,
                        start_x: cursor.x,
                        start_y: cursor.y,
                        widget_start_x: w.x,
                        widget_start_y: w.y,
                    });
                    return CanvasEvent::DragStarted(idx);
                }
            }
        }

        // Hit test all widgets for selection
        if response.clicked() {
            let mut hit = None;
            for (i, w) in design.widgets.iter().enumerate().rev() {
                let [sx, sy] = w.screen_pos(screen_w, screen_h);
                let widget_rect = egui::Rect::from_min_size(
                    egui::pos2(
                        rect.left() + sx * canvas_state.zoom + canvas_state.pan_offset[0],
                        rect.top() + sy * canvas_state.zoom + canvas_state.pan_offset[1],
                    ),
                    egui::vec2(w.w * canvas_state.zoom, w.h * canvas_state.zoom),
                );
                if widget_rect.contains(cursor) {
                    hit = Some(i);
                    break;
                }
            }
            design.selected = hit;
            event = if let Some(i) = hit { CanvasEvent::WidgetSelected(i) } else { CanvasEvent::SelectionCleared };
        }
    }

    // Process ongoing drag
    if let Some(drag) = &canvas_state.dragging {
        if response.dragged() {
            let delta = response.drag_delta();
            let w = &mut design.widgets[drag.widget_index];
            w.x = drag.widget_start_x + delta.x / canvas_state.zoom;
            w.y = drag.widget_start_y + delta.y / canvas_state.zoom;
            if design.snap_to_grid {
                w.x = (w.x / design.grid_size).round() * design.grid_size;
                w.y = (w.y / design.grid_size).round() * design.grid_size;
            }
        } else if response.drag_stopped() {
            canvas_state.dragging = None;
        }
    }

    // Process ongoing resize
    if let Some(resize) = &canvas_state.resizing {
        if response.dragged() {
            let delta = response.drag_delta();
            let w = &mut design.widgets[resize.widget_index];
            let dz = 1.0 / canvas_state.zoom;
            match resize.handle {
                ResizeHandle::Right => w.w = (resize.widget_start_w + delta.x * dz).max(20.0),
                ResizeHandle::Bottom => w.h = (resize.widget_start_h + delta.y * dz).max(16.0),
                ResizeHandle::Left => {
                    let new_w = (resize.widget_start_w - delta.x * dz).max(20.0);
                    w.x = resize.widget_start_x + (resize.widget_start_w - new_w);
                    w.w = new_w;
                }
                ResizeHandle::Top => {
                    let new_h = (resize.widget_start_h - delta.y * dz).max(16.0);
                    w.y = resize.widget_start_y + (resize.widget_start_h - new_h);
                    w.h = new_h;
                }
                _ => {}
            }
        } else if response.drag_stopped() {
            canvas_state.resizing = None;
        }
    }

    // ── Design name label ────────────────────────────────────────────────
    painter.text(
        rect.left_top() + egui::vec2(8.0, 4.0),
        egui::Align2::LEFT_TOP,
        &format!("Design: {} | {} widgets | Zoom: {:.0}%", design.name, design.widgets.len(), canvas_state.zoom * 100.0),
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(140, 140, 160),
    );

    event
}

/// Events emitted by the canvas.
#[derive(Debug)]
pub enum CanvasEvent {
    None,
    WidgetSelected(usize),
    SelectionCleared,
    DragStarted(usize),
    ResizeStarted(usize),
}

/// Get resize handle rectangles for a widget.
fn get_handle_rects(widget_rect: egui::Rect, handle_size: f32) -> Vec<(ResizeHandle, egui::Rect)> {
    let hs = handle_size;
    let _half = hs * 0.5;
    vec![
        (ResizeHandle::TopLeft,      egui::Rect::from_center_size(egui::pos2(widget_rect.left(), widget_rect.top()), egui::vec2(hs, hs))),
        (ResizeHandle::TopRight,     egui::Rect::from_center_size(egui::pos2(widget_rect.right(), widget_rect.top()), egui::vec2(hs, hs))),
        (ResizeHandle::BottomLeft,   egui::Rect::from_center_size(egui::pos2(widget_rect.left(), widget_rect.bottom()), egui::vec2(hs, hs))),
        (ResizeHandle::BottomRight,  egui::Rect::from_center_size(egui::pos2(widget_rect.right(), widget_rect.bottom()), egui::vec2(hs, hs))),
        (ResizeHandle::Top,          egui::Rect::from_center_size(egui::pos2(widget_rect.center().x, widget_rect.top()), egui::vec2(hs, hs))),
        (ResizeHandle::Bottom,       egui::Rect::from_center_size(egui::pos2(widget_rect.center().x, widget_rect.bottom()), egui::vec2(hs, hs))),
        (ResizeHandle::Left,         egui::Rect::from_center_size(egui::pos2(widget_rect.left(), widget_rect.center().y), egui::vec2(hs, hs))),
        (ResizeHandle::Right,        egui::Rect::from_center_size(egui::pos2(widget_rect.right(), widget_rect.center().y), egui::vec2(hs, hs))),
    ]
}
