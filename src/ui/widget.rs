// src/ui/widget.rs
// ──────────────────────────────────────────────────────────────────────────────
// UI widget types — the visual elements users place on the canvas.
//
// Each widget has:
//   - Position/size on the canvas (x, y, w, h)
//   - Anchor point (relative to screen edges or parent)
//   - Visual style (colors, font, border, corner radius, shadow)
//   - Behavior kind (label, button, health bar, etc.)
//   - Runtime data (text, value, visibility, z-order)
//   - Optional parent for grouping
//   - Custom Lua hook for interactivity
// ──────────────────────────────────────────────────────────────────────────────

// ── Widget kind — the type of visual element ──────────────────────────────────
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, Hash)]
pub enum UiWidgetKind {
    Label,
    Button,
    HealthBar,
    ManaBar,
    StaminaBar,
    Counter,
    Slider,
    Toggle,
    Panel,
    ProgressRing,
    Meter,
    Image,
    Tooltip,
    Minimap,
    DamageNumber,
}

impl UiWidgetKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Label => "Label",
            Self::Button => "Button",
            Self::HealthBar => "Health Bar",
            Self::ManaBar => "Mana Bar",
            Self::StaminaBar => "Stamina Bar",
            Self::Counter => "Counter",
            Self::Slider => "Slider",
            Self::Toggle => "Toggle",
            Self::Panel => "Panel",
            Self::ProgressRing => "Progress Ring",
            Self::Meter => "Meter",
            Self::Image => "Image",
            Self::Tooltip => "Tooltip",
            Self::Minimap => "Minimap",
            Self::DamageNumber => "Damage Number",
        }
    }

    pub fn default_height(&self) -> f32 {
        match self {
            Self::Label | Self::Counter | Self::Tooltip | Self::DamageNumber => 24.0,
            Self::Button => 32.0,
            Self::HealthBar | Self::ManaBar | Self::StaminaBar => 16.0,
            Self::Slider => 28.0,
            Self::Toggle => 24.0,
            Self::Panel => 200.0,
            Self::ProgressRing => 80.0,
            Self::Meter => 40.0,
            Self::Image => 64.0,
            Self::Minimap => 160.0,
        }
    }

    pub fn default_width(&self) -> f32 {
        match self {
            Self::Label => 200.0,
            Self::Button => 120.0,
            Self::HealthBar | Self::ManaBar | Self::StaminaBar => 280.0,
            Self::Counter => 180.0,
            Self::Slider => 220.0,
            Self::Toggle => 60.0,
            Self::Panel => 300.0,
            Self::ProgressRing => 80.0,
            Self::Meter => 200.0,
            Self::Image => 64.0,
            Self::Tooltip => 180.0,
            Self::Minimap => 160.0,
            Self::DamageNumber => 100.0,
        }
    }

    pub fn all() -> &'static [UiWidgetKind] {
        &[
            Self::Label, Self::Button, Self::HealthBar, Self::ManaBar,
            Self::StaminaBar, Self::Counter, Self::Slider, Self::Toggle,
            Self::Panel, Self::ProgressRing, Self::Meter, Self::Image,
            Self::Tooltip, Self::Minimap, Self::DamageNumber,
        ]
    }
}

impl std::fmt::Display for UiWidgetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ── Anchor — relative positioning ────────────────────────────────────────────
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UiAnchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl UiAnchor {
    pub fn all() -> &'static [UiAnchor] {
        &[
            Self::TopLeft, Self::TopCenter, Self::TopRight,
            Self::CenterLeft, Self::Center, Self::CenterRight,
            Self::BottomLeft, Self::BottomCenter, Self::BottomRight,
        ]
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            Self::TopLeft => "TL",
            Self::TopCenter => "TC",
            Self::TopRight => "TR",
            Self::CenterLeft => "CL",
            Self::Center => "C",
            Self::CenterRight => "CR",
            Self::BottomLeft => "BL",
            Self::BottomCenter => "BC",
            Self::BottomRight => "BR",
        }
    }

    /// Compute screen position from widget x/y + screen dimensions.
    pub fn compute_screen_pos(&self, x: f32, y: f32, w: f32, h: f32, screen_w: f32, screen_h: f32) -> [f32; 2] {
        match self {
            Self::TopLeft      => [x, y],
            Self::TopCenter    => [screen_w * 0.5 - w * 0.5 + x, y],
            Self::TopRight     => [screen_w - w - x, y],
            Self::CenterLeft   => [x, screen_h * 0.5 - h * 0.5 + y],
            Self::Center       => [screen_w * 0.5 - w * 0.5 + x, screen_h * 0.5 - h * 0.5 + y],
            Self::CenterRight  => [screen_w - w - x, screen_h * 0.5 - h * 0.5 + y],
            Self::BottomLeft   => [x, screen_h - h - y],
            Self::BottomCenter => [screen_w * 0.5 - w * 0.5 + x, screen_h - h - y],
            Self::BottomRight  => [screen_w - w - x, screen_h - h - y],
        }
    }
}

impl Default for UiAnchor {
    fn default() -> Self { Self::TopLeft }
}

impl std::fmt::Display for UiAnchor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

// ── Widget style — visual properties ─────────────────────────────────────────
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UiWidgetStyle {
    pub text_color: [f32; 4],
    pub bg_color: [f32; 4],
    pub border_color: [f32; 4],
    pub border_width: f32,
    pub corner_radius: f32,
    pub font_size: f32,
    pub shadow_offset: [f32; 2],
    pub shadow_color: [f32; 4],
    pub glow_enabled: bool,
    pub glow_color: [f32; 4],
    pub glow_radius: f32,
    pub opacity: f32,
    pub bar_fill_color: [f32; 4],
    pub bar_bg_color: [f32; 4],
    pub bar_corner_radius: f32,
}

impl Default for UiWidgetStyle {
    fn default() -> Self {
        Self {
            text_color: [1.0, 1.0, 1.0, 1.0],
            bg_color: [0.0, 0.0, 0.0, 0.5],
            border_color: [0.3, 0.3, 0.3, 0.8],
            border_width: 1.0,
            corner_radius: 4.0,
            font_size: 14.0,
            shadow_offset: [2.0, 2.0],
            shadow_color: [0.0, 0.0, 0.0, 0.5],
            glow_enabled: false,
            glow_color: [0.5, 0.8, 1.0, 0.6],
            glow_radius: 4.0,
            opacity: 1.0,
            bar_fill_color: [0.2, 0.8, 0.2, 1.0],
            bar_bg_color: [0.15, 0.15, 0.15, 0.8],
            bar_corner_radius: 3.0,
        }
    }
}

impl UiWidgetStyle {
    /// Default style for health/mana/stamina bars.
    pub fn bar_style(fill: [f32; 4]) -> Self {
        Self {
            bar_fill_color: fill,
            bar_bg_color: [0.15, 0.15, 0.15, 0.8],
            bar_corner_radius: 3.0,
            border_width: 1.0,
            border_color: [0.4, 0.4, 0.4, 0.6],
            ..Default::default()
        }
    }

    /// Style for damage numbers (yellow, no background, larger font).
    pub fn damage_number_style() -> Self {
        Self {
            text_color: [1.0, 1.0, 0.0, 1.0],
            bg_color: [0.0, 0.0, 0.0, 0.0],
            font_size: 20.0,
            glow_enabled: true,
            glow_color: [1.0, 0.8, 0.0, 0.8],
            glow_radius: 6.0,
            border_width: 0.0,
            ..Default::default()
        }
    }
}

// ── UiWidget — a single placed widget ────────────────────────────────────────
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct UiWidget {
    pub id: String,
    pub kind: UiWidgetKind,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub visible: bool,
    pub z_order: i32,
    pub anchor: UiAnchor,
    pub style: UiWidgetStyle,
    pub text: String,
    pub value: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub locked: bool,
    pub parent_id: Option<String>,
    pub lua_hook: String,
}

impl UiWidget {
    pub fn new(id: &str, kind: UiWidgetKind) -> Self {
        let style = match kind {
            UiWidgetKind::HealthBar => UiWidgetStyle::bar_style([0.8, 0.2, 0.2, 1.0]),
            UiWidgetKind::ManaBar => UiWidgetStyle::bar_style([0.2, 0.4, 0.9, 1.0]),
            UiWidgetKind::StaminaBar => UiWidgetStyle::bar_style([0.9, 0.8, 0.2, 1.0]),
            UiWidgetKind::DamageNumber => UiWidgetStyle::damage_number_style(),
            _ => UiWidgetStyle::default(),
        };
        Self {
            id: id.to_string(),
            kind,
            x: 24.0,
            y: 24.0,
            w: kind.default_width(),
            h: kind.default_height(),
            visible: true,
            z_order: 0,
            anchor: UiAnchor::default(),
            style,
            text: String::new(),
            value: 1.0,
            min_value: 0.0,
            max_value: 1.0,
            locked: false,
            parent_id: None,
            lua_hook: String::new(),
        }
    }

    /// Screen position after anchor resolution.
    pub fn screen_pos(&self, screen_w: f32, screen_h: f32) -> [f32; 2] {
        self.anchor.compute_screen_pos(self.x, self.y, self.w, self.h, screen_w, screen_h)
    }

    /// Check if a point (screen coords) is inside this widget.
    pub fn contains_point(&self, px: f32, py: f32, screen_w: f32, screen_h: f32) -> bool {
        let [sx, sy] = self.screen_pos(screen_w, screen_h);
        px >= sx && px <= sx + self.w && py >= sy && py <= sy + self.h
    }
}

// ── Preset widgets for the library ───────────────────────────────────────────
impl UiWidget {
    pub fn preset_player_hud() -> Vec<UiWidget> {
        vec![
            UiWidget { id: "player_health".to_string(), kind: UiWidgetKind::HealthBar,
                anchor: UiAnchor::TopLeft, x: 24.0, y: 24.0, w: 280.0, h: 16.0,
                style: UiWidgetStyle::bar_style([0.8, 0.2, 0.2, 1.0]),
                text: "100".to_string(), max_value: 100.0, ..Self::placeholder() },
            UiWidget { id: "player_mana".to_string(), kind: UiWidgetKind::ManaBar,
                anchor: UiAnchor::TopLeft, x: 24.0, y: 48.0, w: 280.0, h: 16.0,
                style: UiWidgetStyle::bar_style([0.2, 0.4, 0.9, 1.0]),
                text: "100".to_string(), max_value: 100.0, ..Self::placeholder() },
            UiWidget { id: "player_stamina".to_string(), kind: UiWidgetKind::StaminaBar,
                anchor: UiAnchor::TopLeft, x: 24.0, y: 72.0, w: 280.0, h: 16.0,
                style: UiWidgetStyle::bar_style([0.9, 0.8, 0.2, 1.0]),
                text: "100".to_string(), max_value: 100.0, ..Self::placeholder() },
            UiWidget { id: "coin_counter".to_string(), kind: UiWidgetKind::Counter,
                anchor: UiAnchor::TopRight, x: 24.0, y: 24.0, w: 180.0, h: 24.0,
                text: "0".to_string(), ..Self::placeholder() },
        ]
    }

    pub fn preset_damage_popup() -> Vec<UiWidget> {
        vec![
            UiWidget { id: "damage_number".to_string(), kind: UiWidgetKind::DamageNumber,
                anchor: UiAnchor::Center, x: 0.0, y: -50.0, w: 100.0, h: 24.0,
                text: "-0".to_string(), visible: false, ..Self::placeholder() },
        ]
    }

    pub fn preset_minimap_hud() -> Vec<UiWidget> {
        vec![
            UiWidget { id: "minimap".to_string(), kind: UiWidgetKind::Minimap,
                anchor: UiAnchor::TopRight, x: 16.0, y: 60.0, w: 160.0, h: 160.0,
                style: UiWidgetStyle { corner_radius: 8.0, border_width: 2.0,
                    border_color: [0.4, 0.6, 0.8, 0.8], ..Default::default() },
                ..Self::placeholder() },
        ]
    }

    fn placeholder() -> Self {
        Self {
            id: String::new(), kind: UiWidgetKind::Label,
            x: 0.0, y: 0.0, w: 100.0, h: 24.0,
            visible: true, z_order: 0, anchor: UiAnchor::TopLeft,
            style: UiWidgetStyle::default(), text: String::new(),
            value: 1.0, min_value: 0.0, max_value: 1.0,
            locked: false, parent_id: None, lua_hook: String::new(),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widget_kind_names() {
        assert_eq!(UiWidgetKind::Label.name(), "Label");
        assert_eq!(UiWidgetKind::HealthBar.name(), "Health Bar");
        assert_eq!(UiWidgetKind::DamageNumber.name(), "Damage Number");
    }

    #[test]
    fn widget_kind_dimensions() {
        assert_eq!(UiWidgetKind::HealthBar.default_height(), 16.0);
        assert_eq!(UiWidgetKind::Button.default_height(), 32.0);
        assert_eq!(UiWidgetKind::Panel.default_width(), 300.0);
    }

    #[test]
    fn anchor_screen_positions() {
        // TopLeft: position is just (x, y)
        let pos = UiAnchor::TopLeft.compute_screen_pos(10.0, 20.0, 100.0, 50.0, 800.0, 600.0);
        assert_eq!(pos, [10.0, 20.0]);

        // Center: centered on screen with offset
        let pos = UiAnchor::Center.compute_screen_pos(0.0, 0.0, 100.0, 50.0, 800.0, 600.0);
        assert_eq!(pos, [350.0, 275.0]);

        // BottomRight: offset from bottom-right corner
        let pos = UiAnchor::BottomRight.compute_screen_pos(10.0, 20.0, 100.0, 50.0, 800.0, 600.0);
        assert_eq!(pos, [690.0, 530.0]);
    }

    #[test]
    fn widget_contains_point() {
        let w = UiWidget { x: 100.0, y: 100.0, w: 200.0, h: 50.0, ..UiWidget::new("test", UiWidgetKind::Label) };
        assert!(w.contains_point(150.0, 120.0, 800.0, 600.0)); // inside
        assert!(!w.contains_point(50.0, 50.0, 800.0, 600.0));   // outside
        assert!(!w.contains_point(350.0, 150.0, 800.0, 600.0)); // right edge
    }

    #[test]
    fn widget_presets() {
        let hud = UiWidget::preset_player_hud();
        assert_eq!(hud.len(), 4);
        assert_eq!(hud[0].kind, UiWidgetKind::HealthBar);
        assert_eq!(hud[0].max_value, 100.0);

        let dmg = UiWidget::preset_damage_popup();
        assert_eq!(dmg.len(), 1);
        assert_eq!(dmg[0].kind, UiWidgetKind::DamageNumber);
        assert!(!dmg[0].visible);
    }

    #[test]
    fn widget_serialization_roundtrip() {
        let w = UiWidget::new("hp", UiWidgetKind::HealthBar);
        let json = serde_json::to_string(&w).unwrap();
        let w2: UiWidget = serde_json::from_str(&json).unwrap();
        assert_eq!(w2.id, "hp");
        assert_eq!(w2.kind, UiWidgetKind::HealthBar);
        assert_eq!(w2.w, 280.0);
    }
}
