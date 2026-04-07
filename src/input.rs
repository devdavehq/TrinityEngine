// We need HashSet from Rust's standard library.
// std::collections is a module in Rust's built-in library.
// HashSet is a collection that holds unique values — no duplicates.
use std::collections::HashSet;
use gilrs::{Axis, Button, EventType, Gilrs};

// KeyCode represents a physical key position on the keyboard.
// PhysicalKey is the wrapper type winit uses — we unwrap it to get KeyCode.
use winit::keyboard::{KeyCode, PhysicalKey};

// "pub" on the struct means: other files (like main.rs) can use InputState.
// Without pub it would be invisible outside this file.
pub struct InputState {
    // This set contains every KeyCode that is currently held down.
    // When you press W, W is inserted.
    // When you release W, W is removed.
    // Asking "is W in here?" is very fast — O(1) time.
    held: HashSet<KeyCode>,
    gamepad_enabled: bool,
    deadzone: f32,
    left_x: f32,
    left_y: f32,
    south_pressed: bool,
    gilrs: Option<Gilrs>,
}

// "impl InputState" means: here are the functions that belong to InputState.
// Like a class's methods in other languages.
impl InputState {
    // "pub fn new()" creates a fresh InputState.
    // "-> Self" means it returns an InputState (Self = the type we're implementing).
    // This is the constructor pattern in Rust.
    pub fn new() -> Self {
        let gilrs = Gilrs::new().ok();
        Self {
            // HashSet::new() creates an empty set.
            // Nothing is held down at startup.
            held: HashSet::new(),
            gamepad_enabled: true,
            deadzone: 0.2,
            left_x: 0.0,
            left_y: 0.0,
            south_pressed: false,
            gilrs,
        }
    }

    pub fn configure_gamepad(&mut self, enabled: bool, deadzone: f32) {
        self.gamepad_enabled = enabled;
        self.deadzone = deadzone.clamp(0.0, 0.95);
    }

    // Call this when the OS tells you a key was pressed.
    // "&mut self" means: I need to modify this InputState (add to the set).
    // "key: KeyCode" is the key that was pressed.
    pub fn handle_key(&mut self, key: PhysicalKey, pressed: bool) {
        // PhysicalKey is an enum — it could be Code(KeyCode) or Unidentified.
        // "if let" means: if it IS a Code, extract the inner KeyCode as "code".
        // If it's Unidentified, do nothing — the if let just skips.
        if let PhysicalKey::Code(code) = key {
            if pressed {
                // insert() puts the key into the set.
                // If it's already there (key held and repeated), nothing changes.
                self.held.insert(code);
            } else {
                // remove() takes the key out.
                // The & is because remove() needs a reference, not the value itself.
                self.held.remove(&code);
            }
        }
    }

    // Call this every frame to ask "is this key currently held?"
    // "&self" means: I only read, I don't modify anything.
    // Returns a bool — true if held, false if not.
    pub fn is_held(&self, key: KeyCode) -> bool {
        // contains() checks if the key is in the set.
        // Returns true or false.
        self.held.contains(&key)
    }

    pub fn update_gamepads(&mut self) {
        if !self.gamepad_enabled {
            return;
        }
        if let Some(g) = &mut self.gilrs {
            while let Some(ev) = g.next_event() {
                match ev.event {
                    EventType::AxisChanged(Axis::LeftStickX, val, _) => self.left_x = val,
                    EventType::AxisChanged(Axis::LeftStickY, val, _) => self.left_y = val,
                    EventType::ButtonPressed(Button::South, _) => self.south_pressed = true,
                    EventType::ButtonReleased(Button::South, _) => self.south_pressed = false,
                    _ => {}
                }
            }
        }
    }

    pub fn is_virtual_key_held(&self, key: &str) -> bool {
        let stick_left = self.left_x < -self.deadzone;
        let stick_right = self.left_x > self.deadzone;
        let stick_up = self.left_y > self.deadzone;
        let stick_down = self.left_y < -self.deadzone;

        match key {
            "W" | "ArrowUp" => {
                self.is_held(KeyCode::KeyW) || self.is_held(KeyCode::ArrowUp) || stick_up
            }
            "S" | "ArrowDown" => {
                self.is_held(KeyCode::KeyS) || self.is_held(KeyCode::ArrowDown) || stick_down
            }
            "A" | "ArrowLeft" => {
                self.is_held(KeyCode::KeyA) || self.is_held(KeyCode::ArrowLeft) || stick_left
            }
            "D" | "ArrowRight" => {
                self.is_held(KeyCode::KeyD) || self.is_held(KeyCode::ArrowRight) || stick_right
            }
            "Space" => self.is_held(KeyCode::Space) || self.south_pressed,
            "Shift" => self.is_held(KeyCode::ShiftLeft) || self.is_held(KeyCode::ShiftRight),
            "Ctrl" => self.is_held(KeyCode::ControlLeft) || self.is_held(KeyCode::ControlRight),
            "E" => self.is_held(KeyCode::KeyE),
            "Q" => self.is_held(KeyCode::KeyQ),
            "R" => self.is_held(KeyCode::KeyR),
            "F" => self.is_held(KeyCode::KeyF),
            _ => false,
        }
    }
}