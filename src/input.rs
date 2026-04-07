// We need HashSet from Rust's standard library.
// std::collections is a module in Rust's built-in library.
// HashSet is a collection that holds unique values — no duplicates.
use std::collections::HashSet;

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
}

// "impl InputState" means: here are the functions that belong to InputState.
// Like a class's methods in other languages.
impl InputState {
    // "pub fn new()" creates a fresh InputState.
    // "-> Self" means it returns an InputState (Self = the type we're implementing).
    // This is the constructor pattern in Rust.
    pub fn new() -> Self {
        Self {
            // HashSet::new() creates an empty set.
            // Nothing is held down at startup.
            held: HashSet::new(),
        }
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
}