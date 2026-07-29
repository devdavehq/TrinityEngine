// src/core.rs
// ──────────────────────────────────────────────────────────────────────────────
// Core engine infrastructure.
//
// Everything in this module is foundational. Other systems depend on core,
// but core never depends on other systems. That's the rule.
//
// MODULES:
//   event_bus  — type-safe pub/sub event dispatch (the engine's nervous system)
//   events     — all engine event type definitions
//   systems    — System trait + SystemScheduler for composable frame loop
// ──────────────────────────────────────────────────────────────────────────────

pub mod event_bus;
pub mod events;
pub mod hierarchy;
pub mod systems;

pub use event_bus::EventBus;
pub use events::*;
