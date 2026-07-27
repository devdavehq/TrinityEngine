// src/ai.rs
// ──────────────────────────────────────────────────────────────────────────────
// AI system for TrinEngine.
//
// Behavior tree framework with composites (Sequence, Selector, Parallel),
// decorators (Inverter, Repeater, Cooldown, Conditional), leaf actions
// (MoveTo, Patrol, Wait, Log), a Blackboard key-value store, and
// Lua bindings so users can build AI from scripts.
//
// Sub-modules:
//   behavior_tree — BT node types, BehaviorTree struct, BTContext
//   blackboard    — per-entity key-value data store
//   components    — AiAgent ECS component, AiRegistry, ai_system()
// ──────────────────────────────────────────────────────────────────────────────

pub mod behavior_tree;
pub mod blackboard;
pub mod components;

pub use behavior_tree::*;
pub use blackboard::*;
pub use components::*;
