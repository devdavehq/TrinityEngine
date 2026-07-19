// src/render.rs
// ──────────────────────────────────────────────────────────────────────────────
// Rendering subsystem.
//
// This module owns all GPU rendering: pipelines, shaders, resources, draw calls.
// The render graph defines pass ordering. The instancing manager batches draws.
//
// MODULES:
//   instancing — GPU instancing for batching identical meshes
//   graph      — render pass dependency graph (topological sort)
//
// EXISTING (in src/renderer.rs, will be migrated here over time):
//   renderer   — the current monolithic renderer (to be split)
//   pipeline   — render pipeline creation
//   shadow     — shadow mapping
//   ibl        — image-based lighting
// ──────────────────────────────────────────────────────────────────────────────

pub mod instancing;
pub mod graph;
pub mod shader_manager;

pub use instancing::{InstanceData, InstancingManager};
pub use graph::{RenderGraph, ResourceId, ResourceDesc, build_default_graph};
pub use shader_manager::ShaderManager;
