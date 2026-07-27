// src/animation/anim_graph.rs
// Node-based animation graph — state machines, transition conditions, layers.
//
// ── Architecture ─────────────────────────────────────────────────────────────
// An AnimationGraph is a directed graph of animation states connected by
// transitions. Each transition has one or more conditions that must ALL be
// true for the transition to fire.
//
// Example Walk/Run/Idle graph:
//
//   ┌──────────┐   speed > 0.5   ┌──────────┐   speed > 3.0   ┌─────────┐
//   │   Idle   │ ──────────────► │   Walk   │ ──────────────► │   Run   │
//   └──────────┘                 └──────────┘                 └─────────┘
//        ▲                            │                            │
//        │      speed < 0.5           │      speed < 3.0           │
//        └────────────────────────────┘◄───────────────────────────┘
//
// ── How it works ─────────────────────────────────────────────────────────────
// 1. Each entity has an AnimGraphComponent with a graph + parameters.
// 2. Each frame, the anim_graph_system:
//    a. Reads parameters (speed, ai_state, is_grounded, etc.)
//    b. Evaluates conditions on all transitions from the current state.
//    c. If a transition fires, sets the new state + triggers crossfade.
//    d. Calls SkeletalAnimator::play() with the new state's clip.
// 3. The existing animation_blending_system handles the actual crossfade.
//
// ── Why this matters ─────────────────────────────────────────────────────────
// WITHOUT AnimGraph:
//   - BT must manually write "ai_state" for EVERY animation change
//   - Walk→Run transition needs custom code
//   - Upper body attack while running requires special-case blending
//   - Jump sequence (launch→fall→land) needs manual state tracking
//
// WITH AnimGraph:
//   - BT just writes parameters (speed, ai_state, is_grounded)
//   - Graph auto-evaluates which state to be in
//   - Transitions are data-driven (no code per transition)
//   - Layers allow independent upper/lower body states
//   - Designers can modify transitions without touching code
//
// ── UE5 comparison ───────────────────────────────────────────────────────────
// This is equivalent to UE5's Animation Blueprint (AnimBP):
//   - AnimState = our AnimStateNode
//   - Transition Rule = our TransitionCondition
//   - AnimLayer = our AnimLayer (parallel state machines)
//   - Blend Poses by Bool/Enum/Float = our conditions

use std::collections::HashMap;

use crate::animation::skeletal::{
    AnimationClip, AnimationStateMap, BoneHierarchy, SkeletalAnimator, TransformKeyframe,
    blend_animated_locals, compute_joint_matrices,
};
use glam::{Mat4, Vec3};

// AiAgent carries the per-entity Blackboard that the BT writes to.
// We read it here to sync blackboard values into AnimGraph parameters.
use crate::ai::components::AiAgent;

// ═══════════════════════════════════════════════════════════════════════════════
// PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════════
// Parameters are the inputs to the animation graph.
// Game logic / BT writes parameters; the graph reads them to evaluate transitions.

/// A typed parameter value that conditions can evaluate against.
#[derive(Clone, Debug, PartialEq)]
pub enum AnimParamValue {
    Float(f32),
    Bool(bool),
    Enum(u32),
    String(String),
}

impl AnimParamValue {
    pub fn as_float(&self) -> f32 {
        match self {
            AnimParamValue::Float(v) => *v,
            AnimParamValue::Bool(v) => if *v { 1.0 } else { 0.0 },
            AnimParamValue::Enum(v) => *v as f32,
            _ => 0.0,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            AnimParamValue::Bool(v) => *v,
            AnimParamValue::Float(v) => *v > 0.5,
            AnimParamValue::Enum(v) => *v != 0,
            _ => false,
        }
    }

    pub fn as_enum(&self) -> u32 {
        match self {
            AnimParamValue::Enum(v) => *v,
            AnimParamValue::Float(v) => *v as u32,
            AnimParamValue::Bool(v) => if *v { 1 } else { 0 },
            _ => 0,
        }
    }

    pub fn as_string(&self) -> &str {
        match self {
            AnimParamValue::String(v) => v.as_str(),
            _ => "",
        }
    }
}

/// Parameter container — key-value store of named animation parameters.
#[derive(Clone, Debug, Default)]
pub struct AnimParameters {
    params: HashMap<String, AnimParamValue>,
}

impl AnimParameters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_float(&mut self, name: &str, value: f32) {
        self.params.insert(name.to_string(), AnimParamValue::Float(value));
    }

    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.params.insert(name.to_string(), AnimParamValue::Bool(value));
    }

    pub fn set_enum(&mut self, name: &str, value: u32) {
        self.params.insert(name.to_string(), AnimParamValue::Enum(value));
    }

    pub fn set_string(&mut self, name: &str, value: &str) {
        self.params.insert(name.to_string(), AnimParamValue::String(value.to_string()));
    }

    pub fn get(&self, name: &str) -> Option<&AnimParamValue> {
        self.params.get(name)
    }

    pub fn get_float(&self, name: &str) -> f32 {
        self.get(name).map(|v| v.as_float()).unwrap_or(0.0)
    }

    pub fn get_bool(&self, name: &str) -> bool {
        self.get(name).map(|v| v.as_bool()).unwrap_or(false)
    }

    pub fn get_enum(&self, name: &str) -> u32 {
        self.get(name).map(|v| v.as_enum()).unwrap_or(0)
    }

    pub fn get_string(&self, name: &str) -> String {
        self.get(name).map(|v| v.as_string().to_string()).unwrap_or_default()
    }

    pub fn has(&self, name: &str) -> bool {
        self.params.contains_key(name)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSITION CONDITIONS
// ═══════════════════════════════════════════════════════════════════════════════
// Conditions are evaluated every frame. ALL conditions on a transition must
// be true for the transition to fire.

/// A single condition that can be evaluated against parameters.
#[derive(Clone, Debug)]
pub enum TransitionCondition {
    /// Parameter value comparison: Float
    FloatGreaterThan { param: String, threshold: f32 },
    FloatLessThan { param: String, threshold: f32 },
    FloatInRange { param: String, min: f32, max: f32 },

    /// Parameter value comparison: Bool
    BoolEquals { param: String, expected: bool },

    /// Parameter value comparison: Enum
    EnumEquals { param: String, expected: u32 },

    /// Parameter value comparison: String
    StringEquals { param: String, expected: String },

    /// Time spent in current state
    TimeInStateGreaterThan { seconds: f32 },

    /// Custom condition (closure-based, for one-off logic)
    Custom(fn(&AnimParameters) -> bool),
}

impl TransitionCondition {
    /// Evaluate this condition against the current parameters and time in state.
    pub fn evaluate(&self, params: &AnimParameters, time_in_state: f32) -> bool {
        match self {
            TransitionCondition::FloatGreaterThan { param, threshold } => {
                params.get_float(param) > *threshold
            }
            TransitionCondition::FloatLessThan { param, threshold } => {
                params.get_float(param) < *threshold
            }
            TransitionCondition::FloatInRange { param, min, max } => {
                let v = params.get_float(param);
                v >= *min && v <= *max
            }
            TransitionCondition::BoolEquals { param, expected } => {
                params.get_bool(param) == *expected
            }
            TransitionCondition::EnumEquals { param, expected } => {
                params.get_enum(param) == *expected
            }
            TransitionCondition::StringEquals { param, expected } => {
                params.get_string(param) == *expected
            }
            TransitionCondition::TimeInStateGreaterThan { seconds } => {
                time_in_state >= *seconds
            }
            TransitionCondition::Custom(f) => f(params),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATE NODES
// ═══════════════════════════════════════════════════════════════════════════════
// Each state node plays one animation clip and has outgoing transitions.

/// A single state in the animation graph.
///
/// Each state:
///   - Plays a specific AnimationClip (by index)
///   - Has outgoing transitions to other states
///   - Has a name for debugging/editor display
#[derive(Clone)]
pub struct AnimStateNode {
    /// Human-readable name (e.g., "Idle", "Walk", "Attack").
    pub name: String,
    /// Index into the AnimationClips library for this state's clip.
    pub clip_index: usize,
    /// Outgoing transitions from this state.
    pub transitions: Vec<AnimTransition>,
}

impl AnimStateNode {
    pub fn new(name: &str, clip_index: usize) -> Self {
        Self {
            name: name.to_string(),
            clip_index,
            transitions: Vec::new(),
        }
    }

    /// Add a transition to this state.
    pub fn add_transition(&mut self, transition: AnimTransition) {
        self.transitions.push(transition);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSITIONS
// ═══════════════════════════════════════════════════════════════════════════════
// A transition connects two states with a set of conditions.

/// A transition between two animation states.
///
/// ALL conditions must be true for the transition to fire.
/// Transitions have a priority — higher priority transitions are checked first.
#[derive(Clone)]
pub struct AnimTransition {
    /// Index of the target state to transition to.
    pub target_state: usize,
    /// Conditions that must ALL be true for this transition.
    pub conditions: Vec<TransitionCondition>,
    /// Blend duration for this transition (0 = use graph default).
    pub blend_duration: f32,
    /// Priority (higher = checked first). Default 0.
    pub priority: i32,
}

impl AnimTransition {
    pub fn new(target_state: usize) -> Self {
        Self {
            target_state,
            conditions: Vec::new(),
            blend_duration: 0.0,
            priority: 0,
        }
    }

    /// Add a condition to this transition.
    pub fn with_condition(mut self, condition: TransitionCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Set blend duration.
    pub fn with_blend_duration(mut self, duration: f32) -> Self {
        self.blend_duration = duration;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Check if ALL conditions are met.
    pub fn should_fire(&self, params: &AnimParameters, time_in_state: f32) -> bool {
        self.conditions.iter().all(|c| c.evaluate(params, time_in_state))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANIMATION GRAPH
// ═══════════════════════════════════════════════════════════════════════════════
// The main animation graph — a state machine with nodes and transitions.

/// A complete animation graph — a directed state machine.
///
/// Usage:
/// ```
/// let mut graph = AnimGraph::new("Character");
/// let idle = graph.add_state(AnimStateNode::new("Idle", 0));
/// let walk = graph.add_state(AnimStateNode::new("Walk", 1));
/// let run  = graph.add_state(AnimStateNode::new("Run", 2));
///
/// // Idle → Walk when speed > 0.5
/// graph.add_transition(idle, AnimTransition::new(walk)
///     .with_condition(TransitionCondition::FloatGreaterThan {
///         param: "speed".to_string(), threshold: 0.5 }));
///
/// // Walk → Run when speed > 3.0
/// graph.add_transition(walk, AnimTransition::new(run)
///     .with_condition(TransitionCondition::FloatGreaterThan {
///         param: "speed".to_string(), threshold: 3.0 }));
///
/// // Walk → Idle when speed < 0.5
/// graph.add_transition(walk, AnimTransition::new(idle)
///     .with_condition(TransitionCondition::FloatLessThan {
///         param: "speed".to_string(), threshold: 0.5 }));
///
/// graph.set_initial_state(idle);
/// ```
#[derive(Clone)]
pub struct AnimGraph {
    /// Human-readable name.
    pub name: String,
    /// All states in the graph.
    pub states: Vec<AnimStateNode>,
    /// Index of the currently active state.
    pub current_state: usize,
    /// Index of the previous state (for blending).
    pub previous_state: Option<usize>,
    /// Default blend duration for transitions (seconds).
    pub default_blend_duration: f32,
    /// Time spent in the current state (seconds).
    pub time_in_state: f32,
    /// Animation parameters (inputs to conditions).
    pub parameters: AnimParameters,
    /// Whether the graph is enabled.
    pub enabled: bool,
}

impl AnimGraph {
    /// Create a new empty animation graph.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            states: Vec::new(),
            current_state: 0,
            previous_state: None,
            default_blend_duration: 0.25,
            time_in_state: 0.0,
            parameters: AnimParameters::new(),
            enabled: true,
        }
    }

    /// Add a state to the graph and return its index.
    pub fn add_state(&mut self, state: AnimStateNode) -> usize {
        let idx = self.states.len();
        self.states.push(state);
        idx
    }

    /// Add a transition from one state to another.
    pub fn add_transition(&mut self, from_state: usize, transition: AnimTransition) {
        self.states[from_state].add_transition(transition);
    }

    /// Set the initial (entry) state.
    pub fn set_initial_state(&mut self, state: usize) {
        self.current_state = state;
        self.time_in_state = 0.0;
    }

    /// Set a parameter value.
    pub fn set_param(&mut self, name: &str, value: AnimParamValue) {
        self.parameters.params.insert(name.to_string(), value);
    }

    /// Set a float parameter.
    pub fn set_float(&mut self, name: &str, value: f32) {
        self.parameters.set_float(name, value);
    }

    /// Set a bool parameter.
    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.parameters.set_bool(name, value);
    }

    /// Set a string parameter.
    pub fn set_string(&mut self, name: &str, value: &str) {
        self.parameters.set_string(name, value);
    }

    /// Set an enum parameter.
    pub fn set_enum(&mut self, name: &str, value: u32) {
        self.parameters.set_enum(name, value);
    }

    /// Force the graph to a specific state (bypass conditions).
    pub fn force_state(&mut self, state: usize, blend_duration: f32) -> AnimGraphTransition {
        let old_state = self.current_state;
        self.previous_state = Some(old_state);
        self.current_state = state;
        self.time_in_state = 0.0;
        AnimGraphTransition {
            from_state: old_state,
            to_state: state,
            blend_duration: blend_duration.max(0.001),
            from_clip_index: self.states[old_state].clip_index,
            to_clip_index: self.states[state].clip_index,
        }
    }

    /// Evaluate the graph for one frame. Returns Some(AnimGraphTransition) if a
    /// state change occurred, None otherwise.
    ///
    /// This is the core logic:
    ///   1. Sort outgoing transitions by priority (highest first).
    ///   2. Check each transition's conditions against parameters.
    ///   3. If any transition fires, switch states and return the transition info.
    pub fn evaluate(&mut self, dt: f32) -> Option<AnimGraphTransition> {
        if !self.enabled {
            return None;
        }

        self.time_in_state += dt;

        // Gather and sort transitions by priority (highest first).
        let mut transitions: Vec<(usize, i32)> = self.states[self.current_state]
            .transitions
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.priority))
            .collect();
        transitions.sort_by(|a, b| b.1.cmp(&a.1)); // Highest priority first.

        // Check each transition.
        for (trans_idx, _) in &transitions {
            let transition = &self.states[self.current_state].transitions[*trans_idx];
            if transition.should_fire(&self.parameters, self.time_in_state) {
                // Transition fires!
                let target = transition.target_state;
                let blend = if transition.blend_duration > 0.0 {
                    transition.blend_duration
                } else {
                    self.default_blend_duration
                };

                let old_state = self.current_state;
                let from_clip = self.states[old_state].clip_index;
                let to_clip = self.states[target].clip_index;

                self.previous_state = Some(old_state);
                self.current_state = target;
                self.time_in_state = 0.0;

                return Some(AnimGraphTransition {
                    from_state: old_state,
                    to_state: target,
                    blend_duration: blend,
                    from_clip_index: from_clip,
                    to_clip_index: to_clip,
                });
            }
        }

        None
    }

    /// Get the current state's clip index.
    pub fn current_clip_index(&self) -> usize {
        self.states[self.current_state].clip_index
    }

    /// Get the current state's name.
    pub fn current_state_name(&self) -> &str {
        &self.states[self.current_state].name
    }
}

/// Information about a transition that just occurred.
#[derive(Clone, Debug)]
pub struct AnimGraphTransition {
    /// Index of the state we're transitioning from.
    pub from_state: usize,
    /// Index of the state we're transitioning to.
    pub to_state: usize,
    /// Blend duration in seconds.
    pub blend_duration: f32,
    /// Clip index of the source state.
    pub from_clip_index: usize,
    /// Clip index of the target state.
    pub to_clip_index: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANIMATION LAYERS
// ═══════════════════════════════════════════════════════════════════════════════
// Layers allow independent state machines for different body parts.
// For example: Lower body runs a Walk/Run graph, upper body runs an Attack graph.

/// A bone mask — defines which bones a layer affects.
///
/// Each bone has a weight (0.0 = not affected, 1.0 = fully affected).
/// Blending between layers uses these weights.
#[derive(Clone, Debug)]
pub struct BoneMask {
    /// Name for debugging.
    pub name: String,
    /// Per-bone weights. Key = bone name, Value = weight (0..1).
    pub weights: HashMap<String, f32>,
}

impl BoneMask {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            weights: HashMap::new(),
        }
    }

    /// Set weight for a bone.
    pub fn set_weight(&mut self, bone_name: &str, weight: f32) {
        self.weights.insert(bone_name.to_string(), weight.clamp(0.0, 1.0));
    }

    /// Get weight for a bone (default 0.0 if not set).
    pub fn get_weight(&self, bone_name: &str) -> f32 {
        self.weights.get(bone_name).copied().unwrap_or(0.0)
    }

    /// Create a lower body mask (hips + legs = 1.0, spine+arms = 0.0).
    pub fn lower_body() -> Self {
        let mut mask = Self::new("Lower Body");
        for bone in &["Hips", "LeftThigh", "LeftShin", "LeftFoot",
                       "RightThigh", "RightShin", "RightFoot"] {
            mask.set_weight(bone, 1.0);
        }
        mask
    }

    /// Create an upper body mask (spine + arms = 1.0, legs = 0.0).
    pub fn upper_body() -> Self {
        let mut mask = Self::new("Upper Body");
        for bone in &["Spine01", "Spine02", "Neck", "Head",
                       "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
                       "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand"] {
            mask.set_weight(bone, 1.0);
        }
        mask
    }

    /// Create a full body mask (all bones = 1.0).
    pub fn full_body() -> Self {
        Self::new("Full Body")
    }
}

/// An animation layer — an independent state machine that affects a subset of bones.
///
/// Example:
///   - Layer 0: "Full Body" — Walk/Run/Idle state machine, full body mask
///   - Layer 1: "Upper Body" — Attack/Reload state machine, upper body mask
///
/// The final pose is computed by blending layers from bottom (0) to top,
/// using bone masks to blend only the affected bones.
#[derive(Clone)]
pub struct AnimLayer {
    /// Name for debugging.
    pub name: String,
    /// The animation graph for this layer.
    pub graph: AnimGraph,
    /// Which bones this layer affects.
    pub bone_mask: BoneMask,
    /// Blend weight for this layer (0.0 = inactive, 1.0 = fully active).
    pub weight: f32,
    /// Blend mode: Replace (override lower layers) or Additive (add to lower layers).
    pub blend_mode: AnimLayerBlendMode,
    /// Is this layer enabled?
    pub enabled: bool,
}

/// How a layer blends with layers below it.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum AnimLayerBlendMode {
    /// Override — this layer's pose replaces lower layers for masked bones.
    Replace,
    /// Additive — this layer adds to the lower layers' pose.
    Additive,
}

impl AnimLayer {
    pub fn new(name: &str, graph: AnimGraph, bone_mask: BoneMask) -> Self {
        Self {
            name: name.to_string(),
            graph,
            bone_mask,
            weight: 1.0,
            blend_mode: AnimLayerBlendMode::Replace,
            enabled: true,
        }
    }

    /// Create a simple full-body layer.
    pub fn full_body(name: &str, graph: AnimGraph) -> Self {
        Self::new(name, graph, BoneMask::full_body())
    }

    /// Create an upper body layer.
    pub fn upper_body(name: &str, graph: AnimGraph) -> Self {
        Self::new(name, graph, BoneMask::upper_body())
    }

    /// Create a lower body layer.
    pub fn lower_body(name: &str, graph: AnimGraph) -> Self {
        Self::new(name, graph, BoneMask::lower_body())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANIMATION GRAPH COMPONENT (ECS)
// ═══════════════════════════════════════════════════════════════════════════════
// Attach this to an entity to give it a full node-based animation setup.

/// ECS component for node-based animation.
///
/// Contains one or more `AnimLayer`s, each with its own state machine.
/// The bottom layer (index 0) is the base; higher layers blend on top.
///
/// Typical setup:
///   - Layer 0: Full body Walk/Run/Idle
///   - Layer 1: Upper body Attack (with Attack trigger)
#[derive(Clone)]
pub struct AnimGraphComponent {
    /// Animation layers, ordered bottom to top.
    pub layers: Vec<AnimLayer>,
    /// Shared parameters for all layers (convenience — layers also have their own).
    pub parameters: AnimParameters,
}

impl AnimGraphComponent {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            parameters: AnimParameters::new(),
        }
    }

    /// Add a layer.
    pub fn add_layer(&mut self, layer: AnimLayer) {
        self.layers.push(layer);
    }

    /// Get the base layer (index 0) for parameter access.
    pub fn base_layer(&self) -> Option<&AnimLayer> {
        self.layers.first()
    }

    /// Get a mutable reference to a layer by name.
    pub fn layer_mut(&mut self, name: &str) -> Option<&mut AnimLayer> {
        self.layers.iter_mut().find(|l| l.name == name)
    }

    /// Set a parameter on the shared parameter store (synced to all layers).
    pub fn set_param(&mut self, name: &str, value: AnimParamValue) {
        self.parameters.params.insert(name.to_string(), value.clone());
        // Sync to all layer graphs.
        for layer in &mut self.layers {
            layer.graph.parameters.params.insert(name.to_string(), value.clone());
        }
    }

    /// Set a float parameter.
    pub fn set_float(&mut self, name: &str, value: f32) {
        self.set_param(name, AnimParamValue::Float(value));
    }

    /// Set a bool parameter.
    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.set_param(name, AnimParamValue::Bool(value));
    }

    /// Set a string parameter.
    pub fn set_string(&mut self, name: &str, value: &str) {
        self.set_param(name, AnimParamValue::String(value.to_string()));
    }

    /// Force a state on a specific layer (for debugging/teleport).
    pub fn force_layer_state(&mut self, layer_name: &str, state: usize, blend: f32) -> Option<AnimGraphTransition> {
        self.layers.iter_mut()
            .find(|l| l.name == layer_name)
            .map(|l| l.graph.force_state(state, blend))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANIM GRAPH SYSTEM (ECS)
// ═══════════════════════════════════════════════════════════════════════════════
// Per-frame system that evaluates all animation graphs and produces blended poses.

use crate::animation::blending::BlendedPose;

/// Per-frame system that evaluates animation graphs.
///
/// For each entity with AnimGraphComponent + AnimationClips + SkeletalAnimator:
///   1. Evaluate each layer's graph → get active clip per layer.
///   2. For each layer, evaluate the active clip at the current time.
///   3. Blend layers together using bone masks.
///   4. Compute final joint matrices.
///
/// This system REPLACES the simple animation_blending_system for entities
/// that use node-based animation. Entities without AnimGraphComponent still
/// use the direct BT→blackboard→SkeletalAnimator path.
pub fn anim_graph_system(
    world: &mut hecs::World,
    dt: f32,
) {
    // ── Auto-sync BT blackboard to AnimGraph parameters ──────────────────
    // The AI behavior tree writes decision state to AiAgent.blackboard
    // (e.g., "speed", "is_grounded", "is_attacking", "vertical_speed").
    // This pass pushes those values into AnimGraphComponent's shared
    // parameter store so graph transitions can react to AI state on the
    // same frame. Only keys that exist in the blackboard are synced --
    // missing keys are silently skipped (no error).
    {
        // Collect entity IDs first to avoid borrow conflicts between
        // the immutable AiAgent read and the mutable AnimGraphComponent write.
        let sync_entities: Vec<hecs::Entity> = world
            .query::<(hecs::Entity, &AiAgent, &AnimGraphComponent)>()
            .iter()
            .map(|(e, _, _)| e)
            .collect();

        for entity in sync_entities {
            // Phase 1: Read all blackboard values in an inner scope so the
            // immutable Ref<AiAgent> is dropped before Phase 2 borrows mutably.
            let (speed, is_grounded, is_attacking, vertical_speed) = {
                let Ok(agent) = world.get::<&AiAgent>(entity) else { continue };
                (
                    agent.blackboard.get_float("speed"),
                    agent.blackboard.get_bool("is_grounded"),
                    agent.blackboard.get_bool("is_attacking"),
                    agent.blackboard.get_float("vertical_speed"),
                )
                // agent (Ref<AiAgent>) dropped here — world borrow released.
            };

            // Phase 2: Apply values to AnimGraphComponent (mutable borrow).
            if let Ok(mut ag) = world.get::<&mut AnimGraphComponent>(entity) {
                // Float parameter: "speed" (locomotion speed scalar).
                if let Some(v) = speed {
                    ag.set_float("speed", v);
                }
                // Bool parameter: "is_grounded" (ground contact state).
                if let Some(v) = is_grounded {
                    ag.set_bool("is_grounded", v);
                }
                // Bool parameter: "is_attacking" (melee/ranged attack flag).
                if let Some(v) = is_attacking {
                    ag.set_bool("is_attacking", v);
                }
                // Float parameter: "vertical_speed" (Y velocity for jump/fall).
                if let Some(v) = vertical_speed {
                    ag.set_float("vertical_speed", v);
                }
            }
            // Entity has no matching blackboard keys -- skip silently.
        }
    }

    // Collect entities to process (avoid borrow conflicts).
    let entities: Vec<hecs::Entity> = world
        .query::<(hecs::Entity, &AnimGraphComponent)>()
        .iter()
        .map(|(e, _)| e)
        .collect();

    for entity in entities {
        // Evaluate all layer graphs.
        let transitions: Vec<Option<AnimGraphTransition>> = {
            let Ok(mut ag) = world.get::<&mut AnimGraphComponent>(entity) else { continue };
            ag.layers.iter_mut()
                .filter(|l| l.enabled)
                .map(|l| l.graph.evaluate(dt))
                .collect()
        };

        // If any layer transitioned, update the SkeletalAnimator.
        if let Some(Some(t)) = transitions.first() {
            if let Ok(mut animator) = world.get::<&mut SkeletalAnimator>(entity) {
                if let Ok(ag) = world.get::<&AnimGraphComponent>(entity) {
                    // Use the base layer's current state clip.
                    if let Some(base) = ag.layers.first() {
                        let clip_idx = base.graph.states[base.graph.current_state].clip_index;
                        animator.play_with_duration(clip_idx, t.blend_duration);
                    }
                }
            }
        }

        // Evaluate per-layer clips and blend them.
        let blended = {
            let Ok(ag) = world.get::<&AnimGraphComponent>(entity) else { continue };
            let Ok(anim_clips) = world.get::<&crate::animation::blending::AnimationClips>(entity) else { continue };
            let Ok(animator) = world.get::<&SkeletalAnimator>(entity) else { continue };

            let bone_count = anim_clips.hierarchy.bone_count();

            // Start with rest pose.
            let mut result_locals: Vec<TransformKeyframe> = (0..bone_count)
                .map(|_| TransformKeyframe::identity(0.0))
                .collect();

            // Accumulate layers bottom to top.
            for layer in &ag.layers {
                if !layer.enabled || layer.weight <= 0.0 {
                    continue;
                }

                let graph = &layer.graph;
                if graph.states.is_empty() { continue; }

                let state = &graph.states[graph.current_state];
                if state.clip_index >= anim_clips.clips.len() { continue; }

                let clip = &anim_clips.clips[state.clip_index];
                let locals = clip.evaluate(animator.time);

                // Blend this layer into result using bone mask.
                for bone_idx in 0..bone_count.min(locals.len()) {
                    let bone_name = if bone_idx < anim_clips.hierarchy.bones.len() {
                        &anim_clips.hierarchy.bones[bone_idx].name
                    } else {
                        continue;
                    };

                    let mask_weight = layer.bone_mask.get_weight(bone_name) * layer.weight;
                    if mask_weight <= 0.001 {
                        continue;
                    }

                    let src = &locals[bone_idx];
                    let dst = &mut result_locals[bone_idx];

                    match layer.blend_mode {
                        AnimLayerBlendMode::Replace => {
                            dst.position = dst.position.lerp(src.position, mask_weight);
                            dst.rotation = dst.rotation.slerp(src.rotation, mask_weight);
                            dst.scale = dst.scale.lerp(src.scale, mask_weight);
                        }
                        AnimLayerBlendMode::Additive => {
                            dst.position += src.position * mask_weight;
                            let add_rot = src.rotation;
                            dst.rotation = dst.rotation.slerp(dst.rotation * add_rot, mask_weight);
                            dst.scale += (src.scale - Vec3::ONE) * mask_weight;
                        }
                    }
                }
            }

            // Compute joint matrices.
            let joint_matrices = compute_joint_matrices(
                &anim_clips.hierarchy,
                &result_locals,
                &anim_clips.inverse_bind_poses,
            );

            BlendedPose {
                joint_matrices,
                is_blending: animator.is_blending(),
                blend_weight: animator.blend_weight,
            }
        };

        // Write the blended pose.
        if let Ok(mut pose) = world.get::<&mut BlendedPose>(entity) {
            *pose = blended;
        } else {
            world.insert_one(entity, blended).ok();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: Build common animation graphs
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a standard humanoid locomotion graph (Idle → Walk → Run).
///
/// Parameters:
///   - "speed" (f32): entity movement speed
///
/// Transitions:
///   - Idle → Walk: speed > 0.5
///   - Walk → Run: speed > 3.0
///   - Walk → Idle: speed < 0.5
///   - Run → Walk: speed < 3.0
pub fn build_locomotion_graph(
    idle_clip: usize,
    walk_clip: usize,
    run_clip: usize,
) -> AnimGraph {
    let mut graph = AnimGraph::new("Locomotion");

    let idle = graph.add_state(AnimStateNode::new("Idle", idle_clip));
    let walk = graph.add_state(AnimStateNode::new("Walk", walk_clip));
    let run = graph.add_state(AnimStateNode::new("Run", run_clip));

    // Idle → Walk
    graph.add_transition(idle, AnimTransition::new(walk)
        .with_condition(TransitionCondition::FloatGreaterThan {
            param: "speed".to_string(),
            threshold: 0.5,
        }));

    // Walk → Run
    graph.add_transition(walk, AnimTransition::new(run)
        .with_condition(TransitionCondition::FloatGreaterThan {
            param: "speed".to_string(),
            threshold: 3.0,
        }));

    // Walk → Idle
    graph.add_transition(walk, AnimTransition::new(idle)
        .with_condition(TransitionCondition::FloatLessThan {
            param: "speed".to_string(),
            threshold: 0.5,
        }));

    // Run → Walk
    graph.add_transition(run, AnimTransition::new(walk)
        .with_condition(TransitionCondition::FloatLessThan {
            param: "speed".to_string(),
            threshold: 3.0,
        }));

    graph.set_initial_state(idle);
    graph
}

/// Build a jump graph (Idle → JumpLaunch → JumpFall → JumpLand → Idle).
///
/// Parameters:
///   - "is_grounded" (bool): true when on ground
///   - "vertical_speed" (f32): upward velocity
pub fn build_jump_graph(
    idle_clip: usize,
    launch_clip: usize,
    fall_clip: usize,
    land_clip: usize,
) -> AnimGraph {
    let mut graph = AnimGraph::new("Jump");

    let idle = graph.add_state(AnimStateNode::new("Idle", idle_clip));
    let launch = graph.add_state(AnimStateNode::new("JumpLaunch", launch_clip));
    let fall = graph.add_state(AnimStateNode::new("JumpFall", fall_clip));
    let land = graph.add_state(AnimStateNode::new("JumpLand", land_clip));

    // Idle → Launch: not grounded
    graph.add_transition(idle, AnimTransition::new(launch)
        .with_condition(TransitionCondition::BoolEquals {
            param: "is_grounded".to_string(),
            expected: false,
        }));

    // Launch → Fall: after 0.3s OR vertical_speed < 0
    graph.add_transition(launch, AnimTransition::new(fall)
        .with_condition(TransitionCondition::FloatLessThan {
            param: "vertical_speed".to_string(),
            threshold: 0.0,
        }));

    // Fall → Land: grounded
    graph.add_transition(fall, AnimTransition::new(land)
        .with_condition(TransitionCondition::BoolEquals {
            param: "is_grounded".to_string(),
            expected: true,
        }));

    // Land → Idle: after 0.4s (land animation plays then transitions)
    graph.add_transition(land, AnimTransition::new(idle)
        .with_condition(TransitionCondition::TimeInStateGreaterThan {
            seconds: 0.4,
        }));

    graph.set_initial_state(idle);
    graph
}

/// Build a combat graph (Idle → Attack → Idle).
///
/// Parameters:
///   - "is_attacking" (bool): true when attack is active
pub fn build_combat_graph(
    idle_clip: usize,
    attack_clip: usize,
) -> AnimGraph {
    let mut graph = AnimGraph::new("Combat");

    let idle = graph.add_state(AnimStateNode::new("Idle", idle_clip));
    let attack = graph.add_state(AnimStateNode::new("Attack", attack_clip));

    // Idle → Attack
    graph.add_transition(idle, AnimTransition::new(attack)
        .with_condition(TransitionCondition::BoolEquals {
            param: "is_attacking".to_string(),
            expected: true,
        }));

    // Attack → Idle: after 0.6s (attack clip duration) OR not attacking
    graph.add_transition(attack, AnimTransition::new(idle)
        .with_condition(TransitionCondition::BoolEquals {
            param: "is_attacking".to_string(),
            expected: false,
        })
        .with_condition(TransitionCondition::TimeInStateGreaterThan {
            seconds: 0.5,
        }));

    graph.set_initial_state(idle);
    graph
}

// ═══════════════════════════════════════════════════════════════════════════════
// BUILDER PATTERN
// ═══════════════════════════════════════════════════════════════════════════════

/// Fluent builder for constructing animation graphs from code.
///
/// Usage:
/// ```
/// let graph = AnimGraphBuilder::new("Character")
///     .state("Idle", 0)
///     .state("Walk", 1)
///     .state("Run", 2)
///     .transition("Idle", "Walk")
///         .condition(TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 0.5 })
///         .done()
///     .transition("Walk", "Run")
///         .condition(TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 3.0 })
///         .done()
///     .transition("Walk", "Idle")
///         .condition(TransitionCondition::FloatLessThan { param: "speed".into(), threshold: 0.5 })
///         .done()
///     .transition("Run", "Walk")
///         .condition(TransitionCondition::FloatLessThan { param: "speed".into(), threshold: 3.0 })
///         .done()
///     .initial("Idle")
///     .build();
/// ```
pub struct AnimGraphBuilder {
    graph: AnimGraph,
    state_names: HashMap<String, usize>,
    current_transition: Option<TransitionBuilder>,
}

struct TransitionBuilder {
    from_state: usize,
    target_state: usize,
    conditions: Vec<TransitionCondition>,
    blend_duration: f32,
    priority: i32,
}

impl AnimGraphBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            graph: AnimGraph::new(name),
            state_names: HashMap::new(),
            current_transition: None,
        }
    }

    /// Add a state with a clip index.
    pub fn state(mut self, name: &str, clip_index: usize) -> Self {
        let idx = self.graph.add_state(AnimStateNode::new(name, clip_index));
        self.state_names.insert(name.to_string(), idx);
        self
    }

    /// Start building a transition from state A to state B.
    pub fn transition(mut self, from: &str, to: &str) -> Self {
        // Finalize any previous transition.
        self.finalize_transition();

        let from_idx = *self.state_names.get(from).expect("Unknown source state");
        let to_idx = *self.state_names.get(to).expect("Unknown target state");

        self.current_transition = Some(TransitionBuilder {
            from_state: from_idx,
            target_state: to_idx,
            conditions: Vec::new(),
            blend_duration: 0.0,
            priority: 0,
        });
        self
    }

    /// Add a condition to the current transition being built.
    pub fn condition(mut self, cond: TransitionCondition) -> Self {
        if let Some(ref mut tb) = self.current_transition {
            tb.conditions.push(cond);
        }
        self
    }

    /// Finalize the current transition and return to graph builder.
    pub fn done(mut self) -> Self {
        self.finalize_transition();
        self
    }

    /// Set the initial state.
    pub fn initial(mut self, name: &str) -> Self {
        let idx = *self.state_names.get(name).expect("Unknown initial state");
        self.graph.set_initial_state(idx);
        self
    }

    /// Build the final AnimGraph.
    pub fn build(mut self) -> AnimGraph {
        self.finalize_transition();
        self.graph
    }

    fn finalize_transition(&mut self) {
        if let Some(tb) = self.current_transition.take() {
            let mut transition = AnimTransition::new(tb.target_state);
            transition.conditions = tb.conditions;
            transition.blend_duration = tb.blend_duration;
            transition.priority = tb.priority;
            self.graph.add_transition(tb.from_state, transition);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anim_param_float() {
        let mut params = AnimParameters::new();
        params.set_float("speed", 2.5);
        assert_eq!(params.get_float("speed"), 2.5);
    }

    #[test]
    fn anim_param_bool() {
        let mut params = AnimParameters::new();
        params.set_bool("grounded", true);
        assert!(params.get_bool("grounded"));
        assert!(!params.get_bool("missing"));
    }

    #[test]
    fn anim_param_string() {
        let mut params = AnimParameters::new();
        params.set_string("ai_state", "attack");
        assert_eq!(params.get_string("ai_state"), "attack");
    }

    #[test]
    fn condition_float_greater_than() {
        let mut params = AnimParameters::new();
        params.set_float("speed", 3.0);
        let cond = TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 2.0 };
        assert!(cond.evaluate(&params, 0.0));
        let cond2 = TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 4.0 };
        assert!(!cond2.evaluate(&params, 0.0));
    }

    #[test]
    fn condition_bool_equals() {
        let mut params = AnimParameters::new();
        params.set_bool("attacking", true);
        let cond = TransitionCondition::BoolEquals { param: "attacking".into(), expected: true };
        assert!(cond.evaluate(&params, 0.0));
        let cond2 = TransitionCondition::BoolEquals { param: "attacking".into(), expected: false };
        assert!(!cond2.evaluate(&params, 0.0));
    }

    #[test]
    fn condition_time_in_state() {
        let params = AnimParameters::new();
        let cond = TransitionCondition::TimeInStateGreaterThan { seconds: 1.0 };
        assert!(!cond.evaluate(&params, 0.5));
        assert!(cond.evaluate(&params, 1.5));
    }

    #[test]
    fn graph_initial_state() {
        let mut graph = AnimGraph::new("Test");
        let idle = graph.add_state(AnimStateNode::new("Idle", 0));
        let walk = graph.add_state(AnimStateNode::new("Walk", 1));
        graph.add_transition(idle, AnimTransition::new(walk)
            .with_condition(TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 0.5 }));
        graph.set_initial_state(idle);

        assert_eq!(graph.current_state, 0);
        assert_eq!(graph.current_state, idle);
    }

    #[test]
    fn graph_transition_fires() {
        let mut graph = AnimGraph::new("Test");
        let idle = graph.add_state(AnimStateNode::new("Idle", 0));
        let walk = graph.add_state(AnimStateNode::new("Walk", 1));
        graph.add_transition(idle, AnimTransition::new(walk)
            .with_condition(TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 0.5 }));
        graph.set_initial_state(idle);

        // Speed is 0 — should stay in Idle.
        graph.set_float("speed", 0.0);
        assert!(graph.evaluate(0.01).is_none());
        assert_eq!(graph.current_state, idle);

        // Speed > 0.5 — should transition to Walk.
        graph.set_float("speed", 1.0);
        let result = graph.evaluate(0.01);
        assert!(result.is_some());
        assert_eq!(graph.current_state, walk);
    }

    #[test]
    fn graph_transition_does_not_fire() {
        let mut graph = AnimGraph::new("Test");
        let idle = graph.add_state(AnimStateNode::new("Idle", 0));
        let walk = graph.add_state(AnimStateNode::new("Walk", 1));
        graph.add_transition(idle, AnimTransition::new(walk)
            .with_condition(TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 0.5 }));
        graph.set_initial_state(idle);

        // Speed is 0.3 — should stay in Idle.
        graph.set_float("speed", 0.3);
        assert!(graph.evaluate(0.01).is_none());
        assert_eq!(graph.current_state, idle);
    }

    #[test]
    fn graph_multiple_conditions() {
        let mut graph = AnimGraph::new("Test");
        let idle = graph.add_state(AnimStateNode::new("Idle", 0));
        let attack = graph.add_state(AnimStateNode::new("Attack", 1));

        // Need BOTH conditions: speed < 1.0 AND is_attacking == true
        graph.add_transition(idle, AnimTransition::new(attack)
            .with_condition(TransitionCondition::FloatLessThan { param: "speed".into(), threshold: 1.0 })
            .with_condition(TransitionCondition::BoolEquals { param: "is_attacking".into(), expected: true }));
        graph.set_initial_state(idle);

        // Only attacking — speed too high.
        graph.set_float("speed", 2.0);
        graph.set_bool("is_attacking", true);
        assert!(graph.evaluate(0.01).is_none());

        // Both conditions met.
        graph.set_float("speed", 0.5);
        assert!(graph.evaluate(0.01).is_some());
        assert_eq!(graph.current_state, attack);
    }

    #[test]
    fn graph_priority() {
        let mut graph = AnimGraph::new("Test");
        let idle = graph.add_state(AnimStateNode::new("Idle", 0));
        let walk = graph.add_state(AnimStateNode::new("Walk", 1));
        let run = graph.add_state(AnimStateNode::new("Run", 2));

        // Low priority: walk when speed > 0.5
        graph.add_transition(idle, AnimTransition::new(walk)
            .with_condition(TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 0.5 })
            .with_priority(0));

        // High priority: run when speed > 3.0
        graph.add_transition(idle, AnimTransition::new(run)
            .with_condition(TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 3.0 })
            .with_priority(10));

        graph.set_initial_state(idle);
        graph.set_float("speed", 5.0);

        // Should go to Run (higher priority).
        graph.evaluate(0.01);
        assert_eq!(graph.current_state, run);
    }

    #[test]
    fn locomotion_graph_build() {
        let graph = build_locomotion_graph(0, 1, 2);
        assert_eq!(graph.states.len(), 3);
        assert_eq!(graph.current_state, 0); // Starts in Idle
    }

    #[test]
    fn jump_graph_build() {
        let graph = build_jump_graph(0, 1, 2, 3);
        assert_eq!(graph.states.len(), 4);
        assert_eq!(graph.current_state, 0);
    }

    #[test]
    fn combat_graph_build() {
        let graph = build_combat_graph(0, 1);
        assert_eq!(graph.states.len(), 2);
        assert_eq!(graph.current_state, 0);
    }

    #[test]
    fn bone_mask_lower_body() {
        let mask = BoneMask::lower_body();
        assert_eq!(mask.get_weight("Hips"), 1.0);
        assert_eq!(mask.get_weight("LeftThigh"), 1.0);
        assert_eq!(mask.get_weight("Head"), 0.0);
    }

    #[test]
    fn bone_mask_upper_body() {
        let mask = BoneMask::upper_body();
        assert_eq!(mask.get_weight("Spine01"), 1.0);
        assert_eq!(mask.get_weight("LeftHand"), 1.0);
        assert_eq!(mask.get_weight("LeftFoot"), 0.0);
    }

    #[test]
    fn anim_layer_creation() {
        let graph = AnimGraph::new("Attack");
        let layer = AnimLayer::upper_body("Upper", graph);
        assert_eq!(layer.name, "Upper");
        assert_eq!(layer.bone_mask.get_weight("Spine01"), 1.0);
        assert_eq!(layer.bone_mask.get_weight("LeftFoot"), 0.0);
    }

    #[test]
    fn anim_graph_component() {
        let mut comp = AnimGraphComponent::new();
        let mut graph = AnimGraph::new("Locomotion");
        let idle = graph.add_state(AnimStateNode::new("Idle", 0));
        graph.set_initial_state(idle);

        comp.add_layer(AnimLayer::full_body("Base", graph));
        comp.set_float("speed", 1.5);

        assert_eq!(comp.layers.len(), 1);
        assert_eq!(comp.layers[0].graph.parameters.get_float("speed"), 1.5);
    }

    #[test]
    fn builder_pattern() {
        let graph = AnimGraphBuilder::new("Character")
            .state("Idle", 0)
            .state("Walk", 1)
            .state("Run", 2)
            .transition("Idle", "Walk")
                .condition(TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 0.5 })
                .done()
            .transition("Walk", "Run")
                .condition(TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 3.0 })
                .done()
            .transition("Walk", "Idle")
                .condition(TransitionCondition::FloatLessThan { param: "speed".into(), threshold: 0.5 })
                .done()
            .transition("Run", "Walk")
                .condition(TransitionCondition::FloatLessThan { param: "speed".into(), threshold: 3.0 })
                .done()
            .initial("Idle")
            .build();

        assert_eq!(graph.states.len(), 3);
        assert_eq!(graph.states[0].name, "Idle");
        assert_eq!(graph.states[1].name, "Walk");
        assert_eq!(graph.states[2].name, "Run");

        // Test transitions.
        let mut g = graph;
        g.set_float("speed", 1.0);
        g.evaluate(0.01); // Idle → Walk
        assert_eq!(g.current_state, 1);

        g.set_float("speed", 5.0);
        g.evaluate(0.01); // Walk → Run
        assert_eq!(g.current_state, 2);
    }

    #[test]
    fn graph_force_state() {
        let mut graph = AnimGraph::new("Test");
        let idle = graph.add_state(AnimStateNode::new("Idle", 0));
        let attack = graph.add_state(AnimStateNode::new("Attack", 1));
        graph.set_initial_state(idle);

        let t = graph.force_state(attack, 0.1);
        assert_eq!(t.from_state, idle);
        assert_eq!(t.to_state, attack);
        assert_eq!(graph.current_state, attack);
    }

    #[test]
    fn graph_disabled_no_transition() {
        let mut graph = AnimGraph::new("Test");
        let idle = graph.add_state(AnimStateNode::new("Idle", 0));
        let walk = graph.add_state(AnimStateNode::new("Walk", 1));
        graph.add_transition(idle, AnimTransition::new(walk)
            .with_condition(TransitionCondition::FloatGreaterThan { param: "speed".into(), threshold: 0.5 }));
        graph.set_initial_state(idle);
        graph.enabled = false;

        graph.set_float("speed", 5.0);
        assert!(graph.evaluate(0.01).is_none());
        assert_eq!(graph.current_state, idle);
    }
}
