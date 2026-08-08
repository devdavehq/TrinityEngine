use crate::ai::blackboard::{Blackboard, BlackboardValue};
use crate::ai::components::AiAgent;
use crate::components::Position;
use crate::navigation::NavGrid;
use crate::navmesh::NavMesh;

// ═══════════════════════════════════════════════════════════════════════════════
// BEHAVIOR TREE FRAMEWORK
// ═══════════════════════════════════════════════════════════════════════════════
//
// A behavior tree (BT) is a hierarchical decision-making structure used in
// game AI.  Every tick, the tree is evaluated from the root downward.  Each
// node returns one of three statuses:
//
//   Success — the node completed its task successfully.
//   Failure — the node could not complete its task.
//   Running — the node is still working (will be resumed next tick).
//
// Nodes are organized into four categories:
//
//   Composites  — control flow (Sequence, Selector, Parallel).
//   Decorators  — modify a single child's behavior (Inverter, Repeater, …).
//   Leaves      — actual actions or conditions (MoveTo, Wait, …).
//   The root    — typically a composite that orchestrates the tree.
//
// ── Status ──────────────────────────────────────────────────────────────────
// Every tick returns this enum.  It is the fundamental communication channel
// between parent and child nodes.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Success,
    Failure,
    Running,
}

// ── BTContext ───────────────────────────────────────────────────────────────
// Passed to every node during a tick.  Contains everything a node might need:
// the entity being controlled, mutable access to the ECS world, delta time,
// simulation clock, navigation grid, and the per-agent blackboard.
//
// Nodes use the blackboard to share state: e.g. a "FindTarget" leaf writes
// the target position, and a subsequent "MoveTo" leaf reads it.

pub struct BTContext<'a> {
    pub entity: hecs::Entity,
    pub world: &'a mut hecs::World,
    pub dt: f32,
    pub time_s: f32,
    pub nav_grid: &'a NavGrid,
    /// Optional triangle navmesh. Movement nodes prefer it for paths and fall
    /// back to `nav_grid` when it is absent or cannot route.
    pub navmesh: Option<&'a NavMesh>,
    pub blackboard: &'a mut Blackboard,
}

// ── BehaviorNode trait ──────────────────────────────────────────────────────
// Every node in the tree implements this trait.  `tick()` is called once per
// evaluation and returns a Status.  `name()` is used for debugging/profiling.

pub trait BehaviorNode: Send + Sync {
    fn tick(&mut self, ctx: &mut BTContext) -> Status;
    fn name(&self) -> &str;
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOSITE NODES
// ═══════════════════════════════════════════════════════════════════════════════
// Composites have multiple children and control the order/conditions under
// which they are evaluated.

// ── Sequence ────────────────────────────────────────────────────────────────
// Runs children left-to-right.  If any child returns Failure, the Sequence
// immediately returns Failure.  If all children return Success, the Sequence
// returns Success.  If a child returns Running, the Sequence suspends and
// resumes from that child on the next tick.

pub struct Sequence {
    children: Vec<Box<dyn BehaviorNode>>,
    name: String,
}

impl Sequence {
    pub fn new(name: &str, children: Vec<Box<dyn BehaviorNode>>) -> Self {
        Self {
            children,
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Sequence {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        for child in &mut self.children {
            match child.tick(ctx) {
                // A failing child aborts the entire sequence.
                Status::Failure => return Status::Failure,
                // A running child means the sequence is still in progress;
                // we'll resume from this child next tick.
                Status::Running => return Status::Running,
                // Success: continue to the next child.
                Status::Success => {}
            }
        }
        // All children succeeded → the sequence succeeds.
        Status::Success
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Selector ────────────────────────────────────────────────────────────────
// Runs children left-to-right.  If any child returns Success, the Selector
// immediately returns Success.  If all children return Failure, the Selector
// returns Failure.  If a child returns Running, the Selector suspends and
// resumes from that child on the next tick.
//
// Think of Selector as "try each option until one works."

pub struct Selector {
    children: Vec<Box<dyn BehaviorNode>>,
    name: String,
}

impl Selector {
    pub fn new(name: &str, children: Vec<Box<dyn BehaviorNode>>) -> Self {
        Self {
            children,
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Selector {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        for child in &mut self.children {
            match child.tick(ctx) {
                // A succeeding child means this option worked → done.
                Status::Success => return Status::Success,
                // A running child means we're mid-attempt; resume next tick.
                Status::Running => return Status::Running,
                // Failure: try the next child.
                Status::Failure => {}
            }
        }
        // All children failed → the selector fails.
        Status::Failure
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Parallel ────────────────────────────────────────────────────────────────
// Ticks ALL children every frame (not short-circuiting).  Two configurable
// thresholds determine the overall outcome:
//
//   success_threshold — if this many children succeed, Parallel returns Success.
//   failure_threshold — if this many children fail, Parallel returns Failure.
//
// If neither threshold is met, Parallel returns Running.
//
// This is useful for behaviors that run concurrently, e.g. "move toward target"
// AND "play walk animation" at the same time.

pub struct Parallel {
    children: Vec<Box<dyn BehaviorNode>>,
    success_threshold: usize,
    failure_threshold: usize,
    name: String,
}

impl Parallel {
    pub fn new(
        name: &str,
        children: Vec<Box<dyn BehaviorNode>>,
        success_threshold: usize,
        failure_threshold: usize,
    ) -> Self {
        Self {
            children,
            success_threshold,
            failure_threshold,
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Parallel {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        let mut successes = 0;
        let mut failures = 0;

        // Tick every child — Parallel never short-circuits.
        for child in &mut self.children {
            match child.tick(ctx) {
                Status::Success => successes += 1,
                Status::Failure => failures += 1,
                Status::Running => {}
            }
        }

        // Check failure first — it's the more urgent condition.
        if failures >= self.failure_threshold {
            return Status::Failure;
        }
        if successes >= self.success_threshold {
            return Status::Success;
        }
        // Neither threshold met → still in progress.
        Status::Running
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── RandomSelector ──────────────────────────────────────────────────────────
// Picks one random child each time it is ticked and runs only that child.
// If the selected child returns Running, it continues to be ticked until it
// completes.  Once it completes (Success or Failure), the next tick picks a
// new random child.

pub struct RandomSelector {
    children: Vec<Box<dyn BehaviorNode>>,
    current_index: Option<usize>,
    name: String,
}

impl RandomSelector {
    pub fn new(name: &str, children: Vec<Box<dyn BehaviorNode>>) -> Self {
        Self {
            children,
            current_index: None,
            name: name.to_string(),
        }
    }

    fn pick_random_index(&self) -> usize {
        // Simple LCG-based random to avoid pulling in the `rand` crate.
        // Adequate for AI selection — not cryptographic.
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let s = RandomState::new();
        let mut hasher = s.build_hasher();
        hasher.write_u64(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64);
        (hasher.finish() as usize) % self.children.len()
    }
}

impl BehaviorNode for RandomSelector {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        if self.children.is_empty() {
            return Status::Failure;
        }

        // If we don't have a selected child, pick one at random.
        if self.current_index.is_none() {
            self.current_index = Some(self.pick_random_index());
        }

        let Some(idx) = self.current_index else { return Status::Failure; };
        let status = self.children[idx].tick(ctx);

        // If the child completed (not Running), deselect so next tick picks
        // a new random child.
        if status != Status::Running {
            self.current_index = None;
        }

        status
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DECORATOR NODES
// ═══════════════════════════════════════════════════════════════════════════════
// Decorators wrap a single child and modify its behavior (invert, repeat, etc).

// ── Inverter ────────────────────────────────────────────────────────────────
// Flips Success ↔ Failure.  Running passes through unchanged.
//
//   Child returns Success → Inverter returns Failure.
//   Child returns Failure → Inverter returns Success.
//   Child returns Running → Inverter returns Running.

pub struct Inverter {
    child: Box<dyn BehaviorNode>,
    name: String,
}

impl Inverter {
    pub fn new(name: &str, child: Box<dyn BehaviorNode>) -> Self {
        Self {
            child,
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Inverter {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        match self.child.tick(ctx) {
            Status::Success => Status::Failure,
            Status::Failure => Status::Success,
            Status::Running => Status::Running,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Repeater ────────────────────────────────────────────────────────────────
// Re-ticks its child up to `max_times` (or forever if max_times == 0).
// Each time the child completes (Success or Failure), the counter increments.
// Returns Running while the child is still being repeated.
// Returns Success once all repetitions are done (or never, if infinite).
//
// NOTE: We store a raw pointer to the child to allow re-ticking it within a
// single BTContext borrow.  This is safe because:
//   1. The child is owned exclusively by this Repeater (Box ownership).
//   2. We never create two simultaneous &mut references — the pointer is
//      dereferenced one tick at a time, sequentially.

// SAFETY: Repeater is only used on the main game thread; raw pointer is
// exclusively owned and never shared across threads.
unsafe impl Send for Repeater {}
unsafe impl Sync for Repeater {}

pub struct Repeater {
    child_ptr: *mut Box<dyn BehaviorNode>,
    child: Box<Box<dyn BehaviorNode>>,
    max_times: u32,
    count: u32,
    name: String,
}

impl Repeater {
    pub fn new(name: &str, child: Box<dyn BehaviorNode>, max_times: u32) -> Self {
        let mut boxed_child = Box::new(child);
        let child_ptr = &mut *boxed_child as *mut Box<dyn BehaviorNode>;
        Self {
            child_ptr,
            child: boxed_child,
            max_times,
            count: 0,
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Repeater {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        // SAFETY: child_ptr points to self.child, which is exclusively owned
        // by this Repeater.  No aliasing occurs because we only dereference
        // one pointer at a time.
        let child = unsafe { &mut *self.child_ptr };

        let status = child.tick(ctx);

        match status {
            Status::Running => Status::Running,
            Status::Success | Status::Failure => {
                self.count += 1;
                if self.max_times > 0 && self.count >= self.max_times {
                    // All repetitions done.
                    self.count = 0;
                    Status::Success
                } else {
                    // More repetitions to go — signal Running so we get ticked again.
                    Status::Running
                }
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Succeeder ───────────────────────────────────────────────────────────────
// Always returns Success regardless of what the child does.
// Useful when you want a branch to always "succeed" for flow control purposes.

pub struct Succeeder {
    child: Box<dyn BehaviorNode>,
    name: String,
}

impl Succeeder {
    pub fn new(name: &str, child: Box<dyn BehaviorNode>) -> Self {
        Self {
            child,
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Succeeder {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        // Tick the child to let it run (it might have side effects),
        // but always report Success to the parent.
        let _ = self.child.tick(ctx);
        Status::Success
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── RepeatUntilFail ─────────────────────────────────────────────────────────
// Keeps ticking its child every frame.  As long as the child returns Success
// or Running, the decorator returns Running.  When the child finally returns
// Failure, the decorator returns Success (the loop is "done").
//
// Useful for "keep trying until it doesn't work" patterns, e.g. patrolling
// waypoints until an enemy is spotted.

pub struct RepeatUntilFail {
    child_ptr: *mut Box<dyn BehaviorNode>,
    child: Box<Box<dyn BehaviorNode>>,
    name: String,
}

// SAFETY: exclusively owned raw pointer, main-thread only.
unsafe impl Send for RepeatUntilFail {}
unsafe impl Sync for RepeatUntilFail {}

impl RepeatUntilFail {
    pub fn new(name: &str, child: Box<dyn BehaviorNode>) -> Self {
        let mut boxed_child = Box::new(child);
        let child_ptr = &mut *boxed_child as *mut Box<dyn BehaviorNode>;
        Self {
            child_ptr,
            child: boxed_child,
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for RepeatUntilFail {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        // SAFETY: same reasoning as Repeater — exclusively owned child.
        let child = unsafe { &mut *self.child_ptr };
        match child.tick(ctx) {
            Status::Failure => Status::Success,  // Child failed → we're done.
            Status::Success => Status::Running,   // Child succeeded → keep going.
            Status::Running => Status::Running,   // Still in progress → keep going.
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Cooldown ────────────────────────────────────────────────────────────────
// Prevents the child from being ticked more often than every `duration`
// seconds.  While the cooldown is active, returns Running without ticking
// the child.  Once the cooldown expires, ticks the child normally.
//
// Uses a raw pointer to the child for the same aliasing-safety reasons as
// Repeater: we need to hold cooldown state and tick the child through the
// same &mut BTContext.

pub struct Cooldown {
    child_ptr: *mut Box<dyn BehaviorNode>,
    child: Box<Box<dyn BehaviorNode>>,
    duration: f32,
    last_fire_time: f32,
    name: String,
}

// SAFETY: exclusively owned raw pointer, main-thread only.
unsafe impl Send for Cooldown {}
unsafe impl Sync for Cooldown {}

impl Cooldown {
    pub fn new(name: &str, child: Box<dyn BehaviorNode>, duration: f32) -> Self {
        let mut boxed_child = Box::new(child);
        let child_ptr = &mut *boxed_child as *mut Box<dyn BehaviorNode>;
        Self {
            child_ptr,
            child: boxed_child,
            duration,
            // Start with last_fire_time = 0 so the first tick fires immediately.
            last_fire_time: -1000.0,
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Cooldown {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        let elapsed = ctx.time_s - self.last_fire_time;
        if elapsed < self.duration {
            // Still on cooldown — don't tick the child.
            return Status::Running;
        }
        self.last_fire_time = ctx.time_s;
        // SAFETY: exclusively owned child, no aliasing.
        let child = unsafe { &mut *self.child_ptr };
        child.tick(ctx)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Conditional ─────────────────────────────────────────────────────────────
// A gate node: ticks its child only if the condition key in the blackboard
// is `true`.  If the condition is `false` (or missing), returns Failure
// without ticking the child.
//
// This enables data-driven branching: a "check_health" leaf writes
// "is_low_health" to the blackboard, and a Conditional gate further down
// the tree reads it to decide whether to flee.

pub struct Conditional {
    child_ptr: *mut Box<dyn BehaviorNode>,
    child: Box<Box<dyn BehaviorNode>>,
    condition_key: String,
    name: String,
}

// SAFETY: exclusively owned raw pointer, main-thread only.
unsafe impl Send for Conditional {}
unsafe impl Sync for Conditional {}

impl Conditional {
    pub fn new(name: &str, child: Box<dyn BehaviorNode>, condition_key: &str) -> Self {
        let mut boxed_child = Box::new(child);
        let child_ptr = &mut *boxed_child as *mut Box<dyn BehaviorNode>;
        Self {
            child_ptr,
            child: boxed_child,
            condition_key: condition_key.to_string(),
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Conditional {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        let condition_met = ctx.blackboard.get_bool(&self.condition_key).unwrap_or(false);
        if !condition_met {
            return Status::Failure;
        }
        // SAFETY: exclusively owned child, no aliasing.
        let child = unsafe { &mut *self.child_ptr };
        child.tick(ctx)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEAF NODES (ACTIONS)
// ═══════════════════════════════════════════════════════════════════════════════
// Leaves are the "doing" nodes — they perform actual work (move, wait, log)
// and have no children.

// ── MoveTo ──────────────────────────────────────────────────────────────────
// Uses the NavGrid A* pathfinder to find a path from the entity's current
// position to a target position (read from the blackboard key `target_pos`).
// Each tick, the entity moves toward the next waypoint in the path.
//
// Behavior:
//   1. First tick: compute path, store in blackboard as `current_path`.
//   2. Subsequent ticks: move toward next waypoint.
//   3. When all waypoints are visited → Success.
//   4. If no path exists → Failure.
//
// The entity's position is updated via the ECS Position component.

pub struct MoveTo {
    speed: f32,
    target_key: String,
    name: String,
}

impl MoveTo {
    pub fn new(name: &str, speed: f32, target_key: &str) -> Self {
        Self {
            speed,
            target_key: target_key.to_string(),
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for MoveTo {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        // Read the target position from the blackboard.
        let target_pos = match ctx.blackboard.get_vec3(&self.target_key) {
            Some(p) => p,
            None => return Status::Failure, // No target set.
        };

        // Get the entity's current position.
        let current_pos = match ctx.world.get::<&Position>(ctx.entity) {
            Ok(p) => [p.x, p.y, p.z],
            Err(_) => return Status::Failure,
        };

        // Check if we already arrived.
        let dx = target_pos[0] - current_pos[0];
        let dy = target_pos[1] - current_pos[1];
        let dz = target_pos[2] - current_pos[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq < 0.1 * 0.1 {
            return Status::Success;
        }

        // If we don't have a path yet (or target changed), compute one.
        let needs_new_path = match ctx.blackboard.get_path("current_path") {
            Some(path) => path.is_empty(),
            None => true,
        };

        if needs_new_path {
            let from = [current_pos[0], current_pos[1], current_pos[2]];
            let waypoints = match find_agent_path(ctx, from, target_pos, current_pos[1]) {
                Some(wp) => wp,
                None => return Status::Failure, // No path exists.
            };
            // Store the raw waypoints (we'll pop from the front).
            ctx.blackboard.set(
                "current_path",
                BlackboardValue::Path(waypoints),
            );
            ctx.blackboard.set(
                "path_waypoint_index",
                BlackboardValue::Float(0.0),
            );
        }

        // Pop the next waypoint from the path.
        let wp_index = ctx.blackboard
            .get_float("path_waypoint_index")
            .unwrap_or(0.0) as usize;

        let waypoints = match ctx.blackboard.get_path("current_path") {
            Some(wp) => wp.clone(),
            None => return Status::Failure,
        };

        if wp_index >= waypoints.len() {
            // All waypoints consumed — we've arrived.
            ctx.blackboard.remove("current_path");
            ctx.blackboard.remove("path_waypoint_index");
            return Status::Success;
        }

        let wp = waypoints[wp_index];

        // Move toward this waypoint.
        let dx = wp[0] - current_pos[0];
        let dy = wp[1] - current_pos[1];
        let dz = wp[2] - current_pos[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist < 0.15 {
            // Reached this waypoint — advance to the next.
            ctx.blackboard.set(
                "path_waypoint_index",
                BlackboardValue::Float((wp_index + 1) as f32),
            );
            return Status::Running;
        }

        // Normalize direction and move.
        let nx = dx / dist;
        let ny = dy / dist;
        let nz = dz / dist;
        let step = self.speed * ctx.dt;
        let move_x = nx * step.min(dist);
        let move_y = ny * step.min(dist);
        let move_z = nz * step.min(dist);

        if let Ok(mut pos) = ctx.world.get::<&mut Position>(ctx.entity) {
            pos.x += move_x;
            pos.y += move_y;
            pos.z += move_z;
        }

        Status::Running
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Patrol ──────────────────────────────────────────────────────────────────
// Cycles through a list of waypoints, moving to each in sequence.
// The waypoint list is read from the blackboard key `patrol_points`.
// The current patrol index is stored in `patrol_index`.
//
// On each tick, Patrol delegates to internal MoveTo logic.
// When the entity reaches a waypoint, the index advances and wraps around.

pub struct Patrol {
    speed: f32,
    waypoints_key: String,
    index_key: String,
    name: String,
}

impl Patrol {
    pub fn new(name: &str, speed: f32, waypoints_key: &str) -> Self {
        Self {
            speed,
            waypoints_key: waypoints_key.to_string(),
            index_key: format!("{}_idx", waypoints_key),
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Patrol {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        let waypoints = match ctx.blackboard.get_path(&self.waypoints_key) {
            Some(wp) => wp.clone(),
            None => return Status::Failure,
        };

        if waypoints.is_empty() {
            return Status::Failure;
        }

        let idx = ctx.blackboard.get_float(&self.index_key).unwrap_or(0.0) as usize;
        let target = waypoints[idx % waypoints.len()];

        // Get current position.
        let current_pos = match ctx.world.get::<&Position>(ctx.entity) {
            Ok(p) => [p.x, p.y, p.z],
            Err(_) => return Status::Failure,
        };

        let dx = target[0] - current_pos[0];
        let dy = target[1] - current_pos[1];
        let dz = target[2] - current_pos[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist < 0.3 {
            // Reached waypoint — advance to next.
            let next = (idx + 1) % waypoints.len();
            ctx.blackboard.set(&self.index_key, BlackboardValue::Float(next as f32));
            return Status::Success;
        }

        // Move toward current waypoint.
        let nx = dx / dist;
        let ny = dy / dist;
        let nz = dz / dist;
        let step = self.speed * ctx.dt;

        if let Ok(mut pos) = ctx.world.get::<&mut Position>(ctx.entity) {
            pos.x += nx * step.min(dist);
            pos.y += ny * step.min(dist);
            pos.z += nz * step.min(dist);
        }

        Status::Running
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Wait ────────────────────────────────────────────────────────────────────
// Returns Running for `duration` seconds, then returns Success.
// Simple but essential — used to create pauses in behavior sequences.

pub struct Wait {
    duration: f32,
    elapsed: f32,
    name: String,
}

impl Wait {
    pub fn new(name: &str, duration: f32) -> Self {
        Self {
            duration,
            elapsed: 0.0,
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Wait {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        self.elapsed += ctx.dt;
        if self.elapsed >= self.duration {
            self.elapsed = 0.0;
            Status::Success
        } else {
            Status::Running
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Log ─────────────────────────────────────────────────────────────────────
// Prints a debug message via tracing::info! and returns Success immediately.
// Useful for debugging behavior tree execution flow.

pub struct Log {
    message: String,
    name: String,
}

impl Log {
    pub fn new(name: &str, message: &str) -> Self {
        Self {
            message: message.to_string(),
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Log {
    fn tick(&mut self, _ctx: &mut BTContext) -> Status {
        tracing::info!("[BT:{}] {}", self.name, self.message);
        Status::Success
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── CustomAction ────────────────────────────────────────────────────────────
// Wraps a user-provided closure.  This is the primary extension point for
// game-specific behavior: write any logic you need in a closure and it
// becomes a leaf node.
//
// Example:
//   CustomAction::new("Shoot", |ctx| {
//       // do shooting logic...
//       Status::Success
//   })

pub struct CustomAction {
    action: Box<dyn FnMut(&mut BTContext) -> Status + Send + Sync>,
    name: String,
}

impl CustomAction {
    pub fn new<F>(name: &str, action: F) -> Self
    where
        F: FnMut(&mut BTContext) -> Status + Send + Sync + 'static,
    {
        Self {
            action: Box::new(action),
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for CustomAction {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        (self.action)(ctx)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── SetState ─────────────────────────────────────────────────────────────────
// Writes an animation state name to the blackboard key "ai_state".
// This is the bridge between BT logic and the skeletal animation system:
//   1. BT designer adds SetState::new("chase", "run") to their tree.
//   2. When this node ticks, it writes "ai_state" = "run" to the blackboard.
//   3. animation_blending_system reads "ai_state" and calls play_state().
//   4. SkeletalAnimator triggers a crossfade to the Run clip.
//
// Usage in a BT:
//   Sequence::new("ChasePlayer", vec![
//       Box::new(SetState::new("SetRun", "run")),
//       Box::new(MoveTo::new("MoveToPlayer", 5.0, "target_pos")),
//       Box::new(SetState::new("SetAttack", "attack")),
//   ])

pub struct SetState {
    state_name: String,
    name: String,
}

impl SetState {
    pub fn new(name: &str, state_name: &str) -> Self {
        Self {
            state_name: state_name.to_string(),
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for SetState {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        ctx.blackboard.set(
            "ai_state",
            crate::ai::blackboard::BlackboardValue::String(self.state_name.clone()),
        );
        Status::Success
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PER-AGENT SEEDED RNG
// ═══════════════════════════════════════════════════════════════════════════════
// A lightweight, deterministic LCG stored ON the agent's blackboard so each
// entity has its own reproducible random stream.  This fixes the existing
// `RandomSelector` problem where every agent using the same tree gets the SAME
// random pick (it seeds off SystemTime, ignoring the entity).  Wander/Flee/
// Graze and the editor's random branches all advance this per-agent stream.
//
// The seed lives under a reserved blackboard key `_rng_seed` as a Float.
// It is initialised lazily from a per-agent source (entity bits) so that two
// agents with identical trees behave differently.

/// Reserved blackboard key holding the per-agent RNG state.
pub const AGENT_RNG_KEY: &str = "_rng_seed";

struct AgentRng {
    state: u64,
}

impl AgentRng {
    // FNV-1a style avalanche mix for the Fowler-Noll hash. Non-cryptographic.
    fn mix(a: u64) -> u64 {
        let mut h = a;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    fn from_seed(seed: u64) -> Self {
        Self { state: Self::mix(seed).max(1) }
    }

    /// Uniform float in [0, 1).
    fn next(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let hi = (self.state >> 32) as u32;
        (hi as f64 / (u32::MAX as f64 + 1.0)) as f32
    }

    /// Uniform float in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }

    /// Random point in [0,1)^3 (for velocity spread, etc.).
    fn vec3(&mut self) -> [f32; 3] {
        [self.next(), self.next(), self.next()]
    }
}

/// Obtain (or lazily initialise) a per-agent RNG borrowed from the blackboard.
/// The agent's identity (entity bits) determines its default seed so results
/// differ between agents even when trees are identical.
fn agent_rng<'a>(blackboard: &'a mut Blackboard, entity: hecs::Entity) -> AgentRng {
    match blackboard.get_float(AGENT_RNG_KEY) {
        Some(state) => AgentRng { state: state as u64 },
        None => {
            // Initialise from the entity identifier for per-agent uniqueness.
            // XOR (not OR) with the golden constant so that differing low bits
            // of the entity id actually change the seed.
            let seed = entity.to_bits().get() ^ 0x9E3779B97F4A7C15u64;
            let rng = AgentRng::from_seed(seed);
            // Persist the starting state so it behaves deterministically.
            blackboard.set(AGENT_RNG_KEY, BlackboardValue::Float(rng.state as f32));
            rng
        }
    }
}

/// Persist the RNG state back onto the blackboard after advancing it.
fn store_agent_rng(blackboard: &mut Blackboard, rng: &AgentRng) {
    blackboard.set(AGENT_RNG_KEY, BlackboardValue::Float(rng.state as f32));
}

// ═══════════════════════════════════════════════════════════════════════════════
// PASSIVE OPEN-WORLD BEHAVIOR NODES
// ═══════════════════════════════════════════════════════════════════════════════
// A small set of engine-native behaviors for ambient creatures: Wander, Flee,
// Idle, Graze/Consume, Perception (FindNearestEntity) and a Distance condition.
// Most require the ECS World or the NavGrid — things that are awkward/impossible
// to express purely in Lua — so they are embedded as BehaviorNode impls.  The
// Lua `bt.*` builder (and the BT editor) can compose them into trees.
//
// All of these drive the `ai_state` blackboard value so the animation-blending
// system selects an appropriate clip ("idle", "walk", "run", "graze").

// ── Idle ──────────────────────────────────────────────────────────────────────
// Native idle: drives `ai_state` = "idle".  With `duration == 0` it returns
// Running indefinitely (the agent stands still).  With a positive duration it
// runs for that many seconds then returns Success, letting a parent Selector
// fall through to another behavior.

pub struct Idle {
    duration: f32,
    remained: f32,
    name: String,
}

impl Idle {
    pub fn new(name: &str, duration: f32) -> Self {
        Self { duration, remained: duration, name: name.to_string() }
    }
}

impl BehaviorNode for Idle {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        ctx.blackboard.set("ai_state", BlackboardValue::String("idle".to_string()));
        if self.duration <= 0.0 {
            return Status::Running;
        }
        self.remained -= ctx.dt;
        if self.remained <= 0.0 {
            self.remained = self.duration;
            Status::Success
        } else {
            Status::Running
        }
    }
    fn name(&self) -> &str {
        &self.name
    }
}

// ── Wander ────────────────────────────────────────────────────────────────────
// Picks a random reachable point within `radius` of `anchor` and walks there
// using the NavGrid.  On arrival it picks a fresh target and keeps wandering
// (returns Running while moving).  Uses the per-agent RNG so each creature
// wanders on its own path.  The `home` blackboard key (Vec3) may override the
// fixed anchor so the wanderer stays near a den/territory.

pub struct Wander {
    speed: f32,
    radius: f32,
    anchor: [f32; 3],
    name: String,
}

impl Wander {
    pub fn new(name: &str, speed: f32, radius: f32, anchor: [f32; 3]) -> Self {
        Self { speed, radius, anchor, name: name.to_string() }
    }
}

impl BehaviorNode for Wander {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        // Per-agent RNG for deterministic-but-distinct wandering.
        let mut rng = agent_rng(ctx.blackboard, ctx.entity);

        let anchor = ctx
            .blackboard
            .get_vec3("home")
            .unwrap_or(self.anchor);

        // Move toward the currently-latched target if one exists.
        let target = ctx.blackboard.get_vec3("target_pos");

        if target.is_none() {
            // Pick a new random point within radius around the anchor.
            let mut candidate = [anchor[0], anchor[1], anchor[2]];
            for _ in 0..8 {
                let dir = rng.vec3();
                let dx = (dir[0] * 2.0 - 1.0) * self.radius;
                let dz = (dir[2] * 2.0 - 1.0) * self.radius;
                candidate = [anchor[0] + dx, anchor[1], anchor[2] + dz];
                // Ensure the candidate is walkable; retry with a new point.
                if Self::grid_walkable(ctx, candidate) {
                    break;
                }
            }
            ctx.blackboard.set("target_pos", BlackboardValue::Vec3(candidate));
            ctx.blackboard.set("ai_state", BlackboardValue::String("walk".to_string()));
            store_agent_rng(ctx.blackboard, &rng);
            return Status::Running;
        }

        let target_pos = target.as_ref().copied().unwrap();
        let status = Self::move_step(ctx, target_pos, self.speed);
        store_agent_rng(ctx.blackboard, &rng);

        if status == Status::Success {
            // Arrived — clear the target so the next tick picks a new one.
            ctx.blackboard.remove("target_pos");
            ctx.blackboard.set("ai_state", BlackboardValue::String("idle".to_string()));
            Status::Running
        } else {
            ctx.blackboard.set("ai_state", BlackboardValue::String("walk".to_string()));
            status
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Wander {
    fn grid_walkable(ctx: &mut BTContext, pos: [f32; 3]) -> bool {
        // Prefer the polygon navmesh for reachability when available.
        if let Some(nm) = ctx.navmesh {
            if !nm.is_walkable_at(pos) {
                return false;
            }
        }
        let half_w = ctx.nav_grid.width as f32 * 0.5;
        let half_d = ctx.nav_grid.depth as f32 * 0.5;
        let gx = (pos[0] + half_w).round() as isize;
        let gz = (pos[2] + half_d).round() as isize;
        if gx < 0 || gz < 0 || gx >= ctx.nav_grid.width as isize || gz >= ctx.nav_grid.depth as isize {
            return false;
        }
        ctx.nav_grid.walkable[gz as usize * ctx.nav_grid.width + gx as usize]
    }

    fn move_step(ctx: &mut BTContext, target: [f32; 3], speed: f32) -> Status {
        move_toward(ctx, target, speed)
    }
}

// ── Flee ─────────────────────────────────────────────────────────────────────
// Reads the current threat (either a Vec3 `threat_pos` or an Entity
// `threat`→its position) and moves directly away from it at `run_speed`.
// Continues running until the threat is farther than `safe_distance`, then
// returns Success.  Fails immediately if no threat is recorded.

pub struct Flee {
    run_speed: f32,
    safe_distance: f32,
    threat_pos_key: String,
    threat_entity_key: String,
    name: String,
}

impl Flee {
    pub fn new(name: &str, run_speed: f32, safe_distance: f32) -> Self {
        Self {
            run_speed,
            safe_distance,
            threat_pos_key: "threat_pos".to_string(),
            threat_entity_key: "threat".to_string(),
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Flee {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        let current = match ctx.world.get::<&Position>(ctx.entity) {
            Ok(p) => [p.x, p.y, p.z],
            Err(_) => return Status::Failure,
        };

        // Resolve threat position from either source.
        let mut threat = ctx.blackboard.get_vec3(&self.threat_pos_key);
        if threat.is_none() {
            if let Some(eid) = ctx.blackboard.get_entity(&self.threat_entity_key) {
                if let Some(e) = hecs::Entity::from_bits(eid) {
                    if let Ok(p) = ctx.world.get::<&Position>(e) {
                        threat = Some([p.x, p.y, p.z]);
                    }
                }
            }
        }
        let Some(threat_pos) = threat else {
            return Status::Failure;
        };

        let dx = current[0] - threat_pos[0];
        let dz = current[2] - threat_pos[2];
        let dist = (dx * dx + dz * dz).sqrt();
        if dist > self.safe_distance {
            ctx.blackboard.set("ai_state", BlackboardValue::String("idle".to_string()));
            return Status::Success;
        }

        // Run away: direction from threat, kept on the ground plane.
        let dir_len = dist.max(1e-4);
        let nx = dx / dir_len;
        let nz = dz / dir_len;
        // Step to the escape cell; clamp to nav bounds but allow ground movement.
        let target = [current[0] + nx * 1.0, current[1], current[2] + nz * 1.0];
        move_toward(ctx, target, self.run_speed);
        ctx.blackboard.set("ai_state", BlackboardValue::String("run".to_string()));
        Status::Running
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Graze / Consume ───────────────────────────────────────────────────────────
// Finds the nearest entity tagged as consumable within `radius`, walks to it,
// and on contact "consumes" it (despawns it from the world) and succeeds.
// Fails if no consumable is in range.  This drives the "graze" animation state.

pub struct Graze {
    speed: f32,
    radius: f32,
    consume_tag: String,
    name: String,
}

impl Graze {
    pub fn new(name: &str, speed: f32, radius: f32) -> Self {
        Self {
            speed,
            radius,
            consume_tag: "grazeable".to_string(),
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Graze {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        let current = match ctx.world.get::<&Position>(ctx.entity) {
            Ok(p) => [p.x, p.y, p.z],
            Err(_) => return Status::Failure,
        };

        // Do we already have a latched food target?
        let mut food_bits = ctx.blackboard.get_entity("graze_target");

        if food_bits.is_none() {
            // Scan the world for the nearest grazeable entity.
            food_bits = find_nearest_entity_fast(ctx, current, &self.consume_tag, self.radius);
            let Some(bits) = food_bits else { return Status::Failure; };
            ctx.blackboard.set("graze_target", BlackboardValue::Entity(bits));
        }

        let Some(bits) = food_bits else { return Status::Failure; };
        let Some(food) = hecs::Entity::from_bits(bits) else {
            ctx.blackboard.remove("graze_target");
            return Status::Failure;
        };

        let food_pos = match ctx.world.get::<&Position>(food) {
            Ok(p) => [p.x, p.y, p.z],
            Err(_) => {
                // Target vanished — clear and retry next tick.
                ctx.blackboard.remove("graze_target");
                return Status::Running;
            }
        };

        let dx = food_pos[0] - current[0];
        let dy = food_pos[1] - current[1];
        let dz = food_pos[2] - current[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        ctx.blackboard.set("ai_state", BlackboardValue::String("graze".to_string()));

        if dist < 0.5 {
            // Consume it.
            let _ = ctx.world.despawn(food);
            ctx.blackboard.remove("graze_target");
            ctx.blackboard.set("ai_state", BlackboardValue::String("idle".to_string()));
            return Status::Success;
        }

        move_toward(ctx, food_pos, self.speed);
        Status::Running
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Perception / FindNearestEntity ────────────────────────────────────────────
// A leaf that scans the ECS World for the closest entity within `radius` that
// carries a provided tag (matched against a per-entity "faction"/"tag" string
// stored on the blackboard side; a tag of "*" matches any entity).  When a
// target is found it writes its entity bits to the `result_entity_key` and its
// position to the `result_pos_key`, then returns Success.  Otherwise it
// returns Failure.  This is the engine-side primitive that pure-Lua cannot do.

pub struct Perception {
    radius: f32,
    look_for_tag: String,
    result_entity_key: String,
    result_pos_key: String,
    name: String,
}

impl Perception {
    pub fn new(name: &str, radius: f32, look_for_tag: &str) -> Self {
        Self {
            radius,
            look_for_tag: look_for_tag.to_string(),
            result_entity_key: "perceived_entity".to_string(),
            result_pos_key: "perceived_pos".to_string(),
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for Perception {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        let current = match ctx.world.get::<&Position>(ctx.entity) {
            Ok(p) => [p.x, p.y, p.z],
            Err(_) => return Status::Failure,
        };

        let nearest = find_nearest_entity_fast(ctx, current, &self.look_for_tag, self.radius);
        match nearest {
            Some(bits) => {
                if let Some(e) = hecs::Entity::from_bits(bits) {
                    if let Ok(p) = ctx.world.get::<&Position>(e) {
                        ctx.blackboard.set(&self.result_entity_key, BlackboardValue::Entity(bits));
                        ctx.blackboard.set(&self.result_pos_key, BlackboardValue::Vec3([p.x, p.y, p.z]));
                    }
                }
                Status::Success
            }
            None => Status::Failure,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── DistanceCondition ─────────────────────────────────────────────────────────
// A gate (decorator-style) that ticks its child only while the distance between
// the entity and a reference position (blackboard Vec3 or result of a
// Perception) is within [min, max].  Stores the distance in the blackboard key
// `dist_to_target`.  Used to gate e.g. "attack only when within 5u" or "flee
// only when the creature is within 8u".

pub struct DistanceCondition {
    child_ptr: *mut Box<dyn BehaviorNode>,
    child: Box<Box<dyn BehaviorNode>>,
    reference_pos_key: String,
    min: f32,
    max: f32,
    distance_key: String,
    name: String,
}

// SAFETY: exclusively owned raw pointer, main-thread only (same as Repeater).
unsafe impl Send for DistanceCondition {}
unsafe impl Sync for DistanceCondition {}

impl DistanceCondition {
    pub fn new(
        name: &str,
        child: Box<dyn BehaviorNode>,
        reference_pos_key: &str,
        min: f32,
        max: f32,
    ) -> Self {
        let mut boxed_child = Box::new(child);
        let child_ptr = &mut *boxed_child as *mut Box<dyn BehaviorNode>;
        Self {
            child_ptr,
            child: boxed_child,
            reference_pos_key: reference_pos_key.to_string(),
            min,
            max,
            distance_key: "dist_to_target".to_string(),
            name: name.to_string(),
        }
    }
}

impl BehaviorNode for DistanceCondition {
    fn tick(&mut self, ctx: &mut BTContext) -> Status {
        let Some(reference) = ctx.blackboard.get_vec3(&self.reference_pos_key) else {
            return Status::Failure;
        };
        let current = match ctx.world.get::<&Position>(ctx.entity) {
            Ok(p) => [p.x, p.y, p.z],
            Err(_) => return Status::Failure,
        };
        let dx = current[0] - reference[0];
        let dy = current[1] - reference[1];
        let dz = current[2] - reference[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        ctx.blackboard.set(&self.distance_key, BlackboardValue::Float(dist));

        if dist < self.min || dist > self.max {
            return Status::Failure;
        }
        // SAFETY: exclusively owned child, no aliasing (same as Conditional).
        let child = unsafe { &mut *self.child_ptr };
        child.tick(ctx)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Find a path from `from` to `to`, preferring the triangle navmesh and
/// falling back to the NavGrid A*. Returns world-space waypoints (including
/// both endpoints). The grid fallback keeps its own height on all waypoints,
/// so `ref_height` (typically the agent's current Y) is used there.
fn find_agent_path(
    ctx: &BTContext,
    from: [f32; 3],
    to: [f32; 3],
    ref_height: f32,
) -> Option<Vec<[f32; 3]>> {
    // 1. Real navmesh first — smooth 3D paths on the actual surface.
    if let Some(nm) = ctx.navmesh {
        if let Some(path) = nm.find_path(from, to) {
            if path.len() > 1 {
                return Some(path);
            }
        }
    }

    // 2. Grid fallback — same result as legacy MoveTo.
    let start = world_to_grid(from, ctx.nav_grid.width, ctx.nav_grid.depth);
    let goal = world_to_grid(to, ctx.nav_grid.width, ctx.nav_grid.depth);
    let grid_path = ctx.nav_grid.find_path(start, goal)?;

    let half_w = ctx.nav_grid.width as f32 * 0.5;
    let half_d = ctx.nav_grid.depth as f32 * 0.5;
    let mut waypoints: Vec<[f32; 3]> = Vec::with_capacity(grid_path.len());
    for &(gx, gz) in &grid_path {
        waypoints.push([gx as f32 - half_w, ref_height, gz as f32 - half_d]);
    }
    if waypoints.last().map(|l| *l != to).unwrap_or(true) {
        waypoints.push(to);
    }
    Some(waypoints)
}

/// Convert world-space [f32;3] to NavGrid grid coordinates.
/// The NavGrid spans from (-half_w, -half_d) to (+half_w, +half_d) in
/// world space, with (0,0) at the grid center.
fn world_to_grid(pos: [f32; 3], grid_w: usize, grid_d: usize) -> (usize, usize) {
    let half_w = grid_w as f32 * 0.5;
    let half_d = grid_d as f32 * 0.5;
    let gx = (pos[0] + half_w).round().clamp(0.0, (grid_w - 1) as f32) as usize;
    let gz = (pos[2] + half_d).round().clamp(0.0, (grid_d - 1) as f32) as usize;
    (gx, gz)
}

/// Move the entity one step toward `target` at `speed`.  Returns Success when
/// within snap tolerance, Running otherwise.  Shares the stepping logic between
/// Wander, Flee and Graze so movement is consistent.
fn move_toward(ctx: &mut BTContext, target: [f32; 3], speed: f32) -> Status {
    let current = match ctx.world.get::<&Position>(ctx.entity) {
        Ok(p) => [p.x, p.y, p.z],
        Err(_) => return Status::Failure,
    };
    let dx = target[0] - current[0];
    let dy = target[1] - current[1];
    let dz = target[2] - current[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    if dist < 0.15 {
        return Status::Success;
    }
    let step = speed * ctx.dt.max(0.0);
    let s = step.min(dist);
    if let Ok(mut pos) = ctx.world.get::<&mut Position>(ctx.entity) {
        pos.x += dx / dist * s;
        pos.y += dy / dist * s;
        pos.z += dz / dist * s;
    }
    Status::Running
}

/// Shared nearest-entity scan used by Perception and Graze.
/// Matches any entity with a Position that is not `ctx.entity`, optionally
/// filtered by a tag.  Returns its entity bits, or None.
fn find_nearest_entity_fast(
    ctx: &mut BTContext,
    origin: [f32; 3],
    tag: &str,
    radius: f32,
) -> Option<u64> {
    let mut nearest: Option<(f32, u64)> = None;
    for (entity, pos) in ctx.world.query::<(hecs::Entity, &Position)>().iter() {
        if entity == ctx.entity {
            continue;
        }
        if tag != "*" {
            // Match either an optional EntityTag component or a "tag" value on
            // an AiAgent blackboard.  Entities without a matching tag are skipped.
            let component_tag = ctx.world.get::<&EntityTag>(entity).ok().map(|t| t.0.clone());
            let bb_tag = ctx
                .world
                .get::<&AiAgent>(entity)
                .ok()
                .and_then(|a| a.blackboard.get_string("tag").map(|t| t.to_string()));
            let ok = component_tag
                .or(bb_tag)
                .map_or(false, |t: String| t == tag);
            if !ok {
                continue;
            }
        }
        let dx = pos.x - origin[0];
        let dy = pos.y - origin[1];
        let dz = pos.z - origin[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist <= radius {
            if nearest.map_or(true, |(best, _)| dist < best) {
                nearest = Some((dist, entity.to_bits().get()));
            }
        }
    }
    nearest.map(|(_, bits)| bits)
}

/// Marker that tags an entity for `find_nearest_entity_fast` scanning even when
/// it has no AiAgent component.  This is a lightweight, optional extension used
/// by game code to mark plants, food, or hazards that other AI should perceive.
/// Without it, tags are read from an AiAgent's blackboard.
pub struct EntityTag(pub String);

// ═══════════════════════════════════════════════════════════════════════════════
// BEHAVIOR TREE WRAPPER
// ═══════════════════════════════════════════════════════════════════════════════
// Wraps a root node with a name.  This is what gets registered in the
// AiRegistry and ticked by the ai_system.

pub struct BehaviorTree {
    root: Box<dyn BehaviorNode>,
    tree_name: String,
}

impl BehaviorTree {
    pub fn new(name: &str, root: Box<dyn BehaviorNode>) -> Self {
        Self {
            root,
            tree_name: name.to_string(),
        }
    }

    /// Tick the root node and return the overall status.
    pub fn tick(&mut self, ctx: &mut BTContext) -> Status {
        self.root.tick(ctx)
    }

    pub fn name(&self) -> &str {
        &self.tree_name
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::blackboard::Blackboard;
    use crate::components::Position;
    use crate::navigation::NavGrid;
    use crate::terrain::TerrainGrid;

    /// Helper: create a minimal NavGrid for testing.
    fn test_nav_grid() -> NavGrid {
        NavGrid::from_terrain(&TerrainGrid::new(16, 16, 1.0), 1.0)
    }

    /// Helper: create a BTContext wired to a test world.
    fn make_ctx<'a>(
        world: &'a mut hecs::World,
        entity: hecs::Entity,
        bb: &'a mut Blackboard,
        nav: &'a NavGrid,
    ) -> BTContext<'a> {
        BTContext {
            entity,
            world,
            dt: 0.016,
            time_s: 0.0,
            nav_grid: nav,
            navmesh: None,
            blackboard: bb,
        }
    }

    // ── Stub nodes for testing ──────────────────────────────────────────────
    struct AlwaysSucceed;
    impl BehaviorNode for AlwaysSucceed {
        fn tick(&mut self, _: &mut BTContext) -> Status { Status::Success }
        fn name(&self) -> &str { "AlwaysSucceed" }
    }

    struct AlwaysFail;
    impl BehaviorNode for AlwaysFail {
        fn tick(&mut self, _: &mut BTContext) -> Status { Status::Failure }
        fn name(&self) -> &str { "AlwaysFail" }
    }

    struct AlwaysRunning;
    impl BehaviorNode for AlwaysRunning {
        fn tick(&mut self, _: &mut BTContext) -> Status { Status::Running }
        fn name(&self) -> &str { "AlwaysRunning" }
    }

    // ── Sequence tests ─────────────────────────────────────────────────────

    #[test]
    fn sequence_succeeds_when_all_children_succeed() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        let mut tree = BehaviorTree::new(
            "test_seq_pass",
            Box::new(Sequence::new("seq", vec![
                Box::new(AlwaysSucceed),
                Box::new(AlwaysSucceed),
                Box::new(AlwaysSucceed),
            ])),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Success);
    }

    #[test]
    fn sequence_fails_on_first_failure() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        let mut tree = BehaviorTree::new(
            "test_seq_fail",
            Box::new(Sequence::new("seq", vec![
                Box::new(AlwaysSucceed),
                Box::new(AlwaysFail),
                Box::new(AlwaysSucceed), // Never reached.
            ])),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Failure);
    }

    #[test]
    fn sequence_returns_running_when_child_runs() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        let mut tree = BehaviorTree::new(
            "test_seq_running",
            Box::new(Sequence::new("seq", vec![
                Box::new(AlwaysSucceed),
                Box::new(AlwaysRunning),
                Box::new(AlwaysSucceed), // Never reached.
            ])),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Running);
    }

    // ── Selector tests ─────────────────────────────────────────────────────

    #[test]
    fn selector_succeeds_on_first_success() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        let mut tree = BehaviorTree::new(
            "test_sel_pass",
            Box::new(Selector::new("sel", vec![
                Box::new(AlwaysFail),
                Box::new(AlwaysSucceed),
                Box::new(AlwaysFail), // Never reached.
            ])),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Success);
    }

    #[test]
    fn selector_fails_when_all_fail() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        let mut tree = BehaviorTree::new(
            "test_sel_fail",
            Box::new(Selector::new("sel", vec![
                Box::new(AlwaysFail),
                Box::new(AlwaysFail),
            ])),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Failure);
    }

    // ── Inverter test ──────────────────────────────────────────────────────

    #[test]
    fn inverter_inverts_success_to_failure() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        let mut tree = BehaviorTree::new(
            "test_invert",
            Box::new(Inverter::new("inv", Box::new(AlwaysSucceed))),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Failure);
    }

    #[test]
    fn inverter_inverts_failure_to_success() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        let mut tree = BehaviorTree::new(
            "test_invert_fail",
            Box::new(Inverter::new("inv", Box::new(AlwaysFail))),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Success);
    }

    // ── Blackboard test ────────────────────────────────────────────────────

    #[test]
    fn blackboard_set_get_roundtrip() {
        let mut bb = Blackboard::new();
        bb.set("hp", BlackboardValue::Float(75.0));
        assert_eq!(bb.get_float("hp"), Some(75.0));

        bb.set("target", BlackboardValue::Vec3([1.0, 2.0, 3.0]));
        assert_eq!(bb.get_vec3("target"), Some([1.0, 2.0, 3.0]));

        bb.set("flag", BlackboardValue::Bool(true));
        assert_eq!(bb.get_bool("flag"), Some(true));
    }

    // ── CustomAction test ──────────────────────────────────────────────────

    #[test]
    fn custom_action_executes_closure() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        let mut tree = BehaviorTree::new(
            "test_custom",
            Box::new(CustomAction::new("set_flag", |ctx| {
                ctx.blackboard.set("done", BlackboardValue::Bool(true));
                Status::Success
            })),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Success);
        assert_eq!(bb.get_bool("done"), Some(true));
    }

    // ── Conditional test ───────────────────────────────────────────────────

    #[test]
    fn conditional_gates_child_execution() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        // Condition is false → should return Failure without ticking child.
        let mut tree = BehaviorTree::new(
            "test_cond_false",
            Box::new(Conditional::new(
                "gate",
                Box::new(AlwaysSucceed),
                "go",
            )),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Failure);

        // Set condition to true → should tick child.
        drop(ctx);
        bb.set("go", BlackboardValue::Bool(true));
        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Success);
    }

    // ── Wait test ──────────────────────────────────────────────────────────

    #[test]
    fn wait_returns_running_then_success() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        let mut tree = BehaviorTree::new(
            "test_wait",
            Box::new(Wait::new("wait_1s", 1.0)),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);

        // Tick a few times — should still be running.
        for _ in 0..50 {
            ctx.dt = 0.016;
            ctx.time_s += 0.016;
            assert_eq!(tree.tick(&mut ctx), Status::Running);
        }

        // After ~1 second total, should succeed.
        // 50 * 0.016 = 0.8, need a bit more.
        for _ in 0..20 {
            ctx.dt = 0.016;
            ctx.time_s += 0.016;
            if tree.tick(&mut ctx) == Status::Success {
                return; // Test passed.
            }
        }
        panic!("Wait did not complete within expected time");
    }

    // ── Succeeder test ─────────────────────────────────────────────────────

    #[test]
    fn succeeder_always_returns_success() {
        let nav = test_nav_grid();
        let mut world = hecs::World::new();
        let entity = world.spawn((Position { x: 0.0, y: 0.0, z: 0.0 },));
        let mut bb = Blackboard::new();

        let mut tree = BehaviorTree::new(
            "test_succeeder",
            Box::new(Succeeder::new("suc", Box::new(AlwaysFail))),
        );

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        assert_eq!(tree.tick(&mut ctx), Status::Success);
    }

    #[test]
    fn set_state_writes_ai_state_to_blackboard() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let mut bb = Blackboard::new();
        let nav = test_nav_grid();

        let mut node = SetState::new("SetRun", "run");
        {
            let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
            // Tick SetState.
            assert_eq!(node.tick(&mut ctx), Status::Success);
        }
        // ctx dropped — now we can read bb.
        assert_eq!(bb.get_string("ai_state"), Some("run"));
    }

    #[test]
    fn set_state_overwrites_previous_state() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let mut bb = Blackboard::new();
        bb.set("ai_state", crate::ai::blackboard::BlackboardValue::String("idle".to_string()));
        let nav = test_nav_grid();

        let mut node = SetState::new("SetWalk", "walk");
        {
            let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
            assert_eq!(node.tick(&mut ctx), Status::Success);
        }
        assert_eq!(bb.get_string("ai_state"), Some("walk"));
    }

    // ── Passive open-world behavior node tests ─────────────────────────────

    #[test]
    fn idle_drives_ai_state() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let mut bb = Blackboard::new();
        let nav = test_nav_grid();
        let mut node = Idle::new("Idle", 0.0);
        {
            let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
            assert_eq!(node.tick(&mut ctx), Status::Running);
        }
        assert_eq!(bb.get_string("ai_state"), Some("idle"));
    }

    #[test]
    fn idle_finite_duration_succeeds() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let mut bb = Blackboard::new();
        let nav = test_nav_grid();
        let mut node = Idle::new("Idle", 0.2);
        {
            let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
            ctx.dt = 0.3;
            assert_eq!(node.tick(&mut ctx), Status::Success);
        }
    }

    #[test]
    fn wander_moves_and_writes_seed() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let mut bb = Blackboard::new();
        let nav = test_nav_grid();
        let mut node = Wander::new("Wander", 2.0, 4.0, [0.0, 0.0, 0.0]);

        // First tick picks a target and starts moving.
        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        ctx.dt = 0.1;
        assert_eq!(node.tick(&mut ctx), Status::Running);
        assert!(ctx.blackboard.get_vec3("target_pos").is_some());
        // The per-agent RNG seed must now exist on the blackboard.
        assert!(ctx.blackboard.get_float(AGENT_RNG_KEY).is_some());

        let start = ctx.world.get::<&Position>(entity).unwrap();
        let start_x = start.x;
        let start_z = start.z;
        drop(start);
        let mut moved = false;
        for _ in 0..200 {
            ctx.dt = 0.1;
            node.tick(&mut ctx);
            let p = ctx.world.get::<&Position>(entity).unwrap();
            let dist = ((p.x - start_x).powi(2) + (p.z - start_z).powi(2)).sqrt();
            if dist > 1.0 {
                moved = true;
                break;
            }
        }
        assert!(moved, "Wander never moved the entity");
    }

    #[test]
    fn flee_runs_away_from_threat() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        // Threat at +X, directly adjacent.
        let threat = world.spawn((
            crate::components::Position { x: 2.0, y: 0.0, z: 0.0 },
        ));
        let mut bb = Blackboard::new();
        bb.set("threat", BlackboardValue::Entity(threat.to_bits().get()));
        let nav = test_nav_grid();
        let mut node = Flee::new("Flee", 2.0, 8.0);

        let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
        ctx.dt = 0.1;
        assert_eq!(node.tick(&mut ctx), Status::Running);
        // Entity should have moved away from +X (towards -X).
        let p = ctx.world.get::<&Position>(entity).unwrap();
        assert!(p.x < 0.0, "Flee moved entity toward the threat (x={})", p.x);
        assert_eq!(ctx.blackboard.get_string("ai_state"), Some("run"));
    }

    #[test]
    fn flee_succeeds_when_threat_far() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let mut bb = Blackboard::new();
        bb.set("threat_pos", BlackboardValue::Vec3([100.0, 0.0, 0.0]));
        let nav = test_nav_grid();
        let mut node = Flee::new("Flee", 2.0, 8.0);
        {
            let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
            assert_eq!(node.tick(&mut ctx), Status::Success);
        }
    }

    #[test]
    fn perception_finds_nearest_entity() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        // Two candidates: near and far.
        let near = world.spawn((
            crate::components::Position { x: 1.0, y: 0.0, z: 0.0 },
            EntityTag("prey".to_string()),
        ));
        let _far = world.spawn((
            crate::components::Position { x: 30.0, y: 0.0, z: 0.0 },
            EntityTag("prey".to_string()),
        ));
        let mut bb = Blackboard::new();
        let nav = test_nav_grid();
        let mut node = Perception::new("Perception", 5.0, "prey");
        {
            let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
            assert_eq!(node.tick(&mut ctx), Status::Success);
        }
        assert_eq!(bb.get_entity("perceived_entity"), Some(near.to_bits().get()));
        let pos = bb.get_vec3("perceived_pos").unwrap();
        assert!((pos[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn perception_ignores_out_of_range() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let _far = world.spawn((
            crate::components::Position { x: 100.0, y: 0.0, z: 0.0 },
            EntityTag("prey".to_string()),
        ));
        let mut bb = Blackboard::new();
        let nav = test_nav_grid();
        let mut node = Perception::new("Perception", 5.0, "prey");
        {
            let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
            assert_eq!(node.tick(&mut ctx), Status::Failure);
        }
    }

    #[test]
    fn graze_consumes_grazeable() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let grass = world.spawn((
            crate::components::Position { x: 0.3, y: 0.0, z: 0.0 },
            EntityTag("grazeable".to_string()),
        ));
        let mut bb = Blackboard::new();
        let nav = test_nav_grid();
        let mut node = Graze::new("Graze", 2.0, 5.0);
        {
            let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
            ctx.dt = 0.1;
            // Walk up to it and consume.
            let mut done = false;
            for _ in 0..20 {
                if node.tick(&mut ctx) == Status::Success {
                    done = true;
                    break;
                }
                ctx.dt += 0.1;
            }
            assert!(done, "Graze never consumed the target");
        }
        // The grazeable entity should be despawned.
        assert!(world.get::<&Position>(grass).is_err());
        assert_eq!(bb.get_string("ai_state"), Some("idle"));
    }

    #[test]
    fn distance_condition_gates_child() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let mut bb = Blackboard::new();
        // Reference point is far away → gate fails.
        bb.set("perceived_pos", BlackboardValue::Vec3([50.0, 0.0, 0.0]));
        let nav = test_nav_grid();

        let mut node = DistanceCondition::new(
            "InRange",
            Box::new(AlwaysSucceed),
            "perceived_pos",
            0.0,
            10.0,
        );
        {
            let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
            assert_eq!(node.tick(&mut ctx), Status::Failure);
            let d = bb.get_float("dist_to_target").unwrap();
            assert!(d > 10.0);
        }

        // Bring the reference within range → gate passes and child runs.
        bb.set("perceived_pos", BlackboardValue::Vec3([2.0, 0.0, 0.0]));
        let mut node2 = DistanceCondition::new(
            "InRange2",
            Box::new(AlwaysSucceed),
            "perceived_pos",
            0.0,
            10.0,
        );
        {
            let mut ctx = make_ctx(&mut world, entity, &mut bb, &nav);
            assert_eq!(node2.tick(&mut ctx), Status::Success);
        }
    }

    #[test]
    fn seeded_rng_differs_per_agent() {
        let mut world = hecs::World::new();
        let e1 = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let e2 = world.spawn((
            crate::components::Position { x: 0.0, y: 0.0, z: 0.0 },
        ));
        let nav = test_nav_grid();
        let mut bb1 = Blackboard::new();
        let mut bb2 = Blackboard::new();
        {
            let mut ctx1 = make_ctx(&mut world, e1, &mut bb1, &nav);
            let r1 = agent_rng(ctx1.blackboard, e1);
            let s1 = r1.state;
            drop(ctx1);
            let mut ctx2 = make_ctx(&mut world, e2, &mut bb2, &nav);
            let r2 = agent_rng(ctx2.blackboard, e2);
            assert_ne!(s1, r2.state, "Two agents should have distinct RNG seeds");
        }
    }
}
