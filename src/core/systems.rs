// src/core/systems.rs
// ──────────────────────────────────────────────────────────────────────────────
// System trait + SystemScheduler.
//
// Each engine subsystem (physics, animation, audio, scripting, etc.) implements
// the `System` trait. The `SystemScheduler` owns all systems and runs them in
// dependency order each frame. This replaces the monolithic GameApp update loop
// with a data-driven, composable pipeline.
//
// DESIGN:
//   - Systems declare their run phase (PreUpdate, Update, PostUpdate, Render)
//     so the scheduler can group and order them correctly.
//   - Systems declare dependencies by name. The scheduler topologically sorts
//     them within each phase, catching cycles at registration time.
//   - Each system receives a `SystemContext` with immutable World borrow, dt,
//     time, and settings. Mutable systems request &mut World via a separate
//     `SystemMut` trait to make borrow conflicts explicit.
//   - Systems are identified by &'static str name for cheap comparison.
// ──────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;

use hecs::World;

use crate::settings::RuntimeSettings;

// ── Execution phases ─────────────────────────────────────────────────────────
// Engine systems run in this fixed order each frame. Within a phase,
// dependency order determines the actual sequence.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Phase {
    /// Before simulation (input processing, event dispatch).
    PreUpdate,
    /// Core simulation (physics, animation, scripting, weather, particles).
    Update,
    /// After simulation (post-physics cleanup, event finalization).
    PostUpdate,
    /// GPU submission (rendering, compute).
    Render,
}

impl Phase {
    pub fn all() -> &'static [Phase] {
        &[Phase::PreUpdate, Phase::Update, Phase::PostUpdate, Phase::Render]
    }
}

// ── SystemContext (immutable borrow) ─────────────────────────────────────────
// Passed to read-only systems each frame.

pub struct SystemContext<'a> {
    pub world: &'a World,
    pub dt: f32,
    pub time_s: f32,
    pub settings: &'a RuntimeSettings,
}

// ── System trait (immutable systems) ─────────────────────────────────────────
// Systems that only READ the world implement this trait.
// Most engine systems should use SystemMut for full access.

pub trait System {
    /// Human-readable name for debugging and dependency resolution.
    fn name(&self) -> &'static str;

    /// Which phase this system runs in.
    fn phase(&self) -> Phase;

    /// Names of systems that must run before this one.
    /// The scheduler enforces these orderings within the phase.
    fn dependencies(&self) -> Vec<&'static str> {
        vec![]
    }

    /// Called once per frame. The scheduler guarantees dependencies have
    /// already run (and &mut World is available — see SystemMut).
    fn run(&self, ctx: &mut SystemContext);
}

// ── SystemMut trait (mutable systems) ────────────────────────────────────────
// Systems that WRITE to the world implement this trait.
// The scheduler runs mutable systems one at a time within each phase
// (exclusive World access).

pub trait SystemMut {
    /// Human-readable name for debugging and dependency resolution.
    fn name(&self) -> &'static str;

    /// Which phase this system runs in.
    fn phase(&self) -> Phase;

    /// Names of systems that must run before this one.
    fn dependencies(&self) -> Vec<&'static str> {
        vec![]
    }

    /// Called once per frame with exclusive World access.
    fn run(&self, world: &mut World, dt: f32, time_s: f32, settings: &RuntimeSettings);
}

// ── SystemScheduler ──────────────────────────────────────────────────────────
// Owns all registered systems. Resolves dependency order at registration time
// and runs them in order each frame.

pub struct SystemScheduler {
    /// Mutable systems in registration order (will be topologically sorted).
    mut_systems: Vec<Box<dyn SystemMut>>,
    /// Pre-computed execution order (indices into mut_systems).
    execution_order: Vec<usize>,
    /// Name -> index mapping for dependency resolution.
    name_to_index: HashMap<&'static str, usize>,
    /// Detected dependency cycles (empty = no cycles).
    cycles: Vec<String>,
}

impl SystemScheduler {
    pub fn new() -> Self {
        Self {
            mut_systems: Vec::new(),
            execution_order: Vec::new(),
            name_to_index: HashMap::new(),
            cycles: Vec::new(),
        }
    }

    /// Register a mutable system. Panics if name is already registered.
    pub fn register(&mut self, system: Box<dyn SystemMut>) {
        let name = system.name();
        assert!(
            !self.name_to_index.contains_key(name),
            "System '{}' is already registered",
            name
        );
        let idx = self.mut_systems.len();
        self.name_to_index.insert(name, idx);
        self.mut_systems.push(system);
    }

    /// Resolve topological order from dependencies. Call after all systems
    /// are registered. Returns Ok(()) on success, or a list of cycles.
    pub fn resolve(&mut self) -> Result<(), Vec<String>> {
        self.cycles.clear();
        let n = self.mut_systems.len();
        let mut phase_groups: HashMap<Phase, Vec<usize>> = HashMap::new();

        for idx in 0..n {
            let phase = self.mut_systems[idx].phase();
            phase_groups.entry(phase).or_default().push(idx);
        }

        let mut sorted = Vec::with_capacity(n);

        for phase in Phase::all() {
            let Some(group) = phase_groups.get(phase) else { continue };
            // Kahn's algorithm for topological sort within this phase.
            // Build adjacency for this phase group.
            // Only intra-phase dependencies create ordering constraints.
            // Cross-phase deps are guaranteed by phase ordering (PreUpdate < Update < PostUpdate < Render).
            let mut in_degree: HashMap<usize, usize> = HashMap::new();
            let mut edges: HashMap<usize, Vec<usize>> = HashMap::new();
            for &i in group {
                in_degree.entry(i).or_insert(0);
                for dep_name in self.mut_systems[i].dependencies() {
                    if let Some(&dep_idx) = self.name_to_index.get(dep_name) {
                        // Only enforce if dependency is in the SAME phase.
                        let dep_phase = self.mut_systems[dep_idx].phase();
                        if dep_phase == *phase {
                            edges.entry(dep_idx).or_default().push(i);
                            *in_degree.entry(i).or_insert(0) += 1;
                        }
                    }
                }
            }
            let mut queue: Vec<usize> = group.iter()
                .copied()
                .filter(|i| in_degree.get(i).copied().unwrap_or(0) == 0)
                .collect();
            queue.sort_by_key(|i| self.mut_systems[*i].name());
            let mut phase_sorted = Vec::new();
            while let Some(node) = queue.pop() {
                phase_sorted.push(node);
                if let Some(deps) = edges.get(&node) {
                    for &dep in deps {
                        let deg = in_degree.entry(dep).or_insert(0);
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(dep);
                        }
                    }
                }
            }
            if phase_sorted.len() != group.len() {
                // Cycle detected — report it.
                for &i in group {
                    if !phase_sorted.contains(&i) {
                        let deps: Vec<String> = self.mut_systems[i].dependencies()
                            .iter()
                            .filter(|d| {
                                // Only report deps in the same phase as potential cycles.
                                if let Some(&dep_idx) = self.name_to_index.get(**d) {
                                    self.mut_systems[dep_idx].phase() == *phase
                                } else {
                                    false
                                }
                            })
                            .map(|d| d.to_string())
                            .collect();
                        self.cycles.push(format!(
                            "System '{}' (phase {:?}) has unresolvable deps: {:?}",
                            self.mut_systems[i].name(), *phase, deps
                        ));
                    }
                }
            }
            sorted.extend(phase_sorted);
        }

        if self.cycles.is_empty() {
            self.execution_order = sorted;
            Ok(())
        } else {
            Err(self.cycles.clone())
        }
    }

    /// Run all registered systems in resolved order.
    pub fn run_all(&self, world: &mut World, dt: f32, time_s: f32, settings: &RuntimeSettings) {
        for &idx in &self.execution_order {
            self.mut_systems[idx].run(world, dt, time_s, settings);
        }
    }

    /// Number of registered systems.
    pub fn count(&self) -> usize {
        self.mut_systems.len()
    }

    /// Get the resolved execution order as (name, phase) pairs.
    pub fn execution_order_info(&self) -> Vec<(&str, Phase)> {
        self.execution_order.iter()
            .map(|&i| (self.mut_systems[i].name(), self.mut_systems[i].phase()))
            .collect()
    }

    /// Check if any dependency cycles were detected.
    pub fn has_cycles(&self) -> bool {
        !self.cycles.is_empty()
    }

    /// Get cycle error messages.
    pub fn cycles(&self) -> &[String] {
        &self.cycles
    }
}

impl Default for SystemScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    struct TestPhysics;
    impl SystemMut for TestPhysics {
        fn name(&self) -> &'static str { "physics" }
        fn phase(&self) -> Phase { Phase::Update }
        fn run(&self, _w: &mut World, _dt: f32, _ts: f32, _s: &RuntimeSettings) {}
    }

    struct TestAnimation;
    impl SystemMut for TestAnimation {
        fn name(&self) -> &'static str { "animation" }
        fn phase(&self) -> Phase { Phase::Update }
        fn dependencies(&self) -> Vec<&'static str> { vec!["physics"] }
        fn run(&self, _w: &mut World, _dt: f32, _ts: f32, _s: &RuntimeSettings) {}
    }

    struct TestParticles;
    impl SystemMut for TestParticles {
        fn name(&self) -> &'static str { "particles" }
        fn phase(&self) -> Phase { Phase::Update }
        fn dependencies(&self) -> Vec<&'static str> { vec!["physics"] }
        fn run(&self, _w: &mut World, _dt: f32, _ts: f32, _s: &RuntimeSettings) {}
    }

    struct TestRender;
    impl SystemMut for TestRender {
        fn name(&self) -> &'static str { "render" }
        fn phase(&self) -> Phase { Phase::Render }
        fn dependencies(&self) -> Vec<&'static str> { vec!["physics", "animation", "particles"] }
        fn run(&self, _w: &mut World, _dt: f32, _ts: f32, _s: &RuntimeSettings) {}
    }

    #[test]
    fn scheduler_resolves_dependency_order() {
        let mut sched = SystemScheduler::new();
        sched.register(Box::new(TestRender));
        sched.register(Box::new(TestParticles));
        sched.register(Box::new(TestAnimation));
        sched.register(Box::new(TestPhysics));
        assert!(sched.resolve().is_ok());
        let order = sched.execution_order_info();
        let names: Vec<&str> = order.iter().map(|(n, _)| *n).collect();
        // physics must come before animation, particles, and render
        let physics_pos = names.iter().position(|&n| n == "physics").unwrap();
        let anim_pos = names.iter().position(|&n| n == "animation").unwrap();
        let particle_pos = names.iter().position(|&n| n == "particles").unwrap();
        let render_pos = names.iter().position(|&n| n == "render").unwrap();
        assert!(physics_pos < anim_pos);
        assert!(physics_pos < particle_pos);
        assert!(physics_pos < render_pos);
        assert!(anim_pos < render_pos);
        assert!(particle_pos < render_pos);
    }

    #[test]
    fn scheduler_detects_cycle() {
        struct SysA;
        impl SystemMut for SysA {
            fn name(&self) -> &'static str { "a" }
            fn phase(&self) -> Phase { Phase::Update }
            fn dependencies(&self) -> Vec<&'static str> { vec!["b"] }
            fn run(&self, _w: &mut World, _dt: f32, _ts: f32, _s: &RuntimeSettings) {}
        }
        struct SysB;
        impl SystemMut for SysB {
            fn name(&self) -> &'static str { "b" }
            fn phase(&self) -> Phase { Phase::Update }
            fn dependencies(&self) -> Vec<&'static str> { vec!["a"] }
            fn run(&self, _w: &mut World, _dt: f32, _ts: f32, _s: &RuntimeSettings) {}
        }
        let mut sched = SystemScheduler::new();
        sched.register(Box::new(SysA));
        sched.register(Box::new(SysB));
        assert!(sched.resolve().is_err());
        assert!(sched.has_cycles());
    }
}
