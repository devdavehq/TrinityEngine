use std::collections::HashMap;

use crate::ai::behavior_tree::BehaviorTree;
use crate::ai::blackboard::Blackboard;
use crate::navigation::NavGrid;
use crate::navmesh::NavMesh;
use hecs::World;

// ── AiAgent — ECS component attached to every AI-controlled entity ──────────
// Each agent references a behavior tree by name (looked up in the AiRegistry)
// and carries its own Blackboard for per-entity data (target position, health
// thresholds, patrol index, etc.).
#[derive(Clone)]
pub struct AiAgent {
    /// Name of the behavior tree to use (looked up in AiRegistry).
    pub tree_name: String,
    /// Whether this agent is currently running.
    pub enabled: bool,
    /// Minimum seconds between ticks. 0.0 = every frame.
    pub tick_interval: f32,
    /// Timestamp of the last tick (seconds since engine start).
    pub last_tick: f32,
    /// Per-agent blackboard for storing decision-making state.
    pub blackboard: Blackboard,
}

impl AiAgent {
    pub fn new(tree_name: &str) -> Self {
        Self {
            tree_name: tree_name.to_string(),
            enabled: true,
            tick_interval: 0.0,
            last_tick: 0.0,
            blackboard: Blackboard::new(),
        }
    }
}

// ── AiRegistry — global storage for behavior trees by name ──────────────────
// The engine creates behavior trees once and registers them here.  AiAgent
// components reference trees by name.  This avoids duplicating tree data per
// entity — all agents using the same tree share the template.
pub struct AiRegistry {
    trees: HashMap<String, BehaviorTree>,
}

impl AiRegistry {
    pub fn new() -> Self {
        Self {
            trees: HashMap::new(),
        }
    }

    /// Register a behavior tree under the given name.
    pub fn register(&mut self, name: &str, tree: BehaviorTree) {
        self.trees.insert(name.to_string(), tree);
    }

    /// Get an immutable reference to a registered tree.
    pub fn get(&self, name: &str) -> Option<&BehaviorTree> {
        self.trees.get(name)
    }

    /// Get a mutable reference to a registered tree.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut BehaviorTree> {
        self.trees.get_mut(name)
    }
}

impl Default for AiRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── ai_system — runs every frame in the game loop ───────────────────────────
// Queries all entities with an AiAgent component.  For each enabled agent,
// checks whether enough time has elapsed (tick_interval) and, if so, ticks
// its behavior tree.  The NavGrid is passed through to BTContext so that
// navigation nodes (MoveTo, Patrol) can query paths.
pub fn ai_system(
    world: &mut World,
    registry: &mut AiRegistry,
    nav_grid: &NavGrid,
    navmesh: Option<&NavMesh>,
    dt: f32,
    time_s: f32,
) {
    // Collect entity data first to satisfy the borrow checker:
    // we need &World for the query but &mut World for BTContext.
    let agents: Vec<(hecs::Entity, String, bool, f32, f32)> = world
        .query::<(hecs::Entity, &AiAgent)>()
        .iter()
        .map(|(e, a)| (e, a.tree_name.clone(), a.enabled, a.tick_interval, a.last_tick))
        .collect();

    for (entity, tree_name, enabled, tick_interval, last_tick) in agents {
        if !enabled {
            continue;
        }
        // Skip if the tick interval hasn't elapsed yet.
        if tick_interval > 0.0 && (time_s - last_tick) < tick_interval {
            continue;
        }

        // Look up the behavior tree template in the registry.
        // We only need an immutable reference to tick the tree.
        let Some(_tree) = registry.get(&tree_name) else {
            tracing::warn!(
                "[AI] Entity {:?} references unknown tree '{}'",
                entity, tree_name
            );
            continue;
        };

        // Update last_tick on the agent.
        if let Ok(mut agent) = world.get::<&mut AiAgent>(entity) {
            agent.last_tick = time_s;
        }

        // Build the blackboard by taking it temporarily out of the agent,
        // ticking the tree, and putting it back.
        // This avoids the borrow conflict: we can't borrow the agent
        // immutably (to pass its blackboard) and the world mutably at the
        // same time.
        let mut blackboard = {
            if let Ok(agent) = world.get::<&AiAgent>(entity) {
                agent.blackboard.clone()
            } else {
                continue;
            }
        };

        // Now tick the behavior tree with a mutable BTContext.
        let mut ctx = super::behavior_tree::BTContext {
            entity,
            world,
            dt,
            time_s,
            nav_grid,
            navmesh,
            blackboard: &mut blackboard,
        };

        // Re-borrow the tree mutably for ticking.
        if let Some(tree) = registry.get_mut(&tree_name) {
            let _status = tree.tick(&mut ctx);
        }

        // Write the blackboard back into the agent.
        if let Ok(mut agent) = world.get::<&mut AiAgent>(entity) {
            agent.blackboard = blackboard;
        }
    }
}
