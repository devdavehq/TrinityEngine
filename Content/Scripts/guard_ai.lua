-- Content/Scripts/guard_ai.lua
-- Enemy AI built from engine-native behavior-tree nodes via the `bt.*` API.
-- There is NO hardcoded Enemy/AIController type in the engine — enemy behavior
-- is a behavior tree you compose here and assign to the entity's AiAgent.  The
-- ai_system() ticks that tree every frame.
--
-- ANIMATION: bt.set_state(entity, "chase"/"idle"/"attack"/"hurt") writes
-- "ai_state" to the entity's blackboard.  The animation blending system reads
-- it each frame and crossfades the SkeletalAnimator to the matching clip
-- (see animation/blending.rs and skeletal.rs:474).  So AI decisions drive
-- animation automatically — the exact answer to "how do we animate enemy AI".

function start(entity)
    -- Build the tree once, assign it to this entity's AiAgent.
    build_guard_tree()
    bt.assign(entity, "guard")
    log("guard_ai: tree assigned to " .. entity)
end

-- The Lua bt.* builder is a flat, single-root tree.  Nodes are added in order
-- onto one root; decorators (bt.in_range/cooldown) wrap the previously added
-- node.  Action leaves use blackboard keys set by other systems:
--   * perceive writes "target_pos" (nearest entity with the given tag)
--   * move_to reads "target_pos" and walks the NavGrid A* path
function build_guard_tree()
    bt.create("guard")            -- fresh builder
    bt.sequence("guard")           -- implicit; children run in order

    -- 1. Sense the nearest "player"-tagged entity within 10 units.
    bt.perceive("guard", 10.0, "player")

    -- 2. Chase it (NavGrid A* toward blackboard "target_pos").
    bt.move_to("guard")
end

function update(entity, dt)
    -- Per-frame decision + animation steering that complements the tree.
    local hp, hp_max = get_health(entity)
    if hp <= 0 then
        bt.set_state(entity, "dead")
        return
    end
    if hp < hp_max * 0.25 then
        bt.set_state(entity, "hurt")
    else
        -- If blackboard has a target we're chasing, play the run clip,
        -- otherwise idle. A real game reads bt.get_blackboard_bool/float.
        bt.set_state(entity, "chase")
    end
end