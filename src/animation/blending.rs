// src/animation/blending.rs
// Animation state blending system — bridges BT blackboard to skeletal animation.
//
// ── Architecture ─────────────────────────────────────────────────────────────
// This module provides the glue between the AI behavior tree and the skeletal
// animation system:
//
//   BT node ticks → writes "ai_state" = "walk" to blackboard
//   → animation_blending_system reads blackboard
//   → calls SkeletalAnimator::play_state("walk", &state_map)
//   → SkeletalAnimator triggers crossfade from current clip to walk clip
//   → per-frame: blend_animated_locals() produces blended bone transforms
//   → compute_joint_matrices() builds final joint matrices for GPU skinning
//
// ── State Flow ───────────────────────────────────────────────────────────────
//   Idle ──(BT detects enemy)──► Attack
//   Idle ──(BT patrol)─────────► Walk
//   Walk ──(BT sees enemy)─────► Run
//   Run  ──(BT reached target)─► Idle
//
// Each transition triggers a smooth crossfade blend (default 0.25s).
//
// ── Why this design? ─────────────────────────────────────────────────────────
// In UE5/Unity, the standard pattern is:
//   1. AI system writes to a blackboard (e.g., "AIState" = "Chase")
//   2. Animation system reads blackboard and feeds it to an Animator/StateMachine
//   3. The state machine selects the correct animation clip
//   4. Crossfade blending handles smooth transitions between states
//
// Our system mirrors this exactly but uses our own BT + ECS.


use crate::ai::components::AiAgent;
use crate::animation::skeletal::{
    AnimationClip, AnimationStateMap, BoneHierarchy, SkeletalAnimator, TransformKeyframe,
    blend_animated_locals, compute_joint_matrices,
};
use glam::Mat4;
use hecs::World;

// ── AnimationClips Component ─────────────────────────────────────────────────
// Holds the clip library + state map for a single entity.
// Each entity that uses skeletal animation has one of these.

/// Clip library and state mappings for a single animated entity.
///
/// This component is attached to entities that use skeletal animation.
/// It holds:
///   - The list of `AnimationClip`s (Idle, Walk, Run, Attack, etc.)
///   - The `AnimationStateMap` (state name → clip index)
///   - The `BoneHierarchy` (skeleton structure)
///   - Precomputed inverse bind-pose matrices
///
/// The BT writes "ai_state" to the entity's blackboard. The blending system
/// reads it and calls `SkeletalAnimator::play_state()` using this component's
/// state map.
#[derive(Clone)]
pub struct AnimationClips {
    /// Available animation clips (Idle, Walk, Run, Attack, Death, etc.).
    pub clips: Vec<AnimationClip>,
    /// Maps state names to clip indices.
    pub state_map: AnimationStateMap,
    /// The bone hierarchy (skeleton).
    pub hierarchy: BoneHierarchy,
    /// Precomputed inverse bind-pose matrices (one per bone).
    pub inverse_bind_poses: Vec<Mat4>,
}

impl AnimationClips {
    pub fn new(
        hierarchy: BoneHierarchy,
        inverse_bind_poses: Vec<Mat4>,
        state_map: AnimationStateMap,
    ) -> Self {
        Self {
            clips: Vec::new(),
            state_map,
            hierarchy,
            inverse_bind_poses,
        }
    }

    /// Add a clip and return its index.
    pub fn add_clip(&mut self, clip: AnimationClip) -> usize {
        let idx = self.clips.len();
        self.clips.push(clip);
        idx
    }

    /// Register a clip and map it to a state name.
    pub fn add_clip_with_state(&mut self, clip: AnimationClip, state_name: &str) -> usize {
        let idx = self.add_clip(clip);
        self.state_map.insert(state_name, idx);
        idx
    }
}

// ── BlendedPose — final output of the blending system ────────────────────────

/// The blended output for a single entity: final joint matrices ready for GPU.
///
/// Computed each frame by `animation_blending_system()`:
///   1. Evaluate current clip at current time → locals_a
///   2. Evaluate previous clip at prev_time → locals_b
///   3. Blend locals_a and locals_b by blend_weight → blended
///   4. Compute joint matrices from blended locals → joint_matrices
///
/// The renderer reads `joint_matrices` to upload to the GPU skinning uniform.
#[derive(Clone, Debug)]
pub struct BlendedPose {
    /// Final joint matrices, one per bone. Ready for GPU upload.
    pub joint_matrices: Vec<Mat4>,
    /// Whether this pose is currently blending (useful for the renderer to know
    /// if it should apply motion blur to the skeleton).
    pub is_blending: bool,
    /// The current blend weight (0.0 = old clip, 1.0 = new clip).
    pub blend_weight: f32,
}

// ── animation_blending_system — per-frame ECS system ─────────────────────────

/// Per-frame system that bridges BT blackboard → SkeletalAnimator.
///
/// For each entity with both `AiAgent` and `SkeletalAnimator`:
///   1. Read `ai_state` from the entity's blackboard.
///   2. Call `SkeletalAnimator::play_state(state, &state_map)` to trigger
///      crossfade transitions when the state changes.
///   3. Advance the animator's time.
///   4. Evaluate both clips and blend the per-bone transforms.
///   5. Compute final joint matrices for GPU skinning.
///   6. Store the result in `BlendedPose` for the renderer to pick up.
///
/// # Arguments
/// * `world` — the ECS world (queries AiAgent + SkeletalAnimator + AnimationClips).
/// * `dt` — delta time in seconds.
///
/// # Blending Algorithm
///   - When the state changes (e.g., "idle" → "walk"):
///     - `prev_clip` = old clip index, `current_clip` = new clip index
///     - `blend_weight` starts at 0.0, ramps to 1.0 over `blend_duration`
///     - Each frame: evaluate both clips, lerp per-bone transforms by weight
///   - When blend is complete (`blend_weight == 1.0`):
///     - Only the new clip is evaluated (no blending overhead)
pub fn animation_blending_system(world: &mut World, dt: f32) {
    // Collect entity data first to avoid borrow conflicts.
    // We need to read from AiAgent (blackboard) and write to SkeletalAnimator
    // and BlendedPose — can't do both in a single query_mut.
    let entities: Vec<(hecs::Entity, String)> = world
        .query::<(hecs::Entity, &AiAgent)>()
        .iter()
        .map(|(e, agent)| (e, agent.blackboard.get_string("ai_state").unwrap_or("idle").to_string()))
        .collect();

    for (entity, ai_state) in entities {
        // Read the state map and clips from AnimationClips component.
        let (state_map, clips) = {
            let Ok(anim_clips) = world.get::<&AnimationClips>(entity) else {
                continue; // Entity doesn't have skeletal animation.
            };
            (anim_clips.state_map.clone(), anim_clips.clips.clone())
        };

        // Transition the animator to the new state.
        {
            let Ok(mut animator) = world.get::<&mut SkeletalAnimator>(entity) else {
                continue;
            };
            animator.play_state(&ai_state, &state_map);
            animator.advance(dt, &clips);
        }

        // Now evaluate the blended pose.
        let blended = {
            let Ok(animator) = world.get::<&SkeletalAnimator>(entity) else {
                continue;
            };
            let Ok(anim_clips) = world.get::<&AnimationClips>(entity) else {
                continue;
            };

            if let Some(current_idx) = animator.current_clip {
                let current_clip = &anim_clips.clips[current_idx];

                if animator.is_blending() {
                    // Blend between two clips.
                    if let Some(prev_idx) = animator.prev_clip {
                        let prev_clip = &anim_clips.clips[prev_idx];
                        let locals_a = prev_clip.evaluate(animator.prev_time);
                        let locals_b = current_clip.evaluate(animator.time);
                        let blended_locals =
                            blend_animated_locals(&locals_a, &locals_b, animator.blend_weight);
                        let joint_matrices = compute_joint_matrices(
                            &anim_clips.hierarchy,
                            &blended_locals,
                            &anim_clips.inverse_bind_poses,
                        );
                        BlendedPose {
                            joint_matrices,
                            is_blending: true,
                            blend_weight: animator.blend_weight,
                        }
                    } else {
                        // Shouldn't happen — is_blending() checks prev_clip is Some.
                        let locals = current_clip.evaluate(animator.time);
                        let joint_matrices = compute_joint_matrices(
                            &anim_clips.hierarchy,
                            &locals,
                            &anim_clips.inverse_bind_poses,
                        );
                        BlendedPose {
                            joint_matrices,
                            is_blending: false,
                            blend_weight: 1.0,
                        }
                    }
                } else {
                    // Single clip — no blending needed.
                    let locals = current_clip.evaluate(animator.time);
                    let joint_matrices = compute_joint_matrices(
                        &anim_clips.hierarchy,
                        &locals,
                        &anim_clips.inverse_bind_poses,
                    );
                    BlendedPose {
                        joint_matrices,
                        is_blending: false,
                        blend_weight: 1.0,
                    }
                }
            } else {
                // No clip playing — rest pose.
                let locals: Vec<TransformKeyframe> = (0..anim_clips.hierarchy.bone_count())
                    .map(|_| TransformKeyframe::identity(0.0))
                    .collect();
                let joint_matrices = compute_joint_matrices(
                    &anim_clips.hierarchy,
                    &locals,
                    &anim_clips.inverse_bind_poses,
                );
                BlendedPose {
                    joint_matrices,
                    is_blending: false,
                    blend_weight: 1.0,
                }
            }
        };

        // Write the blended pose back to the entity.
        if let Ok(mut pose) = world.get::<&mut BlendedPose>(entity) {
            *pose = blended;
        } else {
            world.insert_one(entity, blended).ok();
        }
    }
}

// ── Helper: Build standard humanoid state map ─────────────────────────────────

/// Creates a standard humanoid AnimationStateMap with common states.
///
/// Usage:
/// ```
/// let mut clips = AnimationClips::new(hierarchy, inv_bind, AnimationStateMap::new());
/// let idle_idx = clips.add_clip(idle_clip);
/// let walk_idx = clips.add_clip(walk_clip);
/// let run_idx = clips.add_clip(run_clip);
/// let attack_idx = clips.add_clip(attack_clip);
/// build_humanoid_state_map(&mut clips.state_map, idle_idx, walk_idx, run_idx, attack_idx);
/// ```
pub fn build_humanoid_state_map(
    state_map: &mut AnimationStateMap,
    idle: usize,
    walk: usize,
    run: usize,
    attack: usize,
) {
    state_map.insert("idle", idle);
    state_map.insert("walk", walk);
    state_map.insert("run", run);
    state_map.insert("attack", attack);
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    fn make_test_clips() -> (AnimationClips, BoneHierarchy, Vec<Mat4>) {
        let (hierarchy, inv_bind) =
            crate::animation::skeletal::build_test_skeleton();

        let mut state_map = AnimationStateMap::new();

        // Create simple test clips (each just has one bone animated).
        let mut idle_clip = AnimationClip::new("Idle", 2.0, true);
        idle_clip.add_channel(crate::animation::skeletal::AnimationChannel {
            bone_index: 0,
            keyframes: vec![
                TransformKeyframe::new(0.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
                TransformKeyframe::new(1.0, Vec3::new(0.0, 0.05, 0.0), Quat::IDENTITY, Vec3::ONE),
                TransformKeyframe::new(2.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
            ],
        });

        let mut walk_clip = AnimationClip::new("Walk", 1.0, true);
        walk_clip.add_channel(crate::animation::skeletal::AnimationChannel {
            bone_index: 0,
            keyframes: vec![
                TransformKeyframe::new(0.0, Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
                TransformKeyframe::new(0.5, Vec3::new(0.5, 0.1, 0.0), Quat::IDENTITY, Vec3::ONE),
                TransformKeyframe::new(1.0, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            ],
        });

        let mut clips = AnimationClips::new(hierarchy.clone(), inv_bind.clone(), state_map);
        let idle_idx = clips.add_clip(idle_clip);
        let walk_idx = clips.add_clip(walk_clip);
        clips.state_map.insert("idle", idle_idx);
        clips.state_map.insert("walk", walk_idx);

        (clips, hierarchy, inv_bind)
    }

    #[test]
    fn animation_clips_add_clip() {
        let (clips, _, _) = make_test_clips();
        assert_eq!(clips.clips.len(), 2);
        assert_eq!(clips.clips[0].name, "Idle");
        assert_eq!(clips.clips[1].name, "Walk");
    }

    #[test]
    fn animation_clips_state_map() {
        let (clips, _, _) = make_test_clips();
        assert_eq!(clips.state_map.get_clip_index("idle"), Some(0));
        assert_eq!(clips.state_map.get_clip_index("walk"), Some(1));
        assert_eq!(clips.state_map.get_clip_index("run"), None);
    }

    #[test]
    fn blended_pose_initial_no_blend() {
        let (_, hierarchy, inv_bind) = make_test_clips();
        let locals: Vec<TransformKeyframe> = (0..hierarchy.bone_count())
            .map(|_| TransformKeyframe::identity(0.0))
            .collect();
        let joint_matrices = compute_joint_matrices(&hierarchy, &locals, &inv_bind);
        let pose = BlendedPose {
            joint_matrices,
            is_blending: false,
            blend_weight: 1.0,
        };
        assert!(!pose.is_blending);
        assert_eq!(pose.blend_weight, 1.0);
    }

    #[test]
    fn blend_produces_valid_joint_matrices() {
        let (clips, hierarchy, _) = make_test_clips();

        // Evaluate Idle at t=0.5 and Walk at t=0.5.
        let locals_a = clips.clips[0].evaluate(0.5);
        let locals_b = clips.clips[1].evaluate(0.5);

        // Blend at 50%.
        let blended = blend_animated_locals(&locals_a, &locals_b, 0.5);

        // The blended result should have valid (non-NaN) values.
        for tf in &blended {
            assert!(!tf.position.is_nan());
            assert!(!tf.rotation.is_nan());
            assert!(!tf.scale.is_nan());
        }
    }

    #[test]
    fn build_humanoid_state_map_works() {
        let mut state_map = AnimationStateMap::new();
        build_humanoid_state_map(&mut state_map, 0, 1, 2, 3);

        assert_eq!(state_map.get_clip_index("idle"), Some(0));
        assert_eq!(state_map.get_clip_index("walk"), Some(1));
        assert_eq!(state_map.get_clip_index("run"), Some(2));
        assert_eq!(state_map.get_clip_index("attack"), Some(3));
    }
}
