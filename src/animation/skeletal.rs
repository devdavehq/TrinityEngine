// src/animation/skeletal.rs
// Skeletal animation system — bone hierarchies, animation clips, skinning matrices.
//
// ── Architecture ─────────────────────────────────────────────────────────────
// A skeletal mesh has:
//   1. A BoneHierarchy — a tree of bones with parent indices.
//   2. A RestPose — the default T-pose or A-pose transform of each bone.
//   3. One or more AnimationClips — keyframed transforms per bone over time.
//   4. A SkeletalAnimator — tracks which clip is playing, current time, speed.
//
// Each frame:
//   SkeletalAnimator advances time → evaluate AnimationClip → local bone
//   transforms → multiply up the hierarchy → final JointMatrices → upload
//   to GPU via a uniform buffer or vertex texture.
//
// ── Data flow ────────────────────────────────────────────────────────────────
//   AnimationClip + time → local transforms
//   BoneHierarchy + rest pose + local transforms → JointMatrices (world-space)
//   JointMatrices → GPU uniform buffer → vertex shader skinning
//
// ── Why skeletal animation? ─────────────────────────────────────────────────
// Procedural animation (breathing, bob) is great for simple movement, but
// characters with articulated limbs need bone-driven deformation. Skeletal
// animation lets animators author complex motions (walk, run, attack) in
// external tools (Blender, Maya) and play them back in-engine.
//
// ── Performance notes ────────────────────────────────────────────────────────
// Joint matrix computation is O(bones) per entity. With 64 bones and 100
// characters, that's 6400 matrix multiplies per frame — trivial for the CPU.
// GPU skinning (vertex shader) is the standard approach for real-time games.
// For now we compute CPU-side JointMatrices and leave GPU upload for the
// renderer integration phase.

use glam::{Quat, Vec3};

// ── Bone ─────────────────────────────────────────────────────────────────────

/// A single bone in the skeleton hierarchy.
#[derive(Clone, Debug)]
pub struct Bone {
    /// Human-readable name (e.g., "RightArm", "Spine02").
    pub name: String,
    /// Index of the parent bone. `None` for the root bone.
    pub parent: Option<u16>,
    /// Rest pose transform (T-pose). Relative to parent.
    pub rest_position: Vec3,
    pub rest_rotation: Quat,
    pub rest_scale: Vec3,
}

impl Bone {
    pub fn new(name: &str, parent: Option<u16>) -> Self {
        Self {
            name: name.to_string(),
            parent,
            rest_position: Vec3::ZERO,
            rest_rotation: Quat::IDENTITY,
            rest_scale: Vec3::ONE,
        }
    }

    /// Local rest transform as a 4×4 matrix.
    pub fn rest_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(
            self.rest_scale,
            self.rest_rotation,
            self.rest_position,
        )
    }
}

// ── Bone Hierarchy ───────────────────────────────────────────────────────────

/// The full skeleton — an ordered list of bones forming a tree.
#[derive(Clone, Debug)]
pub struct BoneHierarchy {
    pub bones: Vec<Bone>,
}

impl BoneHierarchy {
    pub fn new() -> Self {
        Self { bones: Vec::new() }
    }

    /// Add a bone and return its index.
    pub fn add_bone(&mut self, bone: Bone) -> u16 {
        let idx = self.bones.len() as u16;
        self.bones.push(bone);
        idx
    }

    /// Number of bones in the skeleton.
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Find a bone by name. Returns (index, bone).
    pub fn find(&self, name: &str) -> Option<(u16, &Bone)> {
        self.bones
            .iter()
            .enumerate()
            .find(|(_, b)| b.name == name)
            .map(|(i, b)| (i as u16, b))
    }

    /// Get the ancestor chain for a bone (bone itself → parent → grandparent → ...).
    pub fn ancestor_chain(&self, bone_idx: u16) -> Vec<u16> {
        let mut chain = Vec::new();
        let mut current = Some(bone_idx);
        while let Some(idx) = current {
            chain.push(idx);
            current = self.bones[idx as usize].parent;
        }
        chain
    }
}

impl Default for BoneHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

// ── Transform Keyframe ───────────────────────────────────────────────────────

/// A single keyframe for a bone transform.
#[derive(Clone, Debug)]
pub struct TransformKeyframe {
    /// Time in seconds.
    pub time: f32,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl TransformKeyframe {
    pub fn new(time: f32, position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self { time, position, rotation, scale }
    }

    /// Default keyframe at time 0 with identity transform.
    pub fn identity(time: f32) -> Self {
        Self::new(time, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }
}

// ── Animation Channel ────────────────────────────────────────────────────────

/// Keyframes for a single bone within an animation clip.
#[derive(Clone, Debug)]
pub struct AnimationChannel {
    /// Index into BoneHierarchy.bones that this channel targets.
    pub bone_index: u16,
    /// Sorted keyframes (by time).
    pub keyframes: Vec<TransformKeyframe>,
}

impl AnimationChannel {
    pub fn new(bone_index: u16) -> Self {
        Self {
            bone_index,
            keyframes: Vec::new(),
        }
    }

    /// Evaluate this channel at a given time, returning the local transform.
    /// Keyframes are linearly interpolated (lerp position/scale, slerp rotation).
    pub fn evaluate(&self, time: f32) -> TransformKeyframe {
        if self.keyframes.is_empty() {
            return TransformKeyframe::identity(0.0);
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].clone();
        }

        // Find the two keyframes surrounding the current time.
        let mut prev = &self.keyframes[0];
        let mut next = &self.keyframes[self.keyframes.len() - 1];

        for i in 0..self.keyframes.len() - 1 {
            if time >= self.keyframes[i].time && time <= self.keyframes[i + 1].time {
                prev = &self.keyframes[i];
                next = &self.keyframes[i + 1];
                break;
            }
        }

        // Interpolation factor within the keyframe range.
        let range = next.time - prev.time;
        let t = if range > 0.0001 {
            ((time - prev.time) / range).clamp(0.0, 1.0)
        } else {
            0.0
        };

        TransformKeyframe::new(
            time,
            prev.position.lerp(next.position, t),
            prev.rotation.slerp(next.rotation, t),
            prev.scale.lerp(next.scale, t),
        )
    }
}

// ── Animation Clip ───────────────────────────────────────────────────────────

/// A complete animation — e.g., "Walk", "Run", "Idle", "Attack".
#[derive(Clone, Debug)]
pub struct AnimationClip {
    /// Human-readable name.
    pub name: String,
    /// Duration in seconds.
    pub duration: f32,
    /// Whether this clip loops.
    pub looping: bool,
    /// One channel per animated bone.
    pub channels: Vec<AnimationChannel>,
}

impl AnimationClip {
    pub fn new(name: &str, duration: f32, looping: bool) -> Self {
        Self {
            name: name.to_string(),
            duration,
            looping,
            channels: Vec::new(),
        }
    }

    /// Add a channel to this clip.
    pub fn add_channel(&mut self, channel: AnimationChannel) {
        self.channels.push(channel);
    }

    /// Evaluate all channels at a given time, returning per-bone local transforms.
    /// The returned Vec is indexed by bone index.
    pub fn evaluate(&self, time: f32) -> Vec<TransformKeyframe> {
        // We'll fill in a default for every bone, then overwrite with channels.
        // This is simple; for large skeletons with sparse animation, a HashMap
        // would be more memory-efficient, but Vec is cache-friendly.
        let mut result: Vec<TransformKeyframe> = Vec::new();

        // Find max bone index to size the result.
        let max_bone = self.channels.iter()
            .map(|c| c.bone_index as usize)
            .max()
            .unwrap_or(0);

        // Initialize with identity.
        result.resize_with(max_bone + 1, || TransformKeyframe::identity(0.0));

        // Evaluate each channel.
        for channel in &self.channels {
            let idx = channel.bone_index as usize;
            if idx < result.len() {
                result[idx] = channel.evaluate(time);
            }
        }

        result
    }

    /// Compute the effective time, handling looping.
    pub fn effective_time(&self, raw_time: f32) -> f32 {
        if self.looping && self.duration > 0.0 {
            raw_time % self.duration
        } else {
            raw_time.clamp(0.0, self.duration)
        }
    }
}

// ── Skeletal Animator ────────────────────────────────────────────────────────

/// Runtime state for playing back an animation clip on an entity.
#[derive(Clone, Debug)]
pub struct SkeletalAnimator {
    /// Index into the clips array (set by the system that owns the clips).
    pub current_clip: Option<usize>,
    /// Current playback time in seconds.
    pub time: f32,
    /// Playback speed multiplier. 1.0 = normal, 0.5 = half speed, -1.0 = reverse.
    pub speed: f32,
    /// Blending weight when blending between two clips (0.0 = old, 1.0 = new).
    pub blend_weight: f32,
    /// Index of the previous clip being blended from.
    pub prev_clip: Option<usize>,
    /// Time of the previous clip during blending.
    pub prev_time: f32,
}

impl Default for SkeletalAnimator {
    fn default() -> Self {
        Self {
            current_clip: None,
            time: 0.0,
            speed: 1.0,
            blend_weight: 1.0,
            prev_clip: None,
            prev_time: 0.0,
        }
    }
}

impl SkeletalAnimator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance time by dt seconds.
    pub fn advance(&mut self, dt: f32, clips: &[AnimationClip]) {
        self.time += dt * self.speed;

        // Handle blend weight transition.
        if self.blend_weight < 1.0 {
            self.blend_weight = (self.blend_weight + dt * 4.0).min(1.0); // 0.25s blend.
        }

        if let Some(idx) = self.current_clip {
            let clip = &clips[idx];
            self.time = clip.effective_time(self.time);
        }
    }

    /// Switch to a new clip, starting from time 0.
    /// The old clip becomes the blend source.
    pub fn play(&mut self, clip_index: usize, _clips: &[AnimationClip]) {
        if self.current_clip == Some(clip_index) {
            return; // Already playing this clip.
        }
        // Move current clip to blend source.
        self.prev_clip = self.current_clip;
        self.prev_time = self.time;
        self.current_clip = Some(clip_index);
        self.time = 0.0;
        self.blend_weight = 0.0; // Start blending from old to new.
    }

    /// Force-set the clip (no blending, instant switch).
    pub fn play_immediate(&mut self, clip_index: usize) {
        self.current_clip = Some(clip_index);
        self.time = 0.0;
        self.blend_weight = 1.0;
        self.prev_clip = None;
    }
}

// ── Joint Matrix Computation ─────────────────────────────────────────────────

/// Compute the final world-space joint matrices for a skeleton.
///
/// Given a bone hierarchy, rest pose, and per-bone local animation transforms,
/// this function walks the hierarchy and produces the inverse-bind-pose ×
/// world-transform matrix for each bone — exactly what the GPU needs for
/// linear blend skinning (LBS).
///
/// # Arguments
/// * `hierarchy` — the bone tree.
/// * `animated_locals` — per-bone local transforms from the animation clip.
/// * `inverse_bind_poses` — precomputed inverse bind-pose matrices (one per bone).
///
/// # Returns
/// A Vec<Mat4> of joint matrices, one per bone, ready for GPU upload.
pub fn compute_joint_matrices(
    hierarchy: &BoneHierarchy,
    animated_locals: &[TransformKeyframe],
    inverse_bind_poses: &[glam::Mat4],
) -> Vec<glam::Mat4> {
    let bone_count = hierarchy.bones.len();
    assert_eq!(animated_locals.len(), bone_count);
    assert_eq!(inverse_bind_poses.len(), bone_count);

    // Step 1: Compute local transform matrices from animated keyframes + rest pose.
    let local_matrices: Vec<glam::Mat4> = (0..bone_count)
        .map(|i| {
            let bone = &hierarchy.bones[i];
            let anim = &animated_locals[i];

            // Combine rest pose with animation delta.
            let anim_mat = glam::Mat4::from_scale_rotation_translation(
                anim.scale,
                anim.rotation,
                anim.position,
            );
            bone.rest_matrix() * anim_mat
        })
        .collect();

    // Step 2: Accumulate world-space transforms down the hierarchy.
    let mut world_matrices = vec![glam::Mat4::IDENTITY; bone_count];
    for i in 0..bone_count {
        if let Some(parent_idx) = hierarchy.bones[i].parent {
            world_matrices[i] = world_matrices[parent_idx as usize] * local_matrices[i];
        } else {
            world_matrices[i] = local_matrices[i];
        }
    }

    // Step 3: Compute final joint matrix = inverse_bind_pose × world_transform.
    // This transforms from bind-pose space to the current animated pose.
    (0..bone_count)
        .map(|i| world_matrices[i] * inverse_bind_poses[i])
        .collect()
}

// ── Helper: Build a humanoid skeleton ────────────────────────────────────────

/// Creates a simple humanoid bone hierarchy for testing.
/// This is not anatomically accurate — it's a minimal skeleton for validating
/// the animation pipeline.
pub fn build_test_skeleton() -> (BoneHierarchy, Vec<glam::Mat4>) {
    let mut h = BoneHierarchy::new();

    // Root
    let root = h.add_bone(Bone::new("Hips", None));

    // Spine
    let spine1 = h.add_bone(Bone::new("Spine01", Some(root)));
    let spine2 = h.add_bone(Bone::new("Spine02", Some(spine1)));
    let neck = h.add_bone(Bone::new("Neck", Some(spine2)));
    let _head = h.add_bone(Bone::new("Head", Some(neck)));

    // Left arm
    let l_shoulder = h.add_bone(Bone::new("LeftShoulder", Some(spine2)));
    let l_upper = h.add_bone(Bone::new("LeftUpperArm", Some(l_shoulder)));
    let l_lower = h.add_bone(Bone::new("LeftLowerArm", Some(l_upper)));
    let _l_hand = h.add_bone(Bone::new("LeftHand", Some(l_lower)));

    // Right arm
    let r_shoulder = h.add_bone(Bone::new("RightShoulder", Some(spine2)));
    let r_upper = h.add_bone(Bone::new("RightUpperArm", Some(r_shoulder)));
    let r_lower = h.add_bone(Bone::new("RightLowerArm", Some(r_upper)));
    let _r_hand = h.add_bone(Bone::new("RightHand", Some(r_lower)));

    // Left leg
    let l_thigh = h.add_bone(Bone::new("LeftThigh", Some(root)));
    let l_shin = h.add_bone(Bone::new("LeftShin", Some(l_thigh)));
    let _l_foot = h.add_bone(Bone::new("LeftFoot", Some(l_shin)));

    // Right leg
    let r_thigh = h.add_bone(Bone::new("RightThigh", Some(root)));
    let r_shin = h.add_bone(Bone::new("RightShin", Some(r_thigh)));
    let _r_foot = h.add_bone(Bone::new("RightFoot", Some(r_shin)));

    // Generate identity inverse bind poses for testing.
    let inverse_bind_poses = vec![glam::Mat4::IDENTITY; h.bone_count()];

    (h, inverse_bind_poses)
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bone_rest_matrix() {
        let bone = Bone {
            name: "TestBone".into(),
            parent: None,
            rest_position: Vec3::new(1.0, 2.0, 3.0),
            rest_rotation: Quat::IDENTITY,
            rest_scale: Vec3::ONE,
        };
        let m = bone.rest_matrix();
        // The translation column should be (1, 2, 3, 1).
        assert!((m.col(3).x - 1.0).abs() < 0.001);
        assert!((m.col(3).y - 2.0).abs() < 0.001);
        assert!((m.col(3).z - 3.0).abs() < 0.001);
    }

    #[test]
    fn bone_hierarchy_parent_chain() {
        let mut h = BoneHierarchy::new();
        let root = h.add_bone(Bone::new("Root", None));
        let child = h.add_bone(Bone::new("Child", Some(root)));
        let grandchild = h.add_bone(Bone::new("Grandchild", Some(child)));

        let chain = h.ancestor_chain(grandchild);
        assert_eq!(chain, vec![grandchild, child, root]);
    }

    #[test]
    fn bone_find_by_name() {
        let mut h = BoneHierarchy::new();
        h.add_bone(Bone::new("A", None));
        h.add_bone(Bone::new("B", Some(0)));
        let (idx, bone) = h.find("B").unwrap();
        assert_eq!(idx, 1);
        assert_eq!(bone.name, "B");
    }

    #[test]
    fn keyframe_interpolation() {
        let channel = AnimationChannel {
            bone_index: 0,
            keyframes: vec![
                TransformKeyframe::new(0.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
                TransformKeyframe::new(1.0, Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            ],
        };

        // At t=0.5, position should be (5, 0, 0).
        let k = channel.evaluate(0.5);
        assert!((k.position.x - 5.0).abs() < 0.001);
    }

    #[test]
    fn clip_looping() {
        let clip = AnimationClip::new("Test", 2.0, true);
        assert!((clip.effective_time(5.0) - 1.0).abs() < 0.001); // 5.0 % 2.0 = 1.0
        assert!((clip.effective_time(2.0) - 0.0).abs() < 0.001); // 2.0 % 2.0 = 0.0
    }

    #[test]
    fn clip_no_loop_clamped() {
        let clip = AnimationClip::new("Test", 2.0, false);
        assert!((clip.effective_time(5.0) - 2.0).abs() < 0.001); // Clamped at duration.
    }

    #[test]
    fn animator_play_switches_clip() {
        let clips = vec![
            AnimationClip::new("Idle", 2.0, true),
            AnimationClip::new("Walk", 1.0, true),
        ];
        let mut anim = SkeletalAnimator::new();
        anim.play(0, &clips);
        assert_eq!(anim.current_clip, Some(0));
        anim.time = 1.0;
        anim.play(1, &clips);
        assert_eq!(anim.current_clip, Some(1));
        assert_eq!(anim.time, 0.0);
        assert_eq!(anim.prev_clip, Some(0));
        assert!(anim.blend_weight < 1.0);
    }

    #[test]
    fn joint_matrix_identity_rest_pose() {
        let (hierarchy, inv_bind) = build_test_skeleton();
        // Identity animated locals (no animation applied).
        let locals: Vec<TransformKeyframe> = (0..hierarchy.bone_count())
            .map(|_i| TransformKeyframe::identity(0.0))
            .collect();
        let joints = compute_joint_matrices(&hierarchy, &locals, &inv_bind);
        assert_eq!(joints.len(), hierarchy.bone_count());
        // Root joint should be identity (rest pose * identity * inverse_bind).
        // Since inverse_bind = I, root joint = rest_matrix * identity = rest_matrix.
        // For the test skeleton, rest matrices are all identity (default bones).
        assert!((joints[0].col(3).w - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_skeleton_bone_count() {
        let (h, _) = build_test_skeleton();
        // 1 root + 4 spine + 4 left arm + 4 right arm + 3 left leg + 3 right leg = 19
        assert_eq!(h.bone_count(), 19);
    }
}
