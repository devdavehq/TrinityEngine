use crate::components::{Position, Renderable};
use crate::jobs::JobSystem;
use hecs::World;

pub mod skeletal;
pub mod blending;
pub mod anim_graph;

#[derive(Clone, Copy, Debug)]
pub enum AnimState {
    Idle,
    Walk,
    Run,
}

#[derive(Clone, Copy, Debug)]
pub struct Animator {
    pub state: AnimState,
    pub speed: f32,
    pub phase: f32,
    pub base_scale_y: f32,
    pub base_pos_y: f32,
}

impl Animator {
    pub fn humanoid_default() -> Self {
        Self {
            state: AnimState::Idle,
            speed: 1.0,
            phase: 0.0,
            base_scale_y: 1.0,
            base_pos_y: f32::NAN,
        }
    }
}

pub fn animation_system(world: &mut World, dt: f32, jobs: &JobSystem) {
    // Foundation: in future we evaluate bone poses in parallel jobs here.
    let _parallel_enabled = jobs.enabled();

    for (anim, pos, renderable) in world.query_mut::<(&mut Animator, &mut Position, &mut Renderable)>() {
        if anim.base_pos_y.is_nan() {
            anim.base_pos_y = pos.y;
        }
        let rate = match anim.state {
            AnimState::Idle => 1.0,
            AnimState::Walk => 2.0,
            AnimState::Run => 3.2,
        } * anim.speed.max(0.1);
        anim.phase += dt * rate;

        match anim.state {
            AnimState::Idle => {
                // Gentle breathing/sway.
                pos.y = anim.base_pos_y + (anim.phase * 1.1).sin() * 0.005;
                renderable.scale[1] = anim.base_scale_y + (anim.phase * 2.0).sin() * 0.02;
            }
            AnimState::Walk => {
                // Slight bob + sway.
                pos.y = anim.base_pos_y + (anim.phase * 6.0).sin() * 0.03;
                renderable.scale[1] = anim.base_scale_y + (anim.phase * 4.0).sin() * 0.03;
            }
            AnimState::Run => {
                // Stronger bob for running.
                pos.y = anim.base_pos_y + (anim.phase * 9.0).sin() * 0.05;
                renderable.scale[1] = anim.base_scale_y + (anim.phase * 7.0).sin() * 0.05;
            }
        }
    }
}
