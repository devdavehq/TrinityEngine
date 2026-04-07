use crate::components::{Position, Renderable};
use crate::jobs::JobSystem;
use hecs::World;

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
}

impl Animator {
    pub fn humanoid_default() -> Self {
        Self {
            state: AnimState::Idle,
            speed: 1.0,
            phase: 0.0,
            base_scale_y: 1.0,
        }
    }
}

pub fn animation_system(world: &mut World, dt: f32, jobs: &JobSystem) {
    // Foundation: in future we evaluate bone poses in parallel jobs here.
    let _parallel_enabled = jobs.enabled();

    for (anim, pos, renderable) in world.query_mut::<(&mut Animator, &mut Position, &mut Renderable)>() {
        let rate = match anim.state {
            AnimState::Idle => 1.0,
            AnimState::Walk => 2.0,
            AnimState::Run => 3.2,
        } * anim.speed.max(0.1);
        anim.phase += dt * rate;

        match anim.state {
            AnimState::Idle => {
                // Gentle breathing/sway.
                renderable.scale[1] = anim.base_scale_y + (anim.phase * 2.0).sin() * 0.02;
            }
            AnimState::Walk => {
                // Slight bob + sway.
                pos.y += (anim.phase * 6.0).sin() * 0.004;
                renderable.scale[1] = anim.base_scale_y + (anim.phase * 4.0).sin() * 0.03;
            }
            AnimState::Run => {
                // Stronger bob for running.
                pos.y += (anim.phase * 9.0).sin() * 0.007;
                renderable.scale[1] = anim.base_scale_y + (anim.phase * 7.0).sin() * 0.05;
            }
        }
    }
}
