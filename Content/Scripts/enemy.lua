local patrol_phase = 0.0

function start(entity)
    log("enemy script ready for entity " .. entity)
end

function update(entity, dt)
    patrol_phase = patrol_phase + dt
    local _, vy, vz = get_velocity(entity)
    local vx = sin(patrol_phase * 1.2) * 2.0
    set_velocity(entity, vx, vy, vz)
end

function on_collision_enter(entity, other, nx, ny, nz, penetration)
    apply_torque(entity, nx * 0.4)
end
