-- player.lua with physics
local jump_force = 6.0
local move_speed = 4.0

function update(entity, dt)
    -- Horizontal movement: apply force left/right.
    -- We set velocity directly for responsive feel.
    -- apply_force adds to velocity; for horizontal we want direct control.
    local vx = 0.0
    if is_key_held("A") or is_key_held("ArrowLeft")  then vx = -move_speed end
    if is_key_held("D") or is_key_held("ArrowRight") then vx =  move_speed end

    -- set_velocity isn't registered yet — we'll use apply_force with reset.
    -- For now: read position, set directly for horizontal, let physics handle vertical.
    local x, y, z = get_position(entity)
    x = x + vx * dt
    set_position(entity, x, y, z)

    -- Jump: only if on the ground.
    -- is_on_ground() checks the RigidBody component.
    if (is_key_held("W") or is_key_held("ArrowUp") or is_key_held("Space"))
        and is_on_ground(entity)
    then
        apply_force(entity, 0.0, jump_force)
    end
end