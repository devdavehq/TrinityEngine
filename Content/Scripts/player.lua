local move_speed = 6.0
local jump_speed = 7.5

function start(entity)
    log("player script ready for entity " .. entity)
end

function update(entity, dt)
    local vx, vy, vz = get_velocity(entity)
    local input_x = 0.0

    if is_key_held("A") then
        input_x = input_x - 1.0
    end
    if is_key_held("D") then
        input_x = input_x + 1.0
    end

    set_velocity(entity, input_x * move_speed, vy, vz)

    if is_key_held("Space") and is_on_ground(entity) then
        set_velocity(entity, input_x * move_speed, jump_speed, vz)
    end
end

function on_collision_enter(entity, other, nx, ny, nz, penetration)
    log("player collision enter: other=" .. other .. " normal=(" .. nx .. "," .. ny .. "," .. nz .. ")")
end

function on_collision_stay(entity, other, nx, ny, nz, penetration)
    set_ui_value("player_contact_depth", penetration)
end

function on_collision_exit(entity, other, nx, ny, nz, penetration)
    set_ui_value("player_contact_depth", 0.0)
end
