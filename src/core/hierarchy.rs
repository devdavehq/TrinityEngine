// src/core/hierarchy.rs
// ── Entity Hierarchy Management ──────────────────────────────────────────
// Tree traversal, parent-child operations, and recursive transform support.

use crate::components::{Children, Parent, Position};
use hecs::Entity;
use std::collections::{HashMap, HashSet};

/// Builds a complete parent→children map from all Parent components in the world.
/// Returns (root_entities, parent_to_children_map).
pub fn build_hierarchy(
    world: &hecs::World,
) -> (
    Vec<Entity>,
    HashMap<Entity, Vec<Entity>>,
) {
    let mut parent_to_children: HashMap<Entity, Vec<Entity>> = HashMap::new();
    let mut children_set: HashSet<Entity> = HashSet::new();

    for (child, parent) in world.query::<(Entity, &Parent)>().iter() {
        parent_to_children
            .entry(parent.entity)
            .or_default()
            .push(child);
        children_set.insert(child);
    }

    let mut roots: Vec<Entity> = world
        .query::<hecs::Entity>()
        .iter()
        .filter(|e| !children_set.contains(e))
        .collect();

    roots.sort_by_key(|e| e.to_bits());

    for children in parent_to_children.values_mut() {
        children.sort_by_key(|e| e.to_bits());
    }

    (roots, parent_to_children)
}

/// Get the full ancestry chain from an entity to the root.
/// Returns [self, parent, grandparent, ...].
pub fn get_ancestry(world: &hecs::World, entity: Entity) -> Vec<Entity> {
    let mut chain = vec![entity];
    let mut current = entity;

    loop {
        match world.get::<&Parent>(current) {
            Ok(parent) => {
                chain.push(parent.entity);
                current = parent.entity;
            }
            Err(_) => break,
        }
    }

    chain
}

/// Check if `ancestor` is an ancestor of `entity` (not including self).
pub fn is_ancestor(world: &hecs::World, entity: Entity, ancestor: Entity) -> bool {
    let chain = get_ancestry(world, entity);
    chain.len() > 1 && chain[1..].contains(&ancestor)
}

/// Get all descendants of an entity (recursive).
pub fn get_descendants(world: &hecs::World, entity: Entity) -> Vec<Entity> {
    let mut descendants = Vec::new();
    let mut stack = vec![entity];

    while let Some(current) = stack.pop() {
        if let Ok(children) = world.get::<&Children>(current) {
            for &child in &children.entities {
                descendants.push(child);
                stack.push(child);
            }
        }
    }

    descendants
}

/// Compute the world-space position of an entity by accumulating parent positions.
pub fn world_position(world: &hecs::World, entity: Entity) -> Option<[f32; 3]> {
    let chain = get_ancestry(world, entity);
    let mut world_pos = [0.0f32; 3];

    for &e in chain.iter().rev() {
        if let Ok(pos) = world.get::<&Position>(e) {
            world_pos[0] += pos.x;
            world_pos[1] += pos.y;
            world_pos[2] += pos.z;
        }
    }

    Some(world_pos)
}

/// Parent an entity under a new parent. Removes from old parent if present.
pub fn set_parent(world: &mut hecs::World, child: Entity, new_parent: Entity) {
    // Remove from old parent.
    if let Ok(old_parent) = world.get::<&Parent>(child) {
        let old = old_parent.entity;
        if let Ok(mut children) = world.get::<&mut Children>(old) {
            children.entities.retain(|&e| e != child);
        }
    }

    // Set new parent.
    let _ = world.insert_one(child, Parent { entity: new_parent });

    // Add to new parent's children list.
    if let Ok(mut children) = world.get::<&mut Children>(new_parent) {
        if !children.entities.contains(&child) {
            children.entities.push(child);
        }
    }
}

/// Unparent an entity (make it a root).
pub fn unparent(world: &mut hecs::World, child: Entity) {
    if let Ok(old_parent) = world.get::<&Parent>(child) {
        let old = old_parent.entity;
        if let Ok(mut children) = world.get::<&mut Children>(old) {
            children.entities.retain(|&e| e != child);
        }
    }

    let _ = world.remove_one::<Parent>(child);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Children, Renderable};

    fn dummy_mesh() -> crate::assets::Handle<crate::assets::Mesh> {
        crate::assets::Handle::new(0)
    }

    fn spawn_simple(world: &mut hecs::World) -> Entity {
        world.spawn((
            Position { x: 0.0, y: 0.0, z: 0.0 },
            Renderable { mesh: dummy_mesh(), color: [1.0; 3], metallic: 0.0, roughness: 0.5, ao: 1.0, scale: [1.0; 3] },
            Children::new(),
        ))
    }

    #[test]
    fn build_hierarchy_finds_roots() {
        let mut world = hecs::World::new();
        let root = spawn_simple(&mut world);
        let child = spawn_simple(&mut world);

        let _ = world.insert_one(child, Parent { entity: root });
        {
            let mut children = world.get::<&mut Children>(root).unwrap();
            children.entities.push(child);
        }

        let (roots, map) = build_hierarchy(&world);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], root);
        assert_eq!(map.get(&root).unwrap().len(), 1);
    }

    #[test]
    fn world_position_accumulates() {
        let mut world = hecs::World::new();
        let root = world.spawn((Position { x: 1.0, y: 0.0, z: 0.0 },));
        let child = world.spawn((Position { x: 0.0, y: 2.0, z: 0.0 },));

        let _ = world.insert_one(child, Parent { entity: root });

        let pos = world_position(&world, child).unwrap();
        assert_eq!(pos, [1.0, 2.0, 0.0]);
    }

    #[test]
    fn unparent_makes_root() {
        let mut world = hecs::World::new();
        let root = spawn_simple(&mut world);
        let child = spawn_simple(&mut world);

        set_parent(&mut world, child, root);
        {
            let (roots, _) = build_hierarchy(&world);
            assert_eq!(roots.len(), 1);
        }

        unparent(&mut world, child);
        let (roots, _) = build_hierarchy(&world);
        assert!(roots.contains(&child));
    }

    #[test]
    fn get_descendants_works() {
        let mut world = hecs::World::new();
        let root = spawn_simple(&mut world);
        let child1 = spawn_simple(&mut world);
        let child2 = spawn_simple(&mut world);

        let _ = world.insert_one(child1, Parent { entity: root });
        let _ = world.insert_one(child2, Parent { entity: root });
        {
            let mut c = world.get::<&mut Children>(root).unwrap();
            c.entities.push(child1);
            c.entities.push(child2);
        }

        let desc = get_descendants(&world, root);
        assert_eq!(desc.len(), 2);
    }
}
