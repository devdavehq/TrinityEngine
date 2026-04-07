// src/assets/store.rs

use std::collections::HashMap;
use std::marker::PhantomData;

// Handle<T> — a typed ID pointing to an asset in a store.
// Copy + Clone: cheap to duplicate, just a number.
// PartialEq + Eq + Hash: allows using handles as HashMap keys.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Handle<T> {
    pub id: u32,
    // PhantomData makes Handle<Mesh> and Handle<Texture> distinct types
    // without storing any T data. Zero cost at runtime.
    _marker: PhantomData<T>,
}

// AssetStore<T> — a registry of assets indexed by Handle.
pub struct AssetStore<T> {
    assets:  HashMap<u32, T>,
    next_id: u32,
}

impl<T> AssetStore<T> {
    pub fn new() -> Self {
        Self { assets: HashMap::new(), next_id: 0 }
    }

    // add() stores an asset and returns a Handle to reference it later.
    pub fn add(&mut self, asset: T) -> Handle<T> {
        let id = self.next_id;
        self.assets.insert(id, asset);
        self.next_id += 1;
        Handle { id, _marker: PhantomData }
    }

    // get() looks up an asset by handle. Returns None if invalid.
    pub fn get(&self, handle: &Handle<T>) -> Option<&T> {
        self.assets.get(&handle.id)
    }

    // get_mut() for assets that need updating (textures, reloaded meshes).
    pub fn get_mut(&mut self, handle: &Handle<T>) -> Option<&mut T> {
        self.assets.get_mut(&handle.id)
    }

    // replace() swaps an asset in place — used for hot reload.
    // The handle stays valid; the data changes.
    pub fn replace(&mut self, handle: &Handle<T>, new_asset: T) {
        self.assets.insert(handle.id, new_asset);
    }
}