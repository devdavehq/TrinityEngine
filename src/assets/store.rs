// src/assets/store.rs

use std::collections::HashMap;
use std::marker::PhantomData;

// Handle<T> — a typed ID pointing to an asset in a store.
// Copy + Clone: cheap to duplicate, just a number.
// PartialEq + Eq + Hash: allows using handles as HashMap keys.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Handle<T> {
    pub id: u32,
    // PhantomData makes Handle<Mesh> and Handle<Texture> distinct types
    // without storing any T data. Zero cost at runtime.
    _marker: PhantomData<T>,
}

impl<T> Copy for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Handle<T> {
    /// Create a Handle with the given ID and a default PhantomData marker.
    pub fn new(id: u32) -> Self {
        Self { id, _marker: PhantomData }
    }
}

// AssetStore<T> — a registry of assets indexed by Handle.
pub struct AssetStore<T> {
    assets:  HashMap<u32, T>,
    /// Reference counts for assets claimed by levels. An asset is only
    /// eligible for eviction once its count drops to zero (Decima-style).
    refs:    HashMap<u32, usize>,
    next_id: u32,
}

impl<T> AssetStore<T> {
    pub fn new() -> Self {
        Self { assets: HashMap::new(), refs: HashMap::new(), next_id: 0 }
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
    #[allow(dead_code)]
    pub fn get_mut(&mut self, handle: &Handle<T>) -> Option<&mut T> {
        self.assets.get_mut(&handle.id)
    }

    // replace() swaps an asset in place — used for hot reload.
    // The handle stays valid; the data changes.
    pub fn replace(&mut self, handle: &Handle<T>, new_asset: T) {
        self.assets.insert(handle.id, new_asset);
    }

    /// Number of assets currently held.
    pub fn count(&self) -> usize {
        self.assets.len()
    }

    /// Claim one reference to an asset. Call this once per level that loads
    /// a mesh so the eviction pass never frees a still-referenced asset.
    pub fn retain(&mut self, handle: &Handle<T>) {
        *self.refs.entry(handle.id).or_insert(0) += 1;
    }

    /// Release one reference to an asset. Returns the remaining refcount.
    /// When it hits 0 the asset becomes eviction-eligible.
    pub fn release(&mut self, handle: &Handle<T>) -> usize {
        if let Some(r) = self.refs.get_mut(&handle.id) {
            *r = r.saturating_sub(1);
            *r
        } else {
            0
        }
    }

    /// How many live references an asset currently has (0 if never claimed).
    pub fn ref_count(&self, handle: &Handle<T>) -> usize {
        self.refs.get(&handle.id).copied().unwrap_or(0)
    }

    /// Remove every asset whose refcount reached zero. `protected` holds the
    /// store IDs of assets still pinned by the primary scene or mesh cache —
    /// those are never evicted even if a level released them, because the
    /// cache dedups by path and other callers may hold them. Returns the
    /// number of assets evicted.
    pub fn evict_unused(&mut self, protected: &std::collections::HashSet<u32>) -> usize {
        let to_remove: Vec<u32> = self
            .refs
            .iter()
            .filter(|(id, r)| **r == 0 && !protected.contains(id))
            .map(|(id, _)| *id)
            .collect();
        let n = to_remove.len();
        for id in to_remove {
            self.assets.remove(&id);
            self.refs.remove(&id);
        }
        n
    }

    /// Iterate the assets in the store (immutable).
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.assets.iter().map(|(id, asset)| (*id, asset))
    }
}