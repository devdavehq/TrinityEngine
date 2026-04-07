pub mod mesh;
pub mod streaming;
pub mod store;

// Re-export the most commonly used types so callers write
// "use crate::assets::Mesh" instead of "crate::assets::mesh::Mesh"
pub use mesh::Mesh;
pub use streaming::MeshStreamingQueue;
pub use store::{AssetStore, Handle};

// Handle<T> is a typed ID — just a number that refers to an asset.
//
// Why typed? So Handle<Mesh> and Handle<Texture> are different types.
// If you try to pass a Handle<Texture> where a Handle<Mesh> is expected,
// the compiler catches it at compile time. No runtime crashes.
//
// Copy + Clone: handles are cheap to duplicate — they're just numbers.
// The asset data stays in the store; handles are just references to it.
// #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
// pub struct Handle<T> {
//     // The actual ID number — index into the store's HashMap.
//     id: u32,

//     // PhantomData<T> uses zero bytes but tells Rust this handle
//     // is associated with type T. Required for generics to work properly.
//     _marker: PhantomData<T>,
// }

// AssetStore<T> is the library — it holds all assets of one type.
// You'll have one AssetStore<Mesh> for meshes, one AssetStore<Texture> later.
// pub struct AssetStore<T> {
//     // The actual storage: a map from ID number to asset value.
//     assets: HashMap<u32, T>,

//     // Tracks the next ID to hand out.
//     // Starts at 0, increments every time we add an asset.
//     next_id: u32,
// }

// impl<T> means: implement these functions for AssetStore<T> for any T.
// This is how you write methods for a generic struct in Rust.
// impl<T> AssetStore<T> {
//     // Create a new empty store.
//     pub fn new() -> Self {
//         Self {
//             assets:  HashMap::new(),
//             next_id: 0,
//         }
//     }

//     // add() stores an asset and returns a Handle to it.
//     // You call this at startup: let mesh_handle = store.add(my_mesh);
//     // Then you store mesh_handle in a Renderable component.
//     pub fn add(&mut self, asset: T) -> Handle<T> {
//         // Grab the next available ID.
//         let id = self.next_id;

//         // Insert the asset into the map with that ID.
//         self.assets.insert(id, asset);

//         // Increment so the next call gets a different ID.
//         self.next_id += 1;

//         // Return a Handle with this ID.
//         // PhantomData is constructed with PhantomData — it takes no value.
//         Handle { id, _marker: PhantomData }
//     }

//     // get() looks up an asset by handle and returns a reference to it.
//     // Returns Option<&T> — Some(&asset) if found, None if not.
//     // The caller decides what to do if the handle is invalid.
//     //
//     // "&self" — we only read, don't modify the store.
//     // "&Handle<T>" — we borrow the handle to read its ID.
//     // "-> Option<&T>" — either Some reference or None.
//     pub fn get(&self, handle: &Handle<T>) -> Option<&T> {
//         // Look up by the handle's ID number in the HashMap.
//         self.assets.get(&handle.id)
//     }
// }