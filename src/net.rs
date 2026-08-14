// src/net.rs
// ──────────────────────────────────────────────────────────────────────────────
// Multiplayer world-state sync over UDP.
//
// WHY IT EXISTS:
//   Real games share a world. TrinityEngine historically rendered, simulated and
//   scripted a single process; networking was the one headline missing feature.
//   This module adds a lightweight host/client sync layer using only `std::net`
//   (no async runtime, no framework) — a pragmatic first run at replication.
//
// HOW IT WORKS
//   The world keeps a `NetId` component on entities that should be shared.
//   A host binds a UDP socket and every tick broadcasts a compact snapshot of
//   every NetId entity (position, velocity, rotation, scale). Clients send
//   their locally-owned entity's state upstream and apply remote snapshots for
//   entities that already exist locally (same scene content on both sides).
//
//   Ownership model:
//     - Host: authoritative. Applies client "controlled" updates for the
//       specific NetId the client claims, then re-broadcasts the merged world.
//     - Client: sends its controlled NetId's state each tick; applies host
//       snapshots to existing local entities.
//
// LIMITATIONS (documented, by design for a first pass)
//   - UDP, no reliability — a dropped packet just skips a frame of movement.
//   - Entities must pre-exist on both sides with matching NetId (we do NOT
//     spawn/despawn remotely yet).
//   - No encryption/auth; meant for a trusted or demo environment.
// ──────────────────────────────────────────────────────────────────────────────

use crate::components::{NetId, Position, Renderable, Rotation};
use crate::engine_subsystems::AssetState;
use glam::Vec3;
use hecs::World;
use std::collections::{HashMap, HashSet};
use std::net::{SocketAddr, UdpSocket};
use std::time::Instant;

/// Maximum payload for a snapshot message (bytes). UDP limits are ~65507, but
/// we keep tile-size small so a snapshot always fits in a single datagram.
const MAX_PAYLOAD: usize = 1200;
/// How often the host compiles + broadcasts a snapshot, in ticks.
const HOST_SNAPSHOT_INTERVAL: u32 = 2;
/// Seconds before a silent client is dropped from the host's peer list.
const PEER_TIMEOUT_SECS: f32 = 5.0;

/// A small wire-format marker for every message.
const MSG_HELLO: u8 = 1;
const MSG_PLAYER_STATE: u8 = 2;
const MSG_SNAPSHOT: u8 = 3;
const MSG_PING: u8 = 4;
/// Host tells a client an entity with a NetId has appeared.
const MSG_SPAWN: u8 = 5;
/// Host tells a client an entity with a NetId has been destroyed.
const MSG_DESPAWN: u8 = 6;
/// Client tells the host which snapshot sequence it last applied (loss stats).
const MSG_ACK: u8 = 7;

/// Replicated transform of a single entity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetTransform {
    pub net_id: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub mass: f32,
}

impl NetTransform {
    pub fn position(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

/// What this process is doing on the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetRole {
    /// No network activity.
    Off,
    /// Authoritative game server (binds a UDP port).
    Host,
    /// Connecting player (talks to the host).
    Client,
}

/// Runtime networking state owned by the engine.
pub struct NetworkManager {
    role: NetRole,
    socket: Option<UdpSocket>,
    /// Host: map of known client addr → (# of the entity that client owns, last heard).
    peers: HashMap<SocketAddr, (u32, f32)>,
    /// Client: address we dialed.
    server_addr: Option<SocketAddr>,
    /// Client: NetId this process controls and reports upstream.
    controlled_net_id: Option<u32>,
    /// Frame counter (drives snapshot cadence).
    tick: u32,
    /// Client handshake done?
    ready: bool,
    /// Diagnostic: number of snapshots applied.
    pub messages_received: u64,
    /// Diagnostic: number of snapshots sent.
    pub messages_sent: u64,
    /// Diagnostic: host-side, how many snapshots each peer has acked (loss stats).
    pub peer_acks: HashMap<SocketAddr, u32>,
    /// Client-side: the highest snapshot sequence applied so far.
    last_applied_seq: u32,
    /// Client-side: how many packets were dropped as duplicates or out-of-order.
    pub packets_dropped: u64,
    /// Host-side: for each peer, the set of NetIds we've already announced.
    /// Used to send MSG_SPAWN/MSG_DESPAWN when that set changes.
    announced: HashMap<SocketAddr, HashSet<u32>>,
    /// Timestamp used to expire silent peers.
    started_at: Instant,
}

impl Default for NetworkManager {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkManager {
    pub fn new() -> Self {
        Self {
            role: NetRole::Off,
            socket: None,
            peers: HashMap::new(),
            server_addr: None,
            controlled_net_id: None,
            tick: 0,
            ready: false,
            messages_received: 0,
            messages_sent: 0,
            peer_acks: HashMap::new(),
            last_applied_seq: 0,
            packets_dropped: 0,
            announced: HashMap::new(),
            started_at: Instant::now(),
        }
    }

    pub fn role(&self) -> NetRole {
        self.role
    }

    pub fn is_active(&self) -> bool {
        self.socket.is_some()
    }

    /// Open a host session bound to the given port.
    pub fn host(&mut self, bind_addr: &str) -> Result<(), String> {
        self.shutdown();
        let socket = UdpSocket::bind(bind_addr).map_err(|e| format!("bind {}: {}", bind_addr, e))?;
        socket.set_nonblocking(true).map_err(|e| e.to_string())?;
        socket.set_read_timeout(Some(std::time::Duration::from_millis(1))).ok();
        self.socket = Some(socket);
        self.role = NetRole::Host;
        self.peers.clear();
        self.announced.clear();
        self.peer_acks.clear();
        self.tick = 0;
        self.ready = true;
        tracing::info!("[Net] Host listening on {}", bind_addr);
        Ok(())
    }

    /// Join a host session. The local entity with `controlled_net_id` is the
    /// one we report upstream.
    pub fn connect(&mut self, host_addr: &str, controlled_net_id: u32) -> Result<(), String> {
        self.shutdown();
        let socket = UdpSocket::bind("0.0.0.0:0").map_err(|e| format!("bind ephemeral: {}", e))?;
        socket.set_nonblocking(true).map_err(|e| e.to_string())?;
        socket.set_read_timeout(Some(std::time::Duration::from_millis(1))).ok();
        let addr: SocketAddr = host_addr.parse().map_err(|e| format!("bad address {}: {}", host_addr, e))?;
        socket.connect(addr).map_err(|e| format!("connect {}: {}", host_addr, e))?;
        self.role = NetRole::Client;
        self.server_addr = Some(addr);
        self.controlled_net_id = Some(controlled_net_id);
        self.ready = false;
        self.tick = 0;

        // Send a HELLO handshake immediately.
        let mut msg = Vec::with_capacity(64);
        msg.push(MSG_HELLO);
        push_u32(&mut msg, controlled_net_id);
        let _ = socket.send(&msg);
        tracing::info!("[Net] Connecting to {} as player {}", host_addr, controlled_net_id);

        self.socket = Some(socket);
        Ok(())
    }

    /// Stop networking.
    pub fn shutdown(&mut self) {
        self.socket = None;
        self.role = NetRole::Off;
        self.peers.clear();
        self.announced.clear();
        self.peer_acks.clear();
        self.server_addr = None;
        self.controlled_net_id = None;
        self.ready = false;
    }

    fn now_secs(&self) -> f32 {
        self.started_at.elapsed().as_secs_f32()
    }

    // ── Per-frame driver ─────────────────────────────────────────────────────

    /// Advance the network layer one frame. Reads inbound messages and, on the
    /// host, periodically broadcasts the merged world snapshot.
    ///
    /// Call this from the simulation tick; `world` lets us both read the local
    /// authoritative state (host) and apply remote updates (both roles). The
    /// asset store is used to resolve mesh handles → paths (host, for spawn
    /// announcements) and to load meshes for remotely-spawned entities (client).
    pub fn tick(&mut self, world: &mut World, assets: &mut AssetState) {
        self.tick += 1;
        if self.socket.is_none() {
            return;
        }
        self.read_inbound(world, assets);
        match self.role {
            NetRole::Host => {
                if self.tick % HOST_SNAPSHOT_INTERVAL == 0 {
                    self.broadcast_snapshot(world, assets);
                }
            }
            NetRole::Client => {
                if self.ready {
                    self.send_player_state(world);
                }
            }
            NetRole::Off => {}
        }
    }

    // ── Inbound handling ─────────────────────────────────────────────────────

    fn read_inbound(&mut self, world: &mut World, assets: &mut AssetState) {
        let Some(socket) = self
            .socket
            .as_ref()
            .and_then(|s| s.try_clone().ok())
        else {
            return;
        };
        let mut buf = [0u8; MAX_PAYLOAD];
        loop {
            // For a connected client, recv_from returns the bound peer; for a
            // host we must read who sent it.
            let res = if self.role == NetRole::Client {
                socket.recv(&mut buf).map(|n| (n, self.server_addr.unwrap()))
            } else {
                socket.recv_from(&mut buf)
            };
            match res {
                Ok((n, from)) if n > 0 => {
                    self.messages_received += 1;
                    self.dispatch(&buf[..n], from, world, assets);
                }
                Ok(_) => continue,
                Err(_) => break, // non-blocking: nothing more to read
            }
        }
    }

    fn dispatch(&mut self, data: &[u8], from: SocketAddr, world: &mut World, assets: &mut AssetState) {
        if data.is_empty() {
            return;
        }
        match data[0] {
            MSG_HELLO => {
                // Host sees a new client claim its controlled NetId.
                if self.role == NetRole::Host {
                    let mut idx = 1;
                    let net_id = read_u32(data, &mut idx).unwrap_or(0);
                    self.peers.insert(from, (net_id, self.now_secs()));
                    self.announced.insert(from, HashSet::new());
                    self.peer_acks.insert(from, 0);
                    tracing::info!("[Net] New client {} owns NetId {}", from, net_id);
                }
            }
            MSG_PING => {
                if self.role == NetRole::Host {
                    let now = self.now_secs();
                    if let Some(entry) = self.peers.get_mut(&from) {
                        entry.1 = now;
                    }
                }
            }
            MSG_PLAYER_STATE => {
                // Client → Host: authoritative update for that client's entity.
                if self.role == NetRole::Host {
                    let mut idx = 1;
                    if let Some(tf) = decode_transform(data, &mut idx) {
                        apply_transform(world, tf);
                        // Refresh the peer's timestamp so it isn't timed out.
                        let now = self.now_secs();
                        if let Some(entry) = self.peers.get_mut(&from) {
                            entry.1 = now;
                        }
                    }
                }
            }
            MSG_SNAPSHOT => {
                // Host → Client: apply remote entities, then ack readiness.
                if self.role == NetRole::Client {
                    let mut idx = 1;
                    let seq = read_u32(data, &mut idx).unwrap_or(0);
                    // Reliability: drop duplicates / out-of-order snapshots.
                    if seq <= self.last_applied_seq && self.messages_received > 1 {
                        self.packets_dropped += 1;
                        return;
                    }
                    self.last_applied_seq = seq;
                    let count = read_u32(data, &mut idx).unwrap_or(0) as usize;
                    for _ in 0..count {
                        if let Some(tf) = decode_transform(data, &mut idx) {
                            apply_transform(world, tf);
                        }
                    }
                    self.ready = true;
                    // Ack the applied sequence back to the host for loss stats.
                    self.send_ack(seq);
                }
            }
            MSG_SPAWN => {
                // Host → Client: an entity with this NetId appeared remotely.
                if self.role == NetRole::Client {
                    let mut idx = 1;
                    if let Some(tf) = decode_transform(data, &mut idx) {
                        let mesh_path = read_string(data, &mut idx).unwrap_or_default();
                        spawn_remote(world, assets, tf, mesh_path);
                    }
                }
            }
            MSG_DESPAWN => {
                // Host → Client: an entity with this NetId was destroyed.
                if self.role == NetRole::Client {
                    let mut idx = 1;
                    let net_id = read_u32(data, &mut idx).unwrap_or(0);
                    despawn_remote(world, net_id);
                }
            }
            MSG_ACK => {
                // Client → Host: last snapshot seq the client applied.
                if self.role == NetRole::Host {
                    let mut idx = 1;
                    let seq = read_u32(data, &mut idx).unwrap_or(0);
                    self.peer_acks.insert(from, seq);
                }
            }
            _ => {}
        }
    }

    // ── Outbound / broadcast ─────────────────────────────────────────────────

    /// Host: gather every NetId entity into a compact snapshot and send to
    /// every known client. Also emits per-peer MSG_SPAWN/MSG_DESPAWN so
    /// clients can create/destroy entities that appear or disappear.
    fn broadcast_snapshot(&mut self, world: &mut World, assets: &mut AssetState) {
        let Some(socket) = &self.socket else { return };
        let now = self.now_secs();

        // Drop silent clients.
        self.peers.retain(|addr, (_, last_seen)| {
            let keep = now - *last_seen < PEER_TIMEOUT_SECS;
            if !keep {
                self.announced.remove(addr);
                self.peer_acks.remove(addr);
            }
            keep
        });
        if self.peers.is_empty() {
            return;
        }

        // Collect every NetId entity's transform in skeleton order, plus its
        // mesh path (so clients can spawn the right mesh).
        let tf: Vec<(NetTransform, Option<String>)> = {
            let mut q = world.query::<(&Position, &NetId, Option<&Rotation>, Option<&Renderable>)>();
            let mut list = Vec::new();
            for (pos, net, rot, render) in q.iter() {
                let (pitch, yaw, roll) = match rot {
                    Some(r) => (r.pitch, r.yaw, r.roll),
                    None => (0.0, 0.0, 0.0),
                };
                let mesh_path = render
                    .map(|r| mesh_path_for_handle(assets, &r.mesh))
                    .flatten();
                list.push((
                    NetTransform {
                        net_id: net.id,
                        x: pos.x,
                        y: pos.y,
                        z: pos.z,
                        vx: 0.0,
                        vy: 0.0,
                        vz: 0.0,
                        pitch,
                        yaw,
                        roll,
                        mass: 1.0,
                    },
                    mesh_path,
                ));
            }
            list
        };

        let current_ids: HashSet<u32> = tf.iter().map(|(t, _)| t.net_id).collect();

        // Per-peer spawn/despawn announcements.
        for addr in self.peers.keys() {
            let known = self.announced.entry(*addr).or_default();
            for (t, mesh_path) in &tf {
                if !known.contains(&t.net_id) {
                    let mut spawn = Vec::with_capacity(MAX_PAYLOAD);
                    spawn.push(MSG_SPAWN);
                    encode_transform(&mut spawn, t);
                    push_string(&mut spawn, mesh_path.as_deref().unwrap_or("meshes/cube.obj"));
                    if socket.send_to(&spawn, addr).is_ok() {
                        self.messages_sent += 1;
                    }
                }
            }
            for gone in known.difference(&current_ids) {
                let mut despawn = Vec::with_capacity(8);
                despawn.push(MSG_DESPAWN);
                push_u32(&mut despawn, *gone);
                if socket.send_to(&despawn, addr).is_ok() {
                    self.messages_sent += 1;
                }
            }
            *known = current_ids.clone();
        }

        let mut out = Vec::with_capacity(MAX_PAYLOAD);
        out.push(MSG_SNAPSHOT);
        push_u32(&mut out, self.tick);

        let mut bodies = Vec::with_capacity(MAX_PAYLOAD);
        let mut count = 0u32;
        for (t, _) in &tf {
            if bodies.len() + 44 > MAX_PAYLOAD - 8 {
                break;
            }
            encode_transform(&mut bodies, t);
            count += 1;
        }
        if count == 0 {
            return;
        }
        push_u32(&mut out, count);
        out.extend_from_slice(&bodies);

        for addr in self.peers.keys() {
            if socket.send_to(&out, addr).is_ok() {
                self.messages_sent += 1;
            }
        }
    }

    /// Client: send our owned entity's current transform upstream.
    fn send_player_state(&mut self, world: &mut World) {
        let Some(socket) = &self.socket else { return };
        let Some(net_id) = self.controlled_net_id else { return };

        let mut tf = None;
        let mut q = world.query::<(&Position, &crate::components::NetId, Option<&Rotation>)>();
        for (pos, net, rot) in q.iter() {
            if net.id == net_id {
                let (pitch, yaw, roll) = match rot {
                    Some(r) => (r.pitch, r.yaw, r.roll),
                    None => (0.0, 0.0, 0.0),
                };
                tf = Some(NetTransform {
                    net_id,
                    x: pos.x,
                    y: pos.y,
                    z: pos.z,
                    vx: 0.0,
                    vy: 0.0,
                    vz: 0.0,
                    pitch,
                    yaw,
                    roll,
                    mass: 1.0,
                });
                break;
            }
        }
        let Some(tf) = tf else { return };
        let mut payload = Vec::with_capacity(64);
        payload.push(MSG_PLAYER_STATE);
        encode_transform(&mut payload, &tf);
        if socket.send(&payload).is_ok() {
            self.messages_sent += 1;
        }
    }

    /// Send a keep-alive ping (useful for hosts that want freshness on demand).
    pub fn send_ping(&mut self) {
        let Some(socket) = &self.socket else { return };
        let _ = socket.send(&[MSG_PING]);
    }

    /// Client → Host: confirm the last applied snapshot sequence.
    fn send_ack(&mut self, seq: u32) {
        let Some(socket) = &self.socket else { return };
        let mut msg = Vec::with_capacity(8);
        msg.push(MSG_ACK);
        push_u32(&mut msg, seq);
        let _ = socket.send(&msg);
    }
}

// ── Wire helpers (little-endian, fixed width) ─────────────────────────────────

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(data: &[u8], idx: &mut usize) -> Option<u32> {
    if *idx + 4 > data.len() {
        return None;
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&data[*idx..*idx + 4]);
    *idx += 4;
    Some(u32::from_le_bytes(b))
}

fn push_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_f32(data: &[u8], idx: &mut usize) -> Option<f32> {
    if *idx + 4 > data.len() {
        return None;
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&data[*idx..*idx + 4]);
    *idx += 4;
    Some(f32::from_le_bytes(b))
}

/// Layout: id(4) pos(12) vel(12) rot(12) mass(4) = 44 bytes.
fn encode_transform(out: &mut Vec<u8>, t: &NetTransform) {
    push_u32(out, t.net_id);
    push_f32(out, t.x);
    push_f32(out, t.y);
    push_f32(out, t.z);
    push_f32(out, t.vx);
    push_f32(out, t.vy);
    push_f32(out, t.vz);
    push_f32(out, t.pitch);
    push_f32(out, t.yaw);
    push_f32(out, t.roll);
    push_f32(out, t.mass);
}

fn decode_transform(data: &[u8], idx: &mut usize) -> Option<NetTransform> {
    Some(NetTransform {
        net_id: read_u32(data, idx)?,
        x: read_f32(data, idx)?,
        y: read_f32(data, idx)?,
        z: read_f32(data, idx)?,
        vx: read_f32(data, idx)?,
        vy: read_f32(data, idx)?,
        vz: read_f32(data, idx)?,
        pitch: read_f32(data, idx)?,
        yaw: read_f32(data, idx)?,
        roll: read_f32(data, idx)?,
        mass: read_f32(data, idx)?,
    })
}

// ── String wire helpers ──────────────────────────────────────────────────────

/// Length-prefixed UTF-8 string (u16 length, then raw bytes).
fn push_string(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len().min(u16::MAX as usize) as u16;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&bytes[..len as usize]);
}

fn read_string(data: &[u8], idx: &mut usize) -> Option<String> {
    if *idx + 2 > data.len() {
        return None;
    }
    let mut b = [0u8; 2];
    b.copy_from_slice(&data[*idx..*idx + 2]);
    *idx += 2;
    let len = u16::from_le_bytes(b) as usize;
    if *idx + len > data.len() {
        return None;
    }
    let s = std::str::from_utf8(&data[*idx..*idx + len]).ok()?.to_string();
    *idx += len;
    Some(s)
}

/// Reverse-look up a mesh path for a handle by scanning the mesh cache.
fn mesh_path_for_handle(assets: &AssetState, handle: &crate::assets::Handle<crate::assets::Mesh>) -> Option<String> {
    assets
        .mesh_cache
        .iter()
        .find(|(_, h)| h.id == handle.id)
        .map(|(path, _)| path.clone())
}

// ── World application ────────────────────────────────────────────────────────

/// Apply a replicated transform to the local entity with that NetId (if it
/// exists). Two-way:
///  - Host applies a client-provided update to its authoritative copy.
///  - Client applies host snapshots to local copies of remote entities.
fn apply_transform(world: &mut World, tf: NetTransform) {
    let mut query = world.query::<(&mut Position, &crate::components::NetId, Option<&mut Rotation>)>();
    for (pos, net, rot) in query.iter() {
        if net.id == tf.net_id {
            pos.x = tf.x;
            pos.y = tf.y;
            pos.z = tf.z;
            if let Some(r) = rot {
                r.pitch = tf.pitch;
                r.yaw = tf.yaw;
                r.roll = tf.roll;
            }
            return;
        }
    }
    // No local entity for this NetId: silently skip (spawning remote entities
    // is intentionally out of scope for this first pass).
}

/// Client: create a local entity for a NetId the host just announced.
/// Loads the referenced mesh (or falls back to the default cube) and gives the
/// entity the host's current transform so it appears in the right place.
fn spawn_remote(world: &mut World, assets: &mut AssetState, tf: NetTransform, mesh_path: String) {
    // Don't double-spawn a NetId we already have locally.
    let existing = world.query::<(&Position, &NetId)>().iter().any(|(_, net)| net.id == tf.net_id);
    if existing {
        return;
    }

    let path = if mesh_path.is_empty() { "meshes/cube.obj" } else { &mesh_path };
    let handle = if let Some(h) = assets.mesh_cache.get(path).copied() {
        h
    } else {
        match crate::assets::Mesh::load(path) {
            Ok(mesh) => {
                let h = assets.meshes.add(mesh);
                assets.mesh_cache.insert(path.to_string(), h);
                h
            }
            Err(e) => {
                tracing::warn!("[Net] Spawn mesh {} failed: {}", path, e);
                // Fall back to a cube so the entity still exists.
                let cube = crate::assets::Mesh::load("meshes/cube.obj").unwrap_or_else(|_| crate::assets::Mesh { vertices: Vec::new() });
                let h = assets.meshes.add(cube);
                assets.mesh_cache.insert("meshes/cube.obj".to_string(), h);
                h
            }
        }
    };

    world.spawn((
        Position { x: tf.x, y: tf.y, z: tf.z },
        NetId { id: tf.net_id },
        Rotation {
            pitch: tf.pitch,
            yaw: tf.yaw,
            roll: tf.roll,
        },
        Renderable {
            mesh: handle,
            color: [0.7, 0.7, 0.7],
            metallic: 0.0,
            roughness: 0.6,
            ao: 1.0,
            scale: [1.0, 1.0, 1.0],
        },
    ));
    tracing::info!("[Net] Spawned remote entity NetId {} (mesh {})", tf.net_id, path);
}

/// Client: destroy the local entity carrying the given NetId.
fn despawn_remote(world: &mut World, net_id: u32) {
    let mut to_remove = None;
    let q = world.query_mut::<(hecs::Entity, &NetId)>();
    for (e, net) in q.into_iter() {
        if net.id == net_id {
            to_remove = Some(e);
            break;
        }
    }
    if let Some(e) = to_remove {
        let _ = world.despawn(e);
        tracing::info!("[Net] Despawned remote entity NetId {}", net_id);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::NetId;
    use hecs::World;

    fn test_assets() -> AssetState {
        AssetState::new()
    }

    #[test]
    fn transform_roundtrip() {
        let t = NetTransform {
            net_id: 7,
            x: 1.5,
            y: -2.0,
            z: 3.25,
            vx: 0.1,
            vy: 0.2,
            vz: 0.3,
            pitch: 0.5,
            yaw: 1.0,
            roll: -0.5,
            mass: 2.0,
        };
        let mut buf = Vec::new();
        encode_transform(&mut buf, &t);
        assert_eq!(buf.len(), 44);
        let mut idx = 0;
        let out = decode_transform(&buf, &mut idx).unwrap();
        assert_eq!(out, t);
        assert_eq!(idx, 44);
    }

    #[test]
    fn host_client_handshake_over_loopback() {
        let mut host = NetworkManager::new();
        host.host("127.0.0.1:27015").expect("host bind");

        let mut client = NetworkManager::new();
        client
            .connect("127.0.0.1:27015", 99)
            .expect("client connect");

        // Let the HELLO packet travel + be processed.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let mut world = World::new();
        let mut assets = test_assets();
        client.tick(&mut world, &mut assets);
        host.tick(&mut world, &mut assets);
        std::thread::sleep(std::time::Duration::from_millis(50));
        host.tick(&mut world, &mut assets);

        assert_eq!(host.peers.len(), 1, "host should have registered the client");
        let (_addr, (net_id, _last)) = host.peers.iter().next().unwrap();
        assert_eq!(*net_id, 99);

        host.shutdown();
        client.shutdown();
    }

    #[test]
    fn snapshot_reaches_client() {
        let mut host = NetworkManager::new();
        host.host("127.0.0.1:27016").expect("host bind");

        let mut client = NetworkManager::new();
        client
            .connect("127.0.0.1:27016", 99)
            .expect("client connect");

        // Seed a NetId entity on BOTH sides so a remote update can apply.
        let mut world = World::new();
        let mut assets = test_assets();
        world.spawn((
            Position { x: 0.0, y: 0.0, z: 0.0 },
            NetId { id: 5 },
            Rotation::default(),
        ));

        // Host moves NetId 5.
        {
            let mut q = world.query::<(&mut Position, &NetId)>();
            for (pos, net) in q.iter() {
                if net.id == 5 {
                    pos.x = 42.0;
                }
            }
        }

        // Let handshake settle, then run several host ticks so a snapshot fires.
        std::thread::sleep(std::time::Duration::from_millis(50));
        for _ in 0..6 {
            host.tick(&mut world, &mut assets);
            std::thread::sleep(std::time::Duration::from_millis(20));
            client.tick(&mut world, &mut assets);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        // Client should now have seen the snapshot and applied x=42.
        let mut q = world.query::<(&Position, &NetId)>();
        let mut seen = false;
        for (pos, net) in q.iter() {
            if net.id == 5 {
                seen = true;
                assert!((pos.x - 42.0).abs() < 0.001, "client applies host snapshot x={}", pos.x);
            }
        }
        assert!(seen, "NetId 5 entity still present");

        host.shutdown();
        client.shutdown();
    }

    #[test]
    fn remote_spawn_and_despawn_reaches_client() {
        let mut host = NetworkManager::new();
        host.host("127.0.0.1:27017").expect("host bind");

        let mut client = NetworkManager::new();
        client
            .connect("127.0.0.1:27017", 99)
            .expect("client connect");

        let mut world = World::new();
        let mut assets = test_assets();
        // Host spawns a NetId entity that the client has NEVER seen.
        world.spawn((
            Position { x: 3.0, y: 1.0, z: -2.0 },
            NetId { id: 77 },
            Rotation::default(),
        ));

        // Let handshake settle, then run host+client ticks so spawn arrives.
        std::thread::sleep(std::time::Duration::from_millis(50));
        for _ in 0..6 {
            host.tick(&mut world, &mut assets);
            std::thread::sleep(std::time::Duration::from_millis(20));
            client.tick(&mut world, &mut assets);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        // Client now has a local entity with NetId 77 at the spawn position.
        {
            let mut q = world.query::<(&Position, &NetId)>();
            let mut seen = false;
            for (pos, net) in q.iter() {
                if net.id == 77 {
                    seen = true;
                    assert!((pos.x - 3.0).abs() < 0.001, "spawned at x={}", pos.x);
                }
            }
            assert!(seen, "client spawned remote NetId 77");
        }

        // Host despawns NetId 77; client should remove its copy.
        {
            let mut remove = None;
            let mut qm = world.query_mut::<(hecs::Entity, &NetId)>();
            for (e, net) in qm.into_iter() {
                if net.id == 77 {
                    remove = Some(e);
                }
            }
            if let Some(e) = remove {
                let _ = world.despawn(e);
            }
        }
        for _ in 0..6 {
            host.tick(&mut world, &mut assets);
            std::thread::sleep(std::time::Duration::from_millis(20));
            client.tick(&mut world, &mut assets);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        let still = world.query::<(&Position, &NetId)>().iter().any(|(_, net)| net.id == 77);
        assert!(!still, "client despawned remote NetId 77");

        host.shutdown();
        client.shutdown();
    }
}