// src/save_slots.rs
// ────────────────────────────────────────────────────────────────────────────
// Per-profile save slots (#4 save system polish).
//
// Saves live under the per-machine profile dir (<appdata>/TrinityEngine/saves),
// NOT inside the installed game folder — so updates/packaging never wipe them
// and every OS user gets their own slots.
//
// Layout (slot index → two files, written atomically via temp+rename):
//   saves/slot_<n>.toml    — SlotMeta metadata
//   saves/slot_<n>.dat     — payload (scene snapshot, world state, anything)
//
// Slot 0 is reserved for the rolling autosave/checkpoint. Slots 1..MAX_SLOTS
// are explicit player saves. This module is feature-independent (works in the
// editor build, the runtime build, and headless tools alike).
// ────────────────────────────────────────────────────────────────────────────

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

pub const AUTOSAVE_SLOT: u32 = 0;
pub const FIRST_MANUAL_SLOT: u32 = 1;
pub const MAX_SLOTS: u32 = 24;
const META_EXT: &str = "toml";
const DATA_EXT: &str = "dat";
/// Third slot file: the WorldStateManager JSON (gameplay state: health, alive,
/// flags). Kept separate from the .scene payload so the scene loader only ever
/// sees valid scene text and old saves without gameplay state still load.
const STATE_EXT: &str = "state";

/// Metadata describing a saved slot (small enough to read without loading
/// the payload).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlotMeta {
    pub label: String,
    /// Scene / level name the save belongs to.
    pub scene: String,
    /// UTC unix seconds when the save was written.
    pub saved_utc: u64,
    /// Number of persisted entities (informational).
    pub entity_count: u32,
    /// Autosave/checkpoint slots roll over on every write.
    pub autosave: bool,
}

impl SlotMeta {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            saved_utc: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            ..Default::default()
        }
    }

    /// Human-readable timestamp (UTC) for HUD lists.
    pub fn timestamp_string(&self) -> String {
        if self.saved_utc == 0 {
            return "—".to_string();
        }
        let mut buf = [0u8; 64];
        // time crate not available; use a minimal formatting of unix time.
        let days = self.saved_utc / 86_400;
        let h = (self.saved_utc % 86_400) / 3_600;
        let m = (self.saved_utc % 3_600) / 60;
        let s = self.saved_utc % 60;
        let text = format!("day {} {:02}:{:02}:{:02}", days, h, m, s);
        let n = text.len().min(buf.len());
        buf[..n].copy_from_slice(&text.as_bytes()[..n]);
        String::from_utf8_lossy(&buf[..n]).trim_end_matches('\0').to_string()
    }
}

/// A slot as returned by `list()` / `load()`.
#[derive(Debug, Clone)]
pub struct SlotEntry {
    pub slot: u32,
    pub meta: SlotMeta,
    pub payload: String,
}

/// Save-slot manager bound to a directory.
pub struct SaveSlots {
    dir: PathBuf,
}

impl Default for SaveSlots {
    fn default() -> Self {
        Self::new()
    }
}

impl SaveSlots {
    /// Saves under the per-profile app-data directory.
    pub fn new() -> Self {
        Self {
            dir: crate::editor_persist::trinity_data_dir().join("saves"),
        }
    }

    /// For tests / custom targets.
    pub fn with_dir(dir: PathBuf) -> Self {
        Self { dir }
    }

    pub fn dir(&self) -> &PathBuf {
        &self.dir
    }

    fn meta_path(&self, slot: u32) -> PathBuf {
        self.dir.join(format!("slot_{}.{}", slot, META_EXT))
    }

    fn data_path(&self, slot: u32) -> PathBuf {
        self.dir.join(format!("slot_{}.{}", slot, DATA_EXT))
    }

    fn state_path(&self, slot: u32) -> PathBuf {
        self.dir.join(format!("slot_{}.{}", slot, STATE_EXT))
    }

    pub fn slot_valid(slot: u32) -> bool {
        slot <= MAX_SLOTS
    }

    /// Write a slot. Atomically (temp file + rename) so a crash mid-write can
    /// never leave a torn save.
    pub fn save(&self, slot: u32, meta: SlotMeta, payload: &str) -> std::io::Result<()> {
        if !Self::slot_valid(slot) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("slot out of range: {slot}"),
            ));
        }
        std::fs::create_dir_all(&self.dir)?;

        let mut meta = meta;
        meta.saved_utc = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let meta_toml = toml::to_string(&meta).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
        write_atomic(&self.meta_path(slot), meta_toml.as_bytes())?;
        write_atomic(&self.data_path(slot), payload.as_bytes())
    }

    /// Write the gameplay state JSON for a slot (WorldStateManager payload).
    /// Optional — a save without it still loads (state is simply empty).
    pub fn save_state(&self, slot: u32, state_json: &str) -> std::io::Result<()> {
        write_atomic(&self.state_path(slot), state_json.as_bytes())
    }

    /// Read the gameplay state JSON for a slot. Ok(None) when not present.
    pub fn load_state(&self, slot: u32) -> std::io::Result<Option<String>> {
        let path = self.state_path(slot);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(std::fs::read_to_string(path)?))
    }

    /// Read a slot's metadata without touching the payload.
    pub fn meta(&self, slot: u32) -> Option<SlotMeta> {
        let path = self.meta_path(slot);
        let text = std::fs::read_to_string(path).ok()?;
        toml::from_str(&text).ok()
    }

    /// Read a slot (metadata + payload). Ok(None) when the slot is empty.
    pub fn load(&self, slot: u32) -> std::io::Result<Option<SlotEntry>> {
        let meta = match self.meta(slot) {
            Some(m) => m,
            None => return Ok(None),
        };
        let payload = std::fs::read_to_string(self.data_path(slot))?;
        Ok(Some(SlotEntry { slot, meta, payload }))
    }

    /// Remove a slot (all three files).
    pub fn delete(&self, slot: u32) -> std::io::Result<()> {
        let _ = std::fs::remove_file(self.meta_path(slot));
        let _ = std::fs::remove_file(self.data_path(slot));
        let _ = std::fs::remove_file(self.state_path(slot));
        Ok(())
    }

    /// List every occupied slot, highest-slot-first (most recent manual saves
    /// are usually the largest index).
    pub fn list(&self) -> Vec<SlotEntry> {
        let mut out: Vec<SlotEntry> = Vec::new();
        for slot in (AUTOSAVE_SLOT..=MAX_SLOTS).rev() {
            if let Ok(Some(e)) = self.load(slot) {
                out.push(e);
            }
        }
        out
    }

    /// The most recently written manual slot (excludes the autosave slot).
    pub fn latest_manual_slot(&self) -> Option<u32> {
        self.list()
            .iter()
            .find(|e| !e.meta.autosave)
            .map(|e| e.slot)
    }

    /// Number of occupied slots.
    pub fn count(&self) -> usize {
        self.list().len()
    }
}

/// Write a file atomically: write to `<file>.tmp` then rename over the target.
fn write_atomic(path: &PathBuf, data: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(data)?;
    f.sync_all().ok();
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_slots(tag: &str) -> SaveSlots {
        let dir = std::env::temp_dir().join(format!(
            "trinity_saves_{}_{}",
            std::process::id(),
            tag
        ));
        let _ = std::fs::remove_dir_all(&dir);
        SaveSlots::with_dir(dir)
    }

    #[test]
    fn slot_save_load_roundtrip() {
        let slots = tmp_slots("roundtrip");
        let mut meta = SlotMeta::new("W1 Checkpoint");
        meta.scene = "Content/scenes/main.scene".to_string();
        meta.entity_count = 12;
        slots
            .save(1, meta, "[entity]\nname = hero\n")
            .unwrap();
        let entry = slots.load(1).unwrap().unwrap();
        assert_eq!(entry.payload, "[entity]\nname = hero\n");
        assert_eq!(entry.meta.label, "W1 Checkpoint");
        assert_eq!(entry.meta.scene, "Content/scenes/main.scene");
        assert_eq!(entry.meta.entity_count, 12);
        assert!(!entry.meta.timestamp_string().is_empty());
        let _ = slots.delete(1);
    }

    #[test]
    fn slot_autosave_overwrites() {
        let slots = tmp_slots("autosave");
        let mut meta = SlotMeta::new("autosave");
        meta.autosave = true;
        slots.save(AUTOSAVE_SLOT, meta.clone(), "v1").unwrap();
        slots.save(AUTOSAVE_SLOT, meta, "v2").unwrap();
        let entry = slots.load(AUTOSAVE_SLOT).unwrap().unwrap();
        assert_eq!(entry.payload, "v2");
        assert_eq!(slots.count(), 1);
    }

    #[test]
    fn slot_listing_and_empty() {
        let slots = tmp_slots("listing");
        assert_eq!(slots.count(), 0);
        assert!(slots.load(3).unwrap().is_none());

        let mut m3 = SlotMeta::new("third");
        let mut m1 = SlotMeta::new("first");
        slots.save(3, m3.clone(), "data3").unwrap();
        m1.scene = "level2".to_string();
        slots.save(1, m1, "data1").unwrap();
        let list = slots.list();
        assert_eq!(list.len(), 2);
        // Highest-slot-first ordering (3 before 1).
        assert_eq!(list[0].slot, 3);
        assert_eq!(list[1].slot, 1);
        assert_eq!(slots.latest_manual_slot(), Some(3));

        slots.delete(3).unwrap();
        assert!(slots.load(3).unwrap().is_none());
        assert_eq!(slots.count(), 1);
    }

    #[test]
    fn slot_out_of_range_rejected() {
        let slots = tmp_slots("range");
        assert!(slots.save(MAX_SLOTS + 1, SlotMeta::new("x"), "x").is_err());
    }
}