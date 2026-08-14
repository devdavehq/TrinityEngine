// src/vfs/pak.rs
// ──────────────────────────────────────────────────────────────────────────
// .pak file archive — pack a folder of Content into one compact file and
// serve it back through the Vfs trait, so the engine can run entirely from
// a single archive instead of loose files.
//
// Format (v2, little-endian):
//   magic  "TRNP" (4 bytes)
//   u32    version        (= 2)
//   u32    entry count
//   TOC:   for each entry:
//            u32   name length
//            bytes name (UTF-8, forward-slash relative path)
//            u8    compression (0 = raw, 1 = deflate)
//            u32   data length        (uncompressed size)
//            u32   stored length      (bytes stored on disk)
//            bytes data              (raw or deflate-compressed)
//
// v2 adds optional deflate compression per entry; the in-memory Vfs always
// exposes uncompressed bytes, so readers don't care how it was stored.
// v1 files (no compression byte) still load through the fallback reader.
// ──────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::io::Read;

use super::{DirEntry, Vfs};

const MAGIC: &[u8; 4] = b"TRNP";
const VERSION: u32 = 2;

/// Compression code stored on disk per entry.
const COMP_NONE: u8 = 0;
const COMP_DEFLATE: u8 = 1;

/// A packed, immutable file archive exposed through the Vfs trait.
#[derive(Default)]
pub struct PakFile {
    entries: HashMap<String, Vec<u8>>,
}

impl PakFile {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert one file into the pack under a relative path.
    pub fn add_file(&mut self, name: &str, data: Vec<u8>) {
        let name = name.replace('\\', "/");
        // Strip any leading "./" or "/" for consistent keys.
        let name = name.trim_start_matches("./").trim_start_matches('/');
        self.entries.insert(name.to_string(), data);
    }

    /// Like `add_file` but convenient for string content.
    pub fn add_str(&mut self, name: &str, data: &str) {
        self.add_file(name, data.as_bytes().to_vec());
    }

    /// Recursively pack every file under `dir` (relative paths are preserved).
    pub fn build_from_dir(dir: &str) -> std::io::Result<Self> {
        let mut pak = Self::new();
        let files = super::OsFileSystem.walk_dir(dir)?;
        for rel in files {
            let full = if dir.is_empty() || dir == "." {
                rel.clone()
            } else {
                format!("{}/{}", dir.trim_end_matches('/'), rel)
            };
            match std::fs::read(&full) {
                Ok(data) => pak.add_file(&rel, data),
                Err(e) => tracing::warn!("[Pak] Skipping {}: {}", full, e),
            }
        }
        Ok(pak)
    }

    /// Serialize the pack to a .pak file on disk.
    ///
    /// Each entry is deflate-compressed when that shrinks it (most text and
    /// binary content compresses well). The in-memory form stays uncompressed.
    pub fn write_to(&self, path: &str) -> std::io::Result<()> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());

        let mut entries: Vec<(&String, &Vec<u8>)> = self.entries.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());

        for (name, data) in &entries {
            let name_bytes = name.as_bytes();
            out.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(name_bytes);

            // Try deflate; keep raw when it doesn't help (e.g. tiny entries).
            let compressed = flate2::write::DeflateEncoder::new(
                Vec::new(),
                flate2::Compression::default(),
            );
            let mut encoder = compressed;
            let _ = std::io::Write::write_all(&mut encoder, data);
            let stored = match encoder.finish() {
                Ok(b) => b,
                Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, e)),
            };

            let (comp, stored) = if stored.len() < data.len() {
                (COMP_DEFLATE, stored)
            } else {
                (COMP_NONE, data.to_vec())
            };
            out.push(comp);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(stored.len() as u32).to_le_bytes());
            out.extend_from_slice(&stored);
        }

        std::fs::write(path, out)
    }

    /// Load a .pak file from disk into an in-memory VFS.
    pub fn open(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Decode a .pak from raw archive bytes.
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        let mut c = std::io::Cursor::new(bytes);
        let mut magic = [0u8; 4];
        c.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a TrinityEngine .pak file (bad magic)",
            ));
        }
        let version = read_u32(&mut c)?;
        if version != VERSION {
            // v1 files lack the per-entry compression byte/length fields.
            if version == 1 {
                return read_v1(bytes);
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported .pak version {version}"),
            ));
        }
        let count = read_u32(&mut c)?;
        let mut pak = Self::new();
        for _ in 0..count {
            let name_len = read_u32(&mut c)? as usize;
            let mut name = vec![0u8; name_len];
            c.read_exact(&mut name)?;
            let name = String::from_utf8(name).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid name encoding")
            })?;
            let comp = read_u8(&mut c)?;
            let data_len = read_u32(&mut c)? as usize;
            let stored_len = read_u32(&mut c)? as usize;
            let mut stored = vec![0u8; stored_len];
            c.read_exact(&mut stored)?;
            let data = match comp {
                COMP_NONE => stored,
                COMP_DEFLATE => {
                    let mut decoder = flate2::read::DeflateDecoder::new(&stored[..]);
                    let mut buf = Vec::with_capacity(data_len);
                    decoder.read_to_end(&mut buf).map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                    })?;
                    buf
                }
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("unsupported .pak compression code {comp}"),
                    ));
                }
            };
            pak.entries.insert(name, data);
        }
        Ok(pak)
    }

    /// Number of entries in the pack.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total uncompressed byte size of all entries.
    pub fn total_bytes(&self) -> u64 {
        self.entries.values().map(|d| d.len() as u64).sum()
    }
}

fn read_u32(c: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    c.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u8(c: &mut impl Read) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    c.read_exact(&mut buf)?;
    Ok(buf[0])
}

/// Backwards-compatible reader for original v1 archives (no compression field).
fn read_v1(bytes: &[u8]) -> std::io::Result<PakFile> {
    let mut c = std::io::Cursor::new(bytes);
    let mut magic = [0u8; 4];
    c.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a TrinityEngine .pak file (bad magic)",
        ));
    }
    let _version = read_u32(&mut c)?;
    let count = read_u32(&mut c)?;
    let mut pak = PakFile::new();
    for _ in 0..count {
        let name_len = read_u32(&mut c)? as usize;
        let mut name = vec![0u8; name_len];
        c.read_exact(&mut name)?;
        let name = String::from_utf8(name).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid name encoding")
        })?;
        let data_len = read_u32(&mut c)? as usize;
        let mut data = vec![0u8; data_len];
        c.read_exact(&mut data)?;
        pak.entries.insert(name, data);
    }
    Ok(pak)
}

impl Vfs for PakFile {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn read(&self, path: &str) -> std::io::Result<Vec<u8>> {
        self.entries
            .get(path.trim_start_matches("./").replace('\\', "/").as_str())
            .cloned()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, path))
    }

    fn read_to_string(&self, path: &str) -> std::io::Result<String> {
        let bytes = self.read(path)?;
        String::from_utf8(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    // A pak is immutable once built — writes are refused.
    fn write(&self, _path: &str, _data: &[u8]) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "cannot write into a packed .pak archive",
        ))
    }

    fn write_string(&self, _path: &str, _data: &str) -> std::io::Result<()> {
        self.write(_path, _data.as_bytes())
    }

    fn exists(&self, path: &str) -> bool {
        self.entries.contains_key(path.trim_start_matches("./").replace('\\', "/").as_str())
    }

    fn read_dir(&self, path: &str) -> std::io::Result<Vec<DirEntry>> {
        let prefix = if path.is_empty() || path == "." || path == "/" {
            String::new()
        } else {
            let mut p = path.trim_start_matches("./").replace('\\', "/");
            if !p.is_empty() {
                p.push('/');
            }
            p
        };
        let mut dirs_seen = std::collections::HashSet::new();
        let mut entries = Vec::new();
        for (key, data) in &self.entries {
            if key.starts_with(&prefix) {
                let rel = &key[prefix.len()..];
                if let Some(slash) = rel.find('/') {
                    let dir_name = &rel[..slash];
                    if dirs_seen.insert(dir_name.to_string()) {
                        entries.push(DirEntry {
                            name: dir_name.to_string(),
                            is_file: false,
                            is_dir: true,
                            size: 0,
                        });
                    }
                } else {
                    entries.push(DirEntry {
                        name: rel.to_string(),
                        is_file: true,
                        is_dir: false,
                        size: data.len() as u64,
                    });
                }
            }
        }
        Ok(entries)
    }

    fn walk_dir(&self, path: &str) -> std::io::Result<Vec<String>> {
        let prefix = if path.is_empty() || path == "." || path == "/" {
            String::new()
        } else {
            let mut p = path.trim_start_matches("./").replace('\\', "/");
            if !p.is_empty() {
                p.push('/');
            }
            p
        };
        let mut out: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .map(|k| k[prefix.len()..].to_string())
            .collect();
        out.sort();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pak_roundtrip_binary() {
        let mut pak = PakFile::new();
        pak.add_str("scene.txt", "[entity]\nname = a\n");
        pak.add_file("Content/Meshes/cube.obj", vec![1, 2, 3, 4, 5]);
        pak.add_str("Content/Scripts/player.lua", "return {}");

        let bytes = {
            let path = std::env::temp_dir().join(format!(
                "trinity_pak_{}_{}.pak",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let ps = path.to_str().unwrap();
            pak.write_to(ps).unwrap();
            let b = std::fs::read(ps).unwrap();
            let _ = std::fs::remove_file(ps);
            b
        };

        let loaded = PakFile::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(
            loaded.read_to_string("scene.txt").unwrap(),
            "[entity]\nname = a\n"
        );
        assert_eq!(loaded.read("Content/Meshes/cube.obj").unwrap(), vec![1, 2, 3, 4, 5]);
        assert!(loaded.exists("Content/Scripts/player.lua"));
    }

    #[test]
    fn pak_v2_compresses_repetitive_content() {
        // Highly repetitive text should round-trip through deflate exactly.
        let mut pak = PakFile::new();
        let big: String = std::iter::repeat("the quick brown fox jumps over the lazy dog ")
            .take(500)
            .collect();
        pak.add_str("Content/Textures/notes.txt", &big);

        let path = std::env::temp_dir().join(format!(
            "trinity_pak_comp_{}.pak",
            std::process::id()
        ));
        let ps = path.to_str().unwrap();
        pak.write_to(ps).unwrap();
        let on_disk = std::fs::metadata(ps).unwrap().len();
        let _ = std::fs::remove_file(ps);

        // Deflate should make the archive far smaller than the raw text.
        assert!(on_disk < big.len() as u64 / 3, "expected compression, got {on_disk} vs {}", big.len());

        let loaded = PakFile::from_bytes(&{
            let mut p = PakFile::new();
            p.add_str("Content/Textures/notes.txt", &big);
            let mut bytes = Vec::new();
            let mut entries: Vec<_> = p.entries.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            bytes.extend_from_slice(MAGIC);
            bytes.extend_from_slice(&VERSION.to_le_bytes());
            bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            for (name, data) in &entries {
                let nb = name.as_bytes();
                bytes.extend_from_slice(&(nb.len() as u32).to_le_bytes());
                bytes.extend_from_slice(nb);
                let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
                let _ = std::io::Write::write_all(&mut enc, data);
                let stored = enc.finish().unwrap();
                bytes.push(COMP_DEFLATE);
                bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&(stored.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&stored);
            }
            bytes
        })
        .unwrap();
        assert_eq!(
            loaded.read_to_string("Content/Textures/notes.txt").unwrap(),
            big
        );
    }

    #[test]
    fn pak_v1_backward_compat() {
        // A hand-built v1 archive (no compression byte) must still load.
        let mut pak = PakFile::new();
        pak.add_str("legacy.txt", "old format");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes()); // v1
        bytes.extend_from_slice(&1u32.to_le_bytes()); // count
        bytes.extend_from_slice(&10u32.to_le_bytes()); // name len
        bytes.extend_from_slice(b"legacy.txt");
        bytes.extend_from_slice(&10u32.to_le_bytes()); // data len
        bytes.extend_from_slice(b"old format");

        let loaded = PakFile::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.read_to_string("legacy.txt").unwrap(), "old format");
        let _ = pak.len();
    }

    #[test]
    fn pak_rejects_garbage_magic() {
        assert!(PakFile::from_bytes(b"NOPE000000").is_err());
    }

    #[test]
    fn pak_is_readonly() {
        let pak = PakFile::new();
        assert!(pak.write("x.txt", b"x").is_err());
    }

    #[test]
    fn pak_nonexistent_read_fails() {
        let pak = PakFile::new();
        assert!(pak.read("missing.txt").is_err());
        assert!(!pak.exists("missing.txt"));
    }

    #[test]
    fn pak_read_dir_plus_walk() {
        let mut pak = PakFile::new();
        pak.add_str("a.txt", "a");
        pak.add_str("sub/b.txt", "b");
        pak.add_str("sub/c.txt", "c");

        let top = pak.read_dir(".").unwrap();
        assert_eq!(top.len(), 2); // a.txt + sub/
        assert!(top.iter().any(|e| e.name == "a.txt" && e.is_file));
        assert!(top.iter().any(|e| e.name == "sub" && e.is_dir));

        let files = pak.walk_dir("sub").unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.contains(&"b.txt".to_string()));
        assert!(files.contains(&"c.txt".to_string()));
    }
}
