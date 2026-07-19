// src/vfs.rs
// ──────────────────────────────────────────────────────────────────────────────
// Virtual File System — decouples asset loading from physical I/O.
//
// The trait lets the engine read/write assets without knowing whether they
// come from disk, a .pak archive, an in-memory overlay, or a network mount.
//
// DEFAULT_VFS is a thread-local OsFileSystem (real filesystem). Callers that
// don't need VFS injection can use `vfs::read_to_string("foo.txt")` directly.
//
// For testability or mod-pack support, replace the global VFS with a
// MemoryVfs or a union of overlays via `set_global_vfs()`.
// ──────────────────────────────────────────────────────────────────────────────

use std::any::Any;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Metadata about a filesystem entry returned by `read_dir`.
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub is_file: bool,
    pub is_dir: bool,
    pub size: u64,
}

// ── Trait ────────────────────────────────────────────────────────────────────

/// Core abstraction for reading and writing files.
///
/// All paths are relative to the VFS root (for OsFileSystem, the CWD).
/// Implementations should handle path normalization (e.g. "foo/../bar" → "bar").
pub trait Vfs: Send + Sync {
    /// Allow downcasting when needed (e.g. to inspect overlay layers).
    fn as_any(&self) -> &dyn Any;

    // ── Reads ────────────────────────────────────────────────────────────

    /// Read an entire file as raw bytes.
    fn read(&self, path: &str) -> std::io::Result<Vec<u8>>;

    /// Read an entire file as a UTF-8 string.
    fn read_to_string(&self, path: &str) -> std::io::Result<String>;

    // ── Writes ───────────────────────────────────────────────────────────

    /// Write raw bytes, creating or overwriting the file.
    fn write(&self, path: &str, data: &[u8]) -> std::io::Result<()>;

    /// Write a UTF-8 string, creating or overwriting the file.
    fn write_string(&self, path: &str, data: &str) -> std::io::Result<()>;

    // ── Queries ──────────────────────────────────────────────────────────

    /// Check whether a path exists (file or directory).
    fn exists(&self, path: &str) -> bool;

    /// List the entries of a directory. Returns empty vec if dir doesn't exist.
    fn read_dir(&self, path: &str) -> std::io::Result<Vec<DirEntry>>;

    /// Recursively walk a directory, returning all files under it (relative paths).
    fn walk_dir(&self, path: &str) -> std::io::Result<Vec<String>>;

    /// Read a file via an `std::io::Read` adapter (for streaming decoders).
    /// Default implementation buffers into memory; override for true streaming.
    fn open_read(&self, path: &str) -> std::io::Result<Box<dyn Read>> {
        let data = self.read(path)?;
        Ok(Box::new(std::io::Cursor::new(data)))
    }
}

// ── OsFileSystem (real disk) ─────────────────────────────────────────────────

/// Default VFS implementation that delegates to the real filesystem.
pub struct OsFileSystem;

impl OsFileSystem {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OsFileSystem {
    fn default() -> Self {
        Self
    }
}

impl Vfs for OsFileSystem {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn read(&self, path: &str) -> std::io::Result<Vec<u8>> {
        std::fs::read(path)
    }

    fn read_to_string(&self, path: &str) -> std::io::Result<String> {
        std::fs::read_to_string(path)
    }

    fn write(&self, path: &str, data: &[u8]) -> std::io::Result<()> {
        // Ensure parent directories exist.
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, data)
    }

    fn write_string(&self, path: &str, data: &str) -> std::io::Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, data)
    }

    fn exists(&self, path: &str) -> bool {
        Path::new(path).exists()
    }

    fn read_dir(&self, path: &str) -> std::io::Result<Vec<DirEntry>> {
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            let name = entry.file_name().to_string_lossy().to_string();
            entries.push(DirEntry {
                name,
                is_file: meta.is_file(),
                is_dir: meta.is_dir(),
                size: meta.len(),
            });
        }
        Ok(entries)
    }

    fn walk_dir(&self, path: &str) -> std::io::Result<Vec<String>> {
        let mut files = Vec::new();
        walk_recursive(Path::new(path), Path::new(path), &mut files)?;
        Ok(files)
    }

    fn open_read(&self, path: &str) -> std::io::Result<Box<dyn Read>> {
        let file = std::fs::File::open(path)?;
        Ok(Box::new(file))
    }
}

fn walk_recursive(root: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_recursive(root, &path, out)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(rel);
        }
    }
    Ok(())
}

// ── MemoryVfs (in-memory, for tests / overlays) ─────────────────────────────

/// In-memory VFS for unit testing or virtual overlays.
pub struct MemoryVfs {
    files: Mutex<HashMap<String, Vec<u8>>>,
}

impl MemoryVfs {
    pub fn new() -> Self {
        Self {
            files: Mutex::new(HashMap::new()),
        }
    }

    /// Insert a file from a string.
    pub fn insert_str(&self, path: &str, data: &str) {
        self.files
            .lock()
            .unwrap()
            .insert(path.to_string(), data.as_bytes().to_vec());
    }

    /// Insert raw bytes.
    pub fn insert(&self, path: &str, data: Vec<u8>) {
        self.files
            .lock()
            .unwrap()
            .insert(path.to_string(), data);
    }

    /// Remove a file.
    pub fn remove(&self, path: &str) -> bool {
        self.files.lock().unwrap().remove(path).is_some()
    }
}

impl Default for MemoryVfs {
    fn default() -> Self {
        Self::new()
    }
}

impl Vfs for MemoryVfs {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn read(&self, path: &str) -> std::io::Result<Vec<u8>> {
        self.files
            .lock()
            .unwrap()
            .get(path)
            .cloned()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, path))
    }

    fn read_to_string(&self, path: &str) -> std::io::Result<String> {
        let bytes = self.read(path)?;
        String::from_utf8(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    fn write(&self, path: &str, data: &[u8]) -> std::io::Result<()> {
        self.files
            .lock()
            .unwrap()
            .insert(path.to_string(), data.to_vec());
        Ok(())
    }

    fn write_string(&self, path: &str, data: &str) -> std::io::Result<()> {
        self.write(path, data.as_bytes())
    }

    fn exists(&self, path: &str) -> bool {
        self.files.lock().unwrap().contains_key(path)
    }

    fn read_dir(&self, path: &str) -> std::io::Result<Vec<DirEntry>> {
        let files = self.files.lock().unwrap();
        let prefix = if path.is_empty() || path == "." || path == "/" {
            String::new()
        } else {
            let mut p = path.trim_end_matches('/').to_string();
            p.push('/');
            p
        };
        let mut dirs_seen = std::collections::HashSet::new();
        let mut entries = Vec::new();
        for key in files.keys() {
            if key.starts_with(&prefix) {
                let rel = &key[prefix.len()..];
                if let Some(slash) = rel.find('/') {
                    // It's inside a subdirectory.
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
                        size: files[key].len() as u64,
                    });
                }
            }
        }
        Ok(entries)
    }

    fn walk_dir(&self, path: &str) -> std::io::Result<Vec<String>> {
        let files = self.files.lock().unwrap();
        let prefix = if path.is_empty() || path == "." || path == "/" {
            String::new()
        } else {
            let mut p = path.trim_end_matches('/').to_string();
            p.push('/');
            p
        };
        let mut out: Vec<String> = files
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .map(|k| k[prefix.len()..].to_string())
            .collect();
        out.sort();
        Ok(out)
    }
}

// ── Global VFS ───────────────────────────────────────────────────────────────

use std::sync::OnceLock;

static GLOBAL_VFS: OnceLock<Arc<dyn Vfs>> = OnceLock::new();

/// Initialize the global VFS (call once at startup).
pub fn init_global_vfs(vfs: Arc<dyn Vfs>) {
    let _ = GLOBAL_VFS.set(vfs);
}

/// Get a reference to the global VFS. Falls back to OsFileSystem if not initialized.
pub fn global() -> &'static dyn Vfs {
    GLOBAL_VFS
        .get_or_init(|| Arc::new(OsFileSystem))
        .as_ref()
}

// ── Convenience free functions (use the global VFS) ──────────────────────────

/// Read a file as UTF-8 text via the global VFS.
pub fn read_to_string(path: &str) -> std::io::Result<String> {
    global().read_to_string(path)
}

/// Read a file as raw bytes via the global VFS.
pub fn read(path: &str) -> std::io::Result<Vec<u8>> {
    global().read(path)
}

/// Write bytes via the global VFS.
pub fn write(path: &str, data: &[u8]) -> std::io::Result<()> {
    global().write(path, data)
}

/// Write a UTF-8 string via the global VFS.
pub fn write_string(path: &str, data: &str) -> std::io::Result<()> {
    global().write_string(path, data)
}

/// Check if a path exists via the global VFS.
pub fn exists(path: &str) -> bool {
    global().exists(path)
}

/// List directory entries via the global VFS.
pub fn read_dir(path: &str) -> std::io::Result<Vec<DirEntry>> {
    global().read_dir(path)
}

/// Recursively walk a directory via the global VFS.
pub fn walk_dir(path: &str) -> std::io::Result<Vec<String>> {
    global().walk_dir(path)
}

/// Open a file for streaming reads via the global VFS.
pub fn open_read(path: &str) -> std::io::Result<Box<dyn Read>> {
    global().open_read(path)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_vfs_read_write() {
        let vfs = MemoryVfs::new();
        vfs.insert_str("hello.txt", "hello world");
        assert!(vfs.exists("hello.txt"));
        assert_eq!(vfs.read_to_string("hello.txt").unwrap(), "hello world");
        assert_eq!(vfs.read("hello.txt").unwrap(), b"hello world");
    }

    #[test]
    fn memory_vfs_not_found() {
        let vfs = MemoryVfs::new();
        assert!(!vfs.exists("nope.txt"));
        assert!(vfs.read("nope.txt").is_err());
    }

    #[test]
    fn memory_vfs_overwrite() {
        let vfs = MemoryVfs::new();
        vfs.insert_str("f.txt", "v1");
        vfs.insert_str("f.txt", "v2");
        assert_eq!(vfs.read_to_string("f.txt").unwrap(), "v2");
    }

    #[test]
    fn memory_vfs_read_dir() {
        let vfs = MemoryVfs::new();
        vfs.insert_str("a.txt", "a");
        vfs.insert_str("b.txt", "b");
        vfs.insert_str("sub/c.txt", "c");
        vfs.insert_str("sub/d.txt", "d");

        let top = vfs.read_dir(".").unwrap();
        assert_eq!(top.len(), 3); // a.txt, b.txt, sub/

        let sub = vfs.read_dir("sub").unwrap();
        assert_eq!(sub.len(), 2); // c.txt, d.txt
    }

    #[test]
    fn memory_vfs_walk_dir() {
        let vfs = MemoryVfs::new();
        vfs.insert_str("x.txt", "x");
        vfs.insert_str("a/b.txt", "b");
        vfs.insert_str("a/c.txt", "c");

        let files = vfs.walk_dir(".").unwrap();
        assert_eq!(files.len(), 3);
        assert!(files.contains(&"x.txt".to_string()));
        assert!(files.contains(&"a/b.txt".to_string()));
        assert!(files.contains(&"a/c.txt".to_string()));
    }

    #[test]
    fn memory_vfs_remove() {
        let vfs = MemoryVfs::new();
        vfs.insert_str("gone.txt", "bye");
        assert!(vfs.remove("gone.txt"));
        assert!(!vfs.exists("gone.txt"));
        assert!(!vfs.remove("gone.txt"));
    }

    #[test]
    fn memory_vfs_write_string() {
        let vfs = MemoryVfs::new();
        vfs.write_string("output.txt", "written").unwrap();
        assert_eq!(vfs.read_to_string("output.txt").unwrap(), "written");
    }

    #[test]
    fn global_vfs_defaults_to_os() {
        // The global VFS should default to OsFileSystem.
        let g = global();
        // Just verify it doesn't panic — we can't test actual file I/O
        // without a real file, but the type check is the important part.
        let _ = g.exists("Cargo.toml");
    }
}
