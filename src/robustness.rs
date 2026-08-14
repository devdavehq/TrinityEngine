// src/robustness.rs — crash handling, persistent logs, graceful shutdown.
// --------------------------------------------------------------------------
// #5 GC/robustness:
//   • install_panic_hook()  writes every panic + backtrace to <appdata>/crash.log
//   • install()  routes tracing to both the console AND <appdata>/trinity-runtime.log
//   • GPU/resource cleanup is handled by explicit drop ordering in main().

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::PathBuf;

/// Where cross-session data lives (%LOCALAPPDATA%/TrinityEngine or ~/.local/share).
fn data_dir() -> PathBuf {
    crate::editor_persist::trinity_data_dir()
}

/// Install a panic hook that records the crash to a persistent file.
pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "non-string panic payload".to_string()
        };
        let location = info
            .location()
            .map(|l| format!(" at {}:{}", l.file(), l.line()))
            .unwrap_or_default();
        let backtrace = std::backtrace::Backtrace::force_capture();
        let msg = format!(
            "\n===== TrinityEngine panic {} =====\nmsg: {}\nthread: {}{}\n\n{}\n",
            crate::TRINITY_ENGINE_VERSION,
            payload,
            std::thread::current()
                .name()
                .unwrap_or("unknown")
                .to_string(),
            location,
            backtrace,
        );

        // Always print to the console so the crash is visible in the terminal.
        eprint!("{msg}");
        let path = data_dir().join("crash.log");
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "{msg}");
            eprintln!("Crash details appended to {}", path.display());
        }
    }));
}

/// A tracing writer that echoes every line to stdout AND to the runtime log.
pub struct TeeWriter {
    file: std::fs::File,
}

impl TeeWriter {
    fn open_runtime_log() -> Option<Self> {
        let dir = data_dir();
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("trinity-runtime.log");
        // Append across sessions; ~/.trinity or appdata dir is small text.
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok()?;
        Some(Self { file })
    }
}

impl tracing_subscriber::fmt::MakeWriter<'_> for TeeWriter {
    type Writer = TeeOutput;

    fn make_writer(&self) -> Self::Writer {
        // Each writer owns a clone of the file handle; O_APPEND makes writes
        // from multiple threads append safely.
        TeeOutput {
            file: self.file.try_clone().ok(),
            out: io::stdout(),
        }
    }
}

/// One tracing writer instance: stdout + file.
pub struct TeeOutput {
    file: Option<std::fs::File>,
    out: io::Stdout,
}

impl Write for TeeOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.out.write(buf)?;
        if let Some(f) = &mut self.file {
            let _ = f.write(buf);
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()?;
        if let Some(f) = &mut self.file {
            let _ = f.flush();
        }
        Ok(())
    }
}

/// Initialise tracing with an optional persistent file sink.
/// `RUST_LOG` still controls verbosity from the environment.
pub fn install_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(
            // Default: info, but panic/error/warn always shown.
            if cfg!(debug_assertions) { "info" } else { "warn" }
        ));

    if let Some(writer) = TeeWriter::open_runtime_log() {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(writer)
            .with_ansi(true)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}