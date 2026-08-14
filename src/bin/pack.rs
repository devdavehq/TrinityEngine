// src/bin/pack.rs — TrinityEngine content packer.
//
// Packs a directory tree (typically Content/) into a single .pak archive that
// the engine can serve through its VFS layer instead of loose files.
//
// Usage:
//   pack <input-dir> <output.pak>
//   pack Content/ game.pak     <- packs the whole Content folder
//
// Paths inside the archive are relative to <input-dir>.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: pack <input-dir> <output.pak>");
        return ExitCode::FAILURE;
    }
    let dir = &args[1];
    let out = &args[2];

    println!("Packing '{}' -> '{}' ...", dir, out);
    let pak = match triengine::vfs::pak::PakFile::build_from_dir(dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to walk '{}': {}", dir, e);
            return ExitCode::FAILURE;
        }
    };

    if pak.is_empty() {
        eprintln!("No files found under '{}' — nothing to pack.", dir);
        return ExitCode::FAILURE;
    }

    if let Err(e) = pak.write_to(out) {
        eprintln!("Failed to write '{}': {}", out, e);
        return ExitCode::FAILURE;
    }

    println!(
        "Packed {} files ({} bytes) into {}.",
        pak.len(),
        pak.total_bytes(),
        out
    );
    ExitCode::SUCCESS
}
