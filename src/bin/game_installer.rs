// src/bin/game_installer.rs
// ──────────────────────────────────────────────────────────────────────────────
// Per-game installer: installs an exported game folder into the current user's
// %LOCALAPPDATA%\Programs\<GameName> and creates a Start Menu + Desktop
// shortcut. Run from the built game folder:
//
//   game_installer.exe MyGame.exe "My Game"           <- self-installs next to it
//
// Because the game is one self-contained folder (exe + game.pak + settings),
// installing = copying that folder + making shortcuts. No registry writes, no
// admin rights, uninstall = delete the folder.
// ──────────────────────────────────────────────────────────────────────────────

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Args {
    /// The game executable to launch (may be a bare name or full path).
    game_exe: String,
    /// Human-readable game name (used for the shortcuts/folder).
    game_name: String,
    /// Whether to also create a desktop shortcut.
    desktop: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut desktop = false;
    if let Some(pos) = args.iter().position(|a| a == "--desktop") {
        args.remove(pos);
        desktop = true;
    }
    if args.is_empty() {
        return Err("Usage: game_installer <game.exe> \"Game Name\" [--desktop]".to_string());
    }
    let game_exe = args[0].clone();
    let game_name = if args.len() >= 2 {
        args[1].clone()
    } else {
        Path::new(&game_exe)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| game_exe.clone())
    };
    Ok(Args { game_exe, game_name, desktop })
}

fn copy_tree(src: &Path, dst: &Path) -> Result<(), String> {
    if !src.exists() {
        return Err(format!("Missing source: {}", src.to_string_lossy()));
    }
    if src.is_dir() {
        fs::create_dir_all(dst).map_err(|e| e.to_string())?;
        for entry in fs::read_dir(src).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            copy_tree(&entry.path(), &dst.join(entry.file_name()))?;
        }
    } else {
        fs::create_dir_all(dst.parent().unwrap()).map_err(|e| e.to_string())?;
        fs::copy(src, dst).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn make_shortcut(lnk: &Path, target: &Path, working: &Path, icon: &Path) {
    let ps = format!(
        "$w=New-Object -ComObject WScript.Shell;\
         $s=$w.CreateShortcut('{}');\
         $s.TargetPath='{}';\
         $s.WorkingDirectory='{}';\
         $s.IconLocation='{}';\
         $s.Save();",
        lnk.to_string_lossy().replace('\'', "''"),
        target.to_string_lossy().replace('\'', "''"),
        working.to_string_lossy().replace('\'', "''"),
        icon.to_string_lossy().replace('\'', "''"),
    );
    let _ = Command::new("powershell")
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", &ps])
        .status();
}

fn main() -> Result<(), String> {
    let args = parse_args()?;

    // The installer lives inside the built game folder.
    let self_dir = env::current_exe()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or_else(|| "Cannot resolve installer directory".to_string())?
        .to_path_buf();

    let game_exe = if Path::new(&args.game_exe).is_absolute() {
        PathBuf::from(&args.game_exe)
    } else {
        self_dir.join(&args.game_exe)
    };
    if !game_exe.exists() {
        return Err(format!(
            "Missing game executable: {}",
            game_exe.to_string_lossy()
        ));
    }

    let local_app_data =
        env::var("LOCALAPPDATA").map_err(|_| "LOCALAPPDATA is not set".to_string())?;
    let install_dir = PathBuf::from(local_app_data)
        .join("Programs")
        .join(&args.game_name);

    // Remove a previous install (fresh copy, never stale files).
    if install_dir.exists() {
        fs::remove_dir_all(&install_dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(&install_dir).map_err(|e| e.to_string())?;

    // Copy the WHOLE game folder (exe + game.pak + settings) so the installed
    // game is self-contained and runnable from its own directory.
    for entry in fs::read_dir(&self_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        // Skip the installer itself.
        if entry.file_name() == "game_installer.exe" {
            continue;
        }
        let name = entry.file_name();
        copy_tree(&entry.path(), &install_dir.join(&name))?;
    }

    let installed_exe = install_dir.join(game_exe.file_name().unwrap_or_default());
    let programs = env::var("ProgramData").ok().map(|_| {
        // Start Menu (current user).
        PathBuf::from(env::var("APPDATA").unwrap_or_default())
            .join("Microsoft")
            .join("Windows")
            .join("Start Menu")
            .join("Programs")
            .join(format!("{}.lnk", args.game_name))
    });
    if let Some(lnk) = programs {
        make_shortcut(&lnk, &installed_exe, &install_dir, &installed_exe);
    }
    if args.desktop {
        let desktop = PathBuf::from(env::var("USERPROFILE").unwrap_or_default())
            .join("Desktop")
            .join(format!("{}.lnk", args.game_name));
        make_shortcut(&desktop, &installed_exe, &install_dir, &installed_exe);
    }

    println!("Installed '{}' to {}", args.game_name, install_dir.to_string_lossy());
    println!("Run: {}", installed_exe.to_string_lossy());
    Ok(())
}
