use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), String> {
    let exe = env::current_exe().map_err(|e| e.to_string())?;
    let build_dir = exe
        .parent()
        .ok_or_else(|| "Cannot resolve installer directory".to_string())?
        .to_path_buf();
    let source_app = {
        let primary = build_dir.join("Triengine.exe");
        if primary.exists() {
            primary
        } else {
            build_dir.join("TrinityEngine.exe")
        }
    };
    if !source_app.exists() {
        return Err(format!(
            "Missing engine executable next to installer: {}",
            source_app.to_string_lossy()
        ));
    }

    let local_app_data =
        env::var("LOCALAPPDATA").map_err(|_| "LOCALAPPDATA is not set".to_string())?;
    let install_dir = PathBuf::from(local_app_data).join("Programs").join("TrinityEngine");
    fs::create_dir_all(&install_dir).map_err(|e| e.to_string())?;
    let target_exe = install_dir.join("Triengine.exe");
    fs::copy(&source_app, &target_exe).map_err(|e| e.to_string())?;

    let ps = format!(
        "$programs=[Environment]::GetFolderPath('Programs');\
         $lnk=Join-Path $programs 'TrinityEngine.lnk';\
         $w=New-Object -ComObject WScript.Shell;\
         $s=$w.CreateShortcut($lnk);\
         $s.TargetPath='{}';\
         $s.WorkingDirectory='{}';\
         $s.IconLocation='{}';\
         $s.Save();",
        target_exe.to_string_lossy().replace('\'', "''"),
        install_dir.to_string_lossy().replace('\'', "''"),
        target_exe.to_string_lossy().replace('\'', "''")
    );
    let _ = Command::new("powershell")
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", &ps])
        .status();

    tracing::info!("Installed TrinityEngine to {}", install_dir.to_string_lossy());
    tracing::info!("Run app: {}", target_exe.to_string_lossy());
    Ok(())
}
