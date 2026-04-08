# Installs a release build into %LOCALAPPDATA%\Programs\TrinityEngine and adds a Start Menu shortcut.
# Run from repo root after: cargo build --release
# Not a full MSI/MSIX; this is a practical "launcher install" similar in spirit to a per-user app folder.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$exeSource = Join-Path $repoRoot "target\release\Triengine.exe"
if (-not (Test-Path $exeSource)) {
    Write-Error "Release binary not found. Run: cargo build --release"
}
$installDir = Join-Path $env:LOCALAPPDATA "Programs\TrinityEngine"
New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Copy-Item -Force $exeSource (Join-Path $installDir "Triengine.exe")
$icon = Join-Path $repoRoot "assets\trinity_icon.png"
$wsh = New-Object -ComObject WScript.Shell
$programsPath = [System.Environment]::GetFolderPath("Programs")
$shortcutPath = Join-Path $programsPath "TrinityEngine.lnk"
$sc = $wsh.CreateShortcut($shortcutPath)
$sc.TargetPath = Join-Path $installDir "Triengine.exe"
$sc.WorkingDirectory = $installDir
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Save()
Write-Host "Installed to $installDir and shortcut: $shortcutPath"
