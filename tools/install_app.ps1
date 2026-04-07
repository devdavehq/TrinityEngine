param(
    [string]$InstallDir = "$env:LOCALAPPDATA\\TrinityEngine",
    [switch]$CreateDesktopShortcut = $true
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$packageScript = Join-Path $PSScriptRoot "package_build.ps1"

& powershell -ExecutionPolicy Bypass -File $packageScript -OutputDir "TrinityEngineBuild" -Release

$buildDir = Join-Path $projectRoot "TrinityEngineBuild"
if (!(Test-Path $buildDir)) {
    throw "Package directory missing: $buildDir"
}

if (Test-Path $InstallDir) {
    Remove-Item $InstallDir -Recurse -Force
}
New-Item -ItemType Directory -Path $InstallDir | Out-Null
Copy-Item (Join-Path $buildDir "*") $InstallDir -Recurse -Force

$exe = Join-Path $InstallDir "TrinityEngine.exe"
if ($CreateDesktopShortcut) {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $lnkPath = Join-Path $desktop "TrinityEngine.lnk"
    $wsh = New-Object -ComObject WScript.Shell
    $shortcut = $wsh.CreateShortcut($lnkPath)
    $shortcut.TargetPath = $exe
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.IconLocation = $exe
    $shortcut.Save()
}

Write-Host ""
Write-Host "Installed TrinityEngine to:"
Write-Host "  $InstallDir"
Write-Host ""
Write-Host "Executable:"
Write-Host "  $exe"
