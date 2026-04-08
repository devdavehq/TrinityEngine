$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "Building release test binary..."
cargo build --release
cargo build --release --bin installer

$srcExe = Join-Path $projectRoot "target\release\Triengine.exe"
if (!(Test-Path $srcExe)) {
    throw "Release binary not found: $srcExe"
}

$outDir = Join-Path $projectRoot "TrinityEngineBuild"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$dstExe = Join-Path $outDir "Triengine.exe"
Copy-Item -Force $srcExe $dstExe
$srcInstaller = Join-Path $projectRoot "target\release\installer.exe"
if (Test-Path $srcInstaller) {
    Copy-Item -Force $srcInstaller (Join-Path $outDir "Installer.exe")
}

Write-Host ""
Write-Host "Test build ready:"
Write-Host "  $dstExe"
if (Test-Path (Join-Path $outDir "Installer.exe")) {
    Write-Host "Installer ready:"
    Write-Host "  $(Join-Path $outDir 'Installer.exe')"
}
