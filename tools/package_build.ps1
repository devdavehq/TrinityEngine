param(
    [string]$OutputDir = "TrinityEngineBuild",
    [switch]$Release = $true
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if ($Release) {
    cargo build --release
    $exePath = Join-Path $projectRoot "target/release/Triengine.exe"
} else {
    cargo build
    $exePath = Join-Path $projectRoot "target/debug/Triengine.exe"
}

if (!(Test-Path $exePath)) {
    throw "Build output not found: $exePath"
}

$out = Join-Path $projectRoot $OutputDir
if (Test-Path $out) {
    Remove-Item $out -Recurse -Force
}
New-Item -ItemType Directory -Path $out | Out-Null

Copy-Item $exePath (Join-Path $out "TrinityEngine.exe") -Force

$maybeFiles = @(
    "engine_settings.toml"
)
foreach ($f in $maybeFiles) {
    $src = Join-Path $projectRoot $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $out $f) -Force
    }
}

$maybeDirs = @(
    "Content",
    "scenes",
    "meshes",
    "scripts",
    "docs"
)
foreach ($d in $maybeDirs) {
    $src = Join-Path $projectRoot $d
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $out $d) -Recurse -Force
    }
}

Write-Host ""
Write-Host "Portable package ready:"
Write-Host "  $out"
Write-Host ""
Write-Host "Run:"
Write-Host "  $((Join-Path $out 'TrinityEngine.exe'))"
