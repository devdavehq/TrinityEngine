param(
    [string]$OutputDir = "GameBuild",
    [string]$GameName = "MyGame",
    [string]$Scene = "Content/scenes/main.scene",
    [switch]$NoPak = $false,
    [switch]$NoInstaller = $false,
    [switch]$Release = $true
)

# Build a shippable runtime (no editor/egui binaries) and assemble a portable
# game folder you can zip and ship. Launch the result with --game <scene>.
#
# The runtime is a single .exe that CONTAINS the whole engine (Rust compiles
# everything in — rendering, physics, Lua, audio). The game's data lives beside
# it:
#   GameBuild/
#     MyGame.exe             <-- the entire engine + your game bootstrap
#     game.pak               <-- ALL Content/assets packed (compressed) archive
#     engine_settings.toml   <-- settings (writable, runtime-tuned)
#
# The exe auto-detects game.pak at startup and serves all its data from the
# archive. That means the shipped folder is 100% self-contained: no engine
# folders, no source, no loose asset files to poke or pillage.
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if ($Release) {
    cargo build --release --no-default-features --features runtime
    $exePath = Join-Path $projectRoot "target/release/Triengine.exe"
} else {
    cargo build --no-default-features --features runtime
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

Copy-Item $exePath (Join-Path $out "$GameName.exe") -Force

# ── Runtime-tuned engine_settings.toml ─────────────────────────────────────
# Keep the project's [render] and [input] tuning, but write a hardened
# [runtime] section: hot-reload/file-watchers are disabled (the data ships in
# game.pak — watching a folder that isn't there just spawns error logs and
# wasted threads), and the startup scene is baked in.
$srcSettings = Join-Path $projectRoot "engine_settings.toml"
$outSettings = Join-Path $out "engine_settings.toml"

function Get-TomlSections {
    param([string]$Text)
    $sections = [ordered]@{}
    $current = ""
    $sections[$current] = [System.Collections.Generic.List[string]]::new()
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -match '^\s*\[(.+)\]\s*$') {
            $current = $Matches[1].Trim()
            if (-not $sections.Contains($current)) {
                $sections[$current] = [System.Collections.Generic.List[string]]::new()
            }
        } else {
            $sections[$current].Add($line)
        }
    }
    return $sections
}

function Quote-IfString([string]$v) {
    if ($v -eq "true" -or $v -eq "false") { return $v }
    if ($v -match '^-?\d+(\.\d+)?$') { return $v }
    return '"' + $v.Replace('"', '\"') + '"'
}

$newText = ""
if (Test-Path $srcSettings) {
    $sections = Get-TomlSections (Get-Content -Raw $srcSettings)
    foreach ($name in $sections.Keys) {
        if ($name -eq "runtime") { continue }
        $body = $sections[$name]
        if ($body.Count -gt 0 -and $name -ne "") {
            $newText += "[$name]`r`n"
        }
        $newText += ($body -join "`r`n")
        if ($body.Count -gt 0 -and $name -ne "") { $newText += "`r`n`r`n" }
    }
}
$newText += "[runtime]`r`n"
$newText += "startup_scene_path = $(Quote-IfString $Scene)`r`n"
$newText += "script_hot_reload_enabled = false`r`n"
$newText += "asset_hot_reload_enabled = false`r`n"
$newText += "autosave_enabled = true`r`n"
$newText += "autosave_interval_seconds = 30`r`n"
$newText += "max_fps = 60`r`n"
$newText += "window_width = 1280`r`n"
$newText += "window_height = 720`r`n"
$newText += "vsync_enabled = true`r`n"
Set-Content -Path $outSettings -Value $newText -Encoding UTF8
Write-Host "Shipped engine_settings.toml written (runtime-tuned, scene=$Scene)"

if ($NoPak) {
    # Legacy loose-file layout (still works — the engine falls back to disk).
    foreach ($d in @("Content", "assets")) {
        $src = Join-Path $projectRoot $d
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $out $d) -Recurse -Force
        }
    }
} else {
    # Production layout: pack all game data into one compressed archive.
    # Uses the `pack` CLI (src/bin/pack.rs), which reads Content AND assets
    # into a single `game.pak` (v2, deflate-compressed per entry).
    $packBin = Join-Path $projectRoot "target/release/pack.exe"
    if (!(Test-Path $packBin)) {
        Write-Host "Building pack tool..."
        cargo build --release --bin pack
    }
    if (!(Test-Path $packBin)) {
        throw "pack.exe not found after build: $packBin"
    }

    # Stage a data dir with Content + assets side by side so their pak keys
    # are exactly "Content/..." and "assets/...".
    $stage = Join-Path $env:TEMP "trinity_game_stage_$PID"
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    New-Item -ItemType Directory -Path $stage | Out-Null
    foreach ($d in @("Content", "assets")) {
        $src = Join-Path $projectRoot $d
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $stage $d) -Recurse -Force
        }
    }

    & $packBin $stage (Join-Path $out "game.pak")
    if (!(Test-Path (Join-Path $out "game.pak"))) {
        Remove-Item $stage -Recurse -Force
        throw "game.pak was not produced"
    }
    Remove-Item $stage -Recurse -Force
}

# Build and drop the per-game installer next to the game. Running it installs
# the whole folder into the user profile and creates Start Menu / Desktop
# shortcuts. It is skipped in debug/iteration builds via -NoInstaller.
if (-not $NoInstaller) {
    cargo build --release --bin game_installer
    $inst = Join-Path $projectRoot "target/release/game_installer.exe"
    if (Test-Path $inst) {
        Copy-Item $inst (Join-Path $out "game_installer.exe") -Force
    }
}

Write-Host ""
Write-Host "Game build ready:"
Write-Host "  $out"
Write-Host ""
Write-Host "Run (the scene is already baked into engine_settings.toml):"
Write-Host "  $((Join-Path $out \"$GameName.exe\")) --game $Scene"
Write-Host ""
Write-Host "Install to user profile (optional):"
Write-Host "  $((Join-Path $out 'game_installer.exe')) \"$GameName.exe\" \"$GameName\" --desktop"
if (-not $NoPak) {
    Write-Host ""
    Write-Host "Layout (self-contained):"
    Write-Host "  $GameName.exe   <-- whole engine compiled in"
    Write-Host "  game.pak        <-- all Content/assets, compressed archive"
    Write-Host "  game_installer.exe"
    Write-Host "  engine_settings.toml"
}