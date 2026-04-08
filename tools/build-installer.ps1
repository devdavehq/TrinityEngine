$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$testScript = Join-Path $PSScriptRoot "build-test.ps1"

& powershell -ExecutionPolicy Bypass -File $testScript

$installer = Join-Path $projectRoot "TrinityEngineBuild\Installer.exe"
if (!(Test-Path $installer)) {
    throw "Installer output missing: $installer"
}

& $installer

Write-Host ""
Write-Host "Installer executed:"
Write-Host "  $installer"
