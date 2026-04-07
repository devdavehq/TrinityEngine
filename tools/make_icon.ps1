param(
    [string]$SourcePng,
    [string]$OutPng = "assets/trinity_icon.png"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$img = [System.Drawing.Image]::FromFile($SourcePng)
$w = $img.Width
$h = $img.Height
$cropH = [int]($h * 0.62)
$size = [Math]::Min($w, $cropH)
$x = [int](($w - $size) / 2)
$y = [int](($cropH - $size) / 2)
$srcRect = New-Object System.Drawing.Rectangle($x, $y, $size, $size)

$dst = New-Object System.Drawing.Bitmap(256, 256)
$g = [System.Drawing.Graphics]::FromImage($dst)
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$g.DrawImage(
    $img,
    (New-Object System.Drawing.Rectangle(0, 0, 256, 256)),
    $srcRect,
    [System.Drawing.GraphicsUnit]::Pixel
)

$outDir = Split-Path -Parent $OutPng
if ($outDir) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}
$dst.Save($OutPng, [System.Drawing.Imaging.ImageFormat]::Png)

$g.Dispose()
$dst.Dispose()
$img.Dispose()

Write-Host "Icon written to $OutPng"
