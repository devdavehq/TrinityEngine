# Triengine

Triengine is a Rust-based real-time game engine/editor focused on quality visuals with practical low-end toggles.

## Quick Start

1. Install stable Rust.
2. From project root, run:
   - `cargo check`
   - `cargo run`
3. Engine opens with the docked editor UI.

## Build Portable / Install

- Portable package (copies exe + required data):
  - `powershell -ExecutionPolicy Bypass -File tools/package_build.ps1`
  - Output: `TrinityEngineBuild/TrinityEngine.exe`
- Install like an app to your user profile and create desktop shortcut:
  - `powershell -ExecutionPolicy Bypass -File tools/install_app.ps1`
  - Installed app: `TrinityEngine.exe`

## First-Time Creator Flow

1. Open `Content/Textures` and add your texture files (`.png`, `.jpg`, `.jpeg`).
2. Add a primitive from Asset Browser (`Add Cube`, `Add Plane`, `Add Capsule`).
3. Select mesh in Hierarchy.
4. In Inspector:
   - adjust transform (`Move / Rotate / Scale`)
   - set material instance (`matte_black`, `silver_brushed`, `foliage_leaf`)
   - assign texture slots (`Albedo`, `Normal`, `Metallic+Roughness`)
5. Write gameplay Lua in `Content/Scripts` via in-editor Lua panel or external editor.
6. Attach script to selected mesh.
7. Use Play/Pause/Step controls to test behavior.

## Core Docs

- `docs/GETTING_STARTED_GAME_CREATOR.md`
- `docs/MATERIAL_TEXTURE_WORKFLOW.md`
- `docs/RENDER_AND_LIGHTING_GUIDE.md`

## Performance Notes

- Use lower presets if your laptop is under load.
- Fog can be toggled and tuned from Inspector.
- Keep heavy features (voxel GI, high bloom, high shadows) off on low-end profiles.
