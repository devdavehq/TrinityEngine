# Getting Started: Game Creator Flow

This guide is the practical "build your first playable scene" path.

## 1) Launch and verify editor

- Run `cargo run`.
- Confirm panels are visible: Hierarchy, Inspector, Asset Browser, Lua Scripts.

## 2) Build a small scene

- Add a floor (`Add Plane` or `Add Floor`).
- Add gameplay mesh (`Add Cube` or `Add Capsule`).
- Select mesh in Hierarchy to edit transform.

## 3) Set up material and textures

- Put source textures in `Content/Textures`.
- Select mesh.
- In Inspector -> `Material Tools`:
  - assign albedo map
  - assign normal map
  - assign metallic+roughness map

## 4) Add script behavior

- Open `Lua Scripts` panel.
- Create script or select existing `.lua`.
- Attach script to selected mesh.
- Use `Save + Reload` for immediate iteration.

## 5) Simulate and debug

- Use `Play`, `Pause`, and `Step`.
- Check `Errors` window for Lua/runtime issues.

## 6) Save reusable gameplay pieces

- Select configured entity.
- `Save Selected As Prefab`.
- Spawn from `Prefabs` window to reuse setup.

## 7) Tune performance

- Keep `asset_hot_reload_enabled` on during iteration, off for stable profiling.
- Start from lower render presets on laptop hardware.
- Keep fog enabled only as much as needed for scene depth.
