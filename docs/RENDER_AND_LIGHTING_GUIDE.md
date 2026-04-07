# Triengine Render and Lighting Guide (Beginner-Friendly)

This guide explains what each feature does in simple words and how to use it.

## How rendering works right now

- The engine uses **rasterization**.
- Rasterization means: triangles are projected to screen pixels and shaded.
- This is the standard real-time technique used by most game engines.

## Lighting you currently have

- **Directional Light**: one "sun" light for direct light and shadows.
- **PBR Shading**: realistic material response (metal/roughness workflow).
- **IBL (Image-Based Lighting)**: ambient light from an environment map.
- **Shadow filtering (PCF/PCSS toggles)**: softer shadow edges.
- **Post-like feature toggles**: bloom, SSAO-like AO, volumetric fog, voxel GI prototype.

## Why some effects are prototypes right now

Some effects are currently implemented as lightweight in-shader approximations.
Reason: this keeps the engine stable and easy to tune while core architecture
is still being built (scene flow, hot reload, threading, culling, profiling).

True AAA versions require dedicated render passes and extra GPU resources.
Those are next steps and should be added in this order:

1. Real post-process Bloom chain
2. Real SSAO using depth/normal buffers
3. Volumetric fog raymarch pass
4. Real voxel GI pipeline (voxelization + cone tracing/probes)

## Easy-to-use presets

`render.preset` controls many options at once:

- `Mobile`: fastest, safest for low-end systems.
- `Balanced`: quality/performance mix.
- `Cinematic`: highest quality, heavy cost.
- `Custom`: use manual toggle values exactly.

## Feature cheat sheet

- `shadows_enabled`: master shadow toggle.
- `pcf_enabled`: soft shadow filter (medium cost).
- `pcss_enabled`: contact-soft shadows (higher cost).
- `ibl_enabled`: ambient/reflection lighting from environment.
- `probes_enabled`: extra ambient realism via probes.
- `culling_enabled`: skip far objects.
- `frustum_culling_enabled`: skip objects outside camera view.
- `bloom_enabled`: glow around highlights.
- `ssao_enabled`: adds contact darkening in corners.
- `volumetric_fog_enabled`: depth atmosphere/fog.
- `voxel_gi_enabled`: prototype bounced lighting style.

## Performance tip

If FPS drops:

1. switch preset to `Mobile`
2. disable `pcss_enabled`, `volumetric_fog_enabled`, `voxel_gi_enabled`
3. lower `shadow_resolution`
4. keep culling enabled

## Quick controls (current editor foundation)

- `F1` toggle Bloom (prints simple explanation)
- `F2` toggle SSAO (prints simple explanation)
- `F3` toggle Volumetric Fog (prints simple explanation)
- `F4` toggle Voxel GI prototype (prints simple explanation)
- `F5` cycle preset (Mobile/Balanced/Cinematic/Custom)
- `F10` open/close editor shell
- `F11` basic/advanced inspector foldout
- `[` / `]` lower/raise bloom strength (example slider behavior)
- `H` print hierarchy list
- `B` print asset browser list
- `F` add simple foliage patch

This is the first stage before full graphical UE-style panels.

## Build and run

From project root:

- `cargo check` -> compile check only
- `cargo run` -> build and launch engine window
- `cargo run --release` -> optimized build (recommended for performance tests)

## Material instances (master-material style, no node graph)

Simple workflow:

1. pick an entity with `N` / `M`
2. press:
   - `1` = `matte_black`
   - `2` = `silver_brushed`
   - `3` = `foliage_leaf`

These instances come from pre-defined master materials and only change values
(base color tint, metallic, roughness, AO multipliers), similar to UE-style
material instances but without opening a node editor.
