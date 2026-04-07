# Material Texture Workflow

This page covers the polished per-entity map workflow.

## Supported texture slots per mesh entity

- `Albedo` (`sRGB`)
- `Normal` (linear)
- `Metallic+Roughness` (linear; metallic from B, roughness from G)

## Assigning textures

### Inspector method

1. Select a texture in Asset Browser.
2. Select mesh in Hierarchy.
3. In Inspector -> `Material Tools`:
   - `Set Albedo from Selected Texture`
   - `Set Normal from Selected Texture`
   - `Set Metallic+Roughness from Selected Texture`

### Drag/drop-style methods

- Hierarchy drop:
  - In Asset Browser click `Drag Texture To Mesh`.
  - Click target mesh in Hierarchy.
- Viewport drop:
  - Click in Viewport over target mesh while texture drag is armed.

## Persistence

- Prefab save/load stores texture slots:
  - `albedo_tex`
  - `normal_tex`
  - `mr_tex`

## Lua access

- Existing script API remains:
  - `get_texture_path(entity)`
  - `set_texture_path(entity, path)`
- These control the `Albedo` slot for compatibility.

## Troubleshooting

- If a map looks wrong:
  - verify file path exists in `Content/Textures`
  - verify normal map is in the normal slot
  - verify metallic+roughness map uses expected channels
- If map does not update:
  - confirm mesh entity is selected
  - confirm hot reload is enabled for content assets
