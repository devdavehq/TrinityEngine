pub mod backend;

use std::collections::HashMap;
use std::fs;

use crate::assets::{AssetStore, Handle, Mesh};
use crate::components::{Position, Renderable};
use crate::profiler::FrameProfiler;
use crate::renderer::Renderer;
use crate::settings::{EngineSettings, RenderPreset};
use hecs::World;

pub fn describe_toggle(name: &str) -> &'static str {
    match name {
        "bloom" => "Bloom adds glow around bright lights. Nice for neon/sun highlights. Medium GPU cost.",
        "ssao" => "SSAO darkens corners and contact areas so objects feel grounded. Medium GPU cost.",
        "fog" => "Volumetric fog adds atmospheric depth and light shafts style haze. High GPU cost.",
        "voxel" => "Voxel GI prototype adds bounced light style fill. Expensive and still experimental.",
        "pcss" => "PCSS creates realistic soft shadows that get blurrier with distance. Expensive.",
        _ => "No description available for this toggle yet.",
    }
}

pub fn print_hierarchy(world: &World) {
    tracing::info!("[Hierarchy] Entities:");
    for entity in world.query::<hecs::Entity>().iter() {
        tracing::info!("  - {:?}", entity);
    }
}

pub fn print_asset_browser() {
    tracing::info!("[Assets] Browser:");
    for dir in ["meshes", "scenes", "scripts"] {
        tracing::info!("  {}:", dir);
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                tracing::info!("    - {}", entry.path().display());
            }
        }
    }
}

pub fn cycle_preset(current: RenderPreset) -> RenderPreset {
    match current {
        RenderPreset::Mobile => RenderPreset::Balanced,
        RenderPreset::Balanced => RenderPreset::Cinematic,
        RenderPreset::Cinematic => RenderPreset::Custom,
        RenderPreset::Custom => RenderPreset::Mobile,
    }
}

pub struct EditorShell {
    pub visible: bool,
    pub show_advanced: bool,
    pub layout: DockLayout,
}

pub struct DockLayout {
    pub top_toolbar: bool,
    pub left_hierarchy: bool,
    pub bottom_assets: bool,
    pub right_inspector: bool,
    pub profiler_panel: bool,
}

impl EditorShell {
    pub fn new() -> Self {
        Self {
            visible: false,
            show_advanced: false,
            layout: DockLayout {
                top_toolbar: true,
                left_hierarchy: true,
                bottom_assets: true,
                right_inspector: true,
                profiler_panel: true,
            },
        }
    }

    pub fn print_help() {
        tracing::info!("[Editor] Controls:");
        tracing::info!("  F10 -> toggle editor shell");
        tracing::info!("  F11 -> toggle advanced section");
        tracing::info!("  F5  -> cycle quality preset");
        tracing::info!("  [ / ] -> bloom strength down/up");
        tracing::info!("  H -> hierarchy panel, B -> asset browser, F -> add foliage patch");
        tracing::info!("  N/M select renderable, 1/2/3 apply material instance presets");
        tracing::info!("  J/K/L set animation state Idle/Walk/Run on selected entity");
    }

    pub fn render_snapshot(
        &self,
        world: &World,
        settings: &EngineSettings,
        renderer: Option<&Renderer>,
        profiler: &FrameProfiler,
    ) {
        if !self.visible {
            return;
        }

        tracing::info!("==================================================");
        tracing::info!(" Matte Black + Silver Editor Shell (Foundation) ");
        tracing::info!("==================================================");
        tracing::info!("[Theme] matte-black background + silver accents");
        tracing::info!("[Layout] top/left/right/bottom + profiler dock");

        if self.layout.top_toolbar {
            tracing::info!("[TopBar] Preset: {:?}", settings.render.preset);
            tracing::info!(
                "[TopBar] Bloom:{} SSAO:{} Fog:{} Voxel:{}",
                settings.render.bloom_enabled,
                settings.render.ssao_enabled,
                settings.render.volumetric_fog_enabled,
                settings.render.voxel_gi_enabled
            );
        }

        if self.layout.left_hierarchy {
            tracing::info!("\n[Left Dock: Hierarchy]");
            let mut count = 0usize;
            for e in world.query::<hecs::Entity>().iter() {
                tracing::info!(" - {:?}", e);
                count += 1;
                if count > 12 {
                    tracing::info!(" ...");
                    break;
                }
            }
        }

        if self.layout.bottom_assets {
            tracing::info!("\n[Bottom Dock: Asset Browser]");
            for dir in ["meshes", "scenes", "scripts"] {
                print!(" {}:", dir);
                let mut shown = 0usize;
                if let Ok(entries) = fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        if shown < 3 {
                            print!(" {}", entry.file_name().to_string_lossy());
                        }
                        shown += 1;
                    }
                }
                if shown > 3 {
                    print!(" ...");
                }
                println!();
            }
        }

        if self.layout.right_inspector {
            tracing::info!("\n[Right Dock: Inspector - Basic]");
            tracing::info!(" shadows_enabled: {}", settings.render.shadows_enabled);
            tracing::info!(" bloom_enabled:   {}", settings.render.bloom_enabled);
            tracing::info!(" ssao_enabled:    {}", settings.render.ssao_enabled);
            tracing::info!(" fog_enabled:     {}", settings.render.volumetric_fog_enabled);
            tracing::info!(" voxel_enabled:   {}", settings.render.voxel_gi_enabled);

            if self.show_advanced {
                tracing::info!("\n[Right Dock: Inspector - Advanced]");
                tracing::info!(" pcss_enabled: {}", settings.render.pcss_enabled);
                tracing::info!(" shadow_resolution: {}", settings.render.shadow_resolution);
                tracing::info!(" pcf_samples: {}", settings.render.pcf_samples);
                tracing::info!(" bloom_strength: {:.2}", settings.render.bloom_strength);
                tracing::info!(" ssao_strength: {:.2}", settings.render.ssao_strength);
                tracing::info!(" fog_density: {:.3}", settings.render.fog_density);
                tracing::info!(" voxel_gi_strength: {:.2}", settings.render.voxel_gi_strength);
                if let Some(r) = renderer {
                    tracing::info!(
                        " culling: {} dist={:.1} frustum={}",
                        r.features.culling_enabled, r.features.culling_distance, r.features.frustum_culling_enabled
                    );
                }
            }
        }

        if self.layout.profiler_panel {
            if let Some(text) = profiler.overlay_text() {
                tracing::info!("\n[Profiler Dock] {}", text);
            }
        }
        tracing::info!("==================================================");
    }
}

pub fn add_foliage_patch(
    world: &mut World,
    meshes: &mut AssetStore<Mesh>,
    mesh_cache: &mut HashMap<String, Handle<Mesh>>,
) {
    let mesh_path = "meshes/cube.obj".to_string();
    let handle = if let Some(h) = mesh_cache.get(&mesh_path) {
        *h
    } else {
        match Mesh::load(&mesh_path) {
            Ok(mesh) => {
                let h = meshes.add(mesh);
                mesh_cache.insert(mesh_path.clone(), h);
                h
            }
            Err(e) => {
                tracing::error!("[Foliage] Could not load {}: {}", mesh_path, e);
                return;
            }
        }
    };

    // Simple "paint patch": 64 tiny instances with pseudo-random jitter.
    for i in 0..64 {
        let fx = ((i * 17 % 100) as f32 / 100.0) * 8.0 - 4.0;
        let fz = ((i * 43 % 100) as f32 / 100.0) * 8.0 - 4.0;
        let scale = 0.15 + ((i * 29 % 100) as f32 / 1000.0);
        world.spawn((
            Position {
                x: fx,
                y: -0.35,
                z: fz,
            },
            Renderable {
                mesh: handle,
                color: [0.18, 0.42, 0.20],
                metallic: 0.0,
                roughness: 0.9,
                ao: 1.0,
                scale: [scale, scale * 2.4, scale],
            },
        ));
    }

    tracing::info!("[Foliage] Added foliage patch (64 instances).");
    tracing::info!("[Foliage] Goal: one-click easy placement like UE foliage mode.");
}
