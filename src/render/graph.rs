// src/render/graph.rs
// ──────────────────────────────────────────────────────────────────────────────
// Render Graph
//
// WHY IT EXISTS:
//   Right now the renderer has one big draw_world() function that manually
//   executes shadow passes, geometry passes, bloom passes, etc in a fixed order.
//   Adding a new pass means editing that giant function and hoping you don't
//   break resource transitions.
//
//   A render graph declares passes and their resource dependencies. The graph
//   figures out execution order and resource transitions automatically. This is
//   how Frostbite, Frostbite, Horizon Zero Dawn, and all modern engines work.
//
// HOW IT WORKS:
//   1. You declare passes: "shadow pass" reads the scene geometry, writes depth.
//   2. You declare resources: "shadow_map" is a depth texture.
//   3. You connect them: shadow pass OUTPUTS shadow_map, main pass INPUTS shadow_map.
//   4. The graph topologically sorts passes, determines which passes can run
//      in parallel, and inserts resource barriers between dependent passes.
//
// LOW-END SUPPORT:
//   The graph lets us SKIP entire branches on low-end hardware. If bloom is
//   disabled, the bloom pass and its dependencies are simply not compiled into
//   the graph. Zero cost, zero shader complexity. Just toggle and the graph
//   rebuilds.
//
// CURRENT STATE:
//   This is the FOUNDATION. We define the types and the builder pattern.
//   Actual integration with the renderer happens as we restructure.
// ──────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;

// ── Resource Handle ───────────────────────────────────────────────────────────
// Opaque handle to a GPU resource (texture, buffer) owned by the render graph.
// The graph manages creation, lifetime, and reuse of these resources.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(u32);

impl ResourceId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

// ── Resource Description ──────────────────────────────────────────────────────
// What kind of GPU resource a pass needs. The graph creates these automatically.

#[derive(Clone, Debug)]
pub enum ResourceDesc {
    /// A color attachment (e.g. scene color, bloom texture).
    Texture {
        label: &'static str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    },
    /// A depth/stencil attachment (e.g. shadow map, scene depth).
    DepthTexture {
        label: &'static str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    },
    /// A buffer (e.g. instance buffer, readback buffer).
    Buffer {
        label: &'static str,
        size: u64,
        usage: wgpu::BufferUsages,
    },
}

// ── Pass Node ─────────────────────────────────────────────────────────────────
// One render/compute pass in the graph. Each pass declares what it reads
// and writes. The graph uses this to determine ordering.

pub struct PassNode {
    /// Human-readable name for debugging and profiling.
    pub name: &'static str,
    /// Resources this pass READS from (inputs).
    pub reads: Vec<ResourceId>,
    /// Resources this pass WRITES to (outputs).
    pub writes: Vec<ResourceId>,
    /// Whether this pass can be skipped (e.g. bloom disabled on low-end).
    pub enabled: bool,
    /// Priority for ordering passes with no dependencies (lower = earlier).
    pub priority: i32,
}

// ── RenderGraph ───────────────────────────────────────────────────────────────
// The graph itself. Built once (or when settings change), then executed each frame.

pub struct RenderGraph {
    /// All resource descriptions. Resources are created lazily on first use.
    resources: Vec<(ResourceId, ResourceDesc)>,
    resource_map: HashMap<String, ResourceId>,

    /// All passes in declaration order. Sorted on build().
    passes: Vec<PassNode>,

    /// Execution order after topological sort.
    sorted_order: Vec<usize>,

    /// Built and ready to execute?
    dirty: bool,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            resource_map: HashMap::new(),
            passes: Vec::new(),
            sorted_order: Vec::new(),
            dirty: true,
        }
    }

    // ── Resource declaration ──────────────────────────────────────────────

    /// Declare a named resource. Returns a handle for passing to passes.
    pub fn declare_resource(&mut self, name: &str, desc: ResourceDesc) -> ResourceId {
        if let Some(&existing) = self.resource_map.get(name) {
            return existing;
        }
        let id = ResourceId(self.resources.len() as u32);
        self.resource_map.insert(name.to_string(), id);
        self.resources.push((id, desc));
        self.dirty = true;
        id
    }

    /// Get a resource handle by name.
    pub fn get_resource(&self, name: &str) -> Option<ResourceId> {
        self.resource_map.get(name).copied()
    }

    // ── Pass declaration ──────────────────────────────────────────────────

    /// Add a pass that reads some resources and writes others.
    pub fn add_pass(
        &mut self,
        name: &'static str,
        reads: Vec<ResourceId>,
        writes: Vec<ResourceId>,
        priority: i32,
    ) {
        self.passes.push(PassNode {
            name,
            reads,
            writes,
            enabled: true,
            priority,
        });
        self.dirty = true;
    }

    /// Add a conditional pass (can be toggled at runtime).
    pub fn add_conditional_pass(
        &mut self,
        name: &'static str,
        reads: Vec<ResourceId>,
        writes: Vec<ResourceId>,
        priority: i32,
        enabled: bool,
    ) {
        self.passes.push(PassNode {
            name,
            reads,
            writes,
            enabled,
            priority,
        });
        self.dirty = true;
    }

    /// Enable or disable a pass by name (for runtime quality toggles).
    pub fn set_pass_enabled(&mut self, name: &str, enabled: bool) {
        for pass in &mut self.passes {
            if pass.name == name {
                pass.enabled = enabled;
                self.dirty = true;
            }
        }
    }

    // ── Build (topological sort) ──────────────────────────────────────────

    /// Compute execution order based on resource dependencies.
    /// Must be called before execute() if the graph changed.
    pub fn build(&mut self) {
        if !self.dirty {
            return;
        }

        let n = self.passes.len();
        // Simple topological sort: build adjacency from write->read edges.
        let mut in_degree = vec![0u32; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Map: resource_id -> index of the pass that writes it last.
        let mut resource_writer: Vec<Option<usize>> = vec![None; self.resources.len()];

        // Sorted by priority first, then declaration order.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by_key(|&i| (self.passes[i].priority, i));

        for &i in &indices {
            if !self.passes[i].enabled {
                continue;
            }
            // For each resource this pass reads, the writer must come before us.
            for &read_id in &self.passes[i].reads {
                if let Some(writer) = resource_writer[read_id.index()] {
                    adj[writer].push(i);
                    in_degree[i] += 1;
                }
            }
            // This pass now becomes the latest writer for its outputs.
            for &write_id in &self.passes[i].writes {
                resource_writer[write_id.index()] = Some(i);
            }
        }

        // Kahn's algorithm for topological sort.
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for i in 0..n {
            if self.passes[i].enabled && in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        self.sorted_order.clear();
        while let Some(i) = queue.pop_front() {
            self.sorted_order.push(i);
            for &next in &adj[i] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        // Disabled passes get appended at the end (they won't execute anyway).
        for i in 0..n {
            if !self.passes[i].enabled {
                self.sorted_order.push(i);
            }
        }

        self.dirty = false;
    }

    // ── Query ─────────────────────────────────────────────────────────────

    /// Get the ordered list of passes to execute.
    pub fn execution_order(&self) -> impl Iterator<Item = &PassNode> {
        self.sorted_order.iter().map(|&i| &self.passes[i])
    }

    /// How many passes are enabled?
    pub fn active_pass_count(&self) -> usize {
        self.passes.iter().filter(|p| p.enabled).count()
    }

    /// Total resources managed.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Reset the graph (for project switch or settings change).
    pub fn clear(&mut self) {
        self.resources.clear();
        self.resource_map.clear();
        self.passes.clear();
        self.sorted_order.clear();
        self.dirty = true;
    }
}

// ── Default Render Graph Setup ────────────────────────────────────────────────
// Convenience: builds the standard TrinityEngine render graph with all passes.
// Call this during renderer initialization. Toggle passes via set_pass_enabled().

pub fn build_default_graph(
    width: u32,
    height: u32,
    shadow_res: u32,
    bloom_enabled: bool,
    ssao_enabled: bool,
    fog_enabled: bool,
    voxel_gi_enabled: bool,
) -> RenderGraph {
    let mut graph = RenderGraph::new();

    // ── Resources ────────────────────────────────────────────────────────
    let scene_color = graph.declare_resource(
        "scene_color",
        ResourceDesc::Texture {
            label: "Scene Color",
            width,
            height,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        },
    );

    let scene_depth = graph.declare_resource(
        "scene_depth",
        ResourceDesc::DepthTexture {
            label: "Scene Depth",
            width,
            height,
            format: wgpu::TextureFormat::Depth32Float,
        },
    );

    let shadow_map = graph.declare_resource(
        "shadow_map",
        ResourceDesc::DepthTexture {
            label: "Shadow Map",
            width: shadow_res,
            height: shadow_res,
            format: wgpu::TextureFormat::Depth32Float,
        },
    );

    let bloom_a = graph.declare_resource(
        "bloom_a",
        ResourceDesc::Texture {
            label: "Bloom A",
            width: width / 2,
            height: height / 2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
        },
    );

    let bloom_b = graph.declare_resource(
        "bloom_b",
        ResourceDesc::Texture {
            label: "Bloom B",
            width: width / 2,
            height: height / 2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
        },
    );

    let ssao_buffer = graph.declare_resource(
        "ssao_buffer",
        ResourceDesc::Texture {
            label: "SSAO Buffer",
            width,
            height,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
        },
    );

    let fog_volume = graph.declare_resource(
        "fog_volume",
        ResourceDesc::Texture {
            label: "Volumetric Fog",
            width,
            height,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
        },
    );

    // ── Passes ───────────────────────────────────────────────────────────
    // Priority order: lower number = earlier in execution.
    // 0: shadow, 1: geometry, 2: post-process, 3: final composite.

    graph.add_pass(
        "shadow_pass",
        vec![],
        vec![shadow_map],
        0,
    );

    graph.add_pass(
        "geometry_pass",
        vec![shadow_map],
        vec![scene_color, scene_depth],
        1,
    );

    graph.add_conditional_pass(
        "ssao_pass",
        vec![scene_depth],
        vec![ssao_buffer],
        2,
        ssao_enabled,
    );

    graph.add_conditional_pass(
        "bloom_extract",
        vec![scene_color],
        vec![bloom_a],
        3,
        bloom_enabled,
    );

    graph.add_conditional_pass(
        "bloom_blur_h",
        vec![bloom_a],
        vec![bloom_b],
        3,
        bloom_enabled,
    );

    graph.add_conditional_pass(
        "bloom_blur_v",
        vec![bloom_b],
        vec![bloom_a],
        3,
        bloom_enabled,
    );

    graph.add_conditional_pass(
        "bloom_composite",
        vec![scene_color, bloom_a],
        vec![scene_color],
        4,
        bloom_enabled,
    );

    graph.add_conditional_pass(
        "fog_pass",
        vec![scene_color, scene_depth],
        vec![fog_volume],
        5,
        fog_enabled,
    );

    graph.add_conditional_pass(
        "fog_composite",
        vec![scene_color, fog_volume],
        vec![scene_color],
        5,
        fog_enabled,
    );

    graph.add_conditional_pass(
        "voxel_gi_pass",
        vec![scene_color, scene_depth],
        vec![scene_color],
        6,
        voxel_gi_enabled,
    );

    graph.add_pass(
        "tonemap_pass",
        vec![scene_color],
        vec![scene_color],
        7,
    );

    graph.add_pass(
        "final_composite",
        vec![scene_color],
        vec![scene_color],
        8,
    );

    graph.build();
    graph
}
