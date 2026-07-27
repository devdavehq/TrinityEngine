// src/resources.rs
// ──────────────────────────────────────────────────────────────────────────────
// Resource budgets and GPU memory management.
//
// Tracks GPU memory usage across textures, buffers, and other resources.
// Enforces configurable limits to prevent out-of-memory on low-end hardware.
//
// Architecture:
//   ResourceTracker    — tracks all allocations with size and category
//   ResourceBudget     — configurable per-category limits
//   GpuMemoryStats     — current usage snapshot
//
// Usage:
//   Register allocations when creating textures/buffers.
//   Query usage before allocating large resources.
//   Check budgets in texture streaming system.
// ──────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;

/// Resource category for budget grouping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceCategory {
    Textures,
    Buffers,
    Pipelines,
    Meshes,
    RenderTargets,
    Shaders,
    ParticleBuffers,
    AudioBuffers,
}

impl ResourceCategory {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Textures => "Textures",
            Self::Buffers => "Buffers",
            Self::Pipelines => "Pipelines",
            Self::Meshes => "Meshes",
            Self::RenderTargets => "Render Targets",
            Self::Shaders => "Shaders",
            Self::ParticleBuffers => "Particle Buffers",
            Self::AudioBuffers => "Audio Buffers",
        }
    }
}

/// A tracked resource allocation.
#[derive(Clone, Debug)]
pub struct ResourceAllocation {
    pub id: u64,
    pub name: String,
    pub category: ResourceCategory,
    pub size_bytes: u64,
    pub created_frame: u64,
}

/// Memory usage snapshot.
#[derive(Clone, Debug, Default)]
pub struct GpuMemoryStats {
    pub total_bytes: u64,
    pub by_category: HashMap<ResourceCategory, u64>,
    pub allocation_count: usize,
}

impl GpuMemoryStats {
    pub fn total_mb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn category_mb(&self, cat: ResourceCategory) -> f64 {
        self.by_category.get(&cat).copied().unwrap_or(0) as f64 / (1024.0 * 1024.0)
    }
}

/// Budget limits per category.
#[derive(Clone, Debug)]
pub struct ResourceBudget {
    pub max_total_bytes: u64,
    pub per_category: HashMap<ResourceCategory, u64>,
    pub warning_threshold: f32, // 0.0-1.0, warn when usage exceeds this fraction
}

impl Default for ResourceBudget {
    fn default() -> Self {
        let mut per_category = HashMap::new();
        // Default budgets (512MB total, split across categories)
        per_category.insert(ResourceCategory::Textures, 256 * 1024 * 1024);      // 256MB
        per_category.insert(ResourceCategory::Buffers, 128 * 1024 * 1024);       // 128MB
        per_category.insert(ResourceCategory::RenderTargets, 64 * 1024 * 1024);  // 64MB
        per_category.insert(ResourceCategory::Meshes, 64 * 1024 * 1024);         // 64MB
        per_category.insert(ResourceCategory::Pipelines, 16 * 1024 * 1024);      // 16MB
        per_category.insert(ResourceCategory::Shaders, 8 * 1024 * 1024);         // 8MB
        per_category.insert(ResourceCategory::ParticleBuffers, 16 * 1024 * 1024); // 16MB
        per_category.insert(ResourceCategory::AudioBuffers, 32 * 1024 * 1024);   // 32MB

        Self {
            max_total_bytes: 512 * 1024 * 1024,
            per_category,
            warning_threshold: 0.85,
        }
    }
}

impl ResourceBudget {
    /// Low-end preset (256MB total).
    pub fn low_end() -> Self {
        let mut budget = Self::default();
        budget.max_total_bytes = 256 * 1024 * 1024;
        budget.per_category.insert(ResourceCategory::Textures, 128 * 1024 * 1024);
        budget.per_category.insert(ResourceCategory::Buffers, 64 * 1024 * 1024);
        budget.per_category.insert(ResourceCategory::RenderTargets, 32 * 1024 * 1024);
        budget
    }

    /// High-end preset (2GB total).
    pub fn high_end() -> Self {
        let mut budget = Self::default();
        budget.max_total_bytes = 2 * 1024 * 1024 * 1024;
        budget.per_category.insert(ResourceCategory::Textures, 1024 * 1024 * 1024);
        budget.per_category.insert(ResourceCategory::Buffers, 512 * 1024 * 1024);
        budget.per_category.insert(ResourceCategory::RenderTargets, 256 * 1024 * 1024);
        budget.per_category.insert(ResourceCategory::Meshes, 256 * 1024 * 1024);
        budget
    }

    /// Check if an allocation of this size would exceed the budget.
    pub fn can_allocate(&self, stats: &GpuMemoryStats, category: ResourceCategory, size_bytes: u64) -> bool {
        let new_total = stats.total_bytes + size_bytes;
        if new_total > self.max_total_bytes {
            return false;
        }
        let cat_used = stats.by_category.get(&category).copied().unwrap_or(0);
        let cat_limit = self.per_category.get(&category).copied().unwrap_or(u64::MAX);
        cat_used + size_bytes <= cat_limit
    }

    /// Check if any category is near its warning threshold.
    pub fn warnings(&self, stats: &GpuMemoryStats) -> Vec<(ResourceCategory, f32)> {
        let mut warnings = Vec::new();
        for (cat, limit) in &self.per_category {
            let used = stats.by_category.get(cat).copied().unwrap_or(0);
            let ratio = used as f32 / *limit as f32;
            if ratio >= self.warning_threshold {
                warnings.push((*cat, ratio));
            }
        }
        warnings
    }
}

/// Resource tracker — manages all known allocations.
pub struct ResourceTracker {
    allocations: HashMap<u64, ResourceAllocation>,
    next_id: u64,
    frame_count: u64,
}

impl ResourceTracker {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            next_id: 1,
            frame_count: 0,
        }
    }

    /// Register a new allocation.
    pub fn allocate(&mut self, name: &str, category: ResourceCategory, size_bytes: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.allocations.insert(id, ResourceAllocation {
            id,
            name: name.to_string(),
            category,
            size_bytes,
            created_frame: self.frame_count,
        });
        id
    }

    /// Remove a tracked allocation.
    pub fn free(&mut self, id: u64) -> Option<ResourceAllocation> {
        self.allocations.remove(&id)
    }

    /// Get current usage stats.
    pub fn stats(&self) -> GpuMemoryStats {
        let mut by_category = HashMap::new();
        let mut total = 0u64;
        for alloc in self.allocations.values() {
            total += alloc.size_bytes;
            *by_category.entry(alloc.category).or_insert(0) += alloc.size_bytes;
        }
        GpuMemoryStats {
            total_bytes: total,
            by_category,
            allocation_count: self.allocations.len(),
        }
    }

    /// Advance frame counter (call once per frame).
    pub fn advance_frame(&mut self) {
        self.frame_count += 1;
    }

    /// Get total tracked allocations.
    pub fn count(&self) -> usize {
        self.allocations.len()
    }

    /// Find largest allocations.
    pub fn largest(&self, n: usize) -> Vec<&ResourceAllocation> {
        let mut allocs: Vec<&ResourceAllocation> = self.allocations.values().collect();
        allocs.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
        allocs.into_iter().take(n).collect()
    }
}

impl Default for ResourceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_track() {
        let mut tracker = ResourceTracker::new();
        let id = tracker.allocate("test_texture", ResourceCategory::Textures, 1024 * 1024);
        assert!(id > 0);
        let stats = tracker.stats();
        assert_eq!(stats.total_bytes, 1024 * 1024);
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.category_mb(ResourceCategory::Textures), 1.0);
    }

    #[test]
    fn free_reduces_usage() {
        let mut tracker = ResourceTracker::new();
        let id = tracker.allocate("temp", ResourceCategory::Buffers, 512 * 1024);
        assert_eq!(tracker.stats().total_bytes, 512 * 1024);
        tracker.free(id);
        assert_eq!(tracker.stats().total_bytes, 0);
    }

    #[test]
    fn budget_can_allocate() {
        let budget = ResourceBudget::default();
        let stats = GpuMemoryStats::default();
        assert!(budget.can_allocate(&stats, ResourceCategory::Textures, 100 * 1024 * 1024));
        assert!(!budget.can_allocate(&stats, ResourceCategory::Textures, 300 * 1024 * 1024)); // over 256MB limit
    }

    #[test]
    fn budget_warnings() {
        let mut budget = ResourceBudget::default();
        budget.warning_threshold = 0.5; // lower threshold for testing
        let mut stats = GpuMemoryStats::default();
        stats.by_category.insert(ResourceCategory::Shaders, 5 * 1024 * 1024); // 5/8 MB = 62%
        let warnings = budget.warnings(&stats);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].0, ResourceCategory::Shaders);
    }

    #[test]
    fn largest_allocations() {
        let mut tracker = ResourceTracker::new();
        tracker.allocate("small", ResourceCategory::Textures, 100);
        tracker.allocate("big", ResourceCategory::Textures, 1000);
        tracker.allocate("medium", ResourceCategory::Buffers, 500);
        let largest = tracker.largest(2);
        assert_eq!(largest[0].name, "big");
        assert_eq!(largest[1].name, "medium");
    }

    #[test]
    fn category_usage() {
        let mut tracker = ResourceTracker::new();
        tracker.allocate("t1", ResourceCategory::Textures, 1000);
        tracker.allocate("t2", ResourceCategory::Textures, 2000);
        tracker.allocate("b1", ResourceCategory::Buffers, 500);
        let stats = tracker.stats();
        assert_eq!(stats.by_category[&ResourceCategory::Textures], 3000);
        assert_eq!(stats.by_category[&ResourceCategory::Buffers], 500);
    }

    #[test]
    fn preset_budgets() {
        let low = ResourceBudget::low_end();
        assert_eq!(low.max_total_bytes, 256 * 1024 * 1024);
        let high = ResourceBudget::high_end();
        assert_eq!(high.max_total_bytes, 2 * 1024 * 1024 * 1024);
    }
}
