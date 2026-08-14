//! Spline-based road/path tool for terrain modification.
//!
//! Provides Catmull-Rom spline evaluation, terrain flattening along spline paths,
//! and influence sampling for any world-space point. Supports variable-width roads,
//! elevation offsets, material blending, and looping paths.
//!
//! # Overview
//!
//! A `TerrainSpline` defines a path through the world as an ordered list of `SplinePoint`
//! control points. Each control point carries its own width, elevation offset, and material
//! index, allowing the road to widen, narrow, dip, and rise between points. Catmull-Rom
//! interpolation produces smooth curves that pass through every control point.
//!
//! Given a `SplineSettings` configuration, the `sample_spline` function evaluates the
//! nearest segment for any world-space coordinate and returns a `SplineInfluence` describing
//! how strongly the spline affects that point. The `flatten_terrain` function applies this
//! influence across an entire heightmap grid, flattening terrain under the road with smooth
//! falloff and slope blending.

use std::f32;

/// A single control point on a terrain spline.
///
/// Each point defines the road geometry at a specific location: its world position,
/// width, how much terrain should be flattened beneath it, and which surface texture
/// to apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplinePoint {
    /// World-space X coordinate of this control point.
    pub x: f32,
    /// World-space Z coordinate of this control point.
    pub z: f32,
    /// Road width at this control point, in world units. Allows the road to widen or
    /// narrow between points (e.g. an intersection or merge lane).
    pub width: f32,
    /// How much to flatten the terrain below this point, in world units (Y axis).
    /// A value of 2.0 means the terrain will be pushed down by up to 2 units to
    /// create a level surface beneath the road.
    pub elevation_offset: f32,
    /// Index into the terrain's material/texture table for the road surface at this
    /// control point. Materials are blended between adjacent control points.
    pub material_index: u32,
}

impl SplinePoint {
    /// Creates a new `SplinePoint` with zero elevation offset and material index 0.
    pub fn new(x: f32, z: f32, width: f32) -> Self {
        Self {
            x,
            z,
            width,
            elevation_offset: 0.0,
            material_index: 0,
        }
    }

    /// Creates a new `SplinePoint` with all fields specified.
    pub fn full(
        x: f32,
        z: f32,
        width: f32,
        elevation_offset: f32,
        material_index: u32,
    ) -> Self {
        Self {
            x,
            z,
            width,
            elevation_offset,
            material_index,
        }
    }

    /// Linearly interpolates between two control points by parameter `t` (0..1).
    ///
    /// Width and elevation_offset are linearly blended. Material index uses the
    /// nearest control point (nearest-neighbor interpolation) since indices are
    /// discrete.
    pub fn lerp(&self, other: &SplinePoint, t: f32) -> SplinePoint {
        let inv = 1.0 - t;
        SplinePoint {
            x: self.x * inv + other.x * t,
            z: self.z * inv + other.z * t,
            width: self.width * inv + other.width * t,
            elevation_offset: self.elevation_offset * inv + other.elevation_offset * t,
            material_index: if t < 0.5 {
                self.material_index
            } else {
                other.material_index
            },
        }
    }

    /// Distance from this point to another in the XZ plane.
    pub fn distance_to(&self, other: &SplinePoint) -> f32 {
        let dx = other.x - self.x;
        let dz = other.z - self.z;
        (dx * dx + dz * dz).sqrt()
    }
}

/// A terrain spline defined by an ordered list of control points.
///
/// Splines can be open (start and end are distinct) or closed (the path loops back
/// to the start). Catmull-Rom interpolation is used between adjacent control points
/// to produce smooth curves.
#[derive(Debug, Clone)]
pub struct TerrainSpline {
    /// The ordered control points that define the spline path.
    pub points: Vec<SplinePoint>,
    /// If true, the spline forms a closed loop. The segment from the last point
    /// back to the first point is included.
    pub closed: bool,
    /// Human-readable name for this spline, e.g. "Main Highway" or "Forest Trail".
    pub name: String,
    /// Default material index for the entire spline. Individual control points may
    /// override this, but this serves as the fallback.
    pub material_index: u32,
}

impl TerrainSpline {
    /// Creates an empty, open spline with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            points: Vec::new(),
            closed: false,
            name: name.to_string(),
            material_index: 0,
        }
    }

    /// Creates a closed (looping) spline with the given name.
    pub fn new_closed(name: &str) -> Self {
        Self {
            points: Vec::new(),
            closed: true,
            name: name.to_string(),
            material_index: 0,
        }
    }

    /// Returns the number of control points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if the spline has no control points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns the number of segments in the spline.
    ///
    /// For an open spline with N points, there are N-1 segments.
    /// For a closed spline with N points, there are N segments.
    pub fn segment_count(&self) -> usize {
        if self.points.is_empty() {
            return 0;
        }
        if self.closed {
            self.points.len()
        } else {
            self.points.len().saturating_sub(1)
        }
    }
}

/// Settings that control how a spline modifies the terrain.
#[derive(Debug, Clone, Copy)]
pub struct SplineSettings {
    /// Number of intermediate subdivision steps between each pair of control
    /// points when evaluating the spline. Higher values produce smoother curves
    /// but cost more computation. A value of 8-16 is typical.
    pub subdivision_steps: u32,
    /// How far from the spline center line terrain is affected, in world units.
    /// Points farther than this distance are unaffected. This defines the
    /// half-width of the influence zone.
    pub flatten_radius: f32,
    /// Strength of terrain flattening at the spline center (0.0 = no effect,
    /// 1.0 = fully flattened to the elevation offset). The effect falls off
    /// smoothly to zero at `flatten_radius`.
    pub flatten_strength: f32,
    /// Distance in world units over which terrain blends from the flattened
    /// road surface back to the natural slope. Larger values produce gentler
    /// transitions.
    pub slope_blend_distance: f32,
}

impl Default for SplineSettings {
    fn default() -> Self {
        Self {
            subdivision_steps: 10,
            flatten_radius: 12.0,
            flatten_strength: 1.0,
            slope_blend_distance: 6.0,
        }
    }
}

/// The result of sampling a spline at a particular world-space point.
///
/// Describes how strongly the spline influences this point and what the interpolated
/// spline properties are at the nearest location on the path.
#[derive(Debug, Clone, Copy)]
pub struct SplineInfluence {
    /// Distance from the sampled point to the nearest point on the spline center
    /// line, in world units.
    pub distance_from_center: f32,
    /// How strongly the spline affects this point, ranging from 0.0 (no effect)
    /// to 1.0 (maximum effect at the center line). This accounts for the
    /// `flatten_strength` setting and distance falloff.
    pub influence_strength: f32,
    /// Interpolated road width at the nearest point on the spline.
    pub interpolated_width: f32,
    /// Interpolated elevation offset at the nearest point on the spline.
    pub interpolated_elevation: f32,
    /// Blend weight for the road surface material at this point. 0.0 means pure
    /// terrain material, 1.0 means pure road material.
    pub road_surface_weight: f32,
}

// ---------------------------------------------------------------------------
// Core spline math
// ---------------------------------------------------------------------------

/// Catmull-Rom interpolation between four control points.
///
/// Given four control points `p0`, `p1`, `p2`, `p3` and a parameter `t` in [0, 1],
/// returns an interpolated point on the curve between `p1` and `p2`. The curve
/// passes exactly through `p1` at t=0 and `p2` at t=1, with `p0` and `p3`
/// influencing the tangent directions.
///
/// # Arguments
///
/// * `p0` - The control point before the segment start (influences tangent at p1)
/// * `p1` - The start of the segment (curve passes through this at t=0)
/// * `p2` - The end of the segment (curve passes through this at t=1)
/// * `p3` - The control point after the segment end (influences tangent at p2)
/// * `t`  - Interpolation parameter in [0, 1]
pub fn catmull_rom_interpolate(p0: &SplinePoint, p1: &SplinePoint, p2: &SplinePoint, p3: &SplinePoint, t: f32) -> SplinePoint {
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom basis matrix coefficients
    let x = 0.5
        * ((2.0 * p1.x)
            + (-p0.x + p2.x) * t
            + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
            + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3);

    let z = 0.5
        * ((2.0 * p1.z)
            + (-p0.z + p2.z) * t
            + (2.0 * p0.z - 5.0 * p1.z + 4.0 * p2.z - p3.z) * t2
            + (-p0.z + 3.0 * p1.z - 3.0 * p2.z + p3.z) * t3);

    let width = 0.5
        * ((2.0 * p1.width)
            + (-p0.width + p2.width) * t
            + (2.0 * p0.width - 5.0 * p1.width + 4.0 * p2.width - p3.width) * t2
            + (-p0.width + 3.0 * p1.width - 3.0 * p2.width + p3.width) * t3);

    let elevation_offset = 0.5
        * ((2.0 * p1.elevation_offset)
            + (-p0.elevation_offset + p2.elevation_offset) * t
            + (2.0 * p0.elevation_offset - 5.0 * p1.elevation_offset + 4.0 * p2.elevation_offset - p3.elevation_offset) * t2
            + (-p0.elevation_offset + 3.0 * p1.elevation_offset - 3.0 * p2.elevation_offset + p3.elevation_offset) * t3);

    let material_index = if t < 0.5 {
        p1.material_index
    } else {
        p2.material_index
    };

    SplinePoint {
        x,
        z,
        width: width.max(0.0),
        elevation_offset,
        material_index,
    }
}

/// Finds the closest point on a line segment to a given point.
///
/// # Arguments
///
/// * `ax`, `az` - Start of the segment
/// * `bx`, `bz` - End of the segment
/// * `px`, `pz` - The query point
///
/// # Returns
///
/// A tuple `(closest_x, closest_z, t)` where `t` is the parameter along the
/// segment in [0, 1]. The closest world-space position is
/// `(ax + t*(bx-ax), az + t*(bz-az))`. The parameter `t` is clamped to [0, 1].
pub fn closest_point_on_segment(ax: f32, az: f32, bx: f32, bz: f32, px: f32, pz: f32) -> (f32, f32, f32) {
    let dx = bx - ax;
    let dz = bz - az;
    let len_sq = dx * dx + dz * dz;

    if len_sq < f32::EPSILON {
        // Segment is degenerate; return the start point.
        return (ax, az, 0.0);
    }

    let mut t = ((px - ax) * dx + (pz - az) * dz) / len_sq;
    t = t.clamp(0.0, 1.0);

    let cx = ax + t * dx;
    let cz = az + t * dz;
    (cx, cz, t)
}

// ---------------------------------------------------------------------------
// Spline access helpers
// ---------------------------------------------------------------------------

/// Returns a phantom control point for wrapping purposes.
///
/// For open splines, the first and last points are duplicated to allow
/// Catmull-Rom evaluation at the endpoints. For closed splines, the points
/// wrap around naturally.
fn get_extended_point(spline: &TerrainSpline, index: isize) -> SplinePoint {
    let n = spline.points.len() as isize;
    if n == 0 {
        return SplinePoint::new(0.0, 0.0, 0.0);
    }

    if spline.closed {
        let wrapped = ((index % n + n) % n) as usize;
        spline.points[wrapped].clone()
    } else {
        let clamped = index.max(0).min(n - 1) as usize;
        spline.points[clamped].clone()
    }
}

/// Returns the pair of control points bounding segment `seg_index`, plus their
/// Catmull-Rom neighbors.
///
/// Returns `(p0, p1, p2, p3)` where the segment runs from `p1` to `p2`.
fn segment_control_points(spline: &TerrainSpline, seg_index: usize) -> (SplinePoint, SplinePoint, SplinePoint, SplinePoint) {
    let i = seg_index as isize;
    let p0 = get_extended_point(spline, i - 1);
    let p1 = get_extended_point(spline, i);
    let p2 = get_extended_point(spline, i + 1);
    let p3 = get_extended_point(spline, i + 2);
    (p0, p1, p2, p3)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Adds a control point to the spline, maintaining sorted order along the path.
///
/// The new point is inserted after the existing point that is closest along the
/// path direction (minimizing total path length increase). For an empty spline,
/// the point is simply appended.
pub fn add_point(spline: &mut TerrainSpline, point: SplinePoint) {
    if spline.points.is_empty() {
        spline.points.push(point);
        return;
    }

    // Find the insertion index that minimizes the increase in total path length.
    let mut best_index = spline.points.len(); // default: append at end
    let mut best_cost = f32::MAX;

    for i in 0..spline.points.len() {
        let prev_len = if i == 0 {
            0.0
        } else {
            spline.points[i - 1].distance_to(&spline.points[i])
        };

        let new_prev_len = if i == 0 {
            0.0
        } else {
            spline.points[i - 1].distance_to(&point)
        };

        let new_next_len = if i < spline.points.len() {
            point.distance_to(&spline.points[i])
        } else {
            0.0
        };

        // Cost: difference in total path length before and after insertion
        let cost = new_prev_len + new_next_len - prev_len;
        if cost < best_cost {
            best_cost = cost;
            best_index = i;
        }
    }

    spline.points.insert(best_index, point);
}

/// Removes a control point at the given index.
///
/// # Panics
///
/// Panics if `index` is out of bounds.
pub fn remove_point(spline: &mut TerrainSpline, index: usize) {
    assert!(
        index < spline.points.len(),
        "remove_point: index {} out of bounds (len={})",
        index,
        spline.points.len()
    );
    spline.points.remove(index);
}

/// Computes the total path length of the spline in world units.
///
/// For an open spline, this is the sum of distances between consecutive control
/// points. For a closed spline, an additional segment from the last point back
/// to the first is included.
///
/// Note: This computes straight-line distances between control points, not
/// Catmull-Rom arc lengths. For smoother length estimates, use
/// `subdivided_length` with your desired step count.
pub fn total_length(spline: &TerrainSpline) -> f32 {
    if spline.points.len() < 2 {
        return 0.0;
    }

    let mut len = 0.0;
    for i in 1..spline.points.len() {
        len += spline.points[i - 1].distance_to(&spline.points[i]);
    }

    if spline.closed && !spline.points.is_empty() {
        let last = spline.points.len() - 1;
        len += spline.points[last].distance_to(&spline.points[0]);
    }

    len
}

/// Computes total path length using Catmull-Rom subdivision for higher accuracy.
///
/// This evaluates the spline at `steps` subdivisions per segment and sums the
/// arc lengths of the resulting polyline.
pub fn subdivided_length(spline: &TerrainSpline, steps: u32) -> f32 {
    if spline.points.len() < 2 || steps == 0 {
        return total_length(spline);
    }

    let segments = spline.segment_count();
    if segments == 0 {
        return 0.0;
    }

    let mut len = 0.0f32;
    let step_count = steps.max(1);

    for seg in 0..segments {
        let (p0, p1, p2, p3) = segment_control_points(spline, seg);
        let mut prev = catmull_rom_interpolate(&p0, &p1, &p2, &p3, 0.0);

        for s in 1..=step_count {
            let t = s as f32 / step_count as f32;
            let current = catmull_rom_interpolate(&p0, &p1, &p2, &p3, t);
            let dx = current.x - prev.x;
            let dz = current.z - prev.z;
            len += (dx * dx + dz * dz).sqrt();
            prev = current;
        }
    }

    len
}

/// Samples the spline at a world-space position and returns influence information.
///
/// For a given `(world_x, world_z)`, this function:
/// 1. Iterates over all spline segments
/// 2. For each segment, finds the closest point on the Catmull-Rom curve
/// 3. Computes the perpendicular distance from the query point to the curve
/// 4. Returns the influence from the closest segment, if within `flatten_radius`
///
/// # Returns
///
/// `Some(SplineInfluence)` if the point is within range of the spline, `None`
/// otherwise.
pub fn sample_spline(
    spline: &TerrainSpline,
    world_x: f32,
    world_z: f32,
    settings: &SplineSettings,
) -> Option<SplineInfluence> {
    if spline.points.len() < 2 {
        return None;
    }

    let segments = spline.segment_count();
    if segments == 0 {
        return None;
    }

    let mut best_influence: Option<SplineInfluence> = None;
    let mut best_dist = f32::MAX;

    let steps = settings.subdivision_steps.max(1);

    for seg in 0..segments {
        let (p0, p1, p2, p3) = segment_control_points(spline, seg);

        // Subdivide the segment and find the closest sub-segment
        let mut prev_pt = catmull_rom_interpolate(&p0, &p1, &p2, &p3, 0.0);
        for s in 1..=steps {
            let t_local = s as f32 / steps as f32;
            let cur_pt = catmull_rom_interpolate(&p0, &p1, &p2, &p3, t_local);

            // Find closest point on this straight sub-segment
            let (cx, cz, _t) = closest_point_on_segment(
                prev_pt.x,
                prev_pt.z,
                cur_pt.x,
                cur_pt.z,
                world_x,
                world_z,
            );

            let dx = world_x - cx;
            let dz = world_z - cz;
            let dist = (dx * dx + dz * dz).sqrt();

            if dist < best_dist {
                best_dist = dist;

                // Interpolate spline properties at this position
                let global_t = (seg as f32 + _t) / segments as f32;
                let interp_pt =
                    sample_spline_at_t(spline, global_t, segments);

                let influence_norm =
                    (1.0 - dist / settings.flatten_radius).clamp(0.0, 1.0);
                let strength = influence_norm * settings.flatten_strength;

                let half_width = interp_pt.width * 0.5;
                let road_weight = if dist <= half_width {
                    1.0
                } else if dist <= settings.flatten_radius {
                    1.0 - ((dist - half_width)
                        / (settings.flatten_radius - half_width).max(f32::EPSILON))
                } else {
                    0.0
                };

                best_influence = Some(SplineInfluence {
                    distance_from_center: dist,
                    influence_strength: strength,
                    interpolated_width: interp_pt.width,
                    interpolated_elevation: interp_pt.elevation_offset,
                    road_surface_weight: road_weight.clamp(0.0, 1.0),
                });
            }

            prev_pt = cur_pt;
        }
    }

    // Only return influence if within the flatten radius
    if best_dist <= settings.flatten_radius {
        best_influence
    } else {
        None
    }
}

/// Samples the spline at a global parameter `t` in [0, 1].
///
/// `t` is mapped to a segment index and local parameter, then Catmull-Rom
/// interpolation is applied.
fn sample_spline_at_t(
    spline: &TerrainSpline,
    t: f32,
    segments: usize,
) -> SplinePoint {
    if segments == 0 || spline.points.is_empty() {
        return SplinePoint::new(0.0, 0.0, 0.0);
    }

    let t_clamped = t.clamp(0.0, 1.0);
    let seg_f = t_clamped * segments as f32;
    let seg_index = (seg_f as usize).min(segments - 1);
    let local_t = seg_f - seg_index as f32;
    let local_t = local_t.clamp(0.0, 1.0);

    let (p0, p1, p2, p3) = segment_control_points(spline, seg_index);
    catmull_rom_interpolate(&p0, &p1, &p2, &p3, local_t)
}

/// Flattens a heightmap in-place along the spline path.
///
/// For each grid cell within `flatten_radius` of the spline, the terrain height
/// is blended toward the spline's elevation offset using a smooth falloff. The
/// `slope_blend_distance` setting controls how gradually the flattened area
/// transitions back to the natural terrain.
///
/// # Arguments
///
/// * `heights` - Mutable slice of heightmap values, row-major order
/// * `width`   - Number of columns in the heightmap
/// * `depth`   - Number of rows in the heightmap
/// * `cell_size` - World-space size of each grid cell (square cells assumed)
/// * `spline`  - The terrain spline to flatten along
/// * `settings` - Spline flattening settings
///
/// # Heightmap Layout
///
/// The heightmap is indexed as `heights[row * width + col]` where row 0 is the
/// minimum-Z edge and col 0 is the minimum-X edge. The world position of cell
/// `(row, col)` is `(col * cell_size, row * cell_size)` (assuming origin at 0,0).
pub fn flatten_terrain(
    heights: &mut [f32],
    width: usize,
    depth: usize,
    cell_size: f32,
    spline: &TerrainSpline,
    settings: &SplineSettings,
) {
    if spline.points.len() < 2 || cell_size <= 0.0 {
        return;
    }

    // Pre-sample the spline onto a fine grid for fast distance queries.
    let sample_steps = 512u32;
    let segments = spline.segment_count();
    let mut sample_xs: Vec<f32> = Vec::with_capacity((sample_steps + 1) as usize);
    let mut sample_zs: Vec<f32> = Vec::with_capacity((sample_steps + 1) as usize);
    let mut sample_widths: Vec<f32> = Vec::with_capacity((sample_steps + 1) as usize);
    let mut sample_elevs: Vec<f32> = Vec::with_capacity((sample_steps + 1) as usize);

    for i in 0..=sample_steps {
        let t = i as f32 / sample_steps as f32;
        let pt = sample_spline_at_t(spline, t, segments);
        sample_xs.push(pt.x);
        sample_zs.push(pt.z);
        sample_widths.push(pt.width);
        sample_elevs.push(pt.elevation_offset);
    }

    let total_radius = settings.flatten_radius + settings.slope_blend_distance;

    for row in 0..depth {
        let cell_z = (row as f32 + 0.5) * cell_size;
        for col in 0..width {
            let cell_x = (col as f32 + 0.5) * cell_size;
            let idx = row * width + col;

            // Quick AABB rejection test
            let min_spline_x = sample_xs.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_spline_x = sample_xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_spline_z = sample_zs.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_spline_z = sample_zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            if cell_x < min_spline_x - total_radius
                || cell_x > max_spline_x + total_radius
                || cell_z < min_spline_z - total_radius
                || cell_z > max_spline_z + total_radius
            {
                continue;
            }

            // Find closest sample point
            let mut best_dist = f32::MAX;
            let mut best_width = 0.0f32;
            let mut best_elev = 0.0f32;

            for s in 0..sample_xs.len() {
                let dx = cell_x - sample_xs[s];
                let dz = cell_z - sample_zs[s];
                let dist = (dx * dx + dz * dz).sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_width = sample_widths[s];
                    best_elev = sample_elevs[s];
                }
            }

            if best_dist > total_radius {
                continue;
            }

            let original_height = heights[idx];

            // Compute flattening blend factor
            let half_width = best_width * 0.5;
            let blend_factor = if best_dist <= half_width {
                // Fully within the road width: apply full flattening
                settings.flatten_strength
            } else if best_dist <= settings.flatten_radius {
                // In the flatten zone: smooth falloff from road edge to flatten radius
                let t = (best_dist - half_width)
                    / (settings.flatten_radius - half_width).max(f32::EPSILON);
                settings.flatten_strength * (1.0 - smoothstep(t))
            } else if best_dist <= total_radius {
                // In the slope blend zone: gentle transition back to natural terrain
                let t = (best_dist - settings.flatten_radius)
                    / settings.slope_blend_distance.max(f32::EPSILON);
                settings.flatten_strength * 0.5 * (1.0 - smoothstep(t))
            } else {
                0.0
            };

            // Target height: the flattened elevation (original minus offset)
            let target_height = original_height - best_elev;
            heights[idx] = original_height + (target_height - original_height) * blend_factor;
        }
    }
}

/// Smooth Hermite interpolation function (ease-in-ease-out).
///
/// Maps t from [0, 1] to [0, 1] with zero derivatives at both endpoints.
/// Equivalent to `3t^2 - 2t^3`.
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_spline() -> TerrainSpline {
        let mut spline = TerrainSpline::new("Test Road");
        spline.points.push(SplinePoint::full(0.0, 0.0, 8.0, 3.0, 0));
        spline.points.push(SplinePoint::full(100.0, 0.0, 8.0, 3.0, 0));
        spline.points.push(SplinePoint::full(200.0, 50.0, 10.0, 2.0, 1));
        spline.points.push(SplinePoint::full(300.0, 50.0, 6.0, 1.0, 1));
        spline
    }

    /// Spline with collinear points for precise distance testing.
    fn make_straight_spline() -> TerrainSpline {
        let mut spline = TerrainSpline::new("Straight Road");
        spline.points.push(SplinePoint::new(0.0, 10.0, 8.0));
        spline.points.push(SplinePoint::new(100.0, 10.0, 8.0));
        spline.points.push(SplinePoint::new(200.0, 10.0, 8.0));
        spline.points.push(SplinePoint::new(300.0, 10.0, 8.0));
        spline
    }

    #[test]
    fn spline_influence_center_edge_off_spline() {
        // Use a straight spline (collinear points) so Catmull-Rom doesn't deviate
        let spline = make_straight_spline();
        let settings = SplineSettings::default();

        // Point right at the center of the first segment (x=50, z=10)
        let center = sample_spline(&spline, 50.0, 10.0, &settings);
        assert!(center.is_some(), "Point at center should be in range");
        let c = center.unwrap();
        assert!(
            c.distance_from_center < 1.0,
            "Center point distance should be near zero, got {}",
            c.distance_from_center
        );
        assert!(
            c.influence_strength > 0.9,
            "Center influence should be high, got {}",
            c.influence_strength
        );

        // Point near the edge of the flatten radius
        let edge = sample_spline(&spline, 50.0, 15.0, &settings);
        assert!(edge.is_some(), "Point near edge should be in range");
        let e = edge.unwrap();
        assert!(
            e.distance_from_center < settings.flatten_radius,
            "Edge point should be within flatten radius"
        );
        assert!(
            e.influence_strength < c.influence_strength,
            "Edge influence ({}) should be less than center ({})",
            e.influence_strength,
            c.influence_strength
        );

        // Point far away from the spline
        let far = sample_spline(&spline, 50.0, 100.0, &settings);
        assert!(far.is_none(), "Point far away should return None");
    }

    #[test]
    fn terrain_flatten_verification() {
        // Use a straight spline along z=10 for predictable results
        let mut spline = TerrainSpline::new("Flatten Test");
        spline
            .points
            .push(SplinePoint::full(0.0, 10.0, 8.0, 3.0, 0));
        spline
            .points
            .push(SplinePoint::full(100.0, 10.0, 8.0, 3.0, 0));
        spline
            .points
            .push(SplinePoint::full(200.0, 10.0, 8.0, 3.0, 0));
        spline
            .points
            .push(SplinePoint::full(300.0, 10.0, 8.0, 3.0, 0));

        let settings = SplineSettings {
            subdivision_steps: 8,
            flatten_radius: 12.0,
            flatten_strength: 1.0,
            slope_blend_distance: 4.0,
        };

        let width = 64;
        let depth = 32;
        let cell_size = 10.0;

        // Initialize heightmap with a flat terrain at height 10.0
        let mut heights: Vec<f32> = vec![10.0; width * depth];

        flatten_terrain(&mut heights, width, depth, cell_size, &spline, &settings);

        // The spline runs along z=10 from x=0 to x=300
        // Cell at (col=5, row=1) => world (55, 15), distance ~5 from spline center at z=10
        let on_path_idx = width + 5;
        let on_path_height = heights[on_path_idx];
        assert!(
            on_path_height < 9.9,
            "On-path height should be lowered from 10.0, got {}",
            on_path_height
        );
        // With elevation_offset=3.0, target height is 7.0, should be close to that
        assert!(
            on_path_height < 8.0,
            "On-path height should be significantly below 10.0, got {}",
            on_path_height
        );

        // Check a point that should be unaffected (far from the path)
        // Cell at (col=0, row=31) => world (5, 315) which is far from spline
        let off_path_idx = 31 * width;
        let off_path_height = heights[off_path_idx];
        assert!(
            (off_path_height - 10.0).abs() < 0.01,
            "Off-path height should remain ~10.0, got {}",
            off_path_height
        );
    }

    #[test]
    fn total_length_calculation() {
        let spline = make_test_spline();
        let len = total_length(&spline);

        // Expected: 0->100 = 100, 100->(200,50) ≈ 111.8, 200->300 = 100
        // Total ≈ 311.8
        assert!(
            len > 300.0 && len < 330.0,
            "Total length should be ~311.8, got {}",
            len
        );

        // Empty spline
        let empty = TerrainSpline::new("empty");
        assert_eq!(total_length(&empty), 0.0);

        // Single point spline
        let mut single = TerrainSpline::new("single");
        single.points.push(SplinePoint::new(5.0, 5.0, 4.0));
        assert_eq!(total_length(&single), 0.0);
    }

    #[test]
    fn closest_point_on_segment_basic() {
        // Horizontal segment from (0,0) to (10,0), query at (5, 3)
        let (cx, cz, t) = closest_point_on_segment(0.0, 0.0, 10.0, 0.0, 5.0, 3.0);
        assert!((cx - 5.0).abs() < 0.001);
        assert!((cz - 0.0).abs() < 0.001);
        assert!((t - 0.5).abs() < 0.001);

        // Query before start
        let (_, _, t_before) = closest_point_on_segment(0.0, 0.0, 10.0, 0.0, -5.0, 0.0);
        assert!((t_before - 0.0).abs() < 0.001);

        // Query after end
        let (_, _, t_after) = closest_point_on_segment(0.0, 0.0, 10.0, 0.0, 15.0, 0.0);
        assert!((t_after - 1.0).abs() < 0.001);
    }

    #[test]
    fn catmull_rom_passes_through_control_points() {
        let p0 = SplinePoint::new(0.0, 0.0, 4.0);
        let p1 = SplinePoint::new(10.0, 0.0, 6.0);
        let p2 = SplinePoint::new(20.0, 10.0, 8.0);
        let p3 = SplinePoint::new(30.0, 10.0, 10.0);

        let at_0 = catmull_rom_interpolate(&p0, &p1, &p2, &p3, 0.0);
        assert!((at_0.x - p1.x).abs() < 0.001, "Should pass through p1 at t=0");
        assert!((at_0.z - p1.z).abs() < 0.001);

        let at_1 = catmull_rom_interpolate(&p0, &p1, &p2, &p3, 1.0);
        assert!((at_1.x - p2.x).abs() < 0.001, "Should pass through p2 at t=1");
        assert!((at_1.z - p2.z).abs() < 0.001);
    }

    #[test]
    fn add_remove_point_maintains_order() {
        let mut spline = TerrainSpline::new("ordered");
        add_point(&mut spline, SplinePoint::new(200.0, 0.0, 4.0));
        add_point(&mut spline, SplinePoint::new(0.0, 0.0, 4.0));
        add_point(&mut spline, SplinePoint::new(100.0, 0.0, 4.0));

        assert_eq!(spline.points.len(), 3);
        // Should be sorted along path: 0, 100, 200
        assert!((spline.points[0].x - 0.0).abs() < 0.001);
        assert!((spline.points[1].x - 100.0).abs() < 0.001);
        assert!((spline.points[2].x - 200.0).abs() < 0.001);

        remove_point(&mut spline, 1);
        assert_eq!(spline.points.len(), 2);
        assert!((spline.points[0].x - 0.0).abs() < 0.001);
        assert!((spline.points[1].x - 200.0).abs() < 0.001);
    }
}
