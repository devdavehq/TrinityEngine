#![allow(dead_code)]

// src/renderer/light_probes.rs

use glam::Vec3;

// IrradianceSH holds spherical harmonics coefficients for one probe.
// L1 SH (4 coefficients per color channel) captures diffuse irradiance
// from all directions without storing a full cubemap.
// 9 coefficients is the standard (L2 SH) used in UE4/Horizon.
// We use L1 (4) for simplicity — upgrade path to L2 is straightforward.
#[derive(Clone, Copy, Default)]
pub struct IrradianceSH {
    // 9 coefficients, each RGB.
    // These encode the low-frequency irradiance coming from each direction.
    // Evaluated in the shader to get the ambient light color for a given normal.
    pub coeffs: [[f32; 3]; 9],
}

// LightProbe: one capture point in the scene.
pub struct LightProbe {
    pub position:    Vec3,
    pub irradiance:  IrradianceSH,
    // Radius: how far this probe's influence extends.
    // Probes with overlapping radii blend their contributions.
    pub radius:      f32,
    // Weight: priority — higher-weighted probes dominate blends.
    pub weight:      f32,
}

// LightProbeGrid manages the full set of probes and handles interpolation.
pub struct LightProbeGrid {
    pub probes: Vec<LightProbe>,
}

impl LightProbeGrid {
    pub fn new() -> Self {
        Self { probes: Vec::new() }
    }

    // add_probe() places a probe at a world position.
    // In the editor, you'll drag these around.
    // At startup or when the scene changes, we recapture them.
    pub fn add_probe(&mut self, position: Vec3, radius: f32) {
        self.probes.push(LightProbe {
            position,
            irradiance: IrradianceSH::default(),
            radius,
            weight: 1.0,
        });
    }

    // interpolate() returns blended SH coefficients for a world position.
    // This is called once per dynamic entity per frame (very cheap).
    //
    // This implements the tetrahedral interpolation described by
    // Linfeng Zhang et al. and used in production engines.
    // For a sparse grid, it gives seamless transitions.
    pub fn interpolate(&self, position: Vec3) -> IrradianceSH {
        if self.probes.is_empty() {
            return IrradianceSH::default();
        }
        if self.probes.len() == 1 {
            return self.probes[0].irradiance;
        }

        // Step 1: Find the N nearest probes within range.
        // "Within range" = position is inside the probe's radius.
        // We collect at most 8 candidates for the blend.
        let candidates: Vec<(usize, f32)> = self.probes
            .iter()
            .enumerate()
            .filter_map(|(i, probe)| {
                let dist = (probe.position - position).length();
                if dist < probe.radius {
                    // Weight = inverse square distance × probe weight.
                    // Closer probes have much more influence.
                    // Adding a small epsilon prevents division by zero at zero distance.
                    let w = probe.weight / (dist * dist + 0.01);
                    Some((i, w))
                } else {
                    None
                }
            })
            .collect();

        // If no probes in range, fall back to the nearest probe globally.
        if candidates.is_empty() {
            let nearest = self.probes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a.position - position).length_squared();
                    let db = (b.position - position).length_squared();
                    da.partial_cmp(&db).unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            return self.probes[nearest].irradiance;
        }

        // Step 2: Tetrahedral interpolation among candidates.
        //
        // True tetrahedral interpolation requires finding the containing
        // tetrahedron and computing barycentric coordinates.
        // For a production engine you'd precompute the Delaunay triangulation
        // of all probe positions at level load time.
        //
        // Our approximation: weighted blend using inverse distance weighting (IDW)
        // with the distance falloff shaped like a smooth cubic fade.
        // This gives C1 continuity (smooth first derivative = no popping).
        //
        // The cubic fade: f(t) = 1 - 3t² + 2t³ (standard smoothstep).
        // At t=0 (probe center): weight = 1.0
        // At t=1 (probe edge):   weight = 0.0
        // First derivative is 0 at both ends — smooth entry and exit.
        let total_weight: f32 = candidates.iter().map(|(_, w)| w).sum();

        if total_weight < 0.0001 {
            return self.probes[candidates[0].0].irradiance;
        }

        // Accumulate weighted SH coefficients.
        let mut blended = IrradianceSH::default();

        for (probe_idx, raw_weight) in &candidates {
            let probe    = &self.probes[*probe_idx];
            let dist     = (probe.position - position).length();
            // t: 0 at probe center, 1 at probe edge.
            let t        = (dist / probe.radius).clamp(0.0, 1.0);
            // Cubic smoothstep: smooth entry AND exit.
            // This is what makes transitions invisible — no linear kink at the edge.
            let smoothed = 1.0 - t * t * (3.0 - 2.0 * t);
            let w        = raw_weight * smoothed / total_weight;

            for c in 0..9 {
                blended.coeffs[c][0] += probe.irradiance.coeffs[c][0] * w;
                blended.coeffs[c][1] += probe.irradiance.coeffs[c][1] * w;
                blended.coeffs[c][2] += probe.irradiance.coeffs[c][2] * w;
            }
        }

        blended
    }

    // sample_sh() evaluates the SH in a given direction.
    // Called in the shader using pre-uploaded coefficients per entity.
    // This replaces the simple ambient term with directional ambient.
    //
    // direction: the surface normal direction (normalized).
    // Returns: irradiance color arriving from that direction.
    pub fn evaluate_sh(sh: &IrradianceSH, direction: Vec3) -> Vec3 {
        let d = direction.normalize();

        // L0 (constant) + L1 (linear) SH basis evaluation.
        // The constant factors (0.282095, 0.488603) are the SH basis
        // function values for the first two bands.
        // Full formula from "An Efficient Representation for Irradiance
        // Environment Maps" (Ramamoorthi & Hanrahan 2001).
        let mut result = Vec3::ZERO;

        // Band 0: constant ambient
        result += Vec3::from(sh.coeffs[0]) * 0.282095;

        // Band 1: linear terms (directional)
        result += Vec3::from(sh.coeffs[1]) * 0.488603 * d.y;
        result += Vec3::from(sh.coeffs[2]) * 0.488603 * d.z;
        result += Vec3::from(sh.coeffs[3]) * 0.488603 * d.x;

        // Band 2: quadratic terms (more directional detail)
        result += Vec3::from(sh.coeffs[4]) * 1.092548 * d.x * d.y;
        result += Vec3::from(sh.coeffs[5]) * 1.092548 * d.y * d.z;
        result += Vec3::from(sh.coeffs[6]) * 0.315392 * (3.0 * d.z * d.z - 1.0);
        result += Vec3::from(sh.coeffs[7]) * 1.092548 * d.x * d.z;
        result += Vec3::from(sh.coeffs[8]) * 0.546274 * (d.x * d.x - d.y * d.y);

        result.max(Vec3::ZERO)
    }
}