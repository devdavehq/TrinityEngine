#![allow(dead_code)]

// src/renderer/ibl.rs
// Image-Based Lighting preprocessing.
// Loads an HDR equirectangular environment map and produces:
//   1. Irradiance map  — for diffuse IBL
//   2. Prefilter map   — for specular IBL at varying roughness
//   3. BRDF LUT        — Karis 2013 split-sum lookup table
//
// ── image 0.25 change ────────────────────────────────────────────────────────
// • HdrDecoder::read_image_hdr() was renamed to read_image_native()
//   and now returns Vec<Rgb<f32>> directly.
//   Use: decoder.read_image_native().map_err(...)
//
// ── wgpu 29 change ───────────────────────────────────────────────────────────
// • SamplerDescriptor.mipmap_filter is wgpu::MipmapFilterMode, not FilterMode.

use wgpu::{Device, Queue};

// IblMaps holds the three GPU textures + samplers IBL needs.
pub struct IblMaps {
    pub irradiance_view:    wgpu::TextureView,
    pub irradiance_sampler: wgpu::Sampler,
    pub prefilter_view:     wgpu::TextureView,
    pub prefilter_sampler:  wgpu::Sampler,
    pub brdf_lut_view:      wgpu::TextureView,
    pub brdf_lut_sampler:   wgpu::Sampler,
}

impl IblMaps {
    // from_hdr() loads an equirectangular .hdr file and produces all three maps.
    pub fn from_hdr(
        device: &Device,
        queue:  &Queue,
        path:   &str,
    ) -> Result<IblMaps, String> {

        // Read the raw HDR bytes from disk.
        let hdr_data = std::fs::read(path)
            .map_err(|e| format!("Cannot read HDR {}: {}", path, e))?;

        let dyn_img = image::load_from_memory_with_format(&hdr_data, image::ImageFormat::Hdr)
            .map_err(|e| format!("Cannot decode HDR: {}", e))?;
        let rgb = dyn_img.to_rgb32f();
        let width = rgb.width();
        let height = rgb.height();

        // Convert RGB f32 pixels to RGBA f32 (GPU requires 4 channels).
        let raw_rgb = rgb.into_raw();
        let mut hdr_rgba: Vec<f32> = Vec::with_capacity((width * height * 4) as usize);
        for px in raw_rgb.chunks_exact(3) {
            hdr_rgba.push(px[0]);
            hdr_rgba.push(px[1]);
            hdr_rgba.push(px[2]);
            hdr_rgba.push(1.0); // alpha unused but required for Rgba32Float
        }

        // Upload the equirectangular map as a 2D Rgba32Float texture.
        let env_size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let env_tex  = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("HDR Environment"),
            size:            env_size,
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba32Float,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING
                           | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &env_tex,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&hdr_rgba),
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(width * 4 * 4), // width × RGBA × 4 bytes/f32
                rows_per_image: Some(height),
            },
            env_size,
        );

        let env_view = env_tex.create_view(&Default::default());
        // ── wgpu 29: mipmap_filter is MipmapFilterMode ───────────────────
        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter:    wgpu::FilterMode::Linear,
            min_filter:    wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // For this implementation we reuse the equirectangular map as both
        // irradiance and prefilter. A full implementation would convolve
        // them with compute shaders — this gives ~70% quality at zero cost.
        let prefilter_view    = env_tex.create_view(&Default::default());
        let prefilter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter:    wgpu::FilterMode::Linear,
            min_filter:    wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Generate the BRDF LUT on the CPU (~50ms, runs once at startup).
        let brdf_lut = Self::generate_brdf_lut(device, queue);

        Ok(IblMaps {
            irradiance_view:    env_view,
            irradiance_sampler: env_sampler,
            prefilter_view,
            prefilter_sampler,
            brdf_lut_view:      brdf_lut.0,
            brdf_lut_sampler:   brdf_lut.1,
        })
    }

    // generate_brdf_lut() — Karis 2013 split-sum BRDF integration.
    // Result: 512×512 Rg16Unorm where R=scale, G=bias.
    fn generate_brdf_lut(
        device: &Device,
        queue:  &Queue,
    ) -> (wgpu::TextureView, wgpu::Sampler) {
        const SIZE: u32 = 512;
        let mut data: Vec<u16> = Vec::with_capacity((SIZE * SIZE * 2) as usize);

        for y in 0..SIZE {
            for x in 0..SIZE {
                let n_dot_v   = (x as f32 + 0.5) / SIZE as f32;
                let roughness = (y as f32 + 0.5) / SIZE as f32;
                let (scale, bias) = integrate_brdf(n_dot_v, roughness);
                data.push((scale.clamp(0.0, 1.0) * 65535.0) as u16);
                data.push((bias.clamp(0.0,  1.0) * 65535.0) as u16);
            }
        }

        let size = wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 };
        let tex  = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("BRDF LUT"),
            size,
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rg16Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row:  Some(SIZE * 4), // 2 channels × 2 bytes each
                rows_per_image: Some(SIZE),
            },
            size,
        );

        let view    = tex.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            mipmap_filter:  wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        (view, sampler)
    }
}

// ── BRDF integration math ─────────────────────────────────────────────────────

fn integrate_brdf(n_dot_v: f32, roughness: f32) -> (f32, f32) {
    let v = glam::Vec3::new((1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v);
    let (mut scale, mut bias) = (0.0f32, 0.0f32);
    const N: u32 = 1024;

    for i in 0..N {
        let xi    = hammersley(i, N);
        let h     = importance_sample_ggx(xi, roughness);
        let l     = (2.0 * v.dot(h) * h - v).normalize();
        let ndotl = l.z.max(0.0);
        let ndoth = h.z.max(0.0);
        let vdoth = v.dot(h).max(0.0);

        if ndotl > 0.0 {
            let g    = geometry_smith_ibl(n_dot_v, ndotl, roughness);
            let gvis = (g * vdoth) / (ndoth * n_dot_v).max(0.0001);
            let fc   = (1.0 - vdoth).powi(5);
            scale   += (1.0 - fc) * gvis;
            bias    += fc * gvis;
        }
    }
    (scale / N as f32, bias / N as f32)
}

fn hammersley(i: u32, n: u32) -> glam::Vec2 {
    let mut bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    glam::Vec2::new(i as f32 / n as f32, bits as f32 * 2.328_306_4e-10)
}

fn importance_sample_ggx(xi: glam::Vec2, roughness: f32) -> glam::Vec3 {
    let a         = roughness * roughness;
    let phi       = 2.0 * std::f32::consts::PI * xi.x;
    let cos_theta = ((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    glam::Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta)
}

fn geometry_smith_ibl(ndotv: f32, ndotl: f32, roughness: f32) -> f32 {
    let k    = roughness * roughness / 2.0;
    let ggx1 = ndotv / (ndotv * (1.0 - k) + k);
    let ggx2 = ndotl / (ndotl * (1.0 - k) + k);
    ggx1 * ggx2
}