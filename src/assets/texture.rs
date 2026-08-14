// src/assets/texture.rs
// Loads images from disk and uploads them to the GPU as wgpu textures.
//
// ── wgpu 29 change ───────────────────────────────────────────────────────────
// • SamplerDescriptor.mipmap_filter is wgpu::MipmapFilterMode, not FilterMode.
//   FilterMode still applies to mag_filter / min_filter.

use image::GenericImageView;
use wgpu::{
    AddressMode, Device, Extent3d, FilterMode, ImageCopyTexture, ImageDataLayout,
    Origin3d, Queue, SamplerDescriptor, TextureAspect, TextureDimension,
    TextureFormat, TextureUsages, TextureView,
};

// Texture bundles the GPU texture object, its shader view, and sampler.
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view:    TextureView,
    pub sampler: wgpu::Sampler,
    pub width:   u32,
    pub height:  u32,
}

impl Texture {
    // load() reads an image file and uploads it to the GPU.
    //
    // is_srgb: pass true for albedo (colour images painted by artists).
    //          The GPU will automatically linearise the colour on read.
    //          Pass false for normal/metallic/roughness — these store data, not colour.
    pub fn load(
        device:  &Device,
        queue:   &Queue,
        path:    &str,
        is_srgb: bool,
    ) -> Result<Texture, String> {
        // Read through the VFS so textures can come from a packed .pak archive
        // just as easily as from loose files.
        let bytes = crate::vfs::read(path)
            .map_err(|e| format!("Cannot read texture {}: {}", path, e))?;
        let img  = image::load_from_memory(&bytes)
            .map_err(|e| format!("Cannot decode texture {}: {}", path, e))?;
        let rgba = img.to_rgba8();
        let (w, h) = img.dimensions();

        let format = if is_srgb {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Rgba8Unorm
        };

        // Mip count: floor(log2(max(w,h))) + 1
        let mip_count = (w.max(h) as f32).log2().floor() as u32 + 1;

        let size    = Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(path),
            size,
            mip_level_count: mip_count,
            sample_count:    1,
            dimension:       TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING
                 | TextureUsages::COPY_DST
                 | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        queue.write_texture(
            ImageCopyTexture {
                texture:   &texture,
                mip_level: 0,
                origin:    Origin3d::ZERO,
                aspect:    TextureAspect::All,
            },
            &rgba,
            ImageDataLayout {
                offset:         0,
                bytes_per_row:  Some(4 * w),
                rows_per_image: Some(h),
            },
            size,
        );

        let view    = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            label:          Some("Texture Sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter:     FilterMode::Linear,
            min_filter:     FilterMode::Linear,
            // ── wgpu 29: MipmapFilterMode (separate type from FilterMode) ──
            mipmap_filter:  wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        Ok(Texture { texture, view, sampler, width: w, height: h })
    }

    // ── Default fallback textures ─────────────────────────────────────────────
    // Used when an entity has no texture assigned.

    pub fn default_white(device: &Device, queue: &Queue) -> Texture {
        Self::solid(device, queue, [255, 255, 255, 255], true)
    }

    // (128, 128, 255) in a normal map = pointing straight out (+Z in tangent space).
    pub fn default_normal(device: &Device, queue: &Queue) -> Texture {
        Self::solid(device, queue, [128, 128, 255, 255], false)
    }

    // B=0 (metallic=0), G=128 (roughness≈0.5).
    pub fn default_metallic_rough(device: &Device, queue: &Queue) -> Texture {
        Self::solid(device, queue, [0, 128, 0, 255], false)
    }

    fn solid(
        device:  &Device,
        queue:   &Queue,
        rgba:    [u8; 4],
        is_srgb: bool,
    ) -> Texture {
        let format = if is_srgb {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Rgba8Unorm
        };
        let size    = Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("Default Texture"),
            size,
            mip_level_count: 1,
            sample_count:    1,
            dimension:       TextureDimension::D2,
            format,
            usage:           TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats:    &[],
        });
        queue.write_texture(
            ImageCopyTexture {
                texture: &texture, mip_level: 0,
                origin: Origin3d::ZERO, aspect: TextureAspect::All,
            },
            &rgba,
            ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            size,
        );
        let view    = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter:    FilterMode::Nearest,
            min_filter:    FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Texture { texture, view, sampler, width: 1, height: 1 }
    }
}