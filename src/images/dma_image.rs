use std::{
    num::NonZeroU64,
    ops::Not,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    sync::Arc,
};

use drm_fourcc::DrmFourcc;
use glam::{UVec2, UVec3};
use timeline_syncobj::timeline_syncobj::TimelineSyncObj;
use vulkano::{
    device::DeviceOwnedVulkanObject as _,
    format::{Format, FormatFeatures},
    image::{
        Image, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageTiling, ImageType, ImageUsage,
        SampleCount, SubresourceLayout,
        sampler::{ComponentMapping, ComponentSwizzle},
        sys::RawImage,
        view::{ImageView, ImageViewCreateInfo},
    },
    memory::{
        DedicatedAllocation, DeviceMemory, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        MemoryAllocateInfo, MemoryImportInfo, MemoryRequirements, ResourceMemory,
    },
    sync::Sharing,
};

use crate::{
    Context,
    images::{AlphaMode, ImageRefSource, ImageResolution, YcbcrMetadata},
};

pub struct DmaImage {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    timeline: Arc<TimelineSyncObj>,
    alpha_mode: Option<AlphaMode>,
    requires_manual_srgb_mapping: bool,
}

pub enum DmaMemoryPlanes {
    Simple {
        fd: OwnedFd,
        planes: Vec<DmaMemoryPlane>,
    },
    Disjoint(Vec<(OwnedFd, DmaMemoryPlane)>),
}
impl DmaMemoryPlanes {
    fn get_planes(&self) -> impl Iterator<Item = &DmaMemoryPlane> {
        let mut iter_1 = None;
        let mut iter_2 = None;
        match self {
            DmaMemoryPlanes::Simple { fd: _, planes } => iter_1 = Some(planes.iter()),
            DmaMemoryPlanes::Disjoint(items) => iter_2 = Some(items.iter().map(|(_, v)| v)),
        }
        iter_1
            .into_iter()
            .flatten()
            .chain(iter_2.into_iter().flatten())
    }
    fn len(&self) -> usize {
        match self {
            DmaMemoryPlanes::Simple { fd: _, planes } => planes.len(),
            DmaMemoryPlanes::Disjoint(items) => items.len(),
        }
    }
}

pub struct DmaMemoryPlane {
    pub offset: u64,
    pub row_size: u64,
    pub array_element_size: u64,
    pub depth_slice_size: u64,
}

impl DmaImage {
    // TODO: replace format with fourcc and do conversion internally
    pub fn import(
        ctx: &Context,
        fourcc: DrmFourcc,
        modifier: u64,
        is_srgb: bool,
        ycbcr_metadata: Option<YcbcrMetadata>,
        // TODO: should this be an Option? there are formats without alpha (in addition to fourccs
        // that swizzle it to 1)
        alpha_mode: AlphaMode,
        res: ImageResolution,
        layers: u32,
        planes: DmaMemoryPlanes,
        timeline: impl Into<Arc<TimelineSyncObj>>,

        local_usage: ImageUsage,
        debug_name: Option<&str>,
        // TODO: return error instead
    ) -> Self {
        let format_info = map_fourcc(fourcc).unwrap();
        let requires_manual_srgb_mapping = (is_srgb && format_info.native_srgb_format.is_none())
            || alpha_mode == AlphaMode::PremultipliedElectrical;
        let format = if requires_manual_srgb_mapping.not() && let Some(f) = format_info.native_srgb_format {
            f
        } else {
            format_info.linear_format
        };
        let format_props = ctx.phys_dev.format_properties(format).unwrap();
        let format_props = format_props
            .drm_format_modifier_properties
            .iter()
            .find(|v| v.drm_format_modifier == modifier)
            .unwrap();
        if format_props.drm_format_modifier_plane_count != planes.len() as u32 {
            panic!("planes len doesn't match");
        }
        let is_disjoint = format_props
            .drm_format_modifier_tiling_features
            .contains(FormatFeatures::DISJOINT)
            && matches!(&planes, DmaMemoryPlanes::Disjoint(_));
        let raw_image = RawImage::new(
            ctx.device.clone(),
            ImageCreateInfo {
                flags: if is_disjoint {
                    ImageCreateFlags::DISJOINT
                } else {
                    ImageCreateFlags::empty()
                },
                image_type: res.image_type(),
                format,
                view_formats: vec![],
                extent: res.extents(),
                array_layers: layers,
                // TODO: expose?
                mip_levels: 1,
                samples: SampleCount::Sample1,
                tiling: ImageTiling::DrmFormatModifier,
                usage: local_usage,
                // TODO: support stencil usage?
                stencil_usage: None,
                sharing: Sharing::Exclusive,
                initial_layout: ImageLayout::Undefined,
                drm_format_modifiers: vec![modifier],
                drm_format_modifier_plane_layouts: planes
                    .get_planes()
                    .map(|p| SubresourceLayout {
                        offset: p.offset,
                        size: 0,
                        row_pitch: p.row_size,
                        array_pitch: NonZeroU64::new(p.array_element_size).map(|v| v.get()),
                        depth_pitch: NonZeroU64::new(p.depth_slice_size).map(|v| v.get()),
                    })
                    .collect(),
                external_memory_handle_types: ExternalMemoryHandleTypes::DMA_BUF,
                ..Default::default()
            },
        )
        // TODO: error handling
        .unwrap();
        let img_mem_reqs = raw_image.memory_requirements();
        let memories = match planes {
            DmaMemoryPlanes::Simple { fd, planes: _ } => {
                vec![import_mem(ctx, &raw_image, &img_mem_reqs[0], fd)]
            }
            DmaMemoryPlanes::Disjoint(planes) => planes
                .into_iter()
                .enumerate()
                // just indexing here is probably wrong?
                .map(|(i, (fd, _plane))| import_mem(ctx, &raw_image, &img_mem_reqs[i], fd))
                .collect(),
        };
        let image = match raw_image.bind_memory(memories).map(Arc::new) {
            Ok(v) => v,
            Err((err, _, _)) => {
                panic!("{err}")
            }
        };
        if format_info.is_ycbcr && ycbcr_metadata.is_none() {
            panic!("ycbcr format used without ycbcr metadata");
        }
        if !format_info.is_ycbcr && ycbcr_metadata.is_some() {
            panic!("ycbcr metadata provided for non ycbcr format");
        }
        let image_view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                component_mapping: format_info.swizzle,
                ..ImageViewCreateInfo::from_image(&image)
            },
        )
        .unwrap();
        if let Some(debug_name) = debug_name {
            _ = image.set_debug_utils_object_name(Some(debug_name));
            _ = image_view.set_debug_utils_object_name(Some(&format!("{debug_name} View")));
        }

        let format_has_alpha =
            format.components()[3] != 0 && format_info.swizzle.a != ComponentSwizzle::One;
        Self {
            image,
            image_view,
            timeline: timeline.into(),
            requires_manual_srgb_mapping,
            alpha_mode: format_has_alpha.then_some(alpha_mode),
        }
    }

    pub fn image(&self) -> Arc<Image> {
        self.image.clone()
    }
    pub fn image_view(&self) -> Arc<ImageView> {
        self.image_view.clone()
    }
    pub fn image_ref(&self) -> &Arc<Image> {
        &self.image
    }
    pub fn image_view_ref(&self) -> &Arc<ImageView> {
        &self.image_view
    }
}

fn import_mem(
    ctx: &Context,
    raw_image: &RawImage,
    mem_req: &MemoryRequirements,
    fd: OwnedFd,
) -> ResourceMemory {
    let fd_reqs = unsafe {
        ctx.device
            .memory_fd_properties(
                ExternalMemoryHandleType::DmaBuf,
                fd.try_clone().unwrap().into(),
            )
            .unwrap()
    };
    let type_bits = fd_reqs.memory_type_bits & mem_req.memory_type_bits;
    let memory_type_index = (0..32)
        .find(|v| (type_bits & (1 << *v)) != 0)
        .expect("unable to find memory type index for dmaimage");
    let raw_fd = fd.as_raw_fd();
    unsafe {
        match DeviceMemory::import(
            ctx.device.clone(),
            MemoryAllocateInfo {
                allocation_size: 0,
                memory_type_index,
                dedicated_allocation: (mem_req.prefers_dedicated_allocation
                    || mem_req.requires_dedicated_allocation)
                    .then(|| DedicatedAllocation::Image(raw_image)),
                ..Default::default()
            },
            MemoryImportInfo::Fd {
                handle_type: ExternalMemoryHandleType::DmaBuf,
                file: fd.into(),
            },
        )
        .map(ResourceMemory::new_dedicated)
        {
            Ok(v) => v,
            Err(err) => {
                // vulkano does not properly clean up the fd in error cases
                OwnedFd::from_raw_fd(raw_fd);
                // TODO: error handling
                panic!("{err}");
            }
        }
    }
}

impl ImageRefSource for DmaImage {
    fn get_image_ref_info(&self) -> super::ImageRefInfo {
        let requires_unmultiply = self.alpha_mode.is_some_and(|v| match v {
            AlphaMode::PremultipliedElectrical => true,
            AlphaMode::PremultipliedOptical => false,
            AlphaMode::Unpremmultiplied => false,
        });
        let requires_multiply = self.alpha_mode.is_some_and(|v| match v {
            AlphaMode::PremultipliedElectrical => true,
            AlphaMode::PremultipliedOptical => false,
            AlphaMode::Unpremmultiplied => true,
        });
        super::ImageRefInfo {
            requires_software_srgb: self.requires_manual_srgb_mapping,
            requires_unmultiplication: requires_unmultiply,
            requires_multiplication: requires_multiply,
        }
    }

    fn get_image(&self) -> Arc<Image> {
        self.image.clone()
    }

    fn get_image_view(&self) -> Arc<ImageView> {
        self.image_view.clone()
    }
}

pub struct FormatInfo {
    pub linear_format: Format,
    pub native_srgb_format: Option<Format>,
    pub swizzle: ComponentMapping,
    pub is_ycbcr: bool,
}
impl FormatInfo {
    const fn new(
        linear_format: Format,
        native_srgb_format: Option<Format>,
        swizzle: Option<ComponentMapping>,
        is_ycbcr: bool,
    ) -> Self {
        Self {
            linear_format,
            native_srgb_format,
            swizzle: {
                match swizzle {
                    Some(x) => x,
                    None => ComponentMapping {
                        r: ComponentSwizzle::Identity,
                        g: ComponentSwizzle::Identity,
                        b: ComponentSwizzle::Identity,
                        a: ComponentSwizzle::Identity,
                    },
                }
            },
            is_ycbcr,
        }
    }
}
macro_rules! fi {
    ($format:expr) => {
        FormatInfo::new($format, None, None, false)
    };
    ($format:expr, $srgb:expr) => {
        FormatInfo::new($format, Some($srgb), None, false)
    };
    (opaque $format:expr) => {
        FormatInfo::new($format, None, Some(CM_OPAQUE), false)
    };
    (opaque $format:expr, $srgb:expr) => {
        FormatInfo::new($format, Some($srgb), Some(CM_OPAQUE), false)
    };
    (ycbcr $format:expr) => {
        FormatInfo::new($format, None, None, true)
    };
    (ycbcr opaque $format:expr) => {
        FormatInfo::new($format, None, Some(CM_OPAQUE), true)
    };
    (ycbcr_swizzle $format:expr) => {
        FormatInfo::new(
            $format,
            None,
            Some(ComponentMapping {
                r: ComponentSwizzle::Blue,
                g: ComponentSwizzle::Identity,
                b: ComponentSwizzle::Red,
                a: ComponentSwizzle::Identity,
            }),
            true,
        )
    };
    (ycbcr_swizzle opaque $format:expr) => {
        FormatInfo::new(
            $format,
            None,
            Some(ComponentMapping {
                r: ComponentSwizzle::Blue,
                g: ComponentSwizzle::Identity,
                b: ComponentSwizzle::Red,
                a: ComponentSwizzle::One,
            }),
            true,
        )
    };
}
const CM_OPAQUE: ComponentMapping = ComponentMapping {
    r: ComponentSwizzle::Identity,
    g: ComponentSwizzle::Identity,
    b: ComponentSwizzle::Identity,
    a: ComponentSwizzle::One,
};
pub const fn map_fourcc(fourcc: DrmFourcc) -> Option<FormatInfo> {
    if (fourcc as u32 & DrmFourcc::Big_endian as u32) != 0 {
        return None;
    }
    let info = match fourcc {
        DrmFourcc::Abgr1555 => fi!(Format::A1B5G5R5_UNORM_PACK16),
        DrmFourcc::Abgr16161616f => fi!(Format::R16G16B16A16_SFLOAT),
        DrmFourcc::Abgr2101010 => fi!(Format::A2B10G10R10_UNORM_PACK32),
        DrmFourcc::Abgr4444 => fi!(Format::A4B4G4R4_UNORM_PACK16),
        DrmFourcc::Abgr8888 => fi!(Format::R8G8B8A8_UNORM, Format::R8G8B8A8_SRGB),
        DrmFourcc::Argb1555 => fi!(Format::A1R5G5B5_UNORM_PACK16),
        DrmFourcc::Argb16161616f => FormatInfo::new(
            Format::R16G16B16A16_SFLOAT,
            None,
            Some(ComponentMapping {
                r: ComponentSwizzle::Blue,
                g: ComponentSwizzle::Identity,
                b: ComponentSwizzle::Red,
                a: ComponentSwizzle::Identity,
            }),
            false,
        ),
        DrmFourcc::Argb2101010 => fi!(Format::A2R10G10B10_UNORM_PACK32),
        DrmFourcc::Argb4444 => fi!(Format::A4R4G4B4_UNORM_PACK16),
        DrmFourcc::Argb8888 => fi!(Format::B8G8R8A8_UNORM, Format::B8G8R8A8_SRGB),
        DrmFourcc::Axbxgxrx106106106106 => fi!(Format::R10X6G10X6B10X6A10X6_UNORM_4PACK16),
        DrmFourcc::Ayuv => FormatInfo::new(
            Format::R8G8B8A8_UNORM,
            None,
            Some(ComponentMapping {
                r: ComponentSwizzle::Identity,
                g: ComponentSwizzle::Blue,
                b: ComponentSwizzle::Green,
                a: ComponentSwizzle::Identity,
            }),
            true,
        ),
        DrmFourcc::Bgr565 => fi!(Format::B5G6R5_UNORM_PACK16),
        DrmFourcc::Bgr888 => fi!(Format::R8G8B8_UNORM, Format::R8G8B8_SRGB),
        DrmFourcc::Bgra4444 => fi!(Format::B4G4R4A4_UNORM_PACK16),
        DrmFourcc::Bgra5551 => fi!(Format::B5G5R5A1_UNORM_PACK16),
        DrmFourcc::Bgra8888 => FormatInfo::new(
            Format::B8G8R8A8_UNORM,
            Some(Format::B8G8R8A8_SRGB),
            Some(ComponentMapping {
                r: ComponentSwizzle::Alpha,
                g: ComponentSwizzle::Blue,
                b: ComponentSwizzle::Green,
                a: ComponentSwizzle::Red,
            }),
            false,
        ),
        DrmFourcc::Bgrx4444 => fi!(opaque Format::B4G4R4A4_UNORM_PACK16),
        DrmFourcc::Bgrx5551 => fi!(opaque Format::B5G5R5A1_UNORM_PACK16),
        DrmFourcc::Bgrx8888 => fi!(opaque Format::R8G8B8A8_UNORM, Format::R8G8B8A8_SRGB),
        DrmFourcc::Gr1616 => fi!(Format::R16G16_UNORM),
        DrmFourcc::Gr88 => fi!(Format::R8G8_UNORM, Format::R8G8_SRGB),
        DrmFourcc::R16 => fi!(Format::R16_UNORM),
        DrmFourcc::R8 => fi!(Format::R8_UNORM, Format::R8_SRGB),
        DrmFourcc::Rg1616 => FormatInfo::new(
            Format::R16G16_UNORM,
            None,
            Some(ComponentMapping {
                r: ComponentSwizzle::Green,
                g: ComponentSwizzle::Red,
                b: ComponentSwizzle::Identity,
                a: ComponentSwizzle::Identity,
            }),
            false,
        ),
        DrmFourcc::Rg88 => FormatInfo::new(
            Format::R8G8_UNORM,
            Some(Format::R8G8_SRGB),
            Some(ComponentMapping {
                r: ComponentSwizzle::Green,
                g: ComponentSwizzle::Red,
                b: ComponentSwizzle::Identity,
                a: ComponentSwizzle::Identity,
            }),
            false,
        ),
        DrmFourcc::Rgb565 => fi!(Format::R5G6B5_UNORM_PACK16),
        DrmFourcc::Rgb888 => fi!(Format::B8G8R8_UNORM, Format::B8G8R8_SRGB),
        DrmFourcc::Rgba4444 => fi!(Format::R4G4B4A4_UNORM_PACK16),
        DrmFourcc::Rgba5551 => fi!(Format::R5G5B5A1_UNORM_PACK16),
        DrmFourcc::Rgba8888 => FormatInfo::new(
            Format::R8G8B8A8_UNORM,
            Some(Format::R8G8B8A8_SRGB),
            Some(ComponentMapping {
                r: ComponentSwizzle::Alpha,
                g: ComponentSwizzle::Blue,
                b: ComponentSwizzle::Green,
                a: ComponentSwizzle::Red,
            }),
            false,
        ),
        DrmFourcc::Rgbx4444 => fi!(opaque Format::R4G4B4A4_UNORM_PACK16),
        DrmFourcc::Rgbx5551 => fi!(opaque Format::R5G5B5A1_UNORM_PACK16),
        DrmFourcc::Rgbx8888 => FormatInfo::new(
            Format::R8G8B8A8_UNORM,
            Some(Format::R8G8B8A8_SRGB),
            Some(ComponentMapping {
                r: ComponentSwizzle::Alpha,
                g: ComponentSwizzle::Blue,
                b: ComponentSwizzle::Green,
                a: ComponentSwizzle::One,
            }),
            false,
        ),
        DrmFourcc::Xbgr1555 => fi!(opaque Format::A1B5G5R5_UNORM_PACK16),
        DrmFourcc::Xbgr16161616f => fi!(opaque Format::R16G16B16A16_SFLOAT),
        DrmFourcc::Xbgr2101010 => fi!(opaque Format::A2B10G10R10_UNORM_PACK32),
        DrmFourcc::Xbgr4444 => fi!(opaque Format::A4B4G4R4_UNORM_PACK16),
        DrmFourcc::Xbgr8888 => fi!(opaque Format::R8G8B8A8_UNORM, Format::R8G8B8A8_SRGB),
        DrmFourcc::Xrgb1555 => fi!(opaque Format::A1R5G5B5_UNORM_PACK16),
        DrmFourcc::Xrgb16161616f => FormatInfo::new(
            Format::R16G16B16A16_SFLOAT,
            None,
            Some(ComponentMapping {
                r: ComponentSwizzle::Blue,
                g: ComponentSwizzle::Identity,
                b: ComponentSwizzle::Red,
                a: ComponentSwizzle::One,
            }),
            false,
        ),
        DrmFourcc::Xrgb2101010 => fi!(opaque Format::A2R10G10B10_UNORM_PACK32),
        DrmFourcc::Xrgb4444 => fi!(opaque Format::A4R4G4B4_UNORM_PACK16),
        DrmFourcc::Xrgb8888 => fi!(opaque Format::B8G8R8A8_UNORM, Format::B8G8R8A8_SRGB),
        DrmFourcc::Xvyu12_16161616 => {
            fi!(ycbcr_swizzle opaque Format::R12X4G12X4B12X4A12X4_UNORM_4PACK16)
        }
        DrmFourcc::Xvyu16161616 => fi!(ycbcr_swizzle opaque Format::R16G16B16A16_UNORM),
        DrmFourcc::Xvyu2101010 => fi!(ycbcr opaque Format::A2R10G10B10_UNORM_PACK32),
        DrmFourcc::Xyuv8888 => FormatInfo::new(
            Format::R8G8B8A8_UNORM,
            None,
            Some(ComponentMapping {
                r: ComponentSwizzle::Identity,
                g: ComponentSwizzle::Blue,
                b: ComponentSwizzle::Green,
                a: ComponentSwizzle::One,
            }),
            true,
        ),
        DrmFourcc::Y210 => fi!(ycbcr Format::G10X6B10X6G10X6R10X6_422_UNORM_4PACK16),
        DrmFourcc::Y212 => fi!(ycbcr Format::G12X4B12X4G12X4R12X4_422_UNORM_4PACK16),
        DrmFourcc::Y216 => fi!(ycbcr Format::G16B16G16R16_422_UNORM),
        DrmFourcc::Y410 => fi!(ycbcr Format::A2R10G10B10_UNORM_PACK32),
        DrmFourcc::Y412 => fi!(ycbcr_swizzle Format::R12X4G12X4B12X4A12X4_UNORM_4PACK16),
        DrmFourcc::Y416 => fi!(ycbcr_swizzle Format::R16G16B16A16_UNORM),
        DrmFourcc::Yuyv => fi!(ycbcr Format::G8B8G8R8_422_UNORM),
        DrmFourcc::Yvyu => fi!(ycbcr_swizzle Format::G8B8G8R8_422_UNORM),
        DrmFourcc::Uyvy => fi!(ycbcr Format::B8G8R8G8_422_UNORM),
        DrmFourcc::Vyuy => fi!(ycbcr_swizzle Format::B8G8R8G8_422_UNORM),
        DrmFourcc::Yuv420 => fi!(ycbcr Format::G8_B8_R8_3PLANE_420_UNORM),
        DrmFourcc::Yvu420 => fi!(ycbcr_swizzle Format::G8_B8_R8_3PLANE_420_UNORM),
        DrmFourcc::Yuv422 => fi!(ycbcr Format::G8_B8_R8_3PLANE_422_UNORM),
        DrmFourcc::Yvu422 => fi!(ycbcr_swizzle Format::G8_B8_R8_3PLANE_422_UNORM),
        DrmFourcc::Yuv444 => fi!(ycbcr Format::G8_B8_R8_3PLANE_444_UNORM),
        DrmFourcc::Yvu444 => fi!(ycbcr_swizzle Format::G8_B8_R8_3PLANE_444_UNORM),
        DrmFourcc::Nv12 => fi!(ycbcr Format::G8_B8R8_2PLANE_420_UNORM),
        DrmFourcc::Nv21 => fi!(ycbcr_swizzle Format::G8_B8R8_2PLANE_420_UNORM),
        DrmFourcc::Nv16 => fi!(ycbcr Format::G8_B8R8_2PLANE_422_UNORM),
        DrmFourcc::Nv61 => fi!(ycbcr_swizzle Format::G8_B8R8_2PLANE_422_UNORM),
        DrmFourcc::Nv24 => fi!(ycbcr Format::G8_B8R8_2PLANE_444_UNORM),
        DrmFourcc::Nv42 => fi!(ycbcr_swizzle Format::G8_B8R8_2PLANE_444_UNORM),
        DrmFourcc::P010 => fi!(ycbcr Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16),
        DrmFourcc::P012 => fi!(ycbcr Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16),
        DrmFourcc::P016 => fi!(ycbcr Format::G16_B16R16_2PLANE_420_UNORM),
        DrmFourcc::P210 => fi!(ycbcr Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16),
        DrmFourcc::Q401 => fi!(ycbcr Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16),
        DrmFourcc::Q410 => fi!(ycbcr_swizzle Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16),
        DrmFourcc::Vuy888 => FormatInfo::new(
            Format::R8G8B8_UNORM,
            None,
            Some(ComponentMapping {
                r: ComponentSwizzle::Blue,
                g: ComponentSwizzle::Red,
                b: ComponentSwizzle::Green,
                a: ComponentSwizzle::Identity,
            }),
            true,
        ),
        // vulkan has no 233 format
        DrmFourcc::Bgr233 => return None,
        DrmFourcc::Rgb332 => return None,
        // vulkan doesn't have multiplanar rgb at all
        DrmFourcc::Bgr565_a8 => return None,
        DrmFourcc::Bgr888_a8 => return None,
        DrmFourcc::Rgb565_a8 => return None,
        DrmFourcc::Rgb888_a8 => return None,
        DrmFourcc::Xrgb8888_a8 => return None,
        DrmFourcc::Bgrx8888_a8 => return None,
        DrmFourcc::Rgbx8888_a8 => return None,
        DrmFourcc::Xbgr8888_a8 => return None,
        // vulkan doesn't have a format with this bit order
        DrmFourcc::Bgra1010102 => return None,
        DrmFourcc::Bgrx1010102 => return None,
        DrmFourcc::Rgbx1010102 => return None,
        DrmFourcc::Rgba1010102 => return None,
        // this is apparently used as an index into a color map? nope, don't support that
        DrmFourcc::C8 => return None,
        // packed 2x2 subsampled ycbcr, vulkan does not support this
        DrmFourcc::X0l0 => return None,
        DrmFourcc::X0l2 => return None,
        DrmFourcc::Y0l0 => return None,
        DrmFourcc::Y0l2 => return None,
        // packs 4 10bit y values into 5 bytes, not something vulkan suppports
        DrmFourcc::Nv15 => return None,
        // uses 4x subsampling, vulkan only supports 2x
        DrmFourcc::Yuv410 => return None,
        DrmFourcc::Yuv411 => return None,
        DrmFourcc::Yvu410 => return None,
        DrmFourcc::Yvu411 => return None,
        // requires non-linear modifier, or in other words, its modifier dependent
        // and those should not be parsed by userspace
        DrmFourcc::Vuy101010 => return None,
        DrmFourcc::Yuv420_10bit => return None,
        DrmFourcc::Yuv420_8bit => return None,
        // is already ruled out at the top and would cause a bunch of issues anyway
        DrmFourcc::Big_endian => {
            unreachable!()
        }
    };
    Some(info)
}
