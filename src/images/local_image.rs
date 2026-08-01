use std::sync::Arc;

use vulkano::{
    device::DeviceOwnedVulkanObject as _,
    format::Format,
    image::{
        Image, ImageCreateFlags, ImageCreateInfo, ImageTiling, ImageUsage, SampleCount,
        sampler::{ComponentMapping, ComponentSwizzle},
        view::{ImageView, ImageViewCreateInfo},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::Sharing,
};

use crate::{
    Context,
    images::{AlphaMode, ImageRefSource, ImageResolution},
};

/// basic wrapper around a vulkan image
pub struct LocalImage {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    // timeline: Arc<TimelineSyncObj>,
    alpha_mode: Option<AlphaMode>,
    requires_manual_srgb_mapping: bool,
}
// TODO: support exporting as dmabuf
impl LocalImage {
    // TODO: support disjoint? maybe?
    /// if alpha_mode is None the image is treated as opaque
    /// if alpha_mode is Some(PremultipliedElectrical) format needs to be a non *_SRGB variant
    pub fn new(
        ctx: &Context,
        // TODO: ycbcr support?
        software_srgb: bool,
        format: Format,
        alpha_mode: Option<AlphaMode>,
        swizzle: Option<ComponentMapping>,
        res: ImageResolution,
        layers: u32,
        usage: ImageUsage,
        debug_name: Option<&str>,
        memory_type_filter: MemoryTypeFilter,
    ) -> Self {
        let image = Image::new(
            ctx.alloc.clone(),
            ImageCreateInfo {
                flags: ImageCreateFlags::empty(),
                image_type: res.image_type(),
                format,
                view_formats: vec![],
                extent: res.extents(),
                array_layers: layers,
                mip_levels: 1,
                samples: SampleCount::Sample1,
                tiling: ImageTiling::Optimal,
                usage,
                sharing: Sharing::Exclusive,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter,
                ..Default::default()
            },
        )
        .unwrap();
        let mut component_mapping = swizzle.unwrap_or(ComponentMapping::identity());
        if alpha_mode.is_none() {
            component_mapping.a = ComponentSwizzle::One;
        }
        let image_view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                component_mapping,
                ..ImageViewCreateInfo::from_image(&image)
            },
        )
        .unwrap();
        if let Some(debug_name) = debug_name {
            _ = image.set_debug_utils_object_name(Some(debug_name));
            _ = image_view.set_debug_utils_object_name(Some(&format!("{debug_name} View")));
        }
        Self {
            image,
            image_view,
            alpha_mode,
            requires_manual_srgb_mapping: software_srgb
                || alpha_mode.is_some_and(|v| v == AlphaMode::PremultipliedElectrical),
        }
    }
}

impl ImageRefSource for LocalImage {
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
