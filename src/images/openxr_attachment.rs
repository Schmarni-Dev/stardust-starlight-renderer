use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use glam::UVec2;
use timeline_syncobj::timeline_syncobj::TimelineSyncObj;
use vulkano::{
    Handle as _, VulkanError, device::DeviceOwnedVulkanObject as _, format::Format, image::{
        Image, ImageLayout, ImageUsage,
        sys::RawImage,
        view::{ImageView, ImageViewCreateInfo},
    },
};

use crate::{
    Context,
    resources::{WriteOp, image::ImageWriteOp},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenXRAttachmentType {
    Color,
}

#[derive(Clone)]
pub struct OpenXRAttachment {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    timeline: Arc<TimelineSyncObj>,
    next_acquire: Arc<AtomicU64>,
    attachment_type: OpenXRAttachmentType,
}
impl OpenXRAttachment {
    pub fn from_raw(
        ctx: &Context,
        handle: u64,
        format: Format,
        res: UVec2,
        array_layers: u32,
        attachment_type: OpenXRAttachmentType,
    ) -> Self {
        let image = unsafe {
            Arc::new(
                RawImage::from_handle_borrowed(
                    ctx.device.clone(),
                    ash::vk::Image::from_raw(handle),
                    vulkano::image::ImageCreateInfo {
                        image_type: vulkano::image::ImageType::Dim2d,
                        format,
                        extent: [res.x, res.y, 1],
                        array_layers: array_layers,
                        mip_levels: 1,
                        samples: vulkano::image::SampleCount::Sample1,
                        usage: ImageUsage::COLOR_ATTACHMENT,
                        initial_layout: ImageLayout::ColorAttachmentOptimal,
                        ..Default::default()
                    },
                )
                .unwrap()
                .assume_bound(),
            )
        };
        let timeline = TimelineSyncObj::new(&ctx.drm_render_node).unwrap();
        let image_view =
            ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image)).unwrap();
        Self {
            image,
            next_acquire: Arc::new(AtomicU64::new(1)),
            timeline: Arc::new(timeline),
            attachment_type,
            image_view,
        }
    }
    pub fn image_set_debug_label(&self, name: Option<&str>) -> Result<(), VulkanError> {
        self.image.set_debug_utils_object_name(name)
    }
    pub fn image_view_set_debug_label(&self, name: Option<&str>) -> Result<(), VulkanError> {
        self.image_view.set_debug_utils_object_name(name)
    }
    /// # Safety
    /// this needs to be called after openxr_swapchain.acquire_image() and
    /// openxr_swapchain.wait_image() have returned
    pub unsafe fn acquire(&self) -> rustix::io::Result<AcquiredOpenXRAttachment> {
        unsafe {
            self.timeline
                .signal(self.next_acquire.load(Ordering::Relaxed))?
        };
        Ok(AcquiredOpenXRAttachment {
            image: self.image.clone(),
            timeline: self.timeline.clone(),
            next_acquire: self.next_acquire.clone(),
            attachment_type: self.attachment_type,
            image_view: self.image_view.clone(),
        })
    }
}
pub struct AcquiredOpenXRAttachment {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    timeline: Arc<TimelineSyncObj>,
    next_acquire: Arc<AtomicU64>,
    attachment_type: OpenXRAttachmentType,
}
impl AcquiredOpenXRAttachment {
    pub fn write(&self) -> Box<OpenXRAttachmentWriteOp> {
        let acquire = self.next_acquire.fetch_add(1, Ordering::Relaxed);
        let release = acquire + 1;
        Box::new(OpenXRAttachmentWriteOp {
            image: self.image.clone(),
            image_view: self.image_view.clone(),
            timeline: self.timeline.clone(),
            attachment_type: self.attachment_type,
            acquire,
            release,
        })
    }
}
pub struct OpenXRAttachmentWriteOp {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    timeline: Arc<TimelineSyncObj>,
    attachment_type: OpenXRAttachmentType,
    acquire: u64,
    release: u64,
}
impl ImageWriteOp for OpenXRAttachmentWriteOp {
    fn image(&self) -> &Arc<Image> {
        &self.image
    }

    fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }

    fn current_layout(&self) -> ImageLayout {
        match self.attachment_type {
            OpenXRAttachmentType::Color => ImageLayout::ColorAttachmentOptimal,
        }
    }

    fn target_layout(&self) -> ImageLayout {
        self.current_layout()
    }
}
impl WriteOp for OpenXRAttachmentWriteOp {
    fn timeline(&self) -> &TimelineSyncObj {
        &self.timeline
    }

    fn acquire_point(&self) -> u64 {
        self.acquire
    }

    fn release_point(&self) -> u64 {
        self.release
    }
}
