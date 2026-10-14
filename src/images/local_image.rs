use std::sync::{
    Arc, atomic::{AtomicBool, AtomicU64, Ordering},
};

use timeline_syncobj::timeline_syncobj::TimelineSyncObj;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    device::DeviceOwnedVulkanObject as _,
    format::Format,
    image::{
        Image, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageTiling, ImageUsage,
        SampleCount,
        sampler::{ComponentMapping, ComponentSwizzle},
        view::{ImageView, ImageViewCreateInfo},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::Sharing,
};

use crate::{
    Context,
    command_buffer::StarlightCommandBuffer,
    images::{AlphaMode, ImageManager, ImageRef, ImageRefSource, ImageResolution, SamplerRef},
    resources::{
        ReadOp, WriteOp,
        buffer::StarlightSubBuffer,
        image::{ImageReadOp, ImageWriteOp},
    },
};

/// basic wrapper around a vulkan image
pub struct LocalImage {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    timeline: Arc<TimelineSyncObj>,
    next_acquire: AtomicU64,
    alpha_mode: Option<AlphaMode>,
    requires_manual_srgb_mapping: bool,
    seen: Arc<AtomicBool>,
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
        let timeline = TimelineSyncObj::new(&ctx.drm_render_node).unwrap();
        unsafe { timeline.signal(1).unwrap() };
        Self {
            image,
            image_view,
            alpha_mode,
            requires_manual_srgb_mapping: software_srgb
                || alpha_mode.is_some_and(|v| v == AlphaMode::PremultipliedElectrical),
            timeline: Arc::new(timeline),
            next_acquire: AtomicU64::new(1),
            seen: AtomicBool::new(false).into(),
        }
    }
    pub fn new_write_op(&self) -> Box<LocalImageWriteOp> {
        let acquire = self.next_acquire.fetch_add(1, Ordering::Relaxed);
        let release = acquire + 1;
        Box::new(LocalImageWriteOp {
            image: self.image.clone(),
            timeline: self.timeline.clone(),
            acquire,
            release,
            image_view: self.image_view.clone(),
            seen: self.seen.clone()
        })
    }
    pub async fn new_read_op(
        &self,
        ctx: &Context,
        manager: &mut ImageManager,
        sampler: Arc<SamplerRef>,
    ) -> Arc<LocalImageReadOp> {
        let image_ref = manager.alloc(ctx, self, sampler).await;
        let acquire = self.next_acquire.fetch_add(1, Ordering::Relaxed);
        let release = acquire + 1;
        Arc::new(LocalImageReadOp {
            image: self.image.clone(),
            image_view: self.image_view.clone(),
            image_ref,
            timeline: self.timeline.clone(),
            acquire,
            release,
        })
    }
}
pub struct LocalImageReadOp {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    image_ref: Arc<ImageRef>,
    timeline: Arc<TimelineSyncObj>,
    acquire: u64,
    release: u64,
}
impl ImageReadOp for LocalImageReadOp {
    fn image(&self) -> &Arc<Image> {
        &self.image
    }

    fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }

    fn image_ref(&self) -> &ImageRef {
        &self.image_ref
    }
}
impl ReadOp for LocalImageReadOp {
    fn timeline(&self) -> &TimelineSyncObj {
        &self.timeline
    }

    fn acquire_point(&self) -> u64 {
        self.acquire
    }
}
impl Drop for LocalImageReadOp {
    fn drop(&mut self) {
        unsafe { self.timeline.signal(self.release).unwrap() };
    }
}
pub struct LocalImageWriteOp {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    timeline: Arc<TimelineSyncObj>,
    acquire: u64,
    release: u64,
    seen: Arc<AtomicBool>,
}
impl LocalImageWriteOp {
    /// upload data to the image using a temporary staging buffer and queue submit, when uploading often you probably
    /// want to have your own persistent staging buffer or try VK_EXT_host_image_copy
    pub async fn upload_tmp_staged<Data: BufferContents + Copy>(
        self: Box<Self>,
        ctx: &Context,
        data: &[Data],
    ) {
        let buf: Subbuffer<[Data]> = Buffer::new_slice(
            ctx.alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.len() as u64,
        )
        .unwrap();

        // TODO: make this a cpu sided write op
        buf.write().unwrap().copy_from_slice(data);
        let buf = StarlightSubBuffer::from_subbuffer(ctx, buf);
        let buf_read = buf.new_read_op();
        let queue = ctx
            .transfer_queue
            .clone()
            .unwrap_or_else(|| ctx.gfx_queue.clone());
        let mut cmds = StarlightCommandBuffer::new(ctx, queue.queue_family_index()).unwrap();
        cmds.buffer_to_image_copy(buf_read, self);
        queue.submit(cmds, vec![], vec![]).await;
    }
}
impl ImageWriteOp for LocalImageWriteOp {
    fn image(&self) -> &Arc<Image> {
        &self.image
    }

    fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }

    fn current_layout(&self) -> ImageLayout {
        // TODO: handle this better or enable the ext that makes general as good as specifc layouts
        if self.seen.swap(true, Ordering::Relaxed) {
            ImageLayout::General
        } else {
            ImageLayout::Undefined
        }
    }

    fn target_layout(&self) -> ImageLayout {
        // TODO: handle this better or enable the ext that makes general as good as specifc layouts
        ImageLayout::General
    }
}
impl WriteOp for LocalImageWriteOp {
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
