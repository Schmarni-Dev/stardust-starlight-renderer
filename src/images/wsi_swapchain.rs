use std::{
    fs::File,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use timeline_syncobj::timeline_syncobj::TimelineSyncObj;
use vulkano::{
    device::DeviceOwned as _,
    image::{Image, ImageLayout, view::ImageView},
    sync::semaphore::{
        ExternalSemaphoreHandleType, ImportSemaphoreFdInfo, Semaphore, SemaphoreImportFlags,
    },
};

use crate::{
    Context,
    resources::{WriteOp, image::ImageWriteOp},
};

#[derive(Clone)]
// TODO: is there a way to properly integrate the swapchain acquire?
pub struct WsiSwapchainImage {
    timeline: Arc<TimelineSyncObj>,
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    next_acquire: Arc<AtomicU64>,
    seen: Arc<AtomicBool>,
}
impl WsiSwapchainImage {
    pub fn new(ctx: &Context, image: Arc<Image>) -> Self {
        let timeline = Arc::new(TimelineSyncObj::new(&ctx.drm_render_node).unwrap());
        unsafe { timeline.signal(1).unwrap() };
        Self {
            timeline,
            image_view: ImageView::new_default(image.clone()).unwrap(),
            image,
            next_acquire: Arc::new(AtomicU64::new(1)),
            seen: AtomicBool::new(false).into(),
        }
    }
    pub fn image(&self) -> &Arc<Image> {
        &self.image
    }
    pub fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }
    pub fn write_op(&self) -> Box<WsiSwapchainImageWriteOp> {
        let acquire = self.next_acquire.fetch_add(1, Ordering::Relaxed);
        let release = acquire + 1;
        Box::new(WsiSwapchainImageWriteOp {
            image: self.image.clone(),
            image_view: self.image_view.clone(),
            timeline: self.timeline.clone(),
            acquire,
            release,
            seen: self.seen.clone(),
        })
    }
    pub fn get_completion_semaphore(&self) -> Option<Arc<Semaphore>> {
        let sync_file = self
            .timeline
            .export_sync_file_point(self.next_acquire.load(Ordering::Relaxed))
            .inspect_err(|err| tracing::error!("failed to get sync file: {err}"))
            .ok()?;
        let semaphore = Semaphore::from_pool(self.image.device().clone()).unwrap();
        unsafe {
            semaphore
                .import_fd(ImportSemaphoreFdInfo {
                    flags: SemaphoreImportFlags::TEMPORARY,
                    file: Some(File::from(sync_file)),
                    ..ImportSemaphoreFdInfo::handle_type(ExternalSemaphoreHandleType::SyncFd)
                })
                // TODO: don't unwrap here :3
                .unwrap();
        }
        Some(semaphore.into())
    }
}
pub struct WsiSwapchainImageWriteOp {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    timeline: Arc<TimelineSyncObj>,
    acquire: u64,
    release: u64,
    seen: Arc<AtomicBool>,
}
impl ImageWriteOp for WsiSwapchainImageWriteOp {
    fn image(&self) -> &Arc<Image> {
        &self.image
    }

    fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }

    fn current_layout(&self) -> ImageLayout {
        if self.seen.swap(true, Ordering::Relaxed) {
            ImageLayout::PresentSrc
        } else {
            ImageLayout::Undefined
        }
    }

    fn target_layout(&self) -> ImageLayout {
        ImageLayout::PresentSrc
    }
}
impl WriteOp for WsiSwapchainImageWriteOp {
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
