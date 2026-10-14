use std::{fs::File, sync::Arc};

use ash::vk;
use timeline_syncobj::{render_node::DrmRenderNode, timeline_syncobj::TimelineSyncObj};
use tokio::task::JoinSet;
use vulkano::{
    VulkanObject as _,
    device::QueueGuard,
    sync::semaphore::{
        ExternalSemaphoreHandleType, ExternalSemaphoreHandleTypes, ImportSemaphoreFdInfo,
        Semaphore, SemaphoreCreateInfo, SemaphoreImportFlags,
    },
};

use crate::command_buffer::StarlightCommandBuffer;

#[derive(Clone)]
pub struct StarlightQueue {
    warn_on_gpu_wait: bool,
    drm_render_node: DrmRenderNode,
    inner: Arc<vulkano::device::Queue>,
}
impl StarlightQueue {
    pub fn new(queue: Arc<vulkano::device::Queue>, drm_render_node: DrmRenderNode) -> Self {
        Self {
            inner: queue,
            drm_render_node,
            warn_on_gpu_wait: true,
        }
    }
    pub fn with<'a, R>(&'a self, func: impl FnOnce(QueueGuard<'a>) -> R) -> R {
        self.inner.with(func)
    }
    pub fn queue_family_index(&self) -> u32 {
        self.inner.queue_family_index()
    }
    // TODO: add additional wait and signal things
    pub async fn submit(
        &self,
        mut command_buffer: StarlightCommandBuffer,
        additional_wait_points: Vec<SyncPoint>,
        additional_signal_points: Vec<SyncPoint>,
    ) {
        command_buffer
            .used_reads
            .dedup_by(|v1, v2| Arc::ptr_eq(v1, v2));
        let acquires = command_buffer
            .used_reads
            .iter()
            .map(|v| (v.timeline(), v.acquire_point()))
            .chain(
                command_buffer
                    .used_writes
                    .iter()
                    .map(|v| (v.timeline(), v.acquire_point())),
            )
            .collect::<Vec<_>>();
        let objs = acquires.iter().map(|v| v.0).collect::<Vec<_>>();
        // TODO: bubble error up? don't unwrap
        let points = self.drm_render_node.query_signaled_points(&objs).unwrap();
        let mut wait_semaphores = vec![];
        if acquires.iter().enumerate().any(|(i, v)| v.1 > points[i]) {
            if self.warn_on_gpu_wait {
                tracing::warn!(
                    "not all resources where ready at submit time, doing gpu waits, may cause framedrops"
                );
            }
            let join_set = JoinSet::from_iter(
                acquires
                    .iter()
                    .enumerate()
                    .filter(|(i, v)| v.1 > points[*i])
                    // TODO: don't unwrap here, lol
                    .map(|(_, v)| v.0.wait_async(v.1).unwrap()),
            );
            join_set.join_all().await;
            for (_, v) in acquires
                .iter()
                .enumerate()
                .filter(|(i, v)| v.1 > points[*i])
            {
                // TODO: probably remove this unwrap, there shouldn't be an error for a missing point (if used
                // properly by potentially other processes), but we shouldn't unwrap on this
                let sync_file = v.0.export_sync_file_point(v.1).unwrap();
                let semaphore = Semaphore::from_pool(self.inner.device().clone()).unwrap();
                unsafe {
                    semaphore
                        .import_fd(ImportSemaphoreFdInfo {
                            flags: SemaphoreImportFlags::TEMPORARY,
                            file: Some(File::from(sync_file)),
                            ..ImportSemaphoreFdInfo::handle_type(
                                ExternalSemaphoreHandleType::SyncFd,
                            )
                        })
                        // TODO: don't unwrap here :3
                        .unwrap();
                }
                wait_semaphores.push(semaphore);
            }
        }
        let task_semaphore = Semaphore::new(
            self.inner.device().clone(),
            SemaphoreCreateInfo {
                export_handle_types: ExternalSemaphoreHandleTypes::SYNC_FD,
                ..Default::default()
            },
        )
        .unwrap();
        let write_semaphores = command_buffer
            .used_writes
            .iter()
            .map(|_| {
                Semaphore::new(
                    self.inner.device().clone(),
                    SemaphoreCreateInfo {
                        export_handle_types: ExternalSemaphoreHandleTypes::SYNC_FD,
                        ..Default::default()
                    },
                )
                // TODO: probably don't unwrap here
                .unwrap()
            })
            .collect::<Vec<_>>();

        let cmd_buffer = unsafe { command_buffer.inner.end() }.unwrap();
        tracing::trace!("about to submit work");
        // TODO: rework this to use vulkanos stuff after the next vulkano release
        self.inner.with(|_| {
            let queue = self.inner.handle();
            let fns = self.inner.device().fns();
            let ash_dev = ash::Device::from_parts_1_3(
                self.inner.device().handle(),
                fns.v1_0.clone(),
                fns.v1_1.clone(),
                fns.v1_2.clone(),
                fns.v1_3.clone(),
            );
            let signal_semaphores = write_semaphores
                .iter()
                .chain([&task_semaphore])
                .chain(additional_signal_points.iter().map(|v| {
                    let SyncPoint::BinarySemaphore { semaphore } = v;
                    semaphore.as_ref()
                }))
                .map(|v| v.handle())
                .collect::<Vec<_>>();
            let wait_semaphores = wait_semaphores
                .iter()
                .chain(additional_wait_points.iter().map(|v| {
                    let SyncPoint::BinarySemaphore { semaphore } = v;
                    semaphore.as_ref()
                }))
                .map(|v| v.handle())
                .collect::<Vec<_>>();
            // let wait_semaphores = vec![];
            // let signal_semaphores = vec![];
            unsafe {
                ash_dev
                    .queue_submit(
                        queue,
                        &[vk::SubmitInfo::default()
                            // TODO: optimize this maybe?
                            .wait_dst_stage_mask(&[vk::PipelineStageFlags::TOP_OF_PIPE])
                            .wait_semaphores(&wait_semaphores)
                            .command_buffers(&[cmd_buffer.handle()])
                            .signal_semaphores(&signal_semaphores)],
                        vk::Fence::null(),
                    )
                    .unwrap();
            }
        });
        tracing::trace!("work submitted");
        for (sema, op) in write_semaphores
            .iter()
            .zip(command_buffer.used_writes.iter())
        {
            // TODO: stop unwrapping everywhere
            let fd = unsafe { sema.export_fd(ExternalSemaphoreHandleType::SyncFd).unwrap() };
            op.timeline()
                .import_sync_file_point(fd, op.release_point())
                .unwrap();
        }
        let task_timeline = TimelineSyncObj::new(&self.drm_render_node).unwrap();
        let task_fd = unsafe {
            task_semaphore
                .export_fd(ExternalSemaphoreHandleType::SyncFd)
                .unwrap()
        };
        task_timeline.import_sync_file_point(task_fd, 1).unwrap();

        let reads = command_buffer.used_reads;
        let writes = command_buffer.used_writes;
        let objs = command_buffer.used_objects;
        tracing::debug!("submit done");
        tokio::spawn(async move {
            task_timeline.wait_async(1).unwrap().await;
            drop(task_semaphore);
            drop(wait_semaphores);
            drop(write_semaphores);
            drop(cmd_buffer);
            drop(additional_wait_points);
            drop(additional_signal_points);
            drop(reads);
            drop(writes);
            drop(objs);
        });
    }
}

pub enum SyncPoint {
    // TimelineSyncObj {
    //     timeline: Arc<TimelineSyncObj>,
    //     point: u64,
    // },
    BinarySemaphore { semaphore: Arc<Semaphore> },
}
