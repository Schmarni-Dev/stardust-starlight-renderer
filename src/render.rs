use std::sync::Arc;

use glam::{Mat4, UVec2, UVec3};
use tokio::runtime::RuntimeFlavor;
use vulkano::{
    DeviceAddress,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        RecordingCommandBuffer, RenderingAttachmentInfo, RenderingInfo, SubpassContents,
    },
    format::{ClearValue, Format},
    image::{Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage, SampleCount},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::graphics::viewport::{Scissor, Viewport},
    render_pass::AttachmentLoadOp,
    sync::{AccessFlags, DependencyFlags, DependencyInfo, ImageMemoryBarrier, PipelineStages},
};

use crate::{
    Context,
    command_buffer::StarlightCommandBuffer,
    images::{ImageManager, transient_attachment::TransientAttachment},
    materials::MaterialHandle,
    object_manager::RenderInvocationData,
    resources::image::ImageWriteOp,
    subpasses::fullscreen_resolve::FullscreenResolve,
};

pub trait SubPass<Material: MaterialHandle> {
    fn record(
        &mut self,
        data: &RenderInvocationData<Material>,
        images: &ImageManager,
        ctx: &RenderContext,
        cmds: &mut StarlightCommandBuffer,
    );
}
pub struct RenderContext {
    pub opaque: TransientAttachment,
    pub depth: TransientAttachment,
    pub wboit: TransientAttachment,
    pub wboit_revealage: TransientAttachment,
    pub previous_wboit_revealage: Arc<Image>,
    // pub previous_wboit_revealage: Arc<Image>,
    pub views_bda: DeviceAddress,
}

#[derive(Debug, BufferContents, Clone, Copy)]
#[repr(C)]
pub struct ViewData {
    pub world_to_clip: Mat4,
}
pub struct RenderInstance {
    resolution: UVec3,
    previous_wboit_revealage: Arc<Image>,
    views: Subbuffer<[ViewData]>,
    multiview_mask: u32,
    depth_format: Format,
    fullscreen_pass: FullscreenResolve,
    render_ctx: RenderContext,
}

pub enum Texture {
    Image(Arc<Image>),
    Dmatex(Arc<Image>),
}

impl RenderInstance {
    pub fn new(
        ctx: &Context,
        resolution: UVec2,
        views: &[ViewData],
        color_format: Format,
        depth_format: Format,
        msaa: SampleCount,
    ) -> Self {
        let resolution = resolution.extend(views.len() as u32);
        // TODO: msaa, needs custom_resolve i think, which needs custom bindings because even ash
        // doesn't have it yet (in release at least)
        let samples = msaa;
        let views_buffer = Buffer::new_slice(
            ctx.alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            views.len() as u64,
        )
        .unwrap();
        views_buffer.write().unwrap().copy_from_slice(views);

        let opaque = TransientAttachment::new(
            ctx,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: color_format,
                extent: [resolution.x, resolution.y, 1],
                array_layers: resolution.z,
                samples,
                usage: ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            "Transient Opaque Target",
        )
        .unwrap();
        let depth = TransientAttachment::new(
            ctx,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: depth_format,
                extent: [resolution.x, resolution.y, 1],
                array_layers: resolution.z,
                samples,
                // TODO: add support for resolving the depth buffer to vram
                usage: ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            "Transient Depth Target",
        )
        .unwrap();
        let wboit = TransientAttachment::new(
            ctx,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_SFLOAT,
                extent: [resolution.x, resolution.y, 1],
                array_layers: resolution.z,
                samples,
                usage: ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            "Transient WBOIT Target",
        )
        .unwrap();
        let revealage = TransientAttachment::new(
            ctx,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8_UNORM,
                extent: [resolution.x, resolution.y, 1],
                array_layers: resolution.z,
                samples,
                usage: ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            "Transient WBOIT Revealage Target",
        )
        .unwrap();
        let previous_wboit_revealage = Image::new(
            ctx.alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8_UNORM,
                extent: [resolution.x, resolution.y, 1],
                array_layers: views_buffer.len() as u32,
                samples,
                usage: ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            Default::default(),
        )
        .unwrap();
        let render_ctx = RenderContext {
            opaque,
            depth,
            wboit,
            wboit_revealage: revealage,
            previous_wboit_revealage: previous_wboit_revealage.clone(),
            views_bda: views_buffer.device_address().unwrap().get(),
        };
        let mut multiview_mask = 0;
        for i in 0..views.len() {
            multiview_mask |= 1 << i;
        }
        Self {
            resolution,
            previous_wboit_revealage,
            views: views_buffer,
            depth_format,
            fullscreen_pass: FullscreenResolve::new(ctx, color_format, views),
            render_ctx,
            multiview_mask,
        }
    }
    pub fn update_view(&self, view_index: usize, data: ViewData) {
        self.views.write().unwrap()[view_index] = data;
    }
    pub fn render(
        &mut self,
        color_target: Box<dyn ImageWriteOp>,
        render_callback: impl FnMut(&RenderContext, StarlightCommandBuffer) -> StarlightCommandBuffer,
        clear_color: [f32; 4],
        cmds: StarlightCommandBuffer,
    ) -> StarlightCommandBuffer {
        // self.render_inner(ctx, color_target, render_callback, cmds)
        if let Ok(rt) = tokio::runtime::Handle::try_current()
            && rt.runtime_flavor() == RuntimeFlavor::CurrentThread
        {
            self.render_inner(color_target, render_callback, clear_color, cmds)
        } else {
            tokio::task::block_in_place(|| {
                self.render_inner(color_target, render_callback, clear_color, cmds)
            })
        }
    }
    fn render_inner(
        &mut self,
        color_target: Box<dyn ImageWriteOp>,
        mut render_callback: impl FnMut(
            &RenderContext,
            StarlightCommandBuffer,
        ) -> StarlightCommandBuffer,
        clear_color: [f32; 4],
        mut cmds: StarlightCommandBuffer,
    ) -> StarlightCommandBuffer {
        unsafe {
            let RenderContext {
                opaque,
                depth,
                wboit,
                wboit_revealage,
                previous_wboit_revealage: _,
                views_bda: _,
            } = &self.render_ctx;
            cmds.inner()
                .pipeline_barrier(&DependencyInfo {
                    dependency_flags: DependencyFlags::BY_REGION,
                    image_memory_barriers: [
                        opaque.setup_barrier(
                            PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                            AccessFlags::COLOR_ATTACHMENT_WRITE,
                        ),
                        wboit.setup_barrier(
                            PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                            AccessFlags::COLOR_ATTACHMENT_WRITE,
                        ),
                        wboit_revealage.setup_barrier(
                            PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                            AccessFlags::COLOR_ATTACHMENT_WRITE,
                        ),
                        depth.setup_barrier(
                            PipelineStages::EARLY_FRAGMENT_TESTS
                                | PipelineStages::LATE_FRAGMENT_TESTS,
                            AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        ),
                        ImageMemoryBarrier {
                            src_stages: PipelineStages::empty(),
                            src_access: AccessFlags::empty(),
                            dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                            dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                            // TODO: don't assume this
                            old_layout: color_target.current_layout(),
                            new_layout: ImageLayout::ColorAttachmentOptimal,
                            subresource_range: color_target.image().subresource_range(),
                            ..ImageMemoryBarrier::image(color_target.image().clone())
                        },
                    ]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })
                .unwrap()
                .begin_rendering(&RenderingInfo {
                    layer_count: 1,
                    // TODO: handle multiview
                    view_mask: self.multiview_mask,
                    color_attachments: vec![
                        Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Clear,
                            clear_value: Some(ClearValue::Float(clear_color)),
                            image_layout: ImageLayout::General,
                            ..RenderingAttachmentInfo::image_view(opaque.image_view())
                        }),
                        Some(RenderingAttachmentInfo {
                            image_layout: ImageLayout::General,
                            ..RenderingAttachmentInfo::image_view(wboit.image_view())
                        }),
                        Some(RenderingAttachmentInfo {
                            image_layout: ImageLayout::General,
                            ..RenderingAttachmentInfo::image_view(wboit_revealage.image_view())
                        }),
                        Some(RenderingAttachmentInfo {
                            image_layout: ImageLayout::ColorAttachmentOptimal,
                            ..RenderingAttachmentInfo::image_view(color_target.image_view().clone())
                        }),
                    ],
                    depth_attachment: Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Clear,
                        clear_value: Some(ClearValue::Depth(0.0)),
                        image_layout: ImageLayout::General,
                        ..RenderingAttachmentInfo::image_view(depth.image_view())
                    }),
                    // TODO: stencil attachment
                    stencil_attachment: None,
                    contents: SubpassContents::Inline,
                    render_area_extent: [self.resolution.x, self.resolution.y],
                    ..Default::default()
                })
                .unwrap()
                .set_viewport_with_count(&[Viewport {
                    offset: [0.0, color_target.image().extent()[1] as f32],
                    extent: [
                        color_target.image().extent()[0] as f32,
                        -(color_target.image().extent()[1] as f32),
                    ],
                    depth_range: 0.0..=1.0,
                }])
                .unwrap()
                .set_scissor_with_count(&[Scissor::default()])
                .unwrap();
        }
        let mut cmds = render_callback(&self.render_ctx, cmds);
        // local read barrier
        unsafe {
            cmds.inner()
                .pipeline_barrier(&DependencyInfo {
                    dependency_flags: DependencyFlags::BY_REGION,
                    image_memory_barriers: [
                        self.render_ctx.opaque.local_read_barrier(
                            PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                            AccessFlags::COLOR_ATTACHMENT_WRITE,
                        ),
                        self.render_ctx.depth.local_read_barrier(
                            PipelineStages::LATE_FRAGMENT_TESTS,
                            AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })
                .unwrap();
            self.fullscreen_pass.record(&self.render_ctx, cmds.inner());
            cmds.inner().end_rendering().unwrap();
            cmds.inner()
                .pipeline_barrier(&DependencyInfo {
                    dependency_flags: DependencyFlags::BY_REGION,
                    image_memory_barriers: [ImageMemoryBarrier {
                        src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                        src_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_stages: PipelineStages::empty(),
                        dst_access: AccessFlags::empty(),
                        old_layout: ImageLayout::ColorAttachmentOptimal,
                        new_layout: color_target.target_layout(),
                        subresource_range: color_target.image().subresource_range(),
                        ..ImageMemoryBarrier::image(color_target.image().clone())
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })
                .unwrap();
        }
        cmds.used_writes.push(color_target);
        cmds
    }
}
