use std::{any::Any, sync::Arc};

use glam::{Mat4, UVec2, UVec3};
use tokio::runtime::RuntimeFlavor;
use vulkano::{
    DeviceAddress,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        RecordingCommandBuffer, RenderingAttachmentInfo, RenderingInfo, SubpassContents,
    },
    device::DeviceOwnedVulkanObject,
    format::{ClearValue, Format},
    image::{
        Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage, SampleCount,
        view::{ImageView, ImageViewCreateInfo},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::graphics::viewport::{Scissor, Viewport},
    render_pass::AttachmentLoadOp,
    sync::{AccessFlags, DependencyFlags, DependencyInfo, ImageMemoryBarrier, PipelineStages},
};

use crate::{
    Context, materials::MaterialHandle, object_manager::RenderInvocationData,
    subpasses::fullscreen_resolve::FullscreenResolve,
};

pub trait SubPass<Material: MaterialHandle> {
    fn record(
        &mut self,
        data: &RenderInvocationData<Material>,
        ctx: &RenderContext,
        cmd: &mut RecordingCommandBuffer,
    );
}
pub struct RenderContext {
    pub opaque: Arc<Image>,
    pub depth: Arc<Image>,
    pub wboit: Arc<Image>,
    pub wboit_revealage: Arc<Image>,
    pub previous_wboit_revealage: Arc<Image>,
    pub opaque_view: Arc<ImageView>,
    pub depth_view: Arc<ImageView>,
    pub wboit_view: Arc<ImageView>,
    pub wboit_revealage_view: Arc<ImageView>,
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
    depth_format: Format,
    fullscreen_pass: FullscreenResolve,
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
    ) -> Self {
        // TODO: msaa, needs custom_resolve i think, which needs custom bindings because even ash
        // doesn't have it yet (in release at least)
        let samples = SampleCount::Sample1;
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
        Self {
            resolution: resolution.extend(views.len() as u32),
            previous_wboit_revealage: Image::new(
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
            .unwrap(),
            views: views_buffer,
            depth_format,
            fullscreen_pass: FullscreenResolve::new(ctx, color_format),
        }
    }
    pub fn update_view(&self, view_index: usize, data: ViewData) {
        self.views.write().unwrap()[view_index] = data;
    }
    pub fn render(
        &mut self,
        ctx: &Context,
        color_target: Texture,
        render_callback: impl FnMut(
            RenderContext,
            RecordingCommandBuffer,
        ) -> (RenderContext, RecordingCommandBuffer),
        cmds: RecordingCommandBuffer,
    ) -> (RecordingCommandBuffer, Box<dyn Any>) {
        // self.render_inner(ctx, color_target, render_callback, cmds)
        if let Ok(rt) = tokio::runtime::Handle::try_current()
            && rt.runtime_flavor() == RuntimeFlavor::CurrentThread
        {
            self.render_inner(ctx, color_target, render_callback, cmds)
        } else {
            tokio::task::block_in_place(|| {
                self.render_inner(ctx, color_target, render_callback, cmds)
            })
        }
    }
    fn render_inner(
        &mut self,
        ctx: &Context,
        color_target: Texture,
        mut render_callback: impl FnMut(
            RenderContext,
            RecordingCommandBuffer,
        ) -> (RenderContext, RecordingCommandBuffer),
        mut cmds: RecordingCommandBuffer,
    ) -> (RecordingCommandBuffer, Box<dyn Any>) {
        let color_target_image = match &color_target {
            Texture::Image(image) => image,
            Texture::Dmatex(image) => image,
        };
        let format = color_target_image.format();
        // TODO: msaa, needs custom_resolve i think, which needs custom bindings because even ash
        // doesn't have it yet (in release at least)
        let samples = SampleCount::Sample1;
        // TODO: keep these image handles around instead of recreating them
        let opaque = Image::new(
            ctx.alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent: [self.resolution.x, self.resolution.y, 1],
                array_layers: self.resolution.z,
                samples,
                usage: ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            // does this work with transient? do i need to create a RawImage instead?
            Default::default(),
        )
        .unwrap();
        let depth = Image::new(
            ctx.alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: self.depth_format,
                extent: [self.resolution.x, self.resolution.y, 1],
                array_layers: self.resolution.z,
                samples,
                // TODO: add support for resolving the depth buffer to vram
                usage: ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            // does this work with transient? do i need to create a RawImage instead?
            Default::default(),
        )
        .unwrap();
        let wboit = Image::new(
            ctx.alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_SFLOAT,
                extent: [self.resolution.x, self.resolution.y, 1],
                array_layers: self.resolution.z,
                samples,
                usage: ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            // does this work with transient? do i need to create a RawImage instead?
            Default::default(),
        )
        .unwrap();
        let revealage = Image::new(
            ctx.alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8_UNORM,
                extent: [self.resolution.x, self.resolution.y, 1],
                array_layers: self.resolution.z,
                samples,
                usage: ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            // does this work with transient? do i need to create a RawImage instead?
            Default::default(),
        )
        .unwrap();
        let opaque_view =
            ImageView::new(opaque.clone(), ImageViewCreateInfo::from_image(&opaque)).unwrap();
        let depth_view =
            ImageView::new(depth.clone(), ImageViewCreateInfo::from_image(&depth)).unwrap();
        let wboit_view =
            ImageView::new(wboit.clone(), ImageViewCreateInfo::from_image(&wboit)).unwrap();
        let revealage_view = ImageView::new(
            revealage.clone(),
            ImageViewCreateInfo::from_image(&revealage),
        )
        .unwrap();
        let color_target_view = ImageView::new(
            color_target_image.clone(),
            ImageViewCreateInfo::from_image(&color_target_image),
        )
        .unwrap();
        opaque
            .set_debug_utils_object_name(Some("Transient Opaque Target"))
            .unwrap();
        depth
            .set_debug_utils_object_name(Some("Transient Depth Target"))
            .unwrap();
        wboit
            .set_debug_utils_object_name(Some("Transient WBOIT Target"))
            .unwrap();
        revealage
            .set_debug_utils_object_name(Some("Transient WBOIT Revealage Target"))
            .unwrap();
        opaque_view
            .set_debug_utils_object_name(Some("Transient Opaque Target View"))
            .unwrap();
        depth_view
            .set_debug_utils_object_name(Some("Transient Depth Target View"))
            .unwrap();
        wboit_view
            .set_debug_utils_object_name(Some("Transient WBOIT Target View"))
            .unwrap();
        revealage_view
            .set_debug_utils_object_name(Some("Transient WBOIT Revealage Target View"))
            .unwrap();
        unsafe {
            cmds.pipeline_barrier(&DependencyInfo {
                dependency_flags: DependencyFlags::BY_REGION,
                image_memory_barriers: [
                    ImageMemoryBarrier {
                        src_stages: PipelineStages::TOP_OF_PIPE,
                        src_access: AccessFlags::empty(),
                        dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                        dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                        // TODO: optimize?
                        old_layout: ImageLayout::Undefined,
                        // LocalRead would be ideal, but vulkano doesn't have a variant for that
                        new_layout: ImageLayout::General,
                        subresource_range: opaque.subresource_range(),
                        ..ImageMemoryBarrier::image(opaque.clone())
                    },
                    ImageMemoryBarrier {
                        src_stages: PipelineStages::TOP_OF_PIPE,
                        src_access: AccessFlags::empty(),
                        dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                        dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                        // TODO: optimize?
                        old_layout: ImageLayout::Undefined,
                        // LocalRead would be ideal, but vulkano doesn't have a variant for that
                        new_layout: ImageLayout::General,
                        subresource_range: wboit.subresource_range(),
                        ..ImageMemoryBarrier::image(wboit.clone())
                    },
                    ImageMemoryBarrier {
                        src_stages: PipelineStages::TOP_OF_PIPE,
                        src_access: AccessFlags::empty(),
                        dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                        dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                        // TODO: optimize?
                        old_layout: ImageLayout::Undefined,
                        // LocalRead would be ideal, but vulkano doesn't have a variant for that
                        new_layout: ImageLayout::General,
                        subresource_range: revealage.subresource_range(),
                        ..ImageMemoryBarrier::image(revealage.clone())
                    },
                    ImageMemoryBarrier {
                        src_stages: PipelineStages::TOP_OF_PIPE,
                        src_access: AccessFlags::empty(),
                        dst_stages: PipelineStages::EARLY_FRAGMENT_TESTS
                            | PipelineStages::LATE_FRAGMENT_TESTS,
                        dst_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        // TODO: optimize?
                        old_layout: ImageLayout::Undefined,
                        // LocalRead would be ideal, but vulkano doesn't have a variant for that
                        new_layout: ImageLayout::General,
                        subresource_range: depth.subresource_range(),
                        ..ImageMemoryBarrier::image(depth.clone())
                    },
                    ImageMemoryBarrier {
                        src_stages: PipelineStages::empty(),
                        src_access: AccessFlags::empty(),
                        dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                        dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                        // TODO: don't assume this
                        old_layout: ImageLayout::Undefined,
                        new_layout: ImageLayout::ColorAttachmentOptimal,
                        subresource_range: color_target_image.subresource_range(),
                        ..ImageMemoryBarrier::image(color_target_image.clone())
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
                view_mask: 0,
                color_attachments: vec![
                    Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Clear,
                        clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 1.0])),
                        image_layout: ImageLayout::General,
                        ..RenderingAttachmentInfo::image_view(opaque_view.clone())
                    }),
                    Some(RenderingAttachmentInfo {
                        image_layout: ImageLayout::General,
                        ..RenderingAttachmentInfo::image_view(wboit_view.clone())
                    }),
                    Some(RenderingAttachmentInfo {
                        image_layout: ImageLayout::General,
                        ..RenderingAttachmentInfo::image_view(revealage_view.clone())
                    }),
                    Some(RenderingAttachmentInfo {
                        image_layout: ImageLayout::ColorAttachmentOptimal,
                        ..RenderingAttachmentInfo::image_view(color_target_view.clone())
                    }),
                ],
                depth_attachment: Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    clear_value: Some(ClearValue::Depth(0.0)),
                    image_layout: ImageLayout::General,
                    ..RenderingAttachmentInfo::image_view(depth_view.clone())
                }),
                // TODO: stencil attachment
                stencil_attachment: None,
                contents: SubpassContents::Inline,
                render_area_extent: [self.resolution.x, self.resolution.y],
                ..Default::default()
            })
            .unwrap()
            .set_viewport_with_count(&[Viewport {
                offset: [0.0, color_target_image.extent()[1] as f32],
                extent: [
                    color_target_image.extent()[0] as f32,
                    -(color_target_image.extent()[1] as f32),
                ],
                depth_range: 0.0..=1.0,
            }])
            .unwrap()
            .set_scissor_with_count(&[Scissor::default()])
            .unwrap();
        }
        let (render_ctx, mut cmds) = render_callback(
            RenderContext {
                opaque,
                depth,
                wboit,
                wboit_revealage: revealage,
                previous_wboit_revealage: self.previous_wboit_revealage.clone(),
                views_bda: self.views.device_address().unwrap().get(),
                opaque_view,
                depth_view,
                wboit_view,
                wboit_revealage_view: revealage_view,
            },
            cmds,
        );
        unsafe {
            cmds.pipeline_barrier(&DependencyInfo {
                dependency_flags: DependencyFlags::BY_REGION,
                image_memory_barriers: [
                    ImageMemoryBarrier {
                        src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                        src_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_stages: PipelineStages::FRAGMENT_SHADER,
                        dst_access: AccessFlags::INPUT_ATTACHMENT_READ,
                        // LocalRead would be ideal, but vulkano doesn't have a variant for that
                        old_layout: ImageLayout::General,
                        new_layout: ImageLayout::General,
                        subresource_range: render_ctx.opaque.subresource_range(),
                        ..ImageMemoryBarrier::image(render_ctx.opaque.clone())
                    },
                    ImageMemoryBarrier {
                        src_stages: PipelineStages::LATE_FRAGMENT_TESTS,
                        src_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_stages: PipelineStages::FRAGMENT_SHADER,
                        dst_access: AccessFlags::INPUT_ATTACHMENT_READ,
                        old_layout: ImageLayout::General,
                        new_layout: ImageLayout::General,
                        subresource_range: render_ctx.depth.subresource_range(),
                        ..ImageMemoryBarrier::image(render_ctx.depth.clone())
                    },
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            })
            .unwrap();
            self.fullscreen_pass.record(&render_ctx, &mut cmds);
            cmds.end_rendering().unwrap();
        }
        // unsafe {
        //     cmds.copy_image(&CopyImageInfo::images(
        //         render_ctx.opaque,
        //         color_target_image.clone(),
        //     ))
        //     .unwrap();
        // }
        // TODO: properly handle WBOIT
        // unsafe {
        //     cmds.resolve_image(&ResolveImageInfo::images(
        //         render_ctx.opaque,
        //         color_target_image.clone(),
        //     ))
        //     .unwrap();
        // }
        (cmds, Box::new(render_ctx))
    }
}
