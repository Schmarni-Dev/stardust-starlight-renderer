use std::sync::Arc;

use glam::{Mat4, UVec2, UVec3, Vec2};
use timeline_syncobj::timeline_syncobj::TimelineSyncObj;
use tokio::runtime::RuntimeFlavor;
use vulkano::{
    DeviceAddress,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{RenderingAttachmentInfo, RenderingInfo, SubpassContents},
    format::{ClearValue, Format},
    image::{
        Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage, SampleCount,
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
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
    resources::{ReadOp, image::ImageWriteOp},
    subpasses::{fullscreen_resolve::FullscreenResolve, wboit::WboitCdfBuildCompute},
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
    pub wboit_accum: TransientAttachment,
    // accumulated optical depth/modified alpha
    pub wboit_revealage: TransientAttachment,
    pub wboit: Arc<WboitContext>,
    // pub previous_wboit_revealage: Arc<Image>,
    pub views_bda: DeviceAddress,
}
pub struct WboitContext {
    /// histogram data populated in the wboit accum step
    pub histogram: Subbuffer<[f32]>,
    pub cdf_sampler: Arc<Sampler>,
    /// 3D image sampled in the wboit accum step
    /// updated at the end of the render
    /// will contain the data from the last frame
    pub cdf: Arc<Image>,
    pub cdf_view: Arc<ImageView>,
    timeline: TimelineSyncObj,
}
// TODO: handle this correctly :3
impl ReadOp for WboitContext {
    fn timeline(&self) -> &TimelineSyncObj {
        &self.timeline
    }

    fn acquire_point(&self) -> u64 {
        1
    }
}

#[derive(Debug, BufferContents, Clone, Copy)]
#[repr(C)]
pub struct ViewData {
    pub world_to_clip: Mat4,
}
pub struct RenderInstance {
    resolution: UVec3,
    bins: u32,
    views: Subbuffer<[ViewData]>,
    multiview_mask: u32,
    depth_format: Format,
    fullscreen_pass: FullscreenResolve,
    wboit_compute: WboitCdfBuildCompute,
    render_ctx: RenderContext,
    wboit_setup: bool,
}

pub enum HistogramSize {
    Scaled(Vec2),
    Absoulte(UVec2),
}

// TODO: make wboit optional, ideally in a way that allows for different transparency methods
impl RenderInstance {
    pub fn new(
        ctx: &Context,
        resolution: UVec2,
        views: &[ViewData],
        color_format: Format,
        depth_format: Format,
        msaa: SampleCount,
        wboit_depth_bins: u32,
        wboit_historgram_size: HistogramSize,
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
        let histogram_res = match &wboit_historgram_size {
            HistogramSize::Scaled(scale) => UVec2::from_array(
                [resolution.x as f32 * scale.x, resolution.y as f32 * scale.y]
                    .map(|v| v.round() as u32),
            ),
            HistogramSize::Absoulte(res) => *res,
        };
        let create_histogram = |usage: ImageUsage| {
            let res = histogram_res;
            Image::new(
                ctx.alloc.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim3d,
                    format: Format::R32_SFLOAT,
                    extent: [res.x, res.y, wboit_depth_bins],
                    // array_layers: views_buffer.len() as u32,
                    samples,
                    usage,
                    ..Default::default()
                },
                Default::default(),
            )
            .unwrap()
        };
        let histogram = Buffer::new_slice(
            ctx.alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            Default::default(),
            dbg!((histogram_res.x * histogram_res.y * wboit_depth_bins) as u64),
        )
        .unwrap();
        let cdf = create_histogram(ImageUsage::STORAGE | ImageUsage::SAMPLED);
        let cdf_sampler = Sampler::new(
            ctx.device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear()
            },
        )
        .unwrap();
        let timeline = TimelineSyncObj::new(&ctx.drm_render_node).unwrap();
        unsafe { timeline.signal(10).unwrap() };
        let cdf_view = ImageView::new_default(cdf.clone()).unwrap();
        let render_ctx = RenderContext {
            opaque,
            depth,
            wboit_accum: wboit,
            wboit_revealage: revealage,
            wboit: Arc::new(WboitContext {
                histogram,
                cdf_sampler,
                cdf,
                cdf_view,
                timeline,
            }),
            views_bda: views_buffer.device_address().unwrap().get(),
        };
        let mut multiview_mask = 0;
        for i in 0..views.len() {
            multiview_mask |= 1 << i;
        }
        let wboit_compute = WboitCdfBuildCompute::new(ctx);
        Self {
            resolution,
            views: views_buffer,
            depth_format,
            fullscreen_pass: FullscreenResolve::new(ctx, color_format, views),
            render_ctx,
            multiview_mask,
            bins: wboit_depth_bins,
            wboit_setup: false,
            wboit_compute,
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
                wboit_accum,
                wboit_revealage,
                views_bda: _,
                wboit,
            } = &self.render_ctx;
            if !self.wboit_setup {
                self.wboit_setup = true;
                cmds.inner()
                    .pipeline_barrier(&DependencyInfo {
                        dependency_flags: DependencyFlags::BY_REGION,
                        image_memory_barriers: [ImageMemoryBarrier {
                            dst_stages: PipelineStages::FRAGMENT_SHADER,
                            dst_access: AccessFlags::SHADER_SAMPLED_READ,
                            old_layout: ImageLayout::Undefined,
                            new_layout: ImageLayout::ShaderReadOnlyOptimal,
                            subresource_range: wboit.cdf.subresource_range(),
                            ..ImageMemoryBarrier::image(wboit.cdf.clone())
                        }]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    })
                    .unwrap();
            }
            cmds.inner()
                .pipeline_barrier(&DependencyInfo {
                    dependency_flags: DependencyFlags::BY_REGION,
                    image_memory_barriers: [
                        opaque.setup_barrier(
                            PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                            AccessFlags::COLOR_ATTACHMENT_WRITE,
                        ),
                        wboit_accum.setup_barrier(
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
                .unwrap();
            self.wboit_compute.record(&mut cmds, &self.render_ctx);
            cmds.inner()
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
                            load_op: AttachmentLoadOp::Clear,
                            clear_value: Some(ClearValue::Float([0.0; 4])),
                            image_layout: ImageLayout::General,
                            ..RenderingAttachmentInfo::image_view(wboit_accum.image_view())
                        }),
                        Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Clear,
                            clear_value: Some(ClearValue::Float([0.0; 4])),
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
                        self.render_ctx.wboit_accum.local_read_barrier(
                            PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                            AccessFlags::COLOR_ATTACHMENT_WRITE,
                        ),
                        self.render_ctx.wboit_revealage.local_read_barrier(
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
