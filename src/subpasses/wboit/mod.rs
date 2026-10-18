use std::sync::Arc;

use timeline_syncobj::timeline_syncobj::TimelineSyncObj;
use vulkano::{
    buffer::Subbuffer,
    descriptor_set::{WriteDescriptorSet, layout::DescriptorSetLayoutCreateFlags},
    image::{Image, ImageLayout, view::ImageView},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::{
        AccessFlags, BufferMemoryBarrier, DependencyFlags, DependencyInfo, ImageMemoryBarrier,
        PipelineStages,
    },
};

use crate::{
    Context, command_buffer::StarlightCommandBuffer, render::RenderContext, resources::ReadOp,
};

pub struct WboitCdfBuildCompute {
    pipeline: Arc<ComputePipeline>,
    timeline: Arc<TimelineSyncObj>,
}
impl WboitCdfBuildCompute {
    pub fn new(ctx: &Context) -> Self {
        let shader = compute::load(ctx.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(shader);
        let pipeline_layout = PipelineLayout::new(ctx.device.clone(), {
            let mut v = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage]);
            v.set_layouts
                .iter_mut()
                .for_each(|v| v.flags |= DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR);
            v.into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap()
        })
        .unwrap();
        let pipeline = ComputePipeline::new(
            ctx.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout),
        )
        .unwrap();
        let timeline = Arc::new(TimelineSyncObj::new(&ctx.drm_render_node).unwrap());
        unsafe { timeline.signal(1).unwrap() };
        Self { pipeline, timeline }
    }
    pub fn record(&self, cmds: &mut StarlightCommandBuffer, ctx: &RenderContext) {
        // TODO: cache this somewhere lol
        let cdf_view = ImageView::new_default(ctx.cdf.clone()).unwrap();
        unsafe {
            cmds.inner()
                .pipeline_barrier(&DependencyInfo {
                    dependency_flags: DependencyFlags::BY_REGION,
                    buffer_memory_barriers: [BufferMemoryBarrier {
                        src_stages: PipelineStages::FRAGMENT_SHADER,
                        src_access: AccessFlags::SHADER_STORAGE_WRITE,
                        dst_stages: PipelineStages::COMPUTE_SHADER,
                        dst_access: AccessFlags::SHADER_STORAGE_READ,
                        range: ctx.histogram.offset()
                            ..(ctx.histogram.offset() + ctx.histogram.size()),
                        ..BufferMemoryBarrier::buffer(ctx.histogram.buffer().clone())
                    }]
                    .into_iter()
                    .collect(),
                    image_memory_barriers: [ImageMemoryBarrier {
                        src_stages: PipelineStages::FRAGMENT_SHADER,
                        src_access: AccessFlags::SHADER_SAMPLED_READ,
                        dst_stages: PipelineStages::COMPUTE_SHADER,
                        dst_access: AccessFlags::SHADER_STORAGE_WRITE,
                        old_layout: ImageLayout::ShaderReadOnlyOptimal,
                        new_layout: ImageLayout::General,
                        subresource_range: ctx.cdf.subresource_range(),
                        ..ImageMemoryBarrier::image(ctx.cdf.clone())
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })
                .unwrap();
            let [size_x, size_y, _] = ctx.cdf.extent();
            cmds.inner()
                .bind_pipeline_compute(&self.pipeline)
                .unwrap()
                .push_descriptor_set(
                    PipelineBindPoint::Compute,
                    self.pipeline.layout(),
                    0,
                    &[
                        WriteDescriptorSet::buffer(0, ctx.histogram.clone()),
                        WriteDescriptorSet::image_view(1, cdf_view.clone()),
                    ],
                )
                .unwrap()
                .dispatch([size_x.div_ceil(8), size_y.div_ceil(8), 1])
                .unwrap()
                .pipeline_barrier(&DependencyInfo {
                    dependency_flags: DependencyFlags::BY_REGION,
                    buffer_memory_barriers: [BufferMemoryBarrier {
                        src_stages: PipelineStages::COMPUTE_SHADER,
                        src_access: AccessFlags::SHADER_STORAGE_READ,
                        dst_stages: PipelineStages::FRAGMENT_SHADER,
                        dst_access: AccessFlags::SHADER_STORAGE_WRITE,
                        range: ctx.histogram.offset()
                            ..(ctx.histogram.offset() + ctx.histogram.size()),
                        ..BufferMemoryBarrier::buffer(ctx.histogram.buffer().clone())
                    }]
                    .into_iter()
                    .collect(),
                    image_memory_barriers: [ImageMemoryBarrier {
                        src_stages: PipelineStages::COMPUTE_SHADER,
                        src_access: AccessFlags::SHADER_STORAGE_WRITE,
                        dst_stages: PipelineStages::FRAGMENT_SHADER,
                        dst_access: AccessFlags::SHADER_SAMPLED_READ,
                        old_layout: ImageLayout::General,
                        new_layout: ImageLayout::ShaderReadOnlyOptimal,
                        subresource_range: ctx.cdf.subresource_range(),
                        ..ImageMemoryBarrier::image(ctx.cdf.clone())
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })
                .unwrap();
        }
        cmds.used_reads.push(Arc::new(DummyReadOp {
            _cdf: ctx.cdf.clone(),
            _cdf_view: cdf_view,
            _histogram: ctx.histogram.clone(),
            timeline: self.timeline.clone(),
        }));
    }
}

// TODO: handle this correctly :3
struct DummyReadOp {
    _cdf: Arc<Image>,
    _cdf_view: Arc<ImageView>,
    _histogram: Subbuffer<[f32]>,
    timeline: Arc<TimelineSyncObj>,
}
impl ReadOp for DummyReadOp {
    fn timeline(&self) -> &TimelineSyncObj {
        &self.timeline
    }

    fn acquire_point(&self) -> u64 {
        1
    }
}

mod compute {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/pipeline_pieces/wboit_build_cdf.comp",
        include: ["src/materials/", "src/pipeline_pieces/"],
    }
}
