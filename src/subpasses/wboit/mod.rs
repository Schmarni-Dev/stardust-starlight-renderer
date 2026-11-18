use std::sync::Arc;

use vulkano::{
    descriptor_set::{
        WriteDescriptorSet,
        layout::{
            DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags,
            DescriptorSetLayoutCreateInfo, DescriptorType,
        },
    },
    image::ImageLayout,
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::ShaderStages,
    sync::{
        AccessFlags, BufferMemoryBarrier, DependencyFlags, DependencyInfo, ImageMemoryBarrier,
        PipelineStages,
    },
};

use crate::{Context, command_buffer::StarlightCommandBuffer, render::RenderContext};

pub struct WboitCdfBuildCompute {
    pipeline: Arc<ComputePipeline>,
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
        Self { pipeline }
    }
    pub fn record(&self, cmds: &mut StarlightCommandBuffer, ctx: &RenderContext) {
        unsafe {
            cmds.inner()
                .pipeline_barrier(&DependencyInfo {
                    dependency_flags: DependencyFlags::BY_REGION,
                    buffer_memory_barriers: [BufferMemoryBarrier {
                        src_stages: PipelineStages::FRAGMENT_SHADER,
                        src_access: AccessFlags::SHADER_STORAGE_WRITE,
                        dst_stages: PipelineStages::COMPUTE_SHADER,
                        dst_access: AccessFlags::SHADER_STORAGE_READ,
                        range: ctx.wboit.histogram.offset()
                            ..(ctx.wboit.histogram.offset() + ctx.wboit.histogram.size()),
                        ..BufferMemoryBarrier::buffer(ctx.wboit.histogram.buffer().clone())
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
                        subresource_range: ctx.wboit.cdf.subresource_range(),
                        ..ImageMemoryBarrier::image(ctx.wboit.cdf.clone())
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })
                .unwrap();
            let [size_x, size_y, _] = ctx.wboit.cdf.extent();
            cmds.inner()
                .bind_pipeline_compute(&self.pipeline)
                .unwrap()
                .push_descriptor_set(
                    PipelineBindPoint::Compute,
                    self.pipeline.layout(),
                    0,
                    &[
                        WriteDescriptorSet::buffer(0, ctx.wboit.histogram.clone()),
                        WriteDescriptorSet::image_view(1, ctx.wboit.cdf_view.clone()),
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
                        dst_stages: PipelineStages::CLEAR,
                        dst_access: AccessFlags::TRANSFER_WRITE,
                        range: ctx.wboit.histogram.offset()
                            ..(ctx.wboit.histogram.offset() + ctx.wboit.histogram.size()),
                        ..BufferMemoryBarrier::buffer(ctx.wboit.histogram.buffer().clone())
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
                        subresource_range: ctx.wboit.cdf.subresource_range(),
                        ..ImageMemoryBarrier::image(ctx.wboit.cdf.clone())
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })
                .unwrap()
                .fill_buffer(&ctx.wboit.histogram, 0)
                .unwrap()
                .pipeline_barrier(&DependencyInfo {
                    dependency_flags: DependencyFlags::BY_REGION,
                    buffer_memory_barriers: [BufferMemoryBarrier {
                        src_stages: PipelineStages::CLEAR,
                        src_access: AccessFlags::TRANSFER_WRITE,
                        dst_stages: PipelineStages::FRAGMENT_SHADER,
                        dst_access: AccessFlags::SHADER_STORAGE_WRITE,
                        range: ctx.wboit.histogram.offset()
                            ..(ctx.wboit.histogram.offset() + ctx.wboit.histogram.size()),
                        ..BufferMemoryBarrier::buffer(ctx.wboit.histogram.buffer().clone())
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })
                .unwrap();
        }
        cmds.used_reads.push(ctx.wboit.clone());
    }
}

mod compute {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/pipeline_pieces/wboit_build_cdf.comp",
        include: ["src/materials/", "src/pipeline_pieces/"],
    }
}

pub fn push_descriptors(
    cmds: &mut StarlightCommandBuffer,
    ctx: &RenderContext,
    layout: &PipelineLayout,
) {
    // cmds.used_reads.push(ctx.wboit.clone());
    unsafe {
        cmds.inner()
            .push_descriptor_set(
                PipelineBindPoint::Graphics,
                layout,
                1,
                &[
                    WriteDescriptorSet::sampler(0, ctx.wboit.cdf_sampler.clone()),
                    WriteDescriptorSet::image_view(1, ctx.wboit.cdf_view.clone()),
                    WriteDescriptorSet::buffer(2, ctx.wboit.histogram.clone()),
                ],
            )
            .unwrap();
    }
}

/// Needs to be at index 1
pub fn get_descriptor_set_layout_info(stages: ShaderStages) -> DescriptorSetLayoutCreateInfo {
    DescriptorSetLayoutCreateInfo {
        flags: DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR,
        bindings: [
            (
                0,
                DescriptorSetLayoutBinding {
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
                },
            ),
            (
                1,
                DescriptorSetLayoutBinding {
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
                },
            ),
            (
                2,
                DescriptorSetLayoutBinding {
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                },
            ),
        ]
        .into_iter()
        .collect(),
        ..DescriptorSetLayoutCreateInfo::default()
    }
}
