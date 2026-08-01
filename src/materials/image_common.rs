use vulkano::{
    descriptor_set::layout::{
        DescriptorBindingFlags, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags,
        DescriptorSetLayoutCreateInfo, DescriptorType,
    },
    pipeline::layout::{
        PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateFlags, PushConstantRange,
    },
    shader::ShaderStages,
};

use crate::Context;
pub fn get_descriptor_set_layout_info(
    ctx: &Context,
    stages: ShaderStages,
) -> DescriptorSetLayoutCreateInfo {
    if ctx
        .phys_dev
        .properties()
        .max_per_stage_descriptor_update_after_bind_sampled_images
        .is_none()
    {
        tracing::error!("no update_after_bind sampled images!");
    }
    let image_max = ctx
        .phys_dev
        .properties()
        .max_per_stage_descriptor_update_after_bind_sampled_images
        .unwrap_or(0);
    if ctx
        .phys_dev
        .properties()
        .max_per_stage_descriptor_update_after_bind_samplers
        .is_none()
    {
        tracing::error!("no update_after_bind samplers!");
    }
    // TODO: should this always use max? could be memory waste as there probably are a low number of
    // samplers anyway (double digits at most)
    let sampler_max = ctx
        .phys_dev
        .properties()
        .max_per_stage_descriptor_update_after_bind_samplers
        .unwrap_or(0);
    let image_max = image_max.min(128 * 1024);
    let sampler_max = sampler_max.min(256);
    DescriptorSetLayoutCreateInfo {
        flags: DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
        bindings: [
            (
                0,
                DescriptorSetLayoutBinding {
                    binding_flags: DescriptorBindingFlags::UPDATE_AFTER_BIND
                        | DescriptorBindingFlags::PARTIALLY_BOUND,
                    descriptor_count: sampler_max,
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
                },
            ),
            (
                1,
                DescriptorSetLayoutBinding {
                    binding_flags: DescriptorBindingFlags::UPDATE_AFTER_BIND
                        | DescriptorBindingFlags::PARTIALLY_BOUND,
                    descriptor_count: image_max,
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
                },
            ),
        ]
        .into_iter()
        .collect(),
        ..DescriptorSetLayoutCreateInfo::default()
    }
}

pub fn get_pipeline_layout_info(
    push_constant_ranges: Vec<PushConstantRange>,
    stages: ShaderStages,
    ctx: &Context,
) -> PipelineDescriptorSetLayoutCreateInfo {
    PipelineDescriptorSetLayoutCreateInfo {
        flags: PipelineLayoutCreateFlags::empty(),
        set_layouts: vec![get_descriptor_set_layout_info(ctx, stages)],
        push_constant_ranges,
    }
}
