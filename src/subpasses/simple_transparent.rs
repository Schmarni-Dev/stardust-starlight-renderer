use std::{collections::HashSet, sync::Arc};

use vulkano::{
    device::DeviceOwnedVulkanObject,
    format::Format,
    image::SampleCount,
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineCreateFlags, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorBlendStateFlags,
                ColorComponents,
            },
            depth_stencil::{CompareOp, DepthState, DepthStencilState, DepthStencilStateFlags},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
            vertex_input::{Vertex as _, VertexDefinition as _},
            viewport::ViewportState,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::ShaderStages,
};

use crate::{
    Context,
    command_buffer::StarlightCommandBuffer,
    images::ImageManager,
    materials::{image_common, simple_pbr::PbrMaterial},
    mesh_manager::VertexData,
    object_manager::RenderInvocationData,
    render::{RenderContext, SubPass, ViewData},
    subpasses::wboit,
};

pub struct SimpleTransparent {
    pipeline: Arc<GraphicsPipeline>,
    pipeline_layout: Arc<PipelineLayout>,
}
impl SubPass<PbrMaterial> for SimpleTransparent {
    fn record(
        &mut self,
        data: &RenderInvocationData<PbrMaterial>,
        images: &ImageManager,
        ctx: &RenderContext,
        cmds: &mut StarlightCommandBuffer,
    ) {
        // TODO: have a graphics pipeline ReadOp
        unsafe { cmds.inner().bind_pipeline_graphics(&self.pipeline) }.unwrap();
        images
            .bind(cmds, PipelineBindPoint::Graphics, &self.pipeline_layout)
            .unwrap();

        tokio::runtime::Handle::current().block_on(data.bind_buffers(cmds));
        let metadata_ptr = data.metadata_bda(cmds);
        let res = ctx.wboit_accum.image().extent();
        wboit::push_descriptors(cmds, ctx, &self.pipeline_layout);
        unsafe {
            cmds.inner()
                .push_constants(
                    &self.pipeline_layout,
                    0,
                    &crate::pipeline_pieces::mesh_vertex_shader::__pc {
                        cam_ptr: ctx.views_bda,
                        metadata_ptr,
                        resolution: [res[0], res[1]],
                    },
                )
                .unwrap();
        }
        data.draw(cmds);
    }
}

impl SimpleTransparent {
    // TODO: error handling
    pub fn new(
        ctx: &Context,
        opaque_format: Format,
        depth_format: Format,
        views: &[ViewData],
    ) -> Self {
        let mut view_mask = 0;
        for i in 0..views.len() {
            view_mask |= 1 << i;
        }
        let vs = crate::pipeline_pieces::mesh_vertex_shader::load(ctx.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = crate::materials::simple_pbr::fs_wboit::load(ctx.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let vertex_input_state = VertexData::per_vertex().definition(&vs).unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(ctx.device.clone(), {
            let mut v = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
            v.set_layouts = vec![
                image_common::get_descriptor_set_layout_info(ctx, ShaderStages::FRAGMENT),
                wboit::get_descriptor_set_layout_info(ShaderStages::FRAGMENT),
            ];
            v.into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap()
        })
        .unwrap();

        // TODO: cache and use other thingy, yk, the other thing, pipeline library or whatever
        let pipeline = unsafe {
            GraphicsPipeline::new_unchecked(
                ctx.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    flags: PipelineCreateFlags::empty(),
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: Default::default(),
                        scissors: Default::default(),
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::None,
                        front_face: FrontFace::CounterClockwise,
                        depth_bias: None,
                        conservative: None,
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState {
                        // TODO: properly handle MSAA state
                        rasterization_samples: SampleCount::Sample1,
                        alpha_to_coverage_enable: false,
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        flags: DepthStencilStateFlags::empty(),
                        depth: Some(DepthState {
                            write_enable: false,
                            compare_op: CompareOp::Greater,
                        }),
                        depth_bounds: None,
                        // TODO: handle stencil for OpenXR view mesh
                        stencil: None,
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState {
                        flags: ColorBlendStateFlags::empty(),
                        attachments: vec![
                            ColorBlendAttachmentState::default(),
                            ColorBlendAttachmentState {
                                blend: Some(AttachmentBlend::additive()),
                                color_write_mask: ColorComponents::all(),
                                color_write_enable: true,
                            },
                            ColorBlendAttachmentState {
                                blend: Some(AttachmentBlend::additive()),
                                color_write_mask: ColorComponents::all(),
                                color_write_enable: true,
                            },
                            ColorBlendAttachmentState::default(),
                        ],
                        ..Default::default()
                    }),
                    dynamic_state: HashSet::from_iter([
                        DynamicState::ViewportWithCount,
                        DynamicState::ScissorWithCount,
                    ]),
                    subpass: Some(PipelineSubpassType::BeginRendering(
                        PipelineRenderingCreateInfo {
                            // TODO: properly handle mutliview
                            view_mask,
                            color_attachment_formats: vec![
                                Some(opaque_format),
                                Some(Format::R16G16B16A16_SFLOAT),
                                Some(Format::R8_UNORM),
                                Some(opaque_format),
                            ],
                            depth_attachment_format: Some(depth_format),
                            // TODO: handle stencil
                            stencil_attachment_format: None,
                            ..Default::default()
                        },
                    )),
                    fragment_shading_rate_state: None,
                    ..GraphicsPipelineCreateInfo::layout(layout.clone())
                },
            )
            .unwrap()
        };
        pipeline
            .set_debug_utils_object_name(Some("Simple Transparent Pipeline"))
            .unwrap();
        Self {
            pipeline,
            pipeline_layout: layout,
        }
    }
}
