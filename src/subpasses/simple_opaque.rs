use std::{collections::HashSet, sync::Arc};

use vulkano::{
    DeviceAddress,
    command_buffer::RecordingCommandBuffer,
    device::DeviceOwnedVulkanObject,
    format::Format,
    image::{Image, SampleCount},
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineCreateFlags, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
                ColorBlendStateFlags, ColorComponents,
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
};

use crate::{
    Context,
    materials::simple_pbr::PbrMaterial,
    mesh_manager::VertexData,
    object_manager::RenderInvocationData,
    render::{RenderContext, SubPass},
};

pub struct SimpleOpaque {
    pipeline: Arc<GraphicsPipeline>,
    pipeline_layout: Arc<PipelineLayout>,
}
impl SubPass<PbrMaterial> for SimpleOpaque {
    fn record(
        &mut self,
        data: &RenderInvocationData<PbrMaterial>,
        ctx: &RenderContext,
        cmds: &mut RecordingCommandBuffer,
    ) {
        unsafe { cmds.bind_pipeline_graphics(&self.pipeline) }.unwrap();

        tokio::runtime::Handle::current().block_on(data.bind_buffers(cmds));
        unsafe {
            cmds.push_constants(
                &self.pipeline_layout,
                0,
                &crate::pipeline_pieces::mesh_vertex_shader::__pc {
                    cam_ptr: ctx.views_bda,
                    metadata_ptr: data.metadata_bda(),
                },
            )
            .unwrap();
        }
        data.draw(cmds);
    }
}

impl SimpleOpaque {
    // TODO: error handling
    pub fn new(ctx: &Context, opaque_format: Format, depth_format: Format) -> Self {
        let vs = crate::pipeline_pieces::mesh_vertex_shader::load(ctx.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = crate::materials::simple_pbr::fs::load(ctx.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let vertex_input_state = VertexData::per_vertex().definition(&vs).unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(ctx.device.clone(), {
            let v = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
            // v.set_layouts[1].flags |= DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;
            // v.set_layouts[1]
            //     .bindings
            //     .get_mut(&0)
            //     .unwrap()
            //     .descriptor_count = 1;
            v.into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap()
        })
        .unwrap();

        // TODO: cache and use other thingy, yk, the other thing, pipeline library or whatever
        let pipeline = unsafe { GraphicsPipeline::new_unchecked(
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
                        write_enable: true,
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
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                src_color_blend_factor: BlendFactor::SrcColor,
                                dst_color_blend_factor: BlendFactor::Zero,
                                color_blend_op: BlendOp::Add,
                                src_alpha_blend_factor: BlendFactor::SrcColor,
                                dst_alpha_blend_factor: BlendFactor::Zero,
                                alpha_blend_op: BlendOp::Add,
                            }),
                            color_write_mask: ColorComponents::all(),
                            color_write_enable: true,
                        },
                        ColorBlendAttachmentState::default(),
                        ColorBlendAttachmentState::default(),
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
                        view_mask: 0,
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
        .unwrap() };
        pipeline
            .set_debug_utils_object_name(Some("Simple Opaque Pipeline"))
            .unwrap();
        Self {
            pipeline,
            pipeline_layout: layout,
        }
    }
}
