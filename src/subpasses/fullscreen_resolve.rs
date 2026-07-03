use std::{collections::HashSet, sync::Arc};

use glam::{Vec2, vec2};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::RecordingCommandBuffer,
    descriptor_set::{
        DescriptorImageViewInfo, WriteDescriptorSet, layout::DescriptorSetLayoutCreateFlags,
    },
    device::DeviceOwnedVulkanObject as _,
    format::Format,
    image::{ImageLayout, SampleCount},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineCreateFlags, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
                ColorBlendStateFlags, ColorComponents,
            },
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
            vertex_input::{Vertex, VertexDefinition as _},
            viewport::ViewportState,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
};

use crate::{Context, render::RenderContext};
#[derive(Debug, BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct SimpleVertex {
    #[format(R32G32_SFLOAT)]
    position: Vec2,
}

pub struct FullscreenResolve {
    pipeline: Arc<GraphicsPipeline>,
    pipeline_layout: Arc<PipelineLayout>,
    mesh: Subbuffer<[SimpleVertex]>,
    indecies: Subbuffer<[u16]>,
}
impl FullscreenResolve {
    pub fn record(&mut self, ctx: &RenderContext, cmds: &mut RecordingCommandBuffer) {
        unsafe {
            cmds.bind_vertex_buffers(0, &[self.mesh.as_bytes().clone()])
                .unwrap()
                .bind_index_buffer(&vulkano::buffer::IndexBuffer::U16(self.indecies.clone()))
                .unwrap()
                .push_descriptor_set(
                    PipelineBindPoint::Graphics,
                    &self.pipeline_layout,
                    0,
                    &[WriteDescriptorSet::image_view_with_layout(
                        0,
                        DescriptorImageViewInfo {
                            image_view: ctx.opaque.image_view(),
                            image_layout: ImageLayout::General,
                        },
                    )],
                )
                .unwrap()
                // .set_input_attachment_indecies(&[0, u32::MAX, u32::MAX, u32::MAX], Some(1), None)
                .bind_pipeline_graphics(&self.pipeline)
                .unwrap()
                .draw_indexed(6, 1, 0, 0, 0)
                .unwrap();
        }
    }
    pub fn new(ctx: &Context, opaque_format: Format) -> Self {
        let vs = vs::load(ctx.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(ctx.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let vertex_input_state = SimpleVertex::per_vertex().definition(&vs).unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(ctx.device.clone(), {
            let mut v = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
            v.set_layouts[0].flags |= DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;
            v.set_layouts[0]
                .bindings
                .get_mut(&0)
                .unwrap()
                .descriptor_count = 1;

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
                        cull_mode: CullMode::Back,
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
                    depth_stencil_state: Some(DepthStencilState::default()),
                    color_blend_state: Some(ColorBlendState {
                        flags: ColorBlendStateFlags::empty(),
                        attachments: vec![
                            ColorBlendAttachmentState::default(),
                            ColorBlendAttachmentState::default(),
                            ColorBlendAttachmentState::default(),
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
                            // TODO: fill all formats here
                            color_attachment_formats: vec![
                                Some(opaque_format),
                                Some(Format::R16G16B16A16_SFLOAT),
                                Some(Format::R8_UNORM),
                                Some(opaque_format),
                            ],
                            // TODO: properly expose instead of hardcoding
                            depth_attachment_format: Some(Format::D32_SFLOAT),
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
        let mesh_data = &[
            SimpleVertex {
                position: vec2(-1.0, 1.0),
            },
            SimpleVertex {
                position: vec2(1.0, 1.0),
            },
            SimpleVertex {
                position: vec2(1.0, -1.0),
            },
            SimpleVertex {
                position: vec2(-1.0, -1.0),
            },
        ];
        let indecies_data = &[0, 3, 1, 1, 3, 2];
        let mesh = Buffer::new_slice(
            ctx.alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            mesh_data.len() as u64,
        )
        .unwrap();
        mesh.write().unwrap().copy_from_slice(mesh_data);
        let indecies = Buffer::new_slice(
            ctx.alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indecies_data.len() as u64,
        )
        .unwrap();
        pipeline
            .set_debug_utils_object_name(Some("Fullscreen Resolve Pipeline"))
            .unwrap();
        indecies.write().unwrap().copy_from_slice(indecies_data);
        Self {
            pipeline,
            pipeline_layout: layout,
            mesh,
            indecies,
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
    ty: "vertex",
    src: r"
                #version 450

                layout(location = 0) in vec2 position;

                void main() {
                    gl_Position = vec4(position,0.0,1.0);
                }
            ",

        }
}
mod fs {
    vulkano_shaders::shader! {
    ty: "fragment",
    src: r"
                #version 450

                layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;
                // layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput inputDepth;


                layout(location = 3) out vec4 color;

                void main() {
                    vec2 uv = vec2(gl_FragCoord.r / 800, gl_FragCoord.g / 600);
                    color = subpassLoad(inputColor);
                    // float d = subpassLoad(inputDepth).r;
                    // color = vec4(d,d,d,1.0);
                }
            ",

        }
}
