use std::{any::Any, sync::Arc};

use vulkano::{
    Validated, ValidationError, VulkanError,
    buffer::{BufferContents, IndexBuffer},
    command_buffer::{
        CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, CopyBufferToImageInfo,
        DrawIndexedIndirectCommand, RecordingCommandBuffer,
    },
    descriptor_set::{DescriptorSet, sys::RawDescriptorSet},
    image::ImageLayout,
    pipeline::{PipelineBindPoint, PipelineLayout},
    sync::{AccessFlags, DependencyInfo, ImageMemoryBarrier, PipelineStages},
};

use crate::{
    Context,
    resources::{ReadOp, WriteOp, buffer::BufferReadOp, image::ImageWriteOp},
};

pub struct StarlightCommandBuffer {
    pub(super) inner: RecordingCommandBuffer,
    pub(super) used_reads: Vec<Arc<dyn ReadOp>>,
    pub(super) used_writes: Vec<Box<dyn WriteOp>>,
    pub(super) used_objects: Vec<Arc<dyn Any + Send + Sync>>,
}
impl StarlightCommandBuffer {
    pub fn new(ctx: &Context, queue_family_index: u32) -> Result<Self, Validated<VulkanError>> {
        let inner = RecordingCommandBuffer::new(
            ctx.cmd_buf_alloc.clone(),
            queue_family_index,
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )?;
        Ok(Self {
            inner,
            used_reads: Vec::new(),
            used_writes: Vec::new(),
            used_objects: Vec::new(),
        })
    }
    pub fn inner(&mut self) -> &mut RecordingCommandBuffer {
        &mut self.inner
    }
}
impl StarlightCommandBuffer {
    // TODO: add regions support
    /// src buffer needs to be created with BufferUsage::TRANSFER_SRC
    pub fn buffer_to_image_copy(
        &mut self,
        src_buffer: Arc<dyn BufferReadOp<impl BufferContents + ?Sized>>,
        dst_image: Box<dyn ImageWriteOp>,
    ) {
        unsafe {
            self.inner
                .pipeline_barrier(&DependencyInfo {
                    image_memory_barriers: [ImageMemoryBarrier {
                        src_stages: PipelineStages::empty(),
                        src_access: AccessFlags::empty(),
                        dst_stages: PipelineStages::COPY,
                        dst_access: AccessFlags::TRANSFER_WRITE,
                        old_layout: dst_image.current_layout(),
                        new_layout: ImageLayout::TransferDstOptimal,
                        subresource_range: dst_image.image().subresource_range(),
                        ..ImageMemoryBarrier::image(dst_image.image().clone())
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })
                .unwrap();
            self.inner
                .copy_buffer_to_image(&CopyBufferToImageInfo::buffer_image(
                    src_buffer.bytes().clone(),
                    dst_image.image().clone(),
                ))
                .unwrap();
            self.inner
                .pipeline_barrier(&DependencyInfo {
                    image_memory_barriers: [ImageMemoryBarrier {
                        src_stages: PipelineStages::COPY,
                        src_access: AccessFlags::TRANSFER_WRITE,
                        // TODO: figure out a good way to populate these somehow? maybe?
                        dst_stages: PipelineStages::empty(),
                        dst_access: AccessFlags::empty(),
                        old_layout: ImageLayout::TransferDstOptimal,
                        new_layout: dst_image.target_layout(),
                        subresource_range: dst_image.image().subresource_range(),
                        ..ImageMemoryBarrier::image(dst_image.image().clone())
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })
                .unwrap();
        }
        self.used_reads.push(src_buffer);
        self.used_writes.push(dst_image);
    }
    pub fn bind_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&Arc<DescriptorSet>],
        dynamic_offsets: &[u32],
    ) -> Result<&mut Self, Box<ValidationError>> {
        let raw_descriptor_sets = descriptor_sets
            .iter()
            .map(|v| v.as_raw())
            .collect::<Vec<&RawDescriptorSet>>();
        unsafe {
            self.inner().bind_descriptor_sets(
                pipeline_bind_point,
                pipeline_layout,
                first_set,
                &raw_descriptor_sets,
                dynamic_offsets,
            )?;
        };
        descriptor_sets.iter().for_each(|v| {
            self.used_objects.push((*v).clone());
        });
        Ok(self)
    }
    /// Binds vertex buffers for future draw calls.
    pub fn bind_vertex_buffers<T: BufferContents + ?Sized>(
        &mut self,
        first_binding: u32,
        vertex_buffers: &[Arc<dyn BufferReadOp<T>>],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.used_reads
            .extend(vertex_buffers.iter().map(|v| v.clone() as _));
        let buffers = vertex_buffers
            .iter()
            .map(|v| v.bytes().clone())
            .collect::<Vec<_>>();
        unsafe { self.inner().bind_vertex_buffers(first_binding, &buffers)? };
        Ok(self)
    }
    pub fn bind_index_buffer(
        &mut self,
        index_buffer: &StarlightIndexBuffer,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.used_reads.push(match index_buffer.clone() {
            StarlightIndexBuffer::U8(v) => v as _,
            StarlightIndexBuffer::U16(v) => v as _,
            StarlightIndexBuffer::U32(v) => v as _,
        });
        unsafe {
            self.inner().bind_index_buffer(&match index_buffer {
                StarlightIndexBuffer::U8(v) => IndexBuffer::U8(v.inner().clone()),
                StarlightIndexBuffer::U16(v) => IndexBuffer::U16(v.inner().clone()),
                StarlightIndexBuffer::U32(v) => IndexBuffer::U32(v.inner().clone()),
            })?
        };
        Ok(self)
    }
    pub unsafe fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &Arc<dyn BufferReadOp<[DrawIndexedIndirectCommand]>>,
        draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        unsafe {
            self.inner()
                .draw_indexed_indirect(indirect_buffer.inner(), draw_count, stride)?;
        }
        Ok(self)
    }
}
/// A buffer holding index values, which index into buffers holding vertex data.
#[derive(Clone)]
pub enum StarlightIndexBuffer {
    /// The [`index_type_uint8`] feature must be enabled on the device.
    U8(Arc<dyn BufferReadOp<[u8]>>),

    U16(Arc<dyn BufferReadOp<[u16]>>),

    U32(Arc<dyn BufferReadOp<[u32]>>),
}
