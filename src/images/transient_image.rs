use std::sync::Arc;

use vulkano::{
    Validated, VulkanError,
    device::DeviceOwnedVulkanObject,
    image::{
        Image, ImageCreateInfo, ImageLayout, ImageMemory, ImageUsage,
        sys::RawImage,
        view::{ImageView, ImageViewCreateInfo},
    },
    memory::{
        DedicatedAllocation, DeviceMemory, MemoryAllocateInfo, MemoryPropertyFlags, ResourceMemory,
    },
    sync::{AccessFlags, ImageMemoryBarrier, PipelineStages},
};

use crate::Context;

pub struct TransientAttachment {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
}
impl TransientAttachment {
    pub fn new(
        ctx: &Context,
        mut create_info: ImageCreateInfo,
        debug_name: &str,
    ) -> Result<Self, Validated<VulkanError>> {
        create_info.usage |= ImageUsage::TRANSIENT_ATTACHMENT;
        create_info.usage |= ImageUsage::INPUT_ATTACHMENT;
        let img = RawImage::new(ctx.device.clone(), create_info)?;
        let mem_reqs = img.memory_requirements();
        let mut raw_mems = Vec::with_capacity(mem_reqs.len());
        let mem_types = ctx
            .phys_dev
            .memory_properties()
            .memory_types
            .iter()
            .enumerate()
            .fold(0u32, |v, (i, mem)| {
                if mem
                    .property_flags
                    .contains(MemoryPropertyFlags::LAZILY_ALLOCATED)
                {
                    v | (1 << i)
                } else {
                    v
                }
            });
        for mem_req in mem_reqs {
            let combined_mem_types = mem_types & mem_req.memory_type_bits;
            let mem_types = (combined_mem_types != 0)
                .then_some(combined_mem_types)
                .unwrap_or(mem_req.memory_type_bits);
            let memory_type_index = (0..32)
                .find(|v| (mem_types & (1 << *v)) != 0)
                .expect("no valid memory type index for transient attachment");

            raw_mems.push(ResourceMemory::new_dedicated(DeviceMemory::allocate(
                ctx.device.clone(),
                MemoryAllocateInfo {
                    allocation_size: mem_req.layout.size(),
                    memory_type_index,
                    dedicated_allocation: (mem_req.requires_dedicated_allocation
                        | mem_req.prefers_dedicated_allocation)
                        .then(|| DedicatedAllocation::Image(&img)),
                    ..Default::default()
                },
            )?));
        }
        let image = match img.bind_memory(raw_mems).map(Arc::new) {
            Ok(v) => v,
            Err((err, _, _)) => return Err(err),
        };
        let image_view = ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image))?;
        _ = image.set_debug_utils_object_name(Some(debug_name));
        _ = image_view.set_debug_utils_object_name(Some(&format!("{debug_name} View")));
        Ok(Self { image, image_view })
    }
    pub fn image(&self) -> Arc<Image> {
        self.image.clone()
    }
    pub fn image_view(&self) -> Arc<ImageView> {
        self.image_view.clone()
    }
    pub fn image_ref(&self) -> &Arc<Image> {
        &self.image
    }
    pub fn image_view_ref(&self) -> &Arc<ImageView> {
        &self.image_view
    }
    pub fn setup_barrier(
        &self,
        dst_stages: PipelineStages,
        dst_access: AccessFlags,
    ) -> ImageMemoryBarrier {
        ImageMemoryBarrier {
            src_stages: PipelineStages::TOP_OF_PIPE,
            src_access: AccessFlags::empty(),
            dst_stages,
            dst_access,
            // TODO: optimize?
            old_layout: ImageLayout::Undefined,
            // LocalRead would be ideal, but vulkano doesn't have a variant for that
            new_layout: ImageLayout::General,
            subresource_range: self.image.subresource_range(),
            ..ImageMemoryBarrier::image(self.image.clone())
        }
    }
    pub fn local_read_barrier(
        &self,
        src_stages: PipelineStages,
        src_access: AccessFlags,
    ) -> ImageMemoryBarrier {
        ImageMemoryBarrier {
            src_stages,
            src_access,
            // TODO: is there every a case where we can and want to local read outside a fragment
            // shader?
            dst_stages: PipelineStages::FRAGMENT_SHADER,
            dst_access: AccessFlags::INPUT_ATTACHMENT_READ,
            // LocalRead would be ideal, but vulkano doesn't have a variant for that
            old_layout: ImageLayout::General,
            new_layout: ImageLayout::General,
            subresource_range: self.image.subresource_range(),
            ..ImageMemoryBarrier::image(self.image.clone())
        }
    }
}
