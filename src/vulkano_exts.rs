use ash::vk;
use vulkano::{VulkanObject, command_buffer::RecordingCommandBuffer, device::DeviceOwned};

pub trait RecordingCommandBufferExt {
    /// color_indecies maps the begin_rendering attachments to indecies used in the shader, use
    /// vk::ATTACHMENT_UNUSED (u32::MAX) to ignore an attachment
    fn set_input_attachment_indecies(
        &mut self,
        color_indecies: &[u32],
        depth_index: Option<u32>,
        stencil_index: Option<u32>,
    ) -> &mut Self;
}
impl RecordingCommandBufferExt for RecordingCommandBuffer {
    /// color_indecies maps the begin_rendering attachments to indecies used in the shader, use
    /// vk::ATTACHMENT_UNUSED (u32::MAX) to ignore an attachment
    fn set_input_attachment_indecies(
        &mut self,
        color_indecies: &[u32],
        depth_index: Option<u32>,
        stencil_index: Option<u32>,
    ) -> &mut Self {
        let mut v = vk::RenderingInputAttachmentIndexInfoKHR::default()
            .color_attachment_input_indices(color_indecies);
        if let Some(index) = depth_index.as_ref() {
            v = v.depth_input_attachment_index(index);
        }
        if let Some(index) = stencil_index.as_ref() {
            v = v.stencil_input_attachment_index(index);
        }
        unsafe {
            (self
                .device()
                .fns()
                .khr_dynamic_rendering_local_read
                .cmd_set_rendering_input_attachment_indices_khr)(self.handle(), &v)
        }
        self
    }
}
