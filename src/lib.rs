use std::sync::Arc;

use vulkano::{
    Version, VulkanLibrary,
    command_buffer::{
        CommandBuffer, PrimaryCommandBufferAbstract,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateFlags,
        QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{GenericMemoryAllocatorCreateInfo, StandardMemoryAllocator},
};

pub mod dmatex;
pub mod materials;
pub mod mesh_manager;
pub mod object_manager;
pub mod pipeline_pieces;
pub mod render;
pub mod resource_manager;
pub mod subpasses;
pub mod vulkano_exts;
pub mod images;

pub struct Context {
    pub instance: Arc<Instance>,
    pub phys_dev: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub alloc: Arc<StandardMemoryAllocator>,
    pub cmd_buf_alloc: Arc<StandardCommandBufferAllocator>,
    pub gfx_queue: Arc<Queue>,
}
impl Context {
    // TODO: proper error handling
    pub fn new(
        entry: Arc<VulkanLibrary>,
        additional_instance_exts: InstanceExtensions,
        additional_device_exts: DeviceExtensions,
        additional_device_features: DeviceFeatures,
    ) -> Self {
        let required_exts = InstanceExtensions {
            khr_external_memory_capabilities: true,
            khr_external_semaphore_capabilities: true,
            // is this needed on 1.3?
            khr_get_physical_device_properties2: true,
            ext_debug_utils: true,
            ..Default::default()
        } | additional_instance_exts;
        let supported_exts = entry.supported_extensions();
        if !supported_exts.contains(&required_exts) {
            panic!("required exts are not supported: {:#?}", supported_exts);
        }
        let instance = Instance::new(
            entry,
            InstanceCreateInfo {
                application_name: Some("Test App".into()),
                application_version: Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                },
                engine_name: Some("stardust-vulkan".into()),
                engine_version: Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                },
                max_api_version: Some(Version::V1_3),
                enabled_extensions: required_exts,
                // TODO: handle debug messages
                ..Default::default()
            },
        )
        .unwrap();
        let mut devs = instance
            .enumerate_physical_devices()
            .unwrap()
            .collect::<Vec<_>>();
        devs.sort_by_key(|dev| match dev.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        });
        let required_device_extensions = DeviceExtensions {
            ext_image_drm_format_modifier: true,
            khr_dynamic_rendering_local_read: true,
            // TODO: add vulkano bindings for global priority, ideally upstream
            // khr_global_priority: true,
            ext_ycbcr_image_arrays: true,
            khr_external_memory: true,
            khr_external_memory_fd: true,
            khr_external_semaphore: true,
            khr_external_semaphore_fd: true,
            khr_push_descriptor: true,
            ..Default::default()
        } | additional_device_exts;
        let required_device_features = DeviceFeatures {
            timeline_semaphore: true,
            multi_draw_indirect: true,
            draw_indirect_count: true,
            descriptor_indexing: true,
            image_cube_array: true,
            shader_input_attachment_array_dynamic_indexing: true,
            shader_input_attachment_array_non_uniform_indexing: true,
            // global_priority_query: true,
            dynamic_rendering: true,
            dynamic_rendering_local_read: true,
            synchronization2: true,
            buffer_device_address: true,
            multiview: true,
            sampler_ycbcr_conversion: true,
            ycbcr_image_arrays: true,
            scalar_block_layout: true,
            shader_int64: true,
            shader_int8: true,
            shader_draw_parameters: true,
            storage_buffer8_bit_access: true,
            independent_blend: true,
            ..Default::default()
        } | additional_device_features;

        let phys_dev = devs
            .into_iter()
            .find(|dev| {
                dbg!(required_device_extensions.difference(dev.supported_extensions()));
                dbg!(required_device_features.difference(dev.supported_features()));
                dev.supported_extensions()
                    .contains(&required_device_extensions)
                    && dev.supported_features().contains(&required_device_features)
            })
            .expect("no suitable device found");

        let info = phys_dev.queue_family_properties();
        let mut graphics_queue_family_index = None;
        for (i, info) in info.into_iter().enumerate() {
            if graphics_queue_family_index.is_none()
                && info.queue_flags.contains(QueueFlags::GRAPHICS)
            {
                graphics_queue_family_index.replace(i);
                if info.queue_count == 1 {
                    continue;
                }
            }
        }
        let gfx_queue_family = graphics_queue_family_index.expect("no graphics queue") as u32;
        let (device, mut queues) = Device::new(
            phys_dev.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    flags: QueueCreateFlags::empty(),
                    queue_family_index: gfx_queue_family,
                    queues: vec![1.0],
                    ..Default::default()
                }],
                enabled_extensions: required_device_extensions,
                enabled_features: required_device_features,
                ..Default::default()
            },
        )
        .unwrap();
        let gfx_queue = queues.next().unwrap();
        Self {
            alloc: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            cmd_buf_alloc: Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default(),
            )),
            instance,
            phys_dev,
            device,
            gfx_queue,
        }
    }
}
