use std::sync::Arc;

use timeline_syncobj::render_node::DrmRenderNode;
use vulkano::{
    Validated, Version, VulkanError, VulkanLibrary,
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::{
        StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateFlags,
        QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::StandardMemoryAllocator,
};

use crate::queue::StarlightQueue;

pub mod command_buffer;
pub mod dmatex;
pub mod images;
pub mod materials;
pub mod mesh_manager;
pub mod object_manager;
pub mod pipeline_pieces;
pub mod queue;
pub mod render;
pub mod resource_manager;
pub mod resources;
pub mod subpasses;
pub mod vulkano_exts;

pub struct Context {
    pub instance: Arc<Instance>,
    pub phys_dev: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub alloc: Arc<StandardMemoryAllocator>,
    pub cmd_buf_alloc: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_alloc: Arc<StandardDescriptorSetAllocator>,
    pub gfx_queue: StarlightQueue,
    pub transfer_queue: Option<StarlightQueue>,
    pub drm_render_node: DrmRenderNode,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplicationInfo {
    pub name: Option<String>,
    pub version: Version,
}
impl Context {
    // TODO: proper error handling
    pub fn new(
        entry: &Arc<VulkanLibrary>,
        app_info: ApplicationInfo,
        additional_instance_exts: InstanceExtensions,
        additional_device_exts: DeviceExtensions,
        additional_device_features: DeviceFeatures,
    ) -> Self {
        Self::new_custom_init(
            entry,
            app_info,
            additional_instance_exts,
            additional_device_exts,
            additional_device_features,
            |info| Instance::new(entry.clone(), info),
            |instance, required_device_extensions, required_device_features| {
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
                Ok(phys_dev)
            },
            |phys_dev, info| Device::new(phys_dev.clone(), info),
        )
    }
    pub fn new_custom_init<I: ExactSizeIterator<Item = Arc<Queue>>>(
        entry: &Arc<VulkanLibrary>,
        app_info: ApplicationInfo,
        additional_instance_exts: InstanceExtensions,
        additional_device_exts: DeviceExtensions,
        additional_device_features: DeviceFeatures,
        instance_create: impl FnOnce(
            InstanceCreateInfo,
        ) -> Result<Arc<Instance>, Validated<VulkanError>>,
        select_phys_dev: impl FnOnce(
            &Arc<Instance>,
            &DeviceExtensions,
            &DeviceFeatures,
        ) -> Result<Arc<PhysicalDevice>, Validated<VulkanError>>,
        device_create: impl FnOnce(
            &Arc<PhysicalDevice>,
            DeviceCreateInfo,
        ) -> Result<(Arc<Device>, I), Validated<VulkanError>>,
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
        let missing = required_exts.difference(&supported_exts);
        if !missing.is_empty() {
            panic!("required exts are not supported: {missing:#?}");
        }
        let instance = instance_create(InstanceCreateInfo {
            application_name: app_info.name,
            application_version: app_info.version,
            engine_name: Some("Starlight Renderer".into()),
            engine_version: Version {
                major: 0,
                minor: 1,
                patch: 0,
            },
            max_api_version: Some(Version::V1_3),
            enabled_extensions: required_exts,
            // TODO: handle debug messages
            ..Default::default()
        })
        .unwrap();
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
            // ext_rgba10x6_formats: true,
            ..Default::default()
        } | additional_device_exts;
        let required_device_features = DeviceFeatures {
            timeline_semaphore: true,
            multi_draw_indirect: true,
            draw_indirect_count: true,
            descriptor_indexing: true,
            runtime_descriptor_array: true,
            descriptor_binding_partially_bound: true,
            descriptor_binding_sampled_image_update_after_bind: true,
            shader_sampled_image_array_non_uniform_indexing: true,
            shader_input_attachment_array_dynamic_indexing: true,
            shader_input_attachment_array_non_uniform_indexing: true,
            // TODO: this is needed for this format to be usable at all, as we do ycbcr in shader,
            // technically optional, but modifies format support
            // format_rgba10x6_without_y_cb_cr_sampler: true,
            image_cube_array: true,
            // global_priority_query: true,
            dynamic_rendering: true,
            dynamic_rendering_local_read: true,
            synchronization2: true,
            buffer_device_address: true,
            multiview: true,
            // sampler_ycbcr_conversion: true,
            // ycbcr_image_arrays: true,
            scalar_block_layout: true,
            shader_int64: true,
            shader_int8: true,
            shader_draw_parameters: true,
            storage_buffer8_bit_access: true,
            fragment_stores_and_atomics: true,
            independent_blend: true,
            ..Default::default()
        } | additional_device_features;
        let phys_dev = select_phys_dev(
            &instance,
            &required_device_extensions,
            &required_device_features,
        )
        .expect("failed to select phys dev");
        let info = phys_dev.queue_family_properties();
        let mut graphics_queue_family_index = None;
        let mut transfer_queue_family_index = None;
        for (i, info) in info.into_iter().enumerate() {
            if graphics_queue_family_index.is_none()
                && info.queue_flags.contains(QueueFlags::GRAPHICS)
            {
                graphics_queue_family_index.replace(i);
            }
            if transfer_queue_family_index.is_none()
                && info.queue_flags.contains(QueueFlags::TRANSFER)
                && !info.queue_flags.contains(QueueFlags::GRAPHICS)
            {
                transfer_queue_family_index.replace(i);
            }
        }
        let gfx_queue_family = graphics_queue_family_index.expect("no graphics queue") as u32;
        let mut queue_create_infos = vec![QueueCreateInfo {
            flags: QueueCreateFlags::empty(),
            queue_family_index: gfx_queue_family,
            queues: vec![1.0],
            ..Default::default()
        }];
        if let Some(transfer_queue_family) = transfer_queue_family_index {
            queue_create_infos.push(QueueCreateInfo {
                flags: QueueCreateFlags::empty(),
                queue_family_index: transfer_queue_family as u32,
                queues: vec![1.0],
                ..Default::default()
            });
        }
        let device_create_info = DeviceCreateInfo {
            queue_create_infos,
            enabled_extensions: required_device_extensions,
            enabled_features: required_device_features,
            ..Default::default()
        };
        let (device, mut queues) = device_create(&phys_dev, device_create_info).unwrap();
        let gfx_queue = queues.next().unwrap();
        let transfer_queue = transfer_queue_family_index
            .is_some()
            .then(|| queues.next().unwrap());
        // TODO: return error instead
        let render_minor = phys_dev
            .properties()
            .render_minor
            .map(i64::cast_unsigned)
            .or_else(|| pci_render_minor(&phys_dev))
            .unwrap();
        // TODO: return error instead here too
        let drm_render_node = DrmRenderNode::new(render_minor).unwrap();
        let gfx_queue = StarlightQueue::new(gfx_queue, drm_render_node.clone());
        gfx_queue
            .set_debug_label(Some("Main Graphics Queue"))
            .unwrap();
        let transfer_queue =
            transfer_queue.map(|q| StarlightQueue::new(q, drm_render_node.clone()));
        if let Some(q) = transfer_queue.as_ref() {
            q.set_debug_label(Some("Transfer Queue")).unwrap();
        }
        Self {
            alloc: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            cmd_buf_alloc: Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default(),
            )),
            descriptor_set_alloc: Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                StandardDescriptorSetAllocatorCreateInfo {
                    update_after_bind: true,
                    ..Default::default()
                },
            )),
            instance,
            phys_dev,
            device,
            gfx_queue,
            drm_render_node,
            transfer_queue,
        }
    }
}

// renderdoc hides VK_EXT_physical_device_drm but passes VK_EXT_pci_bus_info through
fn pci_render_minor(dev: &PhysicalDevice) -> Option<u64> {
    let p = dev.properties();
    let link = std::fs::read_link(format!(
        "/dev/dri/by-path/pci-{:04x}:{:02x}:{:02x}.{}-render",
        p.pci_domain?, p.pci_bus?, p.pci_device?, p.pci_function?
    ))
    .ok()?;
    link.file_name()?.to_str()?.strip_prefix("renderD")?.parse().ok()
}
