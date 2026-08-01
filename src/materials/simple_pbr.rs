use std::sync::Arc;

use glam::Vec4;
use tokio::sync::RwLock;
use vulkano::{
    DeviceSize, Validated,
    buffer::{AllocateBufferError, BufferContents},
};

use crate::{
    Context,
    images::ImageRef,
    materials::MaterialHandle,
    resource_manager::{ResourceHandle, ResourceManager},
};

#[derive(Debug, BufferContents)]
#[repr(C)]
struct PbrBufferData {
    color: Vec4,
    color_image: u64,
}
#[derive(Debug)]
pub struct PbrData {
    pub color: Vec4,
    pub color_image: Option<Arc<ImageRef>>,
}
impl PbrData {
    fn to_buffer_data(&self) -> PbrBufferData {
        PbrBufferData {
            color: self.color,
            color_image: self
                .color_image
                .as_ref()
                .map(|v| v.bda().get())
                .unwrap_or(0),
        }
    }
}

#[derive(Debug)]
pub struct PbrMaterial {
    data: RwLock<PbrData>,
    handle: ResourceHandle<PbrBufferData>,
}
impl MaterialHandle for PbrMaterial {
    unsafe fn get_device_ptr(&self) -> DeviceSize {
        self.handle.device_address().unwrap().get()
    }
}
impl PbrMaterial {
    pub async fn update(&self, new_data: PbrData) {
        *self.handle.write().unwrap() = new_data.to_buffer_data();
        *self.data.write().await = new_data;
    }
}

const MATERIAL_BUFFER_SIZE: u64 = 512;
pub struct PbrMaterialManager {
    manager: Arc<ResourceManager<PbrBufferData>>,
}
impl PbrMaterialManager {
    pub fn new() -> Self {
        Self {
            manager: ResourceManager::new(MATERIAL_BUFFER_SIZE),
        }
    }
    pub async fn alloc(
        &self,
        ctx: &Context,
        data: PbrData,
    ) -> Result<Arc<PbrMaterial>, Validated<AllocateBufferError>> {
        let handle = self.manager.alloc(ctx, data.to_buffer_data()).await?;
        Ok(Arc::new(PbrMaterial {
            handle,
            data: RwLock::new(data),
        }))
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        vulkan_version: "1.3",
        ty: "fragment",
        path: "src/materials/simple_pbr.frag",
        include: ["src/materials/", "src/pipeline_pieces/"],
    }
}
