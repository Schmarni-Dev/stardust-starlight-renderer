use std::sync::Arc;

use glam::Vec4;
use vulkano::{
    DeviceSize, Validated,
    buffer::{AllocateBufferError, BufferContents},
};

use crate::{
    Context,
    materials::MaterialHandle,
    resource_manager::{ResourceHandle, ResourceManager},
};

#[derive(Debug, BufferContents)]
#[repr(C)]
pub struct PbrData {
    pub color: Vec4,
}

#[derive(Debug)]
pub struct PbrMaterial {
    handle: ResourceHandle<PbrData>,
}
impl MaterialHandle for PbrMaterial {
    unsafe fn get_device_ptr(&self) -> DeviceSize {
        self.handle.device_address().unwrap().get()
    }
}
impl PbrMaterial {
    pub fn update(&self, new_data: PbrData) {
        *self.handle.write().unwrap() = new_data;
    }
}

const MATERIAL_BUFFER_SIZE: u64 = 512;
pub struct PbrMaterialManager {
    manager: Arc<ResourceManager<PbrData>>,
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
        let handle = self.manager.alloc(ctx, data).await?;
        Ok(Arc::new(PbrMaterial { handle }))
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/materials/simple_pbr.frag",
        include: ["src/materials/", "src/pipeline_pieces/"],
    }
}
