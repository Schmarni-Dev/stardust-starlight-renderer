use std::sync::Arc;

use glam::Vec4;
use tokio::sync::RwLock;
use vulkano::{
    DeviceSize, Validated,
    buffer::{AllocateBufferError, BufferContents},
};

use crate::{
    Context,
    images::ImageRefData,
    materials::MaterialHandle,
    resource_manager::{ResourceHandle, ResourceManager},
    resources::{ReadOp, buffer::Bda, image::ImageReadOp},
};

#[derive(Debug, BufferContents, Clone, Copy)]
#[repr(C)]
struct PbrBufferData {
    color: Vec4,
    color_image: Option<Bda<'static, ImageRefData>>,
}
#[derive(Debug)]
pub struct PbrData {
    pub color: Vec4,
    pub color_image: Option<Arc<dyn ImageReadOp>>,
}
impl PbrData {
    fn to_buffer_data(&self) -> PbrBufferData {
        PbrBufferData {
            color: self.color,
            color_image: self
                .color_image
                .as_ref()
                // # Safety
                // a handle to this is held for as long as this is potentially accessed
                .map(|v| unsafe { v.image_ref().bda().to_static() }),
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
        // # Safety
        // this surrounding function has the same guarantees that this needs
        unsafe {
            self.handle
                .device_address()
                .unwrap()
                .to_static()
                .as_raw()
                .get()
        }
    }

    fn get_read_ops(&self) -> impl IntoIterator<Item = Arc<dyn ReadOp>> {
        // TODO: figure out how to make this lockless
        // TODO: make this work for both runtime flavors? maybe?
        let v = tokio::task::block_in_place(|| self.data.blocking_read());
        v.color_image.clone().map(|v| v as _)
    }
}
impl PbrMaterial {
    pub async fn update(&self, new_data: PbrData) {
        self.handle
            .new_wite_op()
            .write(|mut v| *v = new_data.to_buffer_data())
            .await
            .unwrap();
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
