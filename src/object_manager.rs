use std::{collections::HashMap, fmt::Debug, sync::Arc};

use glam::Mat4;
use tracing::warn;
use vulkano::{
    DeviceAddress, Validated,
    buffer::{AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::DrawIndexedIndirectCommand,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::{
    Context,
    command_buffer::{StarlightCommandBuffer, StarlightIndexBuffer},
    materials::MaterialHandle,
    mesh_manager::{Mesh, MeshManagerReadOp},
    resource_manager::{ResourceHandle, ResourceManager},
    resources::buffer::{Bda, StarlightSubBuffer},
};

#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
struct ObjectMetadata {
    /// pointer to pipeline dependent material, NonNull
    material: DeviceAddress,
    /// potentially null device pointer
    data: DeviceAddress,
    /// Never None
    global_transform: Option<Bda<'static, Mat4>>,
    /// [`ExtDataType`]
    data_type: u8,
}

#[repr(u8)]
pub enum ExtDataType {
    None = 0,
}

pub struct Object<Material: MaterialHandle> {
    mesh: Arc<Mesh>,
    material: Arc<Material>,
    transform: ResourceHandle<Mat4>,
}

const TRANSFORM_BUFFER_SIZE: u64 = 512;
#[derive(Debug)]
pub struct ObjectManager {
    // for gpu culling parts of this will be copied into another buffer, maybe be fully constructed
    // on the gpu in the future
    // meta_data_buffer: Subbuffer<[ObjectMetadata]>,

    // this is not copied by the gpu ever, its pointed too
    transform_manager: Arc<ResourceManager<Mat4>>,
}
impl ObjectManager {
    pub fn new() -> Self {
        Self {
            transform_manager: ResourceManager::new(TRANSFORM_BUFFER_SIZE),
        }
    }
    pub async fn alloc<Material: MaterialHandle>(
        &self,
        ctx: &Context,
        mesh: &Arc<Mesh>,
        material: &Arc<Material>,
        global_transform: Mat4,
    ) -> Result<Arc<Object<Material>>, Validated<AllocateBufferError>> {
        let handle = self.transform_manager.alloc(ctx, global_transform).await?;
        Ok(Arc::new(Object {
            mesh: mesh.clone(),
            material: material.clone(),
            transform: handle,
        }))
    }
}

impl<M: MaterialHandle> Object<M> {
    // TODO: add update_global_transforms to ObjectManager to make this not terrible
    pub async fn update_global_transform(&self, global_transform: Mat4) {
        self.transform
            .new_wite_op()
            .write(|mut v| *v = global_transform)
            .await
            .unwrap();
    }
}

// TODO: remove?
pub struct RenderInvocationData<Material: MaterialHandle> {
    metadata: StarlightSubBuffer<[ObjectMetadata]>,
    indirect_buf: StarlightSubBuffer<[DrawIndexedIndirectCommand]>,
    indirect_obj_count: u32,
    mesh_data: MeshManagerReadOp,
    // # Safety: this needs to keep the objects alive to make sure the device ptrs stay valid
    objects: Vec<Arc<Object<Material>>>,
}
impl<Material: MaterialHandle> RenderInvocationData<Material> {
    pub fn new<'a>(
        ctx: &Context,
        objects: Vec<Arc<Object<Material>>>,
        mesh_data: &MeshManagerReadOp,
    ) -> Result<Self, Validated<AllocateBufferError>> {
        let metadata_buf = Buffer::new_slice(
            ctx.alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            objects.len() as u64,
        )?;
        let mut objs_map: HashMap<*const Mesh, Vec<&Object<Material>>> = HashMap::new();
        for obj in &objects {
            objs_map
                .entry(Arc::as_ptr(&obj.mesh))
                .or_default()
                .push(obj);
        }
        let indirect_buf = Buffer::new_slice(
            ctx.alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDIRECT_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            objs_map.len() as u64,
        )?;
        let mut indirect_writer = indirect_buf.write().unwrap();
        let mut meta_writer = metadata_buf.write().unwrap();
        let mut metadata_i = 0;
        for (i, (mesh, objs)) in objs_map.into_iter().enumerate() {
            // # Safety: this is fine because the arc is still held by the object
            let first_vertex = unsafe { mesh.as_ref_unchecked().vertex_alloc.offset };
            // # Safety: this is fine because the arc is still held by the object
            let first_index = unsafe { mesh.as_ref_unchecked().index_alloc.offset };
            // # Safety: this is fine because the arc is still held by the object
            let index_len = unsafe { mesh.as_ref_unchecked().index_len };
            let first_instance = metadata_i;
            for obj in &objs {
                if !mesh_data.from_manager(&obj.mesh.manager) {
                    warn!("Object with mesh from different mesh manager");
                }
                meta_writer[metadata_i] = ObjectMetadata {
                    // # Safety: the material handle is kept alive by storing it in the RenderInvocationData
                    material: unsafe { obj.material.get_device_ptr() },
                    // no extra data implemented yet
                    data: 0,
                    // TODO: verify this safety comment, i'm tired
                    // # Safety: this is fine because we hold a handle to the object, i think
                    global_transform: Some(unsafe {
                        obj.transform.device_address().unwrap().to_static()
                    }),
                    // no extra data implemented yet
                    data_type: ExtDataType::None as u8,
                };
                metadata_i += 1;
            }
            indirect_writer[i] = DrawIndexedIndirectCommand {
                index_count: index_len,
                instance_count: dbg!(objs.len() as u32),
                first_index,
                vertex_offset: first_vertex,
                first_instance: first_instance as u32,
            };
        }
        drop(indirect_writer);
        drop(meta_writer);
        let indirect_obj_count = indirect_buf.len() as u32;
        // its fine to do this here since both buffers are constructed here
        let metadata_buf = StarlightSubBuffer::from_subbuffer(ctx, metadata_buf);
        let indirect_buf = StarlightSubBuffer::from_subbuffer(ctx, indirect_buf);

        Ok(Self {
            metadata: metadata_buf,
            indirect_obj_count,
            indirect_buf,
            // TODO: make sure all meshes come from the same manager and handle empty objects vec
            mesh_data: mesh_data.clone(),
            objects,
        })
    }
    pub async fn bind_buffers(&self, cmds: &mut StarlightCommandBuffer) {
        cmds.used_reads
            .extend(self.objects.iter().flat_map(|v| v.material.get_read_ops()));

        // TODO: make sure that this vertex buffer isn't dropped until the gpu work is done,
        // this could happen if the vertex buffer has to grow
        cmds.bind_vertex_buffers(0, &[self.mesh_data.vertex_buffer.clone()])
            .unwrap();
        // TODO: also make sure this survives, same as with the vertex buffer
        cmds.bind_index_buffer(&StarlightIndexBuffer::U32(
            self.mesh_data.index_buffer.clone(),
        ))
        .unwrap();
    }
    pub fn metadata_bda(&self, cmds: &mut StarlightCommandBuffer) -> DeviceAddress {
        cmds.used_reads.push(self.metadata.new_read_op());
        unsafe {
            self.metadata
                .device_address()
                .unwrap()
                .to_static()
                .as_raw()
                .get()
        }
    }
    pub fn draw(&self, cmds: &mut StarlightCommandBuffer) {
        unsafe {
            // TODO: make sure the buffer lives until the gpu work finishes
            cmds.draw_indexed_indirect(
                &(self.indirect_buf.new_read_op() as _),
                self.indirect_obj_count,
                size_of::<DrawIndexedIndirectCommand>() as u32,
            )
            .unwrap();
        }
    }
}
