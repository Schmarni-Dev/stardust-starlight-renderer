use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64, Ordering},
};

use glam::{Vec2, Vec3, Vec4};
use offset_allocator::{Allocation, Allocator};
use tokio::sync::{Mutex, RwLock};
use vulkano::{
    Validated,
    buffer::{
        AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::graphics::vertex_input::Vertex,
};

#[derive(Vertex, BufferContents, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct VertexData {
    #[format(R32G32B32_SFLOAT)]
    pub pos: Vec3,
    #[format(R32G32B32_SFLOAT)]
    pub normal: Vec3,
    #[format(R32G32_SFLOAT)]
    pub uv: Vec2,
    #[format(R32G32B32A32_SFLOAT)]
    /// premultiplied linear RGBA
    pub color: Vec4,
}

#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
pub struct SkinnedVertex {
    pub joint_weights: [f32; 4],
    pub joint_indecies: [u32; 4],
}

pub struct MeshManager {
    id: u32,

    pub(crate) vertex_buffer: RwLock<Subbuffer<[VertexData]>>,
    pub(crate) index_buffer: RwLock<Subbuffer<[u32]>>,
    allocs: Mutex<Allocs>,
    vertex_len: AtomicU64,
    index_len: AtomicU64,
}

struct Allocs {
    vertex: Allocator,
    index: Allocator,
}

pub struct Mesh {
    pub(crate) manager: Arc<MeshManager>,
    pub(crate) vertex_alloc: Allocation,
    pub(crate) vertex_len: u32,
    pub(crate) index_alloc: Allocation,
    pub(crate) index_len: u32,
}

static MESH_MANAGER_ID: AtomicU32 = AtomicU32::new(0);
// TODO: figure out compacting?
impl MeshManager {
    pub fn new(ctx: &crate::Context) -> Result<Arc<Self>, Validated<AllocateBufferError>> {
        let initial_len = 1024;
        let vertex_buffer = Buffer::new_slice(
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
            initial_len,
        )?
        .into();
        let index_buffer = Buffer::new_slice(
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
            initial_len,
        )?
        .into();
        Ok(Self {
            id: MESH_MANAGER_ID.fetch_add(1, Ordering::Relaxed),
            vertex_buffer,
            index_buffer,
            allocs: Mutex::new(Allocs {
                vertex: Allocator::new(u32::MAX - 8),
                index: Allocator::new(u32::MAX - 8),
            }),
            vertex_len: initial_len.into(),
            index_len: initial_len.into(),
        }
        .into())
    }
    pub async fn alloc(
        self: &Arc<Self>,
        ctx: &crate::Context,
        vertex_buffer: &[VertexData],
        indecies: &[u32],
    ) -> Result<Arc<Mesh>, Validated<AllocateBufferError>> {
        debug_assert!(vertex_buffer.len() <= u32::MAX as usize);
        debug_assert!(indecies.len() <= u32::MAX as usize);
        let mut allocs = self.allocs.lock().await;
        // the allocator thinks we have u32::MAX vertex data slots, if this fails we have bigger
        // issues
        let vertex_alloc = allocs.vertex.allocate(indecies.len() as u32).unwrap();
        // TODO: reuse this for both buffers
        while let vertex_len = self.vertex_len.load(Ordering::Relaxed)
            && vertex_alloc.offset as u64 + indecies.len() as u64 > vertex_len
        {
            let vertex_buf = match Buffer::new_slice(
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
                vertex_len * 2,
            ) {
                Ok(v) => v,
                Err(err) => {
                    allocs.vertex.free(vertex_alloc);
                    tracing::error!("failed to grow vertex buffer: {err}");
                    return Err(err);
                }
            };
            self.vertex_len.store(vertex_len * 2, Ordering::Relaxed);
            let read_guard = self.vertex_buffer.read().await;
            let reader = read_guard.read().unwrap();
            let mut writer = vertex_buf.write().unwrap();
            writer
                .iter_mut()
                .zip(reader.iter())
                .for_each(|(v, v2)| *v = *v2);
            drop(reader);
            drop(writer);
            drop(read_guard);
            *self.vertex_buffer.write().await = vertex_buf;
        }
        let index_alloc = allocs.index.allocate(indecies.len() as u32).unwrap();
        while let index_len = self.index_len.load(Ordering::Relaxed)
            && index_alloc.offset as u64 + indecies.len() as u64 > index_len
        {
            let index_buf = match Buffer::new_slice(
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
                index_len * 2,
            ) {
                Ok(v) => v,
                Err(err) => {
                    allocs.vertex.free(vertex_alloc);
                    allocs.index.free(index_alloc);
                    tracing::error!("failed to grow index buffer: {err}");
                    return Err(err);
                }
            };
            self.index_len.store(index_len * 2, Ordering::Relaxed);
            let read_guard = self.index_buffer.read().await;
            let reader = read_guard.read().unwrap();
            let mut writer = index_buf.write().unwrap();
            writer
                .iter_mut()
                .zip(reader.iter())
                .for_each(|(v, v2)| *v = *v2);
            drop(reader);
            drop(read_guard);
            drop(writer);
            *self.index_buffer.write().await = index_buf;
        }

        let vertex_offset = vertex_alloc.offset as usize;
        self.vertex_buffer.read().await.write().unwrap()
            [vertex_offset..(vertex_offset + vertex_buffer.len())]
            .copy_from_slice(vertex_buffer);
        debug_assert_eq!(
            &self.vertex_buffer.read().await.read().unwrap()
                [vertex_offset..(vertex_offset + vertex_buffer.len())],
            vertex_buffer
        );
        let index_offset = index_alloc.offset as usize;
        self.index_buffer.read().await.write().unwrap()
            [index_offset..(index_offset + indecies.len())]
            .copy_from_slice(indecies);
        Ok(Arc::new(Mesh {
            manager: self.clone(),
            vertex_alloc,
            vertex_len: vertex_buffer.len() as u32,
            index_alloc,
            index_len: indecies.len() as u32,
        }))
    }
}
impl Drop for Mesh {
    fn drop(&mut self) {
        let manager = self.manager.clone();
        let vertex_alloc = self.vertex_alloc;
        let index_alloc = self.index_alloc;
        tokio::spawn(async move {
            // TODO: handle gpu sync properly by waiting for final usage to finish, using timeline
            // syncobj
            let mut allocs = manager.allocs.lock().await;
            allocs.vertex.free(vertex_alloc);
            allocs.index.free(index_alloc);
        });
    }
}
