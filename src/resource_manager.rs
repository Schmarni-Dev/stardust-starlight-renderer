use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use offset_allocator::{Allocation, Allocator};
use tokio::sync::{Mutex, RwLock};
use vulkano::{
    Validated,
    buffer::{
        AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::Context;

// TODO: impl trim at some point
#[derive(Debug)]
pub struct ResourceManager<T: BufferContents + Send> {
    buffer_size: u64,
    buffers: RwLock<Vec<Subbuffer<[T]>>>,
    allocator: Mutex<Allocator>,
}

pub struct ResourceHandle<T: BufferContents + Send> {
    buffer: Subbuffer<T>,
    manager: Arc<ResourceManager<T>>,
    alloc: Allocation,
}

impl<T: BufferContents + Send + std::fmt::Debug> std::fmt::Debug for ResourceHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceHandle").finish()
    }
}

impl<T: BufferContents + Send> Deref for ResourceHandle<T> {
    type Target = Subbuffer<T>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
impl<T: BufferContents + Send> DerefMut for ResourceHandle<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}
impl<T: BufferContents + Send> Drop for ResourceHandle<T> {
    fn drop(&mut self) {
        let alloc = self.alloc;
        let manager = self.manager.clone();
        tokio::spawn(async move {
            manager.allocator.lock().await.free(alloc);
        });
    }
}

impl<T: BufferContents + Send> ResourceManager<T> {
    pub fn new(buffer_size: u64) -> Arc<Self> {
        Self {
            buffer_size,
            buffers: Vec::new().into(),
            allocator: Allocator::new(u32::MAX - 8).into(),
        }
        .into()
    }
    pub async fn alloc(
        self: &Arc<Self>,
        ctx: &Context,
        data: T,
    ) -> Result<ResourceHandle<T>, Validated<AllocateBufferError>> {
        let alloc = self.allocator.lock().await.allocate(1).unwrap();
        let buffer_index = alloc.offset as u64 / self.buffer_size;
        let buffer_offset = alloc.offset as u64 % self.buffer_size;
        let mut buffers_read = self.buffers.read().await;
        let new_buffers = (buffer_index + 1).saturating_sub(buffers_read.len() as u64);
        if new_buffers > 0 {
            drop(buffers_read);
            let mut buffers_write = self.buffers.write().await;
            for _ in 0..new_buffers {
                let buf = match Buffer::new_slice(
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
                    self.buffer_size,
                ) {
                    Ok(v) => v,
                    Err(err) => {
                        self.allocator.lock().await.free(alloc);
                        tracing::error!("failed to alloc gtransform buffer: {err}");
                        return Err(err);
                    }
                };
                buffers_write.push(buf);
            }
            drop(buffers_write);
            buffers_read = self.buffers.read().await;
        }

        let buffer = buffers_read[buffer_index as usize]
            .clone()
            .index(buffer_offset);
        *buffer.write().unwrap() = data;
        Ok(ResourceHandle {
            buffer,
            manager: self.clone(),
            alloc,
        })
    }
}
