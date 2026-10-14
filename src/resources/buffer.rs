use std::{
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use timeline_syncobj::timeline_syncobj::TimelineSyncObj;
use vulkano::{
    DeviceAddress, NonNullDeviceAddress, ValidationError,
    buffer::{BufferContents, BufferReadGuard, BufferWriteGuard, Subbuffer},
    device::DeviceOwned,
    sync::HostAccessError,
};

use crate::{
    Context,
    resources::{ReadOp, WriteOp},
};

pub trait BufferReadOp<T: BufferContents + ?Sized>: ReadOp {
    fn bytes(&self) -> &Subbuffer<[u8]>;
    fn inner(&self) -> &Subbuffer<T>;
    fn bda<'a>(&'a self) -> Option<Bda<'a, T>>;
}

pub trait BufferWriteOp: WriteOp {}

#[derive(Debug)]
// #[repr(C)]
pub struct StarlightSubBuffer<T: ?Sized + BufferContents> {
    buf: Subbuffer<T>,
    timeline: Arc<TimelineSyncObj>,
    next_acquire: Arc<AtomicU64>,
}
impl<T: ?Sized + BufferContents> StarlightSubBuffer<T> {
    pub fn from_subbuffer(ctx: &Context, buf: Subbuffer<T>) -> Self {
        debug_assert!(ctx.device == buf.device().clone());
        let timeline = Arc::new(TimelineSyncObj::new(&ctx.drm_render_node).unwrap());
        unsafe {
            timeline.signal(1).unwrap();
        }
        Self {
            buf,
            timeline,
            next_acquire: Arc::new(AtomicU64::new(1)),
        }
    }
    pub fn new_read_op(&self) -> Arc<StarlightSubBufferReadOp<T>> {
        let acquire = self.next_acquire.fetch_add(1, Ordering::Relaxed);
        let release = acquire + 1;
        Arc::new(StarlightSubBufferReadOp {
            buf: self.buf.clone(),
            timeline: self.timeline.clone(),
            acquire,
            release,
        })
    }
    pub fn new_wite_op(&self) -> Box<StarlightSubBufferWriteOp<T>> {
        let acquire = self.next_acquire.fetch_add(1, Ordering::Relaxed);
        let release = acquire + 1;
        Box::new(StarlightSubBufferWriteOp {
            buf: self.buf.clone(),
            timeline: self.timeline.clone(),
            acquire,
            release,
        })
    }
    pub fn into_bytes(self) -> StarlightSubBuffer<[u8]> {
        StarlightSubBuffer {
            buf: self.buf.into_bytes(),
            timeline: self.timeline.clone(),
            next_acquire: self.next_acquire.clone(),
        }
    }
    pub fn device_address<'a>(&'a self) -> Result<Bda<'a, T>, Box<ValidationError>> {
        Ok(Bda(self.buf.device_address()?, PhantomData, PhantomData))
    }
    /// # Safety
    /// this returns the raw representation of the underlying subbuffer, its your job to ensure
    /// access is synchronized properly
    pub unsafe fn inner(&self) -> &Subbuffer<T> {
        &self.buf
    }
    // pub fn as_bytes(&self) -> &StarlightSubBuffer<[u8]> {
    //     assert_eq!(size_of::<StarlightSubBuffer<T>>(), size_of::<StarlightSubBuffer<[u8]>>());
    //     assert_eq!(align_of::<StarlightSubBuffer<T>>(), align_of::<StarlightSubBuffer<[u8]>>());
    //
    //     // SAFETY: All `Subbuffer`s share the same layout.
    //     unsafe { mem::transmute::<&StarlightSubBuffer<T>, &StarlightSubBuffer<[u8]>>(self) }
    // }
}
impl<T: ?Sized + BufferContents> Clone for StarlightSubBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
            timeline: self.timeline.clone(),
            next_acquire: self.next_acquire.clone(),
        }
    }
}
#[derive(Debug)]
pub struct StarlightSubBufferReadOp<T: ?Sized + BufferContents> {
    buf: Subbuffer<T>,
    timeline: Arc<TimelineSyncObj>,
    acquire: u64,
    release: u64,
}
impl<T: ?Sized + BufferContents> StarlightSubBufferReadOp<T> {
    pub async fn read(&self) -> Result<BufferReadGuard<'_, T>, HostAccessError> {
        self.timeline.wait_async(self.acquire).unwrap().await;
        self.buf.read()
    }
}
impl<T: ?Sized + BufferContents> BufferReadOp<T> for StarlightSubBufferReadOp<T> {
    fn bytes(&self) -> &Subbuffer<[u8]> {
        self.buf.as_bytes()
    }

    fn bda<'a>(&'a self) -> Option<Bda<'a, T>> {
        Some(Bda(
            self.buf.device_address().ok()?,
            PhantomData,
            PhantomData,
        ))
    }

    fn inner(&self) -> &Subbuffer<T> {
        &self.buf
    }
}
impl<T: ?Sized + BufferContents> Drop for StarlightSubBufferReadOp<T> {
    fn drop(&mut self) {
        unsafe { self.timeline.signal(self.release).unwrap() }
    }
}
impl<T: ?Sized + BufferContents> ReadOp for StarlightSubBufferReadOp<T> {
    fn timeline(&self) -> &TimelineSyncObj {
        &self.timeline
    }

    fn acquire_point(&self) -> u64 {
        self.acquire
    }
}
pub struct StarlightSubBufferWriteOp<T: ?Sized + BufferContents> {
    buf: Subbuffer<T>,
    timeline: Arc<TimelineSyncObj>,
    acquire: u64,
    release: u64,
}
impl<T: ?Sized + BufferContents> StarlightSubBufferWriteOp<T> {
    pub async fn write(
        self,
        callback: impl for<'a> FnOnce(BufferWriteGuard<'a, T>),
    ) -> Result<(), HostAccessError> {
        self.timeline.wait_async(self.acquire);
        let v = self.buf.write()?;
        callback(v);
        unsafe { self.timeline.signal(self.release) };
        Ok(())
    }
}
impl<T: ?Sized + BufferContents> BufferWriteOp for StarlightSubBufferWriteOp<T> {}
impl<T: ?Sized + BufferContents> WriteOp for StarlightSubBufferWriteOp<T> {
    fn timeline(&self) -> &TimelineSyncObj {
        &self.timeline
    }

    fn acquire_point(&self) -> u64 {
        self.acquire
    }

    fn release_point(&self) -> u64 {
        self.release
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct Bda<'a, T: BufferContents + ?Sized>(
    NonNullDeviceAddress,
    PhantomData<T>,
    PhantomData<&'a ()>,
);

impl<T: BufferContents + ?Sized> Clone for Bda<'_, T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone(), self.2.clone())
    }
}
impl<T: BufferContents + ?Sized> Copy for Bda<'_, T> {}

// TODO: can this 'static lifetime be relaxed?
// # Safety
// this should be fine, its a NonZeroU64 under the hood
unsafe impl<T: BufferContents + ?Sized> bytemuck::ZeroableInOption for Bda<'static, T> {}
// # Safety
// bytemuck explicitly states that pointers are not Pod, but we kinda need it, and we don't deref
// them on the cpu anyway
unsafe impl<T: BufferContents + ?Sized> bytemuck::PodInOption for Bda<'static, T> {}
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct NullableBDA<'a, T: BufferContents + ?Sized>(
    DeviceAddress,
    PhantomData<T>,
    PhantomData<&'a ()>,
);
impl<'a, T: BufferContents + ?Sized> From<Bda<'a, T>> for NullableBDA<'a, T> {
    fn from(value: Bda<'a, T>) -> Self {
        Self(value.0.get(), PhantomData, PhantomData)
    }
}
impl<'a, T: BufferContents + ?Sized> TryFrom<NullableBDA<'a, T>> for Bda<'a, T> {
    type Error = ();

    fn try_from(value: NullableBDA<'a, T>) -> Result<Self, Self::Error> {
        Ok(Self(
            NonNullDeviceAddress::new(value.0).ok_or(())?,
            PhantomData,
            PhantomData,
        ))
    }
}
impl<T: BufferContents + ?Sized> Bda<'_, T> {
    /// # Safety
    /// this casts away the lifetime, you have to ensure that the underlying buffer stays valid while
    /// the returned address could be used
    pub unsafe fn to_static(&self) -> Bda<'static, T> {
        Bda(self.0, PhantomData, PhantomData)
    }
}
impl<T: BufferContents + ?Sized> Bda<'static, T> {
    pub fn as_raw(&self) -> NonNullDeviceAddress {
        self.0
    }
}
