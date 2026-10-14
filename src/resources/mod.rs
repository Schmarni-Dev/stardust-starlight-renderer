pub mod image;
pub mod buffer;

use std::any::TypeId;

use timeline_syncobj::timeline_syncobj::TimelineSyncObj;

pub trait Resource {}
/// should use Drop to release
pub trait ReadOp: Send + Sync {
    fn timeline(&self) -> &TimelineSyncObj;
    fn acquire_point(&self) -> u64;
}
pub trait WriteOp: Send {
    fn timeline(&self) -> &TimelineSyncObj;
    fn acquire_point(&self) -> u64;
    fn release_point(&self) -> u64;
}

// PLAN
//
// Pimitives:
// Arc<dyn ReadOp>, a shareable handle to a read op on a resource
// Box<dyn WriteOp>, a non-clonable handle to a mutating op, for an actual gpu submission this is
// moved into the function/method recording the action
//
//
// Private Resources:
// auto assign aqcuire and release points, create Arc<dyn ReadResource>
// keep returning same Arc<dyn ReadResource> until a mutable operation is scheduled (gpu side or cpu
// side)
// upon a mut op being scheduled advance the acquire and release points
//
// Shared Resources:
// has something like T::read_op(acquire_point: u64, release_point: u64) -> Arc<TSpecificReadOp>
// has something like T::write_op(acquire_point: u64, release_point: u64) -> Box<TSpecificWriteOp>
// they might need to be unsafe, since else memory corruption might happen (2 writes at the same
// time)
//
// TODO: how does this work with secondary command buffers? how does this work with transient
// attachments, would they just go through a different system entirely?

// Transient Attachments stuff:
// RenderAttachment enum:
// Image(Box<dyn ImageWriteOp>),
// TransientAttachment(TransientAttachmentReadOp)
//
// TransientAttachmentReadOp is returned by a method that records the local_read barrier (i think,
// idk how to quite do that)
// idk, just treat transient attachments as unsafe for now, nice and safe abstraction later
