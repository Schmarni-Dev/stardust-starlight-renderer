use std::{fmt::Debug, sync::Arc};
use vulkano::DeviceSize;

use crate::resources::{ReadOp, image::ImageReadOp};

pub mod simple_pbr;
pub mod image_common;

pub trait MaterialHandle: Debug + 'static {
    /// # Safety:
    /// the returned BDA has to stay valid as long as this handle is held
    unsafe fn get_device_ptr(&self) -> DeviceSize;
    fn get_read_ops(&self) -> impl IntoIterator<Item=Arc<dyn ReadOp>>;
}
