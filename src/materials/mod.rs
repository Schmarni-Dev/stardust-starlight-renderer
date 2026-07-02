use std::fmt::Debug;
use vulkano::DeviceSize;

pub mod simple_pbr;

pub trait MaterialHandle: Debug + 'static {
    /// # Safety:
    /// the returned BDA has to stay valid as long as this handle is held
    unsafe fn get_device_ptr(&self) -> DeviceSize;
}
