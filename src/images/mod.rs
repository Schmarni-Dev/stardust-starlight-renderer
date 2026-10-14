use std::sync::Arc;

use glam::{UVec2, UVec3};
use offset_allocator::{Allocation, Allocator};
use tokio::sync::Mutex;
use vulkano::{
    buffer::BufferContents,
    descriptor_set::{DescriptorSet, WriteDescriptorSet, layout::DescriptorSetLayout},
    image::{
        Image, ImageType,
        sampler::{
            Sampler,
            ycbcr::{ChromaLocation, SamplerYcbcrModelConversion, SamplerYcbcrRange},
        },
        view::ImageView,
    },
    pipeline::{PipelineBindPoint, PipelineLayout},
    shader::ShaderStages,
};

use crate::{
    Context,
    command_buffer::StarlightCommandBuffer,
    materials::image_common::get_descriptor_set_layout_info,
    resource_manager::{ResourceHandle, ResourceManager},
    resources::buffer::Bda,
};

pub mod dma_image;
pub mod local_image;
pub mod openxr_attachment;
pub mod transient_attachment;
pub mod wsi_swapchain;

#[derive(Debug, BufferContents, Clone, Copy)]
#[repr(C)]
pub struct ImageRefData {
    // TODO: add two more indices for ycbcr planes
    image_index: u32,
    sampler_index: u32,
    flags: u8,
}
impl ImageRefData {
    /// if true SRGB data is deliverd to the shader and has to be manually be converted to linear
    pub const SOFTWARE_SRGB: u8 = 1 << 0;
    /// if set requires unpremultiplying into a float before applying a manual transfer function
    /// and remultiplying after applying the transfer function (premultiplication being applied
    /// directly to the SRGB value instead of in linear space)
    pub const ALPHA_UNMULTIPLICATION: u8 = 1 << 1;
    /// if set the image uses straight alpha
    pub const ALPHA_MULTIPLICATION: u8 = 1 << 2;
}
pub struct ImageRefInfo {
    /// if true SRGB data is deliverd to the shader and has to be manually be converted to linear
    requires_software_srgb: bool,
    /// if true requires unpremultiplying into a float before applying a manual transfer function
    /// and remultiplying after applying the transfer function (premultiplication being applied
    /// directly to the SRGB value instead of in linear space)
    requires_unmultiplication: bool,
    /// if true the image uses straight alpha
    requires_multiplication: bool,
}

pub trait ImageRefSource {
    fn get_image_ref_info(&self) -> ImageRefInfo;
    fn get_image(&self) -> Arc<Image>;
    fn get_image_view(&self) -> Arc<ImageView>;
}

#[derive(Clone, Copy, Debug)]
pub struct YcbcrMetadata {
    coefficients: SamplerYcbcrModelConversion,
    range: SamplerYcbcrRange,
    chroma_location_x: ChromaLocation,
    chroma_location_y: ChromaLocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Alpha mode used by this Dmatex
pub enum AlphaMode {
    /// The stored color values are premultiplied instead of being premuliplied in linear light.
    /// Wayland uses this by default, consider using PremultipliedOptical instead if you can,
    /// as it's the correct solution for blending.
    PremultipliedElectrical,
    /// The colors where premultiplied in linear light and then encoded into their storage values.
    /// This is the correct way to handle premultiplication, but may be more expensive if you do everything in electrical values (but blending is only correct in linear anyway)
    PremultipliedOptical,
    Unpremmultiplied,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageResolution {
    D1(u32),
    D2(UVec2),
    D3(UVec3),
}

impl ImageResolution {
    pub fn extents(&self) -> [u32; 3] {
        match *self {
            // pretty sure this actually has to be 0 instead of 1?
            ImageResolution::D1(x) => [x, 1, 1],
            ImageResolution::D2(UVec2 { x, y }) => [x, y, 1],
            ImageResolution::D3(UVec3 { x, y, z }) => [x, y, z],
        }
    }
    pub fn image_type(&self) -> ImageType {
        match *self {
            ImageResolution::D1(_) => ImageType::Dim1d,
            ImageResolution::D2(_) => ImageType::Dim2d,
            ImageResolution::D3(_) => ImageType::Dim3d,
        }
    }
}

pub struct SamplerRef {
    sampler: Arc<Sampler>,
    alloc: Allocation,
    allocator: Arc<Mutex<Allocator>>,
}

impl std::fmt::Debug for SamplerRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerRef")
            .field("sampler", &self.sampler)
            .field("alloc", &self.alloc.offset)
            .field("allocator", &self.allocator)
            .finish()
    }
}
impl Drop for SamplerRef {
    fn drop(&mut self) {
        let alloc = self.alloc.clone();
        let sampler = self.sampler.clone();
        let allocator = self.allocator.clone();
        tokio::spawn(async move {
            allocator.lock().await.free(alloc);
            let _sampler = sampler;
        });
    }
}
pub struct ImageManager {
    descriptor_set: DescriptorSet,
    // images: Vec<Arc<ImageView>>,
    refs: Arc<ResourceManager<ImageRefData>>,
    sampler_alloc: Arc<Mutex<Allocator>>,
}
impl ImageManager {
    pub fn new(ctx: &Context, stages: ShaderStages) -> ImageManager {
        let layout = DescriptorSetLayout::new(
            ctx.device.clone(),
            get_descriptor_set_layout_info(ctx, stages),
        )
        // TODO: return this error instead
        .unwrap();
        let count = layout.variable_descriptor_count();
        let descriptor_set =
            DescriptorSet::new_variable(ctx.descriptor_set_alloc.clone(), layout, count, [], [])
                // TODO: return this error
                .unwrap();
        let descriptor_set = Arc::try_unwrap(descriptor_set).unwrap();
        Self {
            // images: Vec::new(),
            refs: ResourceManager::new(64),
            descriptor_set,
            sampler_alloc: Arc::new(Mutex::new(Allocator::new(256))),
        }
    }
    pub async fn alloc_sampler(&mut self, sampler: Arc<Sampler>) -> Option<Arc<SamplerRef>> {
        let alloc = self.sampler_alloc.lock().await.allocate(1)?;
        self.descriptor_set
            .update(
                [WriteDescriptorSet::sampler_array(
                    0,
                    alloc.offset,
                    [sampler.clone()],
                )],
                [],
            )
            .unwrap();
        Some(
            SamplerRef {
                sampler,
                alloc,
                allocator: self.sampler_alloc.clone(),
            }
            .into(),
        )
    }
    pub async fn alloc(
        &mut self,
        ctx: &Context,
        image: &impl ImageRefSource,
        sampler: Arc<SamplerRef>,
    ) -> Arc<ImageRef> {
        // TODO: validate that sampler comes from this instance
        let info = image.get_image_ref_info();
        let image_view = image.get_image_view();
        let mut flags = 0;
        if info.requires_software_srgb {
            flags |= ImageRefData::SOFTWARE_SRGB;
        }
        if info.requires_unmultiplication {
            flags |= ImageRefData::ALPHA_UNMULTIPLICATION;
        }
        if info.requires_multiplication {
            flags |= ImageRefData::ALPHA_MULTIPLICATION;
        }
        let alloc = self
            .refs
            .alloc(
                ctx,
                ImageRefData {
                    image_index: 0,
                    sampler_index: 0,
                    flags,
                },
            )
            .await
            .unwrap();
        alloc
            .new_wite_op()
            .write(|mut v| v.image_index = alloc.alloc_index())
            .await
            .unwrap();
        self.descriptor_set
            .update(
                [WriteDescriptorSet::image_view_array(
                    1,
                    alloc.alloc_index(),
                    [image_view],
                )],
                [],
            )
            .unwrap();
        ImageRef {
            ref_data: alloc,
            sampler,
        }
        .into()
    }
    // TODO: make sure that the descriptor set is not destroyed before the gpu finishes
    pub fn bind(
        &self,
        cmds: &mut StarlightCommandBuffer,
        bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
    ) -> Result<(), Box<vulkano::ValidationError>> {
        unsafe {
            cmds.inner().bind_descriptor_sets(
                bind_point,
                layout,
                0,
                &[self.descriptor_set.as_raw()],
                &[],
            )?;
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct ImageRef {
    sampler: Arc<SamplerRef>,
    ref_data: ResourceHandle<ImageRefData>,
}
impl ImageRef {
    pub fn bda<'a>(&'a self) -> Bda<'a, ImageRefData> {
        self.ref_data.device_address().unwrap()
    }
}
