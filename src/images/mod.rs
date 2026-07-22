use vulkano::image::sampler::ycbcr::{ChromaLocation, SamplerYcbcrModelConversion, SamplerYcbcrRange};

pub mod dma_image;
pub mod transient_attachment;

pub struct ImageRef {
    image_index: u32,
    sampler_index: u32,
    /// if true SRGB data is deliverd to the shader and has to be manually be converted to linear
    requires_software_transform: bool,
    /// if true requires unpremultiplying into a float before applying a manual transfer function
    /// and remultiplying after applying the transfer function (premultiplication being applied
    /// directly to the SRGB value instead of in linear space)
    requires_remultiplication: bool,
    /// if true the image uses straight alpha
    requires_multiplication: bool,
}

pub trait ImageRefSource {}

pub struct YcbcrMetadata {
    coefficients: SamplerYcbcrModelConversion,
    range: SamplerYcbcrRange,
    chroma_location_x: ChromaLocation,
    chroma_location_y: ChromaLocation,
}

/// Alpha mode used by this Dmatex
pub enum AlphaMode {
    /// The stored color values are premultiplied instead of being premuliplied in linear light.
    /// Wayland uses this by default, consider using PremultipliedOptical instead if you can,
    /// as its the correct solution for blending.
    PremultipliedElectrical,
    /// The colors where premultiplied in linear light and then encoded into their storage values.
    /// This is the correct way to handle premultiplication, but may be more expensive if you do everything in electrical values (but blending is only correct in linear anyway)
    PremultipliedOptical,
}
