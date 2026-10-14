use std::{fmt::Debug, sync::Arc};

use vulkano::image::{Image, ImageLayout, view::ImageView};

use crate::{
    images::ImageRef,
    resources::{ReadOp, WriteOp},
};

// TODO: this probably needs image layout transitions, somehow, idk how to do that tho, it kinda
// needs predicting when the last frame with an image is, which is .... not easy
pub trait ImageReadOp: ReadOp {
    fn image(&self) -> &Arc<Image>;
    fn image_view(&self) -> &Arc<ImageView>;
    fn image_ref(&self) -> &ImageRef;
}
// TODO: image metadata for writing, software srgb and stuff, probably an image ref or something?
pub trait ImageWriteOp: WriteOp {
    fn image(&self) -> &Arc<Image>;
    fn image_view(&self) -> &Arc<ImageView>;
    fn current_layout(&self) -> ImageLayout;
    fn target_layout(&self) -> ImageLayout;
}
impl Debug for dyn ImageReadOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("dyn ImageReadOp")
            .field("image", self.image())
            .field("image_view", self.image_view())
            .field("image_ref", self.image_ref())
            .finish()
    }
}
