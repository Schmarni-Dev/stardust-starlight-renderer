#ifndef STARLIGHT_COMMON
#define STARLIGHT_COMMON
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(buffer_reference, scalar) readonly buffer Transform {
    mat4 data;
};

struct Metadata {
    uint64_t material;
    uint64_t data;
    Transform global_transform;
    uint8_t data_type;
};

vec4 blend_premul(vec4 base, vec4 new) {
    return vec4((base * (1 - new.a)) + new);
}

layout(buffer_reference, scalar) buffer ImageRef {
    uint32_t image_index;
    uint32_t sampler_index;
    uint8_t flags;
};

// const uint8_t IMAGE_REF_FLAGS_SOFTWARE_SRGB = 1 << 0;
// const uint8_t IMAGE_REF_FLAGS_ALPHA_REMULTIPLICATION = 1 << 1;
// const uint8_t IMAGE_REF_FLAGS_ALPHA_MULTIPLICATION = 1 << 2;

layout(set = 0, binding = 0) uniform sampler samplers[];
layout(set = 0, binding = 1) uniform texture2D textures[];

float srgb_eotf_conv(float component) {
    if (component <= 0.04045)
        return component / 12.92;
    else
        return pow(((component + 0.055) / 1.055), 2.4);
}
vec4 srgb_eotf(vec4 color) {
    return vec4(srgb_eotf_conv(color.r), srgb_eotf_conv(color.g), srgb_eotf_conv(color.b), color.a);
}

vec4 sample_image(ImageRef image, vec2 uv) {
    vec4 base_sample = texture(sampler2D(textures[nonuniformEXT(image.image_index)], samplers[nonuniformEXT(image.sampler_index)]), uv);
    if ((image.flags & 1) != 0)
        base_sample = vec4(base_sample.rgb / base_sample.a, base_sample.a);
    if ((image.flags & 1 << 1) != 0)
        base_sample = srgb_eotf(base_sample);
    if ((image.flags & 1 << 2) != 0)
        base_sample = vec4(base_sample.rgb * base_sample.a, base_sample.a);

    return base_sample;
}

layout(buffer_reference, scalar) readonly buffer MetadataBuf {
    Metadata[] data;
};

struct CamInfo {
    mat4 proj_mat;
};
layout(buffer_reference, scalar) readonly buffer CamBuf {
    CamInfo[] data;
};

layout(push_constant) uniform _pc {
    CamBuf cam_ptr;
    MetadataBuf metadata_ptr;
	u32vec2 resolution;
} pc;
#endif
