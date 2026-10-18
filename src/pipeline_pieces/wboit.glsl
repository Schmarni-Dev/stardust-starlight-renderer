#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "./common.glsl"

layout(set = 1, binding = 0) uniform readonly sampler cdf_sampler;
layout(set = 1, binding = 1) uniform readonly texture3D cdf;
// TODO: should these be per view?
layout(scalar, set = 0, binding = 2) writeonly buffer _histogram {
    // hope this is an f32
    float data[];
} histogram;
// layout(input_attachment_index = 1, set = 1, binding = 2) uniform readonly subpassInput inputDepth;

// layout(buffer_reference, scalar) buffer histogram {
//     uint32_t data;
// };

const uint32_t BINS = 32;
const uvec2 HISTOGRAM_SIZE = uvec2(1280, 720);

float optical_depth(float alpha) {
    return -log(max(1.0 - alpha, 1e-6));
}
// TODO: play around with this to better allocate the depth bins
float bin_depth(vec4 frag_pos) {
    frag_pos.z
}

/// builds the next frames histogram and returns the value to be added to the wboit accum buffer
WboitOut wboit(vec4 frag_pos, vec4 color) {
    float optical_depth = optical_depth(color.w);
    float bin_depth = bin_depth(frag_pos);

    uint32_t bin_index = uint32_t(bin_depth * float(BINS));
    vec2 normalized_frag_pos = frag_pos.xy / vec2(pc.resolution);
    uvec2 histogram_pos = uvec2(normalized_frag_pos * vec2(HISTOGRAM_SIZE));
    // TODO: maybe average across multiple bins?
    uint index = (xy.x * size.y) + (xy.y * size.z) + bin;
    atomicAdd(histogram.data[index], optical_depth);

    float cdf_value = texture(sampler3D(cdf, cdf_sampler), vec3(frag_pos.xy, bin_depth));
    // TODO: make sure this weight is correct
    float weight = exp(-cdf_value);
    vec4 accum = color * weight * 64.0;
    return WboitOut(accum, optical_depth);
    ;
}
struct WboitOut {
    vec4 accum;
    float revealage;
}
