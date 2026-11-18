#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "./common.glsl"

layout(set = 1, binding = 0) uniform sampler cdf_sampler;
layout(set = 1, binding = 1) uniform texture3D cdf;
// TODO: should these be per view?
layout(scalar, set = 1, binding = 2) buffer _histogram {
    // 16.16 fixed point, RDNA1 has no float atomic add on buffers
    uint data[];
} histogram;
const float HISTOGRAM_SCALE = 65536.0;

float optical_depth(float alpha) {
    return -log(max(1.0 - alpha, 1e-6));
}
// TODO: play around with this to better allocate the depth bins
float bin_depth(vec4 frag_pos) {
    return frag_pos.z;
}

/// builds the next frames histogram and returns the value to be added to the wboit accum buffer
void wboit(vec4 frag_pos, vec4 color, out vec4 accum, out float revealage) {
    ivec3 size = textureSize(sampler3D(cdf, cdf_sampler), 0);
    float optical_depth = optical_depth(color.w);
    float bin_depth = bin_depth(frag_pos);

    uint32_t bin_index = min(uint32_t(bin_depth * float(size.z)), uint32_t(size.z) - 1u);
    vec2 normalized_frag_pos = frag_pos.xy / vec2(pc.resolution);
    uvec2 xy = min(uvec2(normalized_frag_pos * vec2(size.xy)), uvec2(size.xy) - 1u); 
	// TODO: maybe average across multiple bins?
    // bin-major so the cdf build reads coalesce across neighbouring pixels
    uint index = (bin_index * size.y + xy.y) * size.x + xy.x;
    atomicAdd(histogram.data[nonuniformEXT(index)], uint(optical_depth * HISTOGRAM_SCALE));

    float cdf_value = texture(sampler3D(cdf, cdf_sampler), vec3(normalized_frag_pos, bin_depth)).r;
    float weight = exp(-cdf_value);
    accum = color * weight;
    revealage = optical_depth;
}
