#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_GOOGLE_include_directive : enable

#include "../pipeline_pieces/common.glsl"

layout(buffer_reference, scalar) buffer Material {
    vec4 color;
    ImageRef image_ref;
};

float remap(float v, float inMin, float inMax, float outMin, float outMax) {
    float t = (v - inMin) / (inMax - inMin);
    return outMin + (outMax - outMin) * t;
}

vec4 calc_color(uint draw_id, vec4 vertex_color, vec2 vertex_uv, vec3 vertex_normal) {
    Metadata metadata = pc.metadata_ptr.data[draw_id];
    Material material = Material(metadata.material);
    vec4 unlit_color = blend_premul(vertex_color, material.color);
    if (uint64_t(material.image_ref) != 0) {
        vec4 image_sample = sample_image(material.image_ref, vertex_uv);
        unlit_color = blend_premul(unlit_color, image_sample);
    }
    vec3 ref_dir = normalize(vec3(1.0, 1.0, 1.0));
    return vec4(unlit_color.rgb * remap(dot(ref_dir, normalize(vertex_normal)), -1.0, 1.0, 0.4, 1.0), unlit_color.a);
    // float depth = 1 - gl_FragCoord.z;

    // return vec4(exp(gl_FragCoord.z), 0, 0, 1);
}
