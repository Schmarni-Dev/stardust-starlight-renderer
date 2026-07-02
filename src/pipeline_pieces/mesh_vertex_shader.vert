#version 460

#extension GL_EXT_multiview : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 color;

#include "./common.glsl"

layout(location = 0) flat out uint o_draw_id;
layout(location = 1) out vec4 o_color;
layout(location = 2) out vec2 o_uv;
layout(location = 3) out vec3 o_normal;

void main() {
    Metadata metadata = pc.metadata_ptr.data[gl_InstanceIndex];
    mat4 combined_mat = pc.cam_ptr.data[gl_ViewIndex].proj_mat * metadata.global_transform.data;
    gl_Position = combined_mat * vec4(pos, 1.0);
    o_color = color;
    o_uv = uv;
    mat3 normal_mat = transpose(inverse(mat3(metadata.global_transform.data)));
    o_normal = normal_mat * normal;
    o_draw_id = gl_InstanceIndex;
}
