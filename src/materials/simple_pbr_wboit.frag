#version 460

#extension GL_GOOGLE_include_directive : enable

#include "./simple_pbr.glsl"
#include "../pipeline_pieces/wboit.glsl"

layout(location = 0) flat in uint draw_id;
layout(location = 1) in vec4 vertex_color;
layout(location = 2) in vec2 vertex_uv;
layout(location = 3) in vec3 vertex_normal;

layout(location = 1) out vec4 accum;
layout(location = 2) out float revealage;

void main() {
    vec4 color = calc_color(draw_id, vertex_color, vertex_uv, vertex_normal);
	wboit(gl_FragCoord, color, accum, revealage);
}
