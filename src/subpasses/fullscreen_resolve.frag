#version 460

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput inputWboitAccum;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput inputWboitRevealage;

layout(location = 3) out vec4 color;

void main() {
    vec4 opaque = subpassLoad(inputColor);
    vec4 wboit_accum = subpassLoad(inputWboitAccum);
    float wboit_alpha = 1.0 - exp(-subpassLoad(inputWboitRevealage).r);
    vec4 wboit = vec4(wboit_accum.rgb / max(wboit_accum.a, 1e-5) * wboit_alpha, wboit_alpha);
    // float d = 1 - subpassLoad(inputDepth).r;
    // color = vec4(d,d,d,1.0);
    vec4 final_color = vec4((opaque * (1 - wboit.a)) + wboit);
    // color = vec4(wboit_accum.rgb, 1.0);
    color = final_color;
}
