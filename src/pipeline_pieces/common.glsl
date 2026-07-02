#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

layout(buffer_reference, scalar) readonly buffer Transform {
    mat4 data;
};

struct Metadata {
    uint64_t material;
    uint64_t data;
    Transform global_transform;
    uint8_t data_type;
};

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
} pc;
