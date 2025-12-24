#version 460

#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_ARB_shader_draw_parameters: require

#include "common.h"

layout(push_constant) uniform block {
    Globals globals;
};

layout(set = 0, binding = 4) readonly buffer Transforms {
    Transform transforms[];
};

layout (set = 0, binding = 1) readonly buffer Vertices {
    Vertex vertices[];
};

layout(location = 0) out vec3 Normal;
layout(location = 1) out vec2 TexCoords;

void main() {
    Transform transform = transforms[gl_DrawIDARB];
    gl_Position = globals.proj * globals.view * transform.model * vec4(vertices[gl_VertexIndex].vx, vertices[gl_VertexIndex].vy, vertices[gl_VertexIndex].vz, 1.0);
    // decompress normals
    vec3 decompressed = vec3(vertices[gl_VertexIndex].nx, vertices[gl_VertexIndex].ny, vertices[gl_VertexIndex].nz) / 255.0 * 2.0 - 1.0;
    Normal = decompressed;
    TexCoords = vec2(vertices[gl_VertexIndex].tu, vertices[gl_VertexIndex].tv);
}