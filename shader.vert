#version 450

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp;

struct Vertex {
    float vx, vy, vz;
    float nx, ny, nz;
    float tu, tv;
};
layout (set = 0, binding = 2) readonly buffer Vertices {
    Vertex vertices[];
};

layout(location = 0) out vec3 Normal;
layout(location = 1) out vec2 TexCoords;

void main() {
    gl_Position = mvp.proj * mvp.view * mvp.model * vec4(vertices[gl_VertexIndex].vx, vertices[gl_VertexIndex].vy, vertices[gl_VertexIndex].vz, 1.0);
    Normal = vec3(vertices[gl_VertexIndex].nx, vertices[gl_VertexIndex].ny, vertices[gl_VertexIndex].nz);
    TexCoords = vec2(vertices[gl_VertexIndex].tu, vertices[gl_VertexIndex].tv);
}