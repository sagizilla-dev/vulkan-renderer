struct Vertex {
    float vx, vy, vz, vw;
    uint8_t nx, ny, nz, nw;
    float16_t tu, tv;
};

struct Globals {
    mat4 view;
    mat4 proj;
    uint meshletCount;
};

struct Transform {
    mat4 model;
};

// double indexing is important here
// having 126*3 uint32_t indices makes up for 126*3*4 = 1512 bytes
// having 126*3 uint8_t sub indices and 64 uint32_t global indices make up for 126*3 + 64*4 = 634 bytes
// another important feature of double indexing is vertex de duplication
// meaning the global vertex at any index is stored only once in the vertex buffer
// using 124 triangles instead of 126 triangles benefits in case we decide to
// pack 4 uint8_t indices into 1 uint32_t index, which allows us to use
// writePackedIndices4x8NV(...) function in GLSL
struct Meshlet {
    vec4 cone;              // offset 0,   alignment 16
    vec4 coneApex;          // offset 16,  alignment 16
    uint32_t dataOffset;    // offset 32,  alignment 4
    uint8_t triangleCount;  // offset 36,  alignment 1
    uint8_t vertexCount;    // offset 37,  alignment 1
    uint8_t padding[10];    // offset 38,  alignment 1
};
// largest alignment is 16

// alignment is important
// a variable with N-byte alignment must start at a memory address that's a multiple of N
// whole struct's size must be divisible by its largest alignment