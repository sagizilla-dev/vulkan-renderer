struct Vertex {
    float16_t vx, vy, vz, vw;
    uint8_t nx, ny, nz, nw;
    float16_t tu, tv;
};

// double indexing is important here
// having 126*3 uint32_t indices makes up for 126*3*4 = 1512 bytes
// having 126*3 uint8_t sub indices and 64 uint32_t global indices make up for 126*3 + 64*4 = 634 bytes
// another important feature of double indexing is vertex de duplication
// meaning the global vertex at any index is stored only once in the vertex buffer
struct Meshlet {
    vec4 cone;              // offset 0, alignment 16
    vec4 coneApex;          // offset 16, alignment 16
    uint32_t vertices[64];  // offset 32, alignment 4
    uint8_t indices[126*3]; // offset 288, alignment 1
    uint8_t triangleCount;  // offset 666, alignment 1
    uint8_t vertexCount;    // offset 667, alignment 1
    float padding;          // offset 668, alignment 4
};
// largest alignment is 16

// alignment is important
// a variable with N-byte alignment must start at a memory address that's a multiple of N
// whole struct's size must be divisible by its largest alignment