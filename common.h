struct Vertex {
    float16_t vx, vy, vz;
    uint8_t nx, ny, nz, nw;
    float16_t tu, tv;
};

// double indexing is important here
// having 126*3 uint32_t indices makes up for 126*3*4 = 1512 bytes
// having 126*3 uint8_t sub indices and 64 uint32_t global indices make up for 126*3 + 64*4 = 634 bytes
// another important feature of double indexing is vertex de duplication
// meaning the global vertex at any index is stored only once in the vertex buffer
struct Meshlet {
    uint32_t vertices[64];
    uint8_t indices[126*3];
    uint8_t triangleCount;
    uint8_t vertexCount;
};