#pragma once
#include "utils.hpp"

struct QueueFamilies {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> transferFamily;
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value() && transferFamily.has_value();
    }
};
struct SurfaceDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> surfaceFormats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    // ideally we could have uint16_t/float16 for vertex positions, 
    // and it does work, but then the number of meshlets that we can cull with cones
    // decreases a lot. This is a result of cones getting too wide, so we can't cull them
    // and end up having to rely on backface culling later.
    // in fact, having half precision may even collapse a triangle into a 0-area triangle
    // which is supposed to break the program during glm::normalize(...), but for some reason
    // it doesn't happen even with high poly meshes like dragon.obj or buddha.obj
    // matter of fact, high poly meshes shouldn't really benefit from cone culling at all
    // since mesh details are rendered using pure geometry, so we are supposed to end up
    // with too wide cones even with full precision
    // perhaps dragon and buddha are not that high poly?
    float vx, vy, vz, vw; // vw is only used for alignment
    uint8_t nx, ny, nz, nw; // nw is only used for alignment
    uint16_t tu, tv;
    // binding descriptions specify at which rate to load data
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription binding{};
        binding.binding = 0; // all data for a vertex is packed within one array, so we have only one binding
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        binding.stride = sizeof(Vertex);
        return binding;
    }
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescription() {
        std::array<VkVertexInputAttributeDescription, 3> attributes{};
        attributes[0].binding = 0; // which binding to take attribute from
        attributes[0].location = 0; // location referred in vertex shader as input, i.e layout(location=X)
        attributes[0].format = VK_FORMAT_R16G16B16_SFLOAT;
        attributes[0].offset = offsetof(Vertex, vx);
        attributes[1].binding = 0; 
        attributes[1].location = 1;
        attributes[1].format = VK_FORMAT_R8G8B8_UINT;
        attributes[1].offset = offsetof(Vertex, nx);
        attributes[2].binding = 0; 
        attributes[2].location = 2;
        attributes[2].format = VK_FORMAT_R16G16_SFLOAT;
        attributes[2].offset = offsetof(Vertex, tu);
        return attributes;
    }
    bool operator==(const Vertex& v) const {
        return vx == v.vx && vy == v.vy && vz == v.vz && 
                nx == v.nx && ny == v.ny && nz == v.nz &&
                tu == v.tu && tv == v.tv;
    }
};
namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(glm::vec3(vertex.vx, vertex.vy, vertex.vz)) ^
                   (hash<glm::vec3>()(glm::vec3(vertex.nx, vertex.ny, vertex.nz)) << 1)) >> 1) ^
                   (hash<glm::vec2>()(glm::vec2(vertex.tu, vertex.tv)) << 1);
        }
    };
}
struct Meshlet {
    // cone[0:3] describes the cone axis as an average of all normals within the meshlet
    // cone[3] describes the -cos(A+90 deg), where A is the half angle of the cone (angle between the furthest normal and cone axis)
    float cone[4];
    // cone apex is the average of all vertices within the meshlet, it is needed to
    // calculate the view vector
    float coneApex[4]; // vec3 is padded to vec4 anyways
    // this is offset into global meshletData buffer, where meshletData[dataOffset : dataOffset + vertexCount]
    // points to the array of global vertex indices, and meshletData[dataOffset + vertexCount : dataOffset + vertexCount + triangleCount*3] 
    // points to an array of local vertex indices, i.e they point to the first part of meshletData
    uint32_t dataOffset;
    uint8_t triangleCount;
    uint8_t vertexCount; // number of unique vertices
    uint8_t padding[10]; // padding
};

struct MVP {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

// this struct holds data extracted from the SPIR-V
struct DescriptorResourceInfo {
    uint32_t set; // which set the descriptor belongs to
    uint32_t binding; // which binding the descriptor is tied to
    VkDescriptorType type; // type of the descriptor
    uint32_t descriptorCount; // number of descriptors in the array (if it is an array)
    std::string name;
    // since Mesh shader and Vertex shader contain exactly the same descriptors, it is possible
    // to put them into the same struct but define multiple stages using | operator
    VkShaderStageFlags stage;
    // this operator is used to push DescriptorResourceInfo into std::set
    bool operator<(const DescriptorResourceInfo& other) const {
        if (set != other.set) return set < other.set;
        return binding < other.binding;
    }
};
struct Shader {
    VkShaderModule module;
    // a shader can only belong to one stage, therefore use of VkShaderStageFlagBits instead of VkShaderStageFlags
    VkShaderStageFlagBits stage;
    std::vector<uint32_t> code; // vector of words (1 word = uint32_t = 4 bytes)
    uint32_t codeSize; // size in bytes
};

// this struct is only used to provide actual data for descriptor update template
struct DescriptorData {
    union {
        VkDescriptorImageInfo imageInfo;
        VkDescriptorBufferInfo bufferInfo;
    };
};

class Engine {
public:
    Engine();
    ~Engine();
    void run();

    bool meshShadersEnabled = VK_FALSE;
private:
    GLFWwindow* window = nullptr;
    VkInstance instance;
    VkPhysicalDevice pDevice = VK_NULL_HANDLE;
    VkDevice device;
    QueueFamilies queueFamilies;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue transferQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkExtent2D swapchainExtent;
    VkFormat swapchainFormat;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkPipeline graphicsPipeline;
    VkPipeline meshGraphicsPipeline;
    Shader shaders[5]; // collection of all shaders to be compiled
    std::set<DescriptorResourceInfo> descriptorResourceInfos; // descriptor resources information extracted from SPIR-V
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorUpdateTemplate descriptorUpdateTemplate;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    VkPipelineLayout pipelineLayout;
    VkRenderPass renderpass;
    std::vector<VkFramebuffer> framebuffers;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer meshletBuffer;
    VkDeviceMemory meshletBufferMemory;
    VkBuffer meshletDataBuffer;
    VkDeviceMemory meshletDataBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkCommandPool graphicsCmdPool;
    VkCommandPool transferCmdPool;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;
    VkImage depthBuffer;
    VkDeviceMemory depthBufferMemory;
    VkImageView depthBufferImageView;
    VkImage colorBuffer;
    VkDeviceMemory colorBufferMemory;
    VkImageView colorBufferImageView;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Meshlet> meshlets;
    std::vector<uint32_t> meshletData;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    std::vector<VkQueryPool> queryPools;

    void createWindow();
    void loadModel();
    float computeVertexScore(int cachePosition, int valence);
    void optimizeGeometry();
    void buildMeshletCon(Meshlet& meshlet, std::vector<uint32_t> globalIndices, std::vector<uint8_t> localIndices);
    void createMeshletDataBuffer();
    void createMeshlets();
    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapchain();
    void createImageView(VkFormat format, VkImage& image, uint32_t mipLevels, VkImageAspectFlags aspectMask, VkImageView& imageView);
    void createShaders();
    void createDescriptorSetLayout();
    void createDescriptorUpdateTemplate();
    void createDescriptorPool();
    void createDescriptorSets();
    void createGraphicsPipeline();
    void createShader(Shader& shader, std::string path);
    void createShaderModule(Shader& shader);
    void createRenderpass();
    void createFramebuffers();
    void createCommandPool(VkCommandPool& cmdPool, VkCommandPoolCreateFlags flags, uint32_t queueFamily);
    void createCommandBuffer(VkCommandBuffer* cmdBuffer, int count, VkCommandPool& cmdPool);
    void createSemaphore(VkSemaphore& sem);
    void destroySemaphore(VkSemaphore& sem);
    void createFence(VkFence& fence, VkFenceCreateFlags flags = VK_FENCE_CREATE_SIGNALED_BIT);
    void destroyFence(VkFence& fence);
    void cleanupSwapchain();
    void recreateSwapchain();
    void createVertexBuffer();
    void createIndexBuffer();
    void createMeshletBuffer();
    void createBuffer(VkBuffer& buffer, VkDeviceMemory& bufferMemory, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties);
    void createUniformBuffers();
    void createTextureImage();
    void createImage(VkImage& image, VkDeviceMemory& imageMemory, VkSampleCountFlagBits samples, int width, int height, uint32_t mipLevels,
        VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memoryProperties);
    void createTextureSampler();
    void createDepthBuffer();
    void createColorBuffer();
    void createQueryPools();
    VkFormat findDepthFormat();

    const int MAX_FRAMES_IN_FLIGHT = 4;
    uint32_t currentFrame = 0;
    std::vector<float> gpuTimes;
    const std::string MODEL_PATH = std::string(PROJECT_ROOT) + "/assets/kitten.obj";
    const std::string TEXTURE_PATH = std::string(PROJECT_ROOT) + "/textures/viking_room.png";
    const std::string VERT_SHADER_PATH = std::string(PROJECT_ROOT) + "/shaders/shader.vert.spv";
    const std::string MESH_SHADER_PATH = std::string(PROJECT_ROOT) + "/shaders/shader.mesh.spv";
    const std::string FRAG_SHADER_PATH = std::string(PROJECT_ROOT) + "/shaders/shader.frag.spv";
    const std::string TASK_SHADER_PATH = std::string(PROJECT_ROOT) + "/shaders/shader.task.spv";
    std::vector<const char*> instanceExtensions = {
        "VK_KHR_surface",
        "VK_KHR_xcb_surface"
    };
    std::vector<const char*> instanceLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_NV_MESH_SHADER_EXTENSION_NAME
    };
    bool checkInstanceExtensionsSupport();
    bool checkInstanceLayersSupport();
    bool checkDeviceExtensionsSupport(VkPhysicalDevice candidate);
    
    bool isDeviceSuitable(VkPhysicalDevice candidate);
    QueueFamilies findQueueFamilies(VkPhysicalDevice candidate);
    SurfaceDetails getSurfaceDetails(VkPhysicalDevice candidate);
    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
    VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& presentModes);
    VkExtent2D chooseSurfaceExtent(VkSurfaceCapabilitiesKHR capabilities);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void generateMipmaps(VkImage image, VkFormat format, uint32_t width, uint32_t height, uint32_t mipLevels);
    VkSampleCountFlagBits getMaxSampleCount();
    VkCommandBuffer beginRecording(VkCommandPool& cmdPool);
    void stopRecording(VkCommandBuffer& cmdBuffer, VkCommandPool& cmdPool);
    void parseSPIRV(Shader& shader);
    void updateMVP();

    void recordCmdBuffer(VkCommandBuffer& cmdBuffer, uint32_t imageIndex);
    void transitionImageLayout(VkImage image, VkFormat format, uint32_t mipLevels, VkImageLayout oldLayout, VkImageLayout newLayout);
};