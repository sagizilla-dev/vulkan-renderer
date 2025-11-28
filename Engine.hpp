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
    float vx, vy, vz;
    uint8_t nx, ny, nz;
    float tu, tv;
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
        attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[0].offset = offsetof(Vertex, vx);
        attributes[1].binding = 0; 
        attributes[1].location = 1;
        attributes[1].format = VK_FORMAT_R8G8B8_UINT;
        attributes[1].offset = offsetof(Vertex, nx);
        attributes[2].binding = 0; 
        attributes[2].location = 2;
        attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
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


struct MVP {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

class Engine {
public:
    Engine();
    ~Engine();
    void run();
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
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    VkPipelineLayout pipelineLayout;
    VkRenderPass renderpass;
    std::vector<VkFramebuffer> framebuffers;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
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
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    void createWindow();
    void loadModel();
    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapchain();
    void createImageView(VkFormat format, VkImage& image, uint32_t mipLevels, VkImageAspectFlags aspectMask, VkImageView& imageView);
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
    void createGraphicsPipeline();
    void createShaderModule(VkShaderModule& shaderModule, const std::vector<char>& code);
    void createRenderpass();
    void createFramebuffers();
    void createCommandPool(VkCommandPool& cmdPool, VkCommandPoolCreateFlags flags, uint32_t queueFamily);
    void createCommandBuffer(VkCommandBuffer* cmdBuffer, int count, VkCommandPool& cmdPool);
    void createSemaphore(VkSemaphore& sem);
    void destroySemaphore(VkSemaphore& sem);
    // VK_FENCE_CREATE_SIGNALED_BIT means fence is created as already signaled
    void createFence(VkFence& fence, VkFenceCreateFlags flags = VK_FENCE_CREATE_SIGNALED_BIT);
    void destroyFence(VkFence& fence);
    void cleanupSwapchain();
    void recreateSwapchain();
    void createVertexBuffer();
    void createIndexBuffer();
    void createBuffer(VkBuffer& buffer, VkDeviceMemory& bufferMemory, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties);
    void createUniformBuffers();
    void createTextureImage();
    void createImage(VkImage& image, VkDeviceMemory& imageMemory, VkSampleCountFlagBits samples, int width, int height, uint32_t mipLevels,
        VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memoryProperties);
    void createTextureSampler();
    void createDepthBuffer();
    void createColorBuffer();
    VkFormat findDepthFormat();

    const int MAX_FRAMES_IN_FLIGHT = 4;
    uint32_t currentFrame = 0;
    const std::string MODEL_PATH = "../viking_room.obj";
    const std::string TEXTURE_PATH = "../viking_room.png";
    std::vector<const char*> instanceExtensions = {
        "VK_KHR_surface",
        "VK_KHR_xcb_surface"
    };
    std::vector<const char*> instanceLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
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
    void updateMVP();

    void recordCmdBuffer(VkCommandBuffer& cmdBuffer, uint32_t imageIndex);
    void transitionImageLayout(VkImage image, VkFormat format, uint32_t mipLevels, VkImageLayout oldLayout, VkImageLayout newLayout);
};