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
    glm::vec3 position;
    glm::vec3 color;
    // binding descriptions specify at which rate to load data
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription binding{};
        binding.binding = 0; // all data for a vertex is packed within one array, so we have only one binding
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        binding.stride = sizeof(Vertex);
        return binding;
    }
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescription() {
        std::array<VkVertexInputAttributeDescription, 2> attributes{};
        attributes[0].binding = 0; // which binding to take attribute from
        attributes[0].location = 0; // location referred in vertex shader as input
        attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[0].offset = offsetof(Vertex, position);
        attributes[1].binding = 0; 
        attributes[1].location = 1;
        attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[1].offset = offsetof(Vertex, color);
        return attributes;
    }
};
const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}}
};
const std::vector<uint32_t> indices = {
    0, 1, 2, 2, 3, 0
};

struct MVP {
    glm::mat4 view;
    glm::mat4 model;
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

    void createWindow();
    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapchain();
    void createImageView(VkFormat format, VkImage& image, VkImageAspectFlags aspectMask, VkImageView& imageView);
    void createDescriptorSetLayout();
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
    void createUniformBuffer();

    const int MAX_FRAMES_IN_FLIGHT = 4;
    uint32_t currentFrame = 0;
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
    void updateMVP();

    void recordCmdBuffer(VkCommandBuffer& cmdBuffer, uint32_t imageIndex);
};