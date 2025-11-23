#pragma once
#include "utils.hpp"

struct QueueFamilies {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};
struct SurfaceDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> surfaceFormats;
    std::vector<VkPresentModeKHR> presentModes;
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
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkExtent2D swapchainExtent;
    VkFormat swapchainFormat;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkPipeline graphicsPipeline;
    VkPipelineLayout pipelineLayout;
    VkRenderPass renderpass;
    std::vector<VkFramebuffer> framebuffers;

    void createWindow();
    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapchain();
    void createImageView(VkFormat format, VkImage& image, VkImageAspectFlags aspectMask, VkImageView& imageView);
    void createGraphicsPipeline();
    void createShaderModule(VkShaderModule& shaderModule, const std::vector<char>& code);
    void createRenderpass();
    void createFramebuffers();
    void createCommandPool(VkCommandPool& cmdPool, VkCommandPoolCreateFlags flags, uint32_t queueFamily);
    void createCommandBuffer(VkCommandBuffer* cmdBuffer, int count, VkCommandPool& cmdPool);

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

    void recordCmdBuffer(VkCommandBuffer& cmdBuffer, uint32_t imageIndex);
};