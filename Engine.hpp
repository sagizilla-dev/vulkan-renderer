#pragma once
#include "utils.hpp"

struct QueueFamilies {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
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

    void createWindow();
    void createInstance();
    void createSurface();
    void createDevice();

    std::vector<const char*> instanceExtensions = {
        "VK_KHR_surface",
        "VK_KHR_xcb_surface"
    };
    std::vector<const char*> instanceLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    std::vector<const char*> deviceExtensions = {
        
    };
    std::vector<const char*> deviceLayers = {
        
    };
    bool checkInstanceExtensionsSupport();
    bool checkInstanceLayersSupport();
    bool checkDeviceExtensionsSupport(VkPhysicalDevice candidate);
    
    bool isDeviceSuitable(VkPhysicalDevice candidate);
    QueueFamilies findQueueFamilies(VkPhysicalDevice candidate);
};