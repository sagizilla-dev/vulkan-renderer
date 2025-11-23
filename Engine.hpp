#pragma once
#include "utils.hpp"

class Engine {
public:
    Engine();
    ~Engine();
    void run();
private:
    GLFWwindow* window = nullptr;
    VkInstance instance;

    void createWindow();
    void createInstance();

    std::vector<const char*> instanceExtensions = {
        
    };
    bool checkInstanceExtensionsSupport();
    std::vector<const char*> instanceLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    bool checkInstanceLayersSupport();
};