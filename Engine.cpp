#include "Engine.hpp"

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

Engine::Engine() {
    createWindow();
    createInstance();
    createSurface();
    createDevice();
    createSwapchain();
}
Engine::~Engine() {
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}
void Engine::run() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
    }
}

void Engine::createWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
    glfwSetKeyCallback(window, keyCallback);
}
void Engine::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_4;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pApplicationName = "Vulkan application";
    appInfo.pEngineName = "Vulkan engine";

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    if (!checkInstanceExtensionsSupport()) throw std::runtime_error("Requested instance extensions not supported");
    instanceInfo.enabledExtensionCount = instanceExtensions.size();
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
    if (!checkInstanceLayersSupport()) throw std::runtime_error("Requested instance layers not supported");
    instanceInfo.enabledLayerCount = instanceLayers.size();
    instanceInfo.ppEnabledLayerNames = instanceLayers.data();
    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance));
}
void Engine::createSurface() {
    glfwCreateWindowSurface(instance, window, nullptr, &surface);
}

void Engine::createDevice() {
    uint32_t count;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    for (const auto& candidate: devices) {
        if (isDeviceSuitable(candidate)) {
            pDevice = candidate;
            queueFamilies = findQueueFamilies(candidate);
            break;
        }
    }
    if (pDevice == VK_NULL_HANDLE) throw std::runtime_error("No suitable device found");

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.enabledExtensionCount = deviceExtensions.size();
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
    VkPhysicalDeviceFeatures features{};
    features.geometryShader = VK_TRUE;
    deviceInfo.pEnabledFeatures = &features;
    
    std::set<uint32_t> uniqueQueueFamilies = {queueFamilies.graphicsFamily.value(), queueFamilies.presentFamily.value()};
    std::vector<VkDeviceQueueCreateInfo> queueInfos{};
    float priority = 1.0f;
    for (const auto& queueFamily: uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.pQueuePriorities = &priority;
        queueInfo.queueCount = 1;
        queueInfo.queueFamilyIndex = queueFamily;
        queueInfos.push_back(queueInfo);
    }
    deviceInfo.queueCreateInfoCount = queueInfos.size();
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    VK_CHECK(vkCreateDevice(pDevice, &deviceInfo, nullptr, &device));

    vkGetDeviceQueue(device, queueFamilies.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, queueFamilies.presentFamily.value(), 0, &presentQueue);
}
void Engine::createSwapchain() {
    SurfaceDetails surfaceDetails = getSurfaceDetails(pDevice);
    VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(surfaceDetails.surfaceFormats);
    VkPresentModeKHR presentMode = choosePresentMode(surfaceDetails.presentModes);
    VkExtent2D surfaceExtent = chooseSurfaceExtent(surfaceDetails.capabilities);

    uint32_t imageCount = surfaceDetails.capabilities.minImageCount + 1;
    if (imageCount > surfaceDetails.capabilities.maxImageCount && surfaceDetails.capabilities.maxImageCount > 0) {
        imageCount = surfaceDetails.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainInfo.clipped = VK_TRUE;
    swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageColorSpace = surfaceFormat.colorSpace;
    swapchainInfo.imageExtent = surfaceExtent;
    swapchainInfo.imageFormat = surfaceFormat.format;
    std::vector<uint32_t> queueFamilyIndices = {queueFamilies.graphicsFamily.value(), queueFamilies.presentFamily.value()};
    if (queueFamilies.graphicsFamily.value() != queueFamilies.presentFamily.value()) {
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainInfo.queueFamilyIndexCount = 2;
        swapchainInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    } else {
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapchainInfo.queueFamilyIndexCount = 1;
        swapchainInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    }
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchainInfo.minImageCount = imageCount;
    swapchainInfo.presentMode = presentMode;
    swapchainInfo.oldSwapchain = VK_NULL_HANDLE;
    swapchainInfo.preTransform = surfaceDetails.capabilities.currentTransform;
    swapchainInfo.surface = surface;
    VK_CHECK(vkCreateSwapchainKHR(device, &swapchainInfo, nullptr, &swapchain));
}

bool Engine::checkInstanceExtensionsSupport() {
    uint32_t count;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, availableExtensions.data());
    std::set<std::string> requestedExtensions(instanceExtensions.begin(), instanceExtensions.end());
    for (const auto& ext: availableExtensions)
        requestedExtensions.erase(ext.extensionName);
    return requestedExtensions.empty();
}
bool Engine::checkInstanceLayersSupport() {
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> availableLayers(count);
    vkEnumerateInstanceLayerProperties(&count, availableLayers.data());
    std::set<std::string> requestedLayers(instanceLayers.begin(), instanceLayers.end());
    for (const auto& layer: availableLayers)
        requestedLayers.erase(layer.layerName);
    return requestedLayers.empty();
}
bool Engine::checkDeviceExtensionsSupport(VkPhysicalDevice candidate) {
    uint32_t count;
    vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(count);
    vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, availableExtensions.data());
    std::set<std::string> requestedExtensions(deviceExtensions.begin(), deviceExtensions.end());
    for (const auto& ext: availableExtensions)
        requestedExtensions.erase(ext.extensionName);
    return requestedExtensions.empty();
}
bool Engine::isDeviceSuitable(VkPhysicalDevice candidate) {
    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(candidate, &features);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(candidate, &props);

    QueueFamilies _queueFamilies = findQueueFamilies(candidate);

    SurfaceDetails surfaceDetails = getSurfaceDetails(candidate);

    return features.geometryShader == VK_TRUE &&
        props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        checkDeviceExtensionsSupport(candidate) &&
        _queueFamilies.isComplete() &&
        !surfaceDetails.surfaceFormats.empty() && !surfaceDetails.presentModes.empty();
}
QueueFamilies Engine::findQueueFamilies(VkPhysicalDevice candidate) {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
    std::vector<VkQueueFamilyProperties> allQueueFamilies(count);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, allQueueFamilies.data());
    
    int i = 0;
    QueueFamilies _queueFamilies;
    for (const auto& family: allQueueFamilies) {
        if (family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            _queueFamilies.graphicsFamily = i;
        }
        VkBool32 presentSupported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface, &presentSupported);
        if (presentSupported) {
            _queueFamilies.presentFamily = i;
        }
        if (_queueFamilies.isComplete()) {
            break;
        }
        i++;
    }
    return _queueFamilies;
}
SurfaceDetails Engine::getSurfaceDetails(VkPhysicalDevice candidate) {
    SurfaceDetails surfaceDetails;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(candidate, surface, &surfaceDetails.capabilities);
    uint32_t count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface, &count, nullptr);
    surfaceDetails.surfaceFormats.resize(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface, &count, surfaceDetails.surfaceFormats.data());
    vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface, &count, nullptr);
    surfaceDetails.presentModes.resize(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface, &count, surfaceDetails.presentModes.data());
    return surfaceDetails;
}
VkSurfaceFormatKHR Engine::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& format: formats) {
        if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && format.format == VK_FORMAT_B8G8R8A8_SRGB) {
            return format;
        }
    }
    return formats[0];
}
VkPresentModeKHR Engine::choosePresentMode(const std::vector<VkPresentModeKHR>& presentModes) {
    for (const auto& mode: presentModes) {
        if (mode==VK_PRESENT_MODE_MAILBOX_KHR) {
            return mode;
        }
    }
    return presentModes[0];
}
VkExtent2D Engine::chooseSurfaceExtent(VkSurfaceCapabilitiesKHR capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return actualExtent;
    }
}