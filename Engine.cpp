#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include "Engine.hpp"

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

Engine::Engine() {
    createWindow();
    loadModel();
    createInstance();
    createSurface();
    createDevice();
    createSwapchain();
    createCommandPool(graphicsCmdPool, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilies.graphicsFamily.value());
    createCommandPool(transferCmdPool, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queueFamilies.transferFamily.value());
    createDepthBuffer();
    createRenderpass();
    createFramebuffers();
    createUniformBuffers();
    createTextureImage();
    createTextureSampler();
    createDescriptorSetLayout();
    createDescriptorPool();
    createDescriptorSets();
    createGraphicsPipeline();
    createVertexBuffer();
    createIndexBuffer();
}
Engine::~Engine() {
    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);
    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    vkDestroyCommandPool(device, transferCmdPool, nullptr);
    vkDestroyCommandPool(device, graphicsCmdPool, nullptr); // command buffers are freed when command pool is destroyed
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderpass, nullptr);
    cleanupSwapchain();
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}
void Engine::run() {
    // instead of having 1 frame in flight and making CPU wait for GPU to finish work, 
    // we keep pumping work from CPU to GPU by having multiple frames in flight
    // having too many frames in flight introduces big input latency though
    std::vector<VkSemaphore> imageAvailable(MAX_FRAMES_IN_FLIGHT); // indicates the image is acquired 
    for (int i = 0; i<MAX_FRAMES_IN_FLIGHT; i++) createSemaphore(imageAvailable[i]);
    std::vector<VkSemaphore> renderDone(MAX_FRAMES_IN_FLIGHT); // indicates rendering to an image is done
    for (int i = 0; i<MAX_FRAMES_IN_FLIGHT; i++) createSemaphore(renderDone[i]);
    std::vector<VkFence> cmdBufferReady(MAX_FRAMES_IN_FLIGHT); // indicates command buffer is ready to be rerecorded
    for (int i = 0; i<MAX_FRAMES_IN_FLIGHT; i++) createFence(cmdBufferReady[i]);
    std::vector<VkCommandBuffer> cmdBuffer(MAX_FRAMES_IN_FLIGHT);
    createCommandBuffer(cmdBuffer.data(), MAX_FRAMES_IN_FLIGHT, graphicsCmdPool);
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // wait until command buffer is ready to be rerecorded
        vkWaitForFences(device, 1, &cmdBufferReady[currentFrame], VK_TRUE, ~0ull);
        
        uint32_t imageIndex;
        // get the next available image from the swapchain, store the image index in imageIndex
        // and signal to imageAvailable once it is acquired
        VkResult res = vkAcquireNextImageKHR(device, swapchain, ~0ull, imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if (res==VK_ERROR_OUT_OF_DATE_KHR) {
            // take this branch if the swapchain is incompatible with the surface
            recreateSwapchain();
            continue;
        } else if (res!=VK_SUCCESS && res!=VK_SUBOPTIMAL_KHR) {
            // VK_SUBOPTIMAL_KHR means the swapchain can still be used, but the surface properties are no longer matched exactly
            throw std::runtime_error("Couldn't acquire next image");
        }

        // reset the fence only after acquiring the next image - it is important to avoid deadlock 
        // otherwise, if swapchain is recreated, fence is reset but we are still waiting on it
        vkResetFences(device, 1, &cmdBufferReady[currentFrame]);

        vkResetCommandBuffer(cmdBuffer[currentFrame], 0);
        updateMVP();
        recordCmdBuffer(cmdBuffer[currentFrame], imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer[currentFrame];
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailable[currentFrame];
        std::vector<VkPipelineStageFlags> waitStages = {
            // stall operations at this stage to wait for the semaphore
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT 
        };
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderDone[currentFrame];
        VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, cmdBufferReady[currentFrame]));

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderDone[currentFrame];
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;
        res = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (res==VK_ERROR_OUT_OF_DATE_KHR || res==VK_SUBOPTIMAL_KHR) {
            recreateSwapchain();
        } else if (res!=VK_SUCCESS) {
            throw std::runtime_error("Couldn't acquire next image");
        }

        currentFrame = (currentFrame+1) % MAX_FRAMES_IN_FLIGHT;
    }
    vkDeviceWaitIdle(device);
    for (int i = 0; i<MAX_FRAMES_IN_FLIGHT; i++) destroySemaphore(imageAvailable[i]);
    for (int i = 0; i<MAX_FRAMES_IN_FLIGHT; i++) destroySemaphore(renderDone[i]);
    for (int i = 0; i<MAX_FRAMES_IN_FLIGHT; i++) destroyFence(cmdBufferReady[i]);
}

void Engine::createWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
    glfwSetKeyCallback(window, keyCallback);
}
void Engine::loadModel() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;
    std::string warn;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) throw std::runtime_error(err);
    std::unordered_map<Vertex, uint32_t> uniqueVertices{};
    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};
            vertex.position = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            vertex.uv = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
            };

            vertex.color = {1.0f, 1.0f, 1.0f};
            
            if (uniqueVertices.count(vertex)==0) {
                uniqueVertices[vertex] = vertices.size();
                vertices.push_back(vertex);
            }

            indices.push_back(uniqueVertices[vertex]);
        }
    }
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
    features.samplerAnisotropy = VK_TRUE;
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
    vkGetDeviceQueue(device, queueFamilies.transferFamily.value(), 0, &transferQueue);
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
    // set to true if we don't care about pixels that are obscured, i.e there is another window in front of them
    swapchainInfo.clipped = VK_TRUE;
    // specifies if the alpha channel should be used for blending with other windows
    swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; 
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageColorSpace = surfaceFormat.colorSpace;
    swapchainInfo.imageExtent = surfaceExtent;
    swapchainInfo.imageFormat = surfaceFormat.format;
    std::vector<uint32_t> queueFamilyIndices = {queueFamilies.graphicsFamily.value(), queueFamilies.presentFamily.value()};
    if (queueFamilies.graphicsFamily.value() != queueFamilies.presentFamily.value()) {
        // swapchain images are used by two different queues
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

    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
    swapchainImageViews.resize(imageCount);

    swapchainExtent = surfaceExtent;
    swapchainFormat = surfaceFormat.format;
    for (size_t i=0; i<imageCount; i++) {
        createImageView(swapchainFormat, swapchainImages[i], VK_IMAGE_ASPECT_COLOR_BIT, swapchainImageViews[i]);
    }
}
void Engine::createImageView(VkFormat format, VkImage& image, VkImageAspectFlags aspectMask, VkImageView& imageView) {
    VkImageViewCreateInfo imageViewInfo{};
    imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewInfo.format = format;
    imageViewInfo.image = image;
    imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    // swizzling of color components
    imageViewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    // subresourceRange describes what the image's purpose is and which part of the image should be accessed
    imageViewInfo.subresourceRange.aspectMask = aspectMask;
    imageViewInfo.subresourceRange.baseArrayLayer = 0;
    imageViewInfo.subresourceRange.baseMipLevel = 0;
    imageViewInfo.subresourceRange.layerCount = 1;
    imageViewInfo.subresourceRange.levelCount = 1;
    VK_CHECK(vkCreateImageView(device, &imageViewInfo, nullptr, &imageView));
}
void Engine::createDescriptorSetLayout() {
    // this function creates descriptor set layout, which specifies what type of resources
    // are to be passed into the shaders, same way render pass defines what type of attachments to expect
    // descriptor sets define data itself, same way framebuffers define exact attachments
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings(2);
    layoutBindings[0].binding = 0; // referenced in the shader as layout(binding = X)
    layoutBindings[0].descriptorCount = 1; // descriptor can be an array
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{};
    descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutInfo.bindingCount = layoutBindings.size();
    descriptorSetLayoutInfo.pBindings = layoutBindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutInfo, nullptr, &descriptorSetLayout));
}
void Engine::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes(2);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // number of descriptors of a specific time
    poolSizes[0].descriptorCount = MAX_FRAMES_IN_FLIGHT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    // total number of descriptor sets that are to be allocated
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}
void Engine::createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.pSetLayouts = layouts.data();
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.descriptorPool = descriptorPool;
    // allocate descriptor sets, each created based on the descriptor set layout
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()));

    // bind the actual data to descriptor sets
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(MVP);
        
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        std::vector<VkWriteDescriptorSet> writeDescriptors(2);
        writeDescriptors[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptors[0].pBufferInfo = &bufferInfo;
        writeDescriptors[0].descriptorCount = 1; // in case the descriptor is an array
        writeDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptors[0].dstSet = descriptorSets[i];
        writeDescriptors[0].dstBinding = 0;
        writeDescriptors[0].dstArrayElement = 0;

        writeDescriptors[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptors[1].pImageInfo = &imageInfo;
        writeDescriptors[1].descriptorCount = 1; // in case the descriptor is an array
        writeDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptors[1].dstSet = descriptorSets[i];
        writeDescriptors[1].dstBinding = 1;
        writeDescriptors[1].dstArrayElement = 0;

        vkUpdateDescriptorSets(device, writeDescriptors.size(), writeDescriptors.data(), 0, nullptr);
    }
}
void Engine::createGraphicsPipeline() {
    auto vertCode = readFile("../shader.vert.spv");
    auto fragCode = readFile("../shader.frag.spv");
    VkShaderModule vertShaderModule;
    createShaderModule(vertShaderModule, vertCode);
    VkShaderModule fragShaderModule;
    createShaderModule(fragShaderModule, fragCode);
    
    std::vector<VkPipelineShaderStageCreateInfo> shaderStageInfos(2);
    shaderStageInfos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfos[0].module = vertShaderModule;
    shaderStageInfos[0].pName = "main";
    shaderStageInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStageInfos[0].pSpecializationInfo = VK_NULL_HANDLE; // allows us to specify values for shader constants
    shaderStageInfos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfos[1].module = fragShaderModule;
    shaderStageInfos[1].pName = "main";
    shaderStageInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStageInfos[1].pSpecializationInfo = VK_NULL_HANDLE;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = dynamicStates.size();
    dynamicStateInfo.pDynamicStates = dynamicStates.data();

    // this fixed function describes what to expect as inputs to vertex shaders
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    // attribute description just describes data inside the vertex
    auto attributes = Vertex::getAttributeDescription();
    vertexInputInfo.vertexAttributeDescriptionCount = attributes.size();
    vertexInputInfo.pVertexAttributeDescriptions = attributes.data();
    // binding is spacing between data and whether the data is per-vertex or per-instance
    auto bindings = Vertex::getBindingDescription();
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindings;

    VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
    assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    // _STRIP topology forms triangles by using one new vertex and two previous vertices in the buffer
    // useful to decrease the size of the index buffer
    assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    assemblyInfo.primitiveRestartEnable = VK_FALSE; // if set to VK_TRUE, then it is possible to break up lines and triangles in _STRIP topology

    // this will be defined inside the render loop, i.e as dynamic state
    VkPipelineViewportStateCreateInfo viewportInfo{};
    viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportInfo.scissorCount = 1;
    viewportInfo.viewportCount = 1;

    // rasterizer takes geometry that is shaped by vertices from the vertex shader and turns it into
    // fragments to be colored by the fragment shader
    VkPipelineRasterizationStateCreateInfo rasterInfo{};
    rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterInfo.rasterizerDiscardEnable = VK_FALSE;
    rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterInfo.lineWidth = 1.0f;
    rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterInfo.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo msaaInfo{};
    msaaInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msaaInfo.sampleShadingEnable = VK_FALSE;
    msaaInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencilInfo{};
    depthStencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilInfo.depthTestEnable = VK_TRUE; // enable depth testing
    depthStencilInfo.depthWriteEnable = VK_TRUE; // enable writing to the depth buffer
    depthStencilInfo.depthCompareOp = VK_COMPARE_OP_LESS; // lower depth = closer
    depthStencilInfo.depthBoundsTestEnable = VK_FALSE; // if set to VK_TRUE then we can only keep fragments that fall within specific depth range
    depthStencilInfo.minDepthBounds = 0.0f;
    depthStencilInfo.maxDepthBounds = 1.0f;
    depthStencilInfo.stencilTestEnable = VK_FALSE;
    
    // this struct defines color blending settings per framebuffer
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    // this struct defines color blending settings globally
    VkPipelineColorBlendStateCreateInfo colorBlendInfo{};
    colorBlendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendInfo.logicOpEnable = VK_FALSE;
    colorBlendInfo.attachmentCount = 1;
    colorBlendInfo.pAttachments = &colorBlendAttachment;

    // this struct is used to pass uniforms and push constants into shaders
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStageInfos.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &assemblyInfo;
    pipelineInfo.pViewportState = &viewportInfo;
    pipelineInfo.pRasterizationState = &rasterInfo;
    pipelineInfo.pMultisampleState = &msaaInfo;
    pipelineInfo.pDepthStencilState = &depthStencilInfo;
    pipelineInfo.pColorBlendState = &colorBlendInfo;
    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderpass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // in case we want to derive a pipeline from another one

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline));

    // we can delete shader module right away since compilation and
    // linking of shaders are done when pipeline is created
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
}
void Engine::createShaderModule(VkShaderModule& shaderModule, const std::vector<char>& code) {
    VkShaderModuleCreateInfo shaderModuleInfo{};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = code.size();
    shaderModuleInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VK_CHECK(vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shaderModule));
}
void Engine::createRenderpass() {
    // all attachments, i.e color, depth, etc, are passed from the framebuffer in
    // the same order as they were defined during framebuffer creation
    // so this array must have the same order
    std::vector<VkAttachmentDescription> attachmentDescriptions(2);
    attachmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachmentDescriptions[0].format = swapchainFormat;
    attachmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // upon loading we need to clear the image
    attachmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachmentDescriptions[1].format = findDepthFormat();
    attachmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // we don't store the depth buffer after rendering
    attachmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    // each subpass has an array of attachment references
    std::vector<VkAttachmentReference> attachmentReferences(2);
    attachmentReferences[0].attachment = 0; // index of the attachment in the attachment descriptions array
    attachmentReferences[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // layout that the image must have upon entering the subpass
    attachmentReferences[1].attachment = 1;
    attachmentReferences[1].layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // subpass dependency takes care of image layout transitions for each subpass
    // renderpass by default has two implicit image layout transitiosn:
    // at the beginning of the renderpass and at the end
    // once renderpass starts, there is a chance swapchain image is actually still not available
    VkSubpassDependency subpassDependency{};
    subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL; // implicit subpass before or after the renderpass
    subpassDependency.dstSubpass = 0; // current subpass
    // what operations in srcSubpass must complete 
    subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // we wait for the swapchain to finish reading from the image
    // for color attachment it basically means we are waiting for the presentatino engine to finish reading the image
    subpassDependency.srcAccessMask = 0; // wait for all operations within the stage
    // what operations in dstSubpass must wait
    subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // wait before clearing the image

    // technically we don't have to provide subpass dependency for depth buffer since we can manaully transition the layout
    subpassDependency.srcStageMask |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT; // wait for previous frame's depth buffer writes to finish
    subpassDependency.srcAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; // wait for clearing to finish
    subpassDependency.dstStageMask |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    subpassDependency.dstAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    // the index of the attachment in this array is referenced in the fragment shader as layout(location=X) out
    subpass.pColorAttachments = &attachmentReferences[0];
    subpass.pDepthStencilAttachment = &attachmentReferences[1];

    VkRenderPassCreateInfo renderpassInfo{};
    renderpassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderpassInfo.attachmentCount = attachmentDescriptions.size();
    renderpassInfo.pAttachments = attachmentDescriptions.data();
    renderpassInfo.subpassCount = 1;
    renderpassInfo.pSubpasses = &subpass;
    renderpassInfo.dependencyCount = 1;
    renderpassInfo.pDependencies = &subpassDependency;
    VK_CHECK(vkCreateRenderPass(device, &renderpassInfo, nullptr, &renderpass));
}
void Engine::createFramebuffers() {
    framebuffers.resize(swapchainImages.size());
    for (size_t i = 0; i<swapchainImages.size(); i++) {
        // the order of image views/attachments defined here is important
        // image views/attachments are passed to the render pass in the same order
        // as they are defined here
        std::vector<VkImageView> attachments = {swapchainImageViews[i], depthBufferImageView};
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderpass;
        framebufferInfo.attachmentCount = attachments.size();
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapchainExtent.width;
        framebufferInfo.height = swapchainExtent.height;
        framebufferInfo.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]));
    }
}
void Engine::createCommandPool(VkCommandPool& cmdPool, VkCommandPoolCreateFlags flags, uint32_t queueFamily) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // VK_COMMAND_POOL_CREATE_TRANSIENT_BIT means command buffers are rerecorded with new commands very often
    // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT means command buffers can be rerecorded individually
    // without this flag they all have to be reset together
    poolInfo.flags = flags;
    poolInfo.queueFamilyIndex = queueFamily;
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool));
}
void Engine::createCommandBuffer(VkCommandBuffer* cmdBuffer, int count, VkCommandPool& cmdPool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool;
    allocInfo.commandBufferCount = count;
    // VK_COMMAND_BUFFER_LEVEL_PRIMARY means buffer can be submitted to a queue for execution but cannot be called from other buffers
    // VK_COMMAND_BUFFER_LEVEL_SECONDARY means buffer cannot be submitted to a queue for execution but can be called from other buffers
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, cmdBuffer));
}
void Engine::createSemaphore(VkSemaphore& sem) {
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &sem));
}
void Engine::destroySemaphore(VkSemaphore& sem) {
    vkDestroySemaphore(device, sem, nullptr);
}
void Engine::createFence(VkFence& fence, VkFenceCreateFlags flags) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = flags;
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));
}
void Engine::destroyFence(VkFence& fence) {
    vkDestroyFence(device, fence, nullptr);
}
void Engine::cleanupSwapchain() {
    vkDestroyImage(device, depthBuffer, nullptr);
    vkFreeMemory(device, depthBufferMemory, nullptr);
    vkDestroyImageView(device, depthBufferImageView, nullptr);
    for (const auto framebuffer: framebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    for (size_t i=0; i<swapchainImageViews.size(); i++) {
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}
void Engine::recreateSwapchain() {
    vkDeviceWaitIdle(device);
    
    cleanupSwapchain();
    
    createSwapchain();
    createDepthBuffer();
    createFramebuffers();
}
void Engine::createVertexBuffer() {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize vertexBufferSize = sizeof(vertices[0])*vertices.size();
    createBuffer(stagingBuffer, stagingBufferMemory, vertexBufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    // tie CPU memory to GPU memory and copy data
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, vertexBufferSize, 0, &data);
    memcpy(data, vertices.data(), sizeof(vertices[0])*vertices.size());
    vkUnmapMemory(device, stagingBufferMemory); // now the memory is transferred from CPU to GPU memory held by the buffer

    createBuffer(vertexBuffer, vertexBufferMemory, vertexBufferSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copyBuffer(stagingBuffer, vertexBuffer, vertexBufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}
void Engine::createIndexBuffer() {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize indexBufferSize = sizeof(indices[0])*indices.size();
    createBuffer(stagingBuffer, stagingBufferMemory, indexBufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, indexBufferSize, 0, &data);
    memcpy(data, indices.data(), sizeof(indices[0])*indices.size());
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(indexBuffer, indexBufferMemory, indexBufferSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copyBuffer(stagingBuffer, indexBuffer, indexBufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}
void Engine::createBuffer(VkBuffer& buffer, VkDeviceMemory& bufferMemory, VkDeviceSize size, 
VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is only used by graphics queue
    VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

    // VkMemoryRequirements defines the size of the buffer which may differ from VkBufferCreateInfo::size
    // also includes alignment, which is the offset in bytes where the buffer begins in the allocated
    // region of memory, depends on VkBufferCreateInfo::usage and VkBufferCreateInfo::flags
    // also includes memoryTypeBits, which is the bit field of the memory types that are suitable for the buffer
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
    VkMemoryAllocateInfo memAllocInfo{};
    memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllocInfo.allocationSize = memRequirements.size;
    memAllocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, memoryProperties);
    // allocate memory on the GPU for the buffer
    VK_CHECK(vkAllocateMemory(device, &memAllocInfo, nullptr, &bufferMemory));
    
    // now buffer on the GPU holds the memory
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}
void Engine::createUniformBuffers() {
    VkDeviceSize size = sizeof(MVP);
    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(uniformBuffers[i], uniformBuffersMemory[i], size, 
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        // persistent mapping
        vkMapMemory(device, uniformBuffersMemory[i], 0, size, 0, &uniformBuffersMapped[i]);
    }
}
void Engine::createTextureImage() {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth*texHeight*4;
    if (!pixels) throw std::runtime_error("Cannot read the texture file");

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(stagingBuffer, stagingBufferMemory, imageSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, imageSize);
    vkUnmapMemory(device, stagingBufferMemory);
    stbi_image_free(pixels);
    
    createImage(textureImage, textureImageMemory, texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, 
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, textureImage, texWidth, texHeight);
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    createImageView(VK_FORMAT_R8G8B8A8_SRGB, textureImage, VK_IMAGE_ASPECT_COLOR_BIT, textureImageView);
}
void Engine::createImage(VkImage& image, VkDeviceMemory& imageMemory, int width, int height,
VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memoryProperties) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1; // 3rd dimension in the extent matrix
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    // VK_IMAGE_TILING_LINEAR means texels are laid out in a row-major order
    // VK_IMAGE_TILING_OPTIMAL means texels are laid out according to implementation for optimal access
    imageInfo.tiling = tiling;
    // VK_IMAGE_LAYOUT_UNDEFINED means the very first image layout transition will discard texels
    // VK_IMAGE_LAYOUT_PREINITIALIZED means the very first image layout transition will preserve texels 
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // we transition the image layout to TRANSFER_DST and only then copy data
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &image));

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);
    VkMemoryAllocateInfo memAllocInfo{};
    memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllocInfo.allocationSize = memRequirements.size;
    memAllocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, memoryProperties);
    VK_CHECK(vkAllocateMemory(device, &memAllocInfo, nullptr, &imageMemory));

    vkBindImageMemory(device, image, imageMemory, 0);
}
void Engine::createTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR; // how to interpolate texels that are magnified (oversampling) 
    samplerInfo.minFilter = VK_FILTER_LINEAR; // how to interpolate texels that are minified (undersampling)
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE; // to solve undersampling issue
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pDevice, &props);
    samplerInfo.maxAnisotropy = props.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    // if set to VK_TRUE, coordinates are within [0, texWidth) and [0, texHeight)
    samplerInfo.unnormalizedCoordinates = VK_FALSE; // use [0, 1)
    samplerInfo.compareEnable = VK_FALSE; // if set to VK_TRUE, compare texel to a value first and use the result in filtering operations
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler));
}
void Engine::createDepthBuffer() {
    VkFormat depthFormat = findDepthFormat();
    createImage(depthBuffer, depthBufferMemory, swapchainExtent.width, swapchainExtent.height, depthFormat, 
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    createImageView(depthFormat, depthBuffer, VK_IMAGE_ASPECT_DEPTH_BIT, depthBufferImageView);
    transitionImageLayout(depthBuffer, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}
VkFormat Engine::findDepthFormat() {
    std::vector<VkFormat> formats = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    for (const auto& format: formats) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(pDevice, format, &props);
        // we only use optimal tiling for depth buffer
        if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return format;
        }
    }
    throw std::runtime_error("Cannot find a suitable format for depth image");
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

    return features.geometryShader == VK_TRUE && features.samplerAnisotropy == VK_TRUE && 
        props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        checkDeviceExtensionsSupport(candidate) &&
        _queueFamilies.isComplete() &&
        !surfaceDetails.surfaceFormats.empty() && !surfaceDetails.presentModes.empty();
}
QueueFamilies Engine::findQueueFamilies(VkPhysicalDevice candidate) {
    // this function finds indices of all queue families we need
    // some queue families may support same operations,
    // i.e two different queue families may support graphics operations
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
        if (family.queueFlags & VK_QUEUE_TRANSFER_BIT) {
            _queueFamilies.transferFamily = i;
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
    // this function finds details of the surface based on the physical device
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
        // this branch means we are free to set our own extent
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
uint32_t Engine::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    // VkPhysicalDeviceMemoryProperties has two arrays:
    // memoryHeaps: distinct memory resources like VRAM or swap space in RAM
    // memoryTypes: different types of memory within heaps
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(pDevice, &memProperties);
    for (uint32_t i = 0; i<memProperties.memoryTypeCount; i++) {
        // typeFilter consists of flags set for each memory type that is suitable for the buffer
        // order of flags/memory types in typeFilter corresponds to memoryTypes in VkPhysicalDeviceMemoryProperties 
        // VkMemoryPropertyFlags is the set of flags that define how the memory can be used, i.e memory is host visible or device local
        if ((typeFilter & (1<<i)) && (memProperties.memoryTypes[i].propertyFlags & properties)==properties) {
            return i;
        }
    }
    throw std::runtime_error("Cannot find a suitable memory type");
}
void Engine::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBuffer cmdBuffer = beginRecording(transferCmdPool);
    {
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        vkCmdCopyBuffer(cmdBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    }
    stopRecording(cmdBuffer, transferCmdPool);
}
void Engine::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer cmdBuffer = beginRecording(transferCmdPool);
    {
        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0; 
        // next two fields specify how pixels are laid out in memory, i.e. if there is any padding or data is tightly packed
        copyRegion.bufferImageHeight = 0; 
        copyRegion.bufferRowLength = 0;
        copyRegion.imageOffset = {0, 0, 0};
        copyRegion.imageExtent = {width, height, 1};
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageSubresource.mipLevel = 0;
        vkCmdCopyBufferToImage(cmdBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    }
    stopRecording(cmdBuffer, transferCmdPool);
}
VkCommandBuffer Engine::beginRecording(VkCommandPool& cmdPool) {
    VkCommandBuffer cmdBuffer{};
    createCommandBuffer(&cmdBuffer, 1, cmdPool);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);
    return cmdBuffer;
}
void Engine::stopRecording(VkCommandBuffer& cmdBuffer, VkCommandPool& cmdPool) {
    vkEndCommandBuffer(cmdBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;
    VkFence fence{};
    createFence(fence, 0);
    vkQueueSubmit(transferQueue, 1, &submitInfo, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, ~0ull);
    destroyFence(fence);
    vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}
void Engine::updateMVP() {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    MVP mvp{};
    mvp.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    mvp.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    mvp.proj = glm::perspective(glm::radians(45.0f), swapchainExtent.width / (float) swapchainExtent.height, 0.1f, 10.0f);
    mvp.proj[1][1] *= -1;

    memcpy(uniformBuffersMapped[currentFrame], &mvp, sizeof(MVP));
}

void Engine::recordCmdBuffer(VkCommandBuffer& cmdBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT means each recording of the command buffer will only be submitted once, and 
    // the command buffer will be reset and rerecorded again between each submission
    // VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT means the command buffer is secondary and will be executing within a single render pass
    // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT means the command buffer can be resubmitted to any queue of the same
    // queue family while it is already pending
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;
    VK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
    {
        VkRenderPassBeginInfo renderpassBeginInfo{};
        renderpassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderpassBeginInfo.renderPass = renderpass;
        renderpassBeginInfo.framebuffer = framebuffers[imageIndex];
        renderpassBeginInfo.renderArea.extent = swapchainExtent;
        renderpassBeginInfo.renderArea.offset = {0, 0};
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0}; // in Vulkan 1.0 indicates the far view plane, and 0.0 indicates the near view plane
        renderpassBeginInfo.clearValueCount = clearValues.size();
        renderpassBeginInfo.pClearValues = clearValues.data();
        // VK_SUBPASS_CONTENTS_INLINE means the render pass commands will be embedded in the primary buffer command itself
        // VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS means the render pass commands will be executed from secondary command buffer
        vkCmdBeginRenderPass(cmdBuffer, &renderpassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        {
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

            VkDeviceSize offset = {0};
            // this function is used to bind vertex buffer to bindings defined in graphics pipeline creation
            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &vertexBuffer, &offset);
            vkCmdBindIndexBuffer(cmdBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(swapchainExtent.width);
            viewport.height = static_cast<float>(swapchainExtent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
            VkRect2D scissor{};
            scissor.extent = swapchainExtent;
            scissor.offset = {0, 0};
            vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

            // vkCmdDraw(cmdBuffer, vertices.size(), 1, 0, 0);
            vkCmdDrawIndexed(cmdBuffer, indices.size(), 1, 0, 0, 0);
        }
        vkCmdEndRenderPass(cmdBuffer);
    }
    vkEndCommandBuffer(cmdBuffer);
}
void Engine::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer cmdBuffer = beginRecording(transferCmdPool);

    // image layout transitions are done using image memory barriers
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    // next two fields are used to transfer queue ownership
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (format==VK_FORMAT_D32_SFLOAT_S8_UINT || format==VK_FORMAT_D24_UNORM_S8_UINT) {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    } else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;
    barrier.srcAccessMask = 0; // what operations that involve the resource must happen before the barrier
    barrier.dstAccessMask = 0; // what operations that involve the resource must wait on the barrier

    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; // do not copy data until image layout is transitioned
        
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        // do not write or read from depth buffer until the image layout transition is done
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT; // do not execute this stage until image layout transition is done
    } else {
        throw std::runtime_error("Unsupported layout transition");
    }

    vkCmdPipelineBarrier(
        cmdBuffer, 
        srcStage, dstStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    stopRecording(cmdBuffer, transferCmdPool);
}