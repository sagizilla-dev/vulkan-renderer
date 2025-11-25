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
    createRenderpass();
    createFramebuffers();
    createGraphicsPipeline();
    createCommandPool(graphicsCmdPool, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilies.graphicsFamily.value());
    createCommandPool(transferCmdPool, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queueFamilies.transferFamily.value());
    createVertexBuffer();
    createIndexBuffer();
}
Engine::~Engine() {
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    vkDestroyCommandPool(device, transferCmdPool, nullptr);
    vkDestroyCommandPool(device, graphicsCmdPool, nullptr); // command buffers are freed when command pool is destroyed
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
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
    assemblyInfo.primitiveRestartEnable = VK_FALSE; // if set to true, then it is possible to break up lines and triangles in _STRIP topology

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
    rasterInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterInfo.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo msaaInfo{};
    msaaInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msaaInfo.sampleShadingEnable = VK_FALSE;
    msaaInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencilInfo{};
    depthStencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilInfo.depthTestEnable = VK_FALSE;
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
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
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
    std::vector<VkAttachmentDescription> attachmentDescriptions(1);
    attachmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachmentDescriptions[0].format = swapchainFormat;
    attachmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // upon loading we need to clear the image
    attachmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    // each subpass has an array of attachment references
    std::vector<VkAttachmentReference> attachmentReferences(1);
    attachmentReferences[0].attachment = 0; // index of the attachment in the attachment descriptions array
    attachmentReferences[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // layout that the image must have upon entering the subpass

    // subpass dependency takes care of image layout transitions for each subpass
    // renderpass by default has two implicit image layout transitiosn:
    // at the beginning of the renderpass and at the end
    // once renderpass starts, there is a chance swapchain image is actually still not available
    VkSubpassDependency subpassDependency{};
    subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL; // implicit subpass before or after the renderpass
    subpassDependency.dstSubpass = 0; // current subpass
    // what operations in srcSubpass must complete 
    subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // we wait for the swapchain to finish reading from the image
    subpassDependency.srcAccessMask = 0; // wait for all operations within the stage
    // what operations in dstSubpass wait
    subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // wait before clearing the image

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = attachmentReferences.size();
    // the index of the attachment in this array is referenced in the fragment shader as layout(location=X) out
    subpass.pColorAttachments = attachmentReferences.data();

    VkRenderPassCreateInfo renderpassInfo{};
    renderpassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderpassInfo.attachmentCount = 1;
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
        std::vector<VkImageView> attachments = {swapchainImageViews[i]};
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
    createFramebuffers();
}
void Engine::createVertexBuffer() {
    VkBuffer stageBuffer;
    VkDeviceMemory stageBufferMemory;
    VkDeviceSize vertexBufferSize = sizeof(vertices[0])*vertices.size();
    createBuffer(stageBuffer, stageBufferMemory, vertexBufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    // tie CPU memory to GPU memory and copy data
    void* data;
    vkMapMemory(device, stageBufferMemory, 0, vertexBufferSize, 0, &data);
    memcpy(data, vertices.data(), sizeof(vertices[0])*vertices.size());
    vkUnmapMemory(device, stageBufferMemory); // now the memory is transferred from CPU to GPU memory held by the buffer

    createBuffer(vertexBuffer, vertexBufferMemory, vertexBufferSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copyBuffer(stageBuffer, vertexBuffer, vertexBufferSize);

    vkDestroyBuffer(device, stageBuffer, nullptr);
    vkFreeMemory(device, stageBufferMemory, nullptr);
}
void Engine::createIndexBuffer() {
    VkBuffer stageBuffer;
    VkDeviceMemory stageBufferMemory;
    VkDeviceSize indexBufferSize = sizeof(indices[0])*indices.size();
    createBuffer(stageBuffer, stageBufferMemory, indexBufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    void* data;
    vkMapMemory(device, stageBufferMemory, 0, indexBufferSize, 0, &data);
    memcpy(data, indices.data(), sizeof(indices[0])*indices.size());
    vkUnmapMemory(device, stageBufferMemory);

    createBuffer(indexBuffer, indexBufferMemory, indexBufferSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copyBuffer(stageBuffer, indexBuffer, indexBufferSize);

    vkDestroyBuffer(device, stageBuffer, nullptr);
    vkFreeMemory(device, stageBufferMemory, nullptr);
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
    VkCommandBuffer cmdBuffer{};
    createCommandBuffer(&cmdBuffer, 1, transferCmdPool);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);
    {
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        vkCmdCopyBuffer(cmdBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    }
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
}

void Engine::recordCmdBuffer(VkCommandBuffer& cmdBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT means the command buffer will be rerecorded right after executing it once
    // VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT means the command buffer is secondary and will be executing within a single render pass
    // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT means the command buffer can be resubmitted while it is already pending
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
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderpassBeginInfo.clearValueCount = 1;
        renderpassBeginInfo.pClearValues = &clearColor;
        // VK_SUBPASS_CONTENTS_INLINE means the render pass commands will be embedded in the primary buffer command itself
        // VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS means the render pass commands will be executed from secondary command buffer
        vkCmdBeginRenderPass(cmdBuffer, &renderpassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        {
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

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