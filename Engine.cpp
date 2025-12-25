#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include "Engine.hpp"

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (key == GLFW_KEY_M && action == GLFW_PRESS) {
        Engine* eng = (Engine*)glfwGetWindowUserPointer(window);
        eng->meshShadersEnabled = !eng->meshShadersEnabled;
    }
}

Engine::Engine() {
    createWindow();
    loadModel();
    createMeshlets();
    createInstance();
    createSurface();
    createDevice();
    createSwapchain();
    createCommandPool(graphicsCmdPool, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilies.graphicsFamily.value());
    createCommandPool(transferCmdPool, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queueFamilies.transferFamily.value());
    createDepthBuffer();
    createColorBuffer();
    createRenderpass();
    createFramebuffers();
    createTextureImage();
    createTextureSampler();
    createVertexBuffer();
    createIndexBuffer();
    createMeshletDataBuffer();
    createMeshletBuffer();
    createTransformBuffers();
    createIndirectBuffer();
    createShaders();
    createDescriptorSetLayout();
    createDescriptorUpdateTemplate();
    createDescriptorPool();
    createDescriptorSets();
    createGraphicsPipeline();
    createQueryPools();
}
Engine::~Engine() {
    vkDestroyDescriptorUpdateTemplate(device, descriptorUpdateTemplate, nullptr);
    vkDestroyBuffer(device, meshletDataBuffer, nullptr);
    vkFreeMemory(device, meshletDataBufferMemory, nullptr);
    vkDestroyBuffer(device, meshletBuffer, nullptr);
    vkFreeMemory(device, meshletBufferMemory, nullptr);
    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);
    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(device, transformBuffers[i], nullptr);
        vkFreeMemory(device, transformBuffersMemory[i], nullptr);
        vkDestroyQueryPool(device, queryPools[i], nullptr);
    }
    vkDestroyBuffer(device, indirectBuffer, nullptr);
    vkFreeMemory(device, indirectBufferMemory, nullptr);
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    vkDestroyCommandPool(device, transferCmdPool, nullptr);
    vkDestroyCommandPool(device, graphicsCmdPool, nullptr); // command buffers are freed when command pool is destroyed
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipeline(device, meshGraphicsPipeline, nullptr);
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

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pDevice, &props);
    // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1
    float timestampPeriod = props.limits.timestampPeriod;
    int frameCounter = 0;
    
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // wait until command buffer is ready to be rerecorded
        vkWaitForFences(device, 1, &cmdBufferReady[currentFrame], VK_TRUE, ~0ull);

        if (frameCounter >= MAX_FRAMES_IN_FLIGHT) {
            uint64_t timestamps[2];
            // get the result of timestamp queries
            // only works successfully if the queries were reset before using them
            VkResult result = vkGetQueryPoolResults(
                device, queryPools[currentFrame],
                0, 2,
                sizeof(timestamps), timestamps,
                sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT
            );
            
            if (result == VK_SUCCESS) {
                uint64_t delta = timestamps[1] - timestamps[0];
                float gpuTimeMs = (delta * timestampPeriod) / 1000000.0f;
                gpuTimes.push_back(gpuTimeMs);
                // output stats every 200 frames
                if (gpuTimes.size()==200) {
                    char title[256];
                    float avgGpuTime = 0.0f;
                    float avgCpuTime = 0.0f;
                    for (int i=0; i<200; i++) {
                        avgGpuTime+=gpuTimes[i]/200;
                        avgCpuTime+=cpuTimes[i]/200;
                    }
                    double trianglesPerSec = (indices.size()/3) / (avgGpuTime*1e-3);
                    snprintf(title, sizeof(title), "GPU Time: %.3f ms, CPU Time: %.3f ms, %i meshlets, %i triangles, %.2fB tri/sec", 
                            avgGpuTime, avgCpuTime, int(meshlets.size()), int(indices.size())/3, 
                            DRAW_COUNT * trianglesPerSec*1e-9);
                    glfwSetWindowTitle(window, title);
                    gpuTimes.clear();
                }
            }
        }
        
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

        // reset the fence only after acquiring the next image
        // it is important to avoid deadlock, otherwise if swapchain is 
        // recreated, fence is reset but we are still waiting on it
        vkResetFences(device, 1, &cmdBufferReady[currentFrame]);
        
        auto cpuStart = std::chrono::high_resolution_clock::now();
        vkResetCommandBuffer(cmdBuffer[currentFrame], 0);
        
        updateTransforms(currentFrame);
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

        auto cpuEnd = std::chrono::high_resolution_clock::now();
        float cpuTimeMs = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();
        cpuTimes.push_back(cpuTimeMs);

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
        frameCounter++;
    }
    vkDeviceWaitIdle(device); // wait until all GPU operations are done so that it's safe to destroy objects
    for (int i = 0; i<MAX_FRAMES_IN_FLIGHT; i++) destroySemaphore(imageAvailable[i]);
    for (int i = 0; i<MAX_FRAMES_IN_FLIGHT; i++) destroySemaphore(renderDone[i]);
    for (int i = 0; i<MAX_FRAMES_IN_FLIGHT; i++) destroyFence(cmdBufferReady[i]);
}

void Engine::createWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetWindowUserPointer(window, this);
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
            vertex.vx = (attrib.vertices[3 * index.vertex_index + 0]);
            vertex.vy = (attrib.vertices[3 * index.vertex_index + 1]);
            vertex.vz = (attrib.vertices[3 * index.vertex_index + 2]);

            glm::vec3 uncompressed = glm::vec3(0.0f, 0.0f, 0.0f);
            if (!attrib.normals.empty()) {
                uncompressed[0] = attrib.normals[3 * index.normal_index + 0];
                uncompressed[1] = attrib.normals[3 * index.normal_index + 1];
                uncompressed[2] = attrib.normals[3 * index.normal_index + 2];
            }
            uncompressed = glm::normalize(uncompressed); // now the normal vector is between -1 and 1
            vertex.nx = uint8_t((uncompressed[0]*0.5f + 0.5f)*255.0f); // now the vector is within [0, 255]
            vertex.ny = uint8_t((uncompressed[1]*0.5f + 0.5f)*255.0f);
            vertex.nz = uint8_t((uncompressed[2]*0.5f + 0.5f)*255.0f);

            if (!attrib.texcoords.empty()) {
                vertex.tu = floatToHalf(attrib.texcoords[2 * index.texcoord_index + 0]);
                vertex.tv = floatToHalf(1.0f - attrib.texcoords[2 * index.texcoord_index + 1]);
            }
            
            if (uniqueVertices.count(vertex)==0) {
                uniqueVertices[vertex] = vertices.size();
                vertices.push_back(vertex);
            }

            indices.push_back(uniqueVertices[vertex]);
        }
    }
    // geometry optimization here is pretty important.
    // first of all, we need to reorder indices inside the index buffer so that consecutive triangles share vertices.
    // this maximizes vertex cache efficiency, i.e vertices that have already been transformed are stored, 
    // and retrieval doesn't cost us anything.
    optimizeGeometry();
    // secondly, we need to reorder vertex buffer so that vertices that are accessed one after another are stored
    // close to each other.
    // this maximizes memory locality and global memory coalescing
    
    // for geometry and meshlets, bigger meshlets = fewer unique vertices per meshlet = fewer shader invocations.
    // it is important to build meshlets in a way that maximizes the number of triangles per meshlets,
    // not vertices per meshlets
    // otherwise we end up with scatterred triangles belonging to the same meshlet
    // this is actually pretty bad for the current pipeline since we need to perform cone culling
    // which is pretty much useless when meshlets are not contiguous (cones are just too wide).
    // in theory, Forsyth's algorithm is supposed to give us an index buffer where each triangle is connected 
    // to another, therefore creating a big chain of triangles wrapping the entire mesh
    // unfortunately, this is not the case because .obj file contains geometrical duplicates, i.e
    // vertices that share the same position but have different normal or UV coordinates
    // those isolated triangles' vertices have very low valence (1), therefore making Forsyth's push them to
    // the very beginning. There are also vertices that have valence of 2, which is also very low.
}
// calculate vertex score based on its cache position (higher for vertices recently used)
// and valence (prefer vertices shared by fewer remaining triangles so that the vertex can be evicted
// from cache sooner)
float Engine::computeVertexScore(int cachePosition, int valence) {
    // cache is LRU, index 31 is the oldest, 0 is the newest
    
    if (valence == 0) {
        return -1.0f; // dead vertex, no triangles left
    }
    
    // calcualte score based on its cache position
    float cacheScore = 0.0f;
    if (cachePosition < 0) {
        cacheScore = 0.0f; // not in cache
    } else if (cachePosition < 3) {
        cacheScore = 0.75f; // this vertex is hot, it's been used recently
    } else { // in cache but old
        const int CACHE_SIZE = 32;
        float normalizedPosition = (cachePosition - 3)*(1.0f / (CACHE_SIZE - 3)); // map [3, 31] to [0, 1]
        // exponential decay, i.e the older vertex, the lower the score
        // 1.5 is the decay power, meaning the decay is not linear
        cacheScore = std::pow(1.0f - normalizedPosition, 1.5f); 
    }

    // calculate score based on its valence (fewer unprocessed triangles means higher the score)
    // valence boost scale is 2.0f, which defines the strength of the heuristic
    // valence boost power is 0.5f, which produces diminishing returns for vertices that appear in
    // many triangles
    float valenceScore = 2.0f * std::pow(float(valence), -0.5f);
    return cacheScore+valenceScore;
}

// Forsyth's algorithm
void Engine::optimizeGeometry() {
    const uint32_t triangleCount = indices.size()/3;
    const uint32_t vertexCount = vertices.size();
    // connectivity graph
    // maps a vertex index (value in the index buffer) to a list of triangle indices that use it
    std::vector<std::vector<uint32_t>> vertexToTriangles(vertices.size());

    for (size_t i=0; i<triangleCount; i++) {
        // get vertex indices
        uint32_t v0 = indices[i*3+0];
        uint32_t v1 = indices[i*3+1];
        uint32_t v2 = indices[i*3+2];
        
        // for each vertex push the triangle index
        vertexToTriangles[v0].push_back(i);
        vertexToTriangles[v1].push_back(i);
        vertexToTriangles[v2].push_back(i);
    }

    // maps a vertex index to how many unprocessed triangles still need that vertex
    // basically maps vertex index to its valence score (how many unprocessed triangles use this vertex)
    std::vector<int> unprocessedTriangles(vertexCount);
    // maps a vertex index to its position inside the cache
    std::vector<int> vertexCachePosition(vertexCount, -1);
    // maps a vertex index to its score
    std::vector<float> vertexScore(vertexCount);

    // maps a triangle index to its score (sum of scores of all its vertices)
    std::vector<float> triangleScore(triangleCount);
    // whether a triangle has been processed
    std::vector<bool> isTriangleFinished(triangleCount, false);

    // initialize valence: how many unprocessed (at this stage all of them) triangles require this vertex
    for (uint32_t i=0; i<vertexCount; i++) {
        unprocessedTriangles[i] = vertexToTriangles[i].size();
    }
    // calculate initial scores
    for (uint32_t i=0; i<vertexCount; i++) {
        vertexScore[i] = computeVertexScore(vertexCachePosition[i], unprocessedTriangles[i]);
    }
    for (uint32_t i=0; i<triangleCount; i++) {
        triangleScore[i] = vertexScore[indices[i*3+0]] + vertexScore[indices[i*3+1]] + vertexScore[indices[i*3+2]]; 
    }

    // simulate LRU cache that stores vertex indices
    std::vector<int> cache;
    cache.reserve(32);

    std::vector<uint32_t> reorderedIndexBuffer;
    reorderedIndexBuffer.reserve(indices.size());

    int nextBestTriangle = -1;

    for (uint32_t i=0; i<triangleCount; i++) {
        // find the best triangle
        int bestTriangle = nextBestTriangle;
        float bestScore = bestTriangle >= 0 ? triangleScore[bestTriangle] : std::numeric_limits<float>::min();
        if (bestTriangle < 0 || isTriangleFinished[bestTriangle]) { // no candidate or the triangle has been processed
            // this branch is taken either at the very beginning, or if our cache got old, i.e there were
            // no vertices in cache that would lead us to an unprocessed triangle
            // this happens when we processed all isolated triangles, or 
            // we processed a region that is surrounded by isolated triangles, or
            // the bridge between two regions got evicted before we finished the first region
            bestTriangle = -1;
            bestScore = std::numeric_limits<float>::min();
            for (uint32_t j=0; j<triangleCount; j++) {
                if (triangleScore[j] > bestScore && !isTriangleFinished[j]) {
                    bestScore = triangleScore[j];
                    bestTriangle = j;
                }
            }
        }

        uint32_t v0 = indices[bestTriangle*3+0];
        uint32_t v1 = indices[bestTriangle*3+1];
        uint32_t v2 = indices[bestTriangle*3+2];
        reorderedIndexBuffer.push_back(v0);
        reorderedIndexBuffer.push_back(v1);
        reorderedIndexBuffer.push_back(v2);
        isTriangleFinished[bestTriangle] = true;

        // update vertex scores and cache
        std::vector<uint32_t> affectedVertices;
        for (uint32_t v: {v0, v1, v2}) {
            // update cache
            auto it = std::find(cache.begin(), cache.end(), v);
            if (it != cache.end()) {
                cache.erase(it);
            }
            cache.insert(cache.begin(), v);

            if (cache.size() > 32) {
                int evictedVertex = cache.back();
                cache.pop_back();
                vertexCachePosition[evictedVertex] = -1;
                affectedVertices.push_back(evictedVertex);
            }

            // this vertex now has fewer unprocessed triangles that use it 
            affectedVertices.push_back(v);
            unprocessedTriangles[v]--;
        }

        // update cache positions
        for (size_t i=0; i<cache.size(); i++) {
            vertexCachePosition[cache[i]] = i;
        }

        // recalculate scores of affected triangles
        for (uint32_t v: affectedVertices) {
            vertexScore[v] = computeVertexScore(vertexCachePosition[v], unprocessedTriangles[v]);
            for (uint32_t triangle: vertexToTriangles[v]) {
                if (!isTriangleFinished[triangle]) {
                    triangleScore[triangle] = vertexScore[indices[triangle*3+0]] +
                                            vertexScore[indices[triangle*3+1]] + 
                                            vertexScore[indices[triangle*3+2]]; 
                }
            }
        }

        nextBestTriangle = -1;
        bestScore = std::numeric_limits<float>::min();
        // search only cache-adjacent triangles
        for (int v: cache) {
            for (uint32_t triangle: vertexToTriangles[v]) {
                if (!isTriangleFinished[triangle] && triangleScore[triangle] > bestScore) {
                    bestScore = triangleScore[triangle];
                    nextBestTriangle = triangle;
                }
            }
        }
    }
    indices = reorderedIndexBuffer;
}
void Engine::buildMeshletCon(Meshlet& meshlet, std::vector<uint32_t> globalIndices, std::vector<uint8_t> localIndices) {
    // first we need to calculate triangle normals
    float normals[124][3];
    for (uint8_t i=0; i<meshlet.triangleCount; i++) {
        uint8_t localIndex0 = localIndices[i*3+0];
        uint8_t localIndex1 = localIndices[i*3+1];
        uint8_t localIndex2 = localIndices[i*3+2];

        const Vertex& v0 = vertices[globalIndices[localIndex0]];
        const Vertex& v1 = vertices[globalIndices[localIndex1]];
        const Vertex& v2 = vertices[globalIndices[localIndex2]];

        // we return to full precision as half precision messes up the cull test
        glm::vec3 p0 = glm::vec3((v0.vx), (v0.vy), (v0.vz));
        glm::vec3 p1 = glm::vec3((v1.vx), (v1.vy), (v1.vz));
        glm::vec3 p2 = glm::vec3((v2.vx), (v2.vy), (v2.vz));

        glm::vec3 p10 = p1-p0;
        glm::vec3 p20 = p2-p0;

        // direction of normal is defined by the winding order
        glm::vec3 normal = glm::normalize(glm::cross(p10, p20));
        normals[i][0] = normal.x;
        normals[i][1] = normal.y;
        normals[i][2] = normal.z;
    }

    // average normal (cone's axis) for the entire meshlet
    glm::vec3 avgNormal = glm::vec3(0.0f);
    for (uint8_t i=0; i<meshlet.triangleCount; i++) {
        avgNormal+=glm::vec3(normals[i][0], normals[i][1], normals[i][2]);
    }
    avgNormal = glm::normalize(avgNormal);

    // the cosine of the angle between average normal and furthest triangle normal.
    // if this value is 0, some normals are orthogonal to other normals
    // the bigger this value, the smaller the spread of the cone is, which is good for culling!
    float halfAngle = 1.0f;
    for (uint8_t i=0; i<meshlet.triangleCount; i++) {
        float dp = normals[i][0]*avgNormal.x + normals[i][1]*avgNormal.y + normals[i][2]*avgNormal.z;
        halfAngle = std::min(halfAngle, dp);
    }

    // to prove the meshlet is not visible, we need to check whether all normals within the cone
    // point away from the camera, i.e dot product between view vector and any normal is negative as the angle
    // is more than 90 degrees
    // since the angle between average normal and furthest normal is A, we can
    // write the inequality as angle(View, AvgNormal) > 90 deg + A.
    // if it is true, we can cull the entire meshlet
    // in other words, the cone test is: dot(View, AvgNormal) < cos(90+A), or 
    // dot(View, AvgNormal) < -sin(A)
    // note that if the cone's half angle is more than 90 degrees, we cannot
    // cull this meshlet, which is a usual case for high poly meshes
    // if that's the case, clip the angle to 90 degrees
    float coneW = halfAngle < 0.0f ? 1.0f : sqrtf(1-halfAngle*halfAngle);
    meshlet.cone[0] = avgNormal.x;
    meshlet.cone[1] = avgNormal.y;
    meshlet.cone[2] = avgNormal.z;
    meshlet.cone[3] = coneW;

    for (uint8_t i=0; i<meshlet.vertexCount; i++) {
        meshlet.coneApex[0]+=((vertices[globalIndices[i]].vx))/float(meshlet.vertexCount);
        meshlet.coneApex[1]+=((vertices[globalIndices[i]].vy))/float(meshlet.vertexCount);
        meshlet.coneApex[2]+=((vertices[globalIndices[i]].vz))/float(meshlet.vertexCount);
    }

    // calculate radius of the bounding sphere as a max distance to vertices from apex
    float maxDistanceSqr = 0.0f;
    for (int i=0; i<meshlet.vertexCount; i++) {
        float dx = vertices[globalIndices[i]].vx - meshlet.coneApex[0];
        float dy = vertices[globalIndices[i]].vy - meshlet.coneApex[1];
        float dz = vertices[globalIndices[i]].vz - meshlet.coneApex[2];
        maxDistanceSqr = std::max(maxDistanceSqr, dx*dx+dy*dy+dz*dz);
    }
    meshlet.radius = sqrtf(maxDistanceSqr);
}
void Engine::createMeshlets() {
    Meshlet meshlet{};
    meshlet.dataOffset = meshletData.size();
    // this is an array of indices that point to global vertex array, 
    // range  of values is [0, vertices.size()), max size is 64
    std::vector<uint32_t> currentVertices;
    // this is an array of indices that point to local vertex indices (currentVertices), 
    // range is [0, meshlet.vertexCount), max size is 124*3
    std::vector<uint8_t> currentIndices;
    // this maps vertex to its status inside the meshlet, i.e whether it's already been added or not
    // if it has been added, it contains the local vertex index
	std::vector<uint8_t> meshletVertices(vertices.size(), 0xff);
	for (size_t i = 0; i < indices.size(); i += 3) {
        // these are pointers to global array of vertices
		unsigned int globalIndex0 = indices[i + 0];
		unsigned int globalIndex1 = indices[i + 1];
		unsigned int globalIndex2 = indices[i + 2];

        // these are pointers to local array of vertex indices
		uint8_t& localIndex0 = meshletVertices[globalIndex0];
		uint8_t& localIndex1 = meshletVertices[globalIndex1];
		uint8_t& localIndex2 = meshletVertices[globalIndex2];

        int newVerticesCount = (localIndex0 == 0xff) + (localIndex1 == 0xff) + (localIndex2 == 0xff);
		if (currentVertices.size() + newVerticesCount > 64 || currentIndices.size()/3 >= 124) {
            // configure the meshlet
            meshlet.triangleCount = currentIndices.size()/3;
            meshlet.vertexCount = currentVertices.size();
            // pad the indices to fit into uint32_t array
            while (currentIndices.size()%4!=0) {
                currentIndices.push_back(0);
            } 
            meshletData.insert(meshletData.end(), currentVertices.begin(), currentVertices.end());
            uint32_t* packedCurrentIndices = reinterpret_cast<uint32_t*>(currentIndices.data());
            for (size_t j=0; j<currentIndices.size()/4; j++) {
                meshletData.push_back(packedCurrentIndices[j]);
            }
            buildMeshletCon(meshlet, currentVertices, currentIndices);
            meshlets.push_back(meshlet);
            // reset for the next meshlet
			for (size_t j = 0; j < currentVertices.size(); j++) {
				meshletVertices[currentVertices[j]] = 0xff;
            }
            currentVertices.clear();
            currentIndices.clear();
			meshlet = {};
            meshlet.dataOffset = meshletData.size();
		}
        // if av == 0xff, it means the vertex is not in the meshlet array and we need to add it
		if (localIndex0 == 0xff) {
			localIndex0 = currentVertices.size();
            currentVertices.push_back(globalIndex0);
		}
		if (localIndex1 == 0xff) {
            localIndex1 = currentVertices.size();
            currentVertices.push_back(globalIndex1);
		}
		if (localIndex2 == 0xff) {
            localIndex2 = currentVertices.size();
            currentVertices.push_back(globalIndex2);
		}
        currentIndices.push_back(localIndex0);
        currentIndices.push_back(localIndex1);
        currentIndices.push_back(localIndex2);
	}
    // the last meshlet may not have hit any limit on vertices or triangles
	if (currentIndices.size()!=0) {
		meshlet.triangleCount = currentIndices.size()/3;
        meshlet.vertexCount = currentVertices.size();
        while (currentIndices.size()%4!=0) {
            currentIndices.push_back(0);
        } 
        meshletData.insert(meshletData.end(), currentVertices.begin(), currentVertices.end());
        uint32_t* packedCurrentIndices = reinterpret_cast<uint32_t*>(currentIndices.data());
        for (size_t j=0; j<currentIndices.size()/4; j++) {
            meshletData.push_back(packedCurrentIndices[j]);
        }
        buildMeshletCon(meshlet, currentVertices, currentIndices);
        meshlets.push_back(meshlet);
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
            msaaSamples = getMaxSampleCount();
            break;
        }
    }
    if (pDevice == VK_NULL_HANDLE) throw std::runtime_error("No suitable device found");

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.enabledExtensionCount = deviceExtensions.size();
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.features.geometryShader = VK_TRUE;
    features.features.samplerAnisotropy = VK_TRUE;
    features.features.multiDrawIndirect = VK_TRUE;
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.storageBuffer8BitAccess = VK_TRUE;
    features12.shaderInt8 = VK_TRUE;
    VkPhysicalDeviceMeshShaderFeaturesNV meshFeatures{};
    meshFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV;
    meshFeatures.meshShader = VK_TRUE;
    meshFeatures.taskShader = VK_TRUE;
    // this feature allows us to use LocalSizeId to specify the local workgroup size
    VkPhysicalDeviceMaintenance4Features maintenanceFeatures{};
    maintenanceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    maintenanceFeatures.maintenance4 = VK_TRUE;
    VkPhysicalDeviceVulkan11Features features11{};
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features11.storageBuffer16BitAccess = VK_TRUE;
    features11.shaderDrawParameters = VK_TRUE;
    maintenanceFeatures.pNext = &features11;
    meshFeatures.pNext = &maintenanceFeatures;
    features12.pNext = &meshFeatures;
    features.pNext = &features12;
    deviceInfo.pNext = &features;
    
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
        createImageView(swapchainFormat, swapchainImages[i], 1, VK_IMAGE_ASPECT_COLOR_BIT, swapchainImageViews[i]);
    }
}
void Engine::createImageView(VkFormat format, VkImage& image, uint32_t mipLevels, VkImageAspectFlags aspectMask, VkImageView& imageView) {
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
    imageViewInfo.subresourceRange.levelCount = mipLevels;
    VK_CHECK(vkCreateImageView(device, &imageViewInfo, nullptr, &imageView));
}
void Engine::createShaders() {
    // mostly used to create shader modules and parse SPIR-V
    createShader(shaders[0], VERT_SHADER_PATH);
    createShader(shaders[1], FRAG_SHADER_PATH);
    createShader(shaders[2], TASK_SHADER_PATH);
    createShader(shaders[3], MESH_SHADER_PATH);
    createShader(shaders[4], FRAG_SHADER_PATH);
}
void Engine::createDescriptorSetLayout() {
    // this function creates descriptor set layout, which specifies what type of resources
    // are to be passed into the shaders, same way render pass defines what type of attachments to expect
    // descriptor sets define data itself, same way framebuffers define exact attachments
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings(descriptorResourceInfos.size());
    int i = 0;
    for (auto& descriptorResourceInfo: descriptorResourceInfos) {
        layoutBindings[i].binding = descriptorResourceInfo.binding; // referenced in the shader as layout(binding = X)
        layoutBindings[i].descriptorCount = 1; // descriptor can be an array
        layoutBindings[i].descriptorType = descriptorResourceInfo.type;
        layoutBindings[i].stageFlags = descriptorResourceInfo.stage;
        i++;
    }
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{};
    descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutInfo.bindingCount = layoutBindings.size();
    descriptorSetLayoutInfo.pBindings = layoutBindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutInfo, nullptr, &descriptorSetLayout));
}
void Engine::createDescriptorUpdateTemplate() {
    VkDescriptorUpdateTemplateCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
    info.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    info.descriptorSetLayout = descriptorSetLayout;
    info.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
    
    std::vector<VkDescriptorUpdateTemplateEntry> entries(descriptorResourceInfos.size());
    int i = 0;
    // here the descriptor resource infos come in binding order, i.e binding = 0 comes first
    for (auto& descriptorResourceInfo: descriptorResourceInfos) {
        entries[i].descriptorCount = 1; // in case the descriptor is an array
        entries[i].descriptorType = descriptorResourceInfo.type;
        entries[i].dstBinding = descriptorResourceInfo.binding;
        entries[i].dstArrayElement = 0;
        // the next two fields specify how to map the data that is going to be updated
        // since the function vkUpdateDescriptorSetWithTemplate() accepts void* data, we can specify our own
        // data class to hold the data, so we need to provide details about it
        entries[i].stride = sizeof(DescriptorData); // spacing between consecutive descriptors (same for all entries since array is tightly packed)
        entries[i].offset = sizeof(DescriptorData)*i; // offset in the provided data array, is used to find the correct descriptor data
        i++;
    }

    info.descriptorUpdateEntryCount = entries.size();
    info.pDescriptorUpdateEntries = entries.data();
    info.set = 0;

    VK_CHECK(vkCreateDescriptorUpdateTemplate(device, &info, nullptr, &descriptorUpdateTemplate));
}
void Engine::createDescriptorPool() {
    // so the transform buffer is now a storage buffer instead of a uniform
    // since we cannot make a dynamic array inside a uniform block
    std::vector<VkDescriptorPoolSize> poolSizes(2);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // number of descriptors of a specific type
    poolSizes[0].descriptorCount = MAX_FRAMES_IN_FLIGHT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    // Meshlets, meshlet's data (vertices and indices) and vertices, and transformations
    poolSizes[1].descriptorCount = MAX_FRAMES_IN_FLIGHT*4;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    // total number of descriptor sets that are to be allocated
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}
void Engine::createDescriptorSets() {
    // we need to supply a layout for every descriptor set to be created
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
    for (size_t i=0; i<descriptorSets.size(); i++) {
        std::vector<DescriptorData> data(5);
        // image and sampler
        data[0].imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        data[0].imageInfo.imageView = textureImageView;
        data[0].imageInfo.sampler = textureSampler;
        // vertex buffer
        data[1].bufferInfo.buffer = vertexBuffer;
        data[1].bufferInfo.offset = 0;
        data[1].bufferInfo.range = sizeof(vertices[0])*vertices.size();
        // meshlet buffer
        data[2].bufferInfo.buffer = meshletBuffer;
        data[2].bufferInfo.offset = 0;
        data[2].bufferInfo.range = sizeof(meshlets[0])*meshlets.size();
        // meshlet data buffer
        data[3].bufferInfo.buffer = meshletDataBuffer;
        data[3].bufferInfo.offset = 0;
        data[3].bufferInfo.range = sizeof(meshletData[0])*meshletData.size();
        // transform data buffer
        data[4].bufferInfo.buffer = transformBuffers[i];
        data[4].bufferInfo.offset = 0;
        data[4].bufferInfo.range = sizeof(Transform)*DRAW_COUNT;

        // vkUpdateDescriptorSets(device, writeDescriptors.size(), writeDescriptors.data(), 0, nullptr);
        vkUpdateDescriptorSetWithTemplate(device, descriptorSets[i], descriptorUpdateTemplate, data.data());
    }
}
void Engine::createGraphicsPipeline() {
    std::vector<VkPipelineShaderStageCreateInfo> shaderStageInfos(5);
    for (size_t i=0; i<shaderStageInfos.size(); i++) {
        shaderStageInfos[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfos[i].module = shaders[i].module;
        shaderStageInfos[i].pName = "main";
        shaderStageInfos[i].stage = shaders[i].stage;
        shaderStageInfos[i].pSpecializationInfo = VK_NULL_HANDLE; // allows us to specify values for shader constants
    }

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
    // // attribute description just describes data inside the vertex
    // auto attributes = Vertex::getAttributeDescription();
    // vertexInputInfo.vertexAttributeDescriptionCount = attributes.size();
    // vertexInputInfo.pVertexAttributeDescriptions = attributes.data();
    // // binding is spacing between data and whether the data is per-vertex or per-instance
    // auto bindings = Vertex::getBindingDescription();
    // vertexInputInfo.vertexBindingDescriptionCount = 1;
    // vertexInputInfo.pVertexBindingDescriptions = &bindings;

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
    msaaInfo.rasterizationSamples = msaaSamples;

    VkPipelineDepthStencilStateCreateInfo depthStencilInfo{};
    depthStencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilInfo.depthTestEnable = VK_TRUE; // enable depth testing
    depthStencilInfo.depthWriteEnable = VK_TRUE; // enable writing to the depth buffer
    // lower depth = closer, but for reverse-z it's the opposite, so we use great-or-equal
    depthStencilInfo.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
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
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = 0;
    for (const auto& shader: shaders) {
        if (shader.hasPushConstants) {
            pushConstantRange.stageFlags |= shader.stage;
        }
    }
    pushConstantRange.size = sizeof(Globals);
    pushConstantRange.offset = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
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
    pipelineInfo.stageCount = 3;
    pipelineInfo.pStages = shaderStageInfos.data()+2;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &meshGraphicsPipeline));

    // we can delete shader module right away since compilation and
    // linking of shaders are done when pipeline is created
    for (auto& shader: shaders) {
        vkDestroyShaderModule(device, shader.module, nullptr);
    }
}
void Engine::createShader(Shader& shader, std::string path) {
    // SPIR-V is read as a vector of bytes (char), so we need to convert it
    // to a vector of words (std::vector<uint32_t>, so 4 bytes per word)
    auto bytes = readFile(path);
    uint32_t wordCount = bytes.size()/4;
    std::vector<uint32_t> byteCode(wordCount);
    memcpy(byteCode.data(), bytes.data(), bytes.size());
    shader.code = byteCode;
    shader.codeSize = bytes.size();
    shader.hasPushConstants = false;

    // parse SPIR-V to extract information about descriptor sets and what stage the shader belongs to
    parseSPIRV(shader);

    createShaderModule(shader);
}
void Engine::createShaderModule(Shader& shader) {
    VkShaderModuleCreateInfo shaderModuleInfo{};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = shader.codeSize; // size in bytes
    shaderModuleInfo.pCode = shader.code.data(); // must be an array of uint32_t
    VK_CHECK(vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shader.module));
}
void Engine::createRenderpass() {
    // all attachments, i.e color, depth, etc, are passed from the framebuffer in
    // the same order as they were defined during framebuffer creation
    // so this array must have the same order
    std::vector<VkAttachmentDescription> attachmentDescriptions(3);
    // color buffer
    attachmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachmentDescriptions[0].format = swapchainFormat;
    attachmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // upon loading we need to clear the image
    attachmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentDescriptions[0].samples = msaaSamples;
    attachmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // depth buffer
    attachmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachmentDescriptions[1].format = findDepthFormat();
    attachmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // we don't store the depth buffer after rendering
    attachmentDescriptions[1].samples = msaaSamples;
    attachmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // resolve buffer (swapchain image)
    attachmentDescriptions[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescriptions[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachmentDescriptions[2].format = swapchainFormat;
    attachmentDescriptions[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescriptions[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentDescriptions[2].samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescriptions[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescriptions[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    // each subpass has an array of attachment references
    std::vector<VkAttachmentReference> attachmentReferences(3);
    // color buffer
    attachmentReferences[0].attachment = 0; // index of the attachment in the attachment descriptions array
    attachmentReferences[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // layout that the image must have upon entering the subpass
    // depth buffer
    attachmentReferences[1].attachment = 1;
    attachmentReferences[1].layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    // resolve buffer
    attachmentReferences[2].attachment = 2;
    attachmentReferences[2].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

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

    // technically we don't have to provide subpass dependency for depth buffer since we can manually transition the layout
    subpassDependency.srcStageMask |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT; // wait for previous frame's depth buffer writes to finish
    subpassDependency.srcAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; // wait for the previous frame's depth buffer update
    subpassDependency.dstStageMask |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    subpassDependency.dstAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; // wait before clearing the buffer

    subpassDependency.srcAccessMask |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // wait for previous frame's color buffer writes to finish

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    // the index of the attachment in this array is referenced in the fragment shader as layout(location=X) out
    subpass.pColorAttachments = &attachmentReferences[0];
    subpass.pDepthStencilAttachment = &attachmentReferences[1];
    subpass.pResolveAttachments = &attachmentReferences[2];

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
        std::vector<VkImageView> attachments = {colorBufferImageView, depthBufferImageView, swapchainImageViews[i]};
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
    // VK_FENCE_CREATE_SIGNALED_BIT flag means fence is created as already signaled
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = flags;
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));
}
void Engine::destroyFence(VkFence& fence) {
    vkDestroyFence(device, fence, nullptr);
}
void Engine::cleanupSwapchain() {
    vkDestroyImage(device, colorBuffer, nullptr);
    vkFreeMemory(device, colorBufferMemory, nullptr);
    vkDestroyImageView(device, colorBufferImageView, nullptr);
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
    createColorBuffer();
    createFramebuffers();
}
void Engine::createVertexBuffer() {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize vertexBufferSize = sizeof(vertices[0])*vertices.size();
    createBuffer(stagingBuffer, stagingBufferMemory, vertexBufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        // these flags are important
        // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT specifies that we can map GPU memory to CPU memory and therefore
        // copy data from host to device
        //  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT helps us avoid the case when the data is written into cache
        // but has not been flushed yet
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    // map CPU memory to GPU memory and copy data
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, vertexBufferSize, 0, &data);
    memcpy(data, vertices.data(), sizeof(vertices[0])*vertices.size());
    vkUnmapMemory(device, stagingBufferMemory); // now the memory is transferred from CPU to GPU

    createBuffer(vertexBuffer, vertexBufferMemory, vertexBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
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
void Engine::createMeshletDataBuffer() {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize meshletDataBufferSize = sizeof(meshletData[0])*meshletData.size();
    createBuffer(stagingBuffer, stagingBufferMemory, meshletDataBufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, meshletDataBufferSize, 0, &data);
    memcpy(data, meshletData.data(), sizeof(meshletData[0])*meshletData.size());
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(meshletDataBuffer, meshletDataBufferMemory, meshletDataBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT| VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copyBuffer(stagingBuffer, meshletDataBuffer, meshletDataBufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}
void Engine::createMeshletBuffer() {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize meshletBufferSize = sizeof(meshlets[0])*meshlets.size();
    createBuffer(stagingBuffer, stagingBufferMemory, meshletBufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, meshletBufferSize, 0, &data);
    memcpy(data, meshlets.data(), sizeof(meshlets[0])*meshlets.size());
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(meshletBuffer, meshletBufferMemory, meshletBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT| VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copyBuffer(stagingBuffer, meshletBuffer, meshletBufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}
void Engine::createTransformBuffers() {
    transformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    transformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    transformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    VkDeviceSize transformBuffersSize = sizeof(Transform)*DRAW_COUNT;
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(transformBuffers[i], transformBuffersMemory[i], transformBuffersSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        // persistent mapping
        vkMapMemory(device, transformBuffersMemory[i], 0, transformBuffersSize, 0, &transformBuffersMapped[i]);
    }
}
void Engine::createIndirectBuffer() {
    // this stores the indirect buffer information for both vertex pipeline and mesh pipeline
    std::vector<DrawIndirect> drawsIndirect(DRAW_COUNT);
    // the number of task workgroups to dispatch per instance
    // task workgroup consists of 32 threads
    // each thread must work on its own meshlet, and each workgroup's main thread dispatches
    // mesh shaders (the number of dispatches depends on how many meshlets were culled)
    // if the number of meshlets was 31, we'd still need to launch 1 task shader
    // if the number of meshlets was 33, we'd need to launch 2 task shaders
    uint32_t taskWorkgroups = (meshlets.size() + 31) / 32;
    for (int i=0; i<DRAW_COUNT; i++) {
        drawsIndirect[i].commandIndirect.indexCount = indices.size();
        drawsIndirect[i].commandIndirect.instanceCount = 1;
        drawsIndirect[i].commandIndirect.vertexOffset = 0;
        drawsIndirect[i].commandIndirect.firstInstance = 0;
        drawsIndirect[i].commandIndirect.firstIndex = 0;
        drawsIndirect[i].commandMeshIndirect.firstTask = 0;
        drawsIndirect[i].commandMeshIndirect.taskCount = taskWorkgroups;
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkDeviceSize indirectBufferSize = DRAW_COUNT * sizeof(DrawIndirect);
    createBuffer(stagingBuffer, stagingBufferMemory, indirectBufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, indirectBufferSize, 0, &data);
    memcpy(data, drawsIndirect.data(), sizeof(drawsIndirect[0])*drawsIndirect.size());
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(indirectBuffer, indirectBufferMemory, indirectBufferSize, 
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copyBuffer(stagingBuffer, indirectBuffer, indirectBufferSize);

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
void Engine::createTextureImage() {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    uint32_t mipLevels = std::floor(std::log2(std::max(texWidth, texHeight)))+1;
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
    
    createImage(textureImage, textureImageMemory, VK_SAMPLE_COUNT_1_BIT, texWidth, texHeight, mipLevels, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, 
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, mipLevels, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, textureImage, texWidth, texHeight);
    generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    createImageView(VK_FORMAT_R8G8B8A8_SRGB, textureImage, mipLevels, VK_IMAGE_ASPECT_COLOR_BIT, textureImageView);
}
void Engine::createImage(VkImage& image, VkDeviceMemory& imageMemory, VkSampleCountFlagBits samples, int width, int height, uint32_t mipLevels,
VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memoryProperties) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1; // 3rd dimension in the extent matrix
    imageInfo.mipLevels = mipLevels;
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
    imageInfo.samples = samples;
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
    // LOD determines what mipmap to load
    // it is clamped between minLod and maxLod
    // the smaller LOD, the closer the object is, and lower mipmap is loaded for better quality
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler));
}
void Engine::createDepthBuffer() {
    VkFormat depthFormat = findDepthFormat();
    createImage(depthBuffer, depthBufferMemory, msaaSamples, swapchainExtent.width, swapchainExtent.height, 1, depthFormat, 
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    createImageView(depthFormat, depthBuffer, 1, VK_IMAGE_ASPECT_DEPTH_BIT, depthBufferImageView);
    transitionImageLayout(depthBuffer, depthFormat, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}
void Engine::createColorBuffer() {
    // Vulkan specs enforce that images with more than one sample per pixel must have 1 mip level
    createImage(colorBuffer, colorBufferMemory, msaaSamples, swapchainExtent.width, swapchainExtent.height, 1, swapchainFormat, 
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    createImageView(swapchainFormat, colorBuffer, 1, VK_IMAGE_ASPECT_COLOR_BIT, colorBufferImageView);
    transitionImageLayout(colorBuffer, swapchainFormat, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL); 
}
void Engine::createQueryPools() {
    queryPools.resize(MAX_FRAMES_IN_FLIGHT);
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) {
        // need to create query pools for every frame in flight since queries might be 
        // in use while CPU is already processing the next frame
        VkQueryPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        poolInfo.queryCount = 2; // for each query pool we need 2 queries - for start and finish
        VK_CHECK(vkCreateQueryPool(device, &poolInfo, nullptr, &queryPools[i]));
    }   
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
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
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
    // this function copies data to the first mip level of the image,
    // the other levels are still undefined
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
        // it is assumed the image layout has been transitioned to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL prior to calling this function
        vkCmdCopyBufferToImage(cmdBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    }
    stopRecording(cmdBuffer, transferCmdPool);
}
void Engine::generateMipmaps(VkImage image, VkFormat format, uint32_t width, uint32_t height, uint32_t mipLevels) {
    // check if the device supports linear blitting
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(pDevice, format, &props);
    if (!(props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) 
        throw std::runtime_error("Device doesn't support linear blitting");
    // it is assumed the data has already been transferred to the image (mip level 0) prior to calling this function
    // it means the entire image (all mipmaps) layout is VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, but only
    // mip level 0 has data
    VkCommandBuffer cmdBuffer = beginRecording(graphicsCmdPool);
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;
        
        int32_t mipWidth = width;
        int32_t mipHeight = height;
        
        for (uint32_t i = 1; i<mipLevels; i++) {
            // transition source image mipmap from VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL 
            barrier.subresourceRange.baseMipLevel = i-1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; // wait until content is transferred from previous blit or vkCmdCopyBufferToImage
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT; // blit is transfer operation, so we must wait befor reading from this source
            vkCmdPipelineBarrier(
                cmdBuffer, 
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );

            VkImageBlit blit{};
            // srcOffsets and dstOffsets determine the 3D region that data will be blitted from/to
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.srcSubresource.mipLevel = i-1;
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {mipWidth>1 ? mipWidth/2 : 1, mipHeight>1 ? mipHeight/2 : 1, 1};
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;
            blit.dstSubresource.mipLevel = i;
            vkCmdBlitImage(cmdBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit, VK_FILTER_LINEAR);

            // transition source image mipmap from VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL to VK_IMAGE_LAYOUT_SHADER_READ_OPTIMAL
            // for future operations 
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(
                cmdBuffer, 
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );

            if (mipWidth > 1) mipWidth/=2;
            if (mipHeight > 1) mipHeight/=2;
        }

        // at the end of the loop, the last mipmap has layout VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            cmdBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    }
    stopRecording(cmdBuffer, graphicsCmdPool);
}
VkSampleCountFlagBits Engine::getMaxSampleCount() {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pDevice, &props);
    VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;
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
void Engine::parseSPIRV(Shader& shader) {
    uint32_t* spirv = shader.code.data();
    uint32_t codeSize = shader.codeSize/4; // code size in words
    
    assert(spirv[0] == SpvMagicNumber); // verify the file is SPIR-V
    
    std::unordered_map<uint32_t, std::string> names; // map ID to names
    std::unordered_map<uint32_t, uint32_t> descriptorSetDecorations; // map Variable ID to set numbers
    std::unordered_map<uint32_t, uint32_t> descriptorBindingDecorations; // map Variable ID to descriptor binding numbers
    std::unordered_map<uint32_t, uint32_t> typeInfo; // map Variable ID to Type ID
    std::unordered_map<uint32_t, VkDescriptorType> descriptorTypes;  // map Type ID to descriptor types
    
    // actual instructions start at word 5
    const uint32_t* instr = spirv+5;
    while (instr < spirv + codeSize) {
        uint16_t opCode = uint16_t(*instr & 0xffff);
        uint16_t wordCount = uint16_t(*instr >> 16); // number of words in the instruction
        switch (opCode) {
            case SpvOpEntryPoint: {
                // OpEntryPoint: specifies the shader type
                // word 0: instruction
                // word 1: execution model
                // word 2: entry point ID (result ID of an OpFunction instruction)
                // word 3: name
                // word 4+: inputs and outputs
                switch (instr[1]) {
                    case SpvExecutionModelVertex: {
                        shader.stage = VK_SHADER_STAGE_VERTEX_BIT;
                        break;
                    }; 
                    case SpvExecutionModelFragment: {
                        shader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
                        break;
                    }; 
                    case SpvExecutionModelMeshNV: {
                        shader.stage = VK_SHADER_STAGE_MESH_BIT_NV;
                        break;
                    };
                    case SpvExecutionModelTaskNV: {
                        shader.stage = VK_SHADER_STAGE_TASK_BIT_NV;
                        break;
                    };
                    default: {
                        throw std::runtime_error("Cannot map the SPIR-V execution model bytecode to shader stage");
                    };
                }
                break;
            }
            
            case SpvOpName: {
                // OpName: assigns a name string to another instruction's result ID
                // word 0: instruction
                // word 1: target ID
                // word 2: name
                uint32_t targetId = instr[1];
                names[targetId] = reinterpret_cast<const char*>(&instr[2]);
                break;
            }
            
            case SpvOpDecorate: {
                // OpDecorate: adds a decoration (additional information) to target ID
                // word 0: instruction
                // word 1: target ID
                // word 2: decoration
                // word 3: extra information
                if (wordCount >= 4) { // if decoration is a descriptor, we need at least 4 words to specify set or binding
                    uint32_t targetId = instr[1];
                    uint32_t decoration = instr[2];
                    
                    if (decoration == SpvDecorationDescriptorSet) {
                        // we are decorating it with a descriptor set, i.e put this descriptor into set X
                        descriptorSetDecorations[targetId] = instr[3];
                    } else if (decoration == SpvDecorationBinding) {
                        // we are decorating it with a descriptor binding number, i.e put this descriptor at binding X 
                        descriptorBindingDecorations[targetId] = instr[3];
                    }
                }
                break;
            }
            
            case SpvOpTypePointer: {
                // OpTypePointer: declares a new pointer type
                // word 0: instruction
                // word 1: result ID
                // word 2: storage class
                // word 3: target ID of the data struct
                // i.e declare a new pointer pointing to Y data struct located in X storage class
                if (wordCount >= 4) {
                    uint32_t resultId = instr[1];
                    uint32_t storageClass = instr[2];
                    
                    if (storageClass == SpvStorageClassUniform) {
                        descriptorTypes[resultId] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    } else if (storageClass == SpvStorageClassStorageBuffer) {
                        descriptorTypes[resultId] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    } else if (storageClass == SpvStorageClassUniformConstant) {
                        descriptorTypes[resultId] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    } else if (storageClass == SpvStorageClassPushConstant) {
                        shader.hasPushConstants = true;
                    }
                }
                break;
            }
            
            case SpvOpVariable: {
                // OpVariable: allocates an object in memory resulting in a pointer to it
                // word 0: instruction
                // word 1: result type ID (type of this variable), which must be OpTypePointer
                // word 2: result ID, which might have already been decorated
                // word 3: storage class, i.e where memory lives (must be the same as storage class operand of the result type id)
                // word 4: optional, specifies the initial value of the variable's memory content
                if (wordCount >= 4) {
                    uint32_t typeId = instr[1];
                    uint32_t resultId = instr[2];
                    uint32_t storageClass = instr[3];
                    
                    if (storageClass == SpvStorageClassUniform || 
                        storageClass == SpvStorageClassStorageBuffer ||
                        storageClass == SpvStorageClassUniformConstant) {
                        
                        typeInfo[resultId] = typeId;
                        
                        // check if we have already decorated this variable
                        if (descriptorSetDecorations.count(resultId) && descriptorBindingDecorations.count(resultId)) {
                            DescriptorResourceInfo descriptorResourceInfo{};
                            // as per SPIR-V specs, by the time we hit OpVariable, all decorations must have been processed
                            descriptorResourceInfo.set = descriptorSetDecorations[resultId];
                            descriptorResourceInfo.binding = descriptorBindingDecorations[resultId];
                            // as per SPIR-V specs, OpVariable and OpTypePointer are in the same section
                            descriptorResourceInfo.type = descriptorTypes[typeId];
                            descriptorResourceInfo.descriptorCount = 1;
                            descriptorResourceInfo.name = names[resultId];
                            
                            // as per SPIR-V specs, pEntryPoint instruction must have been reached before OpVariable 
                            descriptorResourceInfo.stage = shader.stage;

                            // in case there is another shader that has already defined a descriptor of the same binding 
                            // in the same set, just add additional shader stage to the one already stored
                            auto res = descriptorResourceInfos.insert(descriptorResourceInfo);
                            if (!res.second) {
                                DescriptorResourceInfo existingResource = *res.first;
                                descriptorResourceInfo.stage |= existingResource.stage;
                                descriptorResourceInfos.erase(res.first);
                                descriptorResourceInfos.insert(descriptorResourceInfo);
                            }
                        }
                    }
                }
                break;
            }
        }
        
        instr += wordCount;
    }
}
Globals Engine::createGlobals() {
    Globals globals{};
    globals.view = glm::lookAt(cameraPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    globals.proj = glm::perspective(glm::radians(45.0f), swapchainExtent.width / (float) swapchainExtent.height, 10000.0f, 0.1f);
    globals.proj[1][1] *= -1;
    globals.meshletCount = meshlets.size();
    glm::mat4 vp = globals.proj * globals.view;
    // Gribb-Hartmann method to extract frustum planes from VP matrix 
    globals.frustum[0] = glm::vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]);
    globals.frustum[1] = glm::vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]);
    globals.frustum[2] = glm::vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]);
    globals.frustum[3] = glm::vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]);
    globals.frustum[4] = glm::vec4(vp[0][3] + vp[0][2], vp[1][3] + vp[1][2], vp[2][3] + vp[2][2], vp[3][3] + vp[3][2]);
    globals.frustum[5] = glm::vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]);

    // normalize normals
    for (int i=0; i<6; i++) {
        float len = glm::length(glm::vec3(globals.frustum[i]));
        globals.frustum[i]/=len;
    }

    return globals;
}
void Engine::updateTransforms(int index) {
    // this stores transformations for each instance
    std::vector<Transform> transforms(DRAW_COUNT);
    srand(42);
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    for (int i=0; i<DRAW_COUNT; i++) {
        Transform transform;
        float offsetX = (float(rand())/RAND_MAX)*20.0f-10.0f;
        float offsetY = (float(rand())/RAND_MAX)*20.0f-10.0f;
        float offsetZ = (float(rand())/RAND_MAX)*20.0f-10.0f;
        float scale = 1.0f;
        float rotateX = (float(rand())/RAND_MAX) * time * glm::radians(90.0f);
        float rotateY = (float(rand())/RAND_MAX) * time * glm::radians(45.0f);
        float rotateZ = (float(rand())/RAND_MAX) * time * glm::radians(30.0f);
        transform.model = glm::translate(glm::mat4(1.0f), glm::vec3(offsetX, offsetY, offsetZ));
        transform.model *= glm::rotate(glm::mat4(1.0f), rotateX, glm::vec3(1.0f, 0.0f, 0.0f));
        transform.model *= glm::rotate(glm::mat4(1.0f), rotateY, glm::vec3(0.0f, 1.0f, 0.0f));
        transform.model *= glm::rotate(glm::mat4(1.0f), rotateZ, glm::vec3(0.0f, 0.0f, 1.0f));
        // this is applied only to put models in a correct vertical position
        transform.model *= glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        transform.model *= glm::scale(glm::mat4(1.0f), glm::vec3(scale));
        transforms[i] = transform;
    }
    memcpy(transformBuffersMapped[index], transforms.data(), sizeof(Transform)*DRAW_COUNT);
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
        vkCmdResetQueryPool(cmdBuffer, queryPools[currentFrame], 0, 2);
        vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[currentFrame], 0);

        VkRenderPassBeginInfo renderpassBeginInfo{};
        renderpassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderpassBeginInfo.renderPass = renderpass;
        renderpassBeginInfo.framebuffer = framebuffers[imageIndex];
        renderpassBeginInfo.renderArea.extent = swapchainExtent;
        renderpassBeginInfo.renderArea.offset = {0, 0};
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        // in Vulkan 1.0 indicates the far view plane, and 0.0 indicates the near view plane
        // but since we are using reverse-z, those values are 0.0f and 1.0f respectively
        clearValues[1].depthStencil = {0.0f, 0};
        renderpassBeginInfo.clearValueCount = clearValues.size();
        renderpassBeginInfo.pClearValues = clearValues.data();
        // VK_SUBPASS_CONTENTS_INLINE means the render pass commands will be embedded in the primary buffer command itself
        // VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS means the render pass commands will be executed from secondary command buffer
        vkCmdBeginRenderPass(cmdBuffer, &renderpassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        {
            if (meshShadersEnabled) {
                vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshGraphicsPipeline);
            } else {
                vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
            }

            // vkCmdBindDescriptorSets binds descriptor sets [0...descriptorSetCount-1] to set
            // numbers firstSet...firstSet+descriptorSetCount-1
            // firstSet is the set number of the first descriptor set to be bound in the provided descriptorSets array
            // we bind a different descriptor set every frame to avoid changing transformations buffer when
            // the previous frame is not done reading from it
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

            if (!meshShadersEnabled) {
                vkCmdBindIndexBuffer(cmdBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            }

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

            // PFN_vkCmdDrawMeshTasksNV vkCmdDrawMeshTasksNV = 
            //     (PFN_vkCmdDrawMeshTasksNV)vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksNV");
            PFN_vkCmdDrawMeshTasksIndirectNV vkCmdDrawMeshTasksIndirectNV = 
                (PFN_vkCmdDrawMeshTasksIndirectNV)vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksIndirectNV");

            VkShaderStageFlags pushConstantStages = 0;
            for (const auto& shader: shaders) {
                if (shader.hasPushConstants) {
                    pushConstantStages |= shader.stage;
                }
            }
            
            Globals globals = createGlobals();
            vkCmdPushConstants(cmdBuffer, pipelineLayout, pushConstantStages, 0, sizeof(Globals), &globals);
            
            if (meshShadersEnabled) {
                vkCmdDrawMeshTasksIndirectNV(cmdBuffer, indirectBuffer, offsetof(DrawIndirect, commandMeshIndirect), DRAW_COUNT, sizeof(DrawIndirect));
                // vkCmdDrawMeshTasksNV(cmdBuffer, taskWorkgroups, 0);
                // vkCmdDrawMeshTasksNV(cmdBuffer, uint32_t(meshlets.size()), 0);
            } else {
                vkCmdDrawIndexedIndirect(cmdBuffer, indirectBuffer, offsetof(DrawIndirect, commandIndirect), DRAW_COUNT, sizeof(DrawIndirect));
                // vkCmdDrawIndexed(cmdBuffer, indices.size(), 1, 0, 0, 0);
            }
        }
        vkCmdEndRenderPass(cmdBuffer);
        vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[currentFrame], 1);
    }
    vkEndCommandBuffer(cmdBuffer);
}
void Engine::transitionImageLayout(VkImage image, VkFormat format, uint32_t mipLevels, VkImageLayout oldLayout, VkImageLayout newLayout) {
    // this function transitions the entire image layout, i.e all mipmaps at once

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
    barrier.subresourceRange.levelCount = mipLevels;
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
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; // do not execute this stage until image layout transition is done
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        // do not write or read from depth buffer until the image layout transition is done
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
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