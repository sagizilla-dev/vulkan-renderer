#pragma once
#include "utils.hpp"

struct Shader {
    VkShaderModule module;
    VkShaderStageFlags stage;
    static VkShaderStageFlags getShaderStage(SpvExecutionModel executionModel) {
        switch (executionModel) {
            case SpvExecutionModelVertex: {
                return VK_SHADER_STAGE_VERTEX_BIT;
            }; 
            case SpvExecutionModelFragment: {
                return VK_SHADER_STAGE_FRAGMENT_BIT;
            }; 
            case SpvExecutionModelMeshNV: {
                return VK_SHADER_STAGE_MESH_BIT_NV;
            };
        }
    }
};

struct DescriptorInfo {
    union {
        VkDescriptorImageInfo imageInfo;
        VkDescriptorBufferInfo bufferInfo;
    };
};