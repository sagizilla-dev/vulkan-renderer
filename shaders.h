#pragma once
#include "utils.hpp"

struct DescriptorInfo {
    union {
        VkDescriptorImageInfo imageInfo;
        VkDescriptorBufferInfo bufferInfo;
    };
};