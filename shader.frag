#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 UV;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outFragColor;

void main() {
    outFragColor = texture(texSampler, UV);
}
