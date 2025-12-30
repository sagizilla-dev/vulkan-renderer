#version 460

layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 TexCoords;

layout(set = 0, binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outFragColor;

void main() {
    // outFragColor = texture(texSampler, TexCoords);
    outFragColor = vec4((Normal+1.0)/2, 1.0);
}
