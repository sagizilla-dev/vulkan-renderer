#version 460

layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 TexCoords;

layout(set = 0, binding = 0) uniform sampler2D texSampler;
layout(set = 0, binding = 5) uniform sampler2D hiZDepthBufferSampler;

layout(location = 0) out vec4 outFragColor;

void main() {
    // texture() function expects texture coordinates within [0...1] range, so we need to normalize
    // pixel coordinates
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(hiZDepthBufferSampler, 0));
    // outFragColor = texture(texSampler, TexCoords);
    outFragColor = texture(hiZDepthBufferSampler, uv);
    // outFragColor = vec4(Normal, 1.0);
}
