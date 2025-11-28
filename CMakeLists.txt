cmake_minimum_required(VERSION 3.15)
project(vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED CONFIG)

file(GLOB_RECURSE SHADER_SOURCES
    "*.vert"
    "*.frag"
    "*.mesh"
)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV_OUTPUT "${CMAKE_SOURCE_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV_OUTPUT}
        COMMAND ${GLSLC} ${SHADER} -o ${SPIRV_OUTPUT} --target-env=vulkan1.4
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )
    list(APPEND SPIRV_BINARIES ${SPIRV_OUTPUT})
endforeach()
add_custom_target(shaders DEPENDS ${SPIRV_BINARIES})

add_executable(vulkan 
    main.cpp
    Engine.cpp
)
add_dependencies(vulkan shaders)

target_include_directories(vulkan PRIVATE
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vulkan PRIVATE
    Vulkan::Vulkan
    glfw
)

target_compile_options(vulkan PRIVATE -Wall -Wextra -Wpedantic)