cmake_minimum_required(VERSION 3.15)
project(vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED CONFIG)

file(GLOB_RECURSE SHADER_SOURCES
    "${CMAKE_SOURCE_DIR}/shaders/*.vert"
    "${CMAKE_SOURCE_DIR}/shaders/*.frag"
    "${CMAKE_SOURCE_DIR}/shaders/*.mesh"
)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    get_filename_component(SHADER_DIR  ${SHADER} DIRECTORY)
    set(SPIRV_OUTPUT "${SHADER_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV_OUTPUT}
        COMMAND ${GLSLC} ${SHADER} -o ${SPIRV_OUTPUT} --target-env=vulkan1.4
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )
    list(APPEND SPIRV_BINARIES ${SPIRV_OUTPUT})
endforeach()
add_custom_target(shaders DEPENDS ${SPIRV_BINARIES})
add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
add_executable(vulkan 
    main.cpp
    Engine.cpp
)
add_dependencies(vulkan shaders)

target_include_directories(vulkan PRIVATE external/stb)
target_include_directories(vulkan PRIVATE external/tinyobjloader)

target_include_directories(vulkan PRIVATE
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vulkan PRIVATE
    Vulkan::Vulkan
    glfw
)

target_compile_options(vulkan PRIVATE -Wall -Wextra -Wpedantic)