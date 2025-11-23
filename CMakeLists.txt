cmake_minimum_required(VERSION 3.15)
project(vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED CONFIG)

add_executable(vulkan 
    main.cpp
    Engine.cpp
)

target_include_directories(vulkan PRIVATE
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vulkan PRIVATE
    Vulkan::Vulkan
    glfw
)

target_compile_options(vulkan PRIVATE -Wall -Wextra -Wpedantic)