cmake_minimum_required(VERSION 3.16)

project(ImageCNN
        VERSION 1.0.0
        DESCRIPTION "Convolutional neural network image classifier"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/utf-8)
endif()

add_library(imagecnn_core STATIC
    src/version.cpp
)
target_include_directories(imagecnn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(imagecnn_core PUBLIC cxx_std_17)
