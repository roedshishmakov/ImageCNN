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

include(FetchContent)

# SOURCE_SUBDIR points at a missing directory so that only the headers are
# fetched, without doctest's own (outdated) CMakeLists.txt.
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
    SOURCE_SUBDIR  none
)
FetchContent_MakeAvailable(doctest)

add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE ${doctest_SOURCE_DIR})
add_library(doctest::doctest ALIAS doctest)

add_library(imagecnn_core STATIC
    src/version.cpp
    src/activations.cpp
    src/rng.cpp
    src/neuron.cpp
    src/link.cpp
    src/layer.cpp
    src/network.cpp
    src/spatial.cpp
    src/model.cpp
)
target_include_directories(imagecnn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(imagecnn_core PUBLIC cxx_std_17)

add_executable(imagecnn src/main.cpp)
target_link_libraries(imagecnn PRIVATE imagecnn_core)

enable_testing()

add_executable(imagecnn_tests tests/test_main.cpp)
target_link_libraries(imagecnn_tests PRIVATE imagecnn_core doctest::doctest)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(imagecnn_tests)
