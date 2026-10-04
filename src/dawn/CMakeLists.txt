# Builds libnano_dawn.so: Dawn (WebGPU) at the exact revision Chrome 154's
# model library expects, exposing its proc table through one C function.
cmake_minimum_required(VERSION 3.22)
project(nano_dawn CXX C)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(DAWN_FETCH_DEPENDENCIES OFF)
set(DAWN_BUILD_SAMPLES OFF)
set(DAWN_BUILD_TESTS OFF)
set(DAWN_BUILD_BENCHMARKS OFF)
set(DAWN_ENABLE_INSTALL OFF)
set(DAWN_ENABLE_VULKAN ON)
set(DAWN_ENABLE_OPENGLES OFF)
set(DAWN_ENABLE_DESKTOP_GL OFF)
set(DAWN_ENABLE_NULL OFF)
set(DAWN_USE_X11 OFF)
set(DAWN_USE_WAYLAND OFF)
set(DAWN_USE_GLFW OFF)
set(DAWN_BUILD_MONOLITHIC_LIBRARY OFF)
set(TINT_BUILD_TESTS OFF)
set(TINT_BUILD_BENCHMARKS OFF)
set(TINT_BUILD_CMD_TOOLS OFF)
set(TINT_BUILD_IR_BINARY OFF)
# Dawn's source checkout, passed in by the Makefile.
add_subdirectory(${DAWN_SRC} dawn EXCLUDE_FROM_ALL)

add_library(nano_dawn SHARED shim.cc)
target_link_libraries(nano_dawn PRIVATE dawn::dawn_native)
set_target_properties(nano_dawn PROPERTIES CXX_VISIBILITY_PRESET hidden)
