# Copyright (c) 2017 by ETH Zurich, Thomas Mueller.
cmake_minimum_required(VERSION 2.8.3)

project(visualizer)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(NANOGUI_BUILD_EXAMPLE OFF CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_SHARED ON CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_PYTHON OFF CACHE BOOL " " FORCE)
set(NANOGUI_INSTALL OFF CACHE BOOL " " FORCE)
add_subdirectory(nanogui)

add_executable(visualizer
    src/main.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/nanogui/ext/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/nanogui/ext/glfw/include
    ${CMAKE_CURRENT_SOURCE_DIR}/nanogui/ext/nanovg/src
    ${CMAKE_CURRENT_SOURCE_DIR}/nanogui/include
    ${NANOGUI_EXTRA_INCS}
)

add_definitions(${NANOGUI_EXTRA_DEFS})
target_link_libraries(visualizer nanogui)
