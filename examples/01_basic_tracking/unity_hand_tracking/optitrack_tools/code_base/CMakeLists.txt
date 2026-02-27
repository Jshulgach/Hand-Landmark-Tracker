cmake_minimum_required(VERSION 3.15)
project(optitrack_cam_pybind)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Camera SDK paths based on your current setup
set(CAMERA_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Camera SDK")
set(CAMERA_SDK_INCLUDE "${CAMERA_SDK_DIR}/include")
set(CAMERA_SDK_LIB_DIR "${CAMERA_SDK_DIR}/lib")

# Find the Camera SDK library (handling both static and dynamic versions)
find_library(CAMERALIBRARY_LIB
    NAMES CameraLibrary2015x64S CameraLibrary2015x64D CameraLibrary2013x64S CameraLibrary2008S
    PATHS "${CAMERA_SDK_LIB_DIR}"
    NO_DEFAULT_PATH
)

if(NOT CAMERALIBRARY_LIB)
    message(FATAL_ERROR "Could not find CameraLibrary in ${CAMERA_SDK_LIB_DIR}")
endif()

# Create the pybind11 module
pybind11_add_module(optitrack_cam camera_sdk_pybind.cpp)

# Include directories and definitions
target_include_directories(optitrack_cam PRIVATE "${CAMERA_SDK_INCLUDE}")
target_compile_definitions(optitrack_cam PRIVATE WIN32 CAMERALIBRARY_IMPORTS CORE_IMPORTS)

# Link CameraLibrary and standard Windows dependencies
target_link_libraries(optitrack_cam PRIVATE "${CAMERALIBRARY_LIB}" ws2_32.lib setupapi.lib)

# Set output directory to parent folder
set_target_properties(optitrack_cam PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# Deployment: Copy DLLs to output directory for runtime
if(WIN32)
    file(GLOB CAMERA_SDK_DLLS "${CAMERA_SDK_LIB_DIR}/*.dll")
    foreach(dll_file ${CAMERA_SDK_DLLS})
        add_custom_command(TARGET optitrack_cam POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${dll_file}"
            "$<TARGET_FILE_DIR:optitrack_cam>/"
        )
    endforeach()
endif()