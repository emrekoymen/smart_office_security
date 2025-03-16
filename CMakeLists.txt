cmake_minimum_required(VERSION 3.10)
project(smart_office_security)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Check if TensorFlow has been built
set(TENSORFLOW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/tensorflow")
set(TENSORFLOW_LITE_ROOT "${TENSORFLOW_ROOT}/tensorflow/lite")
set(TENSORFLOW_LITE_LIB "${TENSORFLOW_LITE_ROOT}/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a")

# Check if TensorFlow is available
if(EXISTS ${TENSORFLOW_LITE_LIB})
    message(STATUS "TensorFlow Lite library found: ${TENSORFLOW_LITE_LIB}")
    # Remove the DISABLE_TENSORFLOW definition
    set(USE_TENSORFLOW TRUE)
else()
    message(WARNING "TensorFlow Lite library not found. Using mock implementation.")
    # Continue using the mock implementation
    add_definitions(-DDISABLE_TENSORFLOW)
    set(USE_TENSORFLOW FALSE)
endif()

# TensorFlow include directories
if(USE_TENSORFLOW)
    set(TENSORFLOW_INCLUDE_DIRS
        ${TENSORFLOW_ROOT}
        ${TENSORFLOW_ROOT}/tensorflow/lite/tools/make/downloads/flatbuffers/include
        ${TENSORFLOW_ROOT}/tensorflow/lite/tools/make/downloads/absl
    )
endif()

# Edge TPU library path
set(EDGETPU_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libedgetpu.so")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Add TensorFlow include directories if available
if(USE_TENSORFLOW)
    include_directories(${TENSORFLOW_INCLUDE_DIRS})
endif()

# Source files
file(GLOB SOURCES "src/*.cpp")

# Create executable
add_executable(dual_camera_detector ${SOURCES})

# Link libraries
if(USE_TENSORFLOW)
    # Check if Edge TPU library exists
    if(EXISTS ${EDGETPU_LIBRARY_PATH})
        message(STATUS "Edge TPU library found: ${EDGETPU_LIBRARY_PATH}")
        target_link_libraries(dual_camera_detector
            ${OpenCV_LIBS}
            ${CMAKE_THREAD_LIBS_INIT}
            ${TENSORFLOW_LITE_LIB}
            ${EDGETPU_LIBRARY_PATH}
        )
    else()
        message(WARNING "Edge TPU library not found. Only CPU inference will be available.")
        target_link_libraries(dual_camera_detector
            ${OpenCV_LIBS}
            ${CMAKE_THREAD_LIBS_INIT}
            ${TENSORFLOW_LITE_LIB}
        )
    endif()
else()
    target_link_libraries(dual_camera_detector
        ${OpenCV_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
    )
endif()

# Installation
install(TARGETS dual_camera_detector DESTINATION bin) 