cmake_minimum_required(VERSION 3.10)
project(smart_office_security)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Add Edge TPU library paths (adjust as needed)
set(EDGETPU_LIBRARY_PATH "/usr/lib/libedgetpu.so.1")

# Source files
file(GLOB SOURCES "src/*.cpp")

# Create executable
add_executable(dual_camera_detector ${SOURCES})

# Link libraries
target_link_libraries(dual_camera_detector
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${EDGETPU_LIBRARY_PATH}
)

# Installation
install(TARGETS dual_camera_detector DESTINATION bin) 