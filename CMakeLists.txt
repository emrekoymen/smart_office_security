# Set minimum required CMake version
cMAKE_MINIMUM_REQUIRED(VERSION 3.10)

# Enable project VERSION keyword
cmake_policy(SET CMP0048 NEW)

# Set project name and version
project(dual_camera_detector VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Threads
find_package(Threads REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
# TensorFlow Lite source directory
/home/emrek/cursor/smart_office_security/smart_office_security/external/tensorflow

# FlatBuffers include path
/home/emrek/cursor/smart_office_security/smart_office_security/external/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
)

# Add source files
add_executable(dual_camera_detector 
    src/main.cpp 
    src/dual_camera_detector.cpp 
    src/camera_processor.cpp 
    src/single_camera_detector.cpp
    src/model.cpp
)

# Add Edge TPU library path
# set(EDGETPU_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libedgetpu.so") # Commented out original
set(EDGETPU_LIBRARY_NAME "edgetpu") # Link by name

# TensorFlow Lite library path (Point to downloaded library)
set(TFLITE_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtensorflowlite.so") # Path to downloaded library
# set(TFLITE_LIBRARY_NAME "tensorflowlite") # Commented out linking by name

# Link libraries
target_link_libraries(dual_camera_detector 
    ${OpenCV_LIBS} 
    ${CMAKE_THREAD_LIBS_INIT}
#    ${EDGETPU_LIBRARY_PATH} # Commented out original
    ${EDGETPU_LIBRARY_NAME}    # Link edgetpu using name
    ${TFLITE_LIBRARY_PATH}     # Link tensorflowlite using path
)

# Optional: Enable specific optimizations if needed
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Define compiler definitions (remove the DISABLE_TENSORFLOW)
# add_definitions(-DDISABLE_TENSORFLOW)

# Installation
install(TARGETS dual_camera_detector DESTINATION bin) 