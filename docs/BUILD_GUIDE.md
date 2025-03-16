# Smart Office System C++ Implementation Build Guide

This document provides a comprehensive guide for building and troubleshooting the C++ implementation of the Smart Office System.

## Build Environment

The Smart Office System C++ implementation requires the following:

- Ubuntu Linux or compatible distribution
- CMake 3.10 or higher
- OpenCV 4.x
- TensorFlow Lite 2.4.0 (optional, mock implementation available)
- Edge TPU runtime (optional, for hardware acceleration)

## Build Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/emrekoymen/smart_office_security.git
   cd smart_office_security
   git checkout cpp_implementation
   ```

2. **Setup the build environment**:
   ```bash
   # Create build directory
   mkdir -p build
   
   # Navigate to build directory
   cd build
   
   # Configure with CMake
   cmake ..
   
   # Build
   make -j4
   ```

   Note: If you want to build with TensorFlow Lite support, run the setup script:
   ```bash
   ./build_and_run.sh --setup-tensorflow --build-only
   ```

3. **Run the application**:
   ```bash
   # Single camera mode
   ./dual_camera_detector --single-camera --source=0
   
   # Dual camera mode
   ./dual_camera_detector
   ```

## Implemented Fixes

During the development process, we implemented several fixes to address common issues:

1. **DualCameraDetector Constructor Fix**: Updated the constructor call to use a `CommandLineArgs` struct instead of a string map.

2. **CameraProcessor Enhancements**:
   - Added constructors for different source types
   - Implemented robust frame acquisition methods
   - Added error handling in the video processing pipeline

3. **Path Resolution Fixes**:
   - Updated model and labels paths to use correct relative paths
   - Ensured cross-platform path handling

4. **Error Handling Improvements**:
   - Added null pointer checks
   - Implemented exception handling throughout the code
   - Added retry mechanisms for frame acquisition

## Known Issues

1. **Segmentation Fault**: The application currently encounters a segmentation fault during the frame processing loop, specifically when using video files as input. This issue persists despite adding defensive coding practices.

2. **TensorFlow Lite Integration**: The application can be built without TensorFlow Lite, using a mock implementation, but this limits functionality.

3. **Camera Device Management**: Some camera devices may not be properly identified or accessed.

## Debugging Recommendations

For the segmentation fault issue, we recommend the following debugging approaches:

1. **Use a Debugger**: Run the application with GDB to pinpoint the exact location of the segmentation fault:
   ```bash
   gdb --args ./dual_camera_detector --single-camera --source=../videos/sample.mp4
   ```
   
   Common GDB commands:
   - `run` - Start execution
   - `bt` - Print backtrace when a crash occurs
   - `info locals` - Print local variables in current frame
   - `frame N` - Select frame N from the backtrace

2. **Memory Analysis**: Use Valgrind to detect memory issues:
   ```bash
   valgrind --leak-check=full ./dual_camera_detector --single-camera --source=../videos/sample.mp4
   ```

3. **Direct Implementation Approach**: Consider implementing a simpler version of the video processing without using separate threads:
   ```cpp
   if (!isCamera) {
       cv::VideoCapture directCap(source);
       if (!directCap.isOpened()) {
           std::cerr << "Failed to open video file: " << source << std::endl;
           return;
       }
       
       cv::Mat frame;
       while (directCap.read(frame)) {
           // Process frame directly
           // ...
       }
   }
   ```

## Future Improvements

For future versions, consider implementing the following improvements:

1. **Separate Video and Camera Handling**: Create different classes for video file and camera device processing.

2. **Simplified Threading Model**: Review the multithreaded design for potential race conditions.

3. **Enhanced Error Reporting**: Add more detailed logging and error reporting mechanisms.

4. **Progress Indicators**: Add progress bars or status updates for long-running operations.

5. **Configuration File Support**: Allow loading settings from a configuration file.

## References

- [CMake Documentation](https://cmake.org/documentation/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Lite C++ API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c)
- [Edge TPU Developer Guide](https://coral.ai/docs/edgetpu/api-intro/)

## Support

For additional support or to report bugs, please use the GitHub issue tracker. 