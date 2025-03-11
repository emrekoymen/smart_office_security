#include "dual_camera_detector.h"
#include <iostream>

int main(int argc, char** argv) {
    try {
        // Parse command line arguments
        CommandLineArgs args = DualCameraDetector::parseArgs(argc, argv);
        
        // Create detector
        DualCameraDetector detector(args);
        
        // Initialize detector
        if (!detector.initialize()) {
            std::cerr << "Failed to initialize detector" << std::endl;
            return 1;
        }
        
        // Run detector
        if (!detector.run()) {
            std::cerr << "Error during detection" << std::endl;
            return 1;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 