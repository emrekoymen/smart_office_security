#include "dual_camera_detector.h"
#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments using the detector's parser
        CommandLineArgs args = DualCameraDetector::parseArgs(argc, argv);

        // Create the detector instance
        DualCameraDetector detector(args);

        // Initialize the detector (loads model, opens cameras)
        if (!detector.initialize()) {
            std::cerr << "Failed to initialize detector" << std::endl;
            return 1;
        }

        // Run the detector's main loop
        if (!detector.run()) {
             std::cerr << "Detector encountered an error during run" << std::endl;
             return 1;
        }

        std::cout << "Dual camera detector finished successfully." << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Unhandled Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown unhandled exception occurred." << std::endl;
        return 1;
    }
} 