#ifndef DUAL_CAMERA_DETECTOR_H
#define DUAL_CAMERA_DETECTOR_H

#include <string>
#include <memory>
#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "camera_processor.h"
#include "model.h"
#include "utils.h"

/**
 * @brief Command line arguments structure
 */
struct CommandLineArgs {
    std::string modelPath = "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
    std::string labelsPath = "models/coco_labels.txt";
    float threshold = 0.5;
    std::string camera1 = "0";
    std::string camera2 = "1";
    std::string outputDir = "output";
    int personClassId = 0;
    bool noDisplay = false;
    bool saveVideo = false;
    bool forceCPU = false;
};

/**
 * @brief Dual camera detector class
 */
class DualCameraDetector {
public:
    /**
     * @brief Constructor
     * @param args Command line arguments
     */
    DualCameraDetector(const CommandLineArgs& args);

    /**
     * @brief Destructor
     */
    ~DualCameraDetector();

    /**
     * @brief Initialize the detector
     * @return True if successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Run the detector
     * @return True if successful, false otherwise
     */
    bool run();

    /**
     * @brief Parse command line arguments
     * @param argc Argument count
     * @param argv Argument values
     * @return Command line arguments structure
     */
    static CommandLineArgs parseArgs(int argc, char** argv);

private:
    /**
     * @brief Check if display is available
     * @return True if available, false otherwise
     */
    bool checkDisplayAvailability();

    /**
     * @brief Process the camera feeds
     */
    void processCameraFeeds();

    // Command line arguments
    CommandLineArgs args;

    // Model
    std::shared_ptr<Model> model;

    // Camera processors
    std::unique_ptr<CameraProcessor> camera1;
    std::unique_ptr<CameraProcessor> camera2;

    // Utility classes
    std::unique_ptr<Logger> logger;
    std::unique_ptr<AlertSystem> alertSystem;

    // Video writer
    cv::VideoWriter videoWriter;
    bool videoWriterInitialized;

    // Display availability
    bool displayAvailable;

    // Running flag
    std::atomic<bool> running;

    // Performance metrics
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

#endif // DUAL_CAMERA_DETECTOR_H 