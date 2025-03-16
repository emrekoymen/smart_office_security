#ifndef SINGLE_CAMERA_DETECTOR_H
#define SINGLE_CAMERA_DETECTOR_H

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "camera_processor.h"
#include "model.h"

/**
 * @brief Class to handle single camera or video file processing with person detection
 */
class SingleCameraDetector {
public:
    /**
     * @brief Constructor
     * @param source Source for video (camera index or file path)
     * @param modelPath Path to the TFLite model
     * @param labelsPath Path to the labels file
     * @param threshold Detection threshold
     * @param personClassId ID for person class in the model
     */
    SingleCameraDetector(const std::string& source, 
                         const std::string& modelPath,
                         const std::string& labelsPath,
                         float threshold = 0.5,
                         int personClassId = 0);

    /**
     * @brief Destructor
     */
    ~SingleCameraDetector();

    /**
     * @brief Run the detector
     * @param displayOutput Whether to display the output
     * @param saveOutput Whether to save the output
     * @param outputDir Directory to save output
     * @param forceCPU Force CPU mode (no TPU)
     */
    void run(bool displayOutput = true, 
             bool saveOutput = false,
             const std::string& outputDir = "output",
             bool forceCPU = false);

    /**
     * @brief Get the FPS from the camera processor
     * @return FPS value
     */
    float getFPS() const;

private:
    // Camera processor
    std::unique_ptr<CameraProcessor> cameraProcessor;
    
    // Detection model
    std::unique_ptr<Model> model;
    
    // Detection settings
    float threshold;
    int personClassId;
    
    // Source information
    std::string source;
    bool isCamera;
    bool isRunning;
    
    // Target resolution for processing
    int targetWidth;
    int targetHeight;
    
    /**
     * @brief Process a frame with detection
     * @param frame Frame to process
     * @return Processed frame with detections
     */
    cv::Mat processFrame(const cv::Mat& frame);
    
    /**
     * @brief Determine if a source is a camera or a file
     * @param source Source string
     * @return True if camera, false if file
     */
    bool isSourceCamera(const std::string& source) const;
};

#endif // SINGLE_CAMERA_DETECTOR_H 