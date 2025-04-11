#include "single_camera_detector.h"
#include "utils.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <iomanip>
#include <sstream>
#include <ctime>

namespace fs = std::filesystem;

SingleCameraDetector::SingleCameraDetector(const std::string& source,
                                           const std::string& modelPath,
                                           const std::string& labelsPath,
                                           float threshold,
                                           int personClassId)
    : source(source), modelPath(modelPath), labelsPath(labelsPath), threshold(threshold), personClassId(personClassId), 
      isRunning(false), targetWidth(300), targetHeight(300) {
    
    // Determine if source is a camera or a file
    isCamera = isSourceCamera(source);
    
    std::cout << "Initializing single camera detector for " 
              << (isCamera ? "camera " + source : "video file " + source) << std::endl;
    
    // Create output directory if it doesn't exist
    fs::create_directories("output");
    
    // Initialize the model (as shared_ptr)
    try {
        model = std::make_shared<Model>(modelPath, labelsPath, false);
        targetWidth = model->getInputWidth();
        targetHeight = model->getInputHeight();
        // Interpreter creation removed
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize model: " << e.what() << std::endl;
        model = nullptr;
    }
}

SingleCameraDetector::~SingleCameraDetector() {
    if (cameraProcessor) {
         cameraProcessor->stop(); // Ensure processor is stopped
    }
    std::cout << "Single camera detector destructor called" << std::endl;
}

void SingleCameraDetector::run(bool displayOutput, 
                               bool saveOutput,
                               const std::string& outputDir,
                               bool forceCPU) {
    
    // Create output directory if it doesn't exist
    if (saveOutput) {
        fs::create_directories(outputDir);
    }
    
    // Reinitialize the model if necessary
    bool modelReinitialized = false;
    if (!model || model->isUsingTPU() == forceCPU) { 
        try {
            std::cout << "Reinitializing model (forceCPU=" << forceCPU << ")..." << std::endl;
            model = std::make_shared<Model>(modelPath, labelsPath, forceCPU);
            modelReinitialized = true;
             // Interpreter logic removed from here too
        } catch (const std::exception& e) {
            std::cerr << "Failed to reinitialize model: " << e.what() << std::endl;
            return;
        }
    }

    // Ensure model is valid before proceeding
    if (!model) {
         std::cerr << "Model is not initialized. Cannot run detector." << std::endl;
         return;
    }
    
    // Initialize or reinitialize the camera processor if necessary
    if (!cameraProcessor || modelReinitialized) {
        try {
            std::cout << "Initializing CameraProcessor..." << std::endl;
            int cameraIndex = -1; 
            if (isCamera) {
                try {
                    cameraIndex = std::stoi(source);
                } catch (...) { /* Handle exceptions */ }
            }
            
            // Stop existing processor if reinitializing
            if (cameraProcessor) {
                cameraProcessor->stop();
            }

            // Create CameraProcessor using the correct constructor
            cameraProcessor = std::make_unique<CameraProcessor>(cameraIndex, model, personClassId, threshold);
            
            // Open the camera/video source
            if (!cameraProcessor->openCamera(source)) {
                throw std::runtime_error("Failed to open camera/video source: " + source);
            }
            
            // Start the camera processor
            if (!cameraProcessor->start()) {
                throw std::runtime_error("Failed to start camera processor");
            }
             std::cout << "CameraProcessor started." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize camera processor: " << e.what() << std::endl;
            return;
        }
    }
    
    // Video writer setup (remains largely the same, but get frame from processor)
    cv::VideoWriter videoWriter;
    std::string outputFilePath;
    bool videoWriterOpened = false;
    
    if (saveOutput) {
        // ... (output file naming logic) ...
        std::string sourceName = isCamera ? "camera" + source : fs::path(source).stem().string();
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
        outputFilePath = outputDir + "/" + sourceName + "_processed_" + ss.str() + ".mp4";
        
        // Get initial frame dimensions from processor
        cv::Mat firstFrame;
        for (int attempt = 0; attempt < 10 && firstFrame.empty(); ++attempt) {
             std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Wait for processor
             auto result = cameraProcessor->getLatestResult();
             if (result && !result->frame.empty()) {
                  firstFrame = result->frame;
                  break;
             }
        }
        
        if (!firstFrame.empty()) {
            videoWriter.open(outputFilePath, 
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            cameraProcessor->getFPS() > 0 ? cameraProcessor->getFPS() : 30.0, // Use actual FPS if available
                            cv::Size(firstFrame.cols, firstFrame.rows));
            videoWriterOpened = videoWriter.isOpened();
            if (videoWriterOpened) {
                 std::cout << "Saving output to " << outputFilePath << std::endl;
            } else {
                 std::cerr << "Could not open video writer." << std::endl;
                 saveOutput = false;
            }
        } else {
            std::cerr << "Could not get first frame from processor. Output will not be saved." << std::endl;
            saveOutput = false;
        }
    }
    
    // Main processing loop using CameraProcessor results
    isRunning = true;
    int frameCount = 0;
    double totalFpsAccumulator = 0.0;
    
    std::cout << "Starting detection loop..." << std::endl;
    
    while (isRunning && cameraProcessor && cameraProcessor->isRunning()) {
        auto result = cameraProcessor->getLatestResult();

        if (!result) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
            continue;
        }

        cv::Mat processedFrame = result->frame;
        if (processedFrame.empty()) {
             std::cout << "Received empty processed frame." << std::endl;
             continue; // Skip if frame is empty
        }
        
        float currentFps = result->fps;
        totalFpsAccumulator += currentFps;
        frameCount++;
        
        // Add text overlays (FPS, Mode)
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(currentFps));
        std::string modeText = model->isUsingTPU() ? "TPU" : "CPU";
        cv::putText(processedFrame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(processedFrame, modeText, cv::Point(processedFrame.cols - 100, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Display frame
        if (displayOutput) {
            try {
                cv::imshow("Single Camera Detector", processedFrame);
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q') {
                    isRunning = false;
                }
            } catch (const cv::Exception& e) { // Use cv::Exception
                std::cerr << "Error displaying frame: " << e.what() << std::endl;
                displayOutput = false; 
            }
        }
        
        // Write frame to video
        if (saveOutput && videoWriterOpened) {
            try {
                videoWriter.write(processedFrame);
            } catch (const cv::Exception& e) { // Use cv::Exception
                std::cerr << "Error writing frame to video: " << e.what() << std::endl;
                saveOutput = false; 
            }
        }

        // Stop loop if processor is no longer running
        if (!cameraProcessor->isRunning()) {
             isRunning = false;
        }
    }
    
    std::cout << "Detection loop finished." << std::endl;
    
    // Cleanup
    if (cameraProcessor) {
        cameraProcessor->stop();
    }
    if (videoWriterOpened) {
        videoWriter.release();
    }
    cv::destroyAllWindows();
    
    // Print summary
    if (frameCount > 0) {
        double avgFps = totalFpsAccumulator / frameCount;
        std::cout << "Processed " << frameCount << " frames." << std::endl;
        std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << avgFps << std::endl;
    } else {
         std::cout << "No frames processed." << std::endl;
    }
}

float SingleCameraDetector::getFPS() const {
    return cameraProcessor ? cameraProcessor->getFPS() : 0.0f;
}

bool SingleCameraDetector::isSourceCamera(const std::string& source) const {
    // Try to convert source to integer (camera index)
    try {
        std::stoi(source);
        return true; // Successfully converted, assume it's a camera index
    } catch (const std::invalid_argument& ia) {
        // Not an integer, assume it's a file path
        return false;
    } catch (const std::out_of_range& oor) {
        // Integer but out of range, could be a file path with numbers
        return false;
    }
} 