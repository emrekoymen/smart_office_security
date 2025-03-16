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
    
    // Initialize the model
    try {
        model = std::make_unique<Model>(modelPath, labelsPath, false);
        // Get target dimensions from model
        targetWidth = model->getInputWidth();
        targetHeight = model->getInputHeight();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize model: " << e.what() << std::endl;
    }
}

SingleCameraDetector::~SingleCameraDetector() {
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
    
    // Reinitialize the model with the correct forceCPU setting if needed
    if (model && model->isUsingTPU() != !forceCPU) {
        try {
            model = std::make_unique<Model>(modelPath, labelsPath, forceCPU);
        } catch (const std::exception& e) {
            std::cerr << "Failed to reinitialize model: " << e.what() << std::endl;
        }
    }
    
    // Initialize the camera processor
    try {
        if (isCamera) {
            int cameraIndex = std::stoi(source);
            cameraProcessor = std::make_unique<CameraProcessor>(cameraIndex, 30, 640, 480);
        } else {
            cameraProcessor = std::make_unique<CameraProcessor>(source, 30, 640, 480);
        }
        
        // Start the camera processor
        if (cameraProcessor && !cameraProcessor->start()) {
            std::cerr << "Failed to start camera processor" << std::endl;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize camera processor: " << e.what() << std::endl;
        return;
    }
    
    // Video writer for saving output
    cv::VideoWriter videoWriter;
    std::string outputFilePath;
    
    if (saveOutput) {
        // Get source name for output file
        std::string sourceName;
        if (isCamera) {
            sourceName = "camera" + source;
        } else {
            fs::path p(source);
            sourceName = p.stem().string();
        }
        
        // Create timestamp
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
        
        // Create output file path
        outputFilePath = outputDir + "/" + sourceName + "_processed_" + ss.str() + ".mp4";
        
        // Try to get first frame with multiple attempts
        cv::Mat firstFrame;
        bool frameObtained = false;
        
        for (int attempt = 0; attempt < 5; attempt++) {
            std::cout << "Attempt " << (attempt + 1) << " to get first frame..." << std::endl;
            firstFrame = cameraProcessor->getFrame();
            
            if (!firstFrame.empty()) {
                frameObtained = true;
                break;
            }
            
            std::cerr << "Attempt " << (attempt + 1) << " to get first frame failed. Retrying..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        if (frameObtained) {
            // Initialize video writer
            videoWriter.open(outputFilePath, 
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            30.0, 
                            cv::Size(firstFrame.cols, firstFrame.rows));
            
            if (!videoWriter.isOpened()) {
                std::cerr << "Could not open video writer. Output will not be saved." << std::endl;
                saveOutput = false;
            } else {
                std::cout << "Saving output to " << outputFilePath << std::endl;
            }
        } else {
            std::cerr << "Could not get first frame after multiple attempts. Output will not be saved." << std::endl;
            saveOutput = false;
        }
    }
    
    // Main processing loop
    isRunning = true;
    int frameCount = 0;
    int detectionCount = 0;
    float totalFps = 0.0f;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "Starting detection loop..." << std::endl;
    
    while (isRunning) {
        // Get frame from camera with explicit error handling
        cv::Mat frame;
        try {
            frame = cameraProcessor->getFrame();
        } catch (const std::exception& e) {
            std::cerr << "Error getting frame: " << e.what() << std::endl;
            break;
        }
        
        // Check if frame is valid
        if (frame.empty()) {
            std::cout << "End of video or camera disconnected" << std::endl;
            break;
        }
        
        // Process frame with detection and explicit error handling
        cv::Mat processedFrame;
        try {
            processedFrame = processFrame(frame);
            
            // If processFrame returned an empty frame, skip this iteration
            if (processedFrame.empty()) {
                std::cerr << "Process frame returned empty result, skipping frame" << std::endl;
                continue;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing frame: " << e.what() << std::endl;
            continue;
        }
        
        // Get FPS
        float fps = 0.0f;
        try {
            fps = cameraProcessor->getFPS();
            totalFps += fps;
        } catch (const std::exception& e) {
            std::cerr << "Error getting FPS: " << e.what() << std::endl;
        }
        
        frameCount++;
        
        // Add FPS and TPU/CPU text to frame
        std::string fpsText = "FPS: " + std::to_string(int(fps));
        std::string modeText = model && model->isUsingTPU() ? "TPU" : "CPU";
        
        cv::putText(processedFrame, fpsText, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(processedFrame, modeText, cv::Point(processedFrame.cols - 100, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Display frame if requested
        if (displayOutput) {
            try {
                cv::imshow("Single Camera Detector", processedFrame);
                
                // Exit on ESC or 'q' key
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q') {
                    isRunning = false;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error displaying frame: " << e.what() << std::endl;
                displayOutput = false; // Disable display on error
            }
        }
        
        // Write frame to video if requested
        if (saveOutput && videoWriter.isOpened()) {
            try {
                videoWriter.write(processedFrame);
            } catch (const std::exception& e) {
                std::cerr << "Error writing frame to video: " << e.what() << std::endl;
                saveOutput = false; // Disable video saving on error
            }
        }
        
        // Print progress every 100 frames
        if (frameCount % 100 == 0) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                currentTime - startTime).count() / 1000.0;
            
            std::cout << "Processed " << frameCount << " frames in " 
                     << std::fixed << std::setprecision(1) << duration << " seconds ("
                     << std::fixed << std::setprecision(1) << frameCount / duration << " FPS)" << std::endl;
        }
        
        // Limit processing rate for display mode to avoid excessive CPU usage
        if (displayOutput && fps > 60) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    // Calculate and print statistics
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime).count() / 1000.0;
    
    std::cout << "\n=== Detection Statistics ===" << std::endl;
    std::cout << "Total frames processed: " << frameCount << std::endl;
    std::cout << "Detections: " << detectionCount << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(1) 
              << (frameCount > 0 ? totalFps / frameCount : 0) << std::endl;
    std::cout << "Total processing time: " << std::fixed << std::setprecision(1) 
              << duration << " seconds" << std::endl;
    
    // Release video writer and close windows
    if (saveOutput && videoWriter.isOpened()) {
        videoWriter.release();
        std::cout << "Output video saved to " << outputFilePath << std::endl;
    }
    
    if (displayOutput) {
        cv::destroyAllWindows();
    }
    
    // Stop the camera processor
    if (cameraProcessor) {
        cameraProcessor->stop();
    }
}

cv::Mat SingleCameraDetector::processFrame(const cv::Mat& frame) {
    // Check for valid input
    if (frame.empty()) {
        std::cerr << "Warning: Received empty frame in processFrame()" << std::endl;
        return cv::Mat(); // Return empty mat to indicate error
    }

    // Clone the frame to avoid modifying the original
    cv::Mat outputFrame = frame.clone();
    
    // Run detection
    std::vector<Detection> detections;
    
    if (model) {
        try {
            detections = model->processImage(frame, threshold);
        } catch (const std::exception& e) {
            std::cerr << "Error in model processing: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Warning: Model is null in processFrame()" << std::endl;
    }
    
    // Process detections
    int personCount = 0;
    
    for (const auto& detection : detections) {
        // Check if the detection is a person
        if (detection.id == personClassId) {
            personCount++;
            
            // Get bounding box coordinates
            int x1 = static_cast<int>(detection.bbox.xmin);
            int y1 = static_cast<int>(detection.bbox.ymin);
            int x2 = static_cast<int>(detection.bbox.xmax);
            int y2 = static_cast<int>(detection.bbox.ymax);
            
            // Ensure coordinates are within the frame
            x1 = std::max(0, std::min(x1, frame.cols - 1));
            y1 = std::max(0, std::min(y1, frame.rows - 1));
            x2 = std::max(0, std::min(x2, frame.cols - 1));
            y2 = std::max(0, std::min(y2, frame.rows - 1));
            
            // Draw bounding box
            cv::rectangle(outputFrame, cv::Point(x1, y1), cv::Point(x2, y2), 
                         cv::Scalar(0, 255, 0), 2);
            
            // Add label with confidence
            std::string label = model ? model->getLabel(detection.id) : "Person";
            label += " " + std::to_string(static_cast<int>(detection.score * 100)) + "%";
            
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                               0.5, 1, &baseline);
            
            cv::rectangle(outputFrame, cv::Point(x1, y1 - textSize.height - 5),
                         cv::Point(x1 + textSize.width, y1), 
                         cv::Scalar(0, 255, 0), cv::FILLED);
            
            cv::putText(outputFrame, label, cv::Point(x1, y1 - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }
    
    // If people were detected, add count to the frame
    if (personCount > 0) {
        std::string countText = "People: " + std::to_string(personCount);
        cv::putText(outputFrame, countText, cv::Point(10, frame.rows - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    
    return outputFrame;
}

float SingleCameraDetector::getFPS() const {
    return cameraProcessor ? cameraProcessor->getFPS() : 0.0f;
}

bool SingleCameraDetector::isSourceCamera(const std::string& source) const {
    // Try to parse source as an integer (camera index)
    try {
        std::stoi(source);
        return true;
    } catch (const std::exception&) {
        // If it's not an integer, check if it's an existing file
        return !fs::exists(source);
    }
} 