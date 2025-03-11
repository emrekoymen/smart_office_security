#include "camera_processor.h"
#include <iostream>
#include <chrono>

CameraProcessor::CameraProcessor(int cameraId, std::shared_ptr<Model> model, int personClassId, float threshold)
    : cameraId(cameraId), model(model), personClassId(personClassId), threshold(threshold),
      frameWidth(0), frameHeight(0), fps(30.0), running(false), totalFrames(0), totalDetections(0) {
}

CameraProcessor::~CameraProcessor() {
    stop();
}

bool CameraProcessor::openCamera(const std::string& source) {
    try {
        // If source is a digit string, convert to int for webcam
        if (std::all_of(source.begin(), source.end(), ::isdigit)) {
            cap.open(std::stoi(source));
        } else {
            cap.open(source);
        }
        
        if (!cap.isOpened()) {
            std::cerr << "Could not open video source: " << source << std::endl;
            return false;
        }
        
        // Get camera properties
        frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps = cap.get(cv::CAP_PROP_FPS);
        
        // If FPS is not available, use default
        if (fps <= 0) {
            fps = 30.0;
        }
        
        std::cout << "Camera " << cameraId << " opened: " << frameWidth << "x" << frameHeight 
                  << " @ " << fps << " FPS" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error opening camera " << cameraId << ": " << e.what() << std::endl;
        return false;
    }
}

bool CameraProcessor::start() {
    if (!cap.isOpened()) {
        std::cerr << "Camera " << cameraId << " not opened" << std::endl;
        return false;
    }
    
    running = true;
    
    // Start capture thread
    captureThread = std::thread(&CameraProcessor::captureFrames, this);
    
    // Start processing thread
    processingThread = std::thread(&CameraProcessor::processFrames, this);
    
    return true;
}

void CameraProcessor::stop() {
    running = false;
    
    // Notify processing thread to wake up
    frameQueueCondition.notify_all();
    
    // Join threads
    if (captureThread.joinable()) {
        captureThread.join();
    }
    
    if (processingThread.joinable()) {
        processingThread.join();
    }
    
    // Release camera
    if (cap.isOpened()) {
        cap.release();
    }
}

void CameraProcessor::captureFrames() {
    while (running) {
        cv::Mat frame;
        bool ret = cap.read(frame);
        
        if (!ret || frame.empty()) {
            std::cerr << "Camera " << cameraId << ": End of video stream" << std::endl;
            running = false;
            break;
        }
        
        // Store frame in queue
        {
            std::lock_guard<std::mutex> lock(frameQueueMutex);
            
            // If queue is full, remove oldest frame
            if (frameQueue.size() >= maxQueueSize) {
                frameQueue.pop();
            }
            
            frameQueue.push(frame.clone());
            
            // Store last frame for direct access
            {
                std::lock_guard<std::mutex> frameLock(lastFrameMutex);
                lastFrame = frame.clone();
            }
            
            // Increment frame counter
            totalFrames++;
        }
        
        // Notify processing thread
        frameQueueCondition.notify_one();
        
        // Short sleep to prevent CPU hogging
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void CameraProcessor::processFrames() {
    while (running) {
        cv::Mat frame;
        
        // Get frame from queue
        {
            std::unique_lock<std::mutex> lock(frameQueueMutex);
            
            // Wait for frame or exit signal
            frameQueueCondition.wait(lock, [this] {
                return !frameQueue.empty() || !running;
            });
            
            // Check if we should exit
            if (!running && frameQueue.empty()) {
                break;
            }
            
            // Get frame from queue
            if (!frameQueue.empty()) {
                frame = frameQueue.front();
                frameQueue.pop();
            } else {
                continue;
            }
        }
        
        // Process the frame
        auto frameStartTime = std::chrono::high_resolution_clock::now();
        
        // Process image with model
        std::vector<Detection> detections;
        try {
            detections = model->processImage(frame, threshold);
        } catch (const std::exception& e) {
            std::cerr << "Error processing frame: " << e.what() << std::endl;
            continue;
        }
        
        // Filter for person detections
        std::vector<Detection> personDetections;
        for (const auto& detection : detections) {
            if (detection.id == personClassId) {
                personDetections.push_back(detection);
            }
        }
        
        // Update FPS counter
        fpsCounter.update();
        
        // Process detections
        for (const auto& detection : personDetections) {
            // Draw detection on frame
            drawDetectionBox(
                frame,
                detection.bbox.xmin,
                detection.bbox.ymin,
                detection.bbox.xmax,
                detection.bbox.ymax,
                model->getLabel(detection.id),
                detection.score
            );
        }
        
        // Calculate processing time
        auto processingTime = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - frameStartTime).count() / 1000000.0;
        
        // Draw FPS on frame
        fpsCounter.drawFPS(frame);
        
        // Draw camera ID on frame
        cv::putText(frame, "Camera " + std::to_string(cameraId), cv::Point(10, 90),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        
        // Draw mode indicator (TPU or CPU)
        std::string modeText = model->isUsingTPU() ? "TPU" : "CPU";
        cv::putText(frame, "Mode: " + modeText, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        
        // Create result object
        auto result = std::make_shared<ProcessedResult>();
        result->frame = frame.clone();
        result->detections = personDetections;
        result->processingTime = processingTime;
        result->fps = fpsCounter.getFPS();
        
        // Store result in queue
        {
            std::lock_guard<std::mutex> lock(resultQueueMutex);
            
            // If queue is full, remove oldest result
            if (resultQueue.size() >= maxQueueSize) {
                resultQueue.pop();
            }
            
            resultQueue.push(result);
            
            // Store last result for direct access
            {
                std::lock_guard<std::mutex> resultLock(lastResultMutex);
                lastResult = result;
            }
            
            // Update detection counter
            if (!personDetections.empty()) {
                totalDetections++;
            }
        }
    }
}

std::shared_ptr<ProcessedResult> CameraProcessor::getLatestResult() {
    std::lock_guard<std::mutex> lock(resultQueueMutex);
    
    if (!resultQueue.empty()) {
        auto result = resultQueue.front();
        resultQueue.pop();
        return result;
    }
    
    // If queue is empty, return last result
    std::lock_guard<std::mutex> resultLock(lastResultMutex);
    return lastResult;
} 