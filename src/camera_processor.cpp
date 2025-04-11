#include "camera_processor.h"
#include <iostream>
#include <chrono>

#ifndef DISABLE_TENSORFLOW
#include <tensorflow/lite/interpreter.h>
#endif

CameraProcessor::CameraProcessor(int cameraId, std::shared_ptr<Model> model, int personClassId, float threshold)
    : cameraId(cameraId), model(model), personClassId(personClassId), threshold(threshold),
      frameWidth(0), frameHeight(0), fps(0), running(false), totalFrames(0), totalDetections(0) {
#ifndef DISABLE_TENSORFLOW
    // Create a dedicated interpreter for this processor thread
    try {
        interpreter = model->createInterpreter();
        if (!interpreter) {
            throw std::runtime_error("Failed to create interpreter for CameraProcessor");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error creating interpreter for Camera " << cameraId << ": " << e.what() << std::endl;
        // Handle error appropriately, maybe set a flag?
        throw; // Re-throw for now
    }
#endif
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
        
        // Attempt to set desired resolution (640x480)
        std::cout << "Camera " << cameraId << ": Attempting to set resolution to 640x480" << std::endl;
        bool resWidthSet = cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        bool resHeightSet = cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        if (!resWidthSet || !resHeightSet) {
            std::cout << "Warning: Camera " << cameraId << " failed to set desired resolution." << std::endl;
        }
        
        // Get actual camera properties after attempting to set them
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
        {
            std::unique_lock<std::mutex> lock(frameQueueMutex);
            frameQueueCondition.wait(lock, [this] { return !frameQueue.empty() || !running; });
            if (!running && frameQueue.empty()) {
                break;
            }
            frame = frameQueue.front();
            frameQueue.pop();
        }

        if (frame.empty()) continue;

        auto start = std::chrono::high_resolution_clock::now();
        
        // Perform inference using the local interpreter
        std::vector<Detection> detections = performInference(frame);
        
        auto end = std::chrono::high_resolution_clock::now();
        double processTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        fpsCounter.addFrame(processTime);

        // Create result object
        auto result = std::make_shared<ProcessedResult>();
        result->frame = frame.clone(); // Clone frame for result
        result->detections = detections;
        result->processingTime = processTime;
        result->fps = fpsCounter.getFPS();
        
        // Draw bounding boxes and labels
        for (const auto& det : detections) {
             drawDetectionBox(result->frame, det.bbox.xmin, det.bbox.ymin, 
                             det.bbox.xmax, det.bbox.ymax, model->getLabel(det.id), det.score);
        }
        
        // Update detection count
        if (!detections.empty()) {
            totalDetections++;
        }

        {
            std::lock_guard<std::mutex> lock(resultQueueMutex);
            // Keep only the latest result if queue grows too large (optional)
            if (resultQueue.size() > maxQueueSize) {
                 resultQueue.pop(); // Discard oldest
            }
            resultQueue.push(result);
        }

        {
            std::lock_guard<std::mutex> lock(lastResultMutex);
            lastResult = result;
        }
    }
}

#ifndef DISABLE_TENSORFLOW
std::vector<Detection> CameraProcessor::performInference(const cv::Mat& frame) {
    if (!interpreter) {
        std::cerr << "Interpreter not available for camera " << cameraId << std::endl;
        return {};
    }
    
    // Store original frame dimensions
    int origHeight = frame.rows;
    int origWidth = frame.cols;
    
    // Get model input dimensions
    int inputWidth = model->getInputWidth();
    int inputHeight = model->getInputHeight();
    int inputChannels = model->getInputChannels();

    // Calculate scale factors
    float widthScale = static_cast<float>(origWidth) / inputWidth;
    float heightScale = static_cast<float>(origHeight) / inputHeight;
    
    // Resize and preprocess the image
    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, cv::Size(inputWidth, inputHeight));
    
    cv::Mat rgbFrame;
    if (resizedFrame.channels() == 3 && inputChannels == 3) {
        cv::cvtColor(resizedFrame, rgbFrame, cv::COLOR_BGR2RGB);
    } else if (resizedFrame.channels() == 1 && inputChannels == 1) {
        rgbFrame = resizedFrame; // Grayscale model
    } else {
         std::cerr << "Channel mismatch between input frame and model" << std::endl;
         return {};
    }
    
    // Get input tensor
    // Assuming UINT8 model for now, adjust if using FLOAT32
    uint8_t* inputTensorData = interpreter->typed_input_tensor<uint8_t>(0);
    if (!inputTensorData) {
        std::cerr << "Failed to get input tensor data" << std::endl;
        return {};
    }
    
    // Copy image data to input tensor
    memcpy(inputTensorData, rgbFrame.data, inputWidth * inputHeight * inputChannels);
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter for camera " << cameraId << std::endl;
        return {};
    }
    
    // Get output tensors (adjust indices based on your model's output signature)
    const float* outputLocations = interpreter->typed_output_tensor<float>(0);
    const float* outputClasses = interpreter->typed_output_tensor<float>(1);
    const float* outputScores = interpreter->typed_output_tensor<float>(2);
    const float* numDetections = interpreter->typed_output_tensor<float>(3);

    if (!outputLocations || !outputClasses || !outputScores || !numDetections) {
        std::cerr << "Failed to get output tensors" << std::endl;
        return {};
    }
    
    // Number of detections
    int numDetected = static_cast<int>(*numDetections);
    std::vector<Detection> detections;
    
    // Create detection objects
    for (int i = 0; i < numDetected; i++) {
        float score = outputScores[i];
        
        // Filter by threshold
        if (score >= threshold) {
            int classId = static_cast<int>(outputClasses[i]);
            
            // Check if the detected class is the one we are interested in (personClassId)
            if (classId == personClassId) { 
                // Get bounding box coordinates (in format [ymin, xmin, ymax, xmax])
                // Ensure indices are correct for your model
                float ymin = outputLocations[i * 4 + 0]; 
                float xmin = outputLocations[i * 4 + 1];
                float ymax = outputLocations[i * 4 + 2];
                float xmax = outputLocations[i * 4 + 3];
                
                // Clamp coordinates to [0.0, 1.0]
                ymin = std::max(0.0f, ymin);
                xmin = std::max(0.0f, xmin);
                ymax = std::min(1.0f, ymax);
                xmax = std::min(1.0f, xmax);
                
                // Scale bounding box coordinates to original frame size
                float scaledYmin = ymin * origHeight;
                float scaledXmin = xmin * origWidth;
                float scaledYmax = ymax * origHeight;
                float scaledXmax = xmax * origWidth;
                
                // Create detection object with scaled coordinates
                detections.emplace_back(BBox(scaledYmin, scaledXmin, scaledYmax, scaledXmax), classId, score);
            }
        }
    }
    
    return detections;
}
#endif

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