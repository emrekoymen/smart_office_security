#include "model.h"
#include <fstream>
#include <iostream>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/tools/delegates/delegate_provider.h>

Model::Model(const std::string& modelPath, const std::string& labelsPath, bool forceCPU)
    : usingTPU(false), inputHeight(300), inputWidth(300), inputChannels(3) {
    
    std::cout << "Loading model from " << modelPath << std::endl;
    
    // Load the model
    model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model) {
        throw std::runtime_error("Failed to load model: " + modelPath);
    }
    
    // Create the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    
    if (!interpreter) {
        throw std::runtime_error("Failed to create interpreter");
    }
    
    // Try to use Edge TPU if not forced to CPU
    if (!forceCPU && modelPath.find("edgetpu") != std::string::npos) {
        try {
            std::cout << "Attempting to use Edge TPU" << std::endl;
            edgetpuContext = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
            if (edgetpuContext) {
                interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpuContext.get());
                interpreter->SetNumThreads(1);  // Edge TPU doesn't benefit from multiple threads
                usingTPU = true;
                std::cout << "Successfully loaded Edge TPU model" << std::endl;
            } else {
                std::cout << "Edge TPU not available" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "Edge TPU error: " << e.what() << std::endl;
            std::cout << "Falling back to CPU" << std::endl;
        }
    }
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors");
    }
    
    // Get input tensor dimensions
    auto* inputTensor = interpreter->input_tensor(0);
    if (inputTensor) {
        inputHeight = inputTensor->dims->data[1];
        inputWidth = inputTensor->dims->data[2];
        inputChannels = inputTensor->dims->data[3];
    }
    
    std::cout << "Model input dimensions: " << inputWidth << "x" << inputHeight 
              << "x" << inputChannels << std::endl;
    
    // Load labels
    if (!loadLabels(labelsPath)) {
        std::cout << "Warning: Failed to load labels from " << labelsPath << std::endl;
    }
    
    std::cout << "Model loaded successfully. Using " 
              << (usingTPU ? "Edge TPU" : "CPU") << std::endl;
}

Model::~Model() {
    // Edge TPU context will be automatically released by shared_ptr
}

std::vector<Detection> Model::processImage(const cv::Mat& frame, float threshold) {
    // Store original frame dimensions
    int origHeight = frame.rows;
    int origWidth = frame.cols;
    
    // Calculate scale factors
    float widthScale = static_cast<float>(origWidth) / inputWidth;
    float heightScale = static_cast<float>(origHeight) / inputHeight;
    
    // Resize and preprocess the image
    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, cv::Size(inputWidth, inputHeight));
    
    // Convert BGR to RGB if needed
    cv::Mat rgbFrame;
    if (resizedFrame.channels() == 3) {
        cv::cvtColor(resizedFrame, rgbFrame, cv::COLOR_BGR2RGB);
    } else {
        rgbFrame = resizedFrame;
    }
    
    // Get input tensor
    uint8_t* inputTensorData = interpreter->typed_input_tensor<uint8_t>(0);
    
    // Copy image data to input tensor
    memcpy(inputTensorData, rgbFrame.data, inputWidth * inputHeight * inputChannels);
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Failed to invoke interpreter");
    }
    
    // Get output tensors
    float* outputLocations = interpreter->typed_output_tensor<float>(0);
    float* outputClasses = interpreter->typed_output_tensor<float>(1);
    float* outputScores = interpreter->typed_output_tensor<float>(2);
    float* numDetections = interpreter->typed_output_tensor<float>(3);
    
    // Number of detections
    int numDetected = static_cast<int>(*numDetections);
    
    // Create detection objects
    std::vector<Detection> detections;
    for (int i = 0; i < numDetected; i++) {
        float score = outputScores[i];
        
        // Filter by threshold
        if (score >= threshold) {
            int classId = static_cast<int>(outputClasses[i]);
            
            // Get bounding box coordinates (in format [ymin, xmin, ymax, xmax])
            float ymin = outputLocations[i * 4];
            float xmin = outputLocations[i * 4 + 1];
            float ymax = outputLocations[i * 4 + 2];
            float xmax = outputLocations[i * 4 + 3];
            
            // Scale bounding box coordinates to original frame size
            ymin *= heightScale;
            xmin *= widthScale;
            ymax *= heightScale;
            xmax *= widthScale;
            
            // Create detection object
            detections.emplace_back(BBox(ymin, xmin, ymax, xmax), classId, score);
        }
    }
    
    return detections;
}

bool Model::loadLabels(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open labels file: " << path << std::endl;
        return false;
    }
    
    std::string line;
    int lineNum = 0;
    while (std::getline(file, line)) {
        // Remove trailing whitespace
        line.erase(line.find_last_not_of(" \n\r\t") + 1);
        
        // Add to labels map
        labels[lineNum] = line;
        lineNum++;
    }
    
    std::cout << "Loaded " << labels.size() << " labels" << std::endl;
    return true;
}

std::string Model::getLabel(int id) const {
    auto it = labels.find(id);
    if (it != labels.end()) {
        return it->second;
    }
    return "Unknown";
} 