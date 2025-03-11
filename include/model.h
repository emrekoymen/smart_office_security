#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>

// Conditionally include TensorFlow Lite headers
#ifndef DISABLE_TENSORFLOW
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <edgetpu.h>
#endif

/**
 * @brief Bounding box class for object detection
 */
class BBox {
public:
    BBox(float ymin, float xmin, float ymax, float xmax)
        : ymin(ymin), xmin(xmin), ymax(ymax), xmax(xmax) {}

    float ymin;
    float xmin;
    float ymax;
    float xmax;
};

/**
 * @brief Detection class for object detection results
 */
class Detection {
public:
    Detection(const BBox& bbox, int id, float score)
        : bbox(bbox), id(id), score(score) {}

    BBox bbox;
    int id;
    float score;
};

/**
 * @brief Model class for TFLite model with Edge TPU support
 */
class Model {
public:
    /**
     * @brief Constructor
     * @param modelPath Path to the TFLite model
     * @param labelsPath Path to the labels file
     * @param forceCPU Force CPU mode (no TPU)
     */
    Model(const std::string& modelPath, const std::string& labelsPath, bool forceCPU = false);

    /**
     * @brief Destructor
     */
    ~Model();

    /**
     * @brief Process an image with the model
     * @param frame Input frame
     * @param threshold Detection threshold
     * @return Vector of detections
     */
    std::vector<Detection> processImage(const cv::Mat& frame, float threshold = 0.5);

    /**
     * @brief Load labels from a file
     * @param path Path to the labels file
     * @return True if successful, false otherwise
     */
    bool loadLabels(const std::string& path);

    /**
     * @brief Get the label for a class ID
     * @param id Class ID
     * @return Label string
     */
    std::string getLabel(int id) const;

    /**
     * @brief Check if the model is using the Edge TPU
     * @return True if using TPU, false if using CPU
     */
    bool isUsingTPU() const { return usingTPU; }

private:
    // Labels map
    std::map<int, std::string> labels;
    
    // Flag to track whether we're using the Edge TPU or CPU
    bool usingTPU;
    
    // Input and output tensor dimensions
    int inputHeight;
    int inputWidth;
    int inputChannels;

#ifndef DISABLE_TENSORFLOW
    // TFLite model and interpreter
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    
    // Edge TPU context
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpuContext;
#endif
};

#endif // MODEL_H 