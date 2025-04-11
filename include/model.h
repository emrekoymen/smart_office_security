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
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
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

#ifndef DISABLE_TENSORFLOW
    /**
     * @brief Process an image frame to detect objects.
     * Requires an interpreter instance to be available.
     * @param frame The input image frame (expects BGR format).
     * @param threshold Minimum confidence score for detections.
     * @return A vector of detected objects.
     */
    std::vector<Detection> processImage(const cv::Mat& frame, float threshold);

    /**
     * @brief Create a new TFLite interpreter instance based on the loaded model.
     * Each thread should create its own interpreter.
     * @return A unique pointer to the created interpreter, or nullptr on failure.
     */
    std::unique_ptr<tflite::Interpreter> createInterpreter();
#endif

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
     * @brief Check if the model is configured to attempt using the Edge TPU
     * @return True if TPU usage is intended, false if CPU is forced
     */
    bool isUsingTPU() const { return useTPU; }

    /**
     * @brief Get the input height of the model
     * @return Input height in pixels
     */
    int getInputHeight() const { return inputHeight; }

    /**
     * @brief Get the input width of the model
     * @return Input width in pixels
     */
    int getInputWidth() const { return inputWidth; }

    /**
     * @brief Get the input channels of the model
     * @return Number of input channels
     */
    int getInputChannels() const { return inputChannels; }

private:
    // Labels map
    std::map<int, std::string> labels;
    
    // Input and output tensor dimensions
    int inputHeight;
    int inputWidth;
    int inputChannels;

#ifndef DISABLE_TENSORFLOW
    // TFLite model data (shared across interpreters)
    std::unique_ptr<tflite::FlatBufferModel> model;
    // TFLite interpreter instance (used by default processImage)
    // std::unique_ptr<tflite::Interpreter> interpreter;

    // Hold the Edge TPU context if initialized successfully
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpuContext;

    /**
     * @brief Try to initialize Edge TPU
     * @param modelPath Path to the model file
     * @return Shared pointer to the Edge TPU context if successful, nullptr otherwise.
     */
    std::shared_ptr<edgetpu::EdgeTpuContext> initializeEdgeTPU(const std::string& modelPath);
    
    // Flag indicating if TPU should be used (based on forceCPU and TPU availability)
    bool useTPU;
#else
    // Keep track of forced CPU mode even if TF is disabled
    bool useTPU; // Will be false if TF disabled or forceCPU is true
#endif
};

#endif // MODEL_H 