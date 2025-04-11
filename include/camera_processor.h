#ifndef CAMERA_PROCESSOR_H
#define CAMERA_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory>
#include "model.h"
#include "utils.h"

// Forward declaration for TFLite interpreter if not disabled
#ifndef DISABLE_TENSORFLOW
namespace tflite {
    class Interpreter;
}
#endif

/**
 * @brief Result structure for processed frames
 */
struct ProcessedResult {
    cv::Mat frame;
    std::vector<Detection> detections;
    double processingTime;
    double fps;
};

/**
 * @brief Camera processor class to handle processing of a single camera feed
 */
class CameraProcessor {
public:
    /**
     * @brief Constructor
     * @param cameraId Camera identifier
     * @param model Shared pointer to the model
     * @param personClassId Class ID for "person"
     * @param threshold Detection threshold
     */
    CameraProcessor(int cameraId, std::shared_ptr<Model> model, int personClassId = 0, float threshold = 0.5);

    /**
     * @brief Destructor
     */
    ~CameraProcessor();

    /**
     * @brief Open the camera source
     * @param source Camera source (index or path)
     * @return True if successful, false otherwise
     */
    bool openCamera(const std::string& source);

    /**
     * @brief Start processing the camera feed
     * @return True if successful, false otherwise
     */
    bool start();

    /**
     * @brief Stop processing the camera feed
     */
    void stop();

    /**
     * @brief Get the latest processed result
     * @return Shared pointer to the latest result
     */
    std::shared_ptr<ProcessedResult> getLatestResult();
    
    /**
     * @brief Get the latest frame (for direct access)
     * @return Latest frame from the camera
     */
    cv::Mat getFrame() {
        std::lock_guard<std::mutex> lock(lastFrameMutex);
        if (lastFrame.empty() && cap.isOpened()) {
            // Try to read a frame directly if the stored frame is empty
            cv::Mat frame;
            if (cap.read(frame)) {
                lastFrame = frame.clone();
            } else {
                std::cerr << "Failed to read frame directly from camera/video" << std::endl;
            }
        }
        return lastFrame.clone();
    }

    /**
     * @brief Get the camera ID
     * @return Camera ID
     */
    int getCameraId() const { return cameraId; }

    /**
     * @brief Get the frame width
     * @return Frame width
     */
    int getFrameWidth() const { return frameWidth; }

    /**
     * @brief Get the frame height
     * @return Frame height
     */
    int getFrameHeight() const { return frameHeight; }

    /**
     * @brief Get the camera FPS
     * @return Camera FPS
     */
    double getFPS() const { return fps; }

    /**
     * @brief Get the total number of frames processed
     * @return Total frames
     */
    int getTotalFrames() const { return totalFrames; }

    /**
     * @brief Get the total number of detections
     * @return Total detections
     */
    int getTotalDetections() const { return totalDetections; }

    /**
     * @brief Check if the processor is running
     * @return True if running, false otherwise
     */
    bool isRunning() const { return running; }

    /**
     * @brief Perform inference on a frame using the local interpreter
     * @param frame Input frame
     * @return Vector of detections
     */
    std::vector<Detection> performInference(const cv::Mat& frame);

private:
    /**
     * @brief Thread function to capture frames from the camera
     */
    void captureFrames();

    /**
     * @brief Thread function to process frames from the queue
     */
    void processFrames();

    // Camera properties
    int cameraId;
    cv::VideoCapture cap;
    int frameWidth;
    int frameHeight;
    double fps;

    // Model and detection parameters
    std::shared_ptr<Model> model;
    int personClassId;
    float threshold;

    // Thread control
    std::atomic<bool> running;
    std::thread captureThread;
    std::thread processingThread;

#ifndef DISABLE_TENSORFLOW
    // Each processor gets its own interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;
#endif

    // Frame and result queues
    std::queue<cv::Mat> frameQueue;
    std::mutex frameQueueMutex;
    std::condition_variable frameQueueCondition;

    std::queue<std::shared_ptr<ProcessedResult>> resultQueue;
    std::mutex resultQueueMutex;

    // Latest frame and result for direct access
    cv::Mat lastFrame;
    std::mutex lastFrameMutex;

    std::shared_ptr<ProcessedResult> lastResult;
    std::mutex lastResultMutex;

    // Performance metrics
    FPSCounter fpsCounter;
    std::atomic<int> totalFrames;
    std::atomic<int> totalDetections;

    // Maximum queue sizes
    const int maxQueueSize = 10;
};

#endif // CAMERA_PROCESSOR_H 