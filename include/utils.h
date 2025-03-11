#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <deque>
#include <string>
#include <fstream>
#include <iostream>
#include <mutex>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * @brief FPS counter class to measure and display frames per second
 */
class FPSCounter {
public:
    FPSCounter(int avgFrames = 30) : avgFrames(avgFrames) {}

    /**
     * @brief Update the FPS calculation with a new frame
     */
    void update() {
        auto now = std::chrono::high_resolution_clock::now();
        frameTimes.push_back(now);
        
        // Keep only the last avgFrames times
        while (frameTimes.size() > avgFrames) {
            frameTimes.pop_front();
        }
    }

    /**
     * @brief Get the current FPS
     * @return Current FPS value
     */
    double getFPS() const {
        if (frameTimes.size() <= 1) {
            return 0.0;
        }

        // Calculate time difference between oldest and newest frame
        auto timeDiff = std::chrono::duration_cast<std::chrono::microseconds>(
            frameTimes.back() - frameTimes.front()).count() / 1000000.0;
        
        if (timeDiff <= 0.0) {
            return 0.0;
        }

        // FPS = (number of frames - 1) / time difference
        return (frameTimes.size() - 1) / timeDiff;
    }

    /**
     * @brief Draw FPS on frame
     * @param frame Frame to draw on
     * @return Frame with FPS drawn
     */
    cv::Mat drawFPS(cv::Mat& frame) const {
        double fps = getFPS();
        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)), 
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                    cv::Scalar(0, 255, 0), 2);
        return frame;
    }

private:
    int avgFrames;
    std::deque<std::chrono::time_point<std::chrono::high_resolution_clock>> frameTimes;
};

/**
 * @brief Logger class to handle logging detection events and performance metrics
 */
class Logger {
public:
    Logger(const std::string& logDir = "logs") : logDir(logDir) {
        // Create log directory if it doesn't exist
        if (!fs::exists(logDir)) {
            fs::create_directories(logDir);
        }

        // Create log file with timestamp
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << logDir << "/detection_log_" << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S") << ".log";
        logFile = ss.str();

        // Open log file
        logStream.open(logFile, std::ios::out | std::ios::app);
    }

    ~Logger() {
        if (logStream.is_open()) {
            logStream.close();
        }
    }

    /**
     * @brief Log a detection event
     * @param label Label of the detected object
     * @param confidence Confidence score of the detection
     * @param imagePath Path to the saved image (optional)
     */
    void logDetection(const std::string& label, float confidence, const std::string& imagePath = "") {
        std::lock_guard<std::mutex> lock(logMutex);
        
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        
        std::string logEntry = ss.str() + " - Detected: " + label + " (Confidence: " + 
                              std::to_string(confidence) + ")";
        
        if (!imagePath.empty()) {
            logEntry += " - Image saved: " + imagePath;
        }
        
        if (logStream.is_open()) {
            logStream << logEntry << std::endl;
        }
        
        std::cout << logEntry << std::endl;
    }

    /**
     * @brief Log performance metrics
     * @param fps Current FPS
     * @param processingTime Processing time in seconds
     */
    void logPerformance(double fps, double processingTime) {
        std::lock_guard<std::mutex> lock(logMutex);
        
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        
        std::string logEntry = ss.str() + " - Performance: FPS: " + std::to_string(fps) + 
                              ", Processing time: " + std::to_string(processingTime) + "s";
        
        if (logStream.is_open()) {
            logStream << logEntry << std::endl;
        }
    }

private:
    std::string logDir;
    std::string logFile;
    std::ofstream logStream;
    std::mutex logMutex;
};

/**
 * @brief Alert system class to handle alerts when a person is detected
 */
class AlertSystem {
public:
    AlertSystem(float minConfidence = 0.5, int cooldownPeriod = 10) 
        : minConfidence(minConfidence), cooldownPeriod(cooldownPeriod), lastAlertTime(0), enabled(true) {}

    /**
     * @brief Check if enough time has passed since the last alert
     * @return True if an alert can be triggered, false otherwise
     */
    bool canAlert() const {
        auto currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        return (currentTime - lastAlertTime > cooldownPeriod);
    }

    /**
     * @brief Trigger an alert if confidence is high enough and cooldown has passed
     * @param confidence Confidence score of the detection
     * @param label Label of the detected object
     * @return True if alert was triggered, false otherwise
     */
    bool triggerAlert(float confidence, const std::string& label = "Person") {
        if (!enabled || confidence < minConfidence) {
            return false;
        }
        
        if (canAlert()) {
            lastAlertTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            
            // Print alert to console (in a real system, this could send notifications)
            std::cout << "ALERT: " << label << " detected with " << confidence << " confidence!" << std::endl;
            
            return true;
        }
        
        return false;
    }

    /**
     * @brief Enable or disable the alert system
     * @param enable True to enable, false to disable
     */
    void setEnabled(bool enable) {
        enabled = enable;
    }

private:
    float minConfidence;
    int cooldownPeriod;
    time_t lastAlertTime;
    bool enabled;
};

/**
 * @brief Draw a detection bounding box on a frame
 * @param frame Frame to draw on
 * @param xmin Left coordinate
 * @param ymin Top coordinate
 * @param xmax Right coordinate
 * @param ymax Bottom coordinate
 * @param label Label text
 * @param score Confidence score
 * @return Frame with bounding box drawn
 */
inline cv::Mat drawDetectionBox(cv::Mat& frame, float xmin, float ymin, float xmax, float ymax, 
                         const std::string& label, float score) {
    // Get frame dimensions
    int height = frame.rows;
    int width = frame.cols;
    
    // Check if coordinates are normalized (between 0 and 1) or absolute
    if (xmin <= 1.0f && ymin <= 1.0f && xmax <= 1.0f && ymax <= 1.0f) {
        // Convert normalized coordinates to pixel values
        xmin = xmin * width;
        xmax = xmax * width;
        ymin = ymin * height;
        ymax = ymax * height;
    }
    
    // Convert to int
    int xminInt = static_cast<int>(xmin);
    int yminInt = static_cast<int>(ymin);
    int xmaxInt = static_cast<int>(xmax);
    int ymaxInt = static_cast<int>(ymax);
    
    // Draw bounding box with thicker lines
    cv::rectangle(frame, cv::Point(xminInt, yminInt), cv::Point(xmaxInt, ymaxInt), 
                 cv::Scalar(0, 255, 0), 3);
    
    // Prepare label text with confidence score
    std::string labelText = label + ": " + std::to_string(score).substr(0, 4);
    
    // Get text size for background rectangle
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.7;
    int thickness = 2;
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(labelText, fontFace, fontScale, thickness, &baseline);
    
    // Draw background rectangle for text
    cv::rectangle(frame, 
                 cv::Point(xminInt, yminInt - textSize.height - 10), 
                 cv::Point(xminInt + textSize.width + 10, yminInt), 
                 cv::Scalar(0, 0, 0), 
                 -1);  // -1 means filled rectangle
    
    // Draw label text on the background rectangle
    cv::putText(frame, 
                labelText, 
                cv::Point(xminInt + 5, yminInt - 5), 
                fontFace, 
                fontScale, 
                cv::Scalar(0, 255, 0), 
                thickness);
    
    return frame;
}

/**
 * @brief Save a detection image to disk
 * @param frame Frame to save
 * @param outputDir Output directory
 * @return Path to the saved image
 */
inline std::string saveDetectionImage(const cv::Mat& frame, const std::string& outputDir = "detections") {
    // Create output directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }
    
    // Generate filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << outputDir << "/detection_" << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S") << ".jpg";
    std::string filename = ss.str();
    
    // Save image
    cv::imwrite(filename, frame);
    return filename;
}

#endif // UTILS_H 