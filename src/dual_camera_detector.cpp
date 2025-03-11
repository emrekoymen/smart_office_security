#include "dual_camera_detector.h"
#include <iostream>
#include <filesystem>
#include <thread>
#include <signal.h>
#include <getopt.h>

namespace fs = std::filesystem;

// Global flag for signal handling
static std::atomic<bool> g_running(true);

// Signal handler
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received. Exiting..." << std::endl;
    g_running = false;
}

DualCameraDetector::DualCameraDetector(const CommandLineArgs& args)
    : args(args), videoWriterInitialized(false), displayAvailable(false), running(true) {
    
    // Register signal handler
    signal(SIGINT, signalHandler);
}

DualCameraDetector::~DualCameraDetector() {
    // Release video writer
    if (videoWriterInitialized) {
        videoWriter.release();
    }
}

bool DualCameraDetector::initialize() {
    // Check display availability
    displayAvailable = !args.noDisplay && checkDisplayAvailability();
    
    // If display is not available but was requested, adjust settings
    if (!args.noDisplay && !displayAvailable) {
        std::cout << "Display requested but not available. Running without display." << std::endl;
        args.noDisplay = true;
        
        // Enable video saving if display is not available
        if (!args.saveVideo) {
            std::cout << "Enabling video saving since display is not available." << std::endl;
            args.saveVideo = true;
        }
    }
    
    // Create output directory if saving video
    if (args.saveVideo) {
        fs::create_directories(args.outputDir);
    }
    
    // Check if model exists
    if (!fs::exists(args.modelPath)) {
        std::cerr << "Error: Model file " << args.modelPath << " not found" << std::endl;
        std::cerr << "Please run setup_coral.sh to download the model files" << std::endl;
        return false;
    }
    
    // Load model
    try {
        model = std::make_shared<Model>(args.modelPath, args.labelsPath, args.forceCPU);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
    
    // Get person class label
    std::string personLabel = model->getLabel(args.personClassId);
    std::cout << "Looking for objects of class: " << personLabel 
              << " (ID: " << args.personClassId << ")" << std::endl;
    
    // Initialize utility classes
    logger = std::make_unique<Logger>();
    alertSystem = std::make_unique<AlertSystem>(args.threshold);
    
    // Initialize camera processors
    camera1 = std::make_unique<CameraProcessor>(1, model, args.personClassId, args.threshold);
    camera2 = std::make_unique<CameraProcessor>(2, model, args.personClassId, args.threshold);
    
    // Open cameras
    std::cout << "Opening camera 1: " << args.camera1 << std::endl;
    if (!camera1->openCamera(args.camera1)) {
        std::cerr << "Failed to open camera 1" << std::endl;
        return false;
    }
    
    std::cout << "Opening camera 2: " << args.camera2 << std::endl;
    if (!camera2->openCamera(args.camera2)) {
        std::cerr << "Failed to open camera 2" << std::endl;
        return false;
    }
    
    // Start camera processors
    if (!camera1->start()) {
        std::cerr << "Failed to start camera 1" << std::endl;
        return false;
    }
    
    if (!camera2->start()) {
        std::cerr << "Failed to start camera 2" << std::endl;
        camera1->stop();
        return false;
    }
    
    return true;
}

bool DualCameraDetector::run() {
    // Initialize video writer if saving
    if (args.saveVideo) {
        // Use the first camera's dimensions for the combined output
        int frameWidth = camera1->getFrameWidth() * 2;  // Side by side
        int frameHeight = std::max(camera1->getFrameHeight(), camera2->getFrameHeight());
        
        // Generate output filename
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << args.outputDir << "/dual_camera_processed_" 
           << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S") << ".mp4";
        std::string outputPath = ss.str();
        
        // Define codec and create VideoWriter object
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        videoWriter.open(outputPath, fourcc, 30.0, cv::Size(frameWidth, frameHeight));
        
        if (!videoWriter.isOpened()) {
            std::cerr << "Failed to open video writer" << std::endl;
            return false;
        }
        
        videoWriterInitialized = true;
        std::cout << "Saving processed video to: " << outputPath << std::endl;
    }
    
    // Processing loop
    startTime = std::chrono::high_resolution_clock::now();
    auto lastLogTime = startTime;
    
    // Main processing loop
    processCameraFeeds();
    
    // Print summary
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime).count() / 1000.0;
    
    std::cout << "Camera 1: Processed " << camera1->getTotalFrames() << " frames, detected persons in " 
              << camera1->getTotalDetections() << " frames" << std::endl;
    std::cout << "Camera 2: Processed " << camera2->getTotalFrames() << " frames, detected persons in " 
              << camera2->getTotalDetections() << " frames" << std::endl;
    
    double avgFps1 = camera1->getTotalFrames() / elapsedTime;
    double avgFps2 = camera2->getTotalFrames() / elapsedTime;
    
    std::cout << "Average FPS: Camera 1: " << avgFps1 << ", Camera 2: " << avgFps2 << std::endl;
    
    return true;
}

void DualCameraDetector::processCameraFeeds() {
    auto lastLogTime = std::chrono::high_resolution_clock::now();
    
    while (running && g_running) {
        // Get latest results from both cameras
        auto result1 = camera1->getLatestResult();
        auto result2 = camera2->getLatestResult();
        
        // Skip if either result is nullptr
        if (!result1 || !result2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Get frames from results
        cv::Mat& frame1 = result1->frame;
        cv::Mat& frame2 = result2->frame;
        
        // Skip if either frame is empty
        if (frame1.empty() || frame2.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Process alerts for both cameras
        for (const auto& detection : result1->detections) {
            // Trigger alert
            if (alertSystem->triggerAlert(detection.score, model->getLabel(detection.id))) {
                // Log detection
                logger->logDetection(model->getLabel(detection.id), detection.score);
            }
        }
        
        for (const auto& detection : result2->detections) {
            // Trigger alert
            if (alertSystem->triggerAlert(detection.score, model->getLabel(detection.id))) {
                // Log detection
                logger->logDetection(model->getLabel(detection.id), detection.score);
            }
        }
        
        // Resize frames to same height if different
        if (frame1.rows != frame2.rows) {
            // Resize the smaller frame to match the height of the larger one
            if (frame1.rows < frame2.rows) {
                double scale = static_cast<double>(frame2.rows) / frame1.rows;
                int newWidth = static_cast<int>(frame1.cols * scale);
                cv::resize(frame1, frame1, cv::Size(newWidth, frame2.rows));
            } else {
                double scale = static_cast<double>(frame1.rows) / frame2.rows;
                int newWidth = static_cast<int>(frame2.cols * scale);
                cv::resize(frame2, frame2, cv::Size(newWidth, frame1.rows));
            }
        }
        
        // Combine frames side by side
        cv::Mat combinedFrame;
        cv::hconcat(frame1, frame2, combinedFrame);
        
        // Display if enabled and available
        if (!args.noDisplay) {
            try {
                cv::imshow("Dual Camera Person Detection", combinedFrame);
                
                // Exit on 'q' press
                int key = cv::waitKey(1);
                if (key == 'q' || key == 27) {  // 'q' or ESC
                    running = false;
                    break;
                }
            } catch (const cv::Exception& e) {
                std::cerr << "Display error: " << e.what() << std::endl;
                if (!args.saveVideo) {
                    std::cerr << "No display available and video saving not enabled. Exiting." << std::endl;
                    running = false;
                    break;
                }
            }
        }
        
        // Save frame to video if enabled
        if (args.saveVideo && videoWriterInitialized) {
            videoWriter.write(combinedFrame);
        }
        
        // Log performance occasionally
        auto currentTime = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastLogTime).count() >= 10) {
            // Log performance for both cameras
            logger->logPerformance(result1->fps, result1->processingTime);
            logger->logPerformance(result2->fps, result2->processingTime);
            
            // Print TPU utilization (if available)
            if (model->isUsingTPU()) {
                std::cout << "TPU processing two camera feeds at " << result1->fps << " FPS and " 
                          << result2->fps << " FPS" << std::endl;
            }
            
            lastLogTime = currentTime;
        }
    }
    
    // Stop camera processors
    camera1->stop();
    camera2->stop();
    
    // Clean up
    if (videoWriterInitialized) {
        videoWriter.release();
    }
    
    if (!args.noDisplay) {
        cv::destroyAllWindows();
    }
}

bool DualCameraDetector::checkDisplayAvailability() {
    // First try with default backend
    try {
        cv::namedWindow("Test", cv::WINDOW_NORMAL);
        cv::destroyWindow("Test");
        std::cout << "Display is available with default backend!" << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Default backend failed: " << e.what() << std::endl;
        
        // Try with GTK backend
        try {
            putenv(const_cast<char*>("OPENCV_VIDEOIO_PRIORITY_BACKEND=0"));  // Force GTK
            cv::namedWindow("Test", cv::WINDOW_NORMAL);
            cv::destroyWindow("Test");
            std::cout << "Display is available with GTK backend!" << std::endl;
            return true;
        } catch (const cv::Exception& e) {
            std::cerr << "GTK backend also failed: " << e.what() << std::endl;
            return false;
        }
    }
}

CommandLineArgs DualCameraDetector::parseArgs(int argc, char** argv) {
    CommandLineArgs args;
    
    // Define long options
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"labels", required_argument, 0, 'l'},
        {"threshold", required_argument, 0, 't'},
        {"camera1", required_argument, 0, '1'},
        {"camera2", required_argument, 0, '2'},
        {"output-dir", required_argument, 0, 'o'},
        {"person-class-id", required_argument, 0, 'p'},
        {"no-display", no_argument, 0, 'n'},
        {"save-video", no_argument, 0, 's'},
        {"force-cpu", no_argument, 0, 'c'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    // Parse options
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "m:l:t:1:2:o:p:nsch", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm':
                args.modelPath = optarg;
                break;
            case 'l':
                args.labelsPath = optarg;
                break;
            case 't':
                args.threshold = std::stof(optarg);
                break;
            case '1':
                args.camera1 = optarg;
                break;
            case '2':
                args.camera2 = optarg;
                break;
            case 'o':
                args.outputDir = optarg;
                break;
            case 'p':
                args.personClassId = std::stoi(optarg);
                break;
            case 'n':
                args.noDisplay = true;
                break;
            case 's':
                args.saveVideo = true;
                break;
            case 'c':
                args.forceCPU = true;
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  -m, --model PATH          Path to the detection model" << std::endl;
                std::cout << "  -l, --labels PATH         Path to the labels file" << std::endl;
                std::cout << "  -t, --threshold FLOAT     Detection threshold (default: 0.5)" << std::endl;
                std::cout << "  -1, --camera1 SOURCE      First camera source (default: 0)" << std::endl;
                std::cout << "  -2, --camera2 SOURCE      Second camera source (default: 1)" << std::endl;
                std::cout << "  -o, --output-dir DIR      Directory to save output files (default: output/)" << std::endl;
                std::cout << "  -p, --person-class-id ID  Class ID for 'person' (default: 0 for COCO)" << std::endl;
                std::cout << "  -n, --no-display          Disable display output" << std::endl;
                std::cout << "  -s, --save-video          Save processed video" << std::endl;
                std::cout << "  -c, --force-cpu           Force CPU mode (no TPU)" << std::endl;
                std::cout << "  -h, --help                Show this help message" << std::endl;
                exit(0);
                break;
            default:
                break;
        }
    }
    
    return args;
} 