#include "dual_camera_detector.h"
#include "single_camera_detector.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help                  Show this help message" << std::endl;
    std::cout << "  --single-camera         Use single camera mode (default: dual camera)" << std::endl;
    std::cout << "  --camera1=INDEX         First camera index (default: 0)" << std::endl;
    std::cout << "  --camera2=INDEX         Second camera index (default: 2)" << std::endl;
    std::cout << "  --source=SOURCE         Camera index or video file path for single camera mode (default: 0)" << std::endl;
    std::cout << "  --threshold=VALUE       Detection threshold (default: 0.5)" << std::endl;
    std::cout << "  --no-display            Disable display output" << std::endl;
    std::cout << "  --save-video            Save processed video" << std::endl;
    std::cout << "  --output-dir=DIR        Output directory (default: output)" << std::endl;
    std::cout << "  --force-cpu             Force CPU mode (no TPU)" << std::endl;
}

// Parse command line arguments
std::map<std::string, std::string> parseArgs(int argc, char* argv[]) {
    std::map<std::string, std::string> args;
    
    // Set defaults
    args["single-camera"] = "false";
    args["camera1"] = "0";
    args["camera2"] = "2";
    args["source"] = "0";
    args["threshold"] = "0.5";
    args["display"] = "true";
    args["save-video"] = "false";
    args["output-dir"] = "output";
    args["force-cpu"] = "false";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        } else if (arg == "--single-camera") {
            args["single-camera"] = "true";
        } else if (arg == "--no-display") {
            args["display"] = "false";
        } else if (arg == "--save-video") {
            args["save-video"] = "true";
        } else if (arg == "--force-cpu") {
            args["force-cpu"] = "true";
        } else if (arg.substr(0, 10) == "--camera1=") {
            args["camera1"] = arg.substr(10);
        } else if (arg.substr(0, 10) == "--camera2=") {
            args["camera2"] = arg.substr(10);
        } else if (arg.substr(0, 9) == "--source=") {
            args["source"] = arg.substr(9);
        } else if (arg.substr(0, 12) == "--threshold=") {
            args["threshold"] = arg.substr(12);
        } else if (arg.substr(0, 13) == "--output-dir=") {
            args["output-dir"] = arg.substr(13);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            exit(1);
        }
    }
    
    return args;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    auto args = parseArgs(argc, argv);
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(args["output-dir"]);
    
    // Convert string values to appropriate types
    bool singleCamera = args["single-camera"] == "true";
    bool displayOutput = args["display"] == "true";
    bool saveVideo = args["save-video"] == "true";
    bool forceCPU = args["force-cpu"] == "true";
    float threshold = std::stof(args["threshold"]);
    
    // Get the current directory for absolute paths
    std::filesystem::path currentPath = std::filesystem::current_path();
    
    // Path to model and labels
    std::string modelPath = (std::filesystem::path("../models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")).string();
    std::string labelsPath = (std::filesystem::path("../models/coco_labels.txt")).string();
    
    std::cout << "Model path: " << modelPath << std::endl;
    std::cout << "Labels path: " << labelsPath << std::endl;
    
    try {
        if (singleCamera) {
            std::cout << "Running in single camera mode" << std::endl;
            
            // Create and run the single camera detector
            SingleCameraDetector detector(
                args["source"],
                modelPath,
                labelsPath,
                threshold
            );
            
            detector.run(displayOutput, saveVideo, args["output-dir"], forceCPU);
        } else {
            std::cout << "Running in dual camera mode" << std::endl;
            
            // Create a CommandLineArgs struct from the parsed arguments
            CommandLineArgs cmdArgs;
            cmdArgs.modelPath = modelPath;
            cmdArgs.labelsPath = labelsPath;
            cmdArgs.threshold = threshold;
            cmdArgs.camera1 = args["camera1"];
            cmdArgs.camera2 = args["camera2"];
            cmdArgs.outputDir = args["output-dir"];
            cmdArgs.noDisplay = !displayOutput;
            cmdArgs.saveVideo = saveVideo;
            cmdArgs.forceCPU = forceCPU;
            
            // Create and run the dual camera detector
            DualCameraDetector detector(cmdArgs);
            
            detector.run();
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 