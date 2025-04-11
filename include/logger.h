#ifndef LOGGER_H
#define LOGGER_H

#include <string>

class Logger {
public:
    Logger() = default;
    ~Logger() = default;

    void logDetection(const std::string& label, float score);
    void logPerformance(double fps, double processingTime);
};

#endif // LOGGER_H 