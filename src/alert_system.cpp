#include "alert_system.h"
#include <iostream> // For potential future alert logging

AlertSystem::AlertSystem(float alertThreshold)
    : threshold(alertThreshold) {
    // Constructor logic (if any)
}

bool AlertSystem::triggerAlert(float score, const std::string& label) {
    // Stub implementation: Basic threshold check
    // You might add more complex logic here later (e.g., debouncing, specific label checks)
    if (score >= threshold) {
        // std::cout << "[Alert] Triggered for " << label << " (Score: " << score << ")" << std::endl;
        return true; // Trigger alert if score meets threshold
    }
    return false; // No alert
} 