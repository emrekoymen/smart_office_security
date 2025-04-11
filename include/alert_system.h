#ifndef ALERT_SYSTEM_H
#define ALERT_SYSTEM_H

#include <string>

class AlertSystem {
public:
    explicit AlertSystem(float alertThreshold);
    ~AlertSystem() = default;

    /**
     * @brief Checks if an alert should be triggered based on score and potentially label.
     * 
     * @param score The detection score.
     * @param label The detected object label.
     * @return true If an alert condition is met.
     * @return false Otherwise.
     */
    bool triggerAlert(float score, const std::string& label);

private:
    float threshold; // Alert threshold
};

#endif // ALERT_SYSTEM_H 