#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace vision_core {

/**
 * @brief Common detection result structure
 * Used across all detection and segmentation tasks
 */
struct Detection {
    cv::Rect bbox;           // Bounding box
    float score;             // Confidence score
    int label;               // Class ID
    std::vector<cv::Point> mask;  // Optional: for instance segmentation
    
    Detection() : score(0.0f), label(-1) {}
};

/**
 * @brief Classification result
 */
struct ClassificationResult {
    int class_id;
    float confidence;
    std::string class_name;
};

} // namespace vision_core
