#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vision_core {

/**
 * @brief Common detection result structure
 * 
 * Used across all detection and segmentation tasks.
 * Follows standard C++ conventions with snake_case for members.
 */
struct Detection {
    cv::Rect bbox;                    ///< Bounding box
    float confidence{0.0f};            ///< Confidence score [0.0, 1.0]
    int class_id{-1};                  ///< Class ID (-1 for invalid)
    std::vector<cv::Point> mask;      ///< Optional: instance segmentation mask points
    
    Detection() = default;
    
    Detection(const cv::Rect& box, float conf, int cls)
        : bbox(box), confidence(conf), class_id(cls) {}
};

/**
 * @brief Classification result structure
 */
struct ClassificationResult {
    int class_id{-1};                  ///< Predicted class ID
    float confidence{0.0f};            ///< Confidence score [0.0, 1.0]
    std::string class_name;            ///< Human-readable class name
    
    ClassificationResult() = default;
    
    ClassificationResult(int id, float conf, std::string name = "")
        : class_id(id), confidence(conf), class_name(std::move(name)) {}
};

} // namespace vision_core
