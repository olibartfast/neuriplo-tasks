#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <variant>

namespace vision_core {

/**
 * @brief Classification result structure
 */
struct Classification {
    float class_id{-1.0f};             ///< Predicted class ID
    float class_confidence{0.0f};      ///< Confidence score [0.0, 1.0]
    
    Classification() = default;
    Classification(float id, float conf) : class_id(id), class_confidence(conf) {}
};

/**
 * @brief Detection result structure (extends Classification)
 */
struct Detection : public Classification {
    cv::Rect bbox;                     ///< Bounding box
    
    Detection() = default;
    Detection(const cv::Rect& box, float conf, int cls)
        : Classification(static_cast<float>(cls), conf), bbox(box) {}
};

/**
 * @brief Instance segmentation result structure (extends Detection)
 */
struct InstanceSegmentation : public Detection {
    std::vector<uint8_t> mask_data;    ///< Mask data as a vector
    int mask_height{0};                ///< Mask height
    int mask_width{0};                 ///< Mask width
    cv::Mat mask;                      ///< Binary or soft mask
    
    InstanceSegmentation() = default;
    
    InstanceSegmentation(const cv::Rect& box, float conf, int cls)
        : Detection(box, conf, cls) {}
};

/**
 * @brief Optical flow result structure
 */
struct OpticalFlow {
    cv::Mat flow;                      ///< Colored visualization
    cv::Mat raw_flow;                  ///< Raw flow field (CV_32FC2)
    float max_displacement{0.0f};      ///< Maximum flow magnitude
    
    OpticalFlow() = default;
};

/**
 * @brief Video classification result structure (extends Classification)
 */
struct VideoClassification : public Classification {
    std::string action_label;          ///< Human-readable action name
    std::vector<float> frame_scores;   ///< Confidence scores per frame
    
    VideoClassification() = default;
};

/**
 * @brief Result variant type to hold any task result
 */
using Result = std::variant<Classification, Detection, InstanceSegmentation, OpticalFlow, VideoClassification>;

/**
 * @brief Task type enumeration
 */
enum class TaskType {
    OpticalFlow,
    Classification,
    Detection,
    InstanceSegmentation,
    VideoClassification
};

/**
 * @brief Legacy compatibility: ClassificationResult (maps to Classification)
 */
using ClassificationResult = Classification;

} // namespace vision_core
