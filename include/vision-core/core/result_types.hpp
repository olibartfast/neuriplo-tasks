#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <variant>
#include <vector>

namespace vision_core {

/**
 * @brief Classification result structure
 */
struct Classification {
    float class_id{-1.0f};        ///< Predicted class ID
    float class_confidence{0.0f}; ///< Confidence score [0.0, 1.0]

    Classification() = default;
    Classification(float id, float conf) : class_id(id), class_confidence(conf) {}
};

/**
 * @brief Detection result structure (extends Classification)
 */
struct Detection : public Classification {
    cv::Rect bbox; ///< Bounding box

    Detection() = default;
    Detection(const cv::Rect& box, float conf, int cls) : Classification(static_cast<float>(cls), conf), bbox(box) {}
};

/**
 * @brief Open-vocabulary detection result structure
 */
struct OpenVocabDetection {
    cv::Rect bbox;        ///< Bounding box
    float score{0.0f};    ///< Detection score
    int prompt_index{-1}; ///< Index into the runtime prompt list
    std::string label;    ///< Resolved prompt label

    OpenVocabDetection() = default;
    OpenVocabDetection(const cv::Rect& box, float det_score, int index, std::string prompt_label)
        : bbox(box), score(det_score), prompt_index(index), label(std::move(prompt_label)) {}
};

/**
 * @brief Instance segmentation result structure (extends Detection)
 */
struct InstanceSegmentation : public Detection {
    std::vector<uint8_t> mask_data; ///< Mask data as a vector
    int mask_height{0};             ///< Mask height
    int mask_width{0};              ///< Mask width
    cv::Mat mask;                   ///< Binary or soft mask

    InstanceSegmentation() = default;

    InstanceSegmentation(const cv::Rect& box, float conf, int cls) : Detection(box, conf, cls) {}
};

/**
 * @brief Optical flow result structure
 */
struct OpticalFlow {
    cv::Mat flow;                 ///< Colored visualization
    cv::Mat raw_flow;             ///< Raw flow field (CV_32FC2)
    float max_displacement{0.0f}; ///< Maximum flow magnitude

    OpticalFlow() = default;
};

/**
 * @brief Video classification result structure (extends Classification)
 */
struct VideoClassification : public Classification {
    std::string action_label;        ///< Human-readable action name
    std::vector<float> frame_scores; ///< Confidence scores per frame

    VideoClassification() = default;
};

/**
 * @brief Keypoint structure
 */
struct Keypoint {
    float x{0.0f};
    float y{0.0f};
    float confidence{0.0f};
};

/**
 * @brief Pose estimation result structure
 */
struct PoseEstimation {
    cv::Rect bbox; ///< Bounding box (populated by detection-based models e.g. YOLO pose)
    std::vector<Keypoint> keypoints;
    float score{0.0f};

    PoseEstimation() = default;
};

/**
 * @brief Depth estimation result structure
 */
struct DepthEstimation {
    cv::Mat depth;            ///< Raw depth map (CV_32FC1)
    cv::Mat normalized_depth; ///< Normalized depth map in [0,1] (CV_32FC1)
    float min_depth{0.0f};    ///< Minimum depth value in raw map
    float max_depth{0.0f};    ///< Maximum depth value in raw map

    DepthEstimation() = default;
};

/**
 * @brief Result variant type to hold any task result
 */
using Result = std::variant<Classification, Detection, OpenVocabDetection, InstanceSegmentation, OpticalFlow,
                            VideoClassification, PoseEstimation, DepthEstimation>;

/**
 * @brief Task type enumeration
 */
enum class TaskType : uint8_t {
    OpticalFlow,
    Classification,
    Detection,
    InstanceSegmentation,
    VideoClassification,
    PoseEstimation,
    DepthEstimation,
    OpenVocabDetection
};

} // namespace vision_core
