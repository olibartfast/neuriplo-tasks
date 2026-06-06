#pragma once

#include "vision-core/core/bounding_box.hpp"
#include "vision-core/core/image_matrix.hpp"

#include <cstdint>
#include <string>
#include <utility>
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
    BoundingBox bbox; ///< Bounding box

    Detection() = default;
    Detection(const BoundingBox& box, float conf, int cls) : Classification(static_cast<float>(cls), conf), bbox(box) {}
};

/**
 * @brief Open-vocabulary detection result structure
 */
struct OpenVocabDetection {
    BoundingBox bbox;     ///< Bounding box
    float score{0.0f};    ///< Detection score
    int prompt_index{-1}; ///< Index into the runtime prompt list
    std::string label;    ///< Resolved prompt label

    OpenVocabDetection() = default;
    OpenVocabDetection(const BoundingBox& box, float det_score, int index, std::string prompt_label)
        : bbox(box), score(det_score), prompt_index(index), label(std::move(prompt_label)) {}
};

/**
 * @brief Instance segmentation result structure (extends Detection)
 */
struct InstanceSegmentation : public Detection {
    std::vector<uint8_t> mask_data; ///< Mask data as a vector
    int mask_height{0};             ///< Mask height
    int mask_width{0};              ///< Mask width
    ImageMatrix mask;               ///< Binary or soft mask

    InstanceSegmentation() = default;

    InstanceSegmentation(const BoundingBox& box, float conf, int cls) : Detection(box, conf, cls) {}
};

/**
 * @brief Optical flow result structure
 */
struct OpticalFlow {
    ImageMatrix flow;             ///< Colored visualization
    ImageMatrix raw_flow;         ///< Raw flow field (CV_32FC2)
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
    BoundingBox bbox; ///< Bounding box (populated by detection-based models e.g. YOLO pose)
    std::vector<Keypoint> keypoints;
    float score{0.0f};

    PoseEstimation() = default;
};

/**
 * @brief Depth estimation result structure
 */
struct DepthEstimation {
    ImageMatrix depth;            ///< Raw depth map (CV_32FC1)
    ImageMatrix normalized_depth; ///< Normalized depth map in [0,1] (CV_32FC1)
    float min_depth{0.0f};        ///< Minimum depth value in raw map
    float max_depth{0.0f};        ///< Maximum depth value in raw map

    DepthEstimation() = default;
};

/**
 * @brief Single 3D Gaussian primitive
 *
 * Represents one Gaussian in a 3D Gaussian Splatting scene.
 * Fields follow the standard PLY layout used by the original
 * Kerbl et al. (SIGGRAPH 2023) implementation and by models such
 * as LGM and GRM.
 */
struct Gaussian3D {
    // Position
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};

    // Opacity (pre-sigmoid, raw logit as output by the network)
    float opacity{0.0f};

    // Scale (log-scale, 3 values)
    float scale_x{0.0f};
    float scale_y{0.0f};
    float scale_z{0.0f};

    // Rotation quaternion (w, x, y, z)
    float rot_w{1.0f};
    float rot_x{0.0f};
    float rot_y{0.0f};
    float rot_z{0.0f};

    // Spherical-harmonic DC colour (degree-0, RGB channels)
    float sh_r{0.0f};
    float sh_g{0.0f};
    float sh_b{0.0f};

    Gaussian3D() = default;
};

/**
 * @brief Gaussian Splatting result structure
 *
 * Output of a feed-forward Gaussian Splatting model (e.g. LGM, GRM).
 * Contains per-primitive Gaussian parameters ready for rendering or
 * serialisation to .ply format.
 */
struct GaussianSplatting {
    std::vector<Gaussian3D> gaussians; ///< Predicted 3D Gaussians
    int num_gaussians{0};              ///< Number of Gaussians

    GaussianSplatting() = default;
};

/**
 * @brief Image understanding / text generation result
 */
struct ImageUnderstanding {
    std::string text; ///< Generated text response

    ImageUnderstanding() = default;
    explicit ImageUnderstanding(std::string t) : text(std::move(t)) {}
};

/**
 * @brief Result variant type to hold any task result
 */
using Result =
    std::variant<Classification, Detection, OpenVocabDetection, InstanceSegmentation, OpticalFlow, VideoClassification,
                 PoseEstimation, DepthEstimation, GaussianSplatting, ImageUnderstanding>;

/**
 * @brief Optional visitor helper for Result (forwards to std::visit).
 *
 * Consumers may use std::visit or std::get directly; this wrapper preserves
 * a single include point if the Result alternative set grows.
 */
template <typename Visitor> decltype(auto) visitResult(Result& result, Visitor&& visitor) {
    return std::visit(std::forward<Visitor>(visitor), result);
}

template <typename Visitor> decltype(auto) visitResult(const Result& result, Visitor&& visitor) {
    return std::visit(std::forward<Visitor>(visitor), result);
}

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
    OpenVocabDetection,
    GaussianSplatting,
    ImageUnderstanding
};

} // namespace vision_core
