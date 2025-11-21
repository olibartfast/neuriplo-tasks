#pragma once

#include "vision-core/core/detection.hpp"
#include <cstdint>
#include <variant>
#include <vector>

namespace vision_core {

/// Generic tensor element type (supports common inference engine outputs)
using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief RT-DETR family postprocessing algorithms
 * 
 * Supports RT-DETR, RT-DETRv2, and related transformer-based detectors.
 * These models output pre-filtered detections without requiring NMS.
 */
class RtDetrPostprocessor {
public:
    /**
     * @brief Process RT-DETR output tensors
     * 
     * RT-DETR models typically output two tensors:
     * - Bounding boxes: [batch, num_queries, 4] in [x_center, y_center, w, h] format
     * - Scores: [batch, num_queries, num_classes]
     * 
     * @param bbox_output Bounding box predictions
     * @param score_output Class score predictions
     * @param bbox_shape Shape of bbox tensor
     * @param score_shape Shape of score tensor
     * @param frame_size Original image size
     * @param network_width Network input width
     * @param network_height Network input height
     * @param confidence_threshold Minimum confidence threshold
     * @return Vector of filtered detections
     */
    [[nodiscard]] static std::vector<Detection> postprocess(
        const TensorElement* bbox_output,
        const TensorElement* score_output,
        const std::vector<int64_t>& bbox_shape,
        const std::vector<int64_t>& score_shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold = 0.5f
    );
};

} // namespace vision_core
