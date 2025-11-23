#pragma once

#include "vision-core/core/result_types.hpp"
#include <cstdint>
#include <variant>
#include <vector>

namespace vision_core {

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief RF-DETR postprocessing algorithms
 * 
 * Supports RF-DETR (Receptive Field DETR) models.
 * 
 * Outputs:
 * - Boxes: [batch, num_queries, 4] (cx, cy, w, h)
 * - Scores: [batch, num_queries, num_classes]
 */
class RfDetrPostprocessor {
public:
    /**
     * @brief Process RF-DETR output tensors
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
