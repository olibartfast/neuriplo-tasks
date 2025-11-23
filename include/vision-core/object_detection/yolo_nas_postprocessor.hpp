#pragma once

#include "vision-core/core/result_types.hpp"
#include <cstdint>
#include <variant>
#include <vector>

namespace vision_core {

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief YOLO-NAS postprocessing algorithms
 * 
 * Supports YOLO-NAS models which typically output separated boxes and scores.
 * Unlike standard YOLO, YOLO-NAS often exports with:
 * - Boxes: [batch, num_queries, 4] in [x1, y1, x2, y2] format
 * - Scores: [batch, num_queries, num_classes]
 * 
 * Requires NMS as it is an anchor-free detector but produces many candidates.
 */
class YoloNasPostprocessor {
public:
    /**
     * @brief Process YOLO-NAS output tensors
     * 
     * @param bbox_output Bounding box predictions [batch, num_queries, 4] (xyxy)
     * @param score_output Class score predictions [batch, num_queries, num_classes]
     * @param bbox_shape Shape of bbox tensor
     * @param score_shape Shape of score tensor
     * @param frame_size Original image size
     * @param network_width Network input width
     * @param network_height Network input height
     * @param confidence_threshold Minimum confidence threshold
     * @param nms_threshold IoU threshold for NMS
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
        float confidence_threshold = 0.25f,
        float nms_threshold = 0.45f
    );
};

} // namespace vision_core
