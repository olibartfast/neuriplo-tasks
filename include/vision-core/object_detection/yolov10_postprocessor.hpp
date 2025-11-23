#pragma once

#include "vision-core/core/result_types.hpp"
#include <cstdint>
#include <variant>
#include <vector>

namespace vision_core {

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief YOLOv10 postprocessing algorithms
 * 
 * YOLOv10 is an end-to-end detector that doesn't require NMS.
 * Output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
 */
class Yolov10Postprocessor {
public:
    /**
     * @brief Process YOLOv10 output tensor
     * 
     * @param output Raw output data from inference engine
     * @param shape Output tensor shape [batch, num_detections, 6]
     * @param frame_size Original image size
     * @param network_width Network input width
     * @param network_height Network input height
     * @param confidence_threshold Minimum confidence threshold
     * @return Vector of filtered detections (NMS not needed)
     */
    [[nodiscard]] static std::vector<Detection> postprocess(
        const TensorElement* output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold = 0.5f
    );
};

} // namespace vision_core
