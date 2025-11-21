#pragma once

#include "vision-core/core/detection.hpp"
#include <cstdint>
#include <variant>
#include <vector>

namespace vision_core {

/// Generic tensor element type (supports common inference engine outputs)
using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief YOLO family postprocessing algorithms
 * 
 * Supports multiple YOLO versions with automatic format detection:
 * - YOLOv5, v6, v7: [batch, num_proposals, 4+1+classes] format
 * - YOLOv8, v9, v10, v11, v12: [batch, 4+classes, num_proposals] format
 * 
 * Handles confidence thresholding, NMS, and coordinate transformation.
 */
class YoloPostprocessor {
public:
    /**
     * @brief Process YOLO output tensor and return filtered detections
     * 
     * Automatically detects YOLO version based on tensor shape and applies
     * appropriate postprocessing pipeline including NMS.
     * 
     * @param output Raw output data from inference engine
     * @param shape Output tensor shape (e.g., [1, 25200, 85] or [1, 84, 8400])
     * @param frame_size Original image dimensions
     * @param network_width Network input width
     * @param network_height Network input height
     * @param confidence_threshold Minimum confidence for detection [0.0, 1.0]
     * @param nms_threshold IoU threshold for NMS [0.0, 1.0]
     * @return Vector of detections after filtering and NMS
     */
    [[nodiscard]] static std::vector<Detection> postprocess(
        const TensorElement* output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold = 0.25f,
        float nms_threshold = 0.45f
    );

private:
    /**
     * @brief Process YOLOv5/v6/v7 format: [1, num_proposals, 4+1+classes]
     * 
     * Layout: [x_center, y_center, width, height, objectness, class_0, ..., class_n]
     */
    [[nodiscard]] static std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>>
    process_v567_format(
        const TensorElement* output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold
    );

    /**
     * @brief Process YOLOv8+ format: [1, 4+classes, num_proposals]
     * 
     * Layout: Transposed compared to v5-v7, no explicit objectness score
     */
    [[nodiscard]] static std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>>
    process_ultralytics_format(
        const TensorElement* output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold
    );
};

} // namespace vision_core
