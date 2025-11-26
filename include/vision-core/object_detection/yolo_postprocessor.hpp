#pragma once

#include "vision-core/core/result_types.hpp"
#include <cstdint>
#include <variant>
#include <vector>

namespace vision_core {

/// Generic tensor element type (supports common inference engine outputs)
using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief Unified YOLO family postprocessing algorithms
 * 
 * Supports ALL YOLO versions with automatic format detection:
 * - YOLOv5, v6, v7: [batch, num_proposals, 4+1+classes] format
 * - YOLOv8, v9, v11, v12: [batch, 4+classes, num_proposals] format  
 * - YOLOv10: [batch, num_detections, 6] format (end-to-end, no NMS)
 * - YOLO-NAS: Two tensors - boxes [batch, num_queries, 4] + scores [batch, num_queries, classes]
 * 
 * Handles confidence thresholding, NMS, and coordinate transformation.
 */
class YoloPostprocessor {
public:
    /**
     * @brief Process YOLO output tensor(s) and return filtered detections
     * 
     * Automatically detects YOLO variant based on tensor count and shapes:
     * - Single tensor: Standard YOLO (v5-v9, v11-v12) or YOLOv10
     * - Two tensors: YOLO-NAS (boxes + scores)
     * 
     * @param output Raw output data from inference engine (primary tensor)
     * @param shape Output tensor shape 
     * @param frame_size Original image dimensions
     * @param network_width Network input width
     * @param network_height Network input height
     * @param confidence_threshold Minimum confidence for detection [0.0, 1.0]
     * @param nms_threshold IoU threshold for NMS [0.0, 1.0] (ignored for YOLOv10)
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

    /**
     * @brief Process YOLO-NAS output tensors (two separate tensors)
     * 
     * Handles YOLO-NAS models that output boxes and scores separately.
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
    [[nodiscard]] static std::vector<Detection> postprocess_nas(
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

    /**
     * @brief Process YOLOv10 format: [batch, num_detections, 6]
     * 
     * Layout: [x1, y1, x2, y2, confidence, class_id] - end-to-end, no NMS needed
     */
    [[nodiscard]] static std::vector<Detection> process_v10_format(
        const TensorElement* output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold
    );

    /**
     * @brief Process YOLO-NAS format: separate boxes and scores tensors
     * 
     * Boxes: [batch, num_queries, 4] in [x1, y1, x2, y2] format
     * Scores: [batch, num_queries, num_classes]
     */
    [[nodiscard]] static std::vector<Detection> process_nas_format(
        const TensorElement* bbox_output,
        const TensorElement* score_output,
        const std::vector<int64_t>& bbox_shape,
        const std::vector<int64_t>& score_shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold,
        float nms_threshold
    );
};

} // namespace vision_core
