#pragma once

#include "vision-core/core/result_types.hpp"
#include <cstdint>
#include <variant>
#include <vector>

namespace vision_core {

/// Generic tensor element type
using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief YOLO instance segmentation postprocessor
 * 
 * Supports YOLOv5-seg, YOLOv8-seg, YOLO11-seg with mask processing.
 * Handles both anchor-based (v5) and anchor-free (v8+) variants.
 */
class YoloSegmentationPostprocessor {
public:
    /**
     * @brief Process YOLO segmentation output
     * 
     * @param detection_output Detection tensor (boxes + class + mask coefficients)
     * @param mask_output Mask prototype tensor
     * @param detection_shape Detection tensor shape
     * @param mask_shape Mask tensor shape [batch, num_protos, height, width]
     * @param frame_size Original image size
     * @param network_width Network input width
     * @param network_height Network input height
     * @param confidence_threshold Confidence threshold
     * @param nms_threshold NMS IoU threshold
     * @param mask_threshold Mask binarization threshold
     * @return Vector of instance segmentation results
     */
    [[nodiscard]] static std::vector<InstanceSegmentation> postprocess(
        const TensorElement* detection_output,
        const TensorElement* mask_output,
        const std::vector<int64_t>& detection_shape,
        const std::vector<int64_t>& mask_shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold = 0.25f,
        float nms_threshold = 0.45f,
        float mask_threshold = 0.5f
    );

private:
    /**
     * @brief Calculate letterbox padding for segmentation
     */
    [[nodiscard]] static cv::Rect calculate_padding(
        int network_width,
        int network_height,
        const cv::Size& frame_size
    ) noexcept;

    /**
     * @brief Decode mask from coefficients and prototypes
     */
    [[nodiscard]] static cv::Mat decode_mask(
        const std::vector<float>& mask_coefficients,
        const cv::Mat& mask_prototypes,
        const cv::Rect& bbox,
        const cv::Size& frame_size,
        const cv::Size& mask_size,
        float mask_threshold
    );
};

} // namespace vision_core
