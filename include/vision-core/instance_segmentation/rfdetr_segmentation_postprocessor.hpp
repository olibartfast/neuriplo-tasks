#pragma once

#include "vision-core/core/result_types.hpp"
#include <cstdint>
#include <variant>
#include <vector>
#include <opencv2/opencv.hpp>

namespace vision_core {

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief Instance segmentation result with mask data
 */
struct RfDetrInstanceSegmentation {
    cv::Rect bbox;
    float confidence;
    int class_id;
    cv::Mat mask;  // Binary mask for the instance
    
    RfDetrInstanceSegmentation(const cv::Rect& b, float conf, int cid, const cv::Mat& m)
        : bbox(b), confidence(conf), class_id(cid), mask(m) {}
};

/**
 * @brief RF-DETR Segmentation postprocessing algorithms
 * 
 * Handles RF-DETR models with instance segmentation outputs.
 * Outputs:
 * - Boxes: [batch, num_queries, 4]
 * - Labels: [batch, num_queries]
 * - Masks: [batch, num_queries, H, W]
 */
class RfDetrSegmentationPostprocessor {
public:
    /**
     * @brief Process RF-DETR segmentation output tensors
     * 
     * @param bbox_output Bounding box predictions
     * @param label_output Class label predictions
     * @param mask_output Mask predictions
     * @param bbox_shape Shape of bbox tensor
     * @param label_shape Shape of label tensor
     * @param mask_shape Shape of mask tensor
     * @param frame_size Original image size
     * @param network_width Network input width
     * @param network_height Network input height
     * @param confidence_threshold Minimum confidence threshold
     * @param mask_threshold Threshold for binary mask
     * @return Vector of instance segmentation results
     */
    [[nodiscard]] static std::vector<RfDetrInstanceSegmentation> postprocess(
        const TensorElement* bbox_output,
        const TensorElement* label_output,
        const TensorElement* mask_output,
        const std::vector<int64_t>& bbox_shape,
        const std::vector<int64_t>& label_shape,
        const std::vector<int64_t>& mask_shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold = 0.5f,
        float mask_threshold = 0.5f
    );
};

} // namespace vision_core
