#include "vision-core/object_detection/rtdetr_postprocessor.hpp"
#include <stdexcept>

namespace vision_core {

namespace {

float get_float(const TensorElement& elem) {
    return std::visit([](auto&& arg) -> float {
        return static_cast<float>(arg);
    }, elem);
}

} // anonymous namespace

std::vector<Detection> RtDetrPostprocessor::postprocess(
    const TensorElement* bbox_output,
    const TensorElement* score_output,
    const std::vector<int64_t>& bbox_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size,
    int network_width,
    int network_height,
    float confidence_threshold)
{
    if (bbox_shape.size() < 3 || score_shape.size() < 2) {
        throw std::invalid_argument("Invalid tensor shapes for RT-DETR");
    }

    const int num_detections = bbox_shape[1]; // e.g., 300
    const float scale_width = static_cast<float>(frame_size.width) / network_width;
    const float scale_height = static_cast<float>(frame_size.height) / network_height;

    std::vector<Detection> detections;
    detections.reserve(num_detections);

    for (int i = 0; i < num_detections; ++i) {
        const float score = get_float(score_output[i]);
        
        if (score < confidence_threshold) {
            continue;
        }

        // RT-DETR outputs boxes in [x1, y1, x2, y2] format (normalized 0-1)
        const int bbox_offset = i * 4;
        float x1 = get_float(bbox_output[bbox_offset + 0]);
        float y1 = get_float(bbox_output[bbox_offset + 1]);
        float x2 = get_float(bbox_output[bbox_offset + 2]);
        float y2 = get_float(bbox_output[bbox_offset + 3]);

        // Scale to frame size
        x1 *= scale_width;
        y1 *= scale_height;
        x2 *= scale_width;
        y2 *= scale_height;

        // Create bounding box
        cv::Rect bbox(
            cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
            cv::Point(static_cast<int>(x2), static_cast<int>(y2))
        );

        // Class ID (if available as separate output, otherwise -1)
        const int class_id = (bbox_shape.size() > 3) ? i : -1;

        detections.emplace_back(bbox, score, class_id);
    }

    return detections;
}

} // namespace vision_core
