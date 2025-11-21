#include "vision-core/object_detection/yolov10_postprocessor.hpp"
#include "vision-core/core/bbox_processor.hpp"
#include <stdexcept>

namespace vision_core {

namespace {

float get_float(const TensorElement& elem) {
    return std::visit([](auto&& arg) -> float { 
        return static_cast<float>(arg); 
    }, elem);
}

int get_int(const TensorElement& elem) {
    return std::visit([](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
            return static_cast<int>(arg);
        }
        return static_cast<int>(arg);
    }, elem);
}

} // anonymous namespace

std::vector<Detection> Yolov10Postprocessor::postprocess(
    const TensorElement* output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size,
    int network_width,
    int network_height,
    float confidence_threshold)
{
    if (shape.size() < 3) {
        throw std::invalid_argument("YOLOv10 output shape must have at least 3 dimensions");
    }

    const int64_t num_detections = shape[1]; // e.g., 300
    const int64_t detection_size = shape[2]; // e.g., 6 [x1, y1, x2, y2, conf, class_id]

    if (detection_size != 6) {
        throw std::invalid_argument("YOLOv10 detection size must be 6 [x1, y1, x2, y2, conf, class_id]");
    }

    const float scale_width = static_cast<float>(frame_size.width) / network_width;
    const float scale_height = static_cast<float>(frame_size.height) / network_height;

    std::vector<Detection> detections;
    detections.reserve(num_detections);

    for (int64_t i = 0; i < num_detections; ++i) {
        const int64_t offset = i * detection_size;

        float confidence = get_float(output[offset + 4]);
        
        if (confidence < confidence_threshold) {
            continue;
        }

        // Extract bbox in xyxy format
        float x1 = get_float(output[offset + 0]) * scale_width;
        float y1 = get_float(output[offset + 1]) * scale_height;
        float x2 = get_float(output[offset + 2]) * scale_width;
        float y2 = get_float(output[offset + 3]) * scale_height;

        cv::Rect bbox(
            cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
            cv::Point(static_cast<int>(x2), static_cast<int>(y2))
        );

        int class_id = get_int(output[offset + 5]);

        detections.push_back({bbox, confidence, class_id});
    }

    return detections;
}

} // namespace vision_core
