#include "vision-core/object_detection/dfine_postprocessor.hpp"
#include "vision-core/core/bbox_processor.hpp"
#include <algorithm>
#include <stdexcept>

namespace vision_core {

namespace {

float get_float(const TensorElement& elem) {
    return std::visit([](auto&& arg) -> float { 
        return static_cast<float>(arg); 
    }, elem);
}

} // anonymous namespace

std::vector<Detection> DFinePostprocessor::postprocess(
    const TensorElement* bbox_output,
    const TensorElement* score_output,
    const std::vector<int64_t>& bbox_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size,
    int network_width,
    int network_height,
    float confidence_threshold)
{
    if (bbox_shape.size() < 3 || score_shape.size() < 3) {
        throw std::invalid_argument("D-FINE output shapes must have at least 3 dimensions");
    }

    const int64_t num_queries = bbox_shape[1];
    const int64_t num_classes = score_shape[2];

    if (score_shape[1] != num_queries) {
        throw std::invalid_argument("Mismatch in number of queries between bbox and score tensors");
    }

    std::vector<Detection> detections;
    detections.reserve(num_queries);

    for (int64_t i = 0; i < num_queries; ++i) {
        // Find best class score
        float max_score = 0.0f;
        int best_class_id = -1;

        for (int64_t c = 0; c < num_classes; ++c) {
            float score = get_float(score_output[i * num_classes + c]);
            if (score > max_score) {
                max_score = score;
                best_class_id = static_cast<int>(c);
            }
        }

        if (max_score >= confidence_threshold) {
            // Extract bbox: [cx, cy, w, h]
            float cx = get_float(bbox_output[i * 4 + 0]);
            float cy = get_float(bbox_output[i * 4 + 1]);
            float w  = get_float(bbox_output[i * 4 + 2]);
            float h  = get_float(bbox_output[i * 4 + 3]);

            // Heuristic check for normalized coordinates
            // If all values are <= 1.0, assume normalized and scale to network size
            if (cx <= 1.0f && cy <= 1.0f && w <= 1.0f && h <= 1.0f && w > 0 && h > 0) {
                cx *= network_width;
                cy *= network_height;
                w *= network_width;
                h *= network_height;
            }

            std::vector<float> bbox = {cx, cy, w, h};
            cv::Rect rect = BBoxProcessor::calculate_bounding_box(
                frame_size, bbox, network_width, network_height
            );

            detections.push_back({rect, max_score, best_class_id});
        }
    }

    return detections;
}

} // namespace vision_core
