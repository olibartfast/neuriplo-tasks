#include "vision-core/object_detection/yolo_nas_postprocessor.hpp"
#include "vision-core/core/bbox_processor.hpp"
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <stdexcept>

namespace vision_core {

namespace {

// Helper to get float from TensorElement variant
float get_float(const TensorElement& elem) {
    return std::visit([](auto&& arg) -> float { 
        return static_cast<float>(arg); 
    }, elem);
}

} // anonymous namespace

std::vector<Detection> YoloNasPostprocessor::postprocess(
    const TensorElement* bbox_output,
    const TensorElement* score_output,
    const std::vector<int64_t>& bbox_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size,
    int network_width,
    int network_height,
    float confidence_threshold,
    float nms_threshold)
{
    if (bbox_shape.size() < 3 || score_shape.size() < 3) {
        throw std::invalid_argument("YOLO-NAS output shapes must have at least 3 dimensions");
    }

    const int64_t num_queries = bbox_shape[1];
    const int64_t num_classes = score_shape[2];

    if (score_shape[1] != num_queries) {
        throw std::invalid_argument("Mismatch in number of queries between bbox and score tensors");
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    boxes.reserve(num_queries);
    scores.reserve(num_queries);
    class_ids.reserve(num_queries);

    // Iterate over all queries
    for (int64_t i = 0; i < num_queries; ++i) {
        // Find best class score
        float max_score = 0.0f;
        int best_class_id = -1;

        for (int64_t c = 0; c < num_classes; ++c) {
            // score_output is [batch, num_queries, num_classes]
            // Index: batch*N*C + i*C + c
            // Assuming batch=1 for now (offset 0)
            float score = get_float(score_output[i * num_classes + c]);
            if (score > max_score) {
                max_score = score;
                best_class_id = static_cast<int>(c);
            }
        }

        if (max_score >= confidence_threshold) {
            // Extract bbox: [batch, num_queries, 4] (xyxy)
            // Index: batch*N*4 + i*4 + 0..3
            std::vector<float> raw_bbox(4);
            for (int j = 0; j < 4; ++j) {
                raw_bbox[j] = get_float(bbox_output[i * 4 + j]);
            }

            cv::Rect rect = BBoxProcessor::calculate_bounding_box_from_xyxy(
                frame_size, raw_bbox, network_width, network_height
            );

            boxes.push_back(rect);
            scores.push_back(max_score);
            class_ids.push_back(best_class_id);
        }
    }

    // Apply NMS
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, nms_indices);

    // Build final detections
    std::vector<Detection> detections;
    detections.reserve(nms_indices.size());

    for (int idx : nms_indices) {
        detections.push_back({
            boxes[idx],
            scores[idx],
            class_ids[idx]
        });
    }

    return detections;
}

} // namespace vision_core
