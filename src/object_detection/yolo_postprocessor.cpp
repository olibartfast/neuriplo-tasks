#include "vision-core/object_detection/yolo_postprocessor.hpp"
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

std::vector<Detection> YoloPostprocessor::postprocess(
    const TensorElement* output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size,
    int network_width,
    int network_height,
    float confidence_threshold,
    float nms_threshold)
{
    if (shape.size() < 3) {
        throw std::invalid_argument("YOLO output shape must have at least 3 dimensions");
    }

    // Auto-detect format based on shape
    // YOLOv5-v7: [batch, num_proposals, 4+1+classes] e.g., [1, 25200, 85]
    // YOLOv8+:   [batch, 4+classes, num_proposals] e.g., [1, 84, 8400]
    const bool is_v567_format = shape[1] > shape[2];

    auto [boxes, scores, class_ids] = is_v567_format 
        ? process_v567_format(output, shape, frame_size, network_width, network_height, confidence_threshold)
        : process_ultralytics_format(output, shape, frame_size, network_width, network_height, confidence_threshold);

    // Apply NMS
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, nms_indices);

    // Build final detections
    std::vector<Detection> detections;
    detections.reserve(nms_indices.size());

    for (int idx : nms_indices) {
        detections.emplace_back(boxes[idx], scores[idx], class_ids[idx]);
    }

    return detections;
}

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>>
YoloPostprocessor::process_v567_format(
    const TensorElement* output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size,
    int network_width,
    int network_height,
    float confidence_threshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    const int num_proposals = shape[1];  // e.g., 25200
    const int num_classes = shape[2] - 5; // e.g., 85 - 5 = 80
    const int stride = shape[2];

    for (int i = 0; i < num_proposals; ++i) {
        const TensorElement* proposal = output + (i * stride);
        
        // Get objectness score
        const float objectness = get_float(proposal[4]);
        
        if (objectness < confidence_threshold) {
            continue;
        }

        // Find best class
        float max_class_score = 0.0f;
        int best_class_id = 0;
        
        for (int c = 0; c < num_classes; ++c) {
            const float class_score = get_float(proposal[5 + c]);
            if (class_score > max_class_score) {
                max_class_score = class_score;
                best_class_id = c;
            }
        }

        const float confidence = objectness * max_class_score;
        
        if (confidence < confidence_threshold) {
            continue;
        }

        // Extract bbox coordinates [x_center, y_center, width, height]
        std::vector<float> bbox_coords = {
            get_float(proposal[0]),
            get_float(proposal[1]),
            get_float(proposal[2]),
            get_float(proposal[3])
        };

        cv::Rect bbox = BBoxProcessor::calculate_bounding_box(
            frame_size, bbox_coords, network_width, network_height
        );

        boxes.push_back(bbox);
        scores.push_back(confidence);
        class_ids.push_back(best_class_id);
    }

    return {boxes, scores, class_ids};
}

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>>
YoloPostprocessor::process_ultralytics_format(
    const TensorElement* output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size,
    int network_width,
    int network_height,
    float confidence_threshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    const int num_channels = shape[1];     // e.g., 84
    const int num_proposals = shape[2];    // e.g., 8400
    const int num_classes = num_channels - 4; // 84 - 4 = 80

    // Transpose: [1, 84, 8400] -> [8400, 84]
    std::vector<std::vector<float>> transposed(num_proposals, std::vector<float>(num_channels));
    
    for (int c = 0; c < num_channels; ++c) {
        for (int p = 0; p < num_proposals; ++p) {
            transposed[p][c] = get_float(output[c * num_proposals + p]);
        }
    }

    // Process each proposal
    for (int i = 0; i < num_proposals; ++i) {
        const auto& proposal = transposed[i];
        
        // Find best class (no objectness score in v8+)
        const float* class_scores = proposal.data() + 4;
        const auto max_it = std::max_element(class_scores, class_scores + num_classes);
        const float max_score = *max_it;
        
        if (max_score < confidence_threshold) {
            continue;
        }

        const int best_class_id = std::distance(class_scores, max_it);

        // Extract bbox coordinates [x_center, y_center, width, height]
        std::vector<float> bbox_coords = {
            proposal[0], proposal[1], proposal[2], proposal[3]
        };

        cv::Rect bbox = BBoxProcessor::calculate_bounding_box(
            frame_size, bbox_coords, network_width, network_height
        );

        boxes.push_back(bbox);
        scores.push_back(max_score);
        class_ids.push_back(best_class_id);
    }

    return {boxes, scores, class_ids};
}

} // namespace vision_core
