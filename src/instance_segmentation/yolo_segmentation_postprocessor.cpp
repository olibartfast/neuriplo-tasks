#include "vision-core/instance_segmentation/yolo_segmentation_postprocessor.hpp"
#include "vision-core/core/bbox_processor.hpp"
#include <opencv2/dnn.hpp>
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

cv::Rect YoloSegmentationPostprocessor::calculate_padding(
    int network_width,
    int network_height,
    const cv::Size& frame_size) noexcept
{
    const float ratio_w = static_cast<float>(network_width) / frame_size.width;
    const float ratio_h = static_cast<float>(network_height) / frame_size.height;
    
    if (ratio_h > ratio_w) {
        const int h = static_cast<int>(ratio_w * frame_size.height);
        const int pad = (network_height - h) / 2;
        return cv::Rect(0, pad, network_width, h);
    } else {
        const int w = static_cast<int>(ratio_h * frame_size.width);
        const int pad = (network_width - w) / 2;
        return cv::Rect(pad, 0, w, network_height);
    }
}

cv::Mat YoloSegmentationPostprocessor::decode_mask(
    const std::vector<float>& mask_coefficients,
    const cv::Mat& mask_prototypes,
    const cv::Rect& bbox,
    const cv::Size& frame_size,
    const cv::Size& mask_size,
    float mask_threshold)
{
    // Matrix multiplication: coefficients @ prototypes
    cv::Mat coeffs_mat(1, mask_coefficients.size(), CV_32F, 
                       const_cast<float*>(mask_coefficients.data()));
    cv::Mat mask_result = coeffs_mat * mask_prototypes;
    cv::Mat mask = mask_result.reshape(1, mask_size.height);

    // Apply sigmoid activation
    cv::exp(-mask, mask);
    mask = 1.0 / (1.0 + mask);

    // Resize to frame size
    cv::Mat mask_resized;
    cv::resize(mask, mask_resized, frame_size, 0, 0, cv::INTER_LINEAR);

    // Crop to bbox
    cv::Rect safe_bbox = bbox & cv::Rect(0, 0, frame_size.width, frame_size.height);
    if (safe_bbox.width <= 0 || safe_bbox.height <= 0) {
        return cv::Mat();
    }

    cv::Mat mask_cropped = mask_resized(safe_bbox);

    // Threshold
    cv::Mat mask_binary;
    cv::threshold(mask_cropped, mask_binary, mask_threshold, 1.0, cv::THRESH_BINARY);
    mask_binary.convertTo(mask_binary, CV_8U, 255.0);

    return mask_binary;
}

std::vector<InstanceSegmentation> YoloSegmentationPostprocessor::postprocess(
    const TensorElement* detection_output,
    const TensorElement* mask_output,
    const std::vector<int64_t>& detection_shape,
    const std::vector<int64_t>& mask_shape,
    const cv::Size& frame_size,
    int network_width,
    int network_height,
    float confidence_threshold,
    float nms_threshold,
    float mask_threshold)
{
    if (detection_shape.size() < 3 || mask_shape.size() < 4) {
        throw std::invalid_argument("Invalid tensor shapes for YOLO segmentation");
    }

    const bool is_v567_format = detection_shape[2] < detection_shape[1];
    const int num_mask_coeffs = mask_shape[1]; // e.g., 32
    const int mask_height = mask_shape[2];
    const int mask_width = mask_shape[3];

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> mask_coefficients;

    // Process detection output based on format
    if (is_v567_format) {
        // YOLOv5-seg format: [1, num_proposals, 4+1+classes+mask_coeffs]
        const int num_proposals = detection_shape[1];
        const int num_classes = detection_shape[2] - 5 - num_mask_coeffs;
        const int stride = detection_shape[2];

        for (int i = 0; i < num_proposals; ++i) {
            const TensorElement* proposal = detection_output + (i * stride);
            
            const float objectness = get_float(proposal[4]);
            if (objectness < confidence_threshold) continue;

            // Find best class
            float max_score = 0.0f;
            int best_class = 0;
            for (int c = 0; c < num_classes; ++c) {
                const float score = get_float(proposal[5 + c]);
                if (score > max_score) {
                    max_score = score;
                    best_class = c;
                }
            }

            const float confidence = objectness * max_score;
            if (confidence < confidence_threshold) continue;

            // Extract bbox
            std::vector<float> bbox_coords = {
                get_float(proposal[0]), get_float(proposal[1]),
                get_float(proposal[2]), get_float(proposal[3])
            };

            cv::Rect bbox = BBoxProcessor::calculate_bounding_box(
                frame_size, bbox_coords, network_width, network_height
            );

            // Extract mask coefficients
            std::vector<float> coeffs(num_mask_coeffs);
            for (int m = 0; m < num_mask_coeffs; ++m) {
                coeffs[m] = get_float(proposal[5 + num_classes + m]);
            }

            boxes.push_back(bbox);
            scores.push_back(confidence);
            class_ids.push_back(best_class);
            mask_coefficients.push_back(std::move(coeffs));
        }
    } else {
        // YOLOv8+ format: [1, 4+classes+mask_coeffs, num_proposals]
        const int num_channels = detection_shape[1];
        const int num_proposals = detection_shape[2];
        const int num_classes = num_channels - 4 - num_mask_coeffs;

        // Transpose
        std::vector<std::vector<float>> transposed(num_proposals, std::vector<float>(num_channels));
        for (int c = 0; c < num_channels; ++c) {
            for (int p = 0; p < num_proposals; ++p) {
                transposed[p][c] = get_float(detection_output[c * num_proposals + p]);
            }
        }

        for (int i = 0; i < num_proposals; ++i) {
            const auto& proposal = transposed[i];
            
            // Find best class
            const float* class_scores = proposal.data() + 4;
            const auto max_it = std::max_element(class_scores, class_scores + num_classes);
            const float max_score = *max_it;
            
            if (max_score < confidence_threshold) continue;

            const int best_class = std::distance(class_scores, max_it);

            // Extract bbox
            std::vector<float> bbox_coords = {proposal[0], proposal[1], proposal[2], proposal[3]};
            cv::Rect bbox = BBoxProcessor::calculate_bounding_box(
                frame_size, bbox_coords, network_width, network_height
            );

            // Extract mask coefficients
            std::vector<float> coeffs(num_mask_coeffs);
            for (int m = 0; m < num_mask_coeffs; ++m) {
                coeffs[m] = proposal[4 + num_classes + m];
            }

            boxes.push_back(bbox);
            scores.push_back(max_score);
            class_ids.push_back(best_class);
            mask_coefficients.push_back(std::move(coeffs));
        }
    }

    // Apply NMS
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, nms_indices);

    // Build mask prototypes matrix
    std::vector<float> proto_data(num_mask_coeffs * mask_height * mask_width);
    for (size_t i = 0; i < proto_data.size(); ++i) {
        proto_data[i] = get_float(mask_output[i]);
    }
    cv::Mat prototypes(num_mask_coeffs, mask_height * mask_width, CV_32F, proto_data.data());

    // Calculate padding for mask processing (reserved for future use)
    [[maybe_unused]] cv::Rect padding = calculate_padding(network_width, network_height, frame_size);

    // Build final results with masks
    std::vector<InstanceSegmentation> results;
    results.reserve(nms_indices.size());

    for (int idx : nms_indices) {
        InstanceSegmentation seg(boxes[idx], scores[idx], class_ids[idx]);

        // Decode mask
        cv::Mat mask = decode_mask(
            mask_coefficients[idx],
            prototypes,
            boxes[idx],
            frame_size,
            cv::Size(mask_width, mask_height),
            mask_threshold
        );

        if (!mask.empty()) {
            seg.mask_width = mask.cols;
            seg.mask_height = mask.rows;
            seg.mask_data.assign(mask.data, mask.data + mask.total());
        }

        results.push_back(std::move(seg));
    }

    return results;
}

} // namespace vision_core
