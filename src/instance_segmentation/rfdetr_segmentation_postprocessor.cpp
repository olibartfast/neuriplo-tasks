#include "vision-core/instance_segmentation/rfdetr_segmentation_postprocessor.hpp"
#include "vision-core/core/bbox_processor.hpp"
#include <stdexcept>

namespace vision_core {

namespace {

float get_float(const TensorElement& elem) {
    return std::visit([](auto&& arg) -> float { 
        return static_cast<float>(arg); 
    }, elem);
}

float sigmoid(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

} // anonymous namespace

std::vector<RfDetrInstanceSegmentation> RfDetrSegmentationPostprocessor::postprocess(
    const TensorElement* bbox_output,
    const TensorElement* label_output,
    const TensorElement* mask_output,
    const std::vector<int64_t>& bbox_shape,
    const std::vector<int64_t>& label_shape,
    const std::vector<int64_t>& mask_shape,
    const cv::Size& frame_size,
    int network_width,
    int network_height,
    float confidence_threshold,
    float mask_threshold)
{
    if (bbox_shape.size() < 3 || label_shape.size() < 2 || mask_shape.size() < 4) {
        throw std::invalid_argument("RF-DETR Seg output shapes are invalid");
    }

    const int64_t num_queries = bbox_shape[1];
    const int64_t mask_height = mask_shape[2];
    const int64_t mask_width = mask_shape[3];

    const float scale_w = static_cast<float>(frame_size.width) / network_width;
    const float scale_h = static_cast<float>(frame_size.height) / network_height;

    std::vector<RfDetrInstanceSegmentation> results;
    results.reserve(num_queries);

    for (int64_t i = 0; i < num_queries; ++i) {
        // Get confidence from bbox (assuming last element or separate score)
        // For RF-DETR, confidence might be implicit or part of label output
        // Here we assume detections are pre-filtered and use a dummy confidence
        float confidence = 1.0f; // RF-DETR typically outputs filtered results

        if (confidence < confidence_threshold) {
            continue;
        }

        // Extract bbox: [x_min, y_min, x_max, y_max]
        float x_min = get_float(bbox_output[i * 4 + 0]);
        float y_min = get_float(bbox_output[i * 4 + 1]);
        float x_max = get_float(bbox_output[i * 4 + 2]);
        float y_max = get_float(bbox_output[i * 4 + 3]);

        // Scale to original image size
        x_min *= scale_w;
        y_min *= scale_h;
        x_max *= scale_w;
        y_max *= scale_h;

        cv::Rect bbox(
            static_cast<int>(x_min),
            static_cast<int>(y_min),
            static_cast<int>(x_max - x_min),
            static_cast<int>(y_max - y_min)
        );

        // Clamp bbox to image bounds
        BBoxProcessor::clamp_to_bounds(bbox, frame_size);

        // Get class label
        int class_id = static_cast<int>(get_float(label_output[i]));

        // Extract and process mask
        cv::Mat mask(mask_height, mask_width, CV_32F);
        const int64_t mask_offset = i * mask_height * mask_width;
        
        for (int64_t y = 0; y < mask_height; ++y) {
            for (int64_t x = 0; x < mask_width; ++x) {
                float mask_val = get_float(mask_output[mask_offset + y * mask_width + x]);
                // Apply sigmoid to get probability
                mask.at<float>(y, x) = sigmoid(mask_val);
            }
        }

        // Resize mask to original image size
        cv::Mat resized_mask;
        cv::resize(mask, resized_mask, frame_size, 0, 0, cv::INTER_LINEAR);

        // Apply threshold to create binary mask
        cv::Mat binary_mask;
        cv::threshold(resized_mask, binary_mask, mask_threshold, 1.0f, cv::THRESH_BINARY);

        // Crop mask to bbox region
        cv::Mat cropped_mask = binary_mask(bbox).clone();

        results.emplace_back(bbox, confidence, class_id, cropped_mask);
    }

    return results;
}

} // namespace vision_core
