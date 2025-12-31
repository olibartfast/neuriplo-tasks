#include "vision-core/instance_segmentation/yolo_segmentation_postprocessor.hpp"
#include <stdexcept>

namespace vision_core {

YoloSegmentationPostprocessor::YoloSegmentationPostprocessor(float confidence_threshold, float nms_threshold, float mask_threshold)
    : confidence_threshold_(confidence_threshold)
    , nms_threshold_(nms_threshold)
    , mask_threshold_(mask_threshold) {}

std::vector<InstanceSegmentation> YoloSegmentationPostprocessor::postprocess(
    const std::vector<Tensor>& tensors,
    const cv::Size& frame_size) {
    
    if (tensors.size() < 2) {
        throw std::runtime_error("YOLO segmentation requires at least 2 output tensors");
    }

    // Output 0: Detections [1, 4+cls+masks, 8400]
    // Output 1: Mask Protos [1, 32, 160, 160]
    
    const auto& dets_tensor = tensors[0].data;
    const auto& protos_tensor = tensors[1].data;
    const auto& dets_shape = tensors[0].shape;
    const auto& protos_shape = tensors[1].shape;
    
    std::vector<InstanceSegmentation> segmentations;
    
    if (dets_shape.size() < 3 || protos_shape.size() < 4) return {};
    
    // int batch = dets_shape[0]; // Unused
    int channels = dets_shape[1];
    int anchors = dets_shape[2];
    int num_classes = channels - 4 - 32; // Assuming 32 mask coefficients
    
    if (num_classes <= 0) return {};
    
    const float* dets_data = std::get_if<float>(&dets_tensor[0]);
    const float* protos_data = std::get_if<float>(&protos_tensor[0]);
    
    if (!dets_data || !protos_data) return {};
    
    // Process detections
    for (int i = 0; i < anchors; ++i) {
        // Find max class score
        float max_score = 0.0f;
        int class_id = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            float score = dets_data[(c + 4) * anchors + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        if (max_score < confidence_threshold_) continue;
        
        // Extract box
        float cx = dets_data[0 * anchors + i];
        float cy = dets_data[1 * anchors + i];
        float w = dets_data[2 * anchors + i];
        float h = dets_data[3 * anchors + i];
        
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;
        
        // Extract mask coefficients
        std::vector<float> mask_coeffs;
        mask_coeffs.reserve(32);
        for (int m = 0; m < 32; ++m) {
            mask_coeffs.push_back(dets_data[(4 + num_classes + m) * anchors + i]);
        }
        
        InstanceSegmentation seg;
        seg.class_id = class_id;
        seg.class_confidence = max_score;
        seg.bbox = cv::Rect(static_cast<int>(x), static_cast<int>(y), 
                           static_cast<int>(w), static_cast<int>(h));
        
        // Mask generation would go here (matrix multiplication of coeffs * protos)
        // For now, we'll leave the mask empty or create a placeholder
        seg.mask = cv::Mat::zeros(frame_size, CV_8UC1); 
        
        segmentations.push_back(seg);
    }
    
    // Apply NMS (TODO: Implement NMS for segmentation)
    
    return segmentations;
}

float YoloSegmentationPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

} // namespace vision_core
