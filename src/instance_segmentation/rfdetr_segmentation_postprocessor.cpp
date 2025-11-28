#include "vision-core/instance_segmentation/rfdetr_segmentation_postprocessor.hpp"
#include <stdexcept>

namespace vision_core {

RfDetrSegmentationPostprocessor::RfDetrSegmentationPostprocessor(float confidence_threshold, float mask_threshold)
    : confidence_threshold_(confidence_threshold)
    , mask_threshold_(mask_threshold) {}

std::vector<InstanceSegmentation> RfDetrSegmentationPostprocessor::postprocess(
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes,
    const cv::Size& frame_size) {
    
    if (infer_results.size() < 3 || infer_shapes.size() < 3) {
        throw std::runtime_error("RF-DETR segmentation requires at least 3 output tensors");
    }

    // Boxes, Scores, Masks
    const auto& boxes = infer_results[0];
    const auto& scores = infer_results[1];
    const auto& masks = infer_results[2];
    
    (void)boxes;
    (void)scores;
    (void)masks;
    
    std::vector<InstanceSegmentation> segmentations;
    
    // Similar logic to RT-DETR for boxes/scores, plus mask processing
    // Assuming standard RF-DETR output format
    
    // Placeholder logic for now as RF-DETR specifics can vary
    // We would iterate detections and extract corresponding masks
    
    return segmentations;
}

float RfDetrSegmentationPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

} // namespace vision_core
