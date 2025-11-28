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

    // TODO: Implement actual RF-DETR segmentation postprocessing
    // This is currently a placeholder to match the structure
    std::vector<InstanceSegmentation> segmentations;
    return segmentations;
}

float RfDetrSegmentationPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

} // namespace vision_core
