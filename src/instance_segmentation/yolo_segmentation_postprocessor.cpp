#include "vision-core/instance_segmentation/yolo_segmentation_postprocessor.hpp"
#include <stdexcept>

namespace vision_core {

YoloSegmentationPostprocessor::YoloSegmentationPostprocessor(float confidence_threshold, float nms_threshold, float mask_threshold)
    : confidence_threshold_(confidence_threshold)
    , nms_threshold_(nms_threshold)
    , mask_threshold_(mask_threshold) {}

std::vector<InstanceSegmentation> YoloSegmentationPostprocessor::postprocess(
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes,
    const cv::Size& frame_size) {
    
    if (infer_results.size() < 2 || infer_shapes.size() < 2) {
        throw std::runtime_error("YOLO segmentation requires at least 2 output tensors");
    }

    // TODO: Implement actual YOLO segmentation postprocessing
    // This is currently a placeholder to match the structure
    std::vector<InstanceSegmentation> segmentations;
    return segmentations;
}

float YoloSegmentationPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

} // namespace vision_core
