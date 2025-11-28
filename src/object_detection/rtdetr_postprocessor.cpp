#include "vision-core/object_detection/rtdetr_postprocessor.hpp"
#include <stdexcept>

namespace vision_core {

RtDetrPostprocessor::RtDetrPostprocessor(ObjectDetectionTask::ModelType model_type, float confidence_threshold)
    : model_type_(model_type)
    , confidence_threshold_(confidence_threshold) {}

std::vector<Detection> RtDetrPostprocessor::postprocess(
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes,
    const cv::Size& frame_size) {
    
    if (infer_results.size() < 2 || infer_shapes.size() < 2) {
        throw std::runtime_error("Transformer-based models require at least 2 output tensors");
    }

    std::vector<Detection> detections;

    switch (model_type_) {
        case ObjectDetectionTask::ModelType::RT_DETR_STYLE: {
            detections = postprocessRTDETR(infer_results[0], infer_results[1],
                                         infer_shapes[0], infer_shapes[1], frame_size);
            break;
        }

        case ObjectDetectionTask::ModelType::RT_DETR_UL: {
            detections = postprocessRTDETRUL(infer_results[0], infer_results[1],
                                         infer_shapes[0], infer_shapes[1], frame_size);
            break;
        }
        
        case ObjectDetectionTask::ModelType::RF_DETR: {
            detections = postprocessRFDETR(infer_results[0], infer_results[1],
                                         infer_shapes[0], infer_shapes[1], frame_size);
            break;
        }
        
        default:
            throw std::runtime_error("Unsupported Transformer-based model type");
    }

    return detections;
}

std::vector<Detection> RtDetrPostprocessor::postprocessRTDETR(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    // Placeholder
    return detections;
}

std::vector<Detection> RtDetrPostprocessor::postprocessRTDETRUL(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    // Placeholder
    return detections;
}

std::vector<Detection> RtDetrPostprocessor::postprocessRFDETR(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    // Placeholder
    return detections;
}

float RtDetrPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

} // namespace vision_core
