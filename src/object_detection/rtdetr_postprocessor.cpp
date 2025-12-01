#include "vision-core/object_detection/rtdetr_postprocessor.hpp"
#include <stdexcept>

namespace vision_core {

RtDetrPostprocessor::RtDetrPostprocessor(ObjectDetectionTask::ModelType model_type, 
                                         const cv::Size& input_size,
                                         float confidence_threshold)
    : model_type_(model_type)
    , input_size_(input_size)
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
    
    // Boxes: [1, 300, 4] (cx, cy, w, h), Scores: [1, 300, classes]
    if (box_shape.size() < 3 || score_shape.size() < 3) return {};
    
    int num_dets = box_shape[1];
    int num_classes = score_shape[2];
    
    for (int i = 0; i < num_dets; ++i) {
        float max_score = 0.0f;
        int class_id = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            float score = getTensorFloat(scores[i * num_classes + c]);
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        if (max_score < confidence_threshold_) continue;
        
        float cx = getTensorFloat(boxes[i * 4 + 0]);
        float cy = getTensorFloat(boxes[i * 4 + 1]);
        float w = getTensorFloat(boxes[i * 4 + 2]);
        float h = getTensorFloat(boxes[i * 4 + 3]);
        
        // RT-DETR usually outputs normalized coordinates
        float x = (cx - w / 2.0f) * frame_size.width;
        float y = (cy - h / 2.0f) * frame_size.height;
        float width = w * frame_size.width;
        float height = h * frame_size.height;
        
        Detection det;
        det.class_id = class_id;
        det.class_confidence = max_score;
        det.bbox = cv::Rect(static_cast<int>(x), static_cast<int>(y), 
                           static_cast<int>(width), static_cast<int>(height));
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> RtDetrPostprocessor::postprocessRTDETRUL(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size) {
    
    // Ultralytics RT-DETR export often matches YOLO format or standard RT-DETR
    // Assuming standard RT-DETR format for now, but checking shapes
    // Sometimes UL exports as [1, 300, 4+cls] concatenated
    
    // If shapes match separate tensors:
    return postprocessRTDETR(boxes, scores, box_shape, score_shape, frame_size);
}

std::vector<Detection> RtDetrPostprocessor::postprocessRFDETR(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size) {
    
    // RF-DETR logic is similar to RT-DETR
    return postprocessRTDETR(boxes, scores, box_shape, score_shape, frame_size);
}

float RtDetrPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

} // namespace vision_core
