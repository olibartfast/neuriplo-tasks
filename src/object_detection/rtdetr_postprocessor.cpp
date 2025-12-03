#include "vision-core/object_detection/rtdetr_postprocessor.hpp"
#include <stdexcept>
#include <iostream>

namespace vision_core {

RtDetrPostprocessor::RtDetrPostprocessor(ObjectDetectionTask::ModelType model_type, 
                                         const cv::Size& input_size,
                                         float confidence_threshold,
                                         const std::vector<std::string>& output_names)
    : model_type_(model_type)
    , input_size_(input_size)
    , confidence_threshold_(confidence_threshold) 
{
    findOutputIndices(output_names);
}

void RtDetrPostprocessor::findOutputIndices(const std::vector<std::string>& output_names) {
    scores_idx_ = 0;
    boxes_idx_ = 1;
    labels_idx_ = 2;
    
    if (output_names.empty()) return;
    
    for (size_t i = 0; i < output_names.size(); ++i) {
        const auto& name = output_names[i];
        if (name == "scores") {
            scores_idx_ = static_cast<int>(i);
        } else if (name == "boxes") {
            boxes_idx_ = static_cast<int>(i);
        } else if (name == "labels") {
            labels_idx_ = static_cast<int>(i);
        }
    }
}

std::vector<Detection> RtDetrPostprocessor::postprocess(
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes,
    const cv::Size& frame_size) {

    std::vector<Detection> detections;

    switch (model_type_) {
        case ObjectDetectionTask::ModelType::RT_DETR_STYLE: {
            if (infer_results.size() < 3) {
                throw std::runtime_error("RT-DETR style models require 3 output tensors");
            }
            detections = postprocessRTDETR(
                infer_results[scores_idx_], 
                infer_results[boxes_idx_],
                infer_results[labels_idx_],
                infer_shapes[scores_idx_], 
                infer_shapes[boxes_idx_], 
                frame_size);
            break;
        }

        case ObjectDetectionTask::ModelType::RT_DETR_UL: {
            if (infer_results.empty()) {
                throw std::runtime_error("RT-DETR UL requires at least 1 output tensor");
            }
            detections = postprocessRTDETRUL(
                infer_results[0],
                infer_shapes[0], 
                frame_size);
            break;
        }
        
        default:
            throw std::runtime_error("Unsupported model type for RtDetrPostprocessor");
    }

    return detections;
}

std::vector<Detection> RtDetrPostprocessor::postprocessRTDETR(
    const std::vector<TensorElement>& scores,
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& labels,
    const std::vector<int64_t>& score_shape,
    const std::vector<int64_t>& box_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    
    if (score_shape.size() < 2 || box_shape.size() < 3) {
        return {};
    }
    
    int num_dets = static_cast<int>(score_shape[1]);
    
    float r_w = static_cast<float>(frame_size.width) / static_cast<float>(input_size_.width);
    float r_h = static_cast<float>(frame_size.height) / static_cast<float>(input_size_.height);
    
    for (int i = 0; i < num_dets; ++i) {
        float score = getTensorFloat(scores[i]);
        
        if (score < confidence_threshold_) continue;
        
        int class_id = getTensorInt(labels[i]);
        if (class_id < 0) continue;
        
        float x1 = getTensorFloat(boxes[i * 4 + 0]) * r_w;
        float y1 = getTensorFloat(boxes[i * 4 + 1]) * r_h;
        float x2 = getTensorFloat(boxes[i * 4 + 2]) * r_w;
        float y2 = getTensorFloat(boxes[i * 4 + 3]) * r_h;
        
        Detection det;
        det.class_id = class_id;
        det.class_confidence = score;
        det.bbox = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                           cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> RtDetrPostprocessor::postprocessRTDETRUL(
    const std::vector<TensorElement>& output,
    const std::vector<int64_t>& output_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    
    if (output_shape.size() < 3) return {};
    
    int num_dets = static_cast<int>(output_shape[1]);
    int dims = static_cast<int>(output_shape[2]);
    int num_classes = dims - 4;
    
    if (num_classes <= 0) return {};
    
    float r_w = static_cast<float>(frame_size.width) / static_cast<float>(input_size_.width);
    float r_h = static_cast<float>(frame_size.height) / static_cast<float>(input_size_.height);
    
    for (int i = 0; i < num_dets; ++i) {
        int offset = i * dims;
        
        float max_score = 0.0f;
        int class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = getTensorFloat(output[offset + 4 + c]);
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        if (max_score < confidence_threshold_) continue;
        
        float x1 = getTensorFloat(output[offset + 0]) * r_w;
        float y1 = getTensorFloat(output[offset + 1]) * r_h;
        float x2 = getTensorFloat(output[offset + 2]) * r_w;
        float y2 = getTensorFloat(output[offset + 3]) * r_h;
        
        Detection det;
        det.class_id = class_id;
        det.class_confidence = max_score;
        det.bbox = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                           cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        detections.push_back(det);
    }
    
    return detections;
}

float RtDetrPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

int RtDetrPostprocessor::getTensorInt(const TensorElement& element) {
    return std::visit([](auto&& value) -> int {
        return static_cast<int>(value);
    }, element);
}

} // namespace vision_core
