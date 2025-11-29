#include "vision-core/object_detection/yolo_postprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vision_core {

YoloPostprocessor::YoloPostprocessor(ObjectDetectionTask::ModelType model_type, float confidence_threshold, float nms_threshold)
    : model_type_(model_type)
    , confidence_threshold_(confidence_threshold)
    , nms_threshold_(nms_threshold) {}

std::vector<Detection> YoloPostprocessor::postprocess(
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;

    switch (model_type_) {
        case ObjectDetectionTask::ModelType::YOLO_STANDARD: {
            if (infer_results.empty() || infer_shapes.empty()) return {};
            detections = postprocessYoloStandard(infer_results[0], infer_shapes[0], frame_size);
            break;
        }
        
        case ObjectDetectionTask::ModelType::YOLO_V10: {
            if (infer_results.empty() || infer_shapes.empty()) return {};
            detections = postprocessYoloV10(infer_results[0], infer_shapes[0], frame_size);
            break;
        }
        
        case ObjectDetectionTask::ModelType::YOLO_NAS: {
            if (infer_results.size() < 2) {
                throw std::runtime_error("YOLO-NAS requires 2 output tensors");
            }
            detections = postprocessYoloNAS(infer_results[0], infer_results[1], 
                                          infer_shapes[0], infer_shapes[1], frame_size);
            break;
        }
        
        default:
            throw std::runtime_error("Unsupported YOLO model type");
    }

    return detections;
}

std::vector<Detection> YoloPostprocessor::postprocessYoloStandard(
    const std::vector<TensorElement>& output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    
    // Check shape dimensions
    if (shape.size() < 3) return {};
    
    // int batch = shape[0]; // Unused
    int channels = shape[1];
    int anchors = shape[2];
    
    // Handle transposed output [1, anchors, channels] vs [1, channels, anchors]
    // YOLOv8/v11 typically export as [1, 4+cls, 8400]
    bool is_transposed = (channels < anchors && channels < 100); 
    
    if (!is_transposed) {
        // Swap dimensions for easier processing if needed, or just adjust indexing
        std::swap(channels, anchors);
    }
    
    int num_classes = channels - 4;
    if (num_classes <= 0) return {};

    for (int i = 0; i < anchors; ++i) {
        // Extract class scores
        float max_score = 0.0f;
        int class_id = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            float score;
            if (is_transposed) {
                // [batch, channels, anchors] -> data[c + 4][i]
                score = getTensorFloat(output[(c + 4) * anchors + i]);
            } else {
                // [batch, anchors, channels] -> data[i][c + 4]
                score = getTensorFloat(output[i * channels + (c + 4)]);
            }
            
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        if (max_score < confidence_threshold_) continue;
        
        // Extract box
        float cx, cy, w, h;
        if (is_transposed) {
            cx = getTensorFloat(output[0 * anchors + i]);
            cy = getTensorFloat(output[1 * anchors + i]);
            w  = getTensorFloat(output[2 * anchors + i]);
            h  = getTensorFloat(output[3 * anchors + i]);
        } else {
            cx = getTensorFloat(output[i * channels + 0]);
            cy = getTensorFloat(output[i * channels + 1]);
            w  = getTensorFloat(output[i * channels + 2]);
            h  = getTensorFloat(output[i * channels + 3]);
        }
        
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;
        
        Detection det;
        det.class_id = class_id;
        det.class_confidence = max_score;
        det.bbox = cv::Rect(static_cast<int>(x), static_cast<int>(y), 
                           static_cast<int>(w), static_cast<int>(h));
        detections.push_back(det);
    }
    
    applyNMS(detections);
    return detections;
}

std::vector<Detection> YoloPostprocessor::postprocessYoloV10(
    const std::vector<TensorElement>& output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    
    // YOLOv10 output: [1, 300, 6] (x1, y1, x2, y2, score, class)
    if (shape.size() < 3 || shape[2] < 6) return {};
    
    int num_dets = shape[1];
    int dims = shape[2];
    
    for (int i = 0; i < num_dets; ++i) {
        float score = getTensorFloat(output[i * dims + 4]);
        if (score < confidence_threshold_) continue;
        
        float x1 = getTensorFloat(output[i * dims + 0]);
        float y1 = getTensorFloat(output[i * dims + 1]);
        float x2 = getTensorFloat(output[i * dims + 2]);
        float y2 = getTensorFloat(output[i * dims + 3]);
        int class_id = static_cast<int>(getTensorFloat(output[i * dims + 5]));
        
        Detection det;
        det.class_id = class_id;
        det.class_confidence = score;
        det.bbox = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), 
                           static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        detections.push_back(det);
    }
    
    // YOLOv10 typically doesn't need NMS, but we can apply it if needed
    // applyNMS(detections); 
    return detections;
}

std::vector<Detection> YoloPostprocessor::postprocessYoloNAS(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    
    // Boxes: [1, N, 4], Scores: [1, N, C]
    if (box_shape.size() < 3 || score_shape.size() < 3) return {};
    
    int num_dets = box_shape[1];
    int num_classes = score_shape[2];
    
    for (int i = 0; i < num_dets; ++i) {
        // Find max score for this detection
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
        
        float x1 = getTensorFloat(boxes[i * 4 + 0]);
        float y1 = getTensorFloat(boxes[i * 4 + 1]);
        float x2 = getTensorFloat(boxes[i * 4 + 2]);
        float y2 = getTensorFloat(boxes[i * 4 + 3]);
        
        Detection det;
        det.class_id = class_id;
        det.class_confidence = max_score;
        det.bbox = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), 
                           static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        detections.push_back(det);
    }
    
    applyNMS(detections);
    return detections;
}

void YoloPostprocessor::applyNMS(std::vector<Detection>& detections) {
    if (detections.empty()) return;
    
    std::vector<bool> suppress(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppress[i]) continue;
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppress[j]) continue;
            
            cv::Rect intersection = detections[i].bbox & detections[j].bbox;
            float intersection_area = intersection.area();
            float union_area = detections[i].bbox.area() + detections[j].bbox.area() - intersection_area;
            
            if (union_area > 0) {
                float iou = intersection_area / union_area;
                if (iou > nms_threshold_) {
                    if (detections[i].class_confidence > detections[j].class_confidence) {
                        suppress[j] = true;
                    } else {
                        suppress[i] = true;
                        break;
                    }
                }
            }
        }
    }
    
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
                      [&](const Detection& det) {
                          size_t idx = &det - &detections[0];
                          return suppress[idx];
                      }),
        detections.end()
    );
}

float YoloPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

} // namespace vision_core
