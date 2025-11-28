#include "vision-core/object_detection/object_detection_task.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/core/task_factory.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace vision_core {

ObjectDetectionTask::ObjectDetectionTask(const ModelInfo& model_info, 
                                        const std::string& model_name,
                                        float confidence_threshold,
                                        float nms_threshold)
    : TaskInterface(model_info)
    , model_type_(detectModelType(model_name))
    , model_name_(model_name)
    , confidence_threshold_(confidence_threshold)
    , nms_threshold_(nms_threshold) 
{
    // Extract input dimensions
    cv::Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;
    
    // Create appropriate preprocessor
    preprocessor_ = createPreprocessor(model_type_, input_size);
    
    if (!preprocessor_) {
        throw std::runtime_error("Failed to create preprocessor for model: " + model_name);
    }
}

std::vector<std::vector<uint8_t>> ObjectDetectionTask::preprocess(const std::vector<cv::Mat>& imgs) {
    std::vector<std::vector<uint8_t>> results;
    results.reserve(imgs.size());
    
    for (const auto& img : imgs) {
        if (img.empty()) {
            throw std::invalid_argument("Empty input image provided");
        }
        results.push_back(preprocessor_->preprocess(img));
    }
    
    return results;
}

std::vector<Result> ObjectDetectionTask::postprocess(
    const cv::Size& frame_size,
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes) {
    
    // Validate inputs
    if (!validateTensorInputs(infer_results, infer_shapes)) {
        return {};
    }
    
    std::vector<Detection> detections;
    
    // Route to appropriate postprocessing based on model type
    switch (model_type_) {
        case ModelType::YOLO_STANDARD: {
            detections = postprocessYoloStandard(infer_results[0], infer_shapes[0], frame_size);
            break;
        }
        
        case ModelType::YOLO_V10: {
            detections = postprocessYoloV10(infer_results[0], infer_shapes[0], frame_size);
            break;
        }
        
        case ModelType::YOLO_NAS: {
            if (infer_results.size() < 2) {
                throw std::runtime_error("YOLO-NAS requires 2 output tensors");
            }
            detections = postprocessYoloNAS(infer_results[0], infer_results[1], 
                                          infer_shapes[0], infer_shapes[1], frame_size);
            break;
        }
        
        case ModelType::RT_DETR_STYLE:
        case ModelType::RT_DETR_UL: {
            if (infer_results.size() < 2) {
                throw std::runtime_error("RT-DETR style models require 2 output tensors");
            }
            detections = postprocessRTDETR(infer_results[0], infer_results[1],
                                         infer_shapes[0], infer_shapes[1], frame_size);
            break;
        }
        
        case ModelType::RF_DETR: {
            if (infer_results.size() < 2) {
                throw std::runtime_error("RF-DETR requires 2 output tensors");
            }
            detections = postprocessRFDETR(infer_results[0], infer_results[1],
                                         infer_shapes[0], infer_shapes[1], frame_size);
            break;
        }
        
        default:
            throw std::runtime_error("Unsupported model type for: " + model_name_);
    }
    
    // Convert detections to results
    std::vector<Result> results;
    results.reserve(detections.size());
    for (const auto& detection : detections) {
        results.emplace_back(detection);
    }
    
    return results;
}

ObjectDetectionTask::ModelType ObjectDetectionTask::detectModelType(const std::string& model_name) {
    std::string lower_name = model_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    // YOLO variants
    if (lower_name == "yolov10") {
        return ModelType::YOLO_V10;
    }
    if (lower_name == "yolonas" || lower_name == "yolo-nas") {
        return ModelType::YOLO_NAS;
    }
    if (lower_name.find("yolo") == 0) { // starts with "yolo"
        return ModelType::YOLO_STANDARD;
    }
    
    // Transformer-based models
    if (lower_name == "rfdetr" || lower_name == "rf-detr") {
        return ModelType::RF_DETR;
    }
    if (lower_name == "rtdetrul") {
        return ModelType::RT_DETR_UL;
    }
    if (lower_name == "rtdetr" || lower_name == "rtdetrv2" || 
        lower_name == "dfine" || lower_name == "deim") {
        return ModelType::RT_DETR_STYLE;
    }
    
    // Default to standard YOLO
    return ModelType::YOLO_STANDARD;
}

std::unique_ptr<Preprocessor> ObjectDetectionTask::createPreprocessor(ModelType type, const cv::Size& input_size) {
    switch (type) {
        case ModelType::YOLO_STANDARD:
        case ModelType::YOLO_V10:
        case ModelType::YOLO_NAS:
            return std::make_unique<YoloPreprocessor>(input_size);
            
        case ModelType::RT_DETR_STYLE:
        case ModelType::RT_DETR_UL:
            return std::make_unique<RtDetrPreprocessor>(input_size);
            
        case ModelType::RF_DETR:
            return std::make_unique<RfDetrPreprocessor>(input_size);
            
        default:
            return nullptr;
    }
}

cv::Size ObjectDetectionTask::extractInputSize(const ModelInfo& model_info) {
    int width = 640;  // default
    int height = 640; // default
    
    if (!model_info.input_shapes.empty() && model_info.input_shapes[0].size() >= 3) {
        const auto& shape = model_info.input_shapes[0];
        if (model_info.input_formats[0] == "FORMAT_NCHW") {
            height = static_cast<int>(shape[2]);
            width = static_cast<int>(shape[3]);
        } else if (model_info.input_formats[0] == "FORMAT_NHWC") {
            height = static_cast<int>(shape[1]);
            width = static_cast<int>(shape[2]);
        }
    }
    
    return cv::Size(width, height);
}

bool ObjectDetectionTask::validateTensorInputs(
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes) const {
    
    if (infer_results.empty() || infer_shapes.empty()) {
        return false;
    }
    
    // Model-specific validation
    switch (model_type_) {
        case ModelType::YOLO_STANDARD:
        case ModelType::YOLO_V10:
            return infer_results.size() >= 1 && infer_shapes.size() >= 1;
            
        case ModelType::YOLO_NAS:
        case ModelType::RT_DETR_STYLE:
        case ModelType::RT_DETR_UL:
        case ModelType::RF_DETR:
            return infer_results.size() >= 2 && infer_shapes.size() >= 2;
            
        default:
            return false;
    }
}

// Helper method implementations
float ObjectDetectionTask::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

void ObjectDetectionTask::applyNMS(std::vector<Detection>& detections) {
    if (detections.empty()) return;
    
    // Simple NMS implementation - can be optimized
    std::vector<bool> suppress(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppress[i]) continue;
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppress[j]) continue;
            
            // Calculate IoU
            cv::Rect intersection = detections[i].bbox & detections[j].bbox;
            float intersection_area = intersection.area();
            float union_area = detections[i].bbox.area() + detections[j].bbox.area() - intersection_area;
            
            if (union_area > 0) {
                float iou = intersection_area / union_area;
                if (iou > nms_threshold_) {
                    // Suppress the detection with lower confidence
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
    
    // Remove suppressed detections
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
                      [&](const Detection& det) {
                          size_t idx = &det - &detections[0];
                          return suppress[idx];
                      }),
        detections.end()
    );
}

std::vector<Detection> ObjectDetectionTask::postprocessYoloStandard(
    const std::vector<TensorElement>& output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    // TODO: Implement YOLO standard postprocessing
    // This would include parsing the YOLO output format and applying NMS
    return detections;
}

std::vector<Detection> ObjectDetectionTask::postprocessYoloV10(
    const std::vector<TensorElement>& output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    // TODO: Implement YOLOv10 postprocessing (end-to-end, no NMS)
    return detections;
}

std::vector<Detection> ObjectDetectionTask::postprocessYoloNAS(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    // TODO: Implement YOLO-NAS postprocessing
    return detections;
}

std::vector<Detection> ObjectDetectionTask::postprocessRTDETR(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    // TODO: Implement RT-DETR style postprocessing
    return detections;
}

std::vector<Detection> ObjectDetectionTask::postprocessRFDETR(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    // TODO: Implement RF-DETR postprocessing
    return detections;
}

// Registration function for all detection models
void registerObjectDetectionTasks() {
    // YOLO family
    std::vector<std::string> yolo_variants = {
        "yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", 
        "yolo11", "yolov12", "yolonas"
    };
    
    // Transformer-based
    std::vector<std::string> transformer_variants = {
        "rtdetr", "rtdetrv2", "rtdetrul", "dfine", "deim", "rfdetr"
    };
    
    // Register all variants with unified ObjectDetectionTask
    for (const auto& variant : yolo_variants) {
        TaskFactory::registerTask(variant, [variant](const ModelInfo& info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<ObjectDetectionTask>(info, variant);
        });
    }
    
    for (const auto& variant : transformer_variants) {
        TaskFactory::registerTask(variant, [variant](const ModelInfo& info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<ObjectDetectionTask>(info, variant);
        });
    }
}

} // namespace vision_core