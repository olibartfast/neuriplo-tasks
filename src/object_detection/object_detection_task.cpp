#include "vision-core/object_detection/object_detection_task.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/object_detection/yolo_postprocessor.hpp"
#include "vision-core/object_detection/rtdetr_postprocessor.hpp"
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

    // Create appropriate postprocessor
    postprocessor_ = createPostprocessor(model_type_);
    
    if (!postprocessor_) {
        throw std::runtime_error("Failed to create postprocessor for model: " + model_name);
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
    
    std::vector<Detection> detections = postprocessor_->postprocess(infer_results, infer_shapes, frame_size);
    
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

std::unique_ptr<Postprocessor> ObjectDetectionTask::createPostprocessor(ModelType type) {
    switch (type) {
        case ModelType::YOLO_STANDARD:
        case ModelType::YOLO_V10:
        case ModelType::YOLO_NAS:
            return std::make_unique<YoloPostprocessor>(type, confidence_threshold_, nms_threshold_);
            
        case ModelType::RT_DETR_STYLE:
        case ModelType::RT_DETR_UL:
        case ModelType::RF_DETR:
            return std::make_unique<RtDetrPostprocessor>(type, confidence_threshold_);
            
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

// Registration function for all detection models
void registerObjectDetectionTasks() {
    // YOLO family
    std::vector<std::string> yolo_variants = {
        "yolo", "yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", 
        "yolo11", "yolov12", "yolonas"
    };
    
    // Transformer-based
    std::vector<std::string> transformer_variants = {
        "rtdetr", "rtdetrv2", "rtdetrul", "rtdetr-ultralytics", "dfine", "deim", "rfdetr"
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