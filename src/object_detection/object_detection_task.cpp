#include "vision-core/object_detection/object_detection_task.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/object_detection/yolo_postprocessor.hpp"
#include "vision-core/object_detection/rtdetr_postprocessor.hpp"
#include "vision-core/object_detection/rfdetr_postprocessor.hpp"
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

    // Create appropriate postprocessor with input_size for coordinate transformation
    postprocessor_ = createPostprocessor(model_type_, input_size);
    
    if (!postprocessor_) {
        throw std::runtime_error("Failed to create postprocessor for model: " + model_name);
    }
}

std::vector<std::vector<uint8_t>> ObjectDetectionTask::preprocess(const std::vector<cv::Mat>& imgs) {
    std::vector<std::vector<uint8_t>> results;
    
    // For models with multiple inputs (like RT-DETR/DEIM/DFINE with orig_target_sizes),
    // we need to return one result per model input, not per image
    // Note: RT_DETR_UL (Ultralytics) has single input like YOLO
    if (model_type_ == ModelType::RT_DETR_STYLE) {
        // These models may have multiple inputs (images + orig_target_sizes)
        results.reserve(model_info_.input_shapes.size());
        
        if (imgs.empty() || imgs[0].empty()) {
            throw std::invalid_argument("Empty input image provided");
        }
        
        const cv::Mat& img = imgs[0]; // For now, single image
        
        for (size_t i = 0; i < model_info_.input_names.size(); ++i) {
            const auto& input_name = model_info_.input_names[i];
            const auto& input_shape = model_info_.input_shapes[i];
            
            if (input_shape.size() >= 3) {
                // This is the image input
                results.push_back(preprocessor_->preprocess(img));
            } else if (input_name == "orig_target_sizes" || input_name == "orig_size") {
                // Handle original image size input - use input dimensions, not original frame
                std::vector<int64_t> orig_sizes = {
                    static_cast<int64_t>(input_height_),
                    static_cast<int64_t>(input_width_)
                };
                results.emplace_back(
                    reinterpret_cast<uint8_t*>(orig_sizes.data()),
                    reinterpret_cast<uint8_t*>(orig_sizes.data()) + orig_sizes.size() * sizeof(int64_t)
                );
            } else {
                // Unknown input - send empty for now
                results.emplace_back();
            }
        }
    } else {
        // Standard path for YOLO and RF-DETR (single input)
        results.reserve(imgs.size());
        
        for (const auto& img : imgs) {
            if (img.empty()) {
                throw std::invalid_argument("Empty input image provided");
            }
            results.push_back(preprocessor_->preprocess(img));
        }
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
    if (lower_name == "yolov7e2e" || lower_name == "yolov7-e2e") {
        return ModelType::YOLO_V7_E2E;
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
        case ModelType::YOLO_V7_E2E:
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

std::unique_ptr<Postprocessor> ObjectDetectionTask::createPostprocessor(ModelType type, const cv::Size& input_size) {
    switch (type) {
        case ModelType::YOLO_STANDARD:
        case ModelType::YOLO_V10:
        case ModelType::YOLO_NAS:
        case ModelType::YOLO_V7_E2E:
            return std::make_unique<YoloPostprocessor>(type, input_size, confidence_threshold_, nms_threshold_);
            
        case ModelType::RT_DETR_STYLE:
        case ModelType::RT_DETR_UL:
            return std::make_unique<RtDetrPostprocessor>(type, input_size, confidence_threshold_, model_info_.output_names);
            
        case ModelType::RF_DETR:
            return std::make_unique<RfDetrPostprocessor>(input_size, confidence_threshold_, model_info_.output_names);
            
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
        case ModelType::YOLO_V7_E2E:
        case ModelType::RT_DETR_STYLE:
        case ModelType::RT_DETR_UL:
        case ModelType::RF_DETR:
            return infer_results.size() >= 2 && infer_shapes.size() >= 2;
            
        default:
            return false;
    }
}

} // namespace vision_core