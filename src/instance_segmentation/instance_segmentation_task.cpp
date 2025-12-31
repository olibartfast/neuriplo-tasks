#include "vision-core/instance_segmentation/instance_segmentation_task.hpp"
#include "vision-core/instance_segmentation/yolo_segmentation_postprocessor.hpp"
#include "vision-core/instance_segmentation/rfdetr_segmentation_postprocessor.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include <algorithm>
#include <stdexcept>

namespace vision_core {

InstanceSegmentationTask::InstanceSegmentationTask(const ModelInfo& model_info, 
                                                  const std::string& model_name,
                                                  float confidence_threshold,
                                                  float nms_threshold,
                                                  float mask_threshold)
    : TaskInterface(model_info)
    , model_type_(detectModelType(model_name))
    , model_name_(model_name)
    , confidence_threshold_(confidence_threshold)
    , nms_threshold_(nms_threshold)
    , mask_threshold_(mask_threshold) 
{
    // Extract input dimensions
    cv::Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;
    
    // Create appropriate preprocessor
    preprocessor_ = createPreprocessor(model_type_, input_size);
    
    if (!preprocessor_) {
        throw std::runtime_error("Failed to create preprocessor for segmentation model: " + model_name);
    }

    // Create appropriate postprocessor
    postprocessor_ = createPostprocessor(model_type_);
    
    if (!postprocessor_) {
        throw std::runtime_error("Failed to create postprocessor for segmentation model: " + model_name);
    }
}

std::vector<std::vector<uint8_t>> InstanceSegmentationTask::preprocess(const std::vector<cv::Mat>& imgs) {
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

std::vector<Result> InstanceSegmentationTask::postprocess(
    const cv::Size& frame_size,
    const std::vector<Tensor>& tensors) {
    
    // Validate inputs
    if (!validateTensorInputs(tensors)) {
        return {};
    }
    
    // Delegate to postprocessor
    auto segmentations = postprocessor_->postprocess(tensors, frame_size);
    
    // Convert segmentations to results
    std::vector<Result> results;
    results.reserve(segmentations.size());
    for (const auto& segmentation : segmentations) {
        results.emplace_back(segmentation);
    }
    
    return results;
}

InstanceSegmentationTask::ModelType InstanceSegmentationTask::detectModelType(const std::string& model_name) {
    std::string lower_name = model_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    // RF-DETR segmentation
    if (lower_name.find("rfdetr") != std::string::npos || 
        lower_name.find("rf-detr") != std::string::npos ||
        lower_name.find("rfdetrseg") != std::string::npos) {
        return ModelType::RF_DETR_SEG;
    }
    
    // Default to YOLO segmentation (yoloseg)
    return ModelType::YOLO_SEG;
}

std::unique_ptr<Preprocessor> InstanceSegmentationTask::createPreprocessor(ModelType type, const cv::Size& input_size) {
    switch (type) {
        case ModelType::YOLO_SEG:
            return std::make_unique<YoloPreprocessor>(input_size);
            
        case ModelType::RF_DETR_SEG:
            return std::make_unique<RfDetrPreprocessor>(input_size);
            
        default:
            return nullptr;
    }
}

std::unique_ptr<SegmentationPostprocessor> InstanceSegmentationTask::createPostprocessor(ModelType type) {
    cv::Size input_size(input_width_, input_height_);
    
    switch (type) {
        case ModelType::YOLO_SEG:
            return std::make_unique<YoloSegmentationPostprocessor>(
                confidence_threshold_, nms_threshold_, mask_threshold_);
            
        case ModelType::RF_DETR_SEG:
            return std::make_unique<RfDetrSegmentationPostprocessor>(
                input_size, confidence_threshold_, mask_threshold_, model_info_.output_names);
            
        default:
            return nullptr;
    }
}

cv::Size InstanceSegmentationTask::extractInputSize(const ModelInfo& model_info) {
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

bool InstanceSegmentationTask::validateTensorInputs(
    const std::vector<Tensor>& tensors) const {
    
    if (tensors.empty()) {
        return false;
    }
    
    // Model-specific validation
    switch (model_type_) {
        case ModelType::YOLO_SEG:
            return tensors.size() >= 2;
            
        case ModelType::RF_DETR_SEG:
            return tensors.size() >= 3;
            
        default:
            return false;
    }
}

} // namespace vision_core