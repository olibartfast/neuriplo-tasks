#include "vision-core/instance_segmentation/instance_segmentation_task.hpp"
#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/core/task_factory.hpp"
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
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes) {
    
    // Validate inputs
    if (!validateTensorInputs(infer_results, infer_shapes)) {
        return {};
    }
    
    std::vector<InstanceSegmentation> segmentations;
    
    // Route to appropriate postprocessor based on model type
    switch (model_type_) {
        case ModelType::YOLO_SEG: {
            if (infer_results.size() < 2) {
                throw std::runtime_error("YOLO segmentation requires 2 output tensors");
            }
            segmentations = postprocessYoloSeg(
                infer_results[0], infer_results[1],
                infer_shapes[0], infer_shapes[1],
                frame_size);
            break;
        }
        
        case ModelType::RF_DETR_SEG: {
            if (infer_results.size() < 3) {
                throw std::runtime_error("RF-DETR segmentation requires 3 output tensors");
            }
            segmentations = postprocessRFDETRSeg(
                infer_results[0], infer_results[1], infer_results[2],
                infer_shapes[0], infer_shapes[1], infer_shapes[2],
                frame_size);
            break;
        }
        
        default:
            throw std::runtime_error("Unsupported segmentation model type for: " + model_name_);
    }
    
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
    if (lower_name == "rfdetr" || lower_name == "rf-detr") {
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
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes) const {
    
    if (infer_results.empty() || infer_shapes.empty()) {
        return false;
    }
    
    // Model-specific validation
    switch (model_type_) {
        case ModelType::YOLO_SEG:
            return infer_results.size() >= 2 && infer_shapes.size() >= 2;
            
        case ModelType::RF_DETR_SEG:
            return infer_results.size() >= 3 && infer_shapes.size() >= 3;
            
        default:
            return false;
    }
}

std::vector<InstanceSegmentation> InstanceSegmentationTask::postprocessYoloSeg(
    const std::vector<TensorElement>& detections,
    const std::vector<TensorElement>& mask_protos,
    const std::vector<int64_t>& det_shape,
    const std::vector<int64_t>& mask_shape,
    const cv::Size& frame_size) {
    
    std::vector<InstanceSegmentation> segmentations;
    // TODO: Implement YOLO segmentation postprocessing
    return segmentations;
}

std::vector<InstanceSegmentation> InstanceSegmentationTask::postprocessRFDETRSeg(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& labels,
    const std::vector<TensorElement>& masks,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& label_shape,
    const std::vector<int64_t>& mask_shape,
    const cv::Size& frame_size) {
    
    std::vector<InstanceSegmentation> segmentations;
    // TODO: Implement RF-DETR segmentation postprocessing
    return segmentations;
}

float InstanceSegmentationTask::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

void InstanceSegmentationTask::applyNMS(std::vector<InstanceSegmentation>& segmentations) {
    // TODO: Implement NMS for segmentation results
}

// Registration function for all instance segmentation models
void registerInstanceSegmentationTasks() {
    // Segmentation variants
    std::vector<std::string> segmentation_variants = {
        "yoloseg", "rfdetr"
    };
    
    // Register all variants with unified InstanceSegmentationTask
    for (const auto& variant : segmentation_variants) {
        TaskFactory::registerTask(variant, [variant](const ModelInfo& info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<InstanceSegmentationTask>(info, variant);
        });
    }
}

} // namespace vision_core