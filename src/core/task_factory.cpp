#include "vision-core/core/task_factory.hpp"
#include "vision-core/object_detection/object_detection_task.hpp"
#include "vision-core/classification/classification_task.hpp"
#include "vision-core/classification/classification_postprocessor.hpp"
#include "vision-core/instance_segmentation/instance_segmentation_task.hpp"
#include "vision-core/instance_segmentation/segmentation_postprocessor.hpp"
#include "vision-core/optical_flow/optical_flow_task.hpp"
#include "vision-core/optical_flow/optical_flow_postprocessor.hpp"
#include <algorithm>
#include <stdexcept>

namespace vision_core {

void TaskFactory::validateInputSizes(const std::vector<std::vector<int64_t>>& input_sizes) {
    if (input_sizes.empty()) {
        throw InputDimensionError("Input sizes vector is empty");
    }
    for (const auto& size : input_sizes) {
        if (size.empty()) {
            throw InputDimensionError("An input size vector is empty");
        }
        if (std::any_of(size.begin(), size.end(), [](int64_t s) { return s <= 0; })) {
            throw InputDimensionError("Non-positive input size detected");
        }
    }
}

std::string TaskFactory::normalizeModelType(const std::string& model_type) {
    std::string normalized;
    normalized.reserve(model_type.size());

    for (char c : model_type) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        }
        if (c == '-' || c == '_') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    return normalized;
}

std::unique_ptr<TaskInterface> TaskFactory::createTaskInstance(
    const std::string& model_type,
    const ModelInfo& model_info) {
    
    validateInputSizes(model_info.input_shapes);

    const auto normalized = normalizeModelType(model_type);
    
    if (normalized.empty()) {
        throw std::invalid_argument("Model type string is empty");
    }

    // ============ OBJECT DETECTION ============
    // YOLO variants
    if (normalized == "yolo" || normalized == "yolov7e2e" || normalized == "yolov10" || normalized == "yolonas" || normalized == "yolov4") {
        return std::make_unique<ObjectDetectionTask>(model_info, normalized);
    }
    
    // Transformer-based detectors
    if (normalized == "rtdetr" || normalized == "rtdetrul" || normalized == "rfdetr") {
        return std::make_unique<ObjectDetectionTask>(model_info, normalized);
    }

    // ============ CLASSIFICATION ============
    if (normalized == "torchvisionclassifier" ||
        normalized == "tensorflowclassifier" || normalized == "vitclassifier" ||
        normalized == "timesformer") {
        return std::make_unique<ClassificationTask>(model_info, normalized);
    }

    // ============ INSTANCE SEGMENTATION ============
    if (normalized == "yoloseg" || normalized.find("seg") != std::string::npos) {
        return std::make_unique<InstanceSegmentationTask>(model_info, normalized);
    }

    // ============ OPTICAL FLOW ============
    if (normalized == "raft") {
        return std::make_unique<OpticalFlowTask>(model_info, normalized);
    }

    throw std::invalid_argument("Unrecognized model type: " + model_type);
}

} // namespace vision_core
