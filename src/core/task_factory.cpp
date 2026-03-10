#include "vision-core/core/task_factory.hpp"

#include "vision-core/classification/classification_postprocessor.hpp"
#include "vision-core/classification/classification_task.hpp"
#include "vision-core/depth_estimation/depth_estimation_task.hpp"
#include "vision-core/gaussian_splatting/gaussian_splatting_task.hpp"
#include "vision-core/instance_segmentation/instance_segmentation_task.hpp"
#include "vision-core/instance_segmentation/segmentation_postprocessor.hpp"
#include "vision-core/object_detection/object_detection_task.hpp"
#include "vision-core/optical_flow/optical_flow_postprocessor.hpp"
#include "vision-core/optical_flow/optical_flow_task.hpp"
#include "vision-core/pose_estimation/pose_estimation_task.hpp"
#include "vision-core/video_classification/video_classification_task.hpp"

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
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            continue;
        }
        if (c == '-' || c == '_') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    return normalized;
}

std::unique_ptr<TaskInterface> TaskFactory::createTaskInstance(const std::string& model_type,
                                                               const ModelInfo& model_info, const TaskConfig& config) {

    validateInputSizes(model_info.input_shapes);

    const auto normalized = normalizeModelType(model_type);

    if (normalized.empty()) {
        throw std::invalid_argument("Model type string is empty");
    }

    // ============ INSTANCE SEGMENTATION ============
    // Check seg BEFORE generic yolo prefix so "yoloseg" etc. are routed correctly
    if (normalized == "yoloseg" || normalized == "yolov10seg" || normalized == "yolo26seg" ||
        normalized == "rfdetrseg" ||
        (normalized.size() >= 4 && normalized.substr(0, 4) == "yolo" && normalized.find("seg") != std::string::npos)) {
        return std::make_unique<InstanceSegmentationTask>(model_info, normalized, config.confidence_threshold,
                                                          config.nms_threshold, config.mask_threshold);
    }

    // ============ POSE ESTIMATION (YOLO) ============
    // Check yolo*pose* BEFORE generic yolo prefix so pose models are routed correctly
    if (normalized.size() >= 4 && normalized.substr(0, 4) == "yolo" && normalized.find("pose") != std::string::npos) {
        return std::make_unique<PoseEstimationTask>(model_info, normalized, config.confidence_threshold,
                                                    config.nms_threshold);
    }

    // ============ OBJECT DETECTION ============
    // Any yolo* string (that is not a seg or pose variant, handled above)
    if (normalized.size() >= 4 && normalized.substr(0, 4) == "yolo") {
        return std::make_unique<ObjectDetectionTask>(model_info, normalized, config.confidence_threshold,
                                                     config.nms_threshold);
    }

    // Transformer-based detectors
    if (normalized == "rtdetr" || normalized == "rtdetrul" || normalized == "rtdetrultralytics" ||
        normalized == "rfdetr") {
        return std::make_unique<ObjectDetectionTask>(model_info, normalized, config.confidence_threshold,
                                                     config.nms_threshold);
    }

    // ============ CLASSIFICATION ============
    if (normalized == "torchvisionclassifier" || normalized == "tensorflowclassifier" ||
        normalized == "vitclassifier" || (normalized.size() >= 6 && normalized.substr(0, 6) == "resnet") ||
        normalized.find("tensorflow") != std::string::npos) {
        return std::make_unique<ClassificationTask>(model_info, normalized, config.top_k, config.apply_softmax);
    }

    // ============ DEPTH ESTIMATION ============
    if (normalized.find("depthanythingv2") != std::string::npos) {
        return std::make_unique<DepthEstimationTask>(model_info, normalized);
    }

    // ============ GAUSSIAN SPLATTING ============
    if (normalized == "lgm" || normalized == "grm" || normalized == "gaussiansplatting" ||
        normalized == "lgmmini" || normalized == "lgm-mini" ||
        normalized.find("splat") != std::string::npos) {
        return std::make_unique<GaussianSplattingTask>(model_info, normalized);
    }

    // ============ VIDEO CLASSIFICATION ============
    if (normalized == "videomae" || normalized == "vivit" || normalized == "timesformer") {
        return std::make_unique<VideoClassificationTask>(model_info, normalized, config.top_k, config.apply_softmax);
    }

    // ============ OPTICAL FLOW ============
    if (normalized == "raft") {
        return std::make_unique<OpticalFlowTask>(model_info, normalized);
    }

    // ============ POSE ESTIMATION ============
    if (normalized == "vitpose") {
        return std::make_unique<PoseEstimationTask>(model_info, normalized, config.confidence_threshold,
                                                    config.nms_threshold);
    }

    throw std::invalid_argument("Unrecognized model type: " + model_type);
}

} // namespace vision_core
