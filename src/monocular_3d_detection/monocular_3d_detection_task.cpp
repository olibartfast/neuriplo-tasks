#include "neuriplo/tasks/monocular_3d_detection/monocular_3d_detection_task.hpp"

#include "neuriplo/tasks/object_detection/detection_preprocessor.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace neuriplo_tasks {

Monocular3DDetectionTask::Monocular3DDetectionTask(const ModelInfo& model_info, const std::string& model_name,
                                                   float confidence_threshold, float nms_threshold)
    : BaseTask(model_info), model_type_(detectModelType(model_name)), model_name_(model_name) {
    vision::Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;

    preprocessor_ = std::make_unique<YoloPreprocessor>(input_size);
    if (preprocessor_ == nullptr) {
        throw std::runtime_error("Failed to create preprocessor for monocular 3D model: " + model_name);
    }
    preprocessor_->applyImageInputType(model_info);

    // v1 and v2 checkpoints both emit ground-first corners (verified
    // empirically on the released weights: indices 0-3 sit lower in the
    // image than 4-7; the paper text states the opposite). No remap.
    const bool end_to_end = (model_type_ == ModelType::URBANOMNI_V2);
    postprocessor_ =
        std::make_unique<Monocular3DPostprocessor>(input_size, confidence_threshold, nms_threshold, end_to_end, false);
    if (postprocessor_ == nullptr) {
        throw std::runtime_error("Failed to create postprocessor for monocular 3D model: " + model_name);
    }
}

Monocular3DDetectionTask::~Monocular3DDetectionTask() = default;

const Preprocessor& Monocular3DDetectionTask::getPreprocessor() const { return *preprocessor_; }

std::vector<Result> Monocular3DDetectionTask::decode(const vision::Size& frame_size,
                                                     const std::vector<Tensor>& tensors) {
    return toResults(postprocessor_->postprocess(tensors, frame_size));
}

Monocular3DDetectionTask::ModelType Monocular3DDetectionTask::detectModelType(const std::string& model_name) {
    std::string lower = model_name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    if (lower.find("urbanomni") == std::string::npos && lower.find("monocular3d") == std::string::npos &&
        lower.find("mono3d") == std::string::npos) {
        throw std::invalid_argument("Unsupported monocular 3D model type: " + model_name);
    }
    // UrbanOmniDetect-2 is YOLO26-based (end-to-end rows, ground-first corners).
    if (lower.find("v2") != std::string::npos || lower.find("26") != std::string::npos) {
        return ModelType::URBANOMNI_V2;
    }
    return ModelType::URBANOMNI;
}

Size Monocular3DDetectionTask::extractInputSize(const ModelInfo& model_info) {
    int width = 640;
    int height = 640;
    if (!model_info.input_shapes.empty() && !model_info.input_shapes[0].empty()) {
        const auto& shape = model_info.input_shapes[0];
        const std::string format = model_info.input_formats.empty() ? "FORMAT_NCHW" : model_info.input_formats[0];
        if (shape.size() == 4) {
            if (format == "FORMAT_NHWC") {
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            } else {
                height = static_cast<int>(shape[2]);
                width = static_cast<int>(shape[3]);
            }
        } else if (shape.size() == 3) {
            if (format == "FORMAT_NHWC") {
                height = static_cast<int>(shape[0]);
                width = static_cast<int>(shape[1]);
            } else {
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            }
        }
    }
    return Size(width, height);
}

} // namespace neuriplo_tasks
