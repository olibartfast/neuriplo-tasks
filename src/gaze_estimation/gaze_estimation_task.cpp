#include "neuriplo/tasks/gaze_estimation/gaze_estimation_task.hpp"

#include "neuriplo/tasks/gaze_estimation/gaze_estimation_preprocessor.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace neuriplo_tasks {

GazeEstimationTask::GazeEstimationTask(const ModelInfo& model_info, const std::string& model_name)
    : BaseTask(model_info), model_type_(detectModelType(model_name)), model_name_(model_name) {
    vision::Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;

    preprocessor_ = std::make_unique<GazePreprocessor>(input_size);
    if (preprocessor_ == nullptr) {
        throw std::runtime_error("Failed to create preprocessor for gaze model: " + model_name);
    }
    preprocessor_->applyImageInputType(model_info);

    postprocessor_ = std::make_unique<GazePostprocessor>();
    if (postprocessor_ == nullptr) {
        throw std::runtime_error("Failed to create postprocessor for gaze model: " + model_name);
    }
}

GazeEstimationTask::~GazeEstimationTask() = default;

const Preprocessor& GazeEstimationTask::getPreprocessor() const { return *preprocessor_; }

std::vector<Result> GazeEstimationTask::decode(const vision::Size& /*frame_size*/, const std::vector<Tensor>& tensors) {
    return toResults(postprocessor_->postprocess(tensors));
}

GazeEstimationTask::ModelType GazeEstimationTask::detectModelType(const std::string& model_name) {
    std::string lower = model_name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    if (lower.find("gazetr") != std::string::npos) {
        return ModelType::GAZETR;
    }
    if (lower.find("gaze") != std::string::npos || lower.find("ethxgaze") != std::string::npos ||
        lower.find("xgaze") != std::string::npos) {
        return ModelType::ETHXGAZE;
    }
    throw std::invalid_argument("Unsupported gaze model type: " + model_name);
}

Size GazeEstimationTask::extractInputSize(const ModelInfo& model_info) {
    int width = 224;
    int height = 224;
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
