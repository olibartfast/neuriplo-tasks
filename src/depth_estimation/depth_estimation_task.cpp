#include "neuriplo/tasks/depth_estimation/depth_estimation_task.hpp"

#include "neuriplo/tasks/depth_estimation/depth_anything_v2_postprocessor.hpp"
#include "neuriplo/tasks/depth_estimation/depth_estimation_preprocessor.hpp"

#include <algorithm>
#include <stdexcept>

namespace neuriplo_tasks {

DepthEstimationTask::DepthEstimationTask(const ModelInfo& model_info, const std::string& model_name)
    : BaseTask(model_info), model_type_(detectModelType(model_name)), model_name_(model_name) {
    vision::Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;

    preprocessor_ = createPreprocessor(model_type_, input_size);
    if (!preprocessor_) {
        throw std::runtime_error("Failed to create preprocessor for depth model: " + model_name);
    }

    postprocessor_ = createPostprocessor(model_type_);
    if (!postprocessor_) {
        throw std::runtime_error("Failed to create postprocessor for depth model: " + model_name);
    }
}

DepthEstimationTask::~DepthEstimationTask() = default;

const Preprocessor& DepthEstimationTask::getPreprocessor() const { return *preprocessor_; }

std::vector<Result> DepthEstimationTask::decode(const vision::Size& frame_size, const std::vector<Tensor>& tensors) {
    auto depths = postprocessor_->postprocess(tensors[0].data, tensors[0].shape, frame_size);

    return toResults(depths);
}

DepthEstimationTask::ModelType DepthEstimationTask::detectModelType(const std::string& model_name) {
    std::string lower_name = model_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    if (lower_name.find("depthanythingv2") != std::string::npos ||
        lower_name.find("depth-anything-v2") != std::string::npos ||
        lower_name.find("depth_anything_v2") != std::string::npos) {
        return ModelType::DEPTH_ANYTHING_V2;
    }

    throw std::invalid_argument("Unsupported depth model type: " + model_name);
}

std::unique_ptr<Preprocessor> DepthEstimationTask::createPreprocessor(ModelType type, const vision::Size& input_size) {
    switch (type) {
    case ModelType::DEPTH_ANYTHING_V2:
        return std::make_unique<DepthAnythingV2Preprocessor>(input_size);

    default:
        return nullptr;
    }
}

std::unique_ptr<DepthEstimationPostprocessor> DepthEstimationTask::createPostprocessor(ModelType type) {
    switch (type) {
    case ModelType::DEPTH_ANYTHING_V2:
        return std::make_unique<DepthAnythingV2Postprocessor>();

    default:
        return nullptr;
    }
}

vision::Size DepthEstimationTask::extractInputSize(const ModelInfo& model_info) {
    int width = 518;
    int height = 518;

    if (!model_info.input_shapes.empty()) {
        const auto& shape = model_info.input_shapes[0];

        if (shape.size() == 4) {
            if (model_info.input_formats[0] == "FORMAT_NHWC") {
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            } else {
                height = static_cast<int>(shape[2]);
                width = static_cast<int>(shape[3]);
            }
        } else if (shape.size() == 3) {
            if (model_info.input_formats[0] == "FORMAT_NHWC") {
                height = static_cast<int>(shape[0]);
                width = static_cast<int>(shape[1]);
            } else {
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            }
        }
    }

    return vision::Size(width, height);
}

} // namespace neuriplo_tasks
