#include "neuriplo/tasks/classification/classification_task.hpp"

#include "neuriplo/tasks/classification/classification_postprocessor.hpp"
#include "neuriplo/tasks/classification/classification_preprocessor.hpp"
#include "neuriplo/tasks/classification/tensorflow_postprocessor.hpp"
#include "neuriplo/tasks/classification/torchvision_postprocessor.hpp"
#include "neuriplo/tasks/classification/vit_postprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neuriplo_tasks {

ClassificationTask::ClassificationTask(const ModelInfo& model_info, const std::string& model_name, int top_k,
                                       bool apply_softmax)
    : BaseTask(model_info), model_type_(detectModelType(model_name)), model_name_(model_name), top_k_(top_k),
      apply_softmax_(apply_softmax) {
    // Extract input dimensions
    vision::Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;

    // Create appropriate preprocessor
    preprocessor_ = createPreprocessor(model_type_, input_size);

    if (!preprocessor_) {
        throw std::runtime_error("Failed to create preprocessor for classifier: " + model_name);
    }

    // Create appropriate postprocessor
    postprocessor_ = createPostprocessor(model_type_);

    if (!postprocessor_) {
        throw std::runtime_error("Failed to create postprocessor for classifier: " + model_name);
    }
}

ClassificationTask::~ClassificationTask() = default;

const Preprocessor& ClassificationTask::getPreprocessor() const { return *preprocessor_; }

std::vector<Result> ClassificationTask::decode(const vision::Size& /*frame_size*/, const std::vector<Tensor>& tensors) {
    auto classifications = postprocessor_->postprocess(tensors[0].data, tensors[0].shape);

    return toResults(classifications);
}

ClassificationTask::ModelType ClassificationTask::detectModelType(const std::string& model_name) {
    std::string lower_name = model_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    // ViT models
    if (lower_name.find("vit") != std::string::npos || lower_name == "vit-classifier") {
        return ModelType::VIT;
    }

    // TensorFlow models
    if (lower_name == "tensorflow-classifier" || lower_name.find("tensorflow") != std::string::npos) {
        return ModelType::TENSORFLOW;
    }

    // Default to Torchvision (ResNet, etc.)
    return ModelType::TORCHVISION;
}

std::unique_ptr<Preprocessor> ClassificationTask::createPreprocessor(ModelType type, const vision::Size& input_size) {
    switch (type) {
    case ModelType::TORCHVISION:
        return std::make_unique<TorchvisionPreprocessor>(input_size);

    case ModelType::TENSORFLOW:
        return std::make_unique<TensorflowPreprocessor>(input_size);

    case ModelType::VIT:
        return std::make_unique<ViTPreprocessor>(input_size);

    default:
        return nullptr;
    }
}

std::unique_ptr<ClassificationPostprocessor> ClassificationTask::createPostprocessor(ModelType type) {
    switch (type) {
    case ModelType::TORCHVISION:
        return std::make_unique<TorchvisionPostprocessor>(top_k_, apply_softmax_);

    case ModelType::VIT:
        return std::make_unique<ViTPostprocessor>(top_k_, apply_softmax_);

    case ModelType::TENSORFLOW:
        return std::make_unique<TensorflowPostprocessor>(top_k_, apply_softmax_);

    default:
        // Fallback to default for others for now
        return std::make_unique<DefaultClassificationPostprocessor>(top_k_, apply_softmax_);
    }
}

vision::Size ClassificationTask::extractInputSize(const ModelInfo& model_info) {
    int width = 224;  // default for most classifiers
    int height = 224; // default for most classifiers

    if (!model_info.input_shapes.empty() && model_info.input_shapes[0].size() >= 3) {
        const auto& shape = model_info.input_shapes[0];
        if (shape.size() == 4) {
            if (model_info.input_formats[0] == "FORMAT_NCHW") {
                height = static_cast<int>(shape[2]);
                width = static_cast<int>(shape[3]);
            } else if (model_info.input_formats[0] == "FORMAT_NHWC") {
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            }
        } else if (shape.size() == 3) {
            // 3D case: CHW or HWC
            if (model_info.input_formats[0] == "FORMAT_NCHW") {
                // CHW
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            } else if (model_info.input_formats[0] == "FORMAT_NHWC") {
                // HWC
                height = static_cast<int>(shape[0]);
                width = static_cast<int>(shape[1]);
            }
        }
    }

    return vision::Size(width, height);
}

} // namespace neuriplo_tasks
