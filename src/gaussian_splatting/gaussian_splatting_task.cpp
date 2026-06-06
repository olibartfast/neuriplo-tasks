#include "neuriplo/tasks/gaussian_splatting/gaussian_splatting_task.hpp"

#include "neuriplo/tasks/gaussian_splatting/gaussian_splatting_preprocessor.hpp"
#include "neuriplo/tasks/gaussian_splatting/lgm_postprocessor.hpp"

#include <algorithm>
#include <stdexcept>

namespace neuriplo_tasks {

GaussianSplattingTask::GaussianSplattingTask(const ModelInfo& model_info, const std::string& model_name)
    : TaskInterface(model_info), model_type_(detectModelType(model_name)), model_name_(model_name) {
    cv::Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;

    preprocessor_ = createPreprocessor(model_type_, input_size);
    if (!preprocessor_) {
        throw std::runtime_error("Failed to create preprocessor for Gaussian Splatting model: " + model_name);
    }

    postprocessor_ = createPostprocessor(model_type_);
    if (!postprocessor_) {
        throw std::runtime_error("Failed to create postprocessor for Gaussian Splatting model: " + model_name);
    }
}

GaussianSplattingTask::~GaussianSplattingTask() = default;

int GaussianSplattingTask::getRequiredFrames() const {
    // LGM and GRM are trained on 4-view images produced by multi-view diffusion.
    // A caller using a single-image path can still pass 1 frame; the network
    // will produce lower-quality results but won't crash.
    switch (model_type_) {
    case ModelType::LGM:
    case ModelType::GRM:
        return 4;
    default:
        return 1;
    }
}

std::vector<std::vector<uint8_t>> GaussianSplattingTask::preprocess(const std::vector<cv::Mat>& imgs) {
    std::vector<std::vector<uint8_t>> results;
    results.reserve(imgs.size());

    for (const auto& img : imgs) {
        if (img.empty()) {
            throw std::invalid_argument("Empty input image provided to GaussianSplattingTask");
        }
        results.push_back(preprocessor_->preprocess(img));
    }

    return results;
}

std::vector<Result> GaussianSplattingTask::postprocess(const cv::Size& /*frame_size*/,
                                                       const std::vector<Tensor>& tensors) {
    if (tensors.empty()) {
        return {};
    }

    GaussianSplatting splat = postprocessor_->postprocess(tensors[0].data, tensors[0].shape);

    return {Result{std::move(splat)}};
}

GaussianSplattingTask::ModelType GaussianSplattingTask::detectModelType(const std::string& model_name) {
    std::string lower = model_name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("grm") != std::string::npos) {
        return ModelType::GRM;
    }

    // LGM, LGM-mini, gaussiansplatting, *splat* all map to LGM postprocessor
    return ModelType::LGM;
}

std::unique_ptr<Preprocessor> GaussianSplattingTask::createPreprocessor(ModelType /*type*/,
                                                                        const cv::Size& input_size) {
    return std::make_unique<GaussianSplattingPreprocessor>(input_size);
}

std::unique_ptr<GaussianSplattingPostprocessor> GaussianSplattingTask::createPostprocessor(ModelType type) {
    switch (type) {
    case ModelType::LGM:
    case ModelType::GRM:
        return std::make_unique<LgmPostprocessor>();
    default:
        return nullptr;
    }
}

cv::Size GaussianSplattingTask::extractInputSize(const ModelInfo& model_info) {
    // Default input size used by LGM / GRM
    int width = 256;
    int height = 256;

    if (!model_info.input_shapes.empty()) {
        const auto& shape = model_info.input_shapes[0];
        const std::string& format = model_info.input_formats.empty() ? "FORMAT_NCHW" : model_info.input_formats[0];

        if (shape.size() == 4) {
            if (format == "FORMAT_NCHW") {
                // [N, C, H, W]
                height = static_cast<int>(shape[2]);
                width = static_cast<int>(shape[3]);
            } else if (format == "FORMAT_NHWC") {
                // [N, H, W, C]
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            }
        } else if (shape.size() == 3) {
            if (format == "FORMAT_NCHW") {
                // [C, H, W]
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            } else if (format == "FORMAT_NHWC") {
                // [H, W, C]
                height = static_cast<int>(shape[0]);
                width = static_cast<int>(shape[1]);
            }
        }
    }

    return {width, height};
}

} // namespace neuriplo_tasks
