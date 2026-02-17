#include "vision-core/pose_estimation/pose_estimation_task.hpp"

#include "vision-core/pose_estimation/vit_pose_postprocessor.hpp"

#include <stdexcept>

namespace vision_core {

PoseEstimationTask::PoseEstimationTask(const ModelInfo& model_info, const std::string& model_type)
    : TaskInterface(model_info), model_type_(model_type) {

    // Initialize Preprocessor
    PreprocessConfig config;
    // TaskInterface initializes input_width_, input_height_, input_channels_
    config.input_size = cv::Size(input_width_, input_height_);

    // ViT Pose usually expects NCHW, RGB, Normalized with ImageNet mean/std
    config.format = ImageFormat::NCHW;
    config.data_type = DataType::FLOAT32;
    config.normalize = true;
    config.apply_imagenet_norm = true;
    config.bgr_to_rgb = true;

    preprocessor_ = std::make_unique<Preprocessor>(config);

    // Initialize Postprocessor
    if (model_type_ == "vitpose") {
        postprocessor_ = std::make_unique<ViTPosePostprocessor>();
    } else {
        throw std::invalid_argument("Unsupported pose model type: " + model_type_);
    }
}

std::vector<std::vector<uint8_t>> PoseEstimationTask::preprocess(const std::vector<cv::Mat>& imgs) {
    return preprocessor_->preprocess(imgs);
}

std::vector<Result> PoseEstimationTask::postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) {

    cv::Size input_size(input_width_, input_height_);

    auto poses = postprocessor_->postprocess(tensors, frame_size, input_size);

    std::vector<Result> results;
    results.reserve(poses.size());
    for (const auto& pose : poses) {
        results.push_back(pose);
    }
    return results;
}

} // namespace vision_core
