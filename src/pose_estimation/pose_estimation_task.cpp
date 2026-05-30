#include "vision-core/pose_estimation/pose_estimation_task.hpp"

#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/pose_estimation/edgecrafter_pose_postprocessor.hpp"
#include "vision-core/pose_estimation/vit_pose_postprocessor.hpp"
#include "vision-core/pose_estimation/yolo_pose_postprocessor.hpp"

#include <cstring>
#include <stdexcept>

namespace vision_core {

PoseEstimationTask::PoseEstimationTask(const ModelInfo& model_info, const std::string& model_type,
                                       float confidence_threshold, float nms_threshold)
    : TaskInterface(model_info), model_type_(model_type) {

    cv::Size input_size(input_width_, input_height_);

    if (model_type_.size() >= 4 && model_type_.substr(0, 4) == "yolo") {
        // YOLO pose: letterbox preprocessing + YOLO pose postprocessing
        preprocessor_ = std::make_unique<YoloPreprocessor>(input_size);
        postprocessor_ = std::make_unique<YoloPosePostprocessor>(input_size, confidence_threshold, nms_threshold);
    } else if (model_type_ == "vitpose") {
        // ViTPose: ImageNet-normalized NCHW preprocessing + heatmap postprocessing
        PreprocessConfig config;
        config.input_size = input_size;
        config.format = ImageFormat::NCHW;
        config.data_type = DataType::FLOAT32;
        config.normalize = true;
        config.apply_imagenet_norm = true;
        config.bgr_to_rgb = true;
        preprocessor_ = std::make_unique<Preprocessor>(config);
        postprocessor_ = std::make_unique<ViTPosePostprocessor>();
    } else if (model_type_.find("ecpose") == 0 ||
               (model_type_.find("edgecrafter") == 0 && model_type_.find("pose") != std::string::npos)) {
        // EdgeCrafter pose: direct resize + ImageNet normalization + EdgeCrafter pose postprocessing
        PreprocessConfig config;
        config.input_size = input_size;
        config.format = ImageFormat::NCHW;
        config.data_type = DataType::FLOAT32;
        config.normalize = true;
        config.apply_imagenet_norm = true;
        config.bgr_to_rgb = true;
        preprocessor_ = std::make_unique<Preprocessor>(config);
        postprocessor_ =
            std::make_unique<EdgeCrafterPosePostprocessor>(confidence_threshold, 0.3F, model_info_.output_names);
    } else {
        throw std::invalid_argument("Unsupported pose model type: " + model_type_);
    }
}

std::vector<std::vector<uint8_t>> PoseEstimationTask::preprocess(const std::vector<cv::Mat>& imgs) {
    if (model_type_.find("ecpose") == 0 ||
        (model_type_.find("edgecrafter") == 0 && model_type_.find("pose") != std::string::npos)) {
        std::vector<std::vector<uint8_t>> results;
        results.reserve(model_info_.input_shapes.size());

        if (imgs.empty() || imgs[0].empty()) {
            throw std::invalid_argument("Empty input image provided");
        }

        const cv::Mat& img = imgs[0];

        for (size_t i = 0; i < model_info_.input_names.size(); ++i) {
            const auto& input_name = model_info_.input_names[i];
            const auto& input_shape = model_info_.input_shapes[i];

            if (input_shape.size() >= 3) {
                results.push_back(preprocessor_->preprocess(img));
            } else if (input_name == "orig_target_sizes" || input_name == "orig_size") {
                std::vector<int64_t> orig_sizes = {static_cast<int64_t>(img.cols), static_cast<int64_t>(img.rows)};
                results.emplace_back(reinterpret_cast<uint8_t*>(orig_sizes.data()),
                                     reinterpret_cast<uint8_t*>(orig_sizes.data()) +
                                         orig_sizes.size() * sizeof(int64_t));
            } else {
                results.emplace_back();
            }
        }

        return results;
    }

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
