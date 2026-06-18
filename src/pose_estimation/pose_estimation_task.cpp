#include "neuriplo/tasks/pose_estimation/pose_estimation_task.hpp"

#include "neuriplo/tasks/object_detection/detection_preprocessor.hpp"
#include "neuriplo/tasks/pose_estimation/edgecrafter_pose_postprocessor.hpp"
#include "neuriplo/tasks/pose_estimation/rfdetr_pose_postprocessor.hpp"
#include "neuriplo/tasks/pose_estimation/vit_pose_postprocessor.hpp"
#include "neuriplo/tasks/pose_estimation/yolo_pose_postprocessor.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace neuriplo_tasks {

class PosePreprocessStrategy {
  public:
    virtual ~PosePreprocessStrategy() = default;

    [[nodiscard]] virtual std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) const = 0;
};

namespace {

std::vector<uint8_t> encodeInt64Pair(int64_t first, int64_t second) {
    std::vector<int64_t> values = {first, second};
    const auto* begin = reinterpret_cast<const uint8_t*>(values.data());
    const auto* end = begin + values.size() * sizeof(int64_t);
    return {begin, end};
}

constexpr float kDefaultRfDetrKeypointUncertaintyAlpha = 0.5F;
constexpr float kDefaultEdgeCrafterKeypointThreshold = 0.3F;

class SingleInputPosePreprocessStrategy final : public PosePreprocessStrategy {
  public:
    explicit SingleInputPosePreprocessStrategy(std::unique_ptr<Preprocessor> preprocessor)
        : preprocessor_(std::move(preprocessor)) {}

    [[nodiscard]] std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) const override {
        return preprocessor_->preprocess(imgs);
    }

  private:
    std::unique_ptr<Preprocessor> preprocessor_;
};

class ModelInputPosePreprocessStrategy final : public PosePreprocessStrategy {
  public:
    ModelInputPosePreprocessStrategy(std::unique_ptr<Preprocessor> preprocessor, const ModelInfo& model_info)
        : preprocessor_(std::move(preprocessor)), model_info_(model_info) {}

    [[nodiscard]] std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) const override {
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
                results.push_back(encodeInt64Pair(static_cast<int64_t>(img.cols), static_cast<int64_t>(img.rows)));
            } else {
                results.emplace_back();
            }
        }

        return results;
    }

  private:
    std::unique_ptr<Preprocessor> preprocessor_;
    ModelInfo model_info_;
};

} // namespace

PoseEstimationTask::PoseEstimationTask(const ModelInfo& model_info, const std::string& model_type,
                                       float confidence_threshold, float nms_threshold)
    : TaskInterface(model_info), model_type_(detectModelType(model_type)), model_name_(model_type) {

    cv::Size input_size(input_width_, input_height_);

    preprocess_strategy_ = createPreprocessStrategy(model_type_, input_size);
    if (!preprocess_strategy_) {
        throw std::invalid_argument("Unsupported pose model type: " + model_name_);
    }

    postprocessor_ = createPostprocessor(model_type_, input_size, confidence_threshold, nms_threshold);
    if (!postprocessor_) {
        throw std::invalid_argument("Unsupported pose model type: " + model_name_);
    }
}

PoseEstimationTask::~PoseEstimationTask() = default;

std::vector<std::vector<uint8_t>> PoseEstimationTask::preprocess(const std::vector<cv::Mat>& imgs) {
    return preprocess_strategy_->preprocess(imgs);
}

std::vector<Result> PoseEstimationTask::postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) {

    cv::Size input_size(input_width_, input_height_);

    auto poses = postprocessor_->postprocess(tensors, frame_size, input_size);

    std::vector<Result> results;
    results.reserve(poses.size());
    for (const auto& pose : poses) {
        results.emplace_back(pose);
    }
    return results;
}

PoseEstimationTask::ModelType PoseEstimationTask::detectModelType(const std::string& model_type) {
    std::string lower_name = model_type;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    if (lower_name.find("rfdetr") == 0 &&
        (lower_name.find("pose") != std::string::npos || lower_name.find("keypoint") != std::string::npos ||
         lower_name.find("kpt") != std::string::npos)) {
        return ModelType::RFDETRPOSE;
    }
    if (lower_name.find("yolo") == 0) {
        return ModelType::YOLO;
    }
    if (lower_name == "vitpose") {
        return ModelType::VITPOSE;
    }
    if (lower_name.find("ecpose") == 0 ||
        (lower_name.find("edgecrafter") == 0 && lower_name.find("pose") != std::string::npos)) {
        return ModelType::EDGECRAFTER;
    }

    return ModelType::UNKNOWN;
}

std::unique_ptr<Preprocessor> PoseEstimationTask::createPreprocessor(ModelType type, const cv::Size& input_size) {
    switch (type) {
    case ModelType::RFDETRPOSE:
        return std::make_unique<RfDetrPreprocessor>(input_size);

    case ModelType::YOLO:
        return std::make_unique<YoloPreprocessor>(input_size);

    case ModelType::VITPOSE:
    case ModelType::EDGECRAFTER: {
        PreprocessConfig config;
        config.input_size = input_size;
        config.format = ImageFormat::NCHW;
        config.data_type = DataType::FLOAT32;
        config.normalize = true;
        config.apply_imagenet_norm = true;
        config.bgr_to_rgb = true;
        return std::make_unique<Preprocessor>(config);
    }

    default:
        return nullptr;
    }
}

std::unique_ptr<PosePreprocessStrategy> PoseEstimationTask::createPreprocessStrategy(ModelType type,
                                                                                     const cv::Size& input_size) {
    auto preprocessor = createPreprocessor(type, input_size);
    if (!preprocessor) {
        return nullptr;
    }

    if (type == ModelType::EDGECRAFTER) {
        return std::make_unique<ModelInputPosePreprocessStrategy>(std::move(preprocessor), model_info_);
    }

    return std::make_unique<SingleInputPosePreprocessStrategy>(std::move(preprocessor));
}

std::unique_ptr<PosePostprocessor> PoseEstimationTask::createPostprocessor(ModelType type, const cv::Size& input_size,
                                                                           float confidence_threshold,
                                                                           float nms_threshold) {
    switch (type) {
    case ModelType::RFDETRPOSE:
        return std::make_unique<RfDetrPosePostprocessor>(input_size, confidence_threshold,
                                                         kDefaultRfDetrKeypointUncertaintyAlpha);

    case ModelType::YOLO:
        return std::make_unique<YoloPosePostprocessor>(input_size, confidence_threshold, nms_threshold);

    case ModelType::VITPOSE:
        return std::make_unique<ViTPosePostprocessor>();

    case ModelType::EDGECRAFTER:
        return std::make_unique<EdgeCrafterPosePostprocessor>(
            confidence_threshold, kDefaultEdgeCrafterKeypointThreshold, model_info_.output_names);

    default:
        return nullptr;
    }
}

} // namespace neuriplo_tasks
