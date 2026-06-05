#pragma once

#include "vision-core/core/preprocessor.hpp"
#include "vision-core/core/task_interface.hpp"
#include "vision-core/pose_estimation/pose_postprocessor.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace vision_core {

class PosePreprocessStrategy;

class PoseEstimationTask : public TaskInterface {
  public:
    explicit PoseEstimationTask(const ModelInfo& model_info, const std::string& model_type,
                                float confidence_threshold = 0.25f, float nms_threshold = 0.45f);
    ~PoseEstimationTask() override;

    TaskType getTaskType() override { return TaskType::PoseEstimation; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;

    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) override;

  private:
    enum class ModelType : uint8_t { UNKNOWN, YOLO, VITPOSE, EDGECRAFTER };

    std::unique_ptr<PosePreprocessStrategy> preprocess_strategy_;
    std::unique_ptr<PosePostprocessor> postprocessor_;
    ModelType model_type_;
    std::string model_name_;

    static ModelType detectModelType(const std::string& model_type);

    std::unique_ptr<Preprocessor> createPreprocessor(ModelType type, const cv::Size& input_size);
    std::unique_ptr<PosePreprocessStrategy> createPreprocessStrategy(ModelType type, const cv::Size& input_size);
    std::unique_ptr<PosePostprocessor> createPostprocessor(ModelType type, const cv::Size& input_size,
                                                           float confidence_threshold, float nms_threshold);
};

} // namespace vision_core
