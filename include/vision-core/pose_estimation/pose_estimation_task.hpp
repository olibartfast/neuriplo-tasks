#pragma once

#include "vision-core/core/preprocessor.hpp"
#include "vision-core/core/task_interface.hpp"
#include "vision-core/pose_estimation/pose_postprocessor.hpp"

#include <memory>
#include <string>

namespace vision_core {

class PoseEstimationTask : public TaskInterface {
  public:
    explicit PoseEstimationTask(const ModelInfo& model_info, const std::string& model_type,
                                float confidence_threshold = 0.25f, float nms_threshold = 0.45f);

    TaskType getTaskType() override { return TaskType::PoseEstimation; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;

    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) override;

  private:
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<PosePostprocessor> postprocessor_;
    std::string model_type_;
};

} // namespace vision_core
