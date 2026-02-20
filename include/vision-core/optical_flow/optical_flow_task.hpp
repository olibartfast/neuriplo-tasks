#pragma once

#include "vision-core/core/preprocessor.hpp"
#include "vision-core/core/task_interface.hpp"

#include <memory>
#include <string>

namespace vision_core {

class OpticalFlowPostprocessor;

class OpticalFlowTask : public TaskInterface {
  public:
    enum class ModelType : uint8_t {
        RAFT // RAFT-based optical flow models
    };

    explicit OpticalFlowTask(const ModelInfo& model_info, const std::string& model_name);

    // TaskInterface implementation
    TaskType getTaskType() override { return TaskType::OpticalFlow; }

    int getRequiredFrames() const override { return 2; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;

    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) override;

  private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<OpticalFlowPostprocessor> postprocessor_;

    static ModelType detectModelType(const std::string& model_name);

    std::unique_ptr<Preprocessor> createPreprocessor(ModelType type, const cv::Size& input_size);
    std::unique_ptr<OpticalFlowPostprocessor> createPostprocessor(ModelType type);

    cv::Size extractInputSize(const ModelInfo& model_info);
};

} // namespace vision_core
